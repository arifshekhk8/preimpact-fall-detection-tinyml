"""nb06 -- E4 part 1: train the deployable model, quantise to INT8, verify on desktop.

Section 9 steps 1 and 2. What this notebook can settle without hardware:
  * the final model, trained on a subject-wise split of the full corpus;
  * full-integer INT8 conversion with a representative dataset drawn from TRAINING
    data only -- drawing it from test leaks test statistics into the quantisation
    parameters and invalidates the result;
  * the FP32 versus INT8 accuracy delta on the same held-out set, which must stay
    under two points. If it does not, the problem is quantisation, and it is far
    easier to fix here than on a microcontroller;
  * model.tflite, model.h and the frozen normalisation constants as a C header.

What it cannot settle: measured inference latency, tensor-arena high-water mark and
battery life. Those require the physical ESP32 and are filled in from
firmware/MEASUREMENT.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working")

_fdlib_root = next((p.parent for p in Path("/kaggle/input").rglob("fdlib/__init__.py")), None)
if _fdlib_root is None:
    raise SystemExit("fdlib dataset not attached")
sys.path.insert(0, str(_fdlib_root.parent))

from fdlib import config as C  # noqa: E402
from fdlib.cv import class_weights, inner_val_split  # noqa: E402
from fdlib.experiment import load_corpus  # noqa: E402
from fdlib.metrics import classification_metrics  # noqa: E402
from fdlib.models import MacroF1, compile_model, count_params, proposed_cnn, set_seeds  # noqa: E402
from fdlib.preprocess import NormConstants, instance_normalise  # noqa: E402
from fdlib.tflite_export import (  # noqa: E402
    estimate_arena_bytes, quantisation_report, to_c_array, to_int8_tflite,
)

TASK = "preimpact"  # the deployable model is the pre-impact one -- that is the product
CORPUS = next(Path("/kaggle/input").rglob(f"windows_{TASK}.npz"))

corpus = load_corpus(CORPUS)
X, y, subjects = corpus["X"], corpus["y"], corpus["subject"]
print(f"corpus {X.shape}  subjects {len(np.unique(subjects))}")
print(f"classes {np.bincount(y, minlength=3).tolist()}  (bkg, alert, fall)")

# Held-out test split BY SUBJECT. The representative dataset and the accuracy
# comparison both depend on this boundary being clean.
rng = np.random.default_rng(C.SEED)
subs = np.unique(subjects)
rng.shuffle(subs)
test_subs = set(subs[: max(1, int(len(subs) * 0.2))].tolist())
te = np.array([s in test_subs for s in subjects])
tr_idx = np.where(~te)[0]
te_idx = np.where(te)[0]
print(f"train {len(tr_idx):,} windows / {len(subs) - len(test_subs)} subjects")
print(f"test  {len(te_idx):,} windows / {len(test_subs)} subjects")

inner_tr, inner_va = inner_val_split(subjects, tr_idx)


def train_variant(name: str, in_graph_norm: bool):
    """Train one variant and quantise it.

    in_graph_norm=True  -- per-window instance normalisation is a LAYER, so the model
                           consumes raw windows and normalises internally.
    in_graph_norm=False -- normalisation happens in the data pipeline, and the same
                           two-pass mean/variance runs in the firmware before invoke.

    The two are mathematically identical in float32. They are not identical after INT8
    quantisation, which is the point of running both.
    """
    print("\n" + "=" * 74)
    print(f"variant: {name}  (instance norm {'in graph' if in_graph_norm else 'in pipeline'})")
    print("=" * 74)

    Xv = X if in_graph_norm else instance_normalise(X)
    set_seeds(C.SEED)
    model = compile_model(proposed_cnn(instance_norm=in_graph_norm),
                          steps_per_epoch=max(1, len(inner_tr) // C.BATCH_SIZE),
                          epochs=C.MAX_EPOCHS)
    cb = MacroF1(Xv[inner_va], y[inner_va])
    model.fit(Xv[inner_tr], y[inner_tr], validation_data=(Xv[inner_va], y[inner_va]),
              epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE, verbose=2,
              class_weight=class_weights(y[inner_tr]), callbacks=[cb])

    keras_path = OUT / f"model_fp32_{name}.keras"
    tfl_path = OUT / f"model_{name}.tflite"
    model.save(keras_path)
    to_int8_tflite(model, Xv[inner_tr], tfl_path)

    rep = quantisation_report(model, tfl_path, Xv[te_idx], y[te_idx])
    rep["variant"] = name
    rep["in_graph_instance_norm"] = in_graph_norm
    rep["params"] = count_params(model)
    rep["estimated_arena_bytes"] = estimate_arena_bytes(tfl_path)
    rep["estimated_arena_kb"] = round(rep["estimated_arena_bytes"] / 1024, 1)
    rep["keras_kb"] = round(keras_path.stat().st_size / 1024, 1)
    print(json.dumps(rep, indent=2))
    return model, tfl_path, rep, Xv


# The first run of this notebook put instance normalisation inside the graph and INT8
# quantisation destroyed the model: macro-F1 fell from 0.660 to 0.184, a 47.6 point
# drop against a 2 point budget, with FP32/INT8 agreement at 31 %. Per-window mean and
# variance produce intermediate tensors whose dynamic range a single global scale
# cannot represent, so the layer is hostile to full-integer quantisation.
#
# Both variants are trained and compared here rather than silently switching, because
# the size of that difference is itself a deployment finding.
m_graph, p_graph, rep_graph, _ = train_variant("ingraph", True)
m_pipe, p_pipe, rep_pipe, Xpipe = train_variant("pipeline", False)

reports = {"ingraph": rep_graph, "pipeline": rep_pipe}
(OUT / "quantisation_comparison.json").write_text(json.dumps(reports, indent=2))

print("\n" + "=" * 74)
print("QUANTISATION ROBUSTNESS -- where instance normalisation lives")
print("=" * 74)
cmp = pd.DataFrame([
    {"variant": k,
     "fp32_macro_f1": v["fp32_macro_f1"],
     "int8_macro_f1": v["int8_macro_f1"],
     "drop_pp": v["macro_f1_drop_pp"],
     "fp32_int8_agreement": round(v["agreement"], 4),
     "int8_specificity": v["int8_specificity"],
     "model_kb": v["model_kb"]}
    for k, v in reports.items()
])
print(cmp.to_string(index=False))
cmp.to_csv(OUT / "quantisation_comparison.csv", index=False)

# Ship whichever survives quantisation; prefer the pipeline variant on ties, since it
# is also the cheaper thing to run on the MCU.
best = "pipeline" if rep_pipe["macro_f1_drop_pp"] <= rep_graph["macro_f1_drop_pp"] else "ingraph"
model = m_pipe if best == "pipeline" else m_graph
rep = reports[best]
src = p_pipe if best == "pipeline" else p_graph
(OUT / "model.tflite").write_bytes(src.read_bytes())
(OUT / "model_fp32.keras").write_bytes(
    (OUT / f"model_fp32_{best}.keras").read_bytes())
print(f"\nshipping the '{best}' variant: INT8 macro-F1 {rep['int8_macro_f1']}, "
      f"drop {rep['macro_f1_drop_pp']} points")

rep["task"] = TASK
rep["signature"] = C.preprocess_signature()
rep["test_subjects"] = sorted(test_subs)
rep["shipped_variant"] = best
(OUT / "quantisation_report.json").write_text(json.dumps(rep, indent=2))

if not rep["within_budget_accuracy"]:
    print(f"\n!! INT8 still costs {rep['macro_f1_drop_pp']:.2f} points of macro-F1, over "
          f"the {C.MAX_QUANT_ACCURACY_DROP_PP} point budget. This is a quantisation "
          "problem, not a hardware one, and must be resolved before flashing.")
if not rep["within_budget_kb"]:
    print(f"\n!! model is {rep['model_kb']} KB, over the {C.MAX_MODEL_KB} KB budget.")

# ------------------------------------------------------------------- C artifacts
to_c_array(OUT / "model.tflite", OUT / "model.h")
nc_path = next(Path("/kaggle/input").rglob("norm_constants.json"), None)
if nc_path:
    nc = NormConstants.load(nc_path)
    (OUT / "norm_constants.h").write_text(nc.to_c_header())
    print(f"\nnorm constants carried through from {nc_path}")

# ---------------------------------------------------------------------- Table IV
t4 = pd.DataFrame([
    {"Metric": "Model size (KB)",
     "FP32 (desktop)": rep["keras_kb"],
     "INT8 (desktop)": rep["model_kb"],
     "INT8 (ESP32, measured)": "PENDING-HW"},
    {"Metric": "Peak RAM / tensor arena (KB)",
     "FP32 (desktop)": "n/a",
     "INT8 (desktop)": f"~{rep['estimated_arena_kb']} (estimate)",
     "INT8 (ESP32, measured)": "PENDING-HW"},
    {"Metric": "Inference latency (ms)",
     "FP32 (desktop)": "n/a", "INT8 (desktop)": "n/a",
     "INT8 (ESP32, measured)": "PENDING-HW"},
    {"Metric": "Macro-F1 on held-out set",
     "FP32 (desktop)": rep["fp32_macro_f1"],
     "INT8 (desktop)": rep["int8_macro_f1"],
     "INT8 (ESP32, measured)": "same model"},
    {"Metric": "Sensitivity",
     "FP32 (desktop)": rep["fp32_sensitivity"],
     "INT8 (desktop)": rep["int8_sensitivity"],
     "INT8 (ESP32, measured)": "same model"},
    {"Metric": "Battery life (hours)",
     "FP32 (desktop)": "n/a", "INT8 (desktop)": "n/a",
     "INT8 (ESP32, measured)": "PENDING-HW"},
])
print("\n" + "=" * 74)
print("TABLE IV -- on-device deployment")
print("=" * 74)
print(t4.to_string(index=False))
t4.to_csv(OUT / "table_IV.csv", index=False)
(OUT / "table_IV.md").write_text(
    "# Table IV -- on-device deployment\n\n" + t4.to_markdown(index=False) + "\n\n"
    "`PENDING-HW` cells require the physical ESP32 and MPU6050. The procedure for\n"
    "filling them is in `firmware/MEASUREMENT.md`; the model, firmware and normalisation\n"
    "constants they depend on are all committed and unchanged by the measurement.\n"
)
print("\nwrote model.tflite, model.h, norm_constants.h, table_IV.csv/md")
