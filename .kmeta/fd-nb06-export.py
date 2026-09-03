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
from fdlib.preprocess import NormConstants  # noqa: E402
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
set_seeds(C.SEED)
model = compile_model(proposed_cnn(),
                      steps_per_epoch=max(1, len(inner_tr) // C.BATCH_SIZE),
                      epochs=C.MAX_EPOCHS)
print(f"\nparameters: {count_params(model):,}")
model.summary()

cb = MacroF1(X[inner_va], y[inner_va])
model.fit(X[inner_tr], y[inner_tr], validation_data=(X[inner_va], y[inner_va]),
          epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE, verbose=2,
          class_weight=class_weights(y[inner_tr]), callbacks=[cb])
model.save(OUT / "model_fp32.keras")

# ------------------------------------------------------------------ quantisation
print("\n" + "=" * 74)
print("INT8 conversion -- representative dataset from TRAINING windows only")
print("=" * 74)
to_int8_tflite(model, X[inner_tr], OUT / "model.tflite")

rep = quantisation_report(model, OUT / "model.tflite", X[te_idx], y[te_idx])
rep["params"] = count_params(model)
rep["estimated_arena_bytes"] = estimate_arena_bytes(model)
rep["estimated_arena_kb"] = round(rep["estimated_arena_bytes"] / 1024, 1)
rep["task"] = TASK
rep["signature"] = C.preprocess_signature()
rep["test_subjects"] = sorted(test_subs)

print(json.dumps({k: v for k, v in rep.items() if k != "test_subjects"}, indent=2))
(OUT / "quantisation_report.json").write_text(json.dumps(rep, indent=2))

if not rep["within_budget_accuracy"]:
    print(f"\n!! INT8 cost {rep['macro_f1_drop_pp']:.2f} points of macro-F1, over the "
          f"{C.MAX_QUANT_ACCURACY_DROP_PP} point budget. This is a quantisation "
          "problem, not a hardware one, and must be fixed before flashing.")
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
     "FP32 (desktop)": round(Path(OUT / "model_fp32.keras").stat().st_size / 1024, 1),
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
