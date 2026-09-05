"""nb07 -- the deployable model, and the pre-impact baselines that validate it.

Two jobs in one session, because only one kernel runs at a time.

PART A -- pre-impact baselines on KFall (the gap in Table I).
    E1 established competitiveness on SisFall for the POST-FALL task, but the model
    that actually ships is the PRE-IMPACT one, and nothing had validated that against
    comparators. Same six comparators, same 5-fold subject-grouped protocol, on the
    task and dataset that carry contribution C2.

PART B -- the final deployment model.
    Differences from nb06's model, both deliberate:

    * UMAFall is EXCLUDED from training. nb01 established its pocket smartphone logs
      no gyroscope, so its three gyro channels are identically zero. Training on it
      teaches the model that a dead gyroscope is normal, which is exactly wrong for a
      device whose MPU6050 always has one. Trained on SisFall + KFall + FallAllD.
    * Instance normalisation lives in the PIPELINE, not the graph. nb06 showed the
      in-graph version collapses under INT8 (macro-F1 0.668 -> 0.359). The firmware
      does the same normalisation in float before quantising.

    Reports trial-level metrics on the protocol Yu et al. use for KFall, so the
    numbers are comparable with published work, and sweeps the operating point so the
    threshold burned into the firmware is chosen on evidence.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working")

_fdlib_root = next((p.parent for p in Path("/kaggle/input").rglob("fdlib/__init__.py")), None)
if _fdlib_root is None:
    raise SystemExit("fdlib dataset not attached")
sys.path.insert(0, str(_fdlib_root.parent))

from fdlib import config as C  # noqa: E402
from fdlib.baselines import make_random_forest, make_svm  # noqa: E402
from fdlib.cv import class_weights, grouped_folds, inner_val_split  # noqa: E402
from fdlib.experiment import (  # noqa: E402
    append_row, evaluate, keras_fit_predict, load_corpus, run_folds,
    sklearn_fit_predict, smv_fit_predict,
)
from fdlib.metrics import classification_metrics, macro_f1_3class  # noqa: E402
from fdlib.models import (  # noqa: E402
    MacroF1, cnn_1d, cnn_lstm, compile_model, count_params, proposed_cnn, set_seeds,
)
from fdlib.preprocess import instance_normalise  # noqa: E402
from fdlib.tflite_export import (  # noqa: E402
    estimate_arena_bytes, quantisation_report, to_c_array, to_int8_tflite,
)

TASK = "preimpact"
CORPUS = next(Path("/kaggle/input").rglob(f"windows_{TASK}.npz"))
TRAIN_SETS = ["sisfall", "kfall", "fallalld"]   # UMAFall excluded: no gyroscope
HZ = C.TARGET_HZ

corpus = load_corpus(CORPUS)
print(f"corpus {corpus['X'].shape}  signature {C.preprocess_signature()}")
print(f"proposed params {count_params(proposed_cnn()):,}")


# ============================================================ PART A: baselines
print("\n" + "=" * 74)
print("PART A -- pre-impact baselines on KFall, 5-fold subject-grouped")
print("=" * 74)

kf = corpus["dataset"] == "kfall"
kfc = {k: v[kf] for k, v in corpus.items()}
print(f"KFall windows {kfc['X'].shape}  subjects {len(np.unique(kfc['subject']))}")
print(f"classes {np.bincount(kfc['y'], minlength=3).tolist()} (bkg, alert, fall)")

CSV_A = OUT / "results_e1_preimpact.csv"
SVM_MAX = 20000


def subsample(fn):
    def _w(Xtr, ytr, Xva, yva, Xte):
        if len(Xtr) > SVM_MAX:
            rng = np.random.default_rng(C.SEED)
            i = rng.choice(len(Xtr), SVM_MAX, replace=False)
            Xtr, ytr = Xtr[i], ytr[i]
        p, info = fn(Xtr, ytr, Xva, yva, Xte)
        info["train_subsampled_to"] = int(len(Xtr))
        return p, info
    return _w


COMPARATORS = {
    "smv_threshold": smv_fit_predict(),
    "svm": subsample(sklearn_fit_predict(make_svm)),
    "random_forest": sklearn_fit_predict(make_random_forest),
    "cnn_1d": keras_fit_predict(lambda: cnn_1d()),
    "cnn_lstm": keras_fit_predict(lambda: cnn_lstm()),
    "proposed": keras_fit_predict(lambda: proposed_cnn()),
}

for name, fn in COMPARATORS.items():
    print(f"\n{name}:")
    t0 = time.time()
    try:
        run_folds(kfc, grouped_folds(kfc["subject"], C.GROUPED_CV_SPLITS), name, fn, CSV_A,
                  extra_cols={"protocol": "grouped5", "dataset": "kfall", "task": TASK})
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED {type(e).__name__}: {e}")
        continue
    print(f"  done in {(time.time() - t0) / 60:.1f} min")

if CSV_A.exists():
    dfa = pd.read_csv(CSV_A)
    tbl = (dfa.groupby("model")
              .agg(folds=("fold", "count"), params=("params", "max"),
                   sensitivity=("bin_sensitivity", "mean"), sens_sd=("bin_sensitivity", "std"),
                   specificity=("bin_specificity", "mean"),
                   macro_f1=("macro_f1_3class", "mean"), f1_sd=("macro_f1_3class", "std"),
                   auc=("bin_auc", "mean"))
              .round(4).sort_values("macro_f1", ascending=False))
    print("\nTABLE I(b) -- pre-impact baselines, KFall")
    print(tbl.to_string())
    tbl.to_csv(OUT / "table_Ib.csv")
    (OUT / "table_Ib.md").write_text(
        "# Table I(b) -- within-dataset pre-impact baselines (KFall)\n\n"
        "5-fold subject-grouped CV, three-class pre-impact task.\n\n"
        + tbl.to_markdown() + "\n")


# ========================================================= PART B: final model
print("\n" + "=" * 74)
print("PART B -- final deployment model")
print("=" * 74)

sel = np.isin(corpus["dataset"], TRAIN_SETS)
X, y = corpus["X"][sel], corpus["y"][sel]
subjects, datasets = corpus["subject"][sel], corpus["dataset"][sel]
trials, edges = corpus["trial"][sel], corpus["edge_impact"][sel]
print(f"training pool {X.shape} from {TRAIN_SETS} (UMAFall excluded: no gyroscope)")
print(f"classes {np.bincount(y, minlength=3).tolist()} (bkg, alert, fall)")

rng = np.random.default_rng(C.SEED)
subs = np.unique(subjects)
rng.shuffle(subs)
test_subs = set(subs[: max(1, int(round(len(subs) * 0.2)))].tolist())
te = np.array([s in test_subs for s in subjects])
tr_idx, te_idx = np.where(~te)[0], np.where(te)[0]
print(f"train {len(tr_idx):,} windows / {len(subs) - len(test_subs)} subjects")
print(f"test  {len(te_idx):,} windows / {len(test_subs)} subjects")

Xn = instance_normalise(X)          # normalisation in the pipeline, not the graph
inner_tr, inner_va = inner_val_split(subjects, tr_idx)

set_seeds(C.SEED)
model = compile_model(proposed_cnn(instance_norm=False),
                      steps_per_epoch=max(1, len(inner_tr) // C.BATCH_SIZE),
                      epochs=C.MAX_EPOCHS)
model.summary()
cb = MacroF1(Xn[inner_va], y[inner_va])
model.fit(Xn[inner_tr], y[inner_tr], validation_data=(Xn[inner_va], y[inner_va]),
          epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE, verbose=2,
          class_weight=class_weights(y[inner_tr]), callbacks=[cb])
model.save(OUT / "model_fp32.keras")

to_int8_tflite(model, Xn[inner_tr], OUT / "model.tflite")
rep = quantisation_report(model, OUT / "model.tflite", Xn[te_idx], y[te_idx])
rep.update({
    "params": count_params(model),
    "estimated_arena_bytes": estimate_arena_bytes(OUT / "model.tflite"),
    "task": TASK, "train_datasets": TRAIN_SETS,
    "signature": C.preprocess_signature(),
    "test_subjects": sorted(test_subs),
    "instance_norm": "pipeline",
})
rep["estimated_arena_kb"] = round(rep["estimated_arena_bytes"] / 1024, 1)
print("\n" + json.dumps({k: v for k, v in rep.items() if k != "test_subjects"}, indent=2))
(OUT / "quantisation_report.json").write_text(json.dumps(rep, indent=2))
to_c_array(OUT / "model.tflite", OUT / "model.h")


# ------------------------------------------------- operating point, trial level
print("\n" + "=" * 74)
print("Operating point sweep -- trial-level, the protocol Yu et al. use for KFall")
print("=" * 74)

import tensorflow as tf  # noqa: E402

interp = tf.lite.Interpreter(model_path=str(OUT / "model.tflite"))
interp.allocate_tensors()
inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]


def int8_predict(A: np.ndarray) -> np.ndarray:
    s, z = inp["quantization"]
    os_, oz = outp["quantization"]
    out = np.zeros((len(A), C.N_CLASSES), np.float32)
    for i in range(len(A)):
        q = np.clip(np.round(A[i] / s) + z, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None, ...])
        interp.invoke()
        out[i] = (interp.get_tensor(outp["index"])[0].astype(np.float32) - oz) * os_
    return out


prob = int8_predict(Xn[te_idx])
yt, tt = y[te_idx], trials[te_idx]
edge, impact = edges[te_idx][:, 0], edges[te_idx][:, 1]

rows = []
for thr in (0.3, 0.5, 0.7, 0.9, 0.95):
    for k in (1, 2, 3, 4):
        score = prob[:, 1:].sum(1)
        above = score >= thr
        det = tot_f = clean = tot_a = 0
        leads = []
        for tid in np.unique(tt):
            m = tt == tid
            o = np.argsort(edge[m])
            s, e = above[m][o], edge[m][o]
            if k > 1:
                run, c = np.zeros(len(s), bool), 0
                for i, a in enumerate(s):
                    c = c + 1 if a else 0
                    run[i] = c >= k
                s = run
            fired = np.where(s)[0]
            imp = impact[m]
            if len(imp) and imp[0] >= 0:
                tot_f += 1
                if len(fired):
                    det += 1
                    leads.append((int(imp[0]) - int(e[fired[0]])) * 1000.0 / HZ)
            else:
                tot_a += 1
                if not len(fired):
                    clean += 1
        L = np.asarray(leads); pre = L[L > 0]
        rows.append({
            "threshold": thr, "k": k,
            "sens_trial": round(det / tot_f, 4) if tot_f else 0,
            "spec_trial": round(clean / tot_a, 4) if tot_a else 0,
            "mean_lead_ms": round(float(pre.mean()), 1) if len(pre) else float("nan"),
            "std_lead_ms": round(float(pre.std()), 1) if len(pre) else float("nan"),
            "preimpact_frac": round(len(pre) / max(det, 1), 4),
            "fall_trials": tot_f, "adl_trials": tot_a,
        })

sweep = pd.DataFrame(rows)
sweep["youden"] = sweep["sens_trial"] + sweep["spec_trial"] - 1
sweep = sweep.sort_values("youden", ascending=False)
print(sweep.to_string(index=False))
sweep.to_csv(OUT / "final_operating_points.csv", index=False)

best = sweep.iloc[0]
print(f"\nRECOMMENDED FIRMWARE SETTING: threshold {best['threshold']}, k {int(best['k'])}")
print(f"  trial sensitivity {best['sens_trial']}  specificity {best['spec_trial']}")
print(f"  mean lead {best['mean_lead_ms']} ms  ({best['preimpact_frac']:.1%} of detections "
      "are pre-impact)")
(OUT / "final_model_card.md").write_text(
    "# Final deployment model\n\n"
    f"- Task: three-class pre-impact (bkg / alert / fall)\n"
    f"- Trained on: {', '.join(TRAIN_SETS)} (UMAFall excluded -- no gyroscope)\n"
    f"- Parameters: {rep['params']:,}\n"
    f"- INT8 size: {rep['model_kb']} KB\n"
    f"- Estimated tensor arena: {rep['estimated_arena_kb']} KB\n"
    f"- FP32 macro-F1 {rep['fp32_macro_f1']}, INT8 {rep['int8_macro_f1']} "
    f"(drop {rep['macro_f1_drop_pp']} pp, agreement {rep['agreement']:.4f})\n"
    f"- Normalisation: per-window instance norm, in the PIPELINE (firmware does it in float)\n\n"
    "## Recommended firmware setting\n\n"
    f"`gThreshold = {best['threshold']}`, `gAgreeK = {int(best['k'])}`\n\n"
    f"Trial-level on held-out subjects: sensitivity {best['sens_trial']}, "
    f"specificity {best['spec_trial']}, mean lead {best['mean_lead_ms']} ms.\n\n"
    "## Operating points\n\n" + sweep.head(12).to_markdown(index=False) + "\n")
print("\nwrote model.tflite, model.h, final_operating_points.csv, final_model_card.md")
