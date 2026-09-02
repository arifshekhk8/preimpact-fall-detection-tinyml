"""nb02 -- E1, the within-dataset baseline (Table I).

Purpose, in the plan's words: prove competitiveness before claiming anything about
generalisation. Weak within-dataset numbers make weak cross-dataset numbers
unpublishable, so this runs first and its result is a gate on the rest.

Protocol:
  * six comparators -- SMV threshold, SVM, Random Forest, 1D-CNN, CNN-LSTM, and the
    proposed separable CNN -- under 5-fold SUBJECT-GROUPED CV;
  * plus the proposed model under full leave-one-subject-out, which is the headline
    number reviewers expect.

Every fold appends its own row to results_e1.csv before the next begins, and folds
already present are skipped, so a session that dies part-way costs one fold rather
than the run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working")

_fdlib_root = next((p.parent for p in Path("/kaggle/input").rglob("fdlib/__init__.py")), None)
if _fdlib_root is None:
    raise SystemExit("fdlib dataset not attached")
sys.path.insert(0, str(_fdlib_root.parent))

from fdlib import config as C  # noqa: E402
from fdlib.baselines import make_random_forest, make_svm  # noqa: E402
from fdlib.cv import grouped_folds, loso_folds  # noqa: E402
from fdlib.experiment import (  # noqa: E402
    keras_fit_predict, load_corpus, run_folds, sklearn_fit_predict, smv_fit_predict,
    summarise,
)
from fdlib.models import cnn_1d, cnn_lstm, count_params, proposed_cnn  # noqa: E402

# ---------------------------------------------------------------- configuration
# E1 is a WITHIN-dataset baseline. The plan runs it on the primary corpus; with the
# Enhanced annotations unrecoverable (nb00), that is the original SisFall on the
# post-fall task.
DATASET = "sisfall"
TASK = "postfall"
CORPUS = next(Path("/kaggle/input").rglob(f"windows_{TASK}.npz"))
CSV = OUT / "results_e1.csv"
START_FOLD = 0

# A smoke run exercises every code path -- all six comparators, both protocols, the
# CSV append and the table build -- in a few minutes, so a crash is found before
# seven GPU-hours are spent discovering it at fold 31.
SMOKE = False
if SMOKE:
    C.MAX_EPOCHS = 2
    C.EARLY_STOP_PATIENCE = 1

print(f"fdlib from {_fdlib_root.parent}")
print(f"corpus  {CORPUS}")

corpus = load_corpus(CORPUS)
mask = corpus["dataset"] == DATASET
corpus = {k: v[mask] for k, v in corpus.items()}

if SMOKE:
    keep = np.isin(corpus["subject"], np.unique(corpus["subject"])[:6])
    corpus = {k: v[keep] for k, v in corpus.items()}
    print("SMOKE RUN -- 6 subjects, 2 epochs. Results are not publishable.")

subjects = corpus["subject"]
n_subj = len(np.unique(subjects))

print(f"\nE1 on {DATASET} / {TASK}")
print(f"  windows {corpus['X'].shape}  subjects {n_subj}")
print(f"  class counts {np.bincount(corpus['y'], minlength=3).tolist()}  (bkg, alert, fall)")
print(f"  preprocess signature {C.preprocess_signature()}")

# ------------------------------------------------------------------ comparators
COMPARATORS = {
    "smv_threshold": smv_fit_predict(),
    "svm": sklearn_fit_predict(make_svm),
    "random_forest": sklearn_fit_predict(make_random_forest),
    "cnn_1d": keras_fit_predict(lambda: cnn_1d()),
    "cnn_lstm": keras_fit_predict(lambda: cnn_lstm()),
    "proposed": keras_fit_predict(lambda: proposed_cnn()),
}

# The SVM is O(n^2) in the number of windows and will not finish on ~114k of them in a
# 12-hour session. It is fitted on a subject-preserving subsample, and the subsample
# rate is recorded so the comparison is not quietly unfair.
SVM_MAX_TRAIN = 20000


def subsample_svm(fn):
    def _wrapped(Xtr, ytr, Xva, yva, Xte):
        if len(Xtr) > SVM_MAX_TRAIN:
            rng = np.random.default_rng(C.SEED)
            idx = rng.choice(len(Xtr), SVM_MAX_TRAIN, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        probs, info = fn(Xtr, ytr, Xva, yva, Xte)
        info["train_subsampled_to"] = int(len(Xtr))
        return probs, info
    return _wrapped


COMPARATORS["svm"] = subsample_svm(COMPARATORS["svm"])

print(f"\nproposed model parameter count: {count_params(proposed_cnn()):,} "
      f"(target ~25,000)")

# --------------------------------------------------------- 5-fold grouped sweep
print("\n" + "=" * 74)
print("E1a -- 5-fold subject-grouped CV, all comparators")
print("=" * 74)

for name, fn in COMPARATORS.items():
    print(f"\n{name}:")
    t0 = time.time()
    run_folds(
        corpus, grouped_folds(subjects, C.GROUPED_CV_SPLITS), name, fn, CSV,
        start_fold=START_FOLD,
        extra_cols={"protocol": "grouped5", "dataset": DATASET, "task": TASK},
    )
    print(f"  {name} done in {(time.time() - t0) / 60:.1f} min")

# ------------------------------------------------- full LOSO for proposed model
print("\n" + "=" * 74)
print(f"E1b -- full leave-one-subject-out ({n_subj} folds), proposed model only")
print("=" * 74)

run_folds(
    corpus, loso_folds(subjects), "proposed_loso", COMPARATORS["proposed"], CSV,
    start_fold=START_FOLD,
    extra_cols={"protocol": "loso", "dataset": DATASET, "task": TASK},
)

# ---------------------------------------------------------------------- summary
summary = summarise(CSV, OUT / "results_e1_summary.json")
print("\n" + "=" * 74)
print("TABLE I -- within-dataset baseline")
print("=" * 74)

import pandas as pd  # noqa: E402

df = pd.read_csv(CSV)
tbl = (df.groupby(["model", "protocol"])
         .agg(folds=("fold", "count"),
              params=("params", "max"),
              sensitivity=("bin_sensitivity", "mean"),
              sens_sd=("bin_sensitivity", "std"),
              specificity=("bin_specificity", "mean"),
              spec_sd=("bin_specificity", "std"),
              macro_f1=("macro_f1_3class", "mean"),
              f1_sd=("macro_f1_3class", "std"),
              auc=("bin_auc", "mean"))
         .round(4))
print(tbl.to_string())
tbl.to_csv(OUT / "table_I.csv")
(OUT / "table_I.md").write_text(
    f"# Table I -- within-dataset baseline ({DATASET}, {TASK})\n\n"
    f"Baselines: 5-fold subject-grouped CV. Proposed model additionally at full "
    f"{n_subj}-fold LOSO.\n\n" + tbl.to_markdown() + "\n"
)
print("\nwrote results_e1.csv, table_I.csv, table_I.md")
