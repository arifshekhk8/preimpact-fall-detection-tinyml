"""Fold runner shared by E1, E2, E3 and E5.

Two rules from section 7.2 of the plan are structural here rather than optional,
because both exist to survive a Kaggle session dying mid-run:

  * results are appended to CSV after EVERY fold, never at the end;
  * every entry point resumes -- a fold already present in the CSV is skipped, so a
    session that dies at fold 34 of 38 costs one fold, not the run.

Every row carries the preprocessing signature and the model name, so a results file
can never be silently mixed across preprocessing versions.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import config as C
from . import metrics as M
from .cv import class_weights, inner_val_split


def load_corpus(path: str | Path) -> dict[str, np.ndarray]:
    """Load a cached window corpus and refuse a stale preprocessing version."""
    z = np.load(path, allow_pickle=True)
    sig = str(z["signature"]) if "signature" in z.files else None
    if sig and sig != C.preprocess_signature():
        raise ValueError(
            f"corpus at {path} was built under preprocessing signature {sig} but the "
            f"current contract is {C.preprocess_signature()}. Rebuild it -- mixing "
            "preprocessing versions invalidates every comparison."
        )
    return {k: z[k] for k in z.files if k != "signature"}


def completed_folds(csv_path: Path, model: str) -> set[str]:
    """Folds already recorded for this model, so a restart resumes rather than repeats."""
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("model") == model:
                done.add(row.get("fold", ""))
    return done


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, extra: dict | None = None) -> dict:
    """The standard metric block written for every fold."""
    out = {
        "macro_f1_3class": M.macro_f1_3class(y_true, y_prob),
        **{f"bin_{k}": v for k, v in M.classification_metrics(y_true, y_prob).items()},
    }
    for name, blk in M.per_class_report(y_true, y_prob).items():
        for k, v in blk.items():
            out[f"{name}_{k}"] = v
    if extra:
        out.update(extra)
    return out


def run_folds(
    corpus: dict[str, np.ndarray],
    folds: Iterable[tuple[str, np.ndarray, np.ndarray]],
    model_name: str,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, dict]],
    csv_path: Path,
    start_fold: int = 0,
    extra_cols: dict | None = None,
) -> None:
    """Run one model across a fold generator, appending a row per fold.

    `fit_predict(Xtr, ytr, Xva, yva, Xte) -> (probabilities on Xte, info dict)`
    receives an inner validation split carved out BY SUBJECT, because early stopping
    on a window-wise split leaks the same way a window-wise test split does.
    """
    X, y = corpus["X"], corpus["y"]
    subjects = corpus["subject"]
    done = completed_folds(csv_path, model_name)

    for i, (fold_name, tr, te) in enumerate(folds):
        if i < start_fold:
            continue
        if fold_name in done:
            print(f"  [{model_name}] fold {fold_name}: already done, skipping", flush=True)
            continue

        t0 = time.time()
        inner_tr, inner_va = inner_val_split(subjects, tr)
        if not len(inner_va):
            inner_tr, inner_va = tr, tr[:1]

        probs, info = fit_predict(X[inner_tr], y[inner_tr], X[inner_va], y[inner_va], X[te])
        row = {
            "model": model_name,
            "fold": fold_name,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_subjects": int(len(np.unique(subjects[tr]))),
            "test_subjects": int(len(np.unique(subjects[te]))),
            "seconds": round(time.time() - t0, 1),
            "signature": C.preprocess_signature(),
            **evaluate(y[te], probs, {**(extra_cols or {}), **info}),
        }
        append_row(csv_path, row)
        print(f"  [{model_name}] fold {fold_name}: macro-F1 {row['macro_f1_3class']:.4f}  "
              f"sens {row['bin_sensitivity']:.3f}  spec {row['bin_specificity']:.3f}  "
              f"({row['seconds']:.0f}s)", flush=True)


# ------------------------------------------------------------------ keras adapter

def keras_fit_predict(build: Callable[[], "object"], epochs: int = C.MAX_EPOCHS,
                      batch: int = C.BATCH_SIZE, verbose: int = 0):
    """Wrap a Keras model factory into the `fit_predict` signature."""
    from .models import MacroF1, compile_model, count_params, set_seeds

    def _fn(Xtr, ytr, Xva, yva, Xte):
        set_seeds(C.SEED)
        model = compile_model(build(), steps_per_epoch=max(1, len(Xtr) // batch), epochs=epochs)
        cb = MacroF1(Xva, yva)
        model.fit(
            Xtr, ytr,
            validation_data=(Xva, yva),
            epochs=epochs, batch_size=batch, verbose=verbose,
            class_weight=class_weights(ytr),
            callbacks=[cb],
        )
        probs = model.predict(Xte, batch_size=512, verbose=0)
        return probs, {"params": count_params(model), "best_val_macro_f1": round(cb.best, 4)}

    return _fn


def sklearn_fit_predict(build: Callable[[], "object"], use_features: bool = True):
    """Wrap an sklearn estimator into the `fit_predict` signature."""
    from .baselines import align_proba, window_features

    def _fn(Xtr, ytr, Xva, yva, Xte):
        ftr = window_features(Xtr) if use_features else Xtr.reshape(len(Xtr), -1)
        fte = window_features(Xte) if use_features else Xte.reshape(len(Xte), -1)
        est = build()
        est.fit(ftr, ytr)
        proba = est.predict_proba(fte)
        classes = getattr(est, "classes_", None)
        if classes is None and hasattr(est, "steps"):
            classes = est.steps[-1][1].classes_
        n_params = int(sum(t.tree_.node_count for t in getattr(est, "estimators_", []))) or 0
        return align_proba(proba, classes), {"params": n_params}

    return _fn


def smv_fit_predict():
    """The SMV threshold detector, fitted on the training split only."""
    from .baselines import SMVThreshold

    def _fn(Xtr, ytr, Xva, yva, Xte):
        det = SMVThreshold().fit(Xtr, ytr)
        return det.predict_proba(Xte), {"params": 2, "lo": det.lo, "hi": det.hi}

    return _fn


def summarise(csv_path: Path, out_json: Path | None = None) -> dict:
    """Mean and standard deviation across folds, per model -- Table I's content."""
    import pandas as pd

    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    num = df.select_dtypes("number").columns
    agg = df.groupby("model")[list(num)].agg(["mean", "std", "count"])
    out = json.loads(agg.to_json())
    if out_json:
        out_json.write_text(json.dumps(out, indent=2))
    return out
