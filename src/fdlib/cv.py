"""Fold generators -- the tiered cross-validation strategy of section 5.1.

The guarantee that matters, and the only one, is that no subject appears in both
train and test. Full leave-one-subject-out is the gold standard but is wasteful
applied to everything, so:

    baselines and all ablations  -> GroupKFold(5) grouped by subject (~8x cheaper)
    the proposed model, headline -> full leave-one-subject-out
    cross-dataset (E2)           -> leave-one-dataset-out; the dataset is the group

A random window split would place near-identical windows on both sides of the
boundary -- at 75% overlap, window i and window i+1 share three quarters of their
samples -- and would report accuracy above 99% that means nothing.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold

from . import config as C


def grouped_folds(
    subjects: np.ndarray, n_splits: int = C.GROUPED_CV_SPLITS
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Subject-grouped k-fold. Yields (fold_name, train_idx, test_idx)."""
    idx = np.arange(len(subjects))
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(idx, groups=subjects)):
        _assert_disjoint(subjects, tr, te, f"grouped fold {k}")
        yield f"fold{k}", tr, te


def loso_folds(subjects: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Full leave-one-subject-out. One fold per distinct subject."""
    uniq = np.unique(subjects)
    for s in uniq:
        te = np.where(subjects == s)[0]
        tr = np.where(subjects != s)[0]
        if not len(te) or not len(tr):
            continue
        yield str(s), tr, te


def lodo_folds(
    datasets: np.ndarray, folds: dict | None = None
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-dataset-out. Train on two, test on a third, no fine-tuning.

    UMAFall's fold trains on all three others and is reported exactly once -- it is
    the only genuinely 'never seen, never tuned' row in the results table, and it
    stops being that the moment it is used for model selection.
    """
    folds = folds or C.LODO_FOLDS
    for name, spec in folds.items():
        tr = np.where(np.isin(datasets, spec["train"]))[0]
        te = np.where(datasets == spec["test"])[0]
        if not len(te):
            continue
        if not len(tr):
            continue
        yield name, tr, te


def inner_val_split(
    subjects: np.ndarray, train_idx: np.ndarray, frac: float = 0.2, seed: int = C.SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation split out of the training indices, again by subject.

    Early stopping needs a validation set, and taking it by window would leak the
    test-set problem one level down into model selection.
    """
    rng = np.random.default_rng(seed)
    tr_subj = np.unique(subjects[train_idx])
    rng.shuffle(tr_subj)
    n_val = max(1, int(round(len(tr_subj) * frac)))
    val_subj = set(tr_subj[:n_val].tolist())
    mask = np.array([s in val_subj for s in subjects[train_idx]])
    return train_idx[~mask], train_idx[mask]


def _assert_disjoint(groups: np.ndarray, tr: np.ndarray, te: np.ndarray, where: str) -> None:
    overlap = set(np.unique(groups[tr])) & set(np.unique(groups[te]))
    if overlap:
        raise AssertionError(f"{where}: subject leakage across the split: {sorted(overlap)}")


def class_weights(y: np.ndarray, n_classes: int = C.N_CLASSES) -> dict[int, float]:
    """Inverse-frequency weights.

    The plan is explicit that imbalance is handled by weighting and NOT by
    oversampling: at 75% overlap, oversampling duplicates near-identical windows
    across the split boundary and manufactures the leakage the folds exist to prevent.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return {i: float(w[i]) for i in range(n_classes)}
