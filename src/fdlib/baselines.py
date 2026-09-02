"""Non-deep comparators for Table I.

The plan's point in E1 is that competitiveness must be proved before any claim about
generalisation is made: weak within-dataset numbers make weak cross-dataset numbers
unpublishable. These are the three classical comparators.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import config as C


# ------------------------------------------------------------- hand-crafted features

def window_features(X: np.ndarray) -> np.ndarray:
    """The standard feature set this literature uses, on (W, T, 6) windows.

    Per channel: mean, std, min, max, range, RMS, skew, kurtosis, mean |diff|.
    Plus, on the acceleration vector magnitude: the same statistics again, because
    magnitude is what threshold detectors actually key on.
    """
    feats = []
    acc_mag = np.linalg.norm(X[:, :, 0:3], axis=2)
    gyr_mag = np.linalg.norm(X[:, :, 3:6], axis=2)
    streams = [X[:, :, c] for c in range(X.shape[2])] + [acc_mag, gyr_mag]

    for s in streams:
        d = np.diff(s, axis=1)
        feats.extend([
            s.mean(1), s.std(1), s.min(1), s.max(1), s.max(1) - s.min(1),
            np.sqrt((s ** 2).mean(1)),
            stats.skew(s, axis=1, bias=False),
            stats.kurtosis(s, axis=1, bias=False),
            np.abs(d).mean(1),
        ])
    out = np.stack(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ SMV threshold

class SMVThreshold:
    """Signal magnitude vector threshold -- the classical detector, and the floor.

    Fires when |a| leaves a band around 1 g. The threshold is fitted on the training
    split by sweeping candidates and keeping the best macro-F1; fitting it on test
    data would flatter it, and the point of including it is an honest floor.
    """

    def __init__(self) -> None:
        self.lo = 0.5
        self.hi = 2.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SMVThreshold":
        from sklearn.metrics import f1_score
        mag = np.linalg.norm(X[:, :, 0:3], axis=2)
        peak, trough = mag.max(1), mag.min(1)
        yb = (y > C.BKG).astype(int)
        best = (-1.0, self.lo, self.hi)
        for hi in np.arange(1.4, 4.01, 0.1):
            for lo in np.arange(0.1, 0.91, 0.1):
                pred = ((peak >= hi) | (trough <= lo)).astype(int)
                f1 = f1_score(yb, pred, zero_division=0)
                if f1 > best[0]:
                    best = (f1, lo, hi)
        _, self.lo, self.hi = best
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        mag = np.linalg.norm(X[:, :, 0:3], axis=2)
        peak, trough = mag.max(1), mag.min(1)
        # a graded score so AUC is meaningful, not just a hard decision
        s = np.maximum((peak - self.hi) / max(self.hi, 1e-6),
                       (self.lo - trough) / max(self.lo, 1e-6))
        s = 1.0 / (1.0 + np.exp(-4.0 * s))
        return np.stack([1 - s, s * 0.5, s * 0.5], axis=1).astype(np.float32)


# ------------------------------------------------------------------ sklearn wrappers

def make_svm():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(
        StandardScaler(),
        SVC(C=10.0, gamma="scale", kernel="rbf", probability=True,
            class_weight="balanced", random_state=C.SEED, cache_size=1000),
    )


def make_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1,
        class_weight="balanced_subsample", random_state=C.SEED,
    )


def align_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int = C.N_CLASSES) -> np.ndarray:
    """Expand an sklearn predict_proba back to the full 3-class layout.

    A fold whose training split happens to contain no ALERT windows returns a
    2-column matrix; silently treating column 1 as ALERT would scramble the metrics.
    """
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    for j, c in enumerate(classes):
        out[:, int(c)] = proba[:, j]
    return out


SKLEARN_BASELINES = {
    "svm": make_svm,
    "random_forest": make_random_forest,
}
