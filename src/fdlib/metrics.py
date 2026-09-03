"""Metrics, including the one this paper actually turns on: lead time.

Accuracy on a fall dataset is close to meaningless -- background dominates so
heavily that predicting "no fall" forever scores well. The reported quantities are
sensitivity, specificity, macro-F1 and AUC, and for contribution C2, lead time and
false alarms per hour of ADL data.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from . import config as C


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Binary metrics derived from the three-class head by merging ALERT and FALL."""
    y_true_b = (np.asarray(y_true) > C.BKG).astype(int)
    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        score = y_prob[:, 1:].sum(1) if y_prob.shape[1] == 3 else y_prob[:, 1]
    else:
        score = np.asarray(y_prob).ravel()
    pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_b, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0

    out = {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if len(np.unique(y_true_b)) > 1:
        out["auc"] = float(roc_auc_score(y_true_b, score))
        out["auprc"] = float(average_precision_score(y_true_b, score))
    else:
        out["auc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def macro_f1_3class(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-F1 over the classes actually PRESENT in y_true -- the early-stopping criterion.

    Averaging over all three labels regardless of support silently scores 0 for any
    absent class. On the post-fall task, where ALERT never occurs, that caps macro-F1
    at two thirds of its real value and -- far worse -- makes early stopping chase a
    distorted target. Restricting to observed labels keeps the metric comparable within
    a task; it is not comparable *across* tasks, which is why the task is recorded in
    every results row.
    """
    y_true = np.asarray(y_true)
    pred = np.asarray(y_prob).argmax(1)
    labels = sorted(set(np.unique(y_true).tolist()))
    return float(f1_score(y_true, pred, average="macro", labels=labels, zero_division=0))


def per_class_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, dict[str, float]]:
    pred = np.asarray(y_prob).argmax(1)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    out = {}
    for i, name in enumerate(C.CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        pre = tp / (tp + fp) if (tp + fp) else 0.0
        out[name] = {
            "recall": float(rec),
            "precision": float(pre),
            "f1": float(2 * pre * rec / (pre + rec)) if (pre + rec) else 0.0,
            "support": int(cm[i].sum()),
        }
    return out


# --------------------------------------------------------------------- lead time

def lead_times(
    y_prob: np.ndarray,
    right_edge: np.ndarray,
    impact_idx: np.ndarray,
    trial_ids: np.ndarray,
    threshold: float,
    hz: int = C.TARGET_HZ,
) -> np.ndarray:
    """Milliseconds between the model's FIRST alarm and the labelled impact.

    One value per fall trial. Computed on the FIRST crossing, not the best one --
    a detector that fires late but confidently has not detected anything pre-impact.

    Positive  = fired before impact, which is the whole point.
    Negative  = fired after impact, i.e. post-hoc detection.
    NaN       = never fired on that trial (a miss; counted by `detection_rate`).
    """
    score = y_prob[:, 1:].sum(1) if y_prob.ndim == 2 and y_prob.shape[1] == 3 else np.asarray(y_prob).ravel()
    fired = score >= threshold

    out = []
    for tid in np.unique(trial_ids):
        m = trial_ids == tid
        imp = impact_idx[m]
        if not len(imp) or imp[0] < 0:
            continue  # not a fall trial, or no impact annotation
        edges = right_edge[m]
        hits = np.where(fired[m])[0]
        if not len(hits):
            out.append(np.nan)
            continue
        # The FIRST alarm, by window right edge -- not the most confident one. A
        # detector that fires late but confidently has not detected anything
        # pre-impact, and scoring it on its best window would hide that.
        first_edge = int(edges[hits].min())
        out.append((int(imp[0]) - first_edge) * 1000.0 / hz)
    return np.asarray(out, dtype=np.float64)


def lead_time_summary(lt: np.ndarray) -> dict[str, float]:
    """Report the distribution, not only the mean -- the plan is explicit about this."""
    n = len(lt)
    ok = lt[~np.isnan(lt)]
    pre = ok[ok > 0]
    return {
        "n_fall_trials": int(n),
        "detection_rate": float(len(ok) / n) if n else 0.0,
        "preimpact_rate": float(len(pre) / n) if n else 0.0,
        "mean_ms": float(np.mean(pre)) if len(pre) else float("nan"),
        "std_ms": float(np.std(pre)) if len(pre) else float("nan"),
        "median_ms": float(np.median(pre)) if len(pre) else float("nan"),
        "p25_ms": float(np.percentile(pre, 25)) if len(pre) else float("nan"),
        "p75_ms": float(np.percentile(pre, 75)) if len(pre) else float("nan"),
        "min_ms": float(np.min(pre)) if len(pre) else float("nan"),
        "max_ms": float(np.max(pre)) if len(pre) else float("nan"),
    }


def false_alarms_per_hour(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    stride_sec: float = C.STRIDE_SEC,
) -> float:
    """False alarms per hour of ADL data.

    This is the metric that decides whether a device is wearable at all, and most
    papers omit it. A detector firing forty times a day is useless however good its
    lead time looks.
    """
    score = y_prob[:, 1:].sum(1) if y_prob.ndim == 2 and y_prob.shape[1] == 3 else np.asarray(y_prob).ravel()
    adl = np.asarray(y_true) == C.BKG
    n_adl = int(adl.sum())
    if not n_adl:
        return float("nan")
    fp = int((score[adl] >= threshold).sum())
    hours = n_adl * stride_sec / 3600.0
    return float(fp / hours) if hours else float("nan")


def sweep_operating_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    right_edge: np.ndarray,
    impact_idx: np.ndarray,
    trial_ids: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """The lead-time-versus-false-alarm curve of E3.

    Presenting this trade-off, rather than a single operating point, is what
    distinguishes a deployable claim from a cherry-picked one.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.00, 0.025), 4)
    rows = []
    for t in thresholds:
        lt = lead_times(y_prob, right_edge, impact_idx, trial_ids, float(t))
        row = {"threshold": float(t)}
        row.update(lead_time_summary(lt))
        row["false_alarms_per_hour"] = false_alarms_per_hour(y_true, y_prob, float(t))
        row.update({k: v for k, v in classification_metrics(y_true, y_prob, float(t)).items()
                    if k in ("sensitivity", "specificity", "f1")})
        rows.append(row)
    return rows
