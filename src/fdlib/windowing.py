"""Windowing and the two labelling rules -- section 4.1 steps 4 and 5.

A trial arrives as a decimated (N, 6) array at 50 Hz plus, where the source
provides them, an alert interval and an impact index. It leaves as (W, 100, 6)
windows with a label per window and a subject ID per window.

The subject ID travels with every window all the way to the fold generator. That is
what makes subject-grouped splitting possible, and subject-grouped splitting is the
only reason any accuracy number in this project means anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

# How long after the impact frame still counts as the FALL class, in seconds.
# One window's worth: the event, not the aftermath.
FALL_EVENT_SEC = C.WINDOW_SEC


@dataclass
class Trial:
    """One recording, already decimated to 50 Hz and in g / deg/s."""

    signal: np.ndarray          # (N, 6) float32
    subject: str                # e.g. "sisfall:SA06" -- namespaced, so IDs never collide
    dataset: str
    trial_id: str
    is_fall: bool
    impact_idx: int | None = None          # sample index of body-ground impact
    alert_span: tuple[int, int] | None = None  # [start, end) of the pre-impact interval
    per_sample: np.ndarray | None = None   # (N,) in {BKG, ALERT, FALL}, when annotated


def window_trial(
    trial: Trial,
    task: str = "preimpact",
    window_len: int = C.WINDOW_LEN,
    stride: int = C.STRIDE_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice one trial into windows and label each one.

    task="postfall":  a window is positive if it CONTAINS the impact.
    task="preimpact": a window is positive if its RIGHT EDGE lies inside the alert
                      interval -- that is, the model must fire from what it has
                      already seen, after alert onset but before impact. This is the
                      only labelling under which "lead time" is a meaningful quantity.

    Returns (windows (W, window_len, 6), labels (W,)).
    """
    sig = trial.signal
    n = sig.shape[0]
    if n < window_len:
        return (np.empty((0, window_len, C.N_CHANNELS), np.float32),
                np.empty((0,), np.int64))

    starts = np.arange(0, n - window_len + 1, stride)
    idx = starts[:, None] + np.arange(window_len)[None, :]
    windows = sig[idx]                       # (W, window_len, 6)
    right = starts + window_len - 1          # inclusive right edge of each window

    labels = np.full(len(starts), C.BKG, dtype=np.int64)

    if trial.per_sample is not None:
        # Annotated source (SisFall Enhanced). The window takes the most severe class
        # present at or before its right edge within the window -- FALL dominates ALERT
        # dominates BKG, because a window containing an impact is not a background window.
        seg = trial.per_sample[idx]
        labels = seg.max(axis=1).astype(np.int64)
        if task == "preimpact":
            # under the pre-impact rule the decision is made at the right edge
            labels = trial.per_sample[right].astype(np.int64)
        return windows.astype(np.float32), labels

    if not trial.is_fall:
        return windows.astype(np.float32), labels

    if task == "postfall":
        if trial.impact_idx is not None:
            inside = (starts <= trial.impact_idx) & (trial.impact_idx <= right)
            labels[inside] = C.FALL
    elif task == "preimpact":
        if trial.alert_span is not None:
            a, b = trial.alert_span
            labels[(right >= a) & (right < b)] = C.ALERT
        if trial.impact_idx is not None:
            # FALL is the impact EVENT, not everything that follows it. Marking every
            # window to the end of the trial as FALL inflates the class enormously --
            # a 20 s FallAllD trial with a 10 s impact would be half "fall" -- and
            # makes the task look far easier than detecting a fall actually is.
            # The window is the fall event itself, matching how the three-class
            # annotations this task imitates are defined.
            end = trial.impact_idx + int(round(FALL_EVENT_SEC * C.TARGET_HZ))
            labels[(right >= trial.impact_idx) & (right <= end)] = C.FALL
    else:
        raise ValueError(f"unknown task {task!r}")

    return windows.astype(np.float32), labels


def window_dataset(
    trials: list[Trial],
    task: str = "preimpact",
    window_len: int = C.WINDOW_LEN,
    stride: int = C.STRIDE_LEN,
) -> dict[str, np.ndarray]:
    """Window a whole dataset, carrying subject / dataset / trial provenance through."""
    X, y, subj, dsets, tids, edges = [], [], [], [], [], []
    for t in trials:
        w, lab = window_trial(t, task=task, window_len=window_len, stride=stride)
        if not len(w):
            continue
        X.append(w)
        y.append(lab)
        subj.append(np.full(len(w), t.subject, dtype=object))
        dsets.append(np.full(len(w), t.dataset, dtype=object))
        tids.append(np.full(len(w), t.trial_id, dtype=object))
        # right-edge sample index and the impact index, needed for lead time in E3
        starts = np.arange(0, t.signal.shape[0] - window_len + 1, stride)
        imp = -1 if t.impact_idx is None else int(t.impact_idx)
        edges.append(np.stack([starts + window_len - 1, np.full(len(starts), imp)], 1))

    if not X:
        # Return an empty corpus with the right keys rather than raising. One ablation
        # arm producing nothing should be reported and skipped, not kill a notebook
        # that still has a dozen useful arms to run.
        empty_f = np.empty((0, window_len, C.N_CHANNELS), np.float32)
        return {
            "X": empty_f,
            "y": np.empty((0,), np.int64),
            "subject": np.empty((0,), dtype=object),
            "dataset": np.empty((0,), dtype=object),
            "trial": np.empty((0,), dtype=object),
            "edge_impact": np.empty((0, 2), np.int64),
        }

    return {
        "X": np.concatenate(X, 0).astype(np.float32),
        "y": np.concatenate(y, 0).astype(np.int64),
        "subject": np.concatenate(subj, 0),
        "dataset": np.concatenate(dsets, 0),
        "trial": np.concatenate(tids, 0),
        "edge_impact": np.concatenate(edges, 0).astype(np.int64),
    }


def class_counts(y: np.ndarray) -> dict[str, int]:
    c = np.bincount(y, minlength=C.N_CLASSES)
    return {name: int(c[i]) for i, name in enumerate(C.CLASS_NAMES)}


def to_binary(y: np.ndarray) -> np.ndarray:
    """Merge ALERT and FALL -- how the three-class head yields binary metrics."""
    return (y > C.BKG).astype(np.int64)
