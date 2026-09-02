"""KFall loader -- the primary pre-impact label source.

KFall is the only source here whose temporal labels were designed for this purpose:
onset and impact frames derived from synchronised video, rather than inferred from the
sensor trace itself. That is why the experimental plan makes it primary for
contribution C2 and treats every other temporal label as corroborating.

Format, confirmed by nb00:

    <root>/**/sensor_data/SAxx/SxxTyyRzz.csv     ~100 Hz, header row
    <root>/**/label_data/SAxx_label.xlsx
        columns: 'Task Code (Task ID)', 'Description', 'Trial ID',
                 'Fall_onset_frame', 'Fall_impact_frame'

The task-code column is sparse -- it labels a *group* of rows and is blank on the
continuation rows, so it must be forward-filled before the join. Getting this wrong
silently attaches trial 3's impact frame to trial 1 and quietly ruins every lead-time
number, which is exactly the sort of error that survives to publication.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..preprocess import to_target_rate, resample_index
from ..windowing import Trial

NATIVE_HZ = C.NATIVE_HZ["kfall"]
FILE_RE = re.compile(r"^S(\d+)T(\d+)R(\d+)\.csv$", re.I)
TASK_ID_RE = re.compile(r"\((\d+)\)")

# Candidate column names, in preference order. KFall mirrors are inconsistent about
# capitalisation and about whether the accelerometer columns are Acc* or Acc[XYZ].
ACC_CANDIDATES = [("AccX", "AccY", "AccZ"), ("Acc_X", "Acc_Y", "Acc_Z"), ("ax", "ay", "az")]
GYR_CANDIDATES = [("GyrX", "GyrY", "GyrZ"), ("Gyr_X", "Gyr_Y", "Gyr_Z"), ("gx", "gy", "gz")]


def _pick(df: pd.DataFrame, candidates: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    lower = {c.lower().replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = [c.lower().replace(" ", "") for c in cand]
        if all(k in lower for k in key):
            return tuple(lower[k] for k in key)  # type: ignore[return-value]
    raise KeyError(f"none of {candidates} found in columns {list(df.columns)}")


def load_labels(label_dir: Path) -> dict[tuple[str, int, int], tuple[int, int]]:
    """Build {(subject, task_id, trial_id): (onset_frame, impact_frame)}.

    Only fall tasks carry frames; ADL rows have NaN and are skipped.
    """
    out: dict[tuple[str, int, int], tuple[int, int]] = {}
    for xl in sorted(label_dir.glob("*.xlsx")):
        subj = re.search(r"(S[AE]\d+)", xl.name)
        if not subj:
            continue
        subj = subj.group(1)
        df = pd.read_excel(xl)
        cols = {c.strip(): c for c in df.columns}
        task_col = next((cols[c] for c in cols if c.lower().startswith("task code")), None)
        trial_col = next((cols[c] for c in cols if c.lower().startswith("trial")), None)
        onset_col = next((cols[c] for c in cols if "onset" in c.lower()), None)
        impact_col = next((cols[c] for c in cols if "impact" in c.lower()), None)
        if not all([task_col, trial_col, onset_col, impact_col]):
            continue

        df[task_col] = df[task_col].ffill()  # the sparse group label -- see module docstring
        for _, row in df.iterrows():
            m = TASK_ID_RE.search(str(row[task_col]))
            if not m or pd.isna(row[impact_col]) or pd.isna(row[trial_col]):
                continue
            try:
                out[(subj, int(m.group(1)), int(row[trial_col]))] = (
                    int(row[onset_col]), int(row[impact_col])
                )
            except (TypeError, ValueError):
                continue
    return out


def load(root: str | Path, target_hz: int = C.TARGET_HZ,
         limit: int | None = None) -> list[Trial]:
    root = Path(root)
    sensor_dir = next((p for p in root.rglob("sensor_data") if p.is_dir()), None)
    label_dir = next((p for p in root.rglob("label_data") if p.is_dir()), None)
    if sensor_dir is None:
        raise FileNotFoundError(f"no sensor_data directory under {root}")
    labels = load_labels(label_dir) if label_dir else {}

    paths = sorted(p for p in sensor_dir.rglob("*.csv") if FILE_RE.match(p.name))
    if limit:
        paths = paths[:limit]

    trials: list[Trial] = []
    for p in paths:
        subj_n, task_n, rep_n = (int(g) for g in FILE_RE.match(p.name).groups())
        subj = f"SA{subj_n:02d}"
        try:
            df = pd.read_csv(p)
            acc = _pick(df, ACC_CANDIDATES)
            gyr = _pick(df, GYR_CANDIDATES)
        except (KeyError, pd.errors.ParserError):
            continue

        raw = np.concatenate(
            [df[list(acc)].to_numpy(np.float64), df[list(gyr)].to_numpy(np.float64)], axis=1
        )
        if not np.isfinite(raw).all():
            raw = np.nan_to_num(raw)
        sig = to_target_rate(raw, NATIVE_HZ, target_hz)
        if sig.shape[0] < C.WINDOW_LEN:
            continue

        key = (subj, task_n, rep_n)
        imp = span = None
        is_fall = key in labels
        if is_fall:
            onset_f, impact_f = labels[key]
            imp = resample_index(impact_f, NATIVE_HZ, target_hz)
            span = (resample_index(onset_f, NATIVE_HZ, target_hz), imp)
            if imp >= sig.shape[0]:
                imp, span, is_fall = None, None, False

        trials.append(
            Trial(
                signal=sig,
                subject=f"kfall:{subj}",
                dataset="kfall",
                trial_id=f"kfall:{p.stem}",
                is_fall=is_fall,
                impact_idx=imp,
                alert_span=span,
            )
        )
    return trials
