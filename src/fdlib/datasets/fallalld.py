"""FallAllD loader -- the genuinely out-of-distribution LODO fold.

FallAllD matters because it is independent hardware from an independent lab. If the
cross-dataset gap is real rather than an artifact of shared recording conditions, it
shows up here.

Format, confirmed by nb00 against `FallAllD.pkl` (the official IEEE DataPort
distribution, a pandas pickle):

    columns: SubjectID, Device, ActivityID, TrialNo, Acc, Gyr, Mag, Bar
    Device in {Neck, Waist, Wrist}      -> Waist only; the plan discards the rest
    Acc, Gyr: int16 (4760, 3)           -> 4760 samples / 20 s = 238 Hz
    Mag (1600, 3), Bar (200, 2)         -> discarded; our hardware has neither
    15 subjects

Sensor scaling follows the FallAllD release: the accelerometer is +/-8 g over 16 bits
and the gyroscope +/-2000 dps over 16 bits. nb01's resting-magnitude gate is what
actually confirms this -- if the constant were wrong, resting |a| would not sit at 1 g.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..preprocess import to_target_rate
from ..windowing import Trial

NATIVE_HZ = C.NATIVE_HZ["fallalld"]
ACC_SCALE = 8.0 / 32768.0      # +/-8 g, 16-bit signed
GYR_SCALE = 2000.0 / 32768.0   # +/-2000 dps, 16-bit signed

# nb00 confirmed `activity_info.pkl` is a plain {id: description} dict, not a table.
# Fall activities are therefore identified from their descriptions.
#
# The 'fail' exclusion is not paranoia: FallAllD's activity 15 is "Fail to stand up",
# which a naive substring match on 'fal' would label a fall. It is an ADL.
_FALL_RE = re.compile(r"\bfall", re.I)
_NOT_FALL_RE = re.compile(r"fail", re.I)


def _fall_ids(root: Path) -> set[int] | None:
    info_path = root / "activity_info.pkl"
    if not info_path.exists():
        info_path = next(iter(sorted(root.rglob("activity_info.pkl"))), None)
    if info_path is None or not info_path.exists():
        return None
    try:
        info = pd.read_pickle(info_path)
    except Exception:  # noqa: BLE001
        return None

    if isinstance(info, dict):
        items = info.items()
    elif isinstance(info, pd.DataFrame):
        id_col = next((c for c in info.columns if "id" in c.lower()), None)
        desc_col = next((c for c in info.columns
                         if any(k in c.lower() for k in ("desc", "name", "activity", "type"))), None)
        if id_col is None or desc_col is None:
            return None
        items = zip(info[id_col].tolist(), info[desc_col].tolist())
    else:
        return None

    ids = {int(k) for k, v in items
           if _FALL_RE.search(str(v)) and not _NOT_FALL_RE.search(str(v))}
    return ids or None


def load(root: str | Path, target_hz: int = C.TARGET_HZ,
         device: str = "Waist", limit: int | None = None) -> list[Trial]:
    root = Path(root)
    pkl = root / "FallAllD.pkl"
    if not pkl.exists():
        pkl = next(iter(sorted(root.rglob("FallAllD.pkl"))), None)
    if pkl is None:
        raise FileNotFoundError(f"FallAllD.pkl not found under {root}")

    df = pd.read_pickle(pkl)
    df = df[df["Device"] == device]
    if limit:
        df = df.head(limit)

    fall_ids = _fall_ids(root)
    trials: list[Trial] = []
    for _, row in df.iterrows():
        acc = np.asarray(row["Acc"], dtype=np.float64) * ACC_SCALE
        gyr = np.asarray(row["Gyr"], dtype=np.float64) * GYR_SCALE
        n = min(len(acc), len(gyr))
        if n < 64:
            continue
        raw = np.concatenate([acc[:n], gyr[:n]], axis=1)
        sig = to_target_rate(raw, NATIVE_HZ, target_hz)
        if sig.shape[0] < C.WINDOW_LEN:
            continue

        aid = int(row["ActivityID"])
        if fall_ids is None:
            raise RuntimeError(
                "cannot determine which FallAllD activities are falls: activity_info.pkl "
                "is missing or unreadable. Refusing to guess -- a wrong fall/ADL split "
                "would silently corrupt every metric."
            )
        is_fall = aid in fall_ids
        # FallAllD ships no temporal labels, so the impact is the acceleration peak --
        # the same documented proxy used for SisFall. FallAllD is a generalisation test,
        # not a lead-time source, so this proxy never feeds a C2 number.
        imp = int(np.argmax(np.linalg.norm(sig[:, 0:3], axis=1))) if is_fall else None
        span = (max(0, imp - target_hz), imp) if imp is not None else None

        subj = f"fallalld:S{int(row['SubjectID']):02d}"
        trials.append(
            Trial(
                signal=sig,
                subject=subj,
                dataset="fallalld",
                trial_id=f"fallalld:S{int(row['SubjectID']):02d}A{aid}T{int(row['TrialNo'])}",
                is_fall=is_fall,
                impact_idx=imp,
                alert_span=span,
            )
        )
    return trials
