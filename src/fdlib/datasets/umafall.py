"""UMAFall loader -- the final held-out stress test.

UMAFall is never used for training, tuning, or model selection in any fold. It is the
one genuinely 'never seen, never tuned' row in the results table, and it stops being
that the moment it influences a decision.

Format, confirmed by nb00:

    <root>/**/UMAFall_Subject_NN_{ADL|FALL}_<Activity>_<rep>_<timestamp>.csv
    A '%'-prefixed header block, then semicolon-separated data.
    The header carries the sensor map, e.g.
        %f8:95:c7:f3:ba:82; 0; RIGHTPOCKET; lge-LG-H815-5.1
        %C4:BE:84:71:A5:02; 2; WAIST;       SensorTag

A correction to the experimental plan. The plan states that UMAFall's waist channel is
the 200 Hz smartphone stream. It is not: the header shows the smartphone at
RIGHTPOCKET, and the WAIST position is a SensorTag, which samples far slower. Since
the project's own rule -- the one used to exclude UP-Fall -- forbids upsampling to
reach 50 Hz, the true waist channel cannot be used if its rate is below 50 Hz.

`load()` therefore reports what it finds and takes `position`/`sensor_id` explicitly,
so the choice between "use the slow waist sensor", "use the fast pocket sensor at a
different body position", and "exclude UMAFall as we excluded UP-Fall" is made in
nb01 on measured rates, and recorded, rather than assumed here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..preprocess import to_target_rate
from ..windowing import Trial

FILE_RE = re.compile(
    r"^UMAFall_Subject_(\d+)_(ADL|FALL)_([A-Za-z]+)_(\d+)_", re.I
)
# Sensor type codes used in the UMAFall data section.
ACC_TYPE, GYR_TYPE = 0, 1


def sensor_map(path: Path) -> dict[int, str]:
    """Parse the header's {sensor id -> body position} table."""
    out: dict[int, str] = {}
    for ln in path.read_text(errors="replace").splitlines():
        if not ln.startswith("%"):
            continue
        parts = [p.strip() for p in ln.lstrip("%").split(";")]
        if len(parts) >= 3 and parts[1].isdigit():
            out[int(parts[1])] = parts[2].upper()
    return out


def read_body(path: Path) -> pd.DataFrame:
    """Read the data section, skipping the '%' header block."""
    lines = path.read_text(errors="replace").splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip() and not ln.strip().startswith("%")), None)
    if start is None:
        raise ValueError(f"{path.name}: no data section")
    hdr = None
    for i in range(start - 1, max(start - 6, -1), -1):
        if "TimeStamp" in lines[i] or "Sample" in lines[i]:
            hdr = [c.strip().strip("%").strip() for c in lines[i].lstrip("%").split(";")]
            break
    df = pd.read_csv(path, skiprows=start, sep=";", header=None, engine="python",
                     comment="%", on_bad_lines="skip")
    if hdr and len(hdr) == df.shape[1]:
        df.columns = hdr
    else:
        df.columns = ["TimeStamp", "SampleNo", "X", "Y", "Z", "SensorType", "SensorID"][: df.shape[1]]
    return df


def measured_rate(df: pd.DataFrame, sensor_id: int, sensor_type: int) -> float:
    """Effective sampling rate of one (sensor, modality) stream, in Hz."""
    sub = df[(df["SensorID"] == sensor_id) & (df["SensorType"] == sensor_type)]
    t = pd.to_numeric(sub["TimeStamp"], errors="coerce").dropna().to_numpy()
    if len(t) < 10:
        return float("nan")
    dt = np.median(np.diff(t))
    if dt <= 0:
        return float("nan")
    # UMAFall timestamps are milliseconds
    return float(1000.0 / dt)


def survey(root: str | Path, n: int = 20) -> pd.DataFrame:
    """Report, per sensor id and position, the measured rate across a sample of trials.

    This is what nb01 uses to decide whether UMAFall's waist channel is usable at all.
    """
    root = Path(root)
    paths = sorted(p for p in root.rglob("*.csv") if FILE_RE.match(p.name))[:n]
    rows = []
    for p in paths:
        smap = sensor_map(p)
        try:
            df = read_body(p)
        except (ValueError, pd.errors.ParserError):
            continue
        for sid in sorted(set(df["SensorID"].dropna().astype(int))):
            rows.append({
                "file": p.name,
                "sensor_id": sid,
                "position": smap.get(sid, "?"),
                "acc_hz": measured_rate(df, sid, ACC_TYPE),
                "gyr_hz": measured_rate(df, sid, GYR_TYPE),
                "n_acc": int(((df["SensorID"] == sid) & (df["SensorType"] == ACC_TYPE)).sum()),
            })
    return pd.DataFrame(rows)


def load(root: str | Path, target_hz: int = C.TARGET_HZ,
         position: str = "WAIST", sensor_id: int | None = None,
         native_hz: float | None = None, limit: int | None = None) -> list[Trial]:
    root = Path(root)
    paths = sorted(p for p in root.rglob("*.csv") if FILE_RE.match(p.name))
    if limit:
        paths = paths[:limit]

    trials: list[Trial] = []
    for p in paths:
        m = FILE_RE.match(p.name)
        subj, kind, activity, rep = m.group(1), m.group(2).upper(), m.group(3), m.group(4)
        smap = sensor_map(p)
        sid = sensor_id
        if sid is None:
            sid = next((k for k, v in smap.items() if v == position.upper()), None)
        if sid is None:
            continue
        try:
            df = read_body(p)
        except (ValueError, pd.errors.ParserError):
            continue

        acc = df[(df["SensorID"] == sid) & (df["SensorType"] == ACC_TYPE)]
        gyr = df[(df["SensorID"] == sid) & (df["SensorType"] == GYR_TYPE)]
        if len(acc) < 32:
            continue
        a = acc[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        if len(gyr) >= len(acc) // 2 and len(gyr) > 0:
            g = gyr[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
            k = min(len(a), len(g))
            raw = np.concatenate([a[:k], g[:k]], axis=1)
        else:
            # some UMAFall sensors log no gyroscope; zero-fill so the channel layout
            # stays fixed, and let nb01 report how many trials this affects
            raw = np.concatenate([a, np.zeros_like(a)], axis=1)
        raw = np.nan_to_num(raw)

        src_hz = native_hz if native_hz else measured_rate(df, sid, ACC_TYPE)
        if not np.isfinite(src_hz) or src_hz < target_hz:
            continue  # never upsample -- the rule that excluded UP-Fall
        sig = to_target_rate(raw, src_hz, target_hz)
        if sig.shape[0] < C.WINDOW_LEN:
            continue

        is_fall = kind == "FALL"
        imp = int(np.argmax(np.linalg.norm(sig[:, 0:3], axis=1))) if is_fall else None
        span = (max(0, imp - target_hz), imp) if imp is not None else None

        trials.append(
            Trial(
                signal=sig,
                subject=f"umafall:S{int(subj):02d}",
                dataset="umafall",
                trial_id=f"umafall:S{int(subj):02d}_{kind}_{activity}_{rep}",
                is_fall=is_fall,
                impact_idx=imp,
                alert_span=span,
            )
        )
    return trials
