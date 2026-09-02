"""SisFall loader -- raw ADC counts to Trial objects.

Format, confirmed by nb00 against the mounted mirror:

    <root>/**/{D|F}NN_S{A|E}NN_RNN.txt
    9 comma-separated integers per line, trailing ';', 200 Hz, no header
    cols 0:3  ADXL345   +/-16 g,  13 bit   -> [(2*16)/2^13] * AD
    cols 3:6  ITG3200   +/-2000 dps, 16 bit -> [(2*2000)/2^16] * RD
    cols 6:9  MMA8451Q  +/-8 g,   14 bit   -> DISCARDED (our hardware has one IMU)

Two limitations that must be stated in the paper rather than discovered by a reviewer:

1. The available mirror carries 25 of the 38 published subjects -- all 23 young, but
   only SE06 and SE15 of the 15 older participants. Since SisFall's distinctive value
   is that it includes genuinely older adults, this materially weakens what SisFall
   contributes here.

2. SisFall ships no temporal labels; the label lives in the filename. Fall trials
   therefore get their impact index from the peak of the acceleration magnitude, which
   is the standard proxy in this literature but is a proxy, not an annotation. This is
   why KFall -- which has video-grounded onset and impact frames -- is the primary
   source for the pre-impact contribution, and SisFall is used for the post-fall task.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import config as C
from ..preprocess import sisfall_to_units, to_target_rate
from ..windowing import Trial

TRIAL_RE = re.compile(r"^([DF]\d+)_(S[AE]\d+)_(R\d+)\.txt$")
NATIVE_HZ = C.NATIVE_HZ["sisfall"]


def read_raw(path: Path) -> np.ndarray:
    """Parse one trial file to (N, 9) float. Tolerates the trailing ';' and stray blanks."""
    txt = path.read_text(errors="replace").replace(";", " ").replace(",", " ")
    v = np.fromstring(txt, sep=" ")  # noqa: NPY003 -- fixed numeric format, this is the fast path
    n = v.size // 9
    if n == 0:
        raise ValueError(f"{path.name}: no parsable rows")
    return v[: n * 9].reshape(n, 9)


def impact_index(sig_50hz: np.ndarray) -> int:
    """Proxy impact frame: the sample of peak acceleration magnitude.

    A fall's body-ground impact is by far the largest acceleration transient in the
    trial, so the peak is a reasonable stand-in. It is a proxy and is labelled as one.
    """
    return int(np.argmax(np.linalg.norm(sig_50hz[:, 0:3], axis=1)))


def load(root: str | Path, target_hz: int = C.TARGET_HZ,
         subjects: list[str] | None = None, limit: int | None = None) -> list[Trial]:
    """Load every SisFall trial under `root` as decimated Trial objects."""
    root = Path(root)
    paths = sorted(p for p in root.rglob("*.txt") if TRIAL_RE.match(p.name))
    if subjects is not None:
        paths = [p for p in paths if TRIAL_RE.match(p.name).group(2) in subjects]
    if limit:
        paths = paths[:limit]

    trials: list[Trial] = []
    for p in paths:
        code, subj, rep = TRIAL_RE.match(p.name).groups()
        try:
            raw = read_raw(p)
        except ValueError:
            continue
        sig = to_target_rate(sisfall_to_units(raw), NATIVE_HZ, target_hz)
        if sig.shape[0] < C.WINDOW_LEN:
            continue
        is_fall = code.startswith("F")
        imp = impact_index(sig) if is_fall else None
        # Pre-impact alert interval: the second preceding the proxy impact. Used only
        # when SisFall is pressed into the pre-impact task; KFall's real annotations
        # are preferred wherever both are available.
        span = (max(0, imp - target_hz), imp) if imp is not None else None
        trials.append(
            Trial(
                signal=sig,
                subject=f"sisfall:{subj}",
                dataset="sisfall",
                trial_id=f"sisfall:{p.stem}",
                is_fall=is_fall,
                impact_idx=imp,
                alert_span=span,
            )
        )
    return trials


def subjects_present(root: str | Path) -> list[str]:
    return sorted({TRIAL_RE.match(p.name).group(2)
                   for p in Path(root).rglob("*.txt") if TRIAL_RE.match(p.name)})
