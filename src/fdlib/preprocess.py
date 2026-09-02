"""The frozen preprocessing contract -- section 4.1 of the experimental plan.

This module is the single definition of how a raw sensor stream becomes a model
input. The training scripts import it; the firmware generator emits C that mirrors
it step for step. If preprocessing during training differs in any way from
preprocessing on the device, the model scores 99% in Python and behaves randomly on
the ESP32 -- so nothing here may be reimplemented anywhere else.

The seven steps, in order:
    1. channel selection   -> waist/low-back, (ax, ay, az, gx, gy, gz)
    2. unit conversion     -> g and deg/s
    3. anti-alias filter, then decimate to 50 Hz
    4. windowing           -> 100 x 6, stride 50
    5. labelling           -> post-fall or pre-impact rule
    6. normalisation       -> channel-wise, constants from TRAIN SUBJECTS ONLY
    7. caching             -> .npz stamped with preprocess_signature()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from . import config as C


# ------------------------------------------------------------------ step 2: units
# Each source stores something different. These are the documented conversions;
# nb00 verifies each one by checking that a resting trial sits at 1.0 g.

SISFALL_ADXL345_G = 2 * 16 / 2 ** 13      # +/-16 g, 13-bit  -> 1 g = 256 counts
SISFALL_ITG3200_DPS = 2 * 2000 / 2 ** 16  # +/-2000 dps, 16-bit
SISFALL_MMA8451Q_G = 2 * 8 / 2 ** 14      # +/-8 g, 14-bit (second accelerometer, DISCARDED)


def sisfall_to_units(raw: np.ndarray) -> np.ndarray:
    """SisFall 9-column raw ADC counts -> (N, 6) in g and deg/s.

    Columns 0:3 are the ADXL345, 3:6 the ITG3200, 6:9 the MMA8451Q. The plan
    discards the second accelerometer -- our hardware has one IMU, so training on
    two would not transfer.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[1] < 6:
        raise ValueError(f"SisFall trial has {raw.shape[1]} columns, expected 9")
    out = np.empty((raw.shape[0], 6), dtype=np.float32)
    out[:, 0:3] = raw[:, 0:3] * SISFALL_ADXL345_G
    out[:, 3:6] = raw[:, 3:6] * SISFALL_ITG3200_DPS
    return out


# ------------------------------------------------------ step 3: filter + decimate

def to_target_rate(sig: np.ndarray, src_hz: float, dst_hz: int = C.TARGET_HZ) -> np.ndarray:
    """Anti-alias filter then decimate. Never array-slice.

    Slicing without filtering aliases the impact transient straight into the
    passband, which quietly corrupts exactly the class we care about.

    `scipy.signal.decimate` filters internally but only accepts an integer factor.
    Where the ratio is not an integer (FallAllD's 238 Hz, UMAFall's phone stream)
    we decimate by the largest integer factor available and then resample the
    remainder linearly, which is safe because the anti-alias step already removed
    everything above the new Nyquist.
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[:, None]
    if abs(src_hz - dst_hz) < 1e-9:
        return sig.astype(np.float32)
    if src_hz < dst_hz:
        raise ValueError(f"cannot upsample {src_hz} Hz to {dst_hz} Hz -- would fabricate signal")

    factor = int(src_hz // dst_hz)
    out = sig
    if factor >= 2:
        # decimate in stages of <=10; large single-shot factors give unstable filters
        remaining = factor
        while remaining >= 2:
            step = min(remaining, 10)
            while remaining % step and step > 2:
                step -= 1
            out = decimate(out, step, ftype="iir", zero_phase=True, axis=0)
            remaining //= step
    eff_hz = src_hz / factor
    if abs(eff_hz - dst_hz) > 1e-6:
        n_out = int(round(out.shape[0] * dst_hz / eff_hz))
        src_t = np.arange(out.shape[0])
        dst_t = np.linspace(0, out.shape[0] - 1, n_out)
        out = np.stack([np.interp(dst_t, src_t, out[:, c]) for c in range(out.shape[1])], axis=1)
    return out.astype(np.float32)


def resample_index(idx: int, src_hz: float, dst_hz: int = C.TARGET_HZ) -> int:
    """Map a sample index (e.g. a labelled impact frame) onto the decimated stream."""
    return int(round(idx * dst_hz / src_hz))


# ------------------------------------------------------------- step 4: orientation

def apply_rotation(sig: np.ndarray, rot: np.ndarray | None) -> np.ndarray:
    """Apply a 3x3 axis-convention correction to accelerometer and gyroscope alike.

    Two labs mounting the same sensor 180 degrees apart produce data that cannot
    transfer, and the failure looks exactly like a genuine domain shift. The
    rotation applied to each dataset is decided in nb01 by the gravity-axis check
    and is printed and committed, never inferred silently at training time.
    """
    if rot is None:
        return sig
    out = sig.copy()
    out[:, 0:3] = sig[:, 0:3] @ rot.T
    out[:, 3:6] = sig[:, 3:6] @ rot.T
    return out


def gravity_axis(sig: np.ndarray) -> tuple[int, float, float]:
    """Return (dominant axis, its signed mean in g, resting |a| in g).

    Used by nb01's sanity gate: resting magnitude must be 1.0 +/- 0.05 g, and the
    gravity axis must agree across datasets after rotation.
    """
    mean = sig[:, 0:3].mean(0)
    axis = int(np.argmax(np.abs(mean)))
    return axis, float(mean[axis]), float(np.linalg.norm(sig[:, 0:3], axis=1).mean())


# The convention every dataset is rotated onto: gravity resting on axis 1 (Y),
# negative. Chosen because SisFall and KFall already use it, so the two largest
# corpora are left untouched and only the smaller ones are transformed.
CANONICAL_AXIS = 1
CANONICAL_SIGN = -1.0


def canonical_rotation(axis: int, sign: float,
                       target_axis: int = CANONICAL_AXIS,
                       target_sign: float = CANONICAL_SIGN) -> np.ndarray:
    """Signed permutation matrix mapping a dataset's gravity axis onto the canonical one.

    nb01 measured that FallAllD rests gravity on axis 0 positive while SisFall and
    KFall rest it on axis 1 negative, and UMAFall on axis 1 positive. Left alone, that
    difference is indistinguishable from a genuine domain shift -- a model would appear
    to fail at cross-dataset transfer when it was really just handed an upside-down
    sensor. This is the single correction that has to happen before any C1 number means
    anything.

    The result is restricted to rotations (determinant +1) rather than arbitrary
    reflections: a reflection would flip the handedness of the coordinate frame and
    silently invert the sense of every gyroscope channel.
    """
    src_sign = 1.0 if sign >= 0 else -1.0
    tgt_sign = 1.0 if target_sign >= 0 else -1.0

    perm = list(range(3))
    perm[target_axis], perm[axis] = perm[axis], perm[target_axis]

    R = np.zeros((3, 3), dtype=np.float64)
    for out_i, in_i in enumerate(perm):
        R[out_i, in_i] = 1.0
    R[target_axis] *= tgt_sign * src_sign

    if np.linalg.det(R) < 0:
        # restore right-handedness by flipping an axis that is not the gravity axis
        other = next(i for i in range(3) if i != target_axis)
        R[other] *= -1.0
    return R


def measure_rotation(signals: list[np.ndarray]) -> np.ndarray:
    """Derive a dataset's correction from its own resting trials."""
    votes = [gravity_axis(s) for s in signals]
    axis = int(np.bincount([v[0] for v in votes], minlength=3).argmax())
    sign = float(np.median([v[1] for v in votes]))
    return canonical_rotation(axis, sign)


# ---------------------------------------------------------- step 6: normalisation

@dataclass
class NormConstants:
    """The twelve frozen numbers. Written to JSON, shipped to the firmware."""

    mean: list[float]
    std: list[float]
    signature: str
    n_windows: int
    source: str

    @classmethod
    def fit(cls, windows: np.ndarray, source: str) -> "NormConstants":
        """Fit on TRAINING SUBJECTS ONLY. Never call this on validation or test data."""
        flat = windows.reshape(-1, windows.shape[-1])
        std = flat.std(0)
        std[std < 1e-8] = 1.0
        return cls(
            mean=flat.mean(0).astype(np.float64).tolist(),
            std=std.astype(np.float64).tolist(),
            signature=C.preprocess_signature(),
            n_windows=int(windows.shape[0]),
            source=source,
        )

    def apply(self, windows: np.ndarray) -> np.ndarray:
        return ((windows - np.asarray(self.mean, np.float32))
                / np.asarray(self.std, np.float32)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NormConstants":
        d = json.loads(Path(path).read_text())
        if d["signature"] != C.preprocess_signature():
            raise ValueError(
                f"normalisation constants were fitted under signature {d['signature']} "
                f"but the current contract is {C.preprocess_signature()} -- refusing to "
                "mix preprocessing versions"
            )
        return cls(**d)

    def to_c_header(self) -> str:
        """Emit the same twelve numbers as C, for the ESP32 firmware."""
        m = ", ".join(f"{v:.8f}f" for v in self.mean)
        s = ", ".join(f"{v:.8f}f" for v in self.std)
        return (
            "// Generated by fdlib.preprocess -- do not edit by hand.\n"
            f"// preprocess signature: {self.signature}\n"
            f"// fitted on {self.n_windows} training windows from: {self.source}\n"
            "#pragma once\n"
            f"static const float kNormMean[6] = {{ {m} }};\n"
            f"static const float kNormStd[6]  = {{ {s} }};\n"
        )


def instance_normalise(windows: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-window, per-channel normalisation -- the first rung of the E2 adaptation ladder.

    Costs almost nothing on the MCU (one pass for mean, one for variance) and is the
    cheapest thing that can recover cross-dataset loss, so the plan says start here.
    """
    m = windows.mean(axis=1, keepdims=True)
    s = windows.std(axis=1, keepdims=True) + eps
    return ((windows - m) / s).astype(np.float32)
