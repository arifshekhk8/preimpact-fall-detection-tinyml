"""nb05 -- E5, ablations.

Four levers, each of which is a real deployment decision rather than a curiosity:

  window length   1.0 / 1.5 / 2.0 s -- shorter means lower latency and less RAM
  sampling rate   25 / 50 / 100 Hz  -- lower rates cut compute; published work
                                       suggests performance survives it
  channels        accelerometer only vs accelerometer + gyroscope -- dropping the
                                       gyroscope saves meaningful power, and if
                                       accuracy holds that is a practical result
  placement       waist vs neck vs wrist, using FallAllD's three mounted positions

The placement arm uses FallAllD rather than UMAFall: nb01 established that UMAFall's
multi-position SensorTags all sample at 20 Hz, below the working rate, and cannot be
raised to it without fabricating signal. FallAllD carries Neck, Waist and Wrist at
238 Hz.

Window length and sampling rate change the input tensor, so those arms re-window from
the trial level rather than reusing the cached 50 Hz corpus.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

IN = Path("/kaggle/input/datasets")
OUT = Path("/kaggle/working")

_fdlib_root = next((p.parent for p in Path("/kaggle/input").rglob("fdlib/__init__.py")), None)
if _fdlib_root is None:
    raise SystemExit("fdlib dataset not attached")
sys.path.insert(0, str(_fdlib_root.parent))

from fdlib import config as C  # noqa: E402
from fdlib.cv import grouped_folds  # noqa: E402
from fdlib.datasets import fallalld, sisfall  # noqa: E402
from fdlib.experiment import append_row, completed_folds, evaluate, keras_fit_predict, load_corpus  # noqa: E402
from fdlib.models import count_params, proposed_cnn  # noqa: E402
from fdlib.preprocess import apply_rotation, canonical_rotation, gravity_axis  # noqa: E402
from fdlib.windowing import window_dataset  # noqa: E402

CSV = OUT / "results_e5.csv"
TASK = "postfall"
SMOKE = False
if SMOKE:
    C.MAX_EPOCHS, C.EARLY_STOP_PATIENCE = 2, 1

print(f"proposed params at default config: {count_params(proposed_cnn()):,}")
print(f"preprocess signature {C.preprocess_signature()}")

# --- preflight -------------------------------------------------------------
# E5 re-windows from the trial level for its window, rate and placement arms, so it
# needs the RAW datasets and not only the cached corpus. Roots are resolved by
# content because Kaggle's mount layout is not stable between kernels -- see
# fdlib.kaggle_paths. A misconfigured kernel must fail in seconds, not after the
# first arm has spent twenty-five minutes.
from fdlib import kaggle_paths as KP  # noqa: E402

SISFALL_ROOT = KP.require("SisFall (window/rate arms)", KP.sisfall_root())
FALLALLD_ROOT = KP.require("FallAllD (placement arm)", KP.fallalld_root())
print("preflight: resolved roots")
for _k, _v in KP.report().items():
    print(f"   {_k:10s} {_v}")


def canonicalise(trials):
    """Apply the same axis correction nb01 applies, so ablations are comparable."""
    if not trials:
        return trials
    adl = [t for t in trials if not t.is_fall] or trials
    votes = [gravity_axis(t.signal) for t in adl[:200]]
    axis = int(np.bincount([v[0] for v in votes], minlength=3).argmax())
    sign = float(np.median([v[1] for v in votes]))
    R = canonical_rotation(axis, sign)
    for t in trials:
        t.signal = apply_rotation(t.signal, R).astype(np.float32)
    return trials


def run_arm(arm: str, variant: str, X, y, subjects, n_channels: int, input_len: int):
    """Run one ablation arm. A failure here is reported and skipped, never fatal --
    losing eleven good arms because the twelfth misbehaved is not a trade worth making."""
    key = f"{arm}:{variant}"
    if key in completed_folds(CSV, "proposed"):
        print(f"  {key}: already done")
        return
    fn = keras_fit_predict(lambda: proposed_cnn(input_len=input_len, n_channels=n_channels))
    accum = []
    t0 = time.time()
    try:
        for fold_name, tr, te in grouped_folds(subjects, C.GROUPED_CV_SPLITS):
            from fdlib.cv import inner_val_split
            itr, iva = inner_val_split(subjects, tr)
            probs, info = fn(X[itr], y[itr], X[iva], y[iva], X[te])
            accum.append(evaluate(y[te], probs, info))
    except Exception as e:  # noqa: BLE001
        print(f"  {key}: FAILED {type(e).__name__}: {e}")
        return
    if not accum:
        print(f"  {key}: no folds completed -- skipping")
        return
    mean = {k: float(np.mean([a[k] for a in accum]))
            for k in accum[0] if isinstance(accum[0][k], (int, float))}
    append_row(CSV, {
        "model": "proposed", "fold": key, "arm": arm, "variant": variant,
        "n_channels": n_channels, "input_len": input_len,
        "params": count_params(proposed_cnn(input_len=input_len, n_channels=n_channels)),
        "folds": len(accum), "seconds": round(time.time() - t0, 1),
        "signature": C.preprocess_signature(),
        **{k: round(v, 5) for k, v in mean.items()},
    })
    print(f"  {key:34s} macro-F1 {mean['macro_f1_3class']:.4f}  "
          f"sens {mean['bin_sensitivity']:.3f}  spec {mean['bin_specificity']:.3f}  "
          f"({(time.time() - t0) / 60:.1f} min)")


# ------------------------------------------- arms 1 & 2: window length and rate
print("\n" + "=" * 74)
print("E5a/E5b -- window length and sampling rate (SisFall)")
print("=" * 74)

# Only one lever varies at a time, so enumerate the needed (rate, window) pairs
# explicitly instead of taking the full cross product and discarding most of it.
# The previous version windowed every combination before deciding to skip it, which
# was both wasteful and the reason a skipped combination could crash the notebook.
PAIRS: list[tuple[int, float, str]] = []
for w in C.ABLATION_WINDOW_SEC:
    PAIRS.append((C.TARGET_HZ, w, "window"))
for h in C.ABLATION_HZ:
    if h != C.TARGET_HZ:
        PAIRS.append((h, C.WINDOW_SEC, "rate"))

by_rate: dict[int, list[tuple[float, str]]] = {}
for h, w, a in PAIRS:
    by_rate.setdefault(h, []).append((w, a))
print(f"  arms: {[(h, w, a) for h, w, a in PAIRS]}")

for hz, specs in sorted(by_rate.items()):
    print(f"\n  loading SisFall at {hz} Hz ...", flush=True)
    trials = canonicalise(sisfall.load(SISFALL_ROOT, target_hz=hz))
    lens = [t.signal.shape[0] for t in trials]
    print(f"  trials {len(trials)}  signal length min {min(lens) if lens else 0} "
          f"max {max(lens) if lens else 0}", flush=True)
    if not trials:
        print(f"  !! no trials at {hz} Hz -- skipping this rate")
        continue
    for win_sec, arm in specs:
        wl = int(round(win_sec * hz))
        stride = max(1, int(round(C.STRIDE_SEC * hz)))
        d = window_dataset(trials, task=TASK, window_len=wl, stride=stride)
        print(f"  {hz}Hz {win_sec}s -> window_len {wl} stride {stride}: "
              f"{d['X'].shape[0]} windows", flush=True)
        if d["X"].shape[0] == 0:
            print(f"  !! no windows for {hz}Hz_{win_sec}s -- skipping")
            del d
            continue
        run_arm(arm, f"{hz}Hz_{win_sec}s", d["X"], d["y"], d["subject"], 6, wl)
        del d
    del trials

# ------------------------------------------------ arm 3: accelerometer only
print("\n" + "=" * 74)
print("E5c -- accelerometer only versus accelerometer + gyroscope")
print("=" * 74)

corpus = load_corpus(next(Path("/kaggle/input").rglob(f"windows_{TASK}.npz")))
sf = corpus["dataset"] == "sisfall"
Xs, ys, ss = corpus["X"][sf], corpus["y"][sf], corpus["subject"][sf]
run_arm("channels", "accel+gyro", Xs, ys, ss, 6, C.WINDOW_LEN)
run_arm("channels", "accel_only", Xs[:, :, 0:3], ys, ss, 3, C.WINDOW_LEN)

# --------------------------------------------------------- arm 4: placement
print("\n" + "=" * 74)
print("E5d -- sensor placement (FallAllD: waist / neck / wrist)")
print("=" * 74)

for pos in ("Waist", "Neck", "Wrist"):
    try:
        tr = canonicalise(fallalld.load(FALLALLD_ROOT, device=pos))
        if not tr:
            print(f"  {pos}: no trials"); continue
        d = window_dataset(tr, task=TASK)
        run_arm("placement", pos.lower(), d["X"], d["y"], d["subject"], 6, C.WINDOW_LEN)
        del d, tr
    except Exception as e:  # noqa: BLE001
        print(f"  {pos}: FAILED {type(e).__name__}: {e}")

# ------------------------------------------------------------------- summary
df = pd.read_csv(CSV)
cols = ["arm", "variant", "params", "input_len", "n_channels",
        "macro_f1_3class", "bin_sensitivity", "bin_specificity", "bin_f1"]
tbl = df[[c for c in cols if c in df.columns]].round(4)
print("\n" + "=" * 74)
print("E5 -- ablation summary")
print("=" * 74)
print(tbl.to_string(index=False))
tbl.to_csv(OUT / "table_ablations.csv", index=False)
(OUT / "table_ablations.md").write_text(
    "# E5 -- ablations\n\n5-fold subject-grouped CV, proposed model.\n\n"
    + tbl.to_markdown(index=False) + "\n\n"
    "Placement uses FallAllD rather than UMAFall: UMAFall's multi-position sensors all\n"
    "sample at 20 Hz, below the 50 Hz working rate, and raising them would fabricate\n"
    "signal -- the same criterion that excluded UP-Fall.\n"
)
print("\nwrote results_e5.csv, table_ablations.csv/md")
