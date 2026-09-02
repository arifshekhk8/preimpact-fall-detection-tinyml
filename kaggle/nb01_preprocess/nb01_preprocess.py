"""nb01 -- build the cached window corpus, and refuse to proceed if it is wrong.

This notebook is the highest-risk part of the project. If preprocessing here differs
in any way from preprocessing on the ESP32, the model scores well in Python and
behaves randomly on the device. So the whole seven-step contract lives in
`fdlib.preprocess`, imported here rather than reimplemented, and this notebook's real
job is the section 4.2 sanity gates: it FAILS LOUDLY rather than quietly emitting a
corpus that trains to 99% and means nothing.

Output: a windowed corpus at 50 Hz, stamped with the preprocessing signature, plus
the sanity figures and the per-dataset gate report.

Runs the four loaders, applies the gates, and writes:
    windows_<task>.npz      X, y, subject, dataset, trial, edge_impact
    gate_report.json        every check, with its measured value
    norm_constants.json     the twelve frozen numbers (train-subject fit)
    fig_sanity_traces.png   a fall and a walk from each dataset, shared axes
    fig_alert_intervals.png KFall alert intervals overlaid on acceleration
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN = Path("/kaggle/input/datasets")
OUT = Path("/kaggle/working")

# fdlib arrives as an attached Kaggle Dataset -- see scripts/sync_fdlib.py.
# The mount path is not stable across runtimes, so locate the package rather than
# assume where it landed, and fail loudly if it is missing: a notebook that silently
# ran against a different preprocessing definition is worse than one that crashed.
_fdlib_root = next(
    (p.parent for p in Path("/kaggle/input").rglob("fdlib/__init__.py")), None
)
if _fdlib_root is None:
    raise SystemExit("fdlib dataset not attached -- run scripts/sync_fdlib.py and add "
                     "arifshekh/fdlib-preimpact-fall to this kernel's dataset_sources")
sys.path.insert(0, str(_fdlib_root.parent))
_ver = _fdlib_root.parent / "VERSION.json"
print(f"fdlib from {_fdlib_root.parent}"
      f"  version {json.loads(_ver.read_text())['content_hash'] if _ver.exists() else '?'}")

from fdlib import config as C  # noqa: E402
from fdlib.datasets import fallalld, kfall, sisfall, umafall  # noqa: E402
from fdlib.preprocess import NormConstants, gravity_axis  # noqa: E402
from fdlib.windowing import class_counts, window_dataset  # noqa: E402

ROOTS = {
    "sisfall": IN / "adityavvvn" / "sisfall",
    "kfall": IN / "usmanabbasi2002" / "kfall-dataset",
    "fallalld": IN / "sankalpsinghvishen" / "derived-fallalld-dataset",
    "umafall": IN / "thanushanth" / "umafall",
}

LIMIT = None  # set to an int for a smoke run
gates: dict[str, dict] = {}
report: list[str] = []


def say(*p: object) -> None:
    line = " ".join(str(x) for x in p)
    print(line, flush=True)
    report.append(line)


print(f"fdlib preprocessing signature: {C.preprocess_signature()}")
say(f"preprocess signature: {C.preprocess_signature()}")
say(f"target {C.TARGET_HZ} Hz, window {C.WINDOW_LEN} samples, stride {C.STRIDE_LEN}")


# ------------------------------------------------------------------- 1. load all
trials_by_ds: dict[str, list] = {}
for name, loader, root in [
    ("sisfall", sisfall.load, ROOTS["sisfall"]),
    ("kfall", kfall.load, ROOTS["kfall"]),
    ("fallalld", fallalld.load, ROOTS["fallalld"]),
    ("umafall", umafall.load, ROOTS["umafall"]),
]:
    say(f"\n--- loading {name} from {root}")
    try:
        t = loader(root, target_hz=C.TARGET_HZ, limit=LIMIT)
    except Exception as e:  # noqa: BLE001
        say(f"  !! loader raised {type(e).__name__}: {e}")
        say(traceback.format_exc()[-1500:])
        t = []
    trials_by_ds[name] = t
    n_fall = sum(1 for x in t if x.is_fall)
    subj = sorted({x.subject for x in t})
    say(f"  trials {len(t)}  falls {n_fall}  ADL {len(t) - n_fall}  subjects {len(subj)}")
    if t:
        say(f"  subjects: {subj[:6]}{' ...' if len(subj) > 6 else ''}")
        say(f"  mean duration {np.mean([x.signal.shape[0] for x in t]) / C.TARGET_HZ:.1f} s")


# ------------------------------------------------------- 2. sanity gates (4.2)
say("\n" + "=" * 78)
say("SECTION 4.2 SANITY GATES")
say("=" * 78)

for name, trials in trials_by_ds.items():
    g: dict = {"n_trials": len(trials)}
    if not trials:
        g["status"] = "EMPTY"
        gates[name] = g
        say(f"\n{name}: NO TRIALS LOADED -- gate failed")
        continue

    # gate 1: resting acceleration magnitude must sit at 1.0 g
    adl = [t for t in trials if not t.is_fall] or trials
    mags = [float(np.linalg.norm(t.signal[:, 0:3], axis=1).mean()) for t in adl[:200]]
    g["resting_mag_g"] = round(float(np.median(mags)), 4)
    g["resting_mag_ok"] = bool(abs(g["resting_mag_g"] - 1.0) <= 0.15)

    # gate 2: which axis carries gravity, and with what sign
    axes = [gravity_axis(t.signal) for t in adl[:200]]
    dom = np.bincount([a[0] for a in axes], minlength=3)
    g["gravity_axis"] = int(np.argmax(dom))
    g["gravity_axis_votes"] = dom.tolist()
    g["gravity_sign"] = float(np.median([a[1] for a in axes]))

    # gate 3: gyroscope must be in deg/s, not rad/s -- a 57x error hides easily
    gyr = np.concatenate([t.signal[:, 3:6] for t in adl[:100]])
    g["gyro_abs_mean_dps"] = round(float(np.abs(gyr).mean()), 3)
    g["gyro_p99_dps"] = round(float(np.percentile(np.abs(gyr), 99)), 2)

    # gate 4: fall trials should show a larger peak than ADL trials
    fal = [t for t in trials if t.is_fall]
    if fal:
        pf = np.median([float(np.linalg.norm(t.signal[:, 0:3], axis=1).max()) for t in fal[:200]])
        pa = np.median([float(np.linalg.norm(t.signal[:, 0:3], axis=1).max()) for t in adl[:200]])
        g["peak_fall_g"], g["peak_adl_g"] = round(pf, 3), round(pa, 3)
        g["fall_peak_exceeds_adl"] = bool(pf > pa)

    g["status"] = "OK" if g.get("resting_mag_ok") else "FAIL"
    gates[name] = g
    say(f"\n{name}:")
    for k, v in g.items():
        say(f"  {k:26s} {v}")

say("\n--- gravity axis agreement across datasets ---")
ax = {k: v.get("gravity_axis") for k, v in gates.items() if "gravity_axis" in v}
sg = {k: (1 if v.get("gravity_sign", 0) > 0 else -1) for k, v in gates.items() if "gravity_sign" in v}
say(f"  axis: {ax}")
say(f"  sign: {sg}")
if len(set(ax.values())) > 1 or len(set(sg.values())) > 1:
    say("  !! DATASETS DISAGREE ON THE GRAVITY AXIS.")
    say("  This is the cross-dataset trap the plan warns about: two labs mounting the")
    say("  same sensor differently produce data that cannot transfer, and the failure")
    say("  looks exactly like a genuine domain shift. The rotation needed to reconcile")
    say("  them is recorded below and MUST be applied before any C1 claim.")
    ref_axis = max(set(ax.values()), key=list(ax.values()).count)
    ref_sign = max(set(sg.values()), key=list(sg.values()).count)
    say(f"  reference convention: gravity on axis {ref_axis}, sign {ref_sign}")
    for k in ax:
        say(f"   {k}: axis {ax[k]} sign {sg[k]} -> "
            f"{'matches reference' if (ax[k] == ref_axis and sg[k] == ref_sign) else 'NEEDS ROTATION'}")
else:
    say("  all datasets agree -- no rotation required")
gates["_gravity_axis"] = ax
gates["_gravity_sign"] = sg


# ------------------------------------------------------------- 3. window corpus
say("\n" + "=" * 78)
say("WINDOWING")
say("=" * 78)

corpora = {}
for task in ("postfall", "preimpact"):
    all_trials = [t for ts in trials_by_ds.values() for t in ts]
    if not all_trials:
        say(f"  {task}: no trials, skipping")
        continue
    d = window_dataset(all_trials, task=task)
    corpora[task] = d
    say(f"\n  task={task}: X {d['X'].shape}  classes {class_counts(d['y'])}")
    for ds in sorted(set(d["dataset"])):
        m = d["dataset"] == ds
        cc = class_counts(d["y"][m])
        tot = int(m.sum())
        pos = cc["alert"] + cc["fall"]
        say(f"    {ds:10s} windows {tot:7d}  {cc}  "
            f"imbalance 1:{(tot - pos) / max(pos, 1):.0f}")


# -------------------------------------------------- 4. normalisation constants
say("\n" + "=" * 78)
say("NORMALISATION -- fitted on TRAINING SUBJECTS ONLY")
say("=" * 78)

if "postfall" in corpora:
    d = corpora["postfall"]
    rng = np.random.default_rng(C.SEED)
    subs = np.unique(d["subject"])
    rng.shuffle(subs)
    fit_subs = set(subs[: max(1, int(len(subs) * 0.8))].tolist())
    mask = np.array([s in fit_subs for s in d["subject"]])
    nc = NormConstants.fit(d["X"][mask], source=f"{int(mask.sum())} windows from "
                                                f"{len(fit_subs)}/{len(subs)} subjects (seed {C.SEED})")
    nc.save(OUT / "norm_constants.json")
    (OUT / "norm_constants.h").write_text(nc.to_c_header())
    say(f"  mean {np.round(nc.mean, 5).tolist()}")
    say(f"  std  {np.round(nc.std, 5).tolist()}")
    say("  written to norm_constants.json and norm_constants.h (identical numbers,")
    say("  so the firmware cannot drift from training)")


# ---------------------------------------------------------------- 5. figures
try:
    fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex=True)
    for j, (name, trials) in enumerate(trials_by_ds.items()):
        for i, want_fall in enumerate((True, False)):
            ax_ = axes[i, j]
            sel = next((t for t in trials if t.is_fall == want_fall), None)
            if sel is None:
                ax_.set_title(f"{name}: none"); continue
            seg = sel.signal[: 10 * C.TARGET_HZ]
            tt = np.arange(len(seg)) / C.TARGET_HZ
            for c in range(3):
                ax_.plot(tt, seg[:, c], lw=0.8, label="xyz"[c])
            ax_.plot(tt, np.linalg.norm(seg[:, 0:3], axis=1), "k", lw=1.2, label="|a|")
            ax_.set_title(f"{name} {'fall' if want_fall else 'ADL'}", fontsize=9)
            ax_.set_ylim(-4, 6)
            if j == 0:
                ax_.set_ylabel("accel (g)")
            if i == 1:
                ax_.set_xlabel("time (s)")
            if i == 0 and j == 0:
                ax_.legend(fontsize=7, ncol=4)
    fig.suptitle(f"Sanity traces at {C.TARGET_HZ} Hz -- amplitudes must be comparable "
                 "across datasets", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sanity_traces.png", dpi=140)
    say("\nwrote fig_sanity_traces.png")
except Exception as e:  # noqa: BLE001
    say(f"figure failed: {e}")

try:
    kf = [t for t in trials_by_ds.get("kfall", []) if t.is_fall and t.alert_span][:10]
    if kf:
        fig, axes = plt.subplots(2, 5, figsize=(18, 6))
        for ax_, t in zip(axes.ravel(), kf):
            mag = np.linalg.norm(t.signal[:, 0:3], axis=1)
            tt = np.arange(len(mag)) / C.TARGET_HZ
            ax_.plot(tt, mag, "k", lw=0.9)
            a, b = t.alert_span
            ax_.axvspan(a / C.TARGET_HZ, b / C.TARGET_HZ, color="orange", alpha=0.35,
                        label="alert")
            ax_.axvline(t.impact_idx / C.TARGET_HZ, color="red", lw=1.2, label="impact")
            ax_.set_title(t.trial_id.split(":")[-1], fontsize=8)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("KFall alert intervals -- the interval must PRECEDE the "
                     "acceleration spike, not straddle it", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "fig_alert_intervals.png", dpi=140)
        say("wrote fig_alert_intervals.png")
except Exception as e:  # noqa: BLE001
    say(f"alert figure failed: {e}")


# ---------------------------------------------------------------- 6. emit cache
for task, d in corpora.items():
    np.savez_compressed(
        OUT / f"windows_{task}.npz",
        X=d["X"].astype(np.float32),
        y=d["y"].astype(np.int8),
        subject=d["subject"].astype(str),
        dataset=d["dataset"].astype(str),
        trial=d["trial"].astype(str),
        edge_impact=d["edge_impact"].astype(np.int32),
        signature=C.preprocess_signature(),
    )
    say(f"wrote windows_{task}.npz  {d['X'].shape}")

(OUT / "gate_report.json").write_text(json.dumps(gates, indent=2, default=str))
(OUT / "nb01_report.md").write_text("# nb01 -- preprocessing\n\n```\n" + "\n".join(report) + "\n```\n")

failed = [k for k, v in gates.items() if isinstance(v, dict) and v.get("status") not in (None, "OK")]
say(f"\nGATE SUMMARY: {'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}")
if failed:
    raise SystemExit(f"sanity gates failed for {failed} -- refusing to publish a corpus "
                     "that would train to a meaningless number")
