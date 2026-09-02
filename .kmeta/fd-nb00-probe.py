"""nb00 -- data reality probe, round 3 (final).

Rounds 1 and 2 settled the mount layout and every schema. Three questions remain, and
all three change what gets built, so they are answered before nb01 is written.

Q1. Is the SisFall Enhanced annotation recoverable?
    Round 2 showed recovery rising monotonically as the raw signal is low-passed
    harder -- 0% unfiltered, 2.9% at 20 Hz, 6.3% at 10 Hz, 26.9% at 5 Hz, all at
    decimation 1x. So the Enhanced tensors are a heavily smoothed version of the
    original recordings at their native 200 Hz.
    The obvious next step is to filter harder still, but that is exactly where the
    experiment starts lying to itself: as bandwidth falls, every window converges on
    every other window and matches appear by chance. Round 3 therefore sweeps down to
    1 Hz WITH A NULL CONTROL -- the same search run against time-reversed queries,
    which cannot have true matches. Recovery is only believable to the extent it
    exceeds that null.

Q2. Does the UMAFall mirror contain fall trials at all?
    Round 2 reported 0 of 746, but tested for '_FALL_' case-sensitively while UMAFall
    names its files '_Fall_'. Re-counted case-insensitively here.

Q3. Which FallAllD ActivityIDs are falls?
    activity_info is a plain dict {id: description}, not the DataFrame the loader
    assumed, so fall IDs must be read out of the descriptions.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

IN = Path("/kaggle/input/datasets")
OUT = Path("/kaggle/working")
REPORT: list[str] = []


def say(*parts: object) -> None:
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    REPORT.append(line)


def head(title: str) -> None:
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


SISFALL = IN / "adityavvvn" / "sisfall"
ENHANCED = IN / "nvnikhil0001" / "sisfall-enhanced"
FALLALLD = IN / "sankalpsinghvishen" / "derived-fallalld-dataset"
UMAFALL = IN / "thanushanth" / "umafall"

TRIAL_RE = re.compile(r"^([DF]\d+)_(S[AE]\d+)_(R\d+)\.txt$")
findings: dict = {}


# ==================================================================== Q2 UMAFall
head("Q2. UMAFall -- fall trials, sensor positions, and measured rates")
um = sorted(UMAFALL.rglob("*.csv"))
say(f"  csv files: {len(um)}")
kinds = {}
for p in um:
    m = re.search(r"UMAFall_Subject_(\d+)_([A-Za-z]+)_", p.name)
    if m:
        kinds[m.group(2).upper()] = kinds.get(m.group(2).upper(), 0) + 1
say(f"  trial kinds (case-insensitive): {kinds}")
fall_files = [p for p in um if re.search(r"_fall_", p.name, re.I)]
say(f"  FALL trials: {len(fall_files)}   examples: {[p.name for p in fall_files[:3]]}")
if fall_files:
    acts = sorted({re.search(r"_[Ff]all_([A-Za-z]+)_", p.name).group(1)
                   for p in fall_files if re.search(r"_[Ff]all_([A-Za-z]+)_", p.name)})
    say(f"  fall activity names: {acts}")

# measured per-sensor rate on a few trials
say("\n  measured sampling rate per sensor position:")
rows = []
for p in um[:6] + fall_files[:3]:
    lines = p.read_text(errors="replace").splitlines()
    smap = {}
    for ln in lines:
        if ln.startswith("%"):
            parts = [q.strip() for q in ln.lstrip("%").split(";")]
            if len(parts) >= 3 and parts[1].isdigit():
                smap[int(parts[1])] = parts[2].upper()
    start = next(i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("%"))
    df = pd.read_csv(p, skiprows=start, sep=";", header=None, engine="python",
                     comment="%", on_bad_lines="skip")
    df = df.iloc[:, :7]
    df.columns = ["ts", "n", "x", "y", "z", "stype", "sid"]
    for sid, pos in sorted(smap.items()):
        sub = df[(df["sid"] == sid) & (df["stype"] == 0)]
        if len(sub) < 10:
            continue
        t = pd.to_numeric(sub["ts"], errors="coerce").dropna().to_numpy()
        span_ms = t.max() - t.min()
        hz = (len(sub) - 1) / (span_ms / 1000.0) if span_ms > 0 else float("nan")
        rows.append({"file": p.name[:42], "sid": sid, "pos": pos, "n_acc": len(sub),
                     "span_s": round(span_ms / 1000.0, 2), "hz": round(hz, 1)})
say(pd.DataFrame(rows).to_string(index=False) if rows else "  (none)")
findings["umafall"] = {"n_csv": len(um), "kinds": kinds, "n_fall": len(fall_files),
                       "rates": rows}


# =================================================================== Q3 FallAllD
head("Q3. FallAllD -- which ActivityIDs are falls")
info = pd.read_pickle(FALLALLD / "activity_info.pkl")
say(f"  activity_info type: {type(info).__name__}, {len(info)} entries")
if isinstance(info, dict):
    for k in sorted(info):
        say(f"   {k:4d}  {info[k]}")
    fall_ids = sorted(k for k, v in info.items()
                      if re.search(r"\bfall", str(v), re.I) and not re.search(r"fail", str(v), re.I))
    say(f"\n  FALL activity ids ({len(fall_ids)}): {fall_ids}")
    findings["fallalld_fall_ids"] = fall_ids
    fa = pd.read_pickle(FALLALLD / "FallAllD.pkl")
    waist = fa[fa["Device"] == "Waist"]
    n_fall = int(waist["ActivityID"].isin(fall_ids).sum())
    say(f"  waist trials: {len(waist)}   of which falls: {n_fall}  "
        f"ADL: {len(waist) - n_fall}")
    findings["fallalld_waist"] = {"trials": len(waist), "falls": n_fall}


# ================================================== Q1 alignment, with null control
head("Q1. SisFall Enhanced re-alignment -- final sweep with a null control")

# the Enhanced tensors live under a 'Three Classes/' subdirectory, so locate by name
enh_files = {p.name: p for p in ENHANCED.rglob("*") if p.is_file()}
yv = np.fromfile(enh_files["y_val_3"], dtype=np.uint8).reshape(-1, 3)
xv = np.fromfile(enh_files["x_val_3"], dtype=np.float32).reshape(len(yv), 256, 6)
qmag_all = np.linalg.norm(xv[:, :, 0:3], axis=2)

sf_trials = [p for p in SISFALL.rglob("*.txt") if TRIAL_RE.match(p.name)]
PROBE = ["SA01", "SA02", "SA03"]
probe_trials = [p for p in sf_trials if TRIAL_RE.match(p.name).group(2) in PROBE]


def load_mag(p: Path) -> np.ndarray:
    txt = p.read_text(errors="replace").replace(";", " ").replace(",", " ")
    v = np.fromstring(txt, sep=" ")  # noqa: NPY003
    v = v[: (v.size // 9) * 9].reshape(-1, 9)
    return np.linalg.norm(v[:, 0:3], axis=1)


raw_cache = {p.name: load_mag(p) for p in probe_trials}
say(f"  probe subjects {PROBE}, trials {len(probe_trials)}")

DESC, ACCEPT, NQ = 32, 0.995, 4000
qsub = qmag_all[:NQ]
qrev = qsub[:, ::-1].copy()   # null control: time-reversed, cannot have a true match


def descriptors(w: np.ndarray) -> np.ndarray:
    d = w.reshape(len(w), DESC, 256 // DESC).mean(2)
    d = d - d.mean(1, keepdims=True)
    return (d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def run(cutoff: float | None, queries: np.ndarray) -> int:
    bank, meta = [], []
    for name, mag in raw_cache.items():
        m = mag
        if cutoff is not None:
            b, a = butter(4, cutoff / 100.0, btype="low")  # Nyquist = 100 Hz at 200 Hz
            m = filtfilt(b, a, m)
        if len(m) < 256:
            continue
        starts = np.arange(0, len(m) - 256 + 1, 2)
        w = m[starts[:, None] + np.arange(256)[None, :]]
        bank.append(descriptors(w))
        meta.extend((name, int(s)) for s in starts)
    B = np.concatenate(bank, 0)
    qd = descriptors(queries)

    best_i = np.empty(len(qd), np.int64)
    best_s = np.empty(len(qd), np.float32)
    for a0 in range(0, len(qd), 512):
        sim = qd[a0:a0 + 512] @ B.T
        best_i[a0:a0 + 512] = sim.argmax(1)
        best_s[a0:a0 + 512] = sim.max(1)

    conf = 0
    for qi in np.argsort(-best_s)[:1200]:
        if best_s[qi] < 0.98:
            break
        name, st = meta[best_i[qi]]
        m = raw_cache[name]
        if cutoff is not None:
            b, a = butter(4, cutoff / 100.0, btype="low")
            m = filtfilt(b, a, m)
        seg = m[st:st + 256]
        if len(seg) < 256:
            continue
        u = seg - seg.mean()
        v = queries[qi] - queries[qi].mean()
        den = np.linalg.norm(u) * np.linalg.norm(v)
        if den and float(u @ v / den) >= ACCEPT:
            conf += 1
    return conf


exp = NQ * len(PROBE) / 38.0
say(f"  queries {NQ}; expected true matches if alignment works: ~{exp:.0f}")
say(f"  {'cutoff':>8} {'real':>6} {'null':>6} {'real-null':>10} {'recovery':>9}")
grid = []
for cutoff in (5.0, 4.0, 3.0, 2.0, 1.0):
    t0 = time.time()
    real = run(cutoff, qsub)
    null = run(cutoff, qrev)
    net = max(real - null, 0)
    grid.append({"cutoff_hz": cutoff, "real": real, "null": null,
                 "net": net, "recovery": round(net / exp, 4)})
    say(f"  {cutoff:8.1f} {real:6d} {null:6d} {net:10d} {net / exp:8.1%}   ({time.time()-t0:.0f}s)")

best = max(grid, key=lambda g: g["net"])
tier = ("1-realign" if best["recovery"] >= 0.90 else
        "2-partial" if best["recovery"] >= 0.50 else "3-kfall-only")
say(f"\n  best: {best}")
say(f"  VERDICT -> TIER {tier}")
if tier == "3-kfall-only":
    say("  The Enhanced per-sample annotation is not recoverable at a usable rate.")
    say("  Applying the experimental plan's risk register:")
    say("   * original SisFall (25 subjects, filename labels, peak-acceleration impact")
    say("     proxy) stays in E1/E2/E5 on the POST-FALL task, so C1 keeps four datasets;")
    say("   * KFall alone carries the pre-impact contribution C2. The plan already")
    say("     names KFall the more authoritative source, so C2 loses corroboration,")
    say("     not its primary evidence.")

findings["alignment"] = {"grid": grid, "best": best, "tier": tier}

head("Artifacts")
(OUT / "probe_report.md").write_text(
    "# nb00 -- data reality probe (round 3, final)\n\n```\n" + "\n".join(REPORT) + "\n```\n")
(OUT / "probe_verdict.json").write_text(json.dumps(findings, indent=2, default=str))
say("wrote probe_report.md and probe_verdict.json")
