"""nb00 -- data reality probe, round 2.

Round 1 established the mount layout (/kaggle/input/datasets/<owner>/<slug>), read
every schema, and ran a first SisFall Enhanced re-alignment attempt. That attempt
recovered 0.8% of the expected windows -- but the 12 matches it did confirm were
unambiguously real (correlation 0.995-0.997 at plausible sample offsets, and a
consistent recovered scale of 1 g -> 0.2632 Enhanced units).

Correlations of 0.996 rather than ~1.0 are the tell: if the Enhanced tensors were a
verbatim slice of the original recordings the matches would be near-exact. They are
not, so the Enhanced signal has been filtered and/or resampled. Round 2 tests that
directly -- low-pass the raw signal across a grid of cutoffs and decimation factors,
and see whether recovery jumps. If it does, contribution C2 keeps its second label
source. If it does not, the plan's risk register applies unchanged: original SisFall
for training, KFall alone for C2.

Round 2 also nails down the four parser questions that must be answered before
fdlib.datasets can be written against reality rather than assumption.
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, filtfilt

IN = Path("/kaggle/input")
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


# ------------------------------------------------------------------- mount roots
# Round 1 established the layout definitively; resolve exactly, no guessing.
BASE = IN / "datasets"
SISFALL = BASE / "adityavvvn" / "sisfall"
ENHANCED = BASE / "nvnikhil0001" / "sisfall-enhanced"
KFALL = BASE / "usmanabbasi2002" / "kfall-dataset"
FALLALLD = BASE / "sankalpsinghvishen" / "derived-fallalld-dataset"
UMAFALL = BASE / "thanushanth" / "umafall"

head("0. Mount roots")
for n, p in [("sisfall", SISFALL), ("enhanced", ENHANCED), ("kfall", KFALL),
             ("fallalld", FALLALLD), ("umafall", UMAFALL)]:
    say(f"  {n:10s} {p}  exists={p.exists()}")


# ============================================================ 1. SisFall coverage
head("1. SisFall subject coverage -- is the mirror complete?")
TRIAL_RE = re.compile(r"^([DF]\d+)_(S[AE]\d+)_(R\d+)\.txt$")
sf_trials = [p for p in SISFALL.rglob("*.txt") if TRIAL_RE.match(p.name)]
sf_subj = sorted({TRIAL_RE.match(p.name).group(2) for p in sf_trials})
sa = [s for s in sf_subj if s.startswith("SA")]
se = [s for s in sf_subj if s.startswith("SE")]
say(f"  trial files: {len(sf_trials)}")
say(f"  subjects present: {len(sf_subj)}  (SA {len(sa)}/23, SE {len(se)}/15)")
say(f"  SE present: {se}")
say(f"  SE missing: {sorted(set(f'SE{i:02d}' for i in range(1, 16)) - set(se))}")
say("  NOTE: the published SisFall has 23 young + 15 older subjects. This mirror is")
say("  short of the older cohort, which is precisely the cohort that makes SisFall")
say("  worth having. Recorded as a stated limitation.")


# ============================================================ 2. KFall parser
head("2. KFall -- sensor columns and label join")
kf_sensor_dir = next((p for p in KFALL.rglob("sensor_data") if p.is_dir()), None)
kf_label_dir = next((p for p in KFALL.rglob("label_data") if p.is_dir()), None)
say(f"  sensor_data: {kf_sensor_dir}")
say(f"  label_data:  {kf_label_dir}")
kf_csvs = sorted(kf_sensor_dir.rglob("*.csv")) if kf_sensor_dir else []
say(f"  sensor csv files: {len(kf_csvs)}")
if kf_csvs:
    ex = kf_csvs[0]
    df = pd.read_csv(ex)
    say(f"\n  --- {ex.parent.name}/{ex.name}: shape {df.shape} ---")
    say(f"  columns: {list(df.columns)}")
    say(textwrap.indent(df.head(4).to_string(), "   | "))
    say(f"  dtypes: {df.dtypes.astype(str).to_dict()}")
    # rate check: KFall documents ~100 Hz
    for tcol in ("TimeStamp", "Time", "timestamp", "time"):
        if tcol in df.columns:
            dt = np.diff(pd.to_numeric(df[tcol], errors="coerce").dropna().values)
            say(f"  median dt on {tcol!r}: {np.median(dt):.6f} -> {1/np.median(dt):.2f} Hz")
            break
    subs = sorted({p.parent.name for p in kf_csvs})
    say(f"  sensor subjects ({len(subs)}): {subs}")
    say(f"  example filenames: {[p.name for p in kf_csvs[:5]]}")
if kf_label_dir:
    lf = sorted(kf_label_dir.glob("*.xlsx"))[0]
    ldf = pd.read_excel(lf)
    say(f"\n  --- label {lf.name}: shape {ldf.shape} ---")
    say(f"  columns: {list(ldf.columns)}")
    say(textwrap.indent(ldf.head(8).to_string(), "   | "))
    say("  NOTE: 'Task Code (Task ID)' is sparse -- it labels a group of rows and must")
    say("  be forward-filled before joining Trial ID to a sensor filename.")
    ff = ldf.copy()
    ff["Task Code (Task ID)"] = ff["Task Code (Task ID)"].ffill()
    say(f"  after ffill, task codes: {ff['Task Code (Task ID)'].dropna().unique()[:8].tolist()}")
    say(f"  onset/impact non-null rows: {ff['Fall_impact_frame'].notna().sum()} of {len(ff)}")


# ============================================================ 3. FallAllD parser
head("3. FallAllD -- waist channel and which ActivityIDs are falls")
fa_pkl = FALLALLD / "FallAllD.pkl"
info_pkl = FALLALLD / "activity_info.pkl"
if fa_pkl.exists():
    fa = pd.read_pickle(fa_pkl)
    waist = fa[fa["Device"] == "Waist"]
    say(f"  full: {fa.shape}   waist rows: {len(waist)}   subjects: {sorted(waist['SubjectID'].unique().tolist())}")
    say(f"  waist Acc shape: {waist['Acc'].iloc[0].shape}  Gyr: {waist['Gyr'].iloc[0].shape}")
    lens = waist["Acc"].apply(lambda a: a.shape[0]).value_counts()
    say(f"  Acc lengths: {lens.head(5).to_dict()}   (4760 samples / 20 s = 238 Hz)")
    a = waist["Acc"].iloc[0].astype(np.float64)
    # FallAllD ships a 16-bit +/-8 g accelerometer and +/-2000 dps gyro
    for scale, name in [(8 / 32768, "+/-8g"), (16 / 32768, "+/-16g")]:
        say(f"  resting |a| with {name} scaling: {np.linalg.norm(a * scale, axis=1).mean():.4f} g")
    g = waist["Gyr"].iloc[0].astype(np.float64)
    say(f"  Gyr raw abs-mean {np.abs(g).mean():.1f}; with 2000/32768 -> "
        f"{np.abs(g * 2000 / 32768).mean():.3f} dps")
if info_pkl.exists():
    info = pd.read_pickle(info_pkl)
    say(f"\n  --- activity_info: type {type(info).__name__} ---")
    if isinstance(info, pd.DataFrame):
        say(f"  shape {info.shape}  columns {list(info.columns)}")
        say(textwrap.indent(info.to_string(max_colwidth=60), "   | ")[:6000])
    else:
        say(f"  {str(info)[:4000]}")


# ============================================================ 4. UMAFall parser
head("4. UMAFall -- data section columns and the waist sensor id")
um_csvs = sorted(UMAFALL.rglob("UMAFall_Subject_*.csv"))
say(f"  csv files: {len(um_csvs)}")
if um_csvs:
    ex = um_csvs[0]
    lines = ex.read_text(errors="replace").splitlines()
    body = [i for i, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith("%")]
    say(f"  first non-comment line index: {body[0] if body else None}")
    say("  header block (sensor map):")
    for ln in lines[:45]:
        if "Sensor_ID" in ln or "WAIST" in ln or "POCKET" in ln or "ANKLE" in ln or "WRIST" in ln or "CHEST" in ln:
            say("   |", ln.strip()[:150])
    say("  first data rows:")
    for ln in lines[body[0]:body[0] + 6]:
        say("   |", ln.strip()[:150])
    df = pd.read_csv(ex, skiprows=body[0] - 1 if body[0] else 0, sep=";", engine="python")
    say(f"  parsed shape {df.shape}  columns {list(df.columns)[:10]}")
    say(textwrap.indent(df.head(4).to_string(), "   | "))
    falls = [p for p in um_csvs if "_FALL_" in p.name]
    say(f"  FALL trials: {len(falls)}  ADL trials: {len(um_csvs) - len(falls)}")
    say(f"  fall activity names: {sorted({re.search(r'_FALL_([A-Za-z]+)_', p.name).group(1) for p in falls if re.search(r'_FALL_([A-Za-z]+)_', p.name)})}")


# ============================================================ 5. re-alignment v2
head("5. SisFall Enhanced re-alignment, attempt 2 -- filtering hypothesis")

enh_files = {p.name: p for p in ENHANCED.rglob("*") if p.is_file()}
verdict: dict = {"attempted": False}

if "x_val_3" in enh_files:
    yv = np.fromfile(enh_files["y_val_3"], dtype=np.uint8).reshape(-1, 3)
    xv = np.fromfile(enh_files["x_val_3"], dtype=np.float32).reshape(len(yv), 256, 6)
    qmag_all = np.linalg.norm(xv[:, :, 0:3], axis=2)

    # Round 1's recovered scale, used as a sanity anchor only.
    say("  round-1 anchor: 1 g -> 0.2632 Enhanced units, 12 confirmed matches")

    # -- diagnostic: what sampling rate does the Enhanced signal look like? --
    def lag1(sig: np.ndarray) -> float:
        s = sig - sig.mean(-1, keepdims=True)
        num = (s[..., :-1] * s[..., 1:]).sum(-1)
        den = (s ** 2).sum(-1)
        return float(np.median(num / np.maximum(den, 1e-12)))

    say(f"\n  lag-1 autocorrelation of Enhanced |acc|: {lag1(qmag_all[:2000]):.5f}")
    probe_subjects = ["SA01", "SA02", "SA03"]
    probe_trials = [p for p in sf_trials
                    if TRIAL_RE.match(p.name).group(2) in probe_subjects]

    def load_raw(p: Path) -> np.ndarray:
        v = np.fromstring(  # noqa: NPY003 - fast path for this fixed format
            p.read_text(errors="replace").replace(";", " ").replace(",", " "), sep=" ")
        return v.reshape(-1, 9)

    ref = load_raw(probe_trials[0])
    refmag = np.linalg.norm(ref[:, 0:3], axis=1)
    for f in (1, 2, 4):
        m = refmag if f == 1 else decimate(refmag, f, ftype="iir", zero_phase=True)
        seg = np.stack([m[i:i + 256] for i in range(0, len(m) - 256, 256)])
        say(f"  lag-1 of raw SisFall |acc| decimated {f}x: {lag1(seg):.5f}")

    # -- grid over (decimation factor, low-pass cutoff) --
    DESC_LEN, ACCEPT = 32, 0.995
    NQ = 4000  # query subset keeps the grid affordable
    qsub = qmag_all[:NQ]
    qd = qsub.reshape(NQ, DESC_LEN, 256 // DESC_LEN).mean(2)
    qd -= qd.mean(1, keepdims=True)
    qd /= np.maximum(np.linalg.norm(qd, axis=1, keepdims=True), 1e-12)
    qd = qd.astype(np.float32)

    raw_cache = {p.name: np.linalg.norm(load_raw(p)[:, 0:3], axis=1) for p in probe_trials}
    say(f"\n  probe trials {len(probe_trials)}, query windows {NQ}")
    say(f"  expected matches if alignment works: "
        f"~{NQ * len(probe_subjects) / 38:.0f} (assuming Enhanced covers 38 subjects)")

    results = []
    for dec in (1, 2, 4):
        for cutoff_hz in (None, 20.0, 10.0, 5.0):
            t0 = time.time()
            src_hz = 200.0 / dec
            if cutoff_hz is not None and cutoff_hz >= src_hz / 2:
                continue
            bank, meta = [], []
            for name, mag in raw_cache.items():
                m = mag if dec == 1 else decimate(mag, dec, ftype="iir", zero_phase=True)
                if cutoff_hz is not None:
                    b, a = butter(4, cutoff_hz / (src_hz / 2), btype="low")
                    m = filtfilt(b, a, m)
                if len(m) < 256:
                    continue
                starts = np.arange(0, len(m) - 256 + 1, 2)
                idx = starts[:, None] + np.arange(256)[None, :]
                w = m[idx]
                d = w.reshape(len(starts), DESC_LEN, 256 // DESC_LEN).mean(2)
                d -= d.mean(1, keepdims=True)
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                bank.append(d.astype(np.float32))
                meta.extend((name, int(s)) for s in starts)
            B = np.concatenate(bank, 0)

            best_i = np.empty(NQ, np.int64)
            best_s = np.empty(NQ, np.float32)
            for a0 in range(0, NQ, 512):
                sim = qd[a0:a0 + 512] @ B.T
                best_i[a0:a0 + 512] = sim.argmax(1)
                best_s[a0:a0 + 512] = sim.max(1)

            # verify the strongest coarse hits at full 256-point resolution
            conf = 0
            for qi in np.argsort(-best_s)[:800]:
                if best_s[qi] < 0.98:
                    break
                name, st = meta[best_i[qi]]
                m = raw_cache[name]
                if dec != 1:
                    m = decimate(m, dec, ftype="iir", zero_phase=True)
                if cutoff_hz is not None:
                    b, a = butter(4, cutoff_hz / (src_hz / 2), btype="low")
                    m = filtfilt(b, a, m)
                seg = m[st:st + 256]
                if len(seg) < 256:
                    continue
                u = seg - seg.mean()
                v = qsub[qi] - qsub[qi].mean()
                den = np.linalg.norm(u) * np.linalg.norm(v)
                if den and float(u @ v / den) >= ACCEPT:
                    conf += 1
            exp = NQ * len(probe_subjects) / 38
            results.append((dec, cutoff_hz, len(B), conf, conf / exp))
            say(f"  dec={dec}x cutoff={str(cutoff_hz):>5}Hz  bank={len(B):7d}  "
                f"confirmed={conf:4d}  recovery={conf / exp:6.1%}  ({time.time() - t0:.0f}s)")

    best = max(results, key=lambda r: r[3])
    say(f"\n  best configuration: dec={best[0]}x cutoff={best[1]} -> recovery {best[4]:.1%}")
    tier = ("1-realign" if best[4] >= 0.90 else
            "2-partial" if best[4] >= 0.30 else "3-kfall-only")
    verdict = {
        "attempted": True,
        "grid": [{"decimate": r[0], "cutoff_hz": r[1], "bank": r[2],
                  "confirmed": r[3], "recovery": round(r[4], 4)} for r in results],
        "best": {"decimate": best[0], "cutoff_hz": best[1], "recovery": round(best[4], 4)},
        "tier": tier,
        "sisfall_subjects_present": sf_subj,
        "sisfall_se_missing": sorted(set(f"SE{i:02d}" for i in range(1, 16)) - set(se)),
    }
    say(f"  VERDICT -> TIER {tier}")
    if tier == "3-kfall-only":
        say("  Enhanced labels are not recoverable. Apply the plan's risk register:")
        say("   * original SisFall (25 subjects, filename labels) stays in E1/E2/E5")
        say("     on the POST-FALL task -- C1 keeps all four datasets;")
        say("   * KFall alone carries the pre-impact contribution C2.")

head("6. Artifacts")
(OUT / "probe_report.md").write_text(
    "# nb00 -- data reality probe (round 2)\n\n```\n" + "\n".join(REPORT) + "\n```\n"
)
(OUT / "probe_verdict.json").write_text(json.dumps(verdict, indent=2))
say("wrote probe_report.md and probe_verdict.json")
