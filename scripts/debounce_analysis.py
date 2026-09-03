#!/usr/bin/env python3
"""How many false alarms per hour does the device actually raise?

E3 reports 293 false alarms per hour at threshold 0.5 and 102 at 0.9. Those numbers
are correct and they are the reason this analysis exists.

The arithmetic is unforgiving. The model makes one decision every 0.5 s, so 7,200
decisions per hour. Even 99 % window-level specificity leaves 72 false alarms per
hour. Reaching one per hour needs specificity of 0.99986. **Window-level specificity
is therefore close to meaningless as a deployment metric**, and any paper reporting
only specificity has not shown its device is wearable.

Two mechanisms the firmware already has close most of the gap, and neither costs
anything on the microcontroller:

  k-of-n agreement -- require k consecutive windows above threshold before firing.
                      A genuine fall spans several overlapping windows; isolated
                      false positives do not.
  cooldown         -- after firing, ignore further alarms for a fixed interval. One
                      fall should raise one alarm, not eight.

This script measures both against the saved E3 predictions, and reports the cost in
lead time -- because requiring k consecutive windows necessarily delays the alarm by
(k-1) strides, and that delay comes straight out of the lead time C2 depends on.

Run: python scripts/debounce_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdlib import config as C  # noqa: E402

PRED = ROOT / "results" / "e3_predictions.npz"
STRIDE_SEC = C.STRIDE_SEC
HZ = C.TARGET_HZ


def load():
    z = np.load(PRED, allow_pickle=True)
    return z["prob"], z["y"], z["edge"], z["impact"], z["trial"]


def alarm_stream(score: np.ndarray, thr: float, k: int) -> np.ndarray:
    """True where k consecutive windows have been above threshold."""
    above = score >= thr
    if k <= 1:
        return above
    run = np.zeros(len(above), dtype=np.int32)
    c = 0
    for i, a in enumerate(above):
        c = c + 1 if a else 0
        run[i] = c
    return run >= k


def evaluate(prob, y, edge, impact, trial, thr: float, k: int, cooldown_s: float):
    """False alarms per hour and lead time under k-of-n agreement plus a cooldown."""
    score = prob[:, 1:].sum(1)

    # ---- false alarms on ADL (background) windows, per trial, with cooldown ----
    fa = 0
    adl_windows = 0
    for tid in np.unique(trial):
        m = trial == tid
        # only background windows count toward false alarms
        bkg = m & (y == C.BKG)
        if not bkg.any():
            continue
        order = np.argsort(edge[bkg])
        s = score[bkg][order]
        e = edge[bkg][order]
        adl_windows += len(s)
        fired = alarm_stream(s, thr, k)
        last = -1e18
        for i in np.where(fired)[0]:
            t = e[i] / HZ
            if t - last >= cooldown_s:
                fa += 1
                last = t

    hours = adl_windows * STRIDE_SEC / 3600.0
    fa_per_hour = fa / hours if hours else float("nan")

    # ---- lead time on fall trials, same alarm logic ----
    leads, detected, total = [], 0, 0
    for tid in np.unique(trial):
        m = trial == tid
        imp = impact[m]
        if not len(imp) or imp[0] < 0:
            continue
        total += 1
        order = np.argsort(edge[m])
        s = score[m][order]
        e = edge[m][order]
        fired = alarm_stream(s, thr, k)
        hits = np.where(fired)[0]
        if not len(hits):
            continue
        detected += 1
        leads.append((int(imp[0]) - int(e[hits[0]])) * 1000.0 / HZ)

    leads = np.asarray(leads)
    pre = leads[leads > 0]
    return {
        "threshold": thr,
        "k_consecutive": k,
        "cooldown_s": cooldown_s,
        "false_alarms_per_hour": round(fa_per_hour, 3),
        "adl_hours": round(hours, 2),
        "detection_rate": round(detected / total, 4) if total else 0.0,
        "preimpact_rate": round(len(pre) / total, 4) if total else 0.0,
        "mean_lead_ms": round(float(pre.mean()), 1) if len(pre) else float("nan"),
        "median_lead_ms": round(float(np.median(pre)), 1) if len(pre) else float("nan"),
        "n_fall_trials": total,
    }


def main() -> int:
    if not PRED.exists():
        sys.exit(f"missing {PRED} -- run nb04_e3 first")
    prob, y, edge, impact, trial = load()
    print(f"loaded {len(prob):,} pooled LOSO windows over {len(np.unique(trial)):,} trials\n")

    rows = []
    for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
        for k in (1, 2, 3, 4):
            rows.append(evaluate(prob, y, edge, impact, trial, thr, k, cooldown_s=5.0))
    df = pd.DataFrame(rows)

    out = ROOT / "results" / "debounce_analysis.csv"
    df.to_csv(out, index=False)

    print("k-of-n agreement + 5 s cooldown, pooled over all LOSO folds")
    print("(k=1, cooldown 0 would reproduce Table III's raw window-level numbers)\n")
    print(df.to_string(index=False))

    # the operating points a wearable could actually ship at
    ok = df[df["false_alarms_per_hour"] <= 1.0].sort_values("mean_lead_ms", ascending=False)
    md = [
        "# Alarm debouncing: what the device actually raises\n",
        "Table III reports **293 false alarms per hour** at threshold 0.5 and 102 at 0.9.",
        "Those numbers are correct, and they are the point.\n",
        "The model decides once every 0.5 s, so 7,200 times per hour. Even 99 % window-level",
        "specificity leaves 72 false alarms per hour; one per hour needs specificity of",
        "0.99986. **Window-level specificity is close to meaningless as a deployment metric**,",
        "and a paper reporting only specificity has not shown its device is wearable.\n",
        "The firmware already carries two mechanisms that cost nothing on the MCU: requiring",
        "*k* consecutive windows above threshold before firing, and a cooldown after firing.",
        "Requiring k windows necessarily delays the alarm by (k-1) strides, and that delay",
        "comes straight out of the lead time — so the trade-off is measured here, not assumed.\n",
        df.to_markdown(index=False) + "\n",
    ]
    if len(ok):
        b = ok.iloc[0]
        md += [
            "\n## Operating points at or under 1 false alarm per hour\n",
            ok.to_markdown(index=False) + "\n",
            f"\nBest lead time among them: **{b['mean_lead_ms']:.0f} ms** at threshold "
            f"{b['threshold']}, k={int(b['k_consecutive'])}, detecting "
            f"{b['preimpact_rate']:.1%} of falls before impact at "
            f"{b['false_alarms_per_hour']:.2f} false alarms per hour.\n",
        ]
    else:
        md += [
            "\n## No configuration reaches 1 false alarm per hour\n",
            "None of the swept configurations gets there on this data. That is a finding, not",
            "a gap in the sweep: it says a single 25 k-parameter window classifier at a 0.5 s",
            "stride is not by itself a deployable alarm, and that a shipped device needs",
            "either a stronger model, a longer agreement window, or a second confirmation",
            "stage. Reported rather than tuned around.\n",
        ]
    (ROOT / "results" / "debounce_analysis.md").write_text("\n".join(md))
    print(f"\nwrote {out} and debounce_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
