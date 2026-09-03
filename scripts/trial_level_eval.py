#!/usr/bin/env python3
"""Re-score E3 on the KFall benchmark's protocol, so the comparison is apples to apples.

Yu et al. (2021) report KFall's benchmarks as counts over TRIALS, not windows:
FN 20/444 and FP 84/507 for the threshold method, 3/444 and 5/507 for ConvLSTM. Their
"specificity" is therefore *the fraction of ADL trials that raise no alarm at all*,
not the fraction of windows classified correctly.

Our Table III reports window-level specificity (0.959). Those two numbers measure
different things and comparing them directly would flatter or damn this work at random.
This script recomputes on their definition:

    sensitivity = fall trials with at least one alarm / all fall trials
    specificity = ADL trials with NO alarm / all ADL trials
    lead time   = mean over detected fall trials of (impact - first alarm)

Both framings are worth reporting in the paper. Trial-level is what makes the numbers
comparable with published work; window-level and false-alarms-per-hour are what say
whether the device is actually wearable. Reporting only the first is how this
literature ends up with 99 % specificity and unwearable devices.
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
HZ = C.TARGET_HZ


def evaluate(prob, y, edge, impact, trial, thr: float, k: int = 1):
    score = prob[:, 1:].sum(1)
    above = score >= thr

    fall_detected = fall_total = 0
    adl_clean = adl_total = 0
    leads = []

    for tid in np.unique(trial):
        m = trial == tid
        order = np.argsort(edge[m])
        s = above[m][order]
        e = edge[m][order]

        if k > 1:  # require k consecutive windows
            run, c = np.zeros(len(s), bool), 0
            for i, a in enumerate(s):
                c = c + 1 if a else 0
                run[i] = c >= k
            s = run

        fired = np.where(s)[0]
        imp = impact[m]
        is_fall = len(imp) and imp[0] >= 0

        if is_fall:
            fall_total += 1
            if len(fired):
                fall_detected += 1
                leads.append((int(imp[0]) - int(e[fired[0]])) * 1000.0 / HZ)
        else:
            adl_total += 1
            if not len(fired):
                adl_clean += 1

    leads = np.asarray(leads)
    pre = leads[leads > 0]
    return {
        "threshold": thr,
        "k": k,
        "sensitivity_trial": round(fall_detected / fall_total, 4) if fall_total else 0.0,
        "specificity_trial": round(adl_clean / adl_total, 4) if adl_total else 0.0,
        "FN": fall_total - fall_detected,
        "fall_trials": fall_total,
        "FP": adl_total - adl_clean,
        "adl_trials": adl_total,
        "mean_lead_ms": round(float(pre.mean()), 1) if len(pre) else float("nan"),
        "std_lead_ms": round(float(pre.std()), 1) if len(pre) else float("nan"),
        "preimpact_frac_of_detected": round(len(pre) / max(fall_detected, 1), 4),
    }


def main() -> int:
    if not PRED.exists():
        sys.exit(f"missing {PRED}")
    z = np.load(PRED, allow_pickle=True)
    prob, y, edge, impact, trial = z["prob"], z["y"], z["edge"], z["impact"], z["trial"]

    rows = []
    for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
        for k in (1, 2, 3):
            rows.append(evaluate(prob, y, edge, impact, trial, thr, k))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "trial_level_eval.csv", index=False)

    print("Trial-level scoring, on the protocol Yu et al. (2021) use for KFall\n")
    print(df.to_string(index=False))

    print("\nPublished KFall benchmarks (Yu et al. 2021, Table 3, same dataset):")
    print("  Threshold  sens 0.9550  spec 0.8343  lead 333 +/- 160 ms")
    print("  SVM        sens 0.9977  spec 0.9487  lead 385 +/- 159 ms")
    print("  ConvLSTM   sens 0.9932  spec 0.9901  lead 403 +/- 163 ms")
    print("\nNote: Yu et al. use a held-out test split; this is pooled 32-fold LOSO,")
    print("which is the stricter protocol.")

    (ROOT / "results" / "trial_level_eval.md").write_text(
        "# Trial-level evaluation (comparable with published KFall benchmarks)\n\n"
        "Yu et al. (2021) score KFall per **trial** — FN 20/444, FP 84/507 — so their\n"
        "specificity is the fraction of ADL trials raising no alarm, not a window-level\n"
        "rate. Table III's window-level numbers are not comparable with theirs. This is\n"
        "the same predictions re-scored on their definition.\n\n"
        + df.to_markdown(index=False) + "\n\n"
        "## Published comparison (same dataset)\n\n"
        "| Method | Sens | Spec | Lead time |\n|---|---:|---:|---|\n"
        "| Threshold (Yu 2021) | 0.9550 | 0.8343 | 333 ± 160 ms |\n"
        "| SVM (Yu 2021) | 0.9977 | 0.9487 | 385 ± 159 ms |\n"
        "| ConvLSTM (Yu 2021) | 0.9932 | 0.9901 | 403 ± 163 ms |\n"
        "| PreFallKD (2023) | — | — | 551.3 ms |\n"
        "| TinyFallNet (2023) | 0.8667 | 0.9797 | 477.7 ± 5.8 ms |\n\n"
        "Yu et al. evaluate on a held-out split; the rows above are pooled 32-fold\n"
        "leave-one-subject-out, which is the stricter protocol.\n"
    )
    print(f"\nwrote results/trial_level_eval.csv and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
