"""nb04 -- E3, pre-impact detection (Table III). This is contribution C2.

The metric of record is LEAD TIME: milliseconds between the model's first alarm and
the labelled impact. Reported as a distribution, not only a mean, because the mean
alone hides the trials where the device fired too late to matter.

KFall is the primary and, after nb00, the only source. Its onset and impact frames
were derived from synchronised video rather than inferred from the sensor trace, which
is why the plan calls it the more authoritative source. The SisFall Enhanced
annotations that would have corroborated it are not recoverable from the available
mirror -- nb00 established a 22.5 % ceiling against a null control -- so this notebook
reports one label source and says so.

Two secondary metrics decide whether the device is wearable at all:
  * false alarms per hour of ADL data -- a detector firing forty times a day is
    useless however good its lead time looks;
  * the lead-time-versus-false-alarm curve, so the trade-off is visible rather than
    represented by a single cherry-picked operating point.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/kaggle/working")

_fdlib_root = next((p.parent for p in Path("/kaggle/input").rglob("fdlib/__init__.py")), None)
if _fdlib_root is None:
    raise SystemExit("fdlib dataset not attached")
sys.path.insert(0, str(_fdlib_root.parent))

from fdlib import config as C  # noqa: E402
from fdlib.cv import class_weights, inner_val_split, loso_folds  # noqa: E402
from fdlib.experiment import append_row, completed_folds, load_corpus  # noqa: E402
from fdlib.metrics import (  # noqa: E402
    classification_metrics, false_alarms_per_hour, lead_time_summary, lead_times,
    macro_f1_3class, sweep_operating_points,
)
from fdlib.models import MacroF1, compile_model, proposed_cnn, set_seeds  # noqa: E402

TASK = "preimpact"
LABEL_SOURCE = "kfall"
CORPUS = next(Path("/kaggle/input").rglob(f"windows_{TASK}.npz"))
CSV = OUT / "results_e3.csv"

SMOKE = False
if SMOKE:
    C.MAX_EPOCHS, C.EARLY_STOP_PATIENCE = 2, 1

corpus = load_corpus(CORPUS)
mask = corpus["dataset"] == LABEL_SOURCE
corpus = {k: v[mask] for k, v in corpus.items()}

if SMOKE:
    keep = np.isin(corpus["subject"], np.unique(corpus["subject"])[:4])
    corpus = {k: v[keep] for k, v in corpus.items()}

X, y = corpus["X"], corpus["y"]
subjects, trials = corpus["subject"], corpus["trial"]
right_edge = corpus["edge_impact"][:, 0]
impact_idx = corpus["edge_impact"][:, 1]

print(f"E3 on {LABEL_SOURCE} / {TASK}")
print(f"  windows {X.shape}  subjects {len(np.unique(subjects))}")
print(f"  classes {np.bincount(y, minlength=3).tolist()}  (bkg, alert, fall)")
print(f"  fall trials with an impact frame: "
      f"{len(np.unique(trials[impact_idx >= 0]))}")
print(f"  preprocess signature {C.preprocess_signature()}")

# --------------------------------------------------------- LOSO, collecting probs
all_prob, all_true, all_edge, all_imp, all_trial, all_subj = [], [], [], [], [], []

print("\n" + "=" * 74)
print(f"leave-one-subject-out across {len(np.unique(subjects))} KFall subjects")
print("=" * 74)

for fold_name, tr, te in loso_folds(subjects):
    if fold_name in completed_folds(CSV, "proposed"):
        print(f"  {fold_name}: already done")
        continue
    t0 = time.time()
    inner_tr, inner_va = inner_val_split(subjects, tr)
    set_seeds(C.SEED)
    model = compile_model(proposed_cnn(),
                          steps_per_epoch=max(1, len(inner_tr) // C.BATCH_SIZE),
                          epochs=C.MAX_EPOCHS)
    cb = MacroF1(X[inner_va], y[inner_va])
    model.fit(X[inner_tr], y[inner_tr], validation_data=(X[inner_va], y[inner_va]),
              epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE, verbose=0,
              class_weight=class_weights(y[inner_tr]), callbacks=[cb])
    probs = model.predict(X[te], batch_size=512, verbose=0)

    all_prob.append(probs); all_true.append(y[te])
    all_edge.append(right_edge[te]); all_imp.append(impact_idx[te])
    all_trial.append(trials[te]); all_subj.append(subjects[te])

    row = {"model": "proposed", "fold": fold_name, "task": TASK,
           "label_source": LABEL_SOURCE, "n_test": int(len(te)),
           "macro_f1": macro_f1_3class(y[te], probs),
           "seconds": round(time.time() - t0, 1),
           "signature": C.preprocess_signature()}
    for thr in C.DECISION_THRESHOLDS:
        lt = lead_times(probs, right_edge[te], impact_idx[te], trials[te], thr)
        s = lead_time_summary(lt)
        row[f"lead_mean_ms@{thr}"] = round(s["mean_ms"], 1) if s["mean_ms"] == s["mean_ms"] else None
        row[f"preimpact_rate@{thr}"] = round(s["preimpact_rate"], 4)
        row[f"fa_per_hour@{thr}"] = round(false_alarms_per_hour(y[te], probs, thr), 3)
    append_row(CSV, row)
    print(f"  {fold_name}: macro-F1 {row['macro_f1']:.4f}  "
          f"lead@0.5 {row.get('lead_mean_ms@0.5')} ms  "
          f"FA/h@0.5 {row.get('fa_per_hour@0.5')}  ({row['seconds']:.0f}s)")

# ------------------------------------------------------- pooled Table III + curve
if all_prob:
    P = np.concatenate(all_prob); Y = np.concatenate(all_true)
    E = np.concatenate(all_edge); I = np.concatenate(all_imp)
    T = np.concatenate(all_trial)

    print("\n" + "=" * 74)
    print("TABLE III -- pre-impact performance (pooled over all LOSO folds)")
    print("=" * 74)
    rows = []
    for thr in C.DECISION_THRESHOLDS:
        lt = lead_times(P, E, I, T, thr)
        s = lead_time_summary(lt)
        m = classification_metrics(Y, P, thr)
        rows.append({
            "label_source": "KFall (primary)",
            "threshold": thr,
            "mean_lead_ms": round(s["mean_ms"], 1),
            "std_ms": round(s["std_ms"], 1),
            "median_ms": round(s["median_ms"], 1),
            "p25_ms": round(s["p25_ms"], 1),
            "p75_ms": round(s["p75_ms"], 1),
            "detection_rate": round(s["detection_rate"], 4),
            "preimpact_rate": round(s["preimpact_rate"], 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "false_alarms_per_hour": round(false_alarms_per_hour(Y, P, thr), 3),
            "n_fall_trials": s["n_fall_trials"],
        })
    t3 = pd.DataFrame(rows)
    print(t3.to_string(index=False))
    t3.to_csv(OUT / "table_III.csv", index=False)
    (OUT / "table_III.md").write_text(
        "# Table III -- pre-impact performance\n\n"
        "Lead time is the interval between the model's FIRST alarm and the labelled\n"
        "impact, one value per fall trial, pooled across leave-one-subject-out folds.\n"
        "Positive means the alarm preceded impact.\n\n"
        + t3.to_markdown(index=False) + "\n\n"
        "**One label source.** KFall's onset and impact frames come from synchronised\n"
        "video. The SisFall Enhanced annotations that would have corroborated them are\n"
        "not recoverable from the available mirror (see nb00: 22.5 % ceiling against a\n"
        "null control), so this result rests on KFall alone.\n"
    )

    # the operating-point curve
    sweep = pd.DataFrame(sweep_operating_points(Y, P, E, I, T))
    sweep.to_csv(OUT / "e3_operating_points.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    ok = sweep.dropna(subset=["mean_ms"])
    a1.plot(ok["false_alarms_per_hour"], ok["mean_ms"], "o-", ms=3)
    for thr in C.DECISION_THRESHOLDS:
        r = sweep.iloc[(sweep["threshold"] - thr).abs().argmin()]
        a1.annotate(f"{r['threshold']:.2f}", (r["false_alarms_per_hour"], r["mean_ms"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    a1.set_xlabel("false alarms per hour of ADL data")
    a1.set_ylabel("mean lead time (ms)")
    a1.set_title("Lead time versus false alarm rate")
    a1.set_xscale("symlog", linthresh=0.1)
    a1.grid(alpha=0.3)

    lt05 = lead_times(P, E, I, T, 0.5)
    pre = lt05[~np.isnan(lt05)]
    pre = pre[pre > 0]
    a2.hist(pre, bins=40, edgecolor="k", alpha=0.8)
    a2.axvline(float(np.mean(pre)), color="red", ls="--",
               label=f"mean {np.mean(pre):.0f} ms")
    a2.axvline(float(np.median(pre)), color="orange", ls="--",
               label=f"median {np.median(pre):.0f} ms")
    a2.set_xlabel("lead time (ms)")
    a2.set_ylabel("fall trials")
    a2.set_title("Lead-time distribution at threshold 0.5")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lead_time.png", dpi=140)
    print("\nwrote table_III.csv/md, e3_operating_points.csv, fig_lead_time.png")

    np.savez_compressed(OUT / "e3_predictions.npz", prob=P, y=Y, edge=E, impact=I, trial=T)
