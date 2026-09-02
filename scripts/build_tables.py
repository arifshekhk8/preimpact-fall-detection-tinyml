#!/usr/bin/env python3
"""Assemble Tables I-IV in paper/ from the CSVs pulled back from Kaggle.

Tables are built by script, never by hand, so a number in the paper can always be
traced to the fold that produced it. Re-running this after a new Kaggle result
refreshes everything.

Table IV's three hardware cells are filled from results/hardware_measurements.json
if it exists and stay marked PENDING-HW if it does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)

HW = RESULTS / "hardware_measurements.json"


def _read(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        print(f"  (skip) {name} not present yet")
        return None
    return pd.read_csv(p)


def table_I() -> None:
    df = _read("results_e1.csv")
    if df is None:
        return
    tbl = (df.groupby(["model", "protocol"])
             .agg(folds=("fold", "count"),
                  params=("params", "max"),
                  sensitivity=("bin_sensitivity", "mean"),
                  sens_sd=("bin_sensitivity", "std"),
                  specificity=("bin_specificity", "mean"),
                  spec_sd=("bin_specificity", "std"),
                  macro_f1=("macro_f1_3class", "mean"),
                  f1_sd=("macro_f1_3class", "std"),
                  auc=("bin_auc", "mean"))
             .round(4))
    (PAPER / "table_I.md").write_text(
        "# Table I — within-dataset baseline\n\n"
        "Baselines under 5-fold subject-grouped CV; the proposed model additionally "
        "under full leave-one-subject-out. Mean and standard deviation across folds.\n\n"
        + tbl.to_markdown() + "\n"
    )
    print(f"  table_I.md   ({len(df)} fold rows)")


def table_II() -> None:
    df = _read("results_e2.csv")
    if df is None:
        return
    piv = df.pivot_table(index=["lodo_fold", "train_datasets", "test_dataset"],
                         columns="method", values="bin_f1").round(4)
    order = [m for m in ("none", "instance_norm", "coral", "dann") if m in piv.columns]
    piv = piv.reindex(columns=order)

    # the headline: how far the cross-dataset number falls below the within-dataset one
    e1 = _read("results_e1.csv")
    note = ""
    if e1 is not None and "none" in piv.columns:
        within = e1[e1["model"] == "proposed"]["bin_f1"].mean()
        drop = (within - piv["none"]).round(4)
        piv.insert(len(piv.columns), "drop_vs_table_I", drop)
        note = (f"\n`drop_vs_table_I` is measured against the proposed model's "
                f"within-dataset F1 of {within:.4f} (Table I).\n")

    (PAPER / "table_II.md").write_text(
        "# Table II — cross-dataset generalisation\n\n"
        "Train on two datasets, test on a third. No fine-tuning, no target labels at "
        "any point. F1 on the unseen dataset.\n\n"
        + piv.to_markdown() + "\n" + note +
        "\n**Fold D (UMAFall)** is the never-trained-on, never-tuned stress test, "
        "reported once. It is accelerometer-only and at a different body position "
        "(pocket, not waist), so it measures a combined domain and placement shift and "
        "is not directly comparable with folds A–C.\n"
    )
    print(f"  table_II.md  ({len(df)} rows)")


def table_III() -> None:
    df = _read("table_III.csv")
    if df is None:
        return
    (PAPER / "table_III.md").write_text(
        "# Table III — pre-impact performance\n\n"
        "Lead time is the interval between the model's **first** alarm and the labelled "
        "impact, one value per fall trial, pooled across leave-one-subject-out folds. "
        "Positive means the alarm preceded impact.\n\n"
        + df.round(4).to_markdown(index=False) + "\n\n"
        "**One label source.** KFall's onset and impact frames derive from synchronised "
        "video. The SisFall Enhanced annotations that would have corroborated them are "
        "not recoverable from the available mirror — nb00 measured a 22.5 % ceiling "
        "against a null control — so this result rests on KFall alone.\n"
    )
    print(f"  table_III.md ({len(df)} rows)")


def table_IV() -> None:
    df = _read("table_IV.csv")
    if df is None:
        return
    hw = json.loads(HW.read_text()) if HW.exists() else {}
    col = "INT8 (ESP32, measured)"

    mapping = {
        "Inference latency (ms)": "inference_latency_ms",
        "Peak RAM / tensor arena (KB)": "arena_high_water_bytes",
        "Battery life (hours)": "battery_life_hours",
    }
    filled = 0
    for i, r in df.iterrows():
        key = mapping.get(str(r["Metric"]))
        if key and hw.get(key) is not None:
            v = hw[key]
            if key == "arena_high_water_bytes":
                v = round(v / 1024, 1)
            df.at[i, col] = v
            filled += 1

    pending = int((df[col].astype(str) == "PENDING-HW").sum())
    note = (
        f"\n{pending} cell(s) still marked `PENDING-HW` require the physical ESP32 and "
        "MPU6050. The procedure is in `firmware/MEASUREMENT.md`; the model, firmware "
        "and normalisation constants they depend on are committed and are not changed "
        "by the measurement.\n"
        if pending else
        f"\nHardware measurements taken on {hw.get('board', '?')} with "
        f"{hw.get('imu', '?')}, averaged over {hw.get('n_inferences_averaged', '?')} "
        f"inferences ({hw.get('date', 'date not recorded')}).\n"
    )
    (PAPER / "table_IV.md").write_text(
        "# Table IV — on-device deployment\n\n" + df.to_markdown(index=False) + "\n" + note
    )
    print(f"  table_IV.md  ({filled} hardware cells filled, {pending} pending)")


def ablations() -> None:
    df = _read("table_ablations.csv")
    if df is None:
        return
    (PAPER / "table_ablations.md").write_text(
        "# E5 — ablations\n\n5-fold subject-grouped CV, proposed model.\n\n"
        + df.round(4).to_markdown(index=False) + "\n\n"
        "Placement uses FallAllD rather than UMAFall: UMAFall's multi-position sensors "
        "all sample at 20 Hz, below the 50 Hz working rate, and raising them would "
        "fabricate signal — the same criterion that excluded UP-Fall.\n"
    )
    print(f"  table_ablations.md ({len(df)} rows)")


def main() -> int:
    print("building tables from results/ ...")
    table_I()
    table_II()
    table_III()
    table_IV()
    ablations()
    print(f"\nwrote to {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
