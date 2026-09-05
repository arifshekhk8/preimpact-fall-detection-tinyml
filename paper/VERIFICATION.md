# Verification: every number in the paper, traced to its source

Audit performed before submission. Each claim in `paper/ieee/main.tex` was checked
against the committed result files. **Three errors were found and fixed**; they are
recorded here rather than quietly corrected.

---

## Errors found and fixed

### 1. Macro-F1 compared against binary F1 (abstract, Section V-B)

The abstract claimed cross-dataset F1 *"falls from 0.95 within-dataset to between 0.34
and 0.84."* Table~III (leave-one-dataset-out) reports **binary** F1, but 0.95 is the
proposed model's **macro**-F1 from Table~I. The two are not comparable.

The correct within-dataset comparator is the proposed model's binary F1:

```
results/results_e1.csv, model=proposed, protocol=grouped5
  macro_f1_3class : 0.9502
  bin_f1          : 0.9067   <- the right number
```

**Fixed:** the paper now states 0.907, and Table~III's caption gives it as the explicit
reference value.

### 2. Two different models conflated (abstract, Section V-F)

The abstract attributed *"0.985 trial-level sensitivity and 0.991 specificity ... 318 ms"*
to **the deployed model**. Those figures come from the nb04 model, trained and
evaluated within KFall alone under 32-fold LOSO. The deployed model (nb07) is trained
on three corpora and reaches 0.958 / 0.836 / 539 ms on held-out subjects.

**Fixed:** the abstract now says "scored per trial on KFall as in the benchmark
literature" without calling it the deployed model. Section V-F reports the deployed
model's own figures and explains why they are lower --- it is held to three
heterogeneous corpora at once rather than one.

### 3. Unverifiable size ratio (abstract)

The abstract claimed the model is *"roughly 1/70"* the size of the ConvLSTM benchmark.
We have no verified parameter count or model size for that network; the ratio was
inferred from a secondary source.

**Fixed:** removed. The paper now states only what is measured --- 7,947 parameters and
22.5 KB --- and says "a substantially larger published benchmark" without a fabricated
ratio. Size ratios against the comparators we trained ourselves (13$\times$ fewer
parameters than the 1D-CNN, 6$\times$ fewer than the CNN-LSTM) are exact and retained.

---

## Claims verified as correct

| Claim in paper | Value | Source |
|---|---|---|
| Model parameters | 7,947 | `results/quantisation_report.json` |
| Flatten would cost 51,200 | $25\times64\times32$ | arithmetic |
| Table I, all rows | — | `results/table_I.csv` |
| Table II (KFall pre-impact), all rows | — | `results/table_Ib.csv` |
| Table III (LODO), all rows | — | `results/table_II.csv` |
| Instance norm gain, folds A–C | +0.031 to +0.044 | `results/table_II.csv` |
| Table IV rows (ours) | 0.985/0.991/318±328; 0.978/0.995/342±373 | `results/trial_level_eval.csv` |
| Published rows in Table IV | as cited | Yu 2021, TinyFallNet 2023 |
| 293 false alarms/hour at $\tau$=0.5 | 292.892 | `results/table_III.csv` |
| FP count 538 → 26 at $k$=3 | 538, 26 of 2,729 | `results/trial_level_eval.csv` |
| Mean lead 663 ms at $k$=1 | 663.5 | `results/table_III.csv` |
| Quantisation: 30.95 pp vs −0.22 pp | 30.951, −0.217 | `results/quantisation_comparison.csv` |
| Agreement 0.686 vs 0.994 | same | `results/quantisation_comparison.csv` |
| Deployed model FP32/INT8 macro-F1 | 0.7109 / 0.7138 | `results/quantisation_report.json` |
| Deployed sensitivity 0.922 | 0.9222 / 0.9224 | `results/quantisation_report.json` |
| Deployed 0.958/0.836/539±509, 68% pre-impact | $\tau$=0.95, $k$=1 | `results/final_operating_points.csv` |
| $k$=2 raises spec to 0.975, pre-impact 16% | 0.9754, 0.1623 | `results/final_operating_points.csv` |
| 814 fall / 1,136 ADL trials | same | `results/final_operating_points.csv` |
| Ablations: 0.804 / 0.871 / 0.903 | window 1.0/1.5/2.0 s | `results/table_ablations.csv` |
| 100 Hz buys 0.006 F1 | 0.9095 − 0.9033 | `results/table_ablations.csv` |
| Gyroscope worth 2.6 points | 0.9054 → 0.8798 | `results/table_ablations.csv` |
| Placement neck/waist/wrist | 0.644 / 0.551 / 0.476 | `results/table_ablations.csv` |
| 1/57 of Random Forest node count | 451,336 / 7,947 = 56.8 | `results/table_Ib.csv` |
| SisFall 25 of 38 subjects | 25 (23 SA, SE06 + SE15) | `results/nb00_probe_report.md` |
| Enhanced recovery ceiling 22.5% | 0.2248 net of null | `results/nb00_probe_verdict.json` |
| UMAFall waist 20.1 Hz | measured per sensor | `results/nb00_probe_report.md` |
| FallAllD gravity axis differs | axis 0 (+) vs axis 1 (−) | `results/gate_report.json` |
| Resting magnitude 1.0 ± 0.05 g | 0.988–1.011 across corpora | `results/gate_report.json` |
| Model size 22.5 KB / FP32 163.3 KB | 22.54 / 163.3 | `results/quantisation_report.json` |

---

## Not yet measured

Four cells in Table~V are marked `[HW]` in red and **must not be submitted as-is**:
tensor arena, inference latency, end-to-end latency, battery life. They require the
physical device; the procedure is in `firmware/MEASUREMENT.md`. Fill
`results/hardware_measurements.json`, run `scripts/build_tables.py`, and rebuild.

## Known softness, disclosed in the paper

- Fold D (UMAFall) confounds domain shift with placement and modality; flagged in
  Table III's footnote and in the limitations.
- The placement ablation is FallAllD-internal and is not comparable with the window,
  rate and channel arms, which come from SisFall.
- False-alarm rates come from laboratory ADL recordings, not free-living wear.
- Table IV compares our pooled 32-fold LOSO against published held-out-split results;
  ours is the stricter protocol, which the text states.
