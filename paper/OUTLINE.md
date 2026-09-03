# Paper outline

Target venues (verify current deadlines and templates on each conference site — do not
rely on last year's dates): ICCIT, ICECE, ICAEEE, STI, IEEE TENSYMP. If results are
strong, MDPI *Sensors* and IEEE *Access* are plausible journal targets, though both
carry article processing charges worth confirming with the supervisor first.

---

## 1. Introduction

Falls as a health burden; why wearable detection; the two gaps this paper addresses.

- **Gap 1 — untested generalisation.** Within-dataset accuracy in this literature is
  saturated near 99 %, but published benchmarking shows performance degrades
  considerably once evaluation extends to other datasets. Most papers never leave their
  training dataset.
- **Gap 2 — unverified deployment.** Nearly all fall-detection papers report inference
  on desktop GPUs. Papers combining leave-one-dataset-out evaluation *and* real MCU
  measurements are rare.

Contributions C1, C2, C3 as stated in [PROJECT_PLAN.md](../PROJECT_PLAN.md).

## 2. Related work

Wearable fall datasets and their annotation quality; deep learning approaches; the
saturation of within-dataset results; the small pre-impact literature; TinyML on MCUs.
Position against `tinyfallnet2023` (closest on C2/C3) and `torti2019embedding` (same
group as the Enhanced annotations, deploys to an MCU — simultaneously the closest
related work and the most useful implementation guide).

## 3. Datasets and preprocessing

Condensed from PROJECT_PLAN §2 and §4. Must include, not bury:

- **The annotation-quality discussion.** KFall's labels are video-grounded; SisFall's
  Enhanced annotations were produced by expert panels working from sensor data alone,
  a process subsequent authors have criticised as necessarily subjective. This is why
  KFall is primary for C2 and why the asymmetry is acknowledged rather than averaged
  away.
- **The UP-Fall exclusion rationale** (§3.3 of the experimental plan): ~18.4 Hz
  accelerometers that cannot reach the 50 Hz working rate without fabricating signal,
  and ~850 GB dominated by camera imagery that is never opened. Explicit exclusion
  criteria are a small contribution in themselves.
- **Three findings that constrain the work**, all from `nb00`/`nb01`:
  1. The SisFall Enhanced annotations are not recoverable from the available mirror.
     A three-round re-alignment with a null control put the ceiling at 22.5 %, so C2
     rests on KFall alone.
  2. The SisFall mirror carries 25 of 38 subjects, missing 13 of the 15 older
     participants — the cohort that gives SisFall its distinctive value.
  3. UMAFall's waist sensor samples at 20 Hz; only its pocket smartphone reaches
     200 Hz, and that stream logs no gyroscope. Fold D therefore measures a combined
     domain and placement shift.
- **The axis-convention correction.** FallAllD rests gravity on a different axis and
  sign from SisFall and KFall. Uncorrected, this is indistinguishable from a genuine
  domain shift, and C1 would have measured an upside-down sensor. The rotation applied
  to each dataset is published.

## 4. Proposed method

Architecture (§6 of the plan), the parameter budget, and why separable convolutions
plus global average pooling rather than a flatten. State plainly that the architecture
as specified comes to **7,947 parameters** rather than the ~25,000 originally targeted,
and that being under budget was kept rather than "fixed" by widening layers.

Quantisation strategy; the domain-adaptation ladder and why it is climbed in ascending
order of MCU cost.

## 5. Experimental setup

E1–E5 protocols; the tiered cross-validation rationale (5-fold subject-grouped for
comparisons, full LOSO for the headline, leave-one-dataset-out for C1); hardware
description; metric definitions — in particular **lead time** and **false alarms per
hour of ADL data**, the latter being the metric that decides whether a device is
wearable at all and which most papers omit.

State the leakage guarantee explicitly: every split is subject-wise or dataset-wise,
never window-wise, because at 75 % overlap a random window split reports >99 % accuracy
that means nothing.

## 6. Results

Tables I–IV plus the lead-time-versus-false-alarm curve and the ablation figures.

**The headline is the gap between Table I and Table II.** Report it honestly. If the
gap is large, that *is* the finding — concealing it is precisely what this paper
accuses the existing literature of doing. "The gap is larger and harder to close than
the literature implies" is a legitimate, publishable result.

## 7. Discussion

What the cross-dataset gap means for deployed systems. Limitations, stated without
hedging:

- simulated falls performed by mostly young participants, not real falls by older
  adults;
- single sensor placement in the primary folds;
- annotation subjectivity in every source except KFall;
- the incomplete SisFall mirror;
- fold D confounds domain shift with placement shift and sensor modality.

## 8. Conclusion and future work

---

## Reproducibility checklist (plan §13.3)

- [x] Public repository with preprocessing, training and firmware code
- [x] Frozen normalisation constants published as JSON (and as the C header the
      firmware actually compiles against — same numbers, one generator)
- [x] `.tflite` model file released (`results/model.tflite`, 22.5 KB INT8)
- [x] Exact dataset versions and access dates recorded
- [x] Random seeds fixed and stated (`config.SEED = 1337`)
