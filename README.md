# Pre-Impact Fall Detection on a Low-Cost Microcontroller

A cross-dataset generalisation and TinyML deployment study: can a ~25k-parameter
separable CNN, quantised to INT8 and running in under 50 ms on a BDT 450 ESP32, retain
usable pre-impact fall detection accuracy on datasets it was never trained on?

Three contributions:

| | Claim | Evidence |
|---|---|---|
| **C1** | Leave-one-dataset-out generalisation, reported honestly | Table II — F1 on an unseen dataset, before and after domain adaptation |
| **C2** | Pre-impact detection, not post-hoc | Table III — lead time in ms, as a distribution |
| **C3** | Verified TinyML deployment | Table IV — measured latency, RAM and battery on real hardware |

The full execution plan, including every deviation from the original experimental
design and why, is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## What makes this reproducible

- **One definition of preprocessing.** `src/fdlib/preprocess.py` is imported by the
  training notebooks *and* generates the C constants the firmware uses. If preprocessing
  drifted between training and the device, the model would score 99 % in Python and
  behave randomly on the ESP32.
- **Every split is by subject or by dataset, never by window.** At 75 % overlap a random
  window split puts near-identical windows on both sides of the boundary and reports
  accuracy above 99 % that means nothing.
- **Every cached artifact is stamped** with `preprocess_signature()`, and loading a
  corpus built under a different signature raises rather than silently proceeding.
- **Seeds fixed** (`config.SEED = 1337`), dataset versions and access dates recorded.

## Layout

```
src/fdlib/            the shared library -- single source of truth
  config.py           every constant: 50 Hz, 2.0 s windows, stride 0.5 s, seeds
  preprocess.py       the frozen seven-step contract + axis canonicalisation
  datasets/           sisfall, kfall, fallalld, umafall -> Trial objects
  windowing.py        windowing and the two labelling rules
  cv.py               subject-grouped / LOSO / leave-one-dataset-out folds
  models.py           the proposed separable CNN + comparators
  baselines.py        SMV threshold, SVM, Random Forest
  adapt.py            instance norm, CORAL, DANN
  metrics.py          sensitivity/specificity/F1/AUC + lead time + false alarms/hour
  experiment.py       fold runner with per-fold CSV append and resume
  tflite_export.py    INT8 conversion, desktop verification, model.h
kaggle/               one directory per notebook: script + kernel-metadata.json
scripts/              sync_fdlib.py, run_kernel.py, build_tables.py
results/              every CSV, figure and report pulled back from Kaggle
firmware/esp32/       sketch, model.h, MEASUREMENT.md
paper/                Tables I-IV, figures, refs.bib
```

## Reproducing

Everything trains on Kaggle; nothing is clicked in a browser.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/sync_fdlib.py          # publish the library to Kaggle
.venv/bin/python scripts/run_kernel.py nb00_probe
.venv/bin/python scripts/run_kernel.py nb01_preprocess
.venv/bin/python scripts/run_kernel.py nb02_e1 nb03_e2 nb04_e3 nb05_e5 nb06_export
```

Notebooks passed in one invocation run **sequentially**, stopping at the first failure.

### Three Kaggle rules the harness enforces

1. **T4 × 2 on every kernel.** `kernel-metadata.json` must set `"machine_shape":
   "NvidiaTeslaT4"`; the runner passes `--accelerator` too and then reads the metadata
   back from Kaggle to confirm the request was honoured. An `"accelerator"` key is
   silently dropped by the CLI and the kernel falls back to a single P100.
2. **One kernel in flight.** A lock file makes a concurrent second run impossible.
3. **Halt on first failure.** A non-`complete` terminal state dumps the kernel log and
   skips everything downstream.

## Data

| Dataset | Source | Role |
|---|---|---|
| SisFall | `adityavvvn/sisfall` | Post-fall training corpus, LODO fold |
| KFall | `usmanabbasi2002/kfall-dataset` | **Primary** pre-impact source (video-grounded onset/impact frames) |
| FallAllD | `sankalpsinghvishen/derived-fallalld-dataset` | Independent hardware, LODO fold, placement ablation |
| UMAFall | `thanushanth/umafall` | Held out throughout; reported once |

Three findings from `nb00` that changed the design, all detailed in
[PROJECT_PLAN.md §2](PROJECT_PLAN.md):

- The SisFall **Enhanced** annotations are not recoverable from the Kaggle mirror
  (shuffled, pre-windowed tensors with no subject IDs). A three-round re-alignment
  attempt with a null control put the ceiling at 22.5 %, so **KFall alone carries C2**.
- The SisFall mirror has **25 of 38 subjects**, missing 13 of the 15 older participants.
- UMAFall's **waist channel runs at 20 Hz**, not the 200 Hz assumed; only the pocket
  smartphone reaches 200 Hz, and it logs no gyroscope.

## Citation obligations

Using these datasets obliges citing Sucerquia et al. 2017 (SisFall), Musci et al.
2018/2020 (the Enhanced annotations), Yu et al. 2021 (KFall), Saleh et al. 2020
(FallAllD) and Casilari et al. (UMAFall). `paper/refs.bib` carries all of them.
