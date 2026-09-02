# Pre-Impact Fall Detection on a Low-Cost Microcontroller — Execution Plan

**Derived from:** `Fall_Detection_Experimental_Plan_v2.pdf` (revision 2, 22 August 2026)
**Execution plan version:** 1.0 — 3 September 2026
**Owner:** Md Arif Shekh (GitHub `arifshekhk8`, Kaggle `arifshekh`)

This document turns the experimental plan into an executable, automated pipeline. The PDF
says *what* to prove; this says *which file runs, on which machine, producing which artifact*.
Where I depart from the PDF, the departure is recorded in §2 with a reason.

---

## 1. What is being built

Three contributions, unchanged from the PDF:

| | Contribution | Concrete artifact that proves it |
|---|---|---|
| **C1** | Leave-one-dataset-out generalisation | Table II — F1 on an unseen dataset, before and after domain adaptation |
| **C2** | Pre-impact detection | Table III — lead time in ms on two independent label sources |
| **C3** | Verified TinyML deployment | Table IV — measured latency / RAM / battery on an ESP32 |

One-sentence pitch: *a ~25k-parameter separable CNN, quantised to INT8, running in under
50 ms on a BDT 450 microcontroller, retains usable pre-impact fall detection accuracy on
datasets it was never trained on.*

---

## 2. Reality check on the data — findings and deviations

I verified all four datasets against Kaggle before writing any code. Three findings change
the plan.

### D1 — The Kaggle "SisFall Enhanced" mirror fails the PDF's §3.2 verification checklist

`nvnikhil0001/sisfall-enhanced` does **not** contain raw signals. It contains six binary
blobs — a pre-computed, pre-split windowed tensor:

```
Three Classes/x_train_3   478,439,424 B   77,871 windows
Three Classes/x_val_3     124,311,552 B   20,233 windows
Three Classes/x_test_3    116,023,296 B   18,884 windows
Three Classes/y_*_3       one-hot uint8, 3 classes
Three Classes/weights_3.txt  class weights: 33.48, 107.67, 1.00
```

Geometry (confirmed arithmetically, exact in all three splits):
**1 window = 6144 bytes = 1536 float32 = 256 samples × 6 channels.** 116,988 windows total.

This fails the checklist on every point that matters:

- No subject IDs → **subject-grouped CV is impossible**, and the shipped split is almost
  certainly by window or trial, so training on it directly would leak.
- Not 200 Hz raw ADC counts → the PDF's unit-conversion and decimation steps cannot be
  applied, and I cannot verify resting magnitude is 1.0 g.
- Already windowed at 256 samples → the window-length ablation (E5) is impossible on it.

The PDF anticipated exactly this (§3.2, and the risk register's first row).

**Response — a three-tier fallback, executed in that order:**

1. **Re-alignment (attempt first, `nb00`).** The windowed tensor is a deterministic function
   of the original recordings. The original SisFall is available raw and *does* carry subject
   IDs in its filenames. So: build an index over the original trials, match each Enhanced
   window back to its source trial by signal content, and thereby recover **both** the Musci
   per-sample label **and** the subject ID. If ≥95 % of windows align unambiguously, C2 keeps
   its second label source and E5 keeps SisFall.
2. **Post-fall only.** If re-alignment fails, use original SisFall with filename-derived
   trial labels for E1/E2/E5 (post-fall task, which needs no temporal labels), and drop
   SisFall from E3.
3. **KFall alone for C2**, exactly as the PDF's risk register prescribes. Costs a
   corroborating source, not the project.

The original SisFall is `adityavvvn/sisfall` — pristine `SisFall_dataset/SA01/D01_SA01_R01.txt`
per-trial files plus the `Readme.txt` that carries the unit-conversion formulas the PDF's
§4.1 step 2 depends on. This is downloaded regardless, as the PDF instructs.

#### D1 resolved — nb00's verdict: tier 3

Three probe rounds settled it. Round 1 matched only 0.8 % of windows, but the 12 it did
confirm were unambiguously real (correlation 0.995–0.997 at plausible offsets, consistent
recovered scale of 1 g → 0.2632 Enhanced units). Correlations near 0.996 rather than ~1.0
say the Enhanced tensors are a *smoothed* version of the originals, so round 2 swept
low-pass cutoffs — recovery rose monotonically, 0 % unfiltered → 27 % at 5 Hz.

That trend invited filtering harder still, which is exactly where the experiment would
have started lying to itself: as bandwidth falls, every window converges on every other
and matches appear by chance. Round 3 therefore swept down to 1 Hz **with a null control**
— the identical search against time-reversed queries, which cannot have true matches:

| cutoff | real | null | net | recovery |
|---|---|---|---|---|
| 5 Hz | 85 | 14 | 71 | **22.5 %** |
| 4 Hz | 31 | 13 | 18 | 5.7 % |
| 3 Hz | 15 | 13 | 2 | 0.6 % |
| 2 Hz | 5 | 3 | 2 | 0.6 % |
| 1 Hz | 1 | 2 | 0 | 0.0 % |

Recovery *peaks* at 5 Hz and collapses below it as the null rises to meet the signal.
22.5 % is the ceiling, not a waypoint. **Tier 3 applies**, on evidence rather than on
giving up:

- original SisFall (25 subjects, filename labels, peak-acceleration impact proxy) stays
  in E1/E2/E5 on the **post-fall** task, so C1 keeps all four datasets;
- **KFall alone carries C2.** The PDF already names KFall the more authoritative source
  (§3.1), so C2 loses corroboration, not its primary evidence.

Additional finding: the available SisFall mirror carries **25 of 38 subjects** — all 23
young, but only SE06 and SE15 of the 15 older participants. Since SisFall's distinctive
value is that it includes genuinely older adults, this is a stated limitation, not a
detail.

#### D5 — UMAFall's waist channel is 20 Hz, so fold D uses the pocket

The PDF states UMAFall's waist channel is the 200 Hz smartphone stream. nb00 measured
every sensor in the release and it is not:

| sensor | position | device | measured rate |
|---|---|---|---|
| 0 | RIGHTPOCKET | smartphone | ~200 Hz |
| 1–4 | CHEST, **WAIST**, WRIST, ANKLE | SensorTag | **~20 Hz** |

The true waist channel runs at 20.1 Hz. Raising it to 50 Hz would fabricate signal
content — which is the exact criterion the PDF used to exclude UP-Fall (§3.3), and it has
to apply here too or that exclusion was never principled.

So UMAFall is used at **RIGHTPOCKET**, and the cost is stated rather than hidden: fold D
measures a combined domain *and* placement shift, not the clean waist-to-waist transfer
of folds A–C. That suits its role as the hardest row — never trained on, never tuned,
reported once. Dropping UMAFall entirely would cost the only fourth dataset to buy a
purity the stress test does not need.

Knock-on: E5's sensor-placement ablation cannot use UMAFall's multi-position channels
(all 20 Hz). It uses **FallAllD**, which carries Neck, Waist and Wrist at 238 Hz.

### D2 — Preprocessing runs on Kaggle, not on the MacBook

The PDF §7 puts preprocessing on the M4 and uploads cached `.npz` to Kaggle. All four raw
datasets are *already hosted on Kaggle*, so downloading ~2.5 GB to the laptop only to upload
a derivative back is wasted work. Preprocessing instead runs as a **CPU-only Kaggle notebook**
(`nb01`), which consumes **zero GPU quota** and writes the cached `.npz` straight into a
Kaggle Dataset. The PDF's actual rule — "never re-parse raw files inside a GPU session"
(§7.2 rule 1) — is honoured, and honoured more cheaply. The M4 keeps its role for
firmware, plotting, and paper writing.

### D3 — Dataset sources, final

| Dataset | Source used | Note |
|---|---|---|
| SisFall (original) | `adityavvvn/sisfall` | Raw `.txt`, 200 Hz, 9 cols, has `Readme.txt`. Signal source of record. |
| SisFall Enhanced | `nvnikhil0001/sisfall-enhanced` | Labels only, via re-alignment (D1). |
| KFall | `usmanabbasi2002/kfall-dataset` | `sensor_data/SA06/S06T01R01.csv` + `label_data/SA06_label.xlsx`. Matches the PDF exactly. |
| FallAllD | `sankalpsinghvishen/derived-fallalld-dataset` | `FallAllD.pkl` — the official IEEE DataPort pandas-pickle distribution, 462 MB. |
| UMAFall | `thanushanth/umafall` | Per-trial CSVs, `UMAFall_Subject_01_ADL_Aplausing_1_*.csv`. |

UP-Fall stays excluded, for the two reasons in PDF §3.3.

### D4 — C3's measured column needs hardware I do not have

E4 requires an ESP32 + MPU6050 on a bench. I can produce, and will: the firmware, the
`model.h`, a compiling build, the FP32→INT8 accuracy delta, and the model's exact flash and
tensor-arena sizes as reported by the toolchain. **Measured latency, peak RAM high-water mark,
and battery life require the physical board.** Table IV ships with those three cells marked
`PENDING-HW` and a `firmware/MEASUREMENT.md` giving the exact procedure to fill them. Nothing
downstream is blocked by this; C1 and C2 complete regardless.

---

## 3. Repository and artifact layout

GitHub repo `arifshekhk8/preimpact-fall-detection-tinyml`, created **private**
(it becomes public at submission — PDF §13.3 — but an unpublished result stays private until
then; flipping it is one command).

```
.
├── PROJECT_PLAN.md              this file
├── README.md                    what it is, how to reproduce
├── requirements.txt
├── src/fdlib/                   THE shared library — single source of truth
│   ├── config.py                every constant: 50 Hz, 2.0 s, stride 0.5 s, seeds
│   ├── preprocess.py            §4.1's seven steps, FROZEN. Used by training AND firmware.
│   ├── datasets/                sisfall.py kfall.py fallalld.py umafall.py — raw → (sig, labels, subject)
│   ├── windowing.py             windowing + the two labelling rules (post-fall, pre-impact)
│   ├── cv.py                    GroupKFold(5) / full LOSO / LODO fold generators
│   ├── models.py                proposed separable CNN + 1D-CNN + CNN-LSTM
│   ├── baselines.py             SMV threshold, SVM, Random Forest
│   ├── adapt.py                 per-window instance norm, CORAL, DANN
│   ├── metrics.py               sens/spec/F1/AUC + lead time + false alarms per ADL hour
│   └── tflite_export.py         INT8 conversion + representative dataset from TRAIN only
├── kaggle/                      one directory per notebook: script + kernel-metadata.json
│   ├── nb00_probe/  nb01_preprocess/  nb02_e1/  nb03_e2/  nb04_e3/  nb05_e5/  nb06_export/
├── firmware/esp32/              sketch, model.h, MEASUREMENT.md
├── scripts/
│   ├── sync_fdlib.py            push src/fdlib → Kaggle Dataset arifshekh/fdlib
│   ├── run_kernel.py            push a kernel, poll to completion, pull output
│   └── build_tables.py          results/*.csv → paper/table_I..IV.md
├── results/                     every CSV pulled back from Kaggle (committed)
└── paper/                       Tables I–IV, figures, refs.bib
```

**How Kaggle sees the code.** A Kaggle kernel cannot clone a private repo without embedding a
token. So `src/fdlib` is mirrored to a Kaggle Dataset `arifshekh/fdlib`, attached to every
notebook, and `sys.path`-inserted. `scripts/sync_fdlib.py` re-uploads a new version on every
library change, and each notebook logs the fdlib version it ran against. One library, one
definition of preprocessing, on both machines — which is the PDF §4's central demand.

---

## 4. Frozen preprocessing contract (PDF §4.1)

Implemented once in `src/fdlib/preprocess.py`. Changing it invalidates every result, so it
is version-stamped and hashed into every output file.

1. **Channels** — waist / low-back only. `ax, ay, az, gx, gy, gz`. Discard magnetometer,
   barometer, and SisFall's second accelerometer (ADXL345 kept, MMA8451Q dropped).
2. **Units** — accelerometer → *g*, gyroscope → *deg/s*, using each dataset's documented
   conversion. SisFall per its `Readme.txt`: `a[g] = (2*Range/2^Res) * raw`.
3. **Anti-alias then decimate** — `scipy.signal.decimate` (which filters internally) to 50 Hz.
   Never array-slice.
4. **Window** — 2.0 s = 100 samples × 6 ch, stride 0.5 s = 50 samples, 75 % overlap.
   Identical stride at inference.
5. **Label** — post-fall: positive if the window contains the impact. Pre-impact: positive if
   the window's **right edge** lies inside the alert interval.
6. **Normalise** — channel-wise mean/std from **training subjects only**. Twelve numbers,
   written to `results/norm_constants.json`, applied unchanged to val/test/firmware.
   Never recomputed on test data.
7. **Cache** — compressed `.npz` → Kaggle Dataset `arifshekh/fall-windows-50hz`.

**Sanity gates (PDF §4.2) — `nb01` fails loudly if any of these fail:**

- Resting acceleration magnitude is 1.0 ± 0.05 g in every dataset.
- Gravity sits on the same axis with the same sign everywhere; the rotation applied to each
  dataset is printed and committed. *This is the cross-dataset trap the PDF warns about — an
  unresolved 180° mount difference is indistinguishable from a genuine domain shift.*
- A fall and a walk trial from each dataset, plotted on shared axes at 50 Hz, are
  qualitatively comparable (`results/fig_sanity_traces.png`).
- Windows per class per dataset, and the imbalance ratios, are tabulated.
- Ten SisFall Enhanced alert intervals overlaid on their acceleration traces
  (`results/fig_alert_intervals.png`) — the only defence against inheriting an annotation error.

---

## 5. Experiments — what runs where

Tiered CV per the PDF's revision: `GroupKFold(5)` grouped by subject for everything
comparative, full 38-fold LOSO for the headline number only.

| ID | Notebook | Protocol | GPU est. | Output |
|---|---|---|---|---|
| **nb00** | probe | — | CPU | `results/probe_report.md`, re-alignment verdict |
| **nb01** | preprocess | — | CPU | Dataset `fall-windows-50hz`, sanity figures |
| **E1** | nb02 | 5-fold subject-grouped, 6 models; + full 38-fold LOSO for proposed | ~7 h | Table I |
| **E2** | nb03 | LODO folds A–D × {none, inst-norm, CORAL, DANN} | ~1.5 h | Table II |
| **E3** | nb04 | KFall LOSO (primary) + SisFall Enh. (corroborating) | ~3 h | Table III + lead-time curve |
| **E5** | nb05 | window 1.0/1.5/2.0 s; rate 25/50/100 Hz; accel-only; placement | ~5 h | Ablation figures |
| **E4** | nb06 | INT8 export, desktop verification | CPU | `model.tflite`, `model.h`, Table IV (desktop) |

Total ≈ 17 GPU-hours against a 30 h/week quota. Compute is not the constraint.

**Fold definitions, fixed now (PDF §5, E2):**

| Fold | Train | Test (unseen) |
|---|---|---|
| A | SisFall Enh. + KFall | FallAllD |
| B | SisFall Enh. + FallAllD | KFall |
| C | KFall + FallAllD | SisFall Enh. |
| D | All three | UMAFall — *held out throughout, reported once* |

UMAFall is never used for training, tuning, or model selection in any fold. That is the
whole point of it.

**The headline result is the gap between E1 and E2, and the gap is reported honestly even
if it is large.** Concealing it is precisely what the PDF accuses the existing literature of.
Adaptation is then attempted in ascending cost order: per-window instance normalisation
first (near-zero MCU cost), then CORAL, then DANN only if the first two are insufficient.

**E3 metric of record** is *lead time* — milliseconds between the model firing and the
labelled impact — reported as a **distribution, not just a mean**, at thresholds 0.5/0.7/0.9,
alongside false alarms per hour of ADL data. Lead time is reported **separately** for KFall
and SisFall Enhanced; if they disagree, that disagreement is discussed, not averaged away.

---

## 6. Model

Target ~25,000 parameters, ~25 KB at INT8. The MCU memory budget is the design driver.

```
Input 100 × 6                          (2.0 s @ 50 Hz, 6 IMU channels)
  ├── Instance normalisation (per-window, per-channel)
  ├── Conv1D(24, k=7, s=2) + BN + ReLU            -> 50 × 24
  ├── SeparableConv1D(48, k=5) + BN + ReLU        -> 50 × 48
  ├── MaxPool1D(2)                                -> 25 × 48
  ├── SeparableConv1D(64, k=3) + BN + ReLU        -> 25 × 64
  ├── GlobalAveragePooling1D                      -> 64
  ├── Dense(32) + ReLU + Dropout(0.3)
  └── Dense(N) + Softmax                          N = 3 (pre-impact) or 2 (post-fall)
```

Three-class head is primary; binary metrics are derived by merging alert+fall, so one trained
model yields both result sets.

Training: Adam, LR 1e-3 with cosine decay, batch 128, ≤100 epochs, early stopping on
validation macro-F1, patience 15. Class imbalance handled by **class weighting, not
oversampling** — at 75 % overlap, oversampling duplicates near-identical windows across the
split.

**Split by subject, never by window.** Every split in this project is subject-wise or
dataset-wise. A random window split yields >99 % accuracy that means nothing.

Seeds fixed (`config.SEED = 1337`) and stated.

---

## 7. Deployment (PDF §9)

1. `TFLiteConverter` with `Optimize.DEFAULT`, `int8` in/out, representative dataset of ~200
   windows drawn **from the training split only** — drawing from test leaks and invalidates.
2. Verify the quantised model in the Python interpreter on the full test set before flashing.
   A drop >2 points is a quantisation bug, not a hardware one, and is fixed on the desktop.
3. `xxd -i model.tflite > model.h`, array marked `const` so it lands in flash.
4. Firmware: 50 Hz hardware timer ISR into a ring buffer; every 25 new samples (0.5 s stride)
   copy a 100 × 6 window, convert to g and deg/s, apply the **frozen** normalisation constants,
   quantise with the model's own scale/zero-point, invoke, dequantise, threshold, fire buzzer.
   No `delay(20)` — timer jitter shifts window content relative to training.
5. Instrument `esp_timer_get_time()` around invoke, average 1,000 runs; read the tensor arena
   high-water mark. These two numbers are C3.

Sensor config: ±16 g (fall impacts exceed 8 g and a clipped peak is unrecoverable),
±2000 deg/s, I²C at 400 kHz.

---

## 7a. Kaggle operating rules (non-negotiable)

These come from how this Kaggle account actually behaves, not from the PDF. They are
enforced in `scripts/run_kernel.py` rather than left to discipline, because every one
of them is the kind of rule that gets forgotten at 2 a.m. on fold 34.

| # | Rule | Enforcement |
|---|---|---|
| **K1** | **Every kernel requests T4 × 2.** Other accelerator selections crash this account's sessions. | `kernel-metadata.json` must carry `"enable_gpu": true` and `"machine_shape": "NvidiaTeslaT4"`; the runner also passes `--accelerator NvidiaTeslaT4` on push, then **reads the metadata back from Kaggle** and aborts if the request was not honoured. |
| **K2** | **One notebook at a time. Never train two concurrently.** Concurrent sessions exhaust the account's resources and fail together, losing both runs. | `.kaggle_run.lock` — a second invocation exits rather than pushing. |
| **K3** | **Any notebook that errors halts the pipeline immediately.** Nothing downstream runs on a broken upstream artifact. | Non-`complete` terminal state → dump the kernel log, return non-zero, skip every remaining notebook and say which were skipped. |

Consequence for K2: the experiment schedule is strictly serial —
`nb01 → nb02 → nb03 → nb04 → nb05 → nb06`. The ~17 GPU-hour budget is wall-clock time,
not something to be compressed by fanning out. This is affordable: the budget is
roughly half of one week's 30-hour quota.

Consequence for K3: per-fold CSV appends and `--start-fold` (PDF §7.2) matter more, not
less. A halt at fold 34 of 38 must cost one fold, not the run.

Two caveats worth stating plainly rather than discovering later:

- The Kaggle CLI exposes no `kernels cancel` verb, so a run that has already started
  **cannot be killed from the command line**. What the tooling guarantees is that a
  failed or hung kernel stops the pipeline and never has a successor pushed alongside
  it; killing a still-billing session requires the web UI.
- An `"accelerator"` key in `kernel-metadata.json` is **silently dropped** by the CLI —
  the kernel then runs on a single P100 while the local file still says otherwise. The
  field Kaggle actually reads is `machine_shape`. This was caught in practice on nb00,
  which is why K1 now verifies by reading the stored metadata back rather than trusting
  the local copy.

---

## 8. Automation

The whole pipeline is driven from the laptop; nothing is clicked in a browser.

```
make sync        # src/fdlib -> Kaggle Dataset (new version), records the version hash
make probe       # nb00
make preprocess  # nb01 -> cached windows dataset
make e1 e2 e3 e5 # one at a time, in this order (rule K2)
make export      # nb06 -> model.tflite, model.h
make tables      # results/*.csv -> paper/table_I..IV.md
make firmware    # compile the ESP32 sketch
```

`scripts/run_kernel.py` handles push → poll (`kaggle kernels status`) → pull
(`kaggle kernels output`), enforces rules K1–K3 above, and writes a line to
`results/experiment_log.csv`: date, notebook, fdlib version, git commit, config hash, result.
That log is the answer when a reviewer asks how a number was produced.

Passing several notebooks in one invocation runs them **sequentially**, stopping at the
first failure:

```
python scripts/run_kernel.py nb02_e1 nb03_e2 nb04_e3 nb05_e5 nb06_export
```

**Budget-protecting rules carried over from PDF §7.2:** results are appended to CSV after
*every fold*, not at the end; every training entry point accepts `--start-fold` so a killed
12-hour session resumes rather than restarts; checkpoints go to `/kaggle/working/`.

**Git discipline:** one commit per completed stage, imperative subject line, body stating what
ran and what it produced. No AI co-author trailer, no generated-by footer.

---

## 9. Execution order

Each step's exit condition is a committed artifact. I do not advance past a red gate.

| # | Step | Exit condition |
|---|---|---|
| 0 | Repo, library skeleton, Kaggle fdlib dataset | `make sync` succeeds |
| 1 | **nb00 probe** — parse one trial from each of the four datasets; attempt Enhanced re-alignment | `results/probe_report.md` committed; D1 tier chosen and recorded |
| 2 | **nb01 preprocess** — seven steps, all §4.2 sanity gates | Windows dataset published; gates green; sanity figures committed |
| 3 | **nb02 / E1** — six comparators, 5-fold grouped; LOSO for proposed | Table I filled |
| 4 | **nb03 / E2** — four LODO folds × adaptation ladder | Table II filled — *the headline* |
| 5 | **nb04 / E3** — lead time, both label sources | Table III + lead-time-vs-false-alarm curve |
| 6 | **nb05 / E5** — ablations | Ablation figures |
| 7 | **nb06 / E4** — INT8 export + desktop verification | `model.tflite`, `model.h`, Table IV desktop columns |
| 8 | Firmware compiles; `MEASUREMENT.md` written | Sketch builds; hardware cells marked `PENDING-HW` |
| 9 | Tables, figures, `refs.bib`, README | `paper/` complete |

Step 1's verdict determines whether C2 has one label source or two. That is the single
highest-leverage unknown, so it is resolved first, before any model is written.

---

## 10. Risks, and what I actually do about them

| Risk | Mitigation — concrete |
|---|---|
| Enhanced re-alignment fails | Tier-2/3 fallback in §2 D1. Costs a corroborating source, not the project. |
| Axis conventions differ between labs | §4.2 gate 2 runs *before* any cross-dataset claim. Rotation per dataset printed and committed. A 180° mount difference looks exactly like a domain shift. |
| Cross-dataset drop doesn't close | "The gap is larger and harder to close than the literature implies" is a legitimate, publishable negative result — and it is C1's actual claim. Frame it, don't bury it. |
| Model exceeds the tensor arena | Window 2.0→1.0 s and rate 50→25 Hz cut input size 4×. E5 generates these numbers anyway. |
| Kaggle session dies mid-LOSO | Per-fold CSV append + `--start-fold`. A death at fold 34 of 38 costs one fold. |
| Dataset host goes offline | All four already mirrored on Kaggle; nb01's cached output is itself a durable Kaggle Dataset. |
| No ESP32 on the bench | §2 D4. Firmware and desktop INT8 numbers ship; three cells marked `PENDING-HW` with a written procedure. |
| Timeline slips | C1 + C3 alone are a complete paper. Drop C2 first. |

---

## 11. Definition of done

- `results/` holds a CSV for every fold of every experiment, plus `experiment_log.csv`.
- `paper/table_I.md` … `table_IV.md` are filled from those CSVs by script, not by hand.
- `model.tflite` and `model.h` are committed; the FP32→INT8 delta is under 2 points.
- `results/norm_constants.json` is committed — the frozen twelve numbers.
- The ESP32 sketch compiles; `MEASUREMENT.md` states exactly how to fill the hardware cells.
- Seeds, dataset versions, and download dates are recorded.
- README reproduces the whole thing from a clean checkout.

---

## 12. Citation obligations

Using the Enhanced annotations obliges citing **both** Sucerquia et al. 2017 (the original
recordings) *and* Musci et al. 2018/2020 (the annotation work). Citing only one is the kind of
omission that makes reviewers doubt everything else. KFall (Yu et al. 2021), FallAllD
(Saleh et al. 2020), and UMAFall (Casilari et al.) are cited on use. `paper/refs.bib` carries
all nine references from PDF §14 and is built in step 9, not left to the end.
