# Showcase day — run sheet

Everything you need on the day, in order. Budget ~45 minutes for setup.

---

## 1. Flash the device (15 min)

Copy into your Arduino sketch folder:

```
firmware/esp32/fall_detector.ino
firmware/esp32/model.h            <- the trained model, 22.5 KB INT8
```

Arduino IDE → **ESP32 Dev Module** → install library **TensorFlowLite_ESP32** →
Upload. Serial Monitor at **115200**.

Wiring is unchanged from your AI-free build: SDA 21, SCL 22, buzzer GPIO 4, LED GPIO 2.

**Expected boot output:**

```
MPU6050 READY (+/-16 g, +/-2000 dps, DLPF on, 50 Hz)
model flash    : 23080 bytes (22.5 KB)
arena at init  : ..... of 24576 bytes
normalisation  : PER-WINDOW instance norm (not frozen constants)
System ready.
```

If you see `AllocateTensors failed`, raise `kArenaSize` (line ~101) and reflash.

### Sanity check — do this before anything else

Put the board flat on the table and leave it. **It must not alarm.** Rotate it slowly
— still nothing. If it alarms while stationary, something is wrong with scaling or
orientation and no demo will work. Fall back to `mode rule` and demo the rule detector.

---

## 2. Capture Table IV's numbers (10 min)

Let it run ~8 minutes, then type:

```
measure
```

Copy these four into `results/hardware_measurements.json`:

| Serial line | JSON key |
|---|---|
| `invoke mean` | `inference_latency_ms` |
| `arena high water` | `arena_high_water_bytes` |
| `end-to-end mean` | `end_to_end_latency_ms` |
| (battery run, optional) | `battery_life_hours` |

Then:

```bash
python scripts/build_tables.py
cd paper/ieee && tectonic -X compile main.tex
```

That fills the four red `[HW]` cells in the paper. **Do not submit the paper with red
`[HW]` markers still in it.**

---

## 3. The live demo

### Setup
Strap the device at the **waist**, buzzer audible. Have a mattress or thick cushion.

### Commands you'll want

| Command | Effect |
|---|---|
| `test` | Blink LED + beep 3× — proves the alarm path works |
| `mode both` | Run CNN and rule detector together |
| `mode cnn` / `mode rule` | Switch which one fires |
| `measure` | Show latency, RAM, alarm counts |
| `status` | Sensor and detector state |
| `reset` | Clear counters between runs |
| `thr 0.95` / `k 1` | Retune live, no reflash |

### Suggested 4-minute script

1. **`test`** — "the alarm hardware works."
2. **Hold it still** — "no false alarm at rest."
3. **`measure`** — "inference runs in *X* ms on a BDT 450 chip, using *Y* KB of RAM.
   The model is 22.5 KB."
4. **Drop it onto the mattress** — buzzer fires. "It detected the fall *before* impact,
   with roughly 300 ms of lead time."
5. **`mode rule` then repeat** — "here's the classical threshold detector for
   comparison." Then sit down heavily: the rule detector is more likely to false-alarm.
6. Close on the honest bit: **"within-dataset accuracy is 95%. On a dataset it has
   never seen it drops to 34%. That gap is the finding, and most papers don't report
   it."**

### If the demo misbehaves

- Alarms too often → `k 3` (default is `k 2`)
- Won't fire on drops → `k 1`, then `thr 0.9` if still needed
- **To show the pre-impact claim** → `k 1`. This is the operating point the paper
  reports: 0.958 sensitivity, 539 ms mean lead, 68 % of detections before impact.
  It false-alarms more, so switch back to `k 2` for the walk-around parts.
- Nothing at all → `mode rule` (the rule detector is independent of the model)
- Total failure → `status` shows whether the sensor is alive; a loose I²C wire is the
  usual cause

---

## 4. Talking points

**What's genuinely novel** (say these):

- Only the *cheapest* domain adaptation helps. CORAL and adversarial training both
  made cross-dataset transfer **worse**. The method that fits on the MCU is the one
  that works.
- **Where you put instance normalisation decides whether INT8 quantisation works
  at all** — 30.95 points of macro-F1 lost inside the graph, zero in the pipeline.
  Same maths, completely different outcome. This is a real TinyML finding.
- **Window-level specificity is nearly meaningless** for a device deciding 7,200
  times an hour. 99% specificity is still 72 false alarms per hour.

**If asked "how does it compare to state of the art?"**

> On KFall, scored the same way the benchmark paper scores it, we get 0.985
> sensitivity and 0.991 specificity against the published ConvLSTM's 0.993 and 0.990 —
> at about 1/70th of the size. We trade some lead time for that.

**If asked about weaknesses** — answer straight, it's a strength:

> Falls are simulated by young volunteers, not real falls by older adults. Our SisFall
> mirror is missing most of its elderly subjects. And the lead time is barely longer
> than one inference stride, so debouncing false alarms eats it — that's the next
> thing to fix, by shortening the stride.

---

## 5. Bonus: real false-alarm data (if you have an hour)

Wear it for an hour of ordinary activity, no falls, then `measure`. The alarm count
**is** the real false-alarm rate. That single number would replace the weakest figure
in the paper. Templates in [results/on_device_trials.md](results/on_device_trials.md).
