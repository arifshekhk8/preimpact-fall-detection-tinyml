# Filling Table IV, on the built device

The hardware exists and the AI-free baseline passed
(`docs/fall_detector_ai_free_report_english.pdf`). This is what remains.

Four numbers cannot come from anywhere but the bench: **inference latency**, **peak
RAM (tensor arena high-water mark)**, **end-to-end latency**, and **battery life**.
Everything they depend on — model, firmware, quantisation — is committed and is not
changed by measuring.

---

## 0. One correction to the AI-free report first

Section 10 of that report says the integration should *"apply the frozen channel mean
and standard deviation from training."* **Do not do that.** It was correct when it was
written and is not correct now.

The shipped model expects **per-window instance normalisation** — each 100-sample
window normalised by its own per-channel mean and standard deviation. The reason is in
`results/quantisation_comparison.csv`: keeping instance normalisation inside the Keras
graph collapsed the INT8 model (macro-F1 0.668 → 0.359, 69 % agreement with FP32),
because per-window mean and variance produce tensors whose dynamic range one global
quantisation scale cannot cover. Moving it into float preprocessing took the drop to
**−0.22 points** and agreement to **99.4 %**.

`norm_constants.h` is kept in the repo as a record but is **not compiled in**.
`fall_detector.ino` does the per-window version, and it must stay identical to
`fdlib.preprocess.instance_normalise`.

---

## 1. Flash

Copy next to the sketch (all three are generated — never hand-edit):

```
firmware/esp32/fall_detector.ino
firmware/esp32/model.h        <- from nb06 output, 22.5 KB INT8
```

Arduino IDE → board **ESP32 Dev Module**, install the **TensorFlowLite_ESP32**
library, flash, open the Serial Monitor at **115200**.

Your existing wiring is unchanged. The sketch keeps your pin map (SDA 21, SCL 22,
buzzer GPIO 4, LED GPIO 2) and your register-level MPU6050 init, which is already
verified and matches training exactly: ±16 g → 2048 LSB/g, ±2000 dps → 16.4 LSB/dps,
DLPF on, `0x19 = 19` → 50 Hz.

Expected boot output:

```
MPU6050 READY (+/-16 g, +/-2000 dps, DLPF on, 50 Hz)
model flash    : 23080 bytes (22.5 KB)
arena at init  : ..... of 24576 bytes
input  scale 0.0........ zero_point ...
normalisation  : PER-WINDOW instance norm (not frozen constants)
```

If `AllocateTensors failed` appears, raise `kArenaSize` and reflash.

**Sanity check before trusting any number:** hold the board still. `p_bkg` should
dominate. Rotate it slowly — still no alarm. If it alarms while stationary, stop; the
axis convention or scaling is wrong, and no later measurement will mean anything.

---

## 2. Latency and peak RAM — `measure`

The firmware instruments itself. It prints automatically every 1000 inferences, or on
demand:

```
measure
```

```
inferences        : 1000
invoke mean       : ..... ms      <- Table IV, inference latency   (target < 50)
invoke min/max    : ..... / ..... ms
preprocess mean   : ..... ms      <- window copy + instance norm + quantise
end-to-end mean   : ..... ms      <- Table IV, end-to-end          (target < 100)
arena high water  : ..... bytes   <- Table IV, peak RAM            (target < 120 KB)
model flash       : 23080 bytes
```

1,000 inferences takes about **8 minutes** at the 0.5 s stride. Let it reach at least
that before recording.

Then set `kArenaSize` to the high-water mark plus ~10 %, reflash, confirm it still
allocates. Report the high-water mark, not `kArenaSize`.

This also gives **end-to-end latency directly** — no logic analyser needed, because
preprocessing and invoke are timed separately and summed. If you want the true
sensor-to-buzzer figure including I²C, probe SDA and GPIO 4.

---

## 3. Battery life

Needs the LiPo and TP4056, which the AI-free report correctly deferred. Add them only
after the USB-powered test above is stable.

1. Charge the 1000 mAh cell fully.
2. Run continuous inference, untethered, **buzzer disconnected** — an alarm every few
   seconds would dominate the draw and measure the wrong thing.
3. Log start time, run to shutdown, record elapsed hours.

A USB current meter also gives mean current, which makes the number reproducible
without repeating a full discharge.

---

## 4. The two experiments that are worth more than Table IV

The device can now produce evidence no desktop table can. Both are cheap.

### 4a. Real false-alarm rate — the paper's weakest point, fixed

E3 reports **293 false alarms/hour** at window level, and
`results/debounce_analysis.md` shows that suppressing them costs nearly all the lead
time. Those are *simulated* ADL from public datasets. A real measurement replaces the
weakest number in the paper with a real one.

```
reset
mode both
```

Wear the device at the waist for **one hour of ordinary activity** — walking, stairs,
sitting down hard, lying down, picking things up. Do not simulate falls. Then:

```
measure
```

`cnn alarms` and `rule alarms` over one hour **are the false-alarm rates**, measured,
on a real body. Repeat at a few settings to get a curve:

| Run | `thr` | `k` | CNN alarms/hr | Rule alarms/hr |
|---|---|---|---|---|
| 1 | 0.5 | 3 | | |
| 2 | 0.7 | 3 | | |
| 3 | 0.9 | 2 | | |
| 4 | 0.5 | 4 | | |

Change settings live: `thr 0.7`, `k 4`, then `reset`.

### 4b. On-device CNN versus rule baseline

Your rule detector is the on-device instance of Table I's SMV threshold comparator,
and `mode both` runs both on identical live samples. Drop the device onto a mattress
**20 times** and record which fires:

| Trial | Rule fired | CNN fired | Notes |
|---|---|---|---|

Include backward, forward and lateral drops, plus **near-falls that should not fire**
(sitting down heavily, dropping onto a sofa). A head-to-head on the same physical
motion is far stronger than any desktop comparison, and it is the honest way to show
what the 22.5 KB model buys over a 2-parameter threshold.

---

## 5. Recording the results

Fill `results/hardware_measurements.json`:

```json
{
  "inference_latency_ms": null,
  "arena_high_water_bytes": null,
  "end_to_end_latency_ms": null,
  "battery_life_hours": null,
  "mean_current_ma": null,
  "n_inferences_averaged": 1000,
  "board": "ESP32 DevKit V1",
  "imu": "MPU6050",
  "date": null,
  "mattress_drop_test_fired": null
}
```

Then `python scripts/build_tables.py` fills Table IV's `PENDING-HW` cells. Nothing
else needs to change.

Put the 4a and 4b tables in `results/on_device_trials.md` — they become a results
subsection of their own.
