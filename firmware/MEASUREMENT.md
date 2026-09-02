# Filling Table IV's hardware cells

Three numbers in Table IV cannot be produced without the physical board, and are
committed as `PENDING-HW` rather than estimated: **inference latency**, **peak RAM
(tensor arena high-water mark)** and **battery life**. Everything they depend on —
the model, the firmware, the normalisation constants — is committed and is not
changed by taking the measurement.

This document is the procedure. Following it end to end takes about two hours plus
the battery run.

## Bill of materials

| Component | Qty | BDT | Note |
|---|---:|---:|---|
| ESP32 DevKit V1 (30 or 38 pin) | 1 | 410–500 | 240 MHz dual core, 320 KB usable RAM, 4 MB flash. Prefer 38-pin for pin access. |
| MPU6050 / GY-521 IMU | 2 | 199–290 | Buy two: units arrive faulty often enough to matter, and a working pair gives the waist-vs-wrist placement check for free. |
| TP4056 charging module **with protection** | 1 | 100 | Buy the version with over-discharge protection. |
| 3.7 V LiPo cell, 1000 mAh | 1 | 250 | The battery-life measurement depends on a known capacity. |
| Active buzzer + LED + resistors | 1 set | 60 | Active buzzer needs only a GPIO high, no PWM tone generation. |
| Perfboard, jumper wires, headers | 1 set | 190 | Breadboard first, then solder. |

Subtotal ≈ **1,450 BDT**.

## Wiring

| From | To | Note |
|---|---|---|
| MPU6050 VCC | ESP32 3V3 | **Not 5 V** — the ESP32's I²C lines are 3.3 V |
| MPU6050 GND | ESP32 GND | |
| MPU6050 SDA | GPIO 21 | I²C data (default) |
| MPU6050 SCL | GPIO 22 | I²C clock (default) |
| MPU6050 AD0 | GND or 3V3 | Selects address 0x68 or 0x69; needed for two IMUs on one bus |
| Buzzer + | GPIO 4 | Through a transistor if it draws more than 12 mA |
| Buzzer − | ESP32 GND | |
| LiPo | TP4056 B+ / B− | |
| TP4056 OUT+ | ESP32 VIN (or 5V pin) | Verify your board's regulator accepts 3.7 V before relying on it |

## Build and flash

1. Arduino IDE, board "ESP32 Dev Module". Install the **TensorFlowLite_ESP32** library
   (or `esp-tflite-micro` if you prefer ESP-IDF — the Espressif port gives more control;
   switching later does not touch the model).
2. Copy the generated artifacts next to the sketch:
   ```
   firmware/esp32/fall_detector.ino
   firmware/esp32/model.h            <- from nb06 output
   firmware/esp32/norm_constants.h   <- from nb01/nb06 output
   ```
   Both headers are generated. **Never edit them by hand** — they are the only thing
   keeping device preprocessing identical to training.
3. Flash, open the serial monitor at 115200.

On boot the firmware prints the input/output quantisation parameters and the arena
usage at init. If `AllocateTensors failed` appears, raise `kArenaSize` and reflash.

## Measurement 1 — inference latency

The firmware already instruments this: `esp_timer_get_time()` brackets `Invoke()`,
and every 1000 inferences it prints

```
MEASURE inferences=1000 mean_latency_us=... mean_latency_ms=... arena_used=... arena_size=...
```

Let it run until at least **1000 inferences** have accumulated (about 8 minutes at the
0.5 s stride), then record `mean_latency_ms`.

**Target: < 50 ms.** Report the measured value whatever it is.

## Measurement 2 — peak RAM

Read `arena_used` from the same line. That is TFLite Micro's high-water mark, which is
the number Table IV wants — not `kArenaSize`, which is only what was reserved.

Once you have it, reduce `kArenaSize` to the high-water mark plus roughly 10 % and
reflash to confirm it still allocates. Report the high-water mark.

**Target: < 120 KB.**

## Measurement 3 — end-to-end latency

Sensor read to buzzer edge, which is what a user actually experiences.

- With a logic analyser: probe SDA and GPIO 4, trigger on the I²C burst, measure to the
  buzzer's rising edge.
- Without one: toggle a spare GPIO high at the top of the window-copy block and low
  immediately after `digitalWrite(kPinBuzzer, HIGH)`, and measure that pulse on a
  second scope channel or a second ESP32 running a pulse timer.

**Target: < 100 ms.**

## Measurement 4 — battery life

1. Charge the 1000 mAh cell fully.
2. Run continuous inference, untethered, buzzer disabled (an alarm every few seconds
   would dominate the power draw and measure the wrong thing).
3. Log the start time; run to shutdown; record the elapsed hours.

Report as measured. If a USB current meter is available, also record the mean current —
it makes the number reproducible without repeating the full discharge.

## Measurement 5 — a real fall

Not a Table IV row, but the week-7 deliverable and the most convincing single item in
the paper: drop the device onto a mattress and confirm the buzzer fires. Record it.

## Recording the results

Put the numbers in `results/hardware_measurements.json`:

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
  "date": null
}
```

Then run `python scripts/build_tables.py`, which fills Table IV's `PENDING-HW` cells
from that file. Nothing else needs to change.
