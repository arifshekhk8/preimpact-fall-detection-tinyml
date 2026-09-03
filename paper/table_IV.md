# Table IV — on-device deployment

| Metric                       |   FP32 (desktop) | INT8 (desktop)   | INT8 (ESP32, measured)   |
|:-----------------------------|-----------------:|:-----------------|:-------------------------|
| Model size (KB)              |         163.3    | 22.54            | PENDING-HW               |
| Peak RAM / tensor arena (KB) |         nan      | ~8.0 (estimate)  | PENDING-HW               |
| Inference latency (ms)       |         nan      | nan              | PENDING-HW               |
| Macro-F1 on held-out set     |           0.7014 | 0.7035           | same model               |
| Sensitivity                  |           0.8168 | 0.8121           | same model               |
| Battery life (hours)         |         nan      | nan              | PENDING-HW               |

4 cell(s) still marked `PENDING-HW` require the physical ESP32 and MPU6050. The procedure is in `firmware/MEASUREMENT.md`; the model, firmware and normalisation constants they depend on are committed and are not changed by the measurement.
