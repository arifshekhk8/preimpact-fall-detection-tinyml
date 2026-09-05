# Final deployment model

- Task: three-class pre-impact (bkg / alert / fall)
- Trained on: sisfall, kfall, fallalld (UMAFall excluded -- no gyroscope)
- Parameters: 7,947
- INT8 size: 22.54 KB
- Estimated tensor arena: 8.0 KB
- FP32 macro-F1 0.7109, INT8 0.7138 (drop -0.288 pp, agreement 0.9944)
- Normalisation: per-window instance norm, in the PIPELINE (firmware does it in float)

## Recommended firmware setting

`gThreshold = 0.7`, `gAgreeK = 4`

Trial-level on held-out subjects: sensitivity 0.9128, specificity 0.9692, mean lead 722.0 ms.

## Operating points

|   threshold |   k |   sens_trial |   spec_trial |   mean_lead_ms |   std_lead_ms |   preimpact_frac |   fall_trials |   adl_trials |   youden |
|------------:|----:|-------------:|-------------:|---------------:|--------------:|-----------------:|--------------:|-------------:|---------:|
|        0.7  |   4 |       0.9128 |       0.9692 |          722   |         837.1 |           0.1844 |           814 |         1136 |   0.882  |
|        0.95 |   2 |       0.8857 |       0.9754 |          302.4 |         258.3 |           0.1623 |           814 |         1136 |   0.8611 |
|        0.9  |   3 |       0.8759 |       0.9815 |          362.4 |         528.5 |           0.1178 |           814 |         1136 |   0.8574 |
|        0.9  |   2 |       0.93   |       0.9261 |          539.1 |         608.5 |           0.3659 |           814 |         1136 |   0.8561 |
|        0.5  |   4 |       0.9496 |       0.8944 |         1063   |        1176   |           0.326  |           814 |         1136 |   0.844  |
|        0.7  |   3 |       0.9545 |       0.8847 |          815.9 |         906.5 |           0.4067 |           814 |         1136 |   0.8392 |
|        0.95 |   3 |       0.8071 |       0.9965 |          172.6 |         179.8 |           0.0289 |           814 |         1136 |   0.8036 |
|        0.95 |   1 |       0.9582 |       0.8363 |          539.5 |         509.1 |           0.6821 |           814 |         1136 |   0.7945 |
|        0.9  |   4 |       0.7801 |       0.9938 |          276.9 |         336   |           0.0205 |           814 |         1136 |   0.7739 |
|        0.3  |   4 |       0.9767 |       0.7694 |         1451.6 |        1581.6 |           0.4717 |           814 |         1136 |   0.7461 |
|        0.5  |   3 |       0.9668 |       0.7438 |         1137.6 |        1266.6 |           0.587  |           814 |         1136 |   0.7106 |
|        0.7  |   2 |       0.9767 |       0.7324 |         1015.9 |        1087.8 |           0.7333 |           814 |         1136 |   0.7091 |
