# Alarm debouncing: what the device actually raises

Table III reports **293 false alarms per hour** at threshold 0.5 and 102 at 0.9.
Those numbers are correct, and they are the point.

The model decides once every 0.5 s, so 7,200 times per hour. Even 99 % window-level
specificity leaves 72 false alarms per hour; one per hour needs specificity of
0.99986. **Window-level specificity is close to meaningless as a deployment metric**,
and a paper reporting only specificity has not shown its device is wearable.

The firmware already carries two mechanisms that cost nothing on the MCU: requiring
*k* consecutive windows above threshold before firing, and a cooldown after firing.
Requiring k windows necessarily delays the alarm by (k-1) strides, and that delay
comes straight out of the lead time — so the trade-off is measured here, not assumed.

|   threshold |   k_consecutive |   cooldown_s |   false_alarms_per_hour |   adl_hours |   detection_rate |   preimpact_rate |   mean_lead_ms |   median_lead_ms |   n_fall_trials |
|------------:|----------------:|-------------:|------------------------:|------------:|-----------------:|-----------------:|---------------:|-----------------:|----------------:|
|        0.5  |               1 |            5 |                 216.106 |        6.88 |           1      |           0.9177 |          663.5 |              560 |            2346 |
|        0.5  |               2 |            5 |                  48.137 |        6.88 |           0.997  |           0.4437 |          379.8 |              280 |            2346 |
|        0.5  |               3 |            5 |                   8.144 |        6.88 |           0.9851 |           0.1049 |          318.5 |              240 |            2346 |
|        0.5  |               4 |            5 |                   1.745 |        6.88 |           0.9604 |           0.0149 |          384.6 |              260 |            2346 |
|        0.7  |               1 |            5 |                 167.533 |        6.88 |           1      |           0.8917 |          588.7 |              500 |            2346 |
|        0.7  |               2 |            5 |                  30.394 |        6.88 |           0.994  |           0.376  |          341.4 |              240 |            2346 |
|        0.7  |               3 |            5 |                   4.945 |        6.88 |           0.9783 |           0.0669 |          341.8 |              260 |            2346 |
|        0.7  |               4 |            5 |                   1.018 |        6.88 |           0.9459 |           0.0107 |          401.6 |              280 |            2346 |
|        0.9  |               1 |            5 |                  84.93  |        6.88 |           0.9979 |           0.8018 |          448.5 |              360 |            2346 |
|        0.9  |               2 |            5 |                  10.325 |        6.88 |           0.9838 |           0.2195 |          263.5 |              180 |            2346 |
|        0.9  |               3 |            5 |                   1.891 |        6.88 |           0.9629 |           0.0234 |          303.6 |              200 |            2346 |
|        0.9  |               4 |            5 |                   0.436 |        6.88 |           0.916  |           0.0021 |          272   |              260 |            2346 |
|        0.95 |               1 |            5 |                  55.844 |        6.88 |           0.9962 |           0.7383 |          393.2 |              340 |            2346 |
|        0.95 |               2 |            5 |                   5.09  |        6.88 |           0.9787 |           0.1573 |          229.2 |              160 |            2346 |
|        0.95 |               3 |            5 |                   1.163 |        6.88 |           0.9544 |           0.0119 |          319.3 |              180 |            2346 |
|        0.95 |               4 |            5 |                   0.436 |        6.88 |           0.8858 |           0.0013 |          353.3 |              320 |            2346 |
|        0.99 |               1 |            5 |                  21.669 |        6.88 |           0.9893 |           0.5482 |          308.4 |              260 |            2346 |
|        0.99 |               2 |            5 |                   0.873 |        6.88 |           0.9557 |           0.0618 |          152.8 |              120 |            2346 |
|        0.99 |               3 |            5 |                   0.291 |        6.88 |           0.9079 |           0.0013 |          146.7 |              200 |            2346 |
|        0.99 |               4 |            5 |                   0.145 |        6.88 |           0.7805 |           0      |          nan   |              nan |            2346 |


## Operating points at or under 1 false alarm per hour

|   threshold |   k_consecutive |   cooldown_s |   false_alarms_per_hour |   adl_hours |   detection_rate |   preimpact_rate |   mean_lead_ms |   median_lead_ms |   n_fall_trials |
|------------:|----------------:|-------------:|------------------------:|------------:|-----------------:|-----------------:|---------------:|-----------------:|----------------:|
|        0.95 |               4 |            5 |                   0.436 |        6.88 |           0.8858 |           0.0013 |          353.3 |              320 |            2346 |
|        0.9  |               4 |            5 |                   0.436 |        6.88 |           0.916  |           0.0021 |          272   |              260 |            2346 |
|        0.99 |               2 |            5 |                   0.873 |        6.88 |           0.9557 |           0.0618 |          152.8 |              120 |            2346 |
|        0.99 |               3 |            5 |                   0.291 |        6.88 |           0.9079 |           0.0013 |          146.7 |              200 |            2346 |
|        0.99 |               4 |            5 |                   0.145 |        6.88 |           0.7805 |           0      |          nan   |              nan |            2346 |


Best lead time among them: **353 ms** at threshold 0.95, k=4, detecting 0.1% of falls before impact at 0.44 false alarms per hour.
