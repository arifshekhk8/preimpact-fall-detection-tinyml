# Table III — pre-impact performance

Lead time is the interval between the model's **first** alarm and the labelled impact, one value per fall trial, pooled across leave-one-subject-out folds. Positive means the alarm preceded impact.

| label_source    |   threshold |   mean_lead_ms |   std_ms |   median_ms |   p25_ms |   p75_ms |   detection_rate |   preimpact_rate |   sensitivity |   specificity |   false_alarms_per_hour |   n_fall_trials |
|:----------------|------------:|---------------:|---------:|------------:|---------:|---------:|-----------------:|-----------------:|--------------:|--------------:|------------------------:|----------------:|
| KFall (primary) |         0.5 |          663.5 |    478.9 |         560 |      340 |      860 |           1      |           0.9177 |        0.9292 |        0.9593 |                 292.892 |            2346 |
| KFall (primary) |         0.7 |          588.7 |    437.7 |         500 |      280 |      760 |           1      |           0.8917 |        0.911  |        0.9701 |                 215.088 |            2346 |
| KFall (primary) |         0.9 |          448.5 |    354.7 |         360 |      220 |      600 |           0.9979 |           0.8018 |        0.8652 |        0.9858 |                 102.236 |            2346 |

**One label source.** KFall's onset and impact frames derive from synchronised video. The SisFall Enhanced annotations that would have corroborated them are not recoverable from the available mirror — nb00 measured a 22.5 % ceiling against a null control — so this result rests on KFall alone.
