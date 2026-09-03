# Table II — cross-dataset generalisation

Train on two datasets, test on a third. No fine-tuning, no target labels at any point. F1 on the unseen dataset.

|                                            |   none |   instance_norm |   coral |   dann |   drop_vs_table_I |
|:-------------------------------------------|-------:|----------------:|--------:|-------:|------------------:|
| ('A', 'kfall+sisfall', 'fallalld')         | 0.3439 |          0.3877 |  0.1793 | 0.3072 |            0.5628 |
| ('B', 'fallalld+sisfall', 'kfall')         | 0.8348 |          0.8731 |  0.6443 | 0.8344 |            0.0719 |
| ('C', 'fallalld+kfall', 'sisfall')         | 0.7987 |          0.8299 |  0.6174 | 0.7281 |            0.108  |
| ('D', 'fallalld+kfall+sisfall', 'umafall') | 0.4745 |          0.423  |  0.0054 | 0.2993 |            0.4322 |

`drop_vs_table_I` is measured against the proposed model's within-dataset F1 of 0.9067 (Table I).

**Fold D (UMAFall)** is the never-trained-on, never-tuned stress test, reported once. It is accelerometer-only and at a different body position (pocket, not waist), so it measures a combined domain and placement shift and is not directly comparable with folds A–C.
