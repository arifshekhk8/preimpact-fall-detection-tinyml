# Table II -- cross-dataset generalisation

Train on two datasets, test on a third. No fine-tuning, no target labels.

|                                            |   none |   instance_norm |   coral |   dann |
|:-------------------------------------------|-------:|----------------:|--------:|-------:|
| ('A', 'kfall+sisfall', 'fallalld')         | 0.3439 |          0.3877 |  0.1793 | 0.3072 |
| ('B', 'fallalld+sisfall', 'kfall')         | 0.8348 |          0.8731 |  0.6443 | 0.8344 |
| ('C', 'fallalld+kfall', 'sisfall')         | 0.7987 |          0.8299 |  0.6174 | 0.7281 |
| ('D', 'fallalld+kfall+sisfall', 'umafall') | 0.4745 |          0.423  |  0.0054 | 0.2993 |

**Fold D (UMAFall)** is the never-trained-on stress test, reported once. It is
accelerometer-only and at a different body position (pocket, not waist), because
UMAFall's waist sensor samples at 20 Hz and its 200 Hz pocket smartphone logs no
gyroscope. Fold D therefore measures a combined domain and placement shift and is
not directly comparable with folds A-C.
