# Trial-level evaluation (comparable with published KFall benchmarks)

Yu et al. (2021) score KFall per **trial** — FN 20/444, FP 84/507 — so their
specificity is the fraction of ADL trials raising no alarm, not a window-level
rate. Table III's window-level numbers are not comparable with theirs. This is
the same predictions re-scored on their definition.

|   threshold |   k |   sensitivity_trial |   specificity_trial |   FN |   fall_trials |   FP |   adl_trials |   mean_lead_ms |   std_lead_ms |   preimpact_frac_of_detected |
|------------:|----:|--------------------:|--------------------:|-----:|--------------:|-----:|-------------:|---------------:|--------------:|-----------------------------:|
|        0.5  |   1 |              1      |              0.8029 |    0 |          2346 |  538 |         2729 |          663.5 |         478.9 |                       0.9177 |
|        0.5  |   2 |              0.997  |              0.9553 |    7 |          2346 |  122 |         2729 |          379.8 |         349.3 |                       0.4451 |
|        0.5  |   3 |              0.9851 |              0.9905 |   35 |          2346 |   26 |         2729 |          318.5 |         328.1 |                       0.1064 |
|        0.7  |   1 |              1      |              0.8465 |    0 |          2346 |  419 |         2729 |          588.7 |         437.7 |                       0.8917 |
|        0.7  |   2 |              0.994  |              0.9703 |   14 |          2346 |   81 |         2729 |          341.4 |         339.2 |                       0.3782 |
|        0.7  |   3 |              0.9783 |              0.9949 |   51 |          2346 |   14 |         2729 |          341.8 |         373.4 |                       0.0684 |
|        0.9  |   1 |              0.9979 |              0.9179 |    5 |          2346 |  224 |         2729 |          448.5 |         354.7 |                       0.8035 |
|        0.9  |   2 |              0.9838 |              0.9883 |   38 |          2346 |   32 |         2729 |          263.5 |         280.3 |                       0.2231 |
|        0.9  |   3 |              0.9629 |              0.9974 |   87 |          2346 |    7 |         2729 |          303.6 |         392.6 |                       0.0243 |
|        0.95 |   1 |              0.9962 |              0.9465 |    9 |          2346 |  146 |         2729 |          393.2 |         306   |                       0.7411 |
|        0.95 |   2 |              0.9787 |              0.9941 |   50 |          2346 |   16 |         2729 |          229.2 |         272.8 |                       0.1607 |
|        0.95 |   3 |              0.9544 |              0.9982 |  107 |          2346 |    5 |         2729 |          319.3 |         486.7 |                       0.0125 |
|        0.99 |   1 |              0.9893 |              0.9784 |   25 |          2346 |   59 |         2729 |          308.4 |         249.2 |                       0.5541 |
|        0.99 |   2 |              0.9557 |              0.9985 |  104 |          2346 |    4 |         2729 |          152.8 |         150.6 |                       0.0647 |
|        0.99 |   3 |              0.9079 |              0.9993 |  216 |          2346 |    2 |         2729 |          146.7 |          89.9 |                       0.0014 |

## Published comparison (same dataset)

| Method | Sens | Spec | Lead time |
|---|---:|---:|---|
| Threshold (Yu 2021) | 0.9550 | 0.8343 | 333 ± 160 ms |
| SVM (Yu 2021) | 0.9977 | 0.9487 | 385 ± 159 ms |
| ConvLSTM (Yu 2021) | 0.9932 | 0.9901 | 403 ± 163 ms |
| PreFallKD (2023) | — | — | 551.3 ms |
| TinyFallNet (2023) | 0.8667 | 0.9797 | 477.7 ± 5.8 ms |

Yu et al. evaluate on a held-out split; the rows above are pooled 32-fold
leave-one-subject-out, which is the stricter protocol.
