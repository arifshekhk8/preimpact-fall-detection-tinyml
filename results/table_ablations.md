# E5 -- ablations

5-fold subject-grouped CV, proposed model.

| arm       | variant    |   params |   input_len |   n_channels |   macro_f1_3class |   bin_sensitivity |   bin_specificity |   bin_f1 |
|:----------|:-----------|---------:|------------:|-------------:|------------------:|------------------:|------------------:|---------:|
| rate      | 25Hz_2.0s  |     7947 |          50 |            6 |            0.9379 |            0.8724 |            0.9931 |   0.8836 |
| window    | 50Hz_1.0s  |     7947 |          50 |            6 |            0.899  |            0.834  |            0.9926 |   0.8041 |
| window    | 50Hz_1.5s  |     7947 |          75 |            6 |            0.9323 |            0.8722 |            0.9937 |   0.8708 |
| window    | 50Hz_2.0s  |     7947 |         100 |            6 |            0.9484 |            0.9131 |            0.9927 |   0.9033 |
| rate      | 100Hz_2.0s |     7947 |         200 |            6 |            0.9517 |            0.9297 |            0.9923 |   0.9095 |
| channels  | accel+gyro |     7947 |         100 |            6 |            0.9496 |            0.8969 |            0.9943 |   0.9054 |
| channels  | accel_only |     7443 |         100 |            3 |            0.9357 |            0.9143 |            0.9889 |   0.8798 |
| placement | waist      |     7947 |         100 |            6 |            0.7685 |            0.5977 |            0.9839 |   0.551  |
| placement | neck       |     7947 |         100 |            6 |            0.8152 |            0.7189 |            0.9824 |   0.6441 |
| placement | wrist      |     7947 |         100 |            6 |            0.7317 |            0.4425 |            0.9903 |   0.4756 |

Placement uses FallAllD rather than UMAFall: UMAFall's multi-position sensors all
sample at 20 Hz, below the 50 Hz working rate, and raising them would fabricate
signal -- the same criterion that excluded UP-Fall.
