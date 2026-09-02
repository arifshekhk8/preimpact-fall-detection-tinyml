# nb01 -- preprocessing

```
preprocess signature: 2574e6104c95

--- UMAFall sensor-type inventory (is there a gyroscope?) ---
  UMAFall_Subject_01_ADL_Aplausing_1_2017-04-14_
    (sensor_id, sensor_type) -> rows: {(0, 0): 2971, (1, 0): 300, (1, 1): 300, (1, 2): 300, (2, 0): 299, (2, 1): 299, (2, 2): 299, (3, 0): 299, (3, 1): 299, (3, 2): 299, (4, 0): 299, (4, 1): 299, (4, 2): 299}
  UMAFall_Subject_01_ADL_Aplausing_2_2017-04-14_
    (sensor_id, sensor_type) -> rows: {(0, 0): 2971, (1, 0): 300, (1, 1): 300, (1, 2): 300, (2, 0): 300, (2, 1): 300, (2, 2): 300, (3, 0): 299, (3, 1): 299, (3, 2): 299, (4, 0): 300, (4, 1): 300, (4, 2): 300}
  UMAFall_Subject_01_ADL_Aplausing_3_2017-04-14_
    (sensor_id, sensor_type) -> rows: {(0, 0): 2972, (1, 0): 296, (1, 1): 296, (1, 2): 296, (2, 0): 300, (2, 1): 300, (2, 2): 300, (3, 0): 300, (3, 1): 300, (3, 2): 300, (4, 0): 299, (4, 1): 299, (4, 2): 299}
  sensor_type 0 = accelerometer, 1 = gyroscope, 2 = magnetometer
target 50 Hz, window 100 samples, stride 25

--- loading sisfall from /kaggle/input/datasets/adityavvvn/sisfall
  trials 3691  falls 1798  ADL 1893  subjects 25
  subjects: ['sisfall:SA01', 'sisfall:SA02', 'sisfall:SA03', 'sisfall:SA04', 'sisfall:SA05', 'sisfall:SA06'] ...
  mean duration 17.0 s

--- loading kfall from /kaggle/input/datasets/usmanabbasi2002/kfall-dataset
  trials 5075  falls 2346  ADL 2729  subjects 32
  subjects: ['kfall:SA06', 'kfall:SA07', 'kfall:SA08', 'kfall:SA09', 'kfall:SA10', 'kfall:SA11'] ...
  mean duration 7.9 s

--- loading fallalld from /kaggle/input/datasets/sankalpsinghvishen/derived-fallalld-dataset
  trials 1798  falls 466  ADL 1332  subjects 14
  subjects: ['fallalld:S01', 'fallalld:S02', 'fallalld:S03', 'fallalld:S04', 'fallalld:S05', 'fallalld:S07'] ...
  mean duration 20.0 s

--- loading umafall from /kaggle/input/datasets/thanushanth/umafall
  trials 577  falls 202  ADL 375  subjects 18
  subjects: ['umafall:S01', 'umafall:S02', 'umafall:S03', 'umafall:S04', 'umafall:S05', 'umafall:S06'] ...
  mean duration 12.8 s

==============================================================================
SECTION 4.2 SANITY GATES
==============================================================================

sisfall:
  n_trials                   3691
  resting_mag_g              1.011
  resting_mag_ok             True
  gravity_axis               1
  gravity_axis_votes         [6, 187, 7]
  gravity_sign               -0.964428573846817
  gyro_abs_mean_dps          19.138
  gyro_p99_dps               129.98
  peak_fall_g                4.759
  peak_adl_g                 1.882
  fall_peak_exceeds_adl      True
  status                     OK

kfall:
  n_trials                   5075
  resting_mag_g              1.0013
  resting_mag_ok             True
  gravity_axis               1
  gravity_axis_votes         [1, 194, 5]
  gravity_sign               -0.9629627764225006
  gyro_abs_mean_dps          13.134
  gyro_p99_dps               128.21
  peak_fall_g                4.371
  peak_adl_g                 1.877
  fall_peak_exceeds_adl      True
  status                     OK

fallalld:
  n_trials                   1798
  resting_mag_g              0.9881
  resting_mag_ok             True
  gravity_axis               0
  gravity_axis_votes         [189, 7, 4]
  gravity_sign               0.9284465909004211
  gyro_abs_mean_dps          12.236
  gyro_p99_dps               89.63
  peak_fall_g                4.159
  peak_adl_g                 1.695
  fall_peak_exceeds_adl      True
  status                     OK

umafall:
  n_trials                   577
  resting_mag_g              0.9883
  resting_mag_ok             True
  gravity_axis               1
  gravity_axis_votes         [2, 198, 0]
  gravity_sign               0.8753665387630463
  gyro_abs_mean_dps          0.0
  gyro_p99_dps               0.0
  peak_fall_g                3.666
  peak_adl_g                 1.823
  fall_peak_exceeds_adl      True
  status                     OK

==============================================================================
AXIS CONVENTION -- the cross-dataset trap
==============================================================================
  before rotation -- axis: {'sisfall': 1, 'kfall': 1, 'fallalld': 0, 'umafall': 1}
  before rotation -- sign: {'sisfall': -1, 'kfall': -1, 'fallalld': 1, 'umafall': 1}
  canonical convention: gravity on axis 1, sign -1
  Two labs mounting the same sensor differently produce data that cannot transfer,
  and the failure is indistinguishable from a genuine domain shift. Reconciling
  this is the one correction that must happen before any C1 number means anything.

  sisfall: axis 1 sign -1 -> identity (already canonical)

  kfall: axis 1 sign -1 -> identity (already canonical)

  fallalld: axis 0 sign +1 -> ROTATING
    R = [[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]]

  umafall: axis 1 sign +1 -> ROTATING
    R = [[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]]

  --- verification after rotation ---
   sisfall    axis 1 sign -0.964  OK
   kfall      axis 1 sign -0.963  OK
   fallalld   axis 1 sign -0.928  OK
   umafall    axis 1 sign -0.875  OK

  all datasets now share one axis convention

  --- gyroscope availability per dataset ---
   sisfall    non-zero gyroscope samples: 100.0%
   kfall      non-zero gyroscope samples: 100.0%
   fallalld   non-zero gyroscope samples: 100.0%
   umafall    non-zero gyroscope samples:   0.0%
     !! umafall HAS NO USABLE GYROSCOPE. A model trained on six channels
        and tested here would fail for a reason that is not domain shift.
        Evaluate this dataset with the accelerometer-only model (E5).

==============================================================================
WINDOWING
==============================================================================

  task=postfall: X (256037, 100, 6)  classes {'bkg': 236895, 'alert': 0, 'fall': 19142}
    fallalld   windows   66526  {'bkg': 64670, 'alert': 0, 'fall': 1856}  imbalance 1:35
    kfall      windows   62295  {'bkg': 52926, 'alert': 0, 'fall': 9369}  imbalance 1:6
    sisfall    windows  114466  {'bkg': 107274, 'alert': 0, 'fall': 7192}  imbalance 1:15
    umafall    windows   12750  {'bkg': 12025, 'alert': 0, 'fall': 725}  imbalance 1:17

  task=preimpact: X (256037, 100, 6)  classes {'bkg': 228546, 'alert': 8110, 'fall': 19381}
    fallalld   windows   66526  {'bkg': 63707, 'alert': 921, 'fall': 1898}  imbalance 1:23
    kfall      windows   62295  {'bkg': 49509, 'alert': 3321, 'fall': 9465}  imbalance 1:4
    sisfall    windows  114466  {'bkg': 103593, 'alert': 3591, 'fall': 7282}  imbalance 1:10
    umafall    windows   12750  {'bkg': 11737, 'alert': 277, 'fall': 736}  imbalance 1:12

==============================================================================
NORMALISATION -- fitted on TRAINING SUBJECTS ONLY
==============================================================================
  mean [0.00259, -0.69132, -0.02014, -0.28654, 2.78818, 0.09101]
  std  [0.40943, 0.56151, 0.46309, 31.7661, 36.64721, 22.56092]
  written to norm_constants.json and norm_constants.h (identical numbers,
  so the firmware cannot drift from training)

wrote fig_sanity_traces.png
wrote fig_alert_intervals.png
wrote windows_postfall.npz  (256037, 100, 6)
wrote windows_preimpact.npz  (256037, 100, 6)
```
