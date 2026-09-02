# nb00 -- data reality probe (round 3, final)

```

==============================================================================
Q2. UMAFall -- fall trials, sensor positions, and measured rates
==============================================================================
  csv files: 746
  trial kinds (case-insensitive): {'ADL': 538, 'FALL': 208}
  FALL trials: 208   examples: ['UMAFall_Subject_02_Fall_backwardFall_1_2016-06-13_20-51-32.csv', 'UMAFall_Subject_02_Fall_backwardFall_2_2016-06-13_20-52-16.csv', 'UMAFall_Subject_02_Fall_backwardFall_3_2016-06-13_20-52-55.csv']
  fall activity names: ['backwardFall', 'forwardFall', 'lateralFall']

  measured sampling rate per sensor position:
                                      file  sid         pos  n_acc  span_s    hz
UMAFall_Subject_01_ADL_Aplausing_1_2017-04    0 RIGHTPOCKET   2971   14.90 199.3
UMAFall_Subject_01_ADL_Aplausing_1_2017-04    1       CHEST    300   14.86  20.1
UMAFall_Subject_01_ADL_Aplausing_1_2017-04    2       WAIST    299   14.85  20.1
UMAFall_Subject_01_ADL_Aplausing_1_2017-04    3       WRIST    299   14.85  20.1
UMAFall_Subject_01_ADL_Aplausing_1_2017-04    4       ANKLE    299   14.85  20.1
UMAFall_Subject_01_ADL_Aplausing_2_2017-04    0 RIGHTPOCKET   2971   14.87 199.8
UMAFall_Subject_01_ADL_Aplausing_2_2017-04    1       CHEST    300   14.84  20.1
UMAFall_Subject_01_ADL_Aplausing_2_2017-04    2       WAIST    300   14.84  20.1
UMAFall_Subject_01_ADL_Aplausing_2_2017-04    3       WRIST    299   14.80  20.1
UMAFall_Subject_01_ADL_Aplausing_2_2017-04    4       ANKLE    300   14.85  20.1
UMAFall_Subject_01_ADL_Aplausing_3_2017-04    0 RIGHTPOCKET   2972   14.90 199.4
UMAFall_Subject_01_ADL_Aplausing_3_2017-04    1       CHEST    296   14.74  20.0
UMAFall_Subject_01_ADL_Aplausing_3_2017-04    2       WAIST    300   14.86  20.1
UMAFall_Subject_01_ADL_Aplausing_3_2017-04    3       WRIST    300   14.84  20.1
UMAFall_Subject_01_ADL_Aplausing_3_2017-04    4       ANKLE    299   14.84  20.1
UMAFall_Subject_01_ADL_HandsUp_1_2017-04-1    0 RIGHTPOCKET   2970   14.89 199.4
UMAFall_Subject_01_ADL_HandsUp_1_2017-04-1    1       CHEST    301   14.86  20.2
UMAFall_Subject_01_ADL_HandsUp_1_2017-04-1    2       WAIST    300   14.97  20.0
UMAFall_Subject_01_ADL_HandsUp_1_2017-04-1    3       WRIST    301   14.87  20.2
UMAFall_Subject_01_ADL_HandsUp_1_2017-04-1    4       ANKLE    301   14.87  20.2
UMAFall_Subject_01_ADL_HandsUp_2_2017-04-1    0 RIGHTPOCKET   2970   14.63 202.9
UMAFall_Subject_01_ADL_HandsUp_2_2017-04-1    1       CHEST    300   14.61  20.5
UMAFall_Subject_01_ADL_HandsUp_2_2017-04-1    2       WAIST    301   14.96  20.0
UMAFall_Subject_01_ADL_HandsUp_2_2017-04-1    3       WRIST    300   14.62  20.5
UMAFall_Subject_01_ADL_HandsUp_2_2017-04-1    4       ANKLE    300   14.62  20.5
UMAFall_Subject_01_ADL_HandsUp_3_2017-04-1    0 RIGHTPOCKET   2971   14.92 199.0
UMAFall_Subject_01_ADL_HandsUp_3_2017-04-1    1       CHEST    301   14.90  20.1
UMAFall_Subject_01_ADL_HandsUp_3_2017-04-1    2       WAIST    300   14.90  20.1
UMAFall_Subject_01_ADL_HandsUp_3_2017-04-1    3       WRIST    300   14.88  20.1
UMAFall_Subject_01_ADL_HandsUp_3_2017-04-1    4       ANKLE    299   14.89  20.0
UMAFall_Subject_02_Fall_backwardFall_1_201    0 RIGHTPOCKET   2977   14.85 200.4
UMAFall_Subject_02_Fall_backwardFall_1_201    1       CHEST    300   14.98  20.0
UMAFall_Subject_02_Fall_backwardFall_1_201    2       WAIST    300   14.84  20.2
UMAFall_Subject_02_Fall_backwardFall_1_201    3       WRIST    299   14.81  20.1
UMAFall_Subject_02_Fall_backwardFall_1_201    4       ANKLE    298   14.75  20.1
UMAFall_Subject_02_Fall_backwardFall_2_201    0 RIGHTPOCKET   2978   14.91 199.7
UMAFall_Subject_02_Fall_backwardFall_2_201    1       CHEST    299   14.86  20.1
UMAFall_Subject_02_Fall_backwardFall_2_201    2       WAIST    299   14.85  20.1
UMAFall_Subject_02_Fall_backwardFall_2_201    3       WRIST    300   14.88  20.1
UMAFall_Subject_02_Fall_backwardFall_2_201    4       ANKLE    299   14.84  20.1
UMAFall_Subject_02_Fall_backwardFall_3_201    0 RIGHTPOCKET   2977   14.64 203.3
UMAFall_Subject_02_Fall_backwardFall_3_201    1       CHEST    300   14.59  20.5
UMAFall_Subject_02_Fall_backwardFall_3_201    2       WAIST    300   14.63  20.4
UMAFall_Subject_02_Fall_backwardFall_3_201    3       WRIST    300   14.62  20.5
UMAFall_Subject_02_Fall_backwardFall_3_201    4       ANKLE    299   14.60  20.4

==============================================================================
Q3. FallAllD -- which ActivityIDs are falls
==============================================================================
  activity_info type: dict, 135 entries
      1  Start clap hands
      2  Clap hands
      3  Stop clap hands
      4  Clap hands 1
      5  Start wave hands
      6  wave hands
      7  Stop wave hands
      8  Raising hand up
      9  Moving hand down
     10  Move hand up -> down
     11  Hand shaking
     12  Beating a table
     13  Sitting down
     14  Standing up
     15  Fail to stand up
     16  Lying down
     17  Turning while lying
     18  Rising up
     19  Start walking
     20  Walking slowly
     21  Stop walking
     22  Walking quickly
     23  Stumbling
     24  Jogging slowly
     25  Jogging quickly
     26  Jumping slightly
     27  Jumping strongly
     28  Bending down
     29  Start going upstairs
     30  Going upstairs
     31  Stop going upstairs
     32  Start going downstairs
     33  Going downstairs
     34  Stop going downstairs
     35  Upstairs quickly
     36  Downstairs quickly
     37  Start ascending, lift
     38  Stop ascending, lift
     39  Start descending, lift
     40  Stop descending, lift
     41  Standing, moving bus
     42  Sitting, moving bus
     43  Start jogging
     44  Stop jogging
     45  Unknown activity
     46  Unknown activity
     47  Unknown activity
     48  Unknown activity
     49  Unknown activity
     50  Unknown activity
     51  Unknown activity
     52  Unknown activity
     53  Unknown activity
     54  Unknown activity
     55  Unknown activity
     56  Unknown activity
     57  Unknown activity
     58  Unknown activity
     59  Unknown activity
     60  Unknown activity
     61  Unknown activity
     62  Unknown activity
     63  Unknown activity
     64  Unknown activity
     65  Unknown activity
     66  Unknown activity
     67  Unknown activity
     68  Unknown activity
     69  Unknown activity
     70  Unknown activity
     71  Unknown activity
     72  Unknown activity
     73  Unknown activity
     74  Unknown activity
     75  Unknown activity
     76  Unknown activity
     77  Unknown activity
     78  Unknown activity
     79  Unknown activity
     80  Unknown activity
     81  Unknown activity
     82  Unknown activity
     83  Unknown activity
     84  Unknown activity
     85  Unknown activity
     86  Unknown activity
     87  Unknown activity
     88  Unknown activity
     89  Unknown activity
     90  Unknown activity
     91  Unknown activity
     92  Unknown activity
     93  Unknown activity
     94  Unknown activity
     95  Unknown activity
     96  Unknown activity
     97  Unknown activity
     98  Unknown activity
     99  Unknown activity
    100  Unknown activity
    101  Fall F, walking, trip
    102  Fall F, walking, trip, rec.
    103  Fall F, walking, slip
    104  Fall F, walking, slip, rec.
    105  Fall F, walking, slip, rot.
    106  Fall F, walking, slip, rot., rec.
    107  Fall B, walking, slip
    108  Fall B, walking, slip, rec.
    109  Fall B, walking, slip, rot.
    110  Fall B, walking, slip rot., rec.
    111  Fall F, walking, syncope
    112  Fall B, walking, syncope
    113  Fall L, walking, syncope
    114  Fall, syncope, table
    115  Fall F, try sit
    116  Fall F, try sit, rec.
    117  Fall B, try sit
    118  Fall B, try sit, rec.
    119  Fall L, try sit
    120  Fall L, try sit, rec.
    121  Fall F, jog, trip
    122  Fall F, jog, trip, rec.
    123  Fall F, jog, slip
    124  Fall F, jog, slip, rev.
    125  Fall F, jog, slip, rot.
    126  Fall F, jog, slip, rot., rec.
    127  Fall L, bed
    128  Fall L, bed, rec.
    129  Fall F, chair, syncope
    130  Fall B, chair, syncope
    131  Fall L, chair, syncope
    132  Fall F, syncope
    133  Fall B, syncope
    134  Fall L, syncope
    135  Fall, syncope, slide over a wall

  FALL activity ids (35): [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135]
  waist trials: 1798   of which falls: 466  ADL: 1332

==============================================================================
Q1. SisFall Enhanced re-alignment -- final sweep with a null control
==============================================================================
  probe subjects ['SA01', 'SA02', 'SA03'], trials 462
  queries 4000; expected true matches if alignment works: ~316
    cutoff   real   null  real-null  recovery
       5.0     85     14         71    22.5%   (15s)
       4.0     31     13         18     5.7%   (15s)
       3.0     15     13          2     0.6%   (15s)
       2.0      5      3          2     0.6%   (14s)
       1.0      1      2          0     0.0%   (14s)

  best: {'cutoff_hz': 5.0, 'real': 85, 'null': 14, 'net': 71, 'recovery': 0.2248}
  VERDICT -> TIER 3-kfall-only
  The Enhanced per-sample annotation is not recoverable at a usable rate.
  Applying the experimental plan's risk register:
   * original SisFall (25 subjects, filename labels, peak-acceleration impact
     proxy) stays in E1/E2/E5 on the POST-FALL task, so C1 keeps four datasets;
   * KFall alone carries the pre-impact contribution C2. The plan already
     names KFall the more authoritative source, so C2 loses corroboration,
     not its primary evidence.

==============================================================================
Artifacts
==============================================================================
```
