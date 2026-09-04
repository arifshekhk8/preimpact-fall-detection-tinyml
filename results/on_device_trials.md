# On-device trials

Fill these from `firmware/MEASUREMENT.md` §4. They become a results subsection.

## 4a. Measured false-alarm rate (one hour of ordinary activity, waist-mounted)

Replaces the simulated-ADL figure from E3 (293 window-level false alarms/hour) with a
real one. No falls performed during these runs.

| Run | Threshold | k | Duration | CNN alarms | Rule alarms | CNN alarms/hr | Rule alarms/hr |
|---|---:|---:|---|---:|---:|---:|---:|
| 1 | 0.5 | 3 | | | | | |
| 2 | 0.7 | 3 | | | | | |
| 3 | 0.9 | 2 | | | | | |
| 4 | 0.5 | 4 | | | | | |

Activities performed: _walking, stairs up/down, sitting down heavily, lying down,
picking objects off the floor, ..._

## 4b. CNN versus rule baseline, same physical motion

`mode both` runs both detectors on identical live samples. The rule detector is the
on-device instance of Table I's SMV threshold comparator.

### Falls (mattress drops)

| Trial | Direction | Rule fired | CNN fired | Note |
|---:|---|:--:|:--:|---|
| 1 | backward | | | |
| 2 | forward | | | |
| 3 | lateral | | | |
| … | | | | |

### Near-falls that should NOT fire

| Trial | Action | Rule fired | CNN fired | Note |
|---:|---|:--:|:--:|---|
| 1 | sit down heavily | | | |
| 2 | drop onto sofa | | | |
| 3 | set device down quickly | | | |
| … | | | | |

### Summary

|  | Falls detected | Near-falls falsely fired |
|---|---:|---:|
| Rule baseline | / | / |
| INT8 CNN | / | / |
