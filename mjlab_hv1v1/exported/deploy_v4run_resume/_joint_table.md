# HV1V1 per-joint table — run `v4run_resume`, iteration 39999

Generated from `deploy.json`. Order is **ctrl order** and is the order the policy's
31 outputs arrive in — index 0 first. Do not reorder.

`kp`/`kd` are the SIMULATION joint-space gains the policy was trained against.
For the six four-bar joints (marked **4B**) these are the DOUBLED values; see
`four_bar_warning` in `deploy.json` before writing them to hardware.

| # | joint | actuator | anchor rad | scale | commandable rad | limits rad | kp | kd | peak N·m |
|---|---|---|---|---|---|---|---|---|---|
| 0 | `right_hip_pitch_joint` | X12-320 | -0.11806 | 0.320 | -0.438 … +0.202 | -2.610 … +2.610 | 250 | 15.0 | 320 |
| 1 | `right_hip_roll_joint` | X12-320 | +0.00337 | 0.320 | -0.317 … +0.323 | -1.830 … +0.430 | 250 | 15.0 | 320 |
| 2 | `right_hip_yaw_joint` | X12-320 | +0.05889 | 0.320 | -0.261 … +0.379 | -2.820 … +2.820 | 250 | 15.0 | 320 |
| 3 | `right_knee_joint` | X12-320 | +0.37554 | 0.320 | +0.056 … +0.696 | -0.174 … +2.094 | 250 | 15.0 | 320 |
| 4 | `right_ankle_pitch_joint` **4B** | X8-120x2 | -0.27378 | 0.180 | -0.454 … -0.094 | -0.870 … +0.520 | 400 | 24.0 | 240 |
| 5 | `right_ankle_roll_joint` **4B** | X8-120x2 | -0.00035 | 0.150 | -0.150 … +0.150 | -0.260 … +0.260 | 400 | 24.0 | 240 |
| 6 | `left_hip_pitch_joint` | X12-320 | -0.09100 | 0.320 | -0.411 … +0.229 | -2.610 … +2.610 | 250 | 15.0 | 320 |
| 7 | `left_hip_roll_joint` | X12-320 | -0.00200 | 0.320 | -0.322 … +0.318 | -0.430 … +1.830 | 250 | 15.0 | 320 |
| 8 | `left_hip_yaw_joint` | X12-320 | -0.04544 | 0.320 | -0.365 … +0.275 | -2.820 … +2.820 | 250 | 15.0 | 320 |
| 9 | `left_knee_joint` | X12-320 | +0.32420 | 0.320 | +0.004 … +0.644 | -0.174 … +2.094 | 250 | 15.0 | 320 |
| 10 | `left_ankle_pitch_joint` **4B** | X8-120x2 | -0.24533 | 0.180 | -0.425 … -0.065 | -0.870 … +0.520 | 400 | 24.0 | 240 |
| 11 | `left_ankle_roll_joint` **4B** | X8-120x2 | +0.00021 | 0.150 | -0.150 … +0.150 | -0.260 … +0.260 | 400 | 24.0 | 240 |
| 12 | `waist_yaw_joint` | X8-120 | +0.00000 | 0.000 | —  (pinned) | -1.700 … +1.700 | 200 | 12.0 | 120 |
| 13 | `waist_roll_joint` **4B** | X8-120x2 | +0.00000 | 0.000 | —  (pinned) | -0.520 … +0.520 | 400 | 24.0 | 240 |
| 14 | `waist_pitch_joint` **4B** | X8-120x2 | +0.00000 | 0.000 | —  (pinned) | -0.520 … +0.520 | 400 | 24.0 | 240 |
| 15 | `right_shoulder_pitch_joint` | X8-120 | +0.20000 | 0.150 | +0.050 … +0.350 | -2.620 … +1.830 | 200 | 12.0 | 120 |
| 16 | `right_shoulder_roll_joint` | X8-120 | -0.12000 | 0.150 | -0.270 … +0.030 | -3.250 … +0.170 | 200 | 12.0 | 120 |
| 17 | `right_shoulder_yaw_joint` | X8-120 | +0.00000 | 0.150 | -0.150 … +0.150 | -2.970 … +2.970 | 200 | 12.0 | 120 |
| 18 | `right_elbow_joint` | X8-120 | +0.90000 | 0.150 | +0.750 … +1.050 | -0.980 … +3.140 | 200 | 12.0 | 120 |
| 19 | `right_wrist_roll_joint` | X6-60 | +0.00000 | 0.000 | —  (pinned) | -2.620 … +2.620 | 200 | 12.0 | 60 |
| 20 | `right_wrist_pitch_joint` | X4-36 | +0.00000 | 0.000 | —  (pinned) | -0.520 … +0.520 | 100 | 6.0 | 34 |
| 21 | `right_wrist_yaw_joint` | X4-36 | +0.00000 | 0.000 | —  (pinned) | -1.080 … +1.080 | 100 | 6.0 | 34 |
| 22 | `left_shoulder_pitch_joint` | X8-120 | +0.20000 | 0.150 | +0.050 … +0.350 | -2.620 … +1.830 | 200 | 12.0 | 120 |
| 23 | `left_shoulder_roll_joint` | X8-120 | +0.12000 | 0.150 | -0.030 … +0.270 | -0.170 … +3.250 | 200 | 12.0 | 120 |
| 24 | `left_shoulder_yaw_joint` | X8-120 | +0.00000 | 0.150 | -0.150 … +0.150 | -2.970 … +2.970 | 200 | 12.0 | 120 |
| 25 | `left_elbow_joint` | X8-120 | +0.90000 | 0.150 | +0.750 … +1.050 | -0.980 … +3.140 | 200 | 12.0 | 120 |
| 26 | `left_wrist_roll_joint` | X6-60 | +0.00000 | 0.000 | —  (pinned) | -2.620 … +2.620 | 200 | 12.0 | 60 |
| 27 | `left_wrist_pitch_joint` | X4-36 | +0.00000 | 0.000 | —  (pinned) | -0.520 … +0.520 | 100 | 6.0 | 34 |
| 28 | `left_wrist_yaw_joint` | X4-36 | +0.00000 | 0.000 | —  (pinned) | -1.080 … +1.080 | 100 | 6.0 | 34 |
| 29 | `head_yaw_joint` | X4-36 | +0.00000 | 0.000 | —  (pinned) | -1.390 … +1.390 | 100 | 6.0 | 34 |
| 30 | `head_pitch_joint` | X4-36 | +0.00000 | 0.000 | —  (pinned) | -0.790 … +0.520 | 100 | 6.0 | 34 |

**20 live**, **11 pinned at scale 0.0** (the policy cannot move them; hold them at the anchor).

Pinned: `waist_yaw_joint`, `waist_roll_joint`, `waist_pitch_joint`, `right_wrist_roll_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint`, `left_wrist_roll_joint`, `left_wrist_pitch_joint`, `left_wrist_yaw_joint`, `head_yaw_joint`, `head_pitch_joint`
