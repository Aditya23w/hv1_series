| # | joint | driven by | actuator | kp | kd | action_scale | home rad | IK range rad | peak N·m | cont N·m |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `right_hip_pitch_joint` | **policy** | X12-320 | 250 | 15 | 0.320 | -0.11483 | — | 320 | 85 |
| 1 | `right_hip_roll_joint` | **policy** | X12-320 | 250 | 15 | 0.320 | +0.00319 | — | 320 | 85 |
| 2 | `right_hip_yaw_joint` | **policy** | X12-320 | 250 | 15 | 0.320 | +0.05729 | — | 320 | 85 |
| 3 | `right_knee_joint` | **policy** | X12-320 | 250 | 15 | 0.320 | +0.37554 | — | 320 | 85 |
| 4 | `right_ankle_pitch_joint` | **policy** | X8-120x2 | 400 | 24 | 0.150 | -0.27660 | — | 240 | 86 |
| 5 | `right_ankle_roll_joint` | **policy** | X8-120x2 | 400 | 24 | 0.150 | -0.00033 | — | 240 | 86 |
| 6 | `left_hip_pitch_joint` | **policy** | X12-320 | 250 | 15 | 0.320 | -0.08776 | — | 320 | 85 |
| 7 | `left_hip_roll_joint` | **policy** | X12-320 | 250 | 15 | 0.320 | -0.00186 | — | 320 | 85 |
| 8 | `left_hip_yaw_joint` | **policy** | X12-320 | 250 | 15 | 0.320 | -0.04382 | — | 320 | 85 |
| 9 | `left_knee_joint` | **policy** | X12-320 | 250 | 15 | 0.320 | +0.32420 | — | 320 | 85 |
| 10 | `left_ankle_pitch_joint` | **policy** | X8-120x2 | 400 | 24 | 0.150 | -0.24814 | — | 240 | 86 |
| 11 | `left_ankle_roll_joint` | **policy** | X8-120x2 | 400 | 24 | 0.150 | +0.00019 | — | 240 | 86 |
| 12 | `waist_yaw_joint` | IK | X8-120 | 200 | 12 | — | +0.00000 | -1.00 .. +1.00 | 120 | 43 |
| 13 | `waist_roll_joint` | IK — **locked at home** | X8-120x2 | 400 | 24 | — | +0.00000 | +0.00 .. +0.00 | 240 | 86 |
| 14 | `waist_pitch_joint` | IK — **locked at home** | X8-120x2 | 400 | 24 | — | +0.00000 | +0.00 .. +0.00 | 240 | 86 |
| 15 | `right_shoulder_pitch_joint` | IK | X8-120 | 200 | 12 | — | +0.20000 | -1.60 .. +0.80 | 120 | 43 |
| 16 | `right_shoulder_roll_joint` | IK | X8-120 | 200 | 12 | — | -0.12000 | -1.20 .. +0.00 | 120 | 43 |
| 17 | `right_shoulder_yaw_joint` | IK | X8-120 | 200 | 12 | — | +0.00000 | -1.00 .. +1.00 | 120 | 43 |
| 18 | `right_elbow_joint` | IK | X8-120 | 200 | 12 | — | +0.90000 | +0.00 .. +2.20 | 120 | 43 |
| 19 | `right_wrist_roll_joint` | IK | X6-60 | 200 | 12 | — | +0.00000 | -1.50 .. +1.50 | 60 | 20 |
| 20 | `right_wrist_pitch_joint` | IK | X4-36 | 100 | 6 | — | +0.00000 | -0.45 .. +0.45 | 34 | 10.5 |
| 21 | `right_wrist_yaw_joint` | IK | X4-36 | 100 | 6 | — | +0.00000 | -0.90 .. +0.90 | 34 | 10.5 |
| 22 | `left_shoulder_pitch_joint` | IK | X8-120 | 200 | 12 | — | +0.20000 | -1.60 .. +0.80 | 120 | 43 |
| 23 | `left_shoulder_roll_joint` | IK | X8-120 | 200 | 12 | — | +0.12000 | +0.00 .. +1.20 | 120 | 43 |
| 24 | `left_shoulder_yaw_joint` | IK | X8-120 | 200 | 12 | — | +0.00000 | -1.00 .. +1.00 | 120 | 43 |
| 25 | `left_elbow_joint` | IK | X8-120 | 200 | 12 | — | +0.90000 | +0.00 .. +2.20 | 120 | 43 |
| 26 | `left_wrist_roll_joint` | IK | X6-60 | 200 | 12 | — | +0.00000 | -1.50 .. +1.50 | 60 | 20 |
| 27 | `left_wrist_pitch_joint` | IK | X4-36 | 100 | 6 | — | +0.00000 | -0.45 .. +0.45 | 34 | 10.5 |
| 28 | `left_wrist_yaw_joint` | IK | X4-36 | 100 | 6 | — | +0.00000 | -0.90 .. +0.90 | 34 | 10.5 |
| 29 | `head_yaw_joint` | IK | X4-36 | 100 | 6 | — | +0.00000 | -1.00 .. +1.00 | 34 | 10.5 |
| 30 | `head_pitch_joint` | IK | X4-36 | 100 | 6 | — | +0.00000 | -0.60 .. +0.40 | 34 | 10.5 |
