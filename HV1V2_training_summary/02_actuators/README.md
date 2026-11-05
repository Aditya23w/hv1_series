# HV1V2 actuators and the simulated plant

| file | what |
|---|---|
| `ACTUATOR_DATASHEET.txt` | vendor data (MyActuator RMD X4-36 / X6-60 / X8-120 / X12-320), every number sourced |
| `hv1v2_constants.py` | how that data becomes the mjlab plant: gains, armature, friction, effort limits, collisions, initial pose |
| `joint_table_v5.md` | all 31 joints for the v5 policy: actuator, kp/kd, action scale, anchor pose, peak / continuous torque |

## Modules on the robot

| module | count | joints | peak N·m | rated (continuous) N·m | gear ratio |
|---|---|---|---|---|---|
| X12-320 | 8 | hip pitch / roll / yaw, knee | 320 | 85 | 20 |
| X8-120 | 15 | ankles and waist roll/pitch (two per joint, four-bar), waist yaw, shoulders, elbows | 120 | 43 | 19.6 |
| X6-60 | 2 | wrist roll | 60 | 20 | 19.6 |
| X4-36 | 4 + 2 | wrist pitch / yaw, head (head actuator inferred) | 34 | 10.5 | 36 |

## Simulated plant (what v1–v8 trained against)

> **v11 plant differs** (`get_hv1v2_v11_robot_cfg` in `hv1v2_constants.py`, compiled as `01_robot/mjcf/hv1v2_v11_sim2sim_scene.xml`):
> - joint clamps are lifted to the actuator **peak** (the forcerange column below);
> - waist yaw runs 400/24 instead of 200/12;
> - the corrected knee and `HOME_KEYFRAME` are used;
> - four-bar stiffness is randomised (ankle pitch ×0.8–1.45, roll ×0.5–1.1, waist roll/pitch ×2.5–4.5).
>
> The table describes v1–v8. v9 and v5ft differ from it only in the knee; v10 also adds ankle-stiffness DR.

Joint-space PD position actuators, 200 Hz physics, 50 Hz policy.

| group | joints | kp | kd | actuator forcerange N·m | **effective joint clamp** N·m | armature | frictionloss | viscous | action scale |
|---|---|---|---|---|---|---|---|---|---|
| X12-320 | 8 | 250 | 15 | 320 | 85 | 0.5160 | 3.8 | 0.390 | 0.32 |
| X8-120 | 9 | 200 | 12 | 120 | 43 | 0.0577 | 1.5 | 0.141 | 0.15 |
| X8-120 ×2 (four-bar) | 6 | 400 | 24 | 240 | ankles 43, waist roll/pitch 85 | 0.1154 | 3.0 | 0.282 | 0.15 |
| X6-60 | 2 | 200 | 12 | 60 | 20 | 0.0254 | 1.6 | 0.016 | 0.075 |
| X4-36 | 6 | 100 | 6 | 34 | 10.5 | 0.0389 | 1.14 | 0.049 | 0.085 |

- **armature** is the vendor rotor inertia × ratio², which assumes the published inertia is motor-side. The vendor doesn't say. It's randomised ×0.6–1.4 during training to hedge.
- **frictionloss** is the vendor back-drive torque.
- **viscous** is inferred from the no-load current; treat it as an upper bound.
- **Torque clamp (important):** the actuator `forcerange` is the vendor peak, but the URDF `effort` values become MuJoCo joint `actuatorfrcrange`, which clamps on top. So every policy v1–v8 trained with each joint capped at its **continuous** rating (ankles 43 N·m, not 240). The real drives can go to peak, so the robot can produce more torque than the policy ever saw. The `effort_limits` DR scales `forcerange` and therefore never binds. Torque metrics in the logs read the pre-clamp value.
- **Four-bar joints:** the ankles and waist roll/pitch are each driven by two X8-120s, so they get doubled parameters. That doubling is what makes standing possible: toppling about the ankles needs about 841 N·m/rad, and two ankle-pitch joints at kp 400 supply 800. **The robot doesn't stand open-loop**; it falls in about 3.8 s with the policy off.
- **On hardware,** PD runs per motor through the four-bar, so the same 400/24 gives a joint stiffness of about 765 N·m/rad in ankle pitch and 550 in roll (≈1.9× and 1.4× the sim's 400), and waist roll/pitch about 3.8–4.2×. Computed from the transmission Jacobian in the robot runtime config, not measured on the bench. Don't write sim kp straight to the motors.
- **Domain randomisation** applied at every episode start: kp/kd ×0.9–1.1, armature ×0.6–1.4, effort ×0.85–1.0, foot friction, encoder bias ±0.015 rad, torso CoM ±25/25/30 mm. From v8 on, ankle stiffness is also randomised ×1–1.9 (pitch) and ×1–1.4 (roll).
