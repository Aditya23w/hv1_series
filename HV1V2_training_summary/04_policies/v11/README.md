# HV1V2 deployment bundle: `v11` (v11g, model_19000)

Decoupled loco-manipulation policy: the network drives the **12 leg joints**. The
waist yaw, arms and head (17 joints) are driven by teleop/IK, and the actor
**observes that upper-body target and its rate**, so it can anticipate arm
motion instead of only reacting to it. Waist roll/pitch are locked at home.

Run `2026-11-02_13-24-36_v11g19k`, task `Mjlab-Velocity-Flat-HV1V2-Decoupled-v11`,
seed 43. Copied from `mjlab_hv1v2/exported/deploy_v11g19k/` (exported
2026-11-02 by `scripts/export_deploy.py`). Full config:
`03_training/runs/v11/config_summary.md`.

| file | |
|---|---|
| `policy.onnx` | 837 inputs, 12 outputs |
| `deploy.json` | joint order, anchor, gains, action scale, observation layout, command ranges, upper-body contract |
| `model_19000.pt` | the training checkpoint it was exported from |
| `checksums.txt` | sha256 of each file |

## Control loop (50 Hz)

1. Update the upper-body command `ub_cmd` (clamped, slewed), **then** build the
   observation from it. The actor sees this step's target.
2. `raw = policy(obs)` gives 12 values.
3. `target[j] = anchor[j] + raw[k] * scale[j]` (hips/knees 0.32, ankles 0.15),
   **clipped to the joint range** (as merai_control clips).
4. The legs get `target`; the upper body gets `ub_cmd`. Both are linearly
   interpolated to 1 kHz (training interpolated across the 4 physics substeps).
5. Keep `raw`: it's next step's `actions` term.

## Observation: 837, term-major, oldest first

| term | dim | history | contributes | source |
|---|---|---|---|---|
| `base_ang_vel` | 3 | 10 | 30 | IMU gyro, pelvis frame |
| `projected_gravity` | 3 | 10 | 30 | `R_pelvis^T @ [0,0,-1]` (unit vector) |
| `joint_pos` | 31 | 10 | 310 | `q - anchor` |
| `joint_vel` | 31 | 10 | 310 | `qdot` |
| `actions` | 12 | 10 | 120 | previous raw output |
| `command` | 3 | 1 | 3 | `[vx, vy, wz]` |
| `ub_target` | 17 | 1 | 17 | `ub_cmd - home` for `policy.ub_obs_joints` |
| `ub_target_rate` | 17 | 1 | 17 | `(ub_cmd_t - ub_cmd_{t-1}) * 50` |

On activation, fill every history slot with the first reading (never zeros),
zero the previous actions, and start the upper-body target at home with zero
rate.

**Commands trained:** vx −0.6..1.0, vy ±0.5, wz ±0.8 m/s, rad/s. Don't go outside them.

## Plant it was trained on (differs from v1–v10)

- Corrected left knee and `HOME_KEYFRAME` (knees 0.350/0.350).
- Joint torque caps at the actuator **peak** (X12 320, X8 120 N·m), not continuous.
- Waist yaw 400/24 (what the robot runs); v1–v10 had 200/12.
- Four-bar stiffness DR: ankle pitch ×0.8–1.45, ankle roll ×0.5–1.1, waist roll/pitch ×2.5–4.5.
- 0–20 ms leg action delay, interpolated targets, hardware-limit clipping, rare 5 ms leg dropout.
- Encoder bias ±3°, IMU roll/pitch bias ±0.75°, foot friction 0.3–2.0.

The `four_bar_warning` in `deploy.json` (written by the generic exporter) still
quotes H1-V1 factors. v11 instead trained across the measured HV1V2 stiffness range above.

## Status (2026-11-02)

- v11f `model_12000` ran on the robot: level torso, flat feet, but at zero
  command it held a leg **squeeze** (hip roll −55/+59 N·m) or a scissor for 40 s.
- v11g, this bundle, adds `stand_leg_torque` and 5× `stand_leg_action_l2`, and widens friction DR to 2.0.
  - Sim squeeze gate (`scripts/check_stand_squeeze.py`): **PASS**. At friction 2.0, mean |hip roll τ| is 6.1/5.5 N·m and |hip pitch R−L| is 10.2 N·m. v11f fails the same gate (13.8/12.8, 25.2).
- **Not yet run on the robot.**
