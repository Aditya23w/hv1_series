# HV1V2 locomotion training: summary package

Everything needed to look at, summarise or reproduce how the HV1V2 humanoid's walking policies were trained in **mjlab 1.6 (MuJoCo-Warp) + rsl_rl PPO**.

- The full-body line runs v1→v5, then v9 and v5ft on the corrected-knee model.
- The decoupled legs-only line runs v6→v8, then v10 and v11.
- **Two policies are bundled:**
  - **v5**, the final full-body policy;
  - **v11** (v11g, `model_19000`), the latest decoupled loco-manipulation policy.

  Both get a detailed walkthrough below. The other bundles (v4, v5ft, v6, v7, v9, v10, v11f) are still in `mjlab_hv1v2/exported/deploy_*`.

Built on 2026-10-14 and updated on 2026-11-02 from `~/Desktop/HV1V2/mjlab_hv1v2`. These are copies; the originals are unchanged.

> **Known model error in v1–v8 (found 2026-10-20).** In the v1–v8 models, the left knee joint is 11.9 mm off the knee bore. That's a SolidWorks export error: the URDF puts `left_knee_joint` at `0.011891 0 -0.31067`, while the bore is at `0 0 -0.31`. The right knee is correct. The init pose's knee asymmetry (right 0.376, left 0.324) was mostly the solver compensating for it. `~/Desktop/HV1V2/mjcf/urdf2mjcf.py` now corrects it on generation, and the live `hv1v2_constants.py` has a re-solved `HOME_KEYFRAME`, with the old pose kept as `HOME_KEYFRAME_V1_V8`. The v1–v8 files in this package are left as they were, as the record of what v1–v8 actually trained on. v9, v5ft, v10 and v11 trained on the corrected model, and `01_robot/mjcf/hv1v2_v11_sim2sim_scene.xml` is the corrected v11 plant.

## Folder map

```
README.md                     this file
01_robot/
  urdf/                       HV1V2_URDF_NEW_FOOT_V1_220926.urdf (+ .csv): the source of every model below
  mjcf/hv1v2_robot.xml        training MJCF, as generated (no actuators/floor; mjlab adds them)
  mjcf/hv1v2_viewer.xml       viewing MJCF: floor, lights, placeholder servos, 2 keyframes
  mjcf/hv1v2_sim2sim_scene.xml  the COMPILED v1-v8 training plant: real gains/armature/friction/limits,
                              foot pads, IMU, init_state keyframe. Runs v5.
  mjcf/hv1v2_v11_sim2sim_scene.xml  the COMPILED v11 plant: corrected knee, HOME_KEYFRAME,
                              peak torque caps, waist yaw 400/24. Runs v11.
  mjcf/meshes/                34 STL meshes
  mjcf/README_mjcf_generator.md, tools/urdf2mjcf.py   how URDF -> MJCF is generated
02_actuators/
  README.md                   module table, simulated plant, torque-clamp caveat, DR
  ACTUATOR_DATASHEET.txt      vendor data (MyActuator RMD X-series), every number sourced
  hv1v2_constants.py          plant definitions used by training (live copy, 2026-11-02: v1-v8,
                              corrected and v11 plants; HOME_KEYFRAME_V1_V8 is v5's pose)
  joint_table_v5.md           all 31 joints: actuator, kp/kd, action scale, anchor, torque ratings
03_training/
  lineage_v1_v5.png           reward / episode length / falls across v1..v5 on one axis
  runs/<v1..v11, v5ft>/
    config_summary.md         readable: rewards, observations, actions, DR events, commands,
                              terminations, curriculum, PPO
    env.yaml, agent.yaml      exact configs as logged by the run
    curves.csv, curves.png    every TensorBoard scalar / key curves (none for v8 or v11: their logs
                              are on the remote PC; v11 is the final v11g stage only)
    checkpoints.txt           which checkpoints exist and where
  source/                     training code snapshot: src/ (env_cfgs.py, mdp.py, rl_cfg.py, __init__.py),
                              scripts/ (train.sh, sim2sim.py, check_model.py, export_deploy.py, ...),
                              mdp_v11.py, symmetry.py, tests/test_symmetry.py, eval and audit scripts,
                              README_mjlab_hv1v2.md (the project's own design notes). Snapshot 2026-11-02.
04_policies/v5/               full-body bundle: policy.onnx, deploy.json, README.md, joint table,
                              checkpoint model_47998.pt
04_policies/v11/              decoupled bundle (v11g): policy.onnx, deploy.json, README.md,
                              checkpoint model_19000.pt
05_media/                     sim2sim videos of v1, v3, v4 and the v4 initial pose
tools/
  play_policy.py              run any bundled policy in the MuJoCo viewer (mujoco + onnxruntime only)
  build_summary.py            regenerates 03_training/runs from the mjlab log dirs
```

## Visualise

```bash
cd ~/Desktop/HV1V2/HV1V2_training_summary
conda activate hv1v2_mjlab          # or any env with mujoco, numpy, onnxruntime

python -m mujoco.viewer --mjcf 01_robot/mjcf/hv1v2_viewer.xml   # just the robot
python tools/play_policy.py                                       # v5 walking, keyboard-driven
python tools/play_policy.py --policy v11                          # v11, upper body held at home
python tools/play_policy.py --headless --vx 1.0 --seconds 10      # numbers, no window
tensorboard --logdir ~/Desktop/HV1V2/mjlab_hv1v2/logs/rsl_rl/hv1v2_velocity   # live curves
```

Keys in `play_policy.py`: Up/Down vx ±0.1, Left/Right yaw rate ±0.1, `,`/`.` vy ±0.1, Space stop, Backspace reset. Commands are clamped to the trained ranges.

`play_policy.py` picks the plant from the bundle: v5 runs on the v1–v8 scene, and v11 runs on the v11 scene.

For v11, the player also:
- interpolates leg targets across the 4 substeps;
- clips them to the joint range;
- feeds `ub_target` and `ub_target_rate` = 0 (arms at home).

Checked on this machine, 2026-11-02:

| policy | command | result over 10 s |
|---|---|---|
| v5 | vx 1.0 | 0.90 m/s, standing (matches the v5 README's 0.89–0.90) |
| v5 | stand, turn wz 0.5 | stays upright, drift ≤ 4 cm |
| v11 | vx 0.5 / 1.0 | 0.40 / 0.76 m/s (same as `mjlab_hv1v2/scripts/sim2sim.py`: 3.957 / 7.571 m) |
| v11 | vx −0.5 / vy 0.3 | 0.47 / 0.24 m/s |
| v11 | stand / turn wz 0.5 | 0.01 m / 0.11 m drift, standing |

The `validate_deploy.py` inside the v5 bundle imports `mjlab_hv1v2/scripts/sim2sim.py`, so it only runs from the original repo. Use `tools/play_policy.py` here.

---

## The robot

- **HV1V2:** 92.68 kg, 34 links, **31 actuated hinge joints**.
  - Legs: 2 × 6 (hip pitch/roll/yaw, knee, ankle pitch/roll).
  - Waist: 3. Arms: 2 × 7 (shoulder ×3, elbow, wrist ×3). Head: 2.
- **Actuators:** MyActuator RMD X12-320 (hips, knees), X8-120 (ankles, waist, shoulders, elbows), X6-60 and X4-36 (wrists, head). See `02_actuators/README.md`.
- **Four-bar joints:** the ankles and waist roll/pitch are parallel four-bars driven by two X8-120s each.
- **Easy to get wrong:**
  - The hip-pitch axes are canted about 30°, so the legs are **not mirror images**. The pose and joint limits are per joint, never mirrored.
  - The sole is two patches (heel and toe) with a raised arch. Each foot has 4 collision pads fitted to those patches.
- **Model pipeline:** SolidWorks URDF → `urdf2mjcf.py` → `hv1v2_robot.xml`. mjlab then adds PD actuators, foot pads, sensors (pelvis IMU, root angular momentum) and the initial pose from `hv1v2_constants.py`.
- **Initial / anchor pose:** solved numerically, with feet flat, stance 0.243 m, pelvis at 0.977 m and the CoM 0.4 mm from the sole centre.
- **It is not stable open-loop.** Ankle stiffness is 95% of what the inverted pendulum needs, so the robot falls in about 3.8 s with the policy off.

## How training works (common to v1–v10; v11's differences are in its own section)

| | |
|---|---|
| Simulator | mjlab 1.6 on MuJoCo-Warp, 4096 parallel envs (8192 for v8), flat terrain |
| Timing | physics 200 Hz (dt 0.005 s), policy **50 Hz** (decimation 4), episodes 20 s |
| Task | track a velocity command `[vx, vy, wz]`, heading-controlled for 30% of envs, 10% standing (30% in v6+) |
| Actuation | joint-space PD position targets: `target = anchor + action × action_scale` |
| Algorithm | PPO (rsl_rl): actor and critic MLP 512-256-128 ELU, obs normalisation, 24 steps/env/iter, 5 epochs × 4 minibatches, lr 1e-3 adaptive (KL 0.01), γ 0.99, λ 0.95, clip 0.2, entropy 0.005, initial action std 1.0 |
| Actor obs (v3+) | IMU angular velocity, projected gravity, 31 joint positions (minus anchor), 31 joint velocities, last actions, each with **5-frame history**, plus the command → 498 |
| Critic obs | privileged: base linear velocity, the same proprioception without history, foot height, air time, contact, contact forces |
| Obs noise | gyro ±0.2, gravity ±0.05, joint pos ±0.01, joint vel ±1.5, plus encoder bias ±0.015 rad per episode |
| Domain randomisation | foot friction 0.3–1.2, torso CoM ±25/25/30 mm, kp/kd ×0.9–1.1, armature ×0.6–1.4, effort ×0.85–1.0; pushes every 2–4 s (±0.5 m/s linear, ±0.52 rad/s roll/pitch) |
| Termination | timeout at 20 s, or torso tilt > 70° (`bad_orientation`) |
| Export | ONNX (batch 1) + `deploy.json` (joint order, anchor, gains, scales, obs layout) |

**Observation layout (deployment-critical):** each term's history is stored oldest → newest, and terms are concatenated term-major: `[ang_vel t-4..t0][gravity t-4..t0][joint_pos …][joint_vel …][actions …][command]`. At start-up every history slot is backfilled with the first reading, never zeros.

---

## v5 in detail: the final full-body policy

**Run** `2026-10-05_15-19-51_v5`, checkpoint `model_47998.pt`, task `Mjlab-Velocity-Flat-HV1V2`. Bundle in `04_policies/v5/`, full config in `03_training/runs/v5/config_summary.md`.

### Lineage: v5 is the result of 48k PPO iterations across four runs

| ver | run | iters (wall-clock) | what changed | outcome |
|---|---|---|---|---|
| v1 | `2026-09-24_20-25-01_v1` | 0→2.8k (0.6 h) | baseline: mjlab G1 velocity recipe ported to HV1V2; single-frame obs; `air_time` weight 0 | shuffled: about 3 mm peak foot height |
| v2 | `2026-09-24_21-01-54_v2` | 0→1.2k (0.25 h) | `air_time` on: weight 0.75, window 0.25–0.6 s | stopped early |
| v3 | `2026-09-24_21-21-00_v3` | 0→12k (2.6 h) | actor gets a **5-frame history** (obs 498) | walks; reward 73, episode length 956/1000 |
| v4 | `2026-09-25_00-09-30_v4` | 0→40k (8.6 h) | from scratch; command curriculum vx −0.8..1.0 → −1.0..1.3 at iter 5k → −1.2..1.5 at iter 15k; wz 0.5 → 0.7 | reward 92; 0.94 m/s at a 1.0 command; long steps (0.40 m) and swinging arms |
| **v5** | `2026-10-05_15-19-51_v5` | **40k→48k (1.8 h), fine-tune of v4 `model_39999`** | gait shaping (below); fixed command ranges | reward 116, episode length 988 |

The curves are in `03_training/lineage_v1_v5.png` and `03_training/runs/*/curves.png`. The two reward drops in v4 are the curriculum stages widening the command range.

### What v5 changed on top of v4

The aim was **shorter steps, shorter swings, quiet arms, waist locked**. The interface is identical to v4.

| change | v4 | v5 |
|---|---|---|
| `air_time` weight / window | 0.75 / 0.25–0.6 s | 0.3 / 0.10–0.35 s |
| `feet_flight` (both feet off the ground) | — | −1.0 |
| `air_time_excess`, `t_max` | — | −2.0, 0.40 s |
| `step_length_excess`, `max_length` | — | −2.0, 0.30 m (0.75 × v4's 0.395 m) |
| `arm_joint_vel` (shoulders + elbows) | — | −0.02 |
| pose std, shoulder pitch walk / run | 0.4 / 0.5 | 0.1 / 0.15 |
| commands vx / vy / wz | −1.2..1.5 / ±1.0 / ±0.7 (curriculum) | **−0.8..1.0 / ±0.5 / ±0.7**, no curriculum |

### v5 reward terms (final)

| term | weight | what it does |
|---|---|---|
| `track_linear_velocity` | +3.0 | exp kernel on xy velocity error, std 0.5 |
| `track_angular_velocity` | +3.0 | exp kernel on yaw-rate error, std 0.71 |
| `upright` | +1.0 | torso orientation, std 0.45 |
| `pose` | +1.0 | stay near the anchor pose; per-joint std, tight when standing (0.05), looser walking/running |
| `air_time` | +0.3 | reward foot air time within 0.10–0.35 s while commanded to move |
| `body_ang_vel` | −0.05 | torso angular velocity |
| `angular_momentum` | −0.00285 | whole-body angular momentum (G1 value scaled by mass²) |
| `dof_pos_limits` | −1.0 | joint limits |
| `action_rate_l2` | −0.1 | action smoothness |
| `foot_clearance` | −2.0 | swing foot height vs 0.08 m target |
| `foot_swing_height` | −0.25 | peak swing height vs 0.08 m |
| `foot_slip` | −0.1 | foot sliding while in contact |
| `soft_landing` | −1e-5 | foot impact force |
| `feet_flight` | −1.0 | both feet off the ground |
| `air_time_excess` | −2.0 | swing longer than 0.40 s |
| `step_length_excess` | −2.0 | step longer than 0.30 m |
| `arm_joint_vel` | −0.02 | shoulder/elbow velocity |
| `self_collisions` | −1.0 | self-contact over 10 N |
| `joint_torques` | −2.85e-7 | torque² |

### v5 actions

- **31 outputs, but 20 are live:** 12 legs, 6 shoulders, 2 elbows.
- **11 are pinned** at `action_scale = 0`, so they hold the anchor pose: waist ×3, wrists ×6, head ×2.
- **Action scale** is 0.25 × effort / kp: 0.32 for hips and knees, 0.15 for ankles, shoulders and elbows.

### v5 results (sim2sim, fixed commands, v4 → v5)

| command | step length | swing time | cadence | arm RMS speed | foot lift |
|---|---|---|---|---|---|
| vx 0.3 | 0.144 → 0.087 m | 0.53 → 0.28 s | 1.71 → 2.93 /s | 0.34 → 0.14 rad/s | 3.2 → 1.3 mm |
| vx 0.6 | 0.246 → 0.204 m | 0.42 → 0.30 s | 2.21 → 2.86 /s | 0.56 → 0.18 rad/s | 4.8 → 4.7 mm |
| vx 1.0 | 0.395 → 0.316 m | 0.35 → 0.23 s | 2.57 → 3.79 /s | 0.80 → 0.30 rad/s | 8.7 → 6.1 mm |
| vy 0.3 | — | 0.45 → 0.31 s | 2.00 → 2.86 /s | 0.27 → 0.17 rad/s | 2.4 → 1.6 mm |
| wz 0.5 | — | 0.50 → 0.29 s | 1.86 → 2.93 /s | 0.32 → 0.12 rad/s | 2.5 → 1.2 mm |

- No falls at any command, and never both feet in the air.
- At vx 1.0 it achieves 0.89–0.90 m/s.
- Over training episodes the worst actuator averaged 129% of its continuous rating (v4: 156%).

### v5 known limits

- **Foot clearance is the main risk:** 1–6 mm against an 80 mm target. It's close to sliding the feet, so a mat edge or cable can catch a foot.
- **Fast cadence:** about 3.8 steps/s at 1.0 m/s.
- **Torque:** the sim clamped every joint at its continuous rating (see `02_actuators/README.md`), so sim torque numbers aren't hardware loads.
- **Unvalidated assumptions:**
  - armature is provisional (motor-side inertia assumed);
  - the four-bar gain amplification hasn't been applied;
  - the kp/kd values are inherited from H1-V1.

---

## After v5

### Full-body line on the corrected knee (v9, v5ft)

| ver | run | iters | what changed | outcome |
|---|---|---|---|---|
| v9 | `2026-10-20_23-37-27_v9` | 20k from scratch | v5's task, rewards and DR on the corrected knee + re-solved `HOME_KEYFRAME` | walks, but loses to v5: 22% more falls under DR, about 5% worse tracking, keeps stepping after a stop on the real-ankle plant (24/30 runs vs v5's 6/30). The gap is training budget (20k vs 48k), not the knee fix |
| v5ft | `2026-10-26_20-07-21_v5ft` | 47998→55998, fine-tune of v5 | v5 + 8k iterations on the corrected model and pose | best on the training plant (DR falls 0.028 vs v5's 0.037 per env-minute, fastest, least tilt); on the real-ankle plant it steps after a stop in 13/30 runs and needs more push-recovery steps, so **v5 stays the full-body pick** |

The comparison report is `mjlab_hv1v2/logs/eval/v5_vs_v9/REPORT.md`. It was produced with `scripts/eval_dr.py`, `compare_gait.py` and `eval_plant.py --plant real`.

### The decoupled line (v6–v11)

For loco-manipulation, the policy was cut down to the **12 leg joints**. The upper body is driven externally by IK or teleop.

| ver | iters | what changed | status |
|---|---|---|---|
| v6 | 30k from scratch (7.6 h) | legs-only actions (obs 403); random upper-body targets with a level curriculum; 0–2 kg hand payload; ±20 N hand forces; 30% standing envs; stand-still foot terms | stands still under arm motion, but the legs are jerky |
| v7 | 30k from scratch (7.75 h) | v6 + six leg-smoothness terms ramped in over iters 1k–5k, calibrated to fixed reward shares | smooth, and ran on the robot; problems: 1–2 Hz hip-pitch rocking from arm motion, torso tilt with every arm move, right foot standing on its outer edge |
| v8 | 15k from scratch at 8192 envs (remote PC) | v7 + Ruckig upper-body clips, interpolated targets, 0–20 ms leg delay, ankle-stiffness DR | rocking −33 to −39%, but new self-sway and heading drift; never shipped |
| v10 | v7 `model_29999` + 8k (to 37998) | warm start from v7 on the corrected model, with stance terms (sole tilt, ankle-roll pose std 0.02) and real ankle stiffness | fixes v7's edge-standing in sim, but won't start walking from a stand at small commands; not deployed |
| **v11** | **19k, fresh, 8192 envs (remote PC)** | fresh training combining v8 + v10, plus a realistic plant, an actor that sees the upper-body target, flat-foot/tilt rewards and a symmetry loss (below) | **latest: v11g `model_19000`**; v11f ran on the robot, v11g not yet |

Only the v11 bundle is in `04_policies/`. The v6, v7 and v10 bundles are in `mjlab_hv1v2/exported/`.

---

## v11 in detail: the latest decoupled policy

**Run** `2026-11-02_13-24-36_v11g19k` (seed 43), checkpoint `model_19000.pt`, task `Mjlab-Velocity-Flat-HV1V2-Decoupled-v11`.
- Bundle: `04_policies/v11/`.
- Config: `03_training/runs/v11/config_summary.md`.
- Design notes: the long comment block above `hv1v2_decoupled_v11_env_cfg` in `03_training/source/src/env_cfgs.py`, and the header of `mdp_v11.py`.

### Why v11 exists

v7 runs on the robot but has three problems:
- the torso tilts with every arm, head or waist move;
- one ankle is held rolled;
- the legs rock at 1–2 Hz.

v7's own config explains why:
- **Tilt was almost free:** the `upright` kernel std was 0.447, which still pays 0.96 at 5°.
- **The actor never saw the upper-body target**, so it could only react to arm motion.
- **Nothing priced a flat foot** while walking.
- **The plant was wrong in the directions that matter:**
  - no latency;
  - zero-order-hold targets;
  - no four-bar stiffness;
  - continuous-only torque;
  - waist yaw at 200/12.

### What v11 changes

| area | v7 | v11 |
|---|---|---|
| plant | old knee, continuous torque caps, waist yaw 200/12, ZOH targets, no delay | corrected knee + `HOME_KEYFRAME`, **peak** torque caps, waist yaw 400/24, interpolated targets, 0–20 ms leg delay, hardware-limit clip, rare 5 ms leg dropout |
| four-bar stiffness DR | none | ankle pitch ×0.8–1.45, roll ×0.5–1.1, waist roll/pitch ×2.5–4.5 |
| sensor DR | encoder ±0.86° | encoder ±3° (legs), IMU roll/pitch bias ±0.75°, gyro bias, rare IMU freezes |
| DR ramp | fixed | DR half-widths 50% → 100%, advancing only while the walking-tracking gate is open |
| upper body | one random speed for all joints, point-mass payload, constant hand wrench | per-joint speeds with caps, 2% single-step holds, payload 0–2 kg/hand with CoM offset and inertia, mid-episode pick-ups as ramped wrenches, smooth hand forces, torso pushes; waist roll/pitch locked |
| actor obs | 5-frame history, no upper-body target | **10-frame history**, plus `ub_target` and `ub_target_rate` (17 joints, 1 frame); 837 inputs |
| commands | | vx −0.6..1.0, vy ±0.5, wz ±0.8; stop and reversal events |
| tilt rewards | `upright` std 0.447, `body_ang_vel` −0.05 | wide `upright` (std 0.35) + precise one (std 0.06, ramped in), `body_ang_vel` to −0.5, `torso_quiet_under_ub` |
| foot rewards | none while walking | `loaded_sole_tilt` (Fz > 20 N), `pad_load_balance`, `ankle_roll_torque_stance` (4 N·m deadband) |
| standing | | `stand_leg_action_l2`, `stand_stance_shape`, `still_stand_quiet`, and in v11g `stand_leg_torque` |
| tracking | one kernel (std² 0.25) | wide (std² 0.25, ×3) + precise (std² 0.05, ×2) kernels; `air_time` pays only while the body tracks the command |
| PPO | | 8192 envs, MLP 512-256-128, **mirror symmetry loss** (`symmetry.py`, FK-checked in `tests/test_symmetry.py`) |

New-term weights were calibrated by rolling v7 out in the new environment (`scripts/calibrate_v11.py`) and pricing each term at a fixed share of v7's per-step reward.

**Reading `env.yaml`:** several weights there are ramp **start** values. Examples: `stand_leg_torque` 0.0, `upright_precise` 0.0, and the posture terms at their resume values. They ramp in on a gated "clock" at runtime.

### How v11 got to v11g: one seed-43 chain, each stage resumed from the last

| stage | iterations | problem in the previous stage | fix |
|---|---|---|---|
| v11 (s42) | 0→2k, stopped | | all posture/stance penalties at full weight from iteration 0. It learned to stand and turn in place and never translate |
| v11b / v11c | 0→800 (the part v11d kept; v11c ran on to ~9.6k) | v11 never walked | penalties ramp 5% → 100%; the upper-body level and DR ramp only advance once walking tracks. In v11c the gate needs forward, backward and lateral tracking plus a minimum swing time |
| v11d | 800→2.4k | v11c walked at iteration 800, then stood still from ~3000 | ankle acceleration/torque-rate weights cut; every ramp keyed to a clock that only advances while the walking error is low |
| v11e | 2.4k→6k | v11d dropped slow (0.3 m/s) commands once the ramp finished | linear tracking split into wide + precise kernels; a slow-command error gate; the clock rewinds when tracking degrades |
| v11f | 6k→14.4k | v11e stood still at ≥ 0.8 m/s commands | wide kernel ×3, precise std² 0.05; fast-command gate. **`model_12000` ran on the robot** |
| **v11g** | **14.4k→19k** | on the robot, v11f had a level torso and flat feet, but at zero command it held a leg squeeze (hip roll −55/+59 N·m) or a scissor (hip pitch +62/−44) for 40 s; it was holding its own last action | `stand_leg_torque` (deadbands 10/15/30 N·m on hip roll/pitch/knee), `stand_leg_action_l2` ×5, foot friction DR 0.3–2.0, both ramped over 500 clock iterations |

The checkpoints for v11b–v11e and their TensorBoard logs are on the remote PC.

### v11g results (sim)

| check | v11f `model_12000` | v11g `model_19000` |
|---|---|---|
| standing squeeze, friction 2.0 (`check_stand_squeeze.py`): mean \|hip roll τ\| R/L | 13.8 / 12.8 N·m | **6.1 / 5.5** |
| same: \|hip pitch R−L\| | 25.2 N·m | **10.2** |
| gate (hip roll ≤ 10, pitch R−L ≤ 15) | FAIL | **PASS** (both frictions) |

sim2sim on the v11 plant (arms held at home, 10 s, no falls):

| command | speed | step length (heading axis) | swing time | cadence | foot lift |
|---|---|---|---|---|---|
| vx 0.5 | 0.40 m/s | 0.162 m | 0.30 s | 2.67 /s | 0.7 mm |
| vx 1.0 | 0.76 m/s | 0.382 m | 0.34 s | 2.56 /s | 2.2 mm |

### v11 known limits

- **Not yet run on the robot.** The squeeze fix is verified in sim only.
- **Under-tracks:** 0.76 m/s at a 1.0 command and 0.40 at 0.5, below the plan's ≥ 0.9 m/s gate.
- **Foot clearance is very low:** 1–2 mm lift, the same risk as v5.
- **Phase-2 gates not reported for v11g yet:** tilt under sweeps, pads, asymmetric payload, latency margin. Run them with `scripts/eval_v11_gates.py`.
- **Armature is still provisional,** and the X8-120 1.16× torque-scale question is absorbed by the stiffness DR rather than settled.
