# mjlab_hv1v2

HV1V2 humanoid robot and velocity-tracking tasks for [mjlab](https://github.com/mujocolab/mjlab) 1.6.0.

```bash
conda activate hv1v2_mjlab
list-envs | grep HV1V2
./scripts/train.sh                       # primary task, 4096 envs
```

| Task | Actor sees | Use |
|---|---|---|
| `Mjlab-Velocity-Flat-HV1V2` | IMU + encoders + actions + command | **primary** — deployable |
| `Mjlab-Velocity-Flat-HV1V2-State-Estimation` | the above + base linear velocity | baseline / ablation |
| `Mjlab-Velocity-Rough-HV1V2` | the above + terrain height scan | rough terrain |
| `Mjlab-Velocity-Flat-HV1V2-Decoupled` | IMU + all 31 encoders + 12 leg actions + command | **v6**: legs only, upper body is an IK-driven disturbance (`exported/deploy_v6`) |
| `Mjlab-Velocity-Flat-HV1V2-Decoupled-Smooth` | same as Decoupled | **v7**: v6 plus leg smoothness shaping (see "Smoothness (v7)") |

## It does not touch the mjlab checkout

This is a real out-of-tree plugin. `pyproject.toml` declares

```toml
[project.entry-points."mjlab.tasks"]
mjlab_hv1v2 = "mjlab_hv1v2"
```

and mjlab's `_import_registered_packages()` imports it on `import mjlab`, which is
when the `register_mjlab_task` calls fire. `../mjlab_src` stays a pristine clone you
can `git pull`. This differs from `g1_lab` / `c1_lab` / `d1_lab` / `h1v1_lab`, which
all vendor the robot *into* their mjlab fork.

Because entry-point import failures are swallowed into a `[WARN]` traceback on
stderr and the tasks just never appear, debug with `python -c "import mjlab_hv1v2"`,
where the traceback is fatal.

## Layout

```
src/mjlab_hv1v2/
  hv1v2_constants.py   plant, collisions, initial state, action scale
  env_cfgs.py          rough + flat + play, all the per-robot blanks
  rl_cfg.py            PPO
  mdp.py               torque-duty metrics
  assets/hv1v2/        hv1v2_robot.xml + meshes (symlinks into ../../meshes)
scripts/
  train.sh play.sh view.py
  check_model.py       audits the COMPILED plant against the intended one
  solve_init_pose.py   solves / re-checks the standing pose
  plant_report.py      step response, PD hold, torque duty, sensor frames
```

The MJCF is generated, not hand-written:

```bash
python ../mjcf/urdf2mjcf.py --target robot --copy-meshes
```

## The plant

Five actuator groups over 31 joints, from `ACTUATOR_DATASHEET.txt`.

| group | n | kp | kd | effort | armature | frictionloss | viscous | action scale |
|---|---|---|---|---|---|---|---|---|
| X12-320 | 8 | 250 | 15 | 320 | 0.5160 | 3.8 | 0.390 | 0.320 |
| X8-120 | 9 | 200 | 12 | 120 | 0.0577 | 1.5 | 0.141 | 0.150 |
| X8-120 ×2 (four-bar) | 6 | 400 | 24 | 240 | 0.1154 | 3.0 | 0.282 | 0.150 |
| X6-60 | 2 | 200 | 12 | 60 | 0.0254 | 1.6 | 0.016 | 0.075 |
| X4-36 | 6 | 100 | 6 | 34 | 0.0389 | 1.14 | 0.049 | 0.085 |

`effort` is the vendor PEAK torque, `frictionloss` the vendor back-drive torque,
kp/kd the H1-V1 *simulation* plant (not its deployed gains, which carry four-bar
pre-division belonging to hardware).

**The ankles and waist roll/pitch are parallel four-bars driven by two X8-120s
each**, so they get doubled parameters as mjlab's own G1 config does. This is
load-bearing, not cosmetic: toppling about the ankles needs
`m·g·z_com = 92.682 × 9.807 × 0.9254 ≈ 841 N·m/rad`, and two ankle-pitch joints at
kp 400 supply 800. At the single-module kp 200 they would supply 400.

**Action space:** all 31 joints, with 11 pinned to `action_scale = 0.0` (waist ×3,
wrists ×6, head ×2), leaving 20 live — 12 legs, 6 shoulders, 2 elbows. Nothing
prices a pinned joint's output, so the network drifts there; unpinning one without
adding a cost in the same change makes it slam.

## Observation history

The actor carries **5 frames** (100 ms at 50 Hz). The critic does not — it already sees
privileged state and gains nothing from a window.

```
base_ang_vel        3 x 5 =  15
projected_gravity   3 x 5 =  15
joint_pos          31 x 5 = 155
joint_vel          31 x 5 = 155
actions            31 x 5 = 155
command             3 x 1 =   3      <- excluded, constant within a resample window
                            ---
                            498      actor        (critic stays 117)
```

`Mjlab-Velocity-Flat-HV1V2-State-Estimation` adds `base_lin_vel` 3x5 → 513.
`Mjlab-Velocity-Rough-HV1V2` is 700: `height_scan` is also excluded, because at 187 numbers
per frame stacking it would add 748 inputs and take the actor to 1448, for a terrain map
that shifts by roughly one 0.1 m grid cell over the window.

Why at all: this env redraws mass, friction, PD gains, armature, payload, push and CoM
offset every episode, and a memoryless actor cannot tell which draw it is on. Base linear
velocity is also in *no* actor term on the primary task — one frame cannot form it, but
`actions` against `joint_pos` a step later can.

**Layout — get this wrong and nothing errors.** mjlab is **term-major, oldest→newest**:
each term's whole history is contiguous, then terms are concatenated, and the *last* D
values of each slice are the current frame.

```
[ang_vel t0..t4][gravity t0..t4][joint_pos t0..t4]...[command t4]
```

Many deploy stacks are frame-major instead. H1-V1 records that its own
`robo_control/docs/ARCHITECTURE.md` §17 claims frame-major and is **wrong**, verified
against the shipped `libpolicy_service_lib.so`.

**Reset:** all 5 slots are backfilled with the first post-reset frame, never zeros, so
there is no cold-start transient. **A deploying runtime must do the same at t=0.**

**Noise** is applied *before* the buffer, so each slot holds an independently corrupted
snapshot — that is what lets the actor filter rather than just remember.

**Deployment:** the ONNX input is simply 498 wide; the buffer is not in the graph, so the
runtime owns it. The export writes per-term `observation_terms_history_length`
(`5,5,5,5,5,0`) into the ONNX `metadata_props`. Note `policy_service` carries a single
`history_length` for all terms, so it will assemble 102 x 5 = **510** and a `Gather` must
drop the 12 stale `command` copies, keeping the **newest** — the same reconciliation
`h1v1_lab/mnn/build_mnn.py` performs, already covered by its `test_gather_index.py` `(5, 1)`
case.

**Caveat:** history is not a proven win on this hardware. H1-V1's own `history-5` run is
the weakest row in its stand table (peak reward 68.8 vs 98.7), and the S1/S2 gains were
attributed to other changes. Its locomotion notes call history "the real fix for the
marching". Set `OBS_HISTORY = 1` in `env_cfgs.py` to restore the single-frame layout
exactly.

## Things that are easy to get wrong here

- **The legs are not mirror images.** The hip-pitch axes are canted ~30° out of
  pure-Y (`right` world axis `[0, 0.866, 0.5]`, `left` `[0, 0.866, -0.5]`), so a
  hip-pitch rotation carries a yaw component of opposite sign per leg. The initial
  pose is listed **per joint**, not with `.*_hip_pitch_joint`-style regexes, and a
  symmetric crouch does not put the feet flat. `right_hip_roll` is `[-1.83, 0.43]`
  against left `[-0.43, 1.83]`, so limits must never be mirrored either.
- **The sole is not a flat plate.** It has a heel patch (~83 × 100 mm) and a toe
  patch (~86 × 117 mm) separated by an arch standing 6–8 mm proud. The four pads
  per foot are fitted to those patches; a bounding-box tiling would fill the arch
  in and merge two contact patches into one.
- **`viscous_damping` is not `damping`.** The first is a passive joint property
  (`dof_damping`), the second the PD derivative gain (`actuator_biasprm[2]`).
  `check_model.py` asserts both separately.
- **An unmatched joint gets action scale 1.0**, silently — `BaseAction` inits
  `torch.ones`. Assert the compiled 31-vector, never the config dict.
- **mjlab's regex filters are prefix-anchored** (`re.match`, not `fullmatch`), so
  every pattern here ends in `$`. Unanchored, `.*_col` would swallow the foot pads.

## Verification

Run in this order; each gates the next.

```bash
python scripts/check_model.py            # compiled plant == intended plant
python scripts/solve_init_pose.py --check # feet flat, CoM in support, no self-contact
python scripts/plant_report.py           # step response, PD hold, torque duty, sensors
RUN=smoke ITERS=200 NUM_ENVS=512 ./scripts/train.sh
```

Measured on an RTX 5080 Laptop:

| | |
|---|---|
| unloaded step response | rise 0.12–0.32 s, ≤8% overshoot, ≤3 mrad error |
| holding the init pose | worst joint 37.9% of its **continuous** rating (ankle pitch) |
| open-loop PD hold | steady ~3.8 s, then topples |
| throughput | 0.37 s/iter @ 512 envs, 0.75 s/iter @ 4096 |
| 200-iteration smoke | episode length 12.8 → 78.1, `fell_over` 12.1 → 7.5 |

**The robot is expected to fall open-loop.** Ankle stiffness is 95% of what
toppling demands, and `ACTUATORS.txt` §7 records the same for H1-V1: "no pose
holds the robot up without a policy at these gains." `plant_report.py` treats the
fall as a characterisation — what it checks is that the fall is slow and
symmetric, the torques are inside duty, and nothing goes numerically wrong.

## Provenance: what is grounded, and what to revisit after v3

Several settings here were copied from the sibling H1-V1 project. That project's own
records are worth reading before trusting any of them: **every H1-V1 standing run peaks
around iteration 550-590 and then degrades for 3,400 more iterations** (end reward 16-56
from peaks of 58-98.7, falls:timeouts up to 27:1), its notes state *"no run has ever been
evaluated at fixed difficulty, so 'better' has never been measurable here"*, and the
shipped S2 policy **has never seen a nonzero command** — it hits the floor at 1.46 s on
`turn 0.8` and its command ranges are clamped to `[0,0,0]`. H1-V1 has produced a
zero-command stand, not a walk. Treat it as a source of hardware facts and measurements,
**not** as a validated configuration.

**Independently grounded — not from H1-V1:**

| setting | basis |
|---|---|
| armature, frictionloss, effort_limit | vendor datasheet |
| four-bar doubling | mjlab's G1 convention + the 841 vs 800 N·m/rad ankle calculation |
| initial pose | solved numerically for HV1V2's own geometry, feet-flat verified |
| foot pads | measured from HV1V2's two-patch sole |
| `angular_momentum` weight | mass² scaling of G1's value |
| `pose` std dicts | G1 |
| `air_time` > 0 | our own v1 measurement (3 mm peak height = shuffling) |
| v5 gait terms, `L_max` 0.30 m, `air_time` 0.3 / 0.10–0.35 s | v4 sim2sim gait table (below) — a deliberate gait choice |
| waist locked | v5 **requirement**, no longer just inherited |

**H1-V1-derived and NOT independently validated — revisit after v3:**

| setting | what is actually unverified |
|---|---|
| **wrists + head pinned** (the waist is now a requirement, see above) | H1-V1 pins its wrists because they are *physically unassembled on tahiti-d1-002*, and its waist because *its* gains are "log-interpolated seeds". Neither is a fact about HV1V2. mjlab's working G1 pins **nothing**. The waist is the cheapest CoM authority on a 93 kg robot and we have it disabled. |
| kp/kd 250/15, 200/12, 400/24, 100/6 | H1-V1's *deployed hardware* gains (~3.5 Hz bandwidth, ζ≈1.9). Defensible for sim2real **only if HV1V2 runs the same control stack** — unconfirmed. G1 instead derives `kp = armature × ω_n²` at 10 Hz, which with our armature would give X12 kp 2037. |
| `entropy_coef` 0.005 | vs G1's 0.01. The reasoning (11 dead + 8 arm actions) is plausible, never tested. |
| knee prior 0.35 in `solve_init_pose.py` | arbitrary inheritance |
| `OBS_HISTORY = 5` | from H1-V1's `history-5` run, the weakest row in its stand table |

## Gait shaping (v5)

v5 is a fine-tune of v4 aimed at **shorter steps, shorter swings, quiet arms, waist
locked**. Same observations, actions and pinned set as v4, so it warm-starts from
`model_39999.pt` and deploys with the same layout.

v4 baseline, `scripts/sim2sim.py --onnx exported/deploy_v4/policy.onnx --headless 750`
(first 1 s dropped; step length is the landing foot ahead of the other one, along the heading):

| command | step length | swing time | cadence | double flight | arm RMS speed |
|---|---|---|---|---|---|
| vx 0.3 | 0.144 m | 0.527 s | 1.71 /s | 0 | 0.34 rad/s |
| vx 0.6 | 0.246 m | 0.424 s | 2.21 /s | 0 | 0.56 rad/s |
| vx 1.0 | 0.395 m | 0.352 s | 2.57 /s | 0 | 0.80 rad/s |
| vy 0.3 | 0.023 m | 0.449 s | 2.00 /s | 0 | 0.27 rad/s |
| wz 0.5 | 0.106 m | 0.504 s | 1.86 /s | 0 | 0.32 rad/s |

v5 result (`2026-10-05_15-19-51_v5`, `model_47998.pt`, 8000 iterations on top of v4), same commands:

| command | step length | swing time | cadence | double flight | arm RMS speed | foot lift |
|---|---|---|---|---|---|---|
| vx 0.3 | 0.087 m | 0.278 s | 2.93 /s | 0 | 0.14 rad/s | 1.3 mm |
| vx 0.6 | 0.204 m | 0.299 s | 2.86 /s | 0 | 0.18 rad/s | 4.7 mm |
| vx 1.0 | 0.316 m | 0.232 s | 3.79 /s | 0 | 0.30 rad/s | 6.1 mm |
| vy 0.3 | 0.061 m | 0.311 s | 2.86 /s | 0 | 0.17 rad/s | 1.6 mm |
| wz 0.5 | 0.014 m | 0.294 s | 2.93 /s | 0 | 0.12 rad/s | 1.2 mm |

No falls at any of these commands. It covers 13.5 m in 15 s at vx 1.0, against
v4's 14.3 m.

**Open issue: foot lift.** v4 already lifted only 3–9 mm, and v5 is worse at slow
speeds and when turning (1.2–1.6 mm). The `foot_clearance` and
`foot_swing_height` targets of 80 mm are not being enforced. Watch `peak_height_mean` in v5: shorter swings
must not turn into shuffling.

| change | v4 | v5 |
|---|---|---|
| `air_time` weight / window | 0.75 / 0.25–0.6 s | 0.3 / 0.10–0.35 s |
| `feet_flight` (both feet up) | — | −1.0 |
| `air_time_excess`, `t_max` | — | −2.0, 0.40 s |
| `step_length_excess`, `max_length` | — | −2.0, 0.30 m (0.75 × 0.395) |
| `arm_joint_vel` (`joint_vel_l2`, shoulders + elbows) | — | −0.02 |
| pose std, shoulder pitch walk / run | 0.4 / 0.5 | 0.1 / 0.15 |
| command vx / vy / wz | up to 1.5 / ±1.0 / ±0.7 | −0.8..1.0 / ±0.5 / ±0.7 |

The new training metrics, `Metrics/step_length_mean`, `swing_time_mean`,
`Episode_Metrics/double_flight_frac` and `arm_joint_speed`, average over the
random command mix, and the contact sensor counts heel/toe rocking as short
swings. **Use them for trends only.** Compare policies with the sim2sim table at
fixed commands.

```bash
RUN=v5 ITERS=8000 ./scripts/train.sh Mjlab-Velocity-Flat-HV1V2 \
  --agent.resume True --agent.load-run '2026-09-25_00-09-30_v4$' \
  --agent.load-checkpoint 'model_39999\.pt'
```

The run regex must be anchored: `2026-09-29_23-07-29_v4` also ends in `_v4`.
Resuming keeps the iteration count (v5 runs 40000→48000) and the action std, and
resets the learning rate to 1e-3.

## Corrected left knee (v9)

v9 is v5's whole-body task (`Mjlab-Velocity-Flat-HV1V2`, same rewards, observations,
DR and commands) trained **from scratch** on the corrected plant:
- `mjcf/urdf2mjcf.py` moves the left knee axis onto its bore. The SolidWorks export had
  it 11.9 mm off.
- `HOME_KEYFRAME` is re-solved: knees 0.350 / 0.350, pelvis 0.97939.

v1–v8 trained on the old model and pose, which is kept as `HOME_KEYFRAME_V1_V8`. Before
the run, the logged env.yaml matched v5's except for `num_envs` and the init pose.
v5 itself was v4 (40k from scratch) plus 8k of fine-tuning, so v9 is the first time the
v5 rewards train from zero. Watch whether it learns to walk at all before trusting the
short-swing terms.

```bash
RUN=v9 ITERS=20000 ./scripts/train.sh Mjlab-Velocity-Flat-HV1V2
```

v5 on the corrected model, for comparison (sim2sim, 15 s): vx 1.0 gives 0.317 m steps,
0.243 s swing, 3.57 /s cadence and 8.3 mm lift. On the old model it gave 0.316 / 0.232 /
3.79 / 6.1.

## Smoothness (v7)

v6 legs were jerky. Measured with `sim2sim.py --headless --smooth` at vx 0.6,
random full-amplitude arm motion, 2 kg per hand:
- mean |Δa| 0.122 (v4 0.094);
- |Δ²a| 0.102 (v4 0.056);
- hip-yaw |Δ²a| 3.2× v4;
- hip/knee target jumps up to 19°;
- hip/knee torque rate about 2.4× v4;
- 3–8 Hz share 0.26 while standing under arm motion (v4 0.04).

v7 adds six terms, ramped linearly in over iterations 1000–5000. Each final
weight is calibrated by rolling v6 out in the v7 env and pricing the term at a
fixed share of v6's per-step reward (7.20):

| term | final weight | share |
|---|---|---|
| `action_rate_l2` | −0.4 (v6 −0.1) | 5.6% |
| `action_acc_l2` | −0.2 | 4.9% |
| `hip_knee_joint_acc` (hip pitch/roll/yaw, knee) | −4e-5 | 3.0% |
| `hip_knee_torque_rate` (same joints) | −2.5e-8 | 2.9% |
| `hip_yaw_action_rate` (extra) | −3.5 | 2.8% |
| `hip_yaw_vel` | −0.14 | 1.5% |

A guessed torque-rate weight of −1e-6 would have cost 116% of the reward. Calibrate, don't guess.

**Acceptance gates**, v7 against v6, all in sim2sim:
- mean |Δa| ≤ 0.08;
- |Δ²a| ≤ 0.056;
- hip-yaw |Δ²a| ≤ 0.03;
- hip/knee target jump max ≤ 12°;
- torque-rate p95 ≤ 600 (hip pitch) / 1200 (knee) N·m/s;
- 3–8 Hz share standing ≤ 0.10;
- with **no regression** in stand foot drift (≤ 10 mm, 0 steps), falls (0) or vx 1.0 speed (≥ 0.82 m/s).

## Open items

- **`armature` is PROVISIONAL.** `J·ratio²` assumes the vendor's published
  "Inertia" is motor-side, which `ACTUATOR_DATASHEET.txt` note 2 says is never
  stated; if it is output-side these are ~400× too large. Hedged with
  `joint_armature` DR at `(0.6, 1.4)`. Settle it with a free-swing pendulum-decay
  on one X12 joint, then add a `_V2` group rather than editing in place.
- **A policy trained here is not plant-compatible with an H1-V1 policy** (armature
  differs 1.4–11×).
- **Four-bar doubling is optimistic on the roll axis**, where the two motors oppose.
- **`head_pitch`'s actuator is inferred**, not documented. Pinned, so inert.
- **Which IMU the deployed state estimator uses is unconfirmed.** The sim uses a
  pelvis-frame IMU (`imu_in_pelvis`); check `hardware.yaml` before trusting a
  transferred policy. `plant_report.py` prints the sim sign conventions.
- **No thermal model exists.** `effort_limit` is the 3–10 s peak rating; the
  `torque_duty_frac` / `torque_frac_of_continuous` metrics are logged every run so
  a gait the hardware cannot hold is visible.
