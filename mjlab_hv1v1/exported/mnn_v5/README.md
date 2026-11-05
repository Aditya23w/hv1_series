# HV1V1 MNN deployment — run `v5`, iteration 39999

The policy, converted to MNN, plus every parameter the robot needs. Nothing here
is inherited from HV1V2: all values were measured on HV1V1's own model and its
own trained policy, and all configs are generated from `deploy.json` by
`mjlab_hv1v1/scripts/export_mnn.py` rather than hand-maintained.

| file | |
|---|---|
| `walk_hv1v1_v5.mnn` | **the model the robot runs** — 498 in, 31 out |
| `policy_service.hv1v1.yaml` | runtime config: obs layout, anchors, gains, limits |
| `ready_pose.hv1v1.yaml` | the pose to servo to before handing over |
| `deploy.json` | the full machine-readable source all of the above derives from |
| `policy.onnx` | the source graph, kept for re-conversion and verification |
| `sim_obs.npz` | 800 real sim2sim frames across 8 commands, for `verify_mnn.py` |
| `verify_mnn.py` | checks the MNN still matches the ONNX numerically |
| `_joint_table.md` | per-joint anchor, scale, gains, limits, in ctrl order |
| `checksums.txt` | sha256 of every file |

## ⚠️ Set the robot's ankle offset to ZERO

This policy was **trained with +2.7° of ankle plantarflexion already in the
anchor** (`left_ankle_pitch −0.19821`, `right_ankle_pitch −0.22666`). That is the
offset you previously applied on the robot to get the weight onto the toes.

**It is now in the policy. Leave the hardware offset on and the robot gets 5.4°.**

This is also why the policy is expected to behave differently from the run-4
policy, which was trained without it and only balanced when the offset was
applied externally.

## Verification — run this before deploying

```bash
python verify_mnn.py            # ONNX vs MNN on the recorded rollout
python verify_mnn.py --random   # plus a random-input stress test
```

Measured on this conversion:

| | max \|onnx − mnn\| | worst joint error |
|---|---|---|
| 800 real sim2sim frames | **1.14e-05** | **0.00005°** |
| 256 random N(0,1) inputs | 7.82e-05 | 0.00056° |

Both are orders of magnitude below the robot's encoder resolution. Re-run it if
you ever re-convert — conversion reporting "Converted Success" says nothing about
whether the numbers agree.

## Control loop

| | |
|---|---|
| policy / target rate | **50 Hz** |
| **joint PD rate** | **≥ 200 Hz — see below** |
| root body | `pelvis` |
| robot mass | 94.309 kg |

Each tick: assemble the observation, run the MNN, map
`target[i] = anchor[i] + action[i] * action_scale[i]`, write joint PD targets.

### The PD must run faster than the policy

In simulation the PD is evaluated every physics step (200 Hz) while the policy
only refreshes the target every 4th step (50 Hz). Running the PD at the policy's
50 Hz is **not** the same system. Measured in sim by dropping the PD to 50 Hz:
the robot falls in 2.3 s with 8× the torque chatter. 100 Hz is already fine, 200
Hz matches training.

Send targets at 50 Hz; let the motor controllers close the loop at 200 Hz or more.

## Observation — 498, term-major, oldest first

| term | dim | history | contributes | source on robot |
|---|---|---|---|---|
| `base_ang_vel` | 3 | 5 | 15 | IMU angular velocity, **pelvis frame**, rad/s |
| `projected_gravity` | 3 | 5 | 15 | `R_pelvis^T @ [0,0,-1]` — a **unit** vector, not 9.81 |
| `joint_pos` | 31 | 5 | 155 | `q − anchor` |
| `joint_vel` | 31 | 5 | 155 | `qdot` |
| `actions` | 31 | 5 | 155 | previous **raw** output, before `action_scale` |
| `command` | 3 | 1 | 3 | operator `[vx, vy, wz]` |

Each term's whole 5-frame history is contiguous, then terms concatenate. Within
a term, **oldest frame first**.

At `t=0` **backfill every history slot with the first reading. Do not zero-fill** —
zeros are not a valid observation and the policy has never seen them.

There is **no base linear velocity** in the actor, deliberately: it would need a
state estimator this robot does not have.

## Commands

`policy_service.hv1v1.yaml` ships a **subset** of the trained envelope on purpose —
the robot has not walked its full range, and a command the policy has never seen
is the cheapest way to put it on the floor.

| | shipped | trained |
|---|---|---|
| `lin_vel_x` | −0.4 … 0.8 | −1.2 … 1.5 |
| `lin_vel_y` | −0.3 … 0.3 | −1.0 … 1.0 |
| `ang_vel_z` | −0.4 … 0.4 | −0.7 … 0.7 |

Widen once it walks. Do not exceed the trained column.

**Known: there is a low-speed dead zone.** Commands at or below ~0.3 m/s produce
*zero* motion, not slow motion. Either clamp small commands to 0 or step straight
to 0.5.

## Four-bar joints — READ BEFORE WRITING GAINS

`left/right_ankle_pitch`, `left/right_ankle_roll`, `waist_roll`, `waist_pitch`.

Each is a parallel four-bar driven by **two X8-120s**. The `kp` of **400** in the
YAML is the joint-level, already-doubled simulation value. Writing it per motor
is wrong twice over:

| joint | yaml kp | ÷2 per module | ÷ amplification | ≈ motor kp |
|---|---|---|---|---|
| `ankle_pitch` | 400 | 200 | 2.41 | **83** |
| `ankle_roll` | 400 | 200 | 1.23 | **163** |
| `waist_roll` | 400 | 200 | 4.00 | **50** |
| `waist_pitch` | 400 | 200 | 4.81 | **42** |

**Those amplification factors are H1-V1's and have never been measured on HV1V1.**
They are a starting point for your own measurement, not values to use. Measure
them on this robot and divide before deploying. `waist_yaw` is a passthrough and
needs no correction.

Every ankle torque figure anywhere in this bundle assumes 1:1 transmission and is
wrong by exactly this unmeasured factor. The 2.7° anchor bias is itself an
ankle-torque effect, so it runs through the same unknown.

## Pinned joints

11 of the 31 outputs have `action_scale = 0.0` — waist ×3, wrists ×6, head ×2.
Hold them at the anchor. Their raw network output is **unconstrained and can be
large**; it is multiplied by zero so it is harmless, but do not log it as a torque
demand or feed it anywhere.

## Measured behaviour

From the final policy in plain MuJoCo:

| metric | v5 (HV1V1) | HV1V2 v4, for scale |
|---|---|---|
| single support | 86.8% | 87.1% |
| landings/sec | 2.65 | 2.66 |
| `air_time_mean` | 0.175 | 0.176 |
| peak foot clearance | 12.3 mm | 7.5 mm |
| soft-limit violation | 0.0006 rad | 0.0005 rad |

## Known limits

- **Low-speed dead zone** below ~0.3 m/s, above.
- **Four-bar amplification unmeasured**, above.
- **`armature` is provisional** — vendor `J·ratio²` assuming the published inertia
  is motor-side, which the datasheet never states. Hedged with ±40% domain
  randomisation in training, but unconfirmed.
- **Flat ground only.** No stairs, no slopes, no terrain.
- **The simulation does not reproduce this robot's hardware behaviour.** An HV1V2
  policy walks on the real HV1V1 but stands still on the HV1V1 *model* at every
  ankle offset tried. Something in the model is wrong — foot geometry, sole plane,
  ankle placement or mass. Sim numbers for HV1V1 should not be trusted to predict
  hardware until that is understood.
- **Vibration was seen with the run-4 policy** on hardware and was never explained.
  Leading candidates are four-bar compliance with the 84%-heavier foot, and the
  unmeasured amplification above. If v5 vibrates too, start there.

## Rebuilding

```bash
python mjlab_hv1v1/scripts/export_deploy.py --run v5   # ONNX bundle
python mjlab_hv1v1/scripts/export_mnn.py    --run v5   # this folder
cd mjlab_hv1v1/exported/mnn_v5 && python verify_mnn.py
```

`export_mnn.py` regenerates every config from `deploy.json`, so the YAML and the
trained anchor cannot drift apart. Do not hand-edit the generated files.
