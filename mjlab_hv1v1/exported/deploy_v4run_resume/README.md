# HV1V1 deployment bundle — run `v4run_resume`, iteration 39999

Everything needed to run this policy on the robot. Nothing here is inherited from
HV1V2: every number was measured on HV1V1's own model and its own trained policy.

| file | |
|---|---|
| `policy.onnx` | the graph — 498 inputs, 31 outputs |
| `deploy.json` | every parameter below, machine-readable |
| `validate_deploy.py` | reference implementation; drives the policy from `deploy.json` alone |
| `_joint_table.md` | per-joint anchor, scale, gains, limits, in ctrl order |
| `model_39999.pt` | the training checkpoint it was exported from |
| `checksums.txt` | sha256 of each file |

**Bundle verified**: `python validate_deploy.py --vx 1.0` reads nothing but
`deploy.json` and the ONNX — no mjlab config in the loop — and walks at
**0.86 m/s for 8 s without falling**. If you change anything, re-run it.

## Control loop

| | |
|---|---|
| policy rate | **50 Hz** (`control_dt` 0.02 s) |
| sim timestep | 0.005 s, decimation 4 |
| root body | `pelvis` |
| robot mass | 94.309 kg |

Each tick: assemble the observation, run the graph, map
`target[i] = anchor[i] + raw_action[i] * action_scale[i]`, write joint PD targets.

## Observation — 498, term-major, oldest first

| term | dim | history | contributes | source on robot |
|---|---|---|---|---|
| `base_ang_vel` | 3 | 5 | 15 | IMU angular velocity, **pelvis frame**, rad/s |
| `projected_gravity` | 3 | 5 | 15 | `R_pelvis^T @ [0,0,-1]` — a **unit** vector, not 9.81 |
| `joint_pos` | 31 | 5 | 155 | `q - anchor` |
| `joint_vel` | 31 | 5 | 155 | `qdot` |
| `actions` | 31 | 5 | 155 | previous **raw** output, before `action_scale` |
| `command` | 3 | 1 | 3 | operator `[vx, vy, wz]` |

Layout is **term-major**: each term's whole 5-frame history is contiguous, then
terms concatenate. Within a term, **oldest frame first**.

At `t=0` **backfill every history slot with the first reading. Do not zero-fill** —
zeros are not a valid observation and the policy has never seen them.

There is **no base linear velocity** in the actor. That is deliberate: it would
need a state estimator this robot does not have.

## READY pose

Servo here with joint PD, then hand over to the policy. Identical to the anchor,
so the handover is bumpless. Values in `_joint_table.md`; pelvis height 0.97697 m,
feet flat, stance width 0.24304 m.

**This pose does not stand by itself.** Held with PD and no policy it topples in
~6.2 s (to a 50 mm drop). That is not a defect specific to HV1V1 — HV1V2's own
anchor manages 3.9 s. Ankle stiffness is ~95% of what the inverted pendulum
demands, so **keep the policy running**.

### Why the CoM is not over the sole centre

HV1V2 puts its CoM at 50.0% along its sole. HV1V1's ankle sits at **22.0% of the
sole** instead of 36.5%, so centring the CoM would require it **78.7 mm ahead of
the ankle** — and that pose diverges open-loop in **0.89 s**.

This pose is at **34.4% of the sole, +34.7 mm ahead of the ankle**, i.e. 44.2 mm
*behind* the sole centre. It measured the slowest open-loop creep of every
candidate tested:

| CoM position | creep | open-loop hold |
|---|---|---|
| **34.4% (this pose)** | **−0.21 mm/s** | **6.2 s** |
| 41.3% | −19.85 mm/s | 1.35 s |
| 49.8% (sole centre) | −47.32 mm/s | 0.89 s |
| HV1V2's own anchor, for scale | −0.83 mm/s | 3.9 s |

### About the 2.5° ankle offset

If you previously ran an **HV1V2** policy on this robot you needed roughly +2.5°
of ankle-pitch offset to stop it walking backwards. **Do not apply that here.**
It was a policy-transfer artifact: HV1V2's angles land at 34.6% of *this* robot's
sole while its policy expects 50%, a ~43 mm foot-relative mismatch. This policy
was trained from scratch on HV1V1 and has no mismatch to correct. The offset is
not in the anchor and must not be added.

## Commands

| | min | max |
|---|---|---|
| `lin_vel_x` | −1.2 | 1.5 |
| `lin_vel_y` | −1.0 | 1.0 |
| `ang_vel_z` | −0.7 | 0.7 |

**Do not command outside these.** The policy has never seen it.

Measured response (plain MuJoCo, this bundle):

| cmd vx | actual | tracking | cadence | step length | foot lift |
|---|---|---|---|---|---|
| 0.1–0.3 | **0.00** | **0%** | — | — | — |
| 0.5 | 0.41 | 82% | 1.01 Hz | 41 cm | 70 mm |
| 0.8 | 0.72 | 90% | 1.13 Hz | 64 cm | 81 mm |
| 1.0 | 0.91 | 91% | 1.13 Hz | 80 cm | 83 mm |
| 1.4 | 1.25 | 89% | 1.26 Hz | 99 cm | 93 mm |

**Known: there is a hard low-speed dead zone.** Commands at or below ~0.3 m/s
produce **zero** motion, not slow motion. Above 0.5 m/s tracking is 82–91%. If the
operator interface allows small commands, either clamp them to 0 or step straight
to 0.5.

Stopping from 1.0 m/s takes **2 steps and settles in 0.94 s**. One step is
physically unavoidable — the capture-point estimate at that speed is ~0.28 m.

## Four-bar joints — READ BEFORE WRITING GAINS

`left/right_ankle_pitch`, `left/right_ankle_roll`, `waist_roll`, `waist_pitch`.

Each is a parallel four-bar driven by **two X8-120s**. The linkage **amplifies**
commanded joint gain, so the simulation `kp` of **400** on these joints is the
doubled value and writing it straight to the motors will over-stiffen them.

**The amplification factor has never been measured on HV1V1.** `deploy.json`
carries H1-V1's factors (ankle_pitch 2.41, ankle_roll 1.23, waist_roll 4.00,
waist_pitch 4.81) **as a starting point for your own measurement, not as values to
use.** Measure them on this robot and divide before deploying. `waist_yaw` is a
passthrough and needs no correction.

Every ankle torque figure quoted anywhere in this bundle assumes 1:1 transmission
and is wrong by exactly this unmeasured factor.

## Pinned joints

11 of the 31 outputs have `action_scale = 0.0` — waist ×3, wrists ×6, head ×2. The
policy cannot move them; hold them at the anchor. Their raw network output is
**unconstrained and can be large** — it is multiplied by zero, so it is harmless,
but do not log it as a torque demand or feed it anywhere.

## Known limits

- **Low-speed dead zone** below ~0.3 m/s, above.
- **Slip is ~35% higher than HV1V2's** (0.087 vs 0.064 m/s mean at contact). Watch
  for it on low-friction floors; training used friction 0.3–1.2.
- **Ankle four-bar amplification unmeasured**, above.
- **`armature` is provisional** — vendor `J·ratio²` assuming the published inertia
  is motor-side, which the datasheet never states. Training hedged it with ±40%
  domain randomisation, so the policy should tolerate the uncertainty, but the
  number itself is not confirmed.
- **Trained on flat ground only.** No stairs, no slopes, no terrain.
- **Push robustness is untested on this policy.** It trained against velocity
  perturbations ramped to ±0.80 m/s (HV1V2 trained at a fixed ±0.5), but the
  recovery ladder measured on HV1V2 has not been repeated here.

## Per-joint table

See `_joint_table.md` — index, actuator, anchor, action scale, commandable window,
joint limits, kp/kd and peak torque, in **ctrl order**. Index 0 is the first of the
policy's 31 outputs. Do not reorder.
