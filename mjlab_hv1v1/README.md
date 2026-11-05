# mjlab_hv1v1

HV1V1 humanoid — the **older-foot** variant of the same body as HV1V2 — for
[mjlab](https://github.com/mujocolab/mjlab) 1.6.0.

```bash
conda activate hv1v2_mjlab        # one env hosts mjlab and BOTH robot packages
list-envs | grep HV1V1
./scripts/train.sh
```

| Task | Actor sees | Use |
|---|---|---|
| `Mjlab-Velocity-Flat-HV1V1` | IMU + encoders + actions + command | **primary** — deployable |
| `Mjlab-Velocity-Flat-HV1V1-State-Estimation` | + base linear velocity | baseline / ablation |
| `Mjlab-Velocity-Rough-HV1V1` | + terrain height scan | rough terrain |

> **Naming.** The source export is `URDF_HV1V2_26092026/`, but the foot geometry
> identifies it as the old-foot robot. The filename is misleading; the foot is not.

## How HV1V1 differs from HV1V2 — only the feet

Verified against both URDFs: every joint name, **limit, velocity, origin xyz and rpy**, and
the entire actuator map are **identical**. Every link mass matches within 1% except the feet.

| | HV1V2 | HV1V1 |
|---|---|---|
| total mass | 92.6824 kg | **94.3088 kg** (+1.63, all feet) |
| foot mass each | 0.9691 kg | **1.7823 kg** (+84%) |
| foot CoM ahead of ankle | 29.5 mm | **65.9 mm** |
| foot inertia about ankle | 0.00535 kg·m² | **0.01879** (3.5×) |
| sole about the ankle | −95.9 … +167.1 mm | **−62.0 … +219.5 mm** |
| ankle along the sole | 36.5% from heel | **22.0% from heel** |
| support centre ahead of ankle | +35.6 mm | **+78.7 mm** |

**The plant is therefore HV1V2's, unchanged** — same five actuator groups, same gains,
same action scales, same 11 pinned joints.

## Why HV1V2's policy walked backwards here

HV1V2's anchor pose puts the CoM over *HV1V2's* support centre. HV1V1's support centre is
**43.1 mm further forward**, so the same joint angles leave the CoM **42.1 mm behind** it —
the robot stands on its heels, pitches back, and forward drive nets backward. A full-FK
solve says the ankle-pitch offset that re-centres it is **−2.70°**, which is the ~2.5° that
had to be dialled in by hand.

**The pose below removes the need for it.** It is not a property of the robot; it was a
symptom of reusing a pose solved for a different foot.

## Initial pose — solved against an ankle-torque budget

HV1V2 places its CoM over the sole centre. Doing that here would put the CoM 78.7 mm ahead
of the ankle and load the ankles to ~42% of continuous by the static measure (and
`plant_report.py` measures roughly 2× the static figure). So HV1V1 targets the **ankle duty**
instead — the rule that reproduces HV1V2's pose exactly and generalises:

> CoM at the x that loads the ankles to **18.8% of continuous**, rear margin ≥ 90 mm.

| | value |
|---|---|
| CoM ahead of the ankle | **+34.6 mm** (target 35.0) |
| ankle duty, static | 18.6% (HV1V2: 18.8%) |
| ankle duty, **measured** by `plant_report.py` | **35.8%** (HV1V2: 37.9%) |
| margins | 96.6 mm rear / 185.2 mm front |
| pelvis height | 0.97698 m |

Margins are asymmetric because the foot only has 62 mm of sole behind the ankle. That is
the foot, not the pose. The left/right leg values also differ — the hip-pitch axes are
canted ~30° and the legs are not mirror images, exactly as on HV1V2.

## Foot pads

Four box pads per foot, fitted to the two measured contact patches (sole plane z = −0.0374):

| patch | left | right |
|---|---|---|
| heel — **straddles the ankle** | x −0.0620 … +0.0320 | −0.0617 … +0.0323 |
| toe | x +0.1140 … +0.2195 | +0.1143 … +0.2198 |

Unlike HV1V2, where the ankle sits in the arch gap, HV1V1's ankle sits *on* the rear pad.

## Verification

```bash
python scripts/check_model.py            # compiled plant, action map, obs widths
python scripts/solve_init_pose.py --check # pose, ankle duty, margins, self-contacts
python scripts/plant_report.py           # step response, PD hold, torque duty, sensors
RUN=smoke ITERS=200 NUM_ENVS=512 ./scripts/train.sh
```

Measured at build time:

| | |
|---|---|
| unloaded step response | rise 0.136–0.194 s, 0% overshoot, ≤2 mrad error |
| holding the init pose | **35.8%** of continuous (ankle pitch) |
| open-loop PD hold | steady ~6.2 s then topples (HV1V2: 3.8 s) |
| actor / critic obs | 498 / 117 |

**The open-loop fall is expected**, as on HV1V2 — ankle stiffness is close to what the
inverted pendulum demands, so nothing holds it up without a policy.

## Open items

Inherited from HV1V2 and equally unresolved here:

- **`armature` is provisional** — vendor `J·ratio²` assuming a motor-side inertia
  convention the vendor never states. Hedged with DR `(0.6, 1.4)`.
- **Four-bar amplification is unmeasured for HV1V1.** The ankles and waist roll/pitch are
  parallel four-bars; the linkage amplifies commanded gain, so the sim `kp = 400` must not
  be written to the motors unmeasured. Nothing may be inherited from HV1V2's figures.
- **No thermal model.** `effort` is the vendor PEAK (a 3–10 s rating).

Specific to HV1V1:

- **Swing-leg inertia is 3.5× HV1V2's** on the same ankle actuator. Foot-clearance and
  swing-height rewards were tuned on the lighter foot; 0.08 m is a starting point.
- **Only 62 mm of heel lever** (vs 95.9) — far less heel-strike braking, far more toe-off.
  Expect a different gait, and watch for backward drift as the signature of the CoM
  sitting too far back.
