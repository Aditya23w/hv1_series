# HV1V2 — MuJoCo model

## Two models, one generator

| target | output | for |
|---|---|---|
| `--target viewer` (default) | `mjcf/hv1v2.xml` | looking at the robot; floor, lights, skybox, placeholder servos, 2 keyframes |
| `--target robot` | `../mjlab_hv1v2/src/mjlab_hv1v2/assets/hv1v2/hv1v2_robot.xml` | mjlab training; no floor/lights/actuators/keyframes, foot pads, IMU site, the sensors mjlab needs |

```bash
python mjcf/urdf2mjcf.py --target viewer
python mjcf/urdf2mjcf.py --target robot --copy-meshes
```

The training model must carry **zero keyframes**: mjlab's `Scene._add_entities()`
takes `spec.keys[0]` and only warns about the rest, so a leftover keyframe would
silently replace the configured initial state. It also drops the floor plane and
lights, which would otherwise be attached into every training env.

`hv1v2.xml` is generated from `urdf/HV1V2_URDF_NEW_FOOT_V1_220926.urdf` and loads
the STLs from `../meshes` directly. Nothing in the URDF tree was re-authored by
hand: link poses, joint axes, joint limits, masses and inertia tensors are copied
across verbatim and verified to match the URDF's forward kinematics to 1e-10.

## Viewing

```bash
python3 mjcf/view.py            # arms-down "home" pose, physics running
python3 mjcf/view.py --key stand --static
python3 -m mujoco.viewer --mjcf=mjcf/hv1v2.xml
```

## What the model contains

| | |
|---|---|
| Bodies | 34 links + world, floating base at `pelvis` (`freejoint` "floating_base") |
| Joints | 31 hinges; `pelvis_imu_link` and `torso_imu_link` are fixed to their parents |
| Mass | 92.682 kg |
| Standing height | pelvis at 0.99453 m, top of head 1.841 m |
| Actuators | 31 `position` servos, one per joint, named after the joint |
| Sensors | framequat / gyro / accelerometer at both IMU sites |
| Keyframes | `stand` (URDF zero pose), `home` (elbows folded so the arms hang) |

Geom groups: **2** = visual meshes, **3** = collision hulls, toggled with the
number keys in the viewer.

## Deviations from the URDF, and why

Everything here is an addition the URDF could not supply, not a change to it.

- **`meshes/torso_link_mjcf.STL`** — MuJoCo's STL reader rejects meshes over
  200k faces and `torso_link.STL` has 233,472. `decimate_stl.py` produced a
  187,771-face copy by 0.66 mm vertex clustering; the bounding box is identical
  to the original. The source mesh is untouched and still what the URDF uses.
- **Collision geometry** is the convex hull of each link mesh, capped at 64
  hull vertices. Real links are not convex, so hulls of a link and its
  *grandparent* interpenetrate at rest; `<contact><exclude>` drops those pairs
  (MuJoCo already drops parent-child). `pelvis`/`torso_link` is excluded too,
  because the torso hull fills in the concave waist and clips the pelvis once
  the waist joints leave zero. No self-collisions remain in either keyframe.
- **Joint damping / armature / friction loss** — the URDF declares no
  `<dynamics>`, so `damping="1" armature="0.01" frictionloss="0.1"` are
  placeholders chosen to make the viewer well-behaved. Replace them with real
  actuator values before using this for control work.
- **Actuator gains** — `kp = 40 x effort`, `kv = 1.2 x effort`, taken from each
  joint's URDF effort limit. These are stiff enough that the robot holds either
  keyframe under gravity without a balance controller. `forcerange` is the URDF
  effort limit verbatim. The URDF velocity limits are *not* applied.
- **Floor, lights, skybox, materials** — viewing scaffolding. Link colours come
  from the URDF `<material>` rgba values, deduplicated into 10 materials.

## Regenerating

```bash
python3 mjcf/urdf2mjcf.py    # re-reads the URDF, rewrites hv1v2.xml
```

Keyframe pelvis heights are not hardcoded: the script compiles the model, finds
the lowest mesh vertex in each pose, and writes the height that puts the soles
on the floor.

## Notes on the URDF worth knowing

- The hip-pitch joint origins carry large `rpy` values (`2.618 0 -3.1416` on the
  right, `-0.5236 0 0` on the left) that the hip-roll origins cancel out. So
  `right_hip_pitch_link` and `left_hip_pitch_link` sit in rotated frames at the
  zero pose, and their joint axes are *not* world-aligned there
  (`right_hip_pitch_joint` points along `[0, 0.866, 0.5]`). Everything from
  `hip_roll` down is axis-aligned again. `head_yaw`/`head_pitch` do the same
  trick with a 1.7815 rad yaw. This is a SolidWorks export artifact, faithfully
  preserved.
- At the zero pose the forearms point straight forward, not down — hence the
  `home` keyframe.
