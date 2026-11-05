#!/usr/bin/env python3
"""Open the HV1V2 model in the MuJoCo viewer.

  python3 mjcf/view.py              # passive viewer, physics running, "home" pose
  python3 mjcf/view.py --key stand  # start from the URDF zero pose
  python3 mjcf/view.py --static     # no physics, just look at the model

In the viewer: press Tab for the control panel (the sliders drive the 31
position actuators), and use the Group keys 0-4 to toggle geom groups -
group 2 is the visual meshes, group 3 the collision hulls.
"""
import argparse, os, time
import mujoco
import mujoco.viewer

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hv1v2.xml")

ap = argparse.ArgumentParser()
ap.add_argument("--key", default="home", help="keyframe to start from (stand|home)")
ap.add_argument("--static", action="store_true", help="freeze physics")
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.key)
if k < 0:
    raise SystemExit(f"no keyframe {args.key!r}; have: " + ", ".join(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)))
mujoco.mj_resetDataKeyframe(model, data, k)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.0, 140, -10
    v.cam.lookat[:] = [0, 0, 0.85]
    while v.is_running():
        step_start = time.time()
        if not args.static:
            mujoco.mj_step(model, data)
        v.sync()
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
