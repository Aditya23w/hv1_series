#!/usr/bin/env python3
"""Scroll the HV1V2 joints through their limits by hand with the viewer sliders.

Kinematic only (no physics): every frame the 31 Control sliders are copied
straight into the joint angles and mj_forward is run, so a slider IS the joint
angle. The actuator ctrlranges equal the joint ranges, so each slider's ends
are exactly that joint's limits. The pelvis is held in the air so the feet
never touch the floor. Self-collisions between collision hulls are printed and
shown on screen as they appear.

  python3 mjcf/sweep_limits.py              # start from the "home" pose
  python3 mjcf/sweep_limits.py --key stand  # start from the URDF zero pose

In the viewer: Tab opens the left panel, Shift+Tab the right one; the joint
sliders are under "Control". Backspace zeroes every slider (the "stand" pose).
"""
import argparse, math, os, time
import mujoco
import mujoco.viewer
import numpy as np

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hv1v2.xml")

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=XML, help="MJCF to load (default: viewer model)")
ap.add_argument("--key", default="home", help="keyframe to start from (stand|home)")
ap.add_argument("--height", type=float, default=1.3, help="pelvis height, keeps the feet off the floor")
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(args.xml)
data = mujoco.MjData(model)

k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.key)
if k < 0:
    raise SystemExit(f"no keyframe {args.key!r}; have: " + ", ".join(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)))
base = model.key_qpos[k][:7].copy()
base[2] = args.height

# Each slider writes the joint its position actuator drives.
qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
assert all(np.allclose(model.actuator_ctrlrange[a], model.jnt_range[model.actuator_trnid[a, 0]])
           for a in range(model.nu)), "a slider range differs from its joint limits"
data.ctrl[:] = model.key_qpos[k][qadr]

def self_contacts():
    """(geom1, geom2, depth_mm) for every robot-robot contact at the current pose."""
    return [(model.geom(c.geom1).name, model.geom(c.geom2).name, -c.dist * 1e3)
            for c in data.contact[:data.ncon]
            if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]  # not the floor

active = set()
with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.0, 140, -10
    v.cam.lookat[:] = [0, 0, args.height - 0.15]
    while v.is_running():
        with v.lock():
            data.qpos[:7] = base
            data.qpos[qadr] = data.ctrl
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            hits = self_contacts()
        pairs = {(a, b) for a, b, _ in hits}
        for a, b, depth in hits:
            if (a, b) not in active:
                print(f"contact: {a} x {b} ({depth:.1f} mm)")
        active = pairs
        # set_texts takes the viewer lock itself; calling it under v.lock() deadlocks.
        v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "\n".join(f"{a} x {b}  {d:.1f} mm" for a, b, d in hits) or "no self-contact",
                     None))
        v.sync()
        time.sleep(1 / 60)
