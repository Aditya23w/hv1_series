#!/usr/bin/env python3
"""Play a bundled HV1V2 policy in plain MuJoCo. Needs only mujoco, numpy, onnxruntime.

Physics comes from the compiled mjlab training plant (same gains, armature,
frictionloss, effort limits, foot pads, init_state keyframe):
  v5   01_robot/mjcf/hv1v2_sim2sim_scene.xml      (v1-v8 plant: old knee, continuous caps)
  v11  01_robot/mjcf/hv1v2_v11_sim2sim_scene.xml  (corrected knee, peak caps, waist yaw 400/24) Everything about the policy -- joint order, anchor,
action scale, observation layout, history -- is read from the bundle's
deploy.json, exactly as a robot runtime would.

v11 is decoupled: it drives only the 12 leg joints and also observes the
upper-body target and its rate. Here the upper body holds its home pose (target
rel. home = 0, rate = 0), and leg targets are linearly interpolated across the 4
physics substeps and clipped to the joint range, as merai_control does and as
v11 trained.

  python tools/play_policy.py                      # v5, viewer
  python tools/play_policy.py --policy v11
  python tools/play_policy.py --headless --vx 1.0 --seconds 10

Viewer keys: Up/Down vx +-0.1, Left/Right wz +-0.1, , / . vy +-0.1,
             Space zero command, Backspace reset.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "01_robot" / "mjcf" / "hv1v2_sim2sim_scene.xml"
SCENE_V11 = ROOT / "01_robot" / "mjcf" / "hv1v2_v11_sim2sim_scene.xml"
GRAVITY_W = np.array([0.0, 0.0, -1.0])


class Policy:
  def __init__(self, bundle: Path):
    self.cfg = json.loads((bundle / "deploy.json").read_text())
    self.all_joints = [j["name"] for j in self.cfg["joints"]]
    act = self.cfg["action"]
    self.act_joints = act.get("joint_names", self.all_joints)
    anchor = self.cfg["anchor"]["default_joint_pos"]
    self.anchor_all = np.array([anchor[n] for n in self.all_joints])
    self.act_idx = np.array([self.all_joints.index(n) for n in self.act_joints])
    self.scale = np.array([act["scale"][n] for n in self.act_joints])
    p = self.cfg["policy"]
    self.terms = p["observation"]
    self.obs_dim = int(p["obs_dim"])
    self.in_name = p["input_name"]
    self.sess = ort.InferenceSession(str(bundle / p["file"]),
                                     providers=["CPUExecutionProvider"])
    self.ranges = self.cfg["commands"]["trained_ranges"]
    self.v11 = any(t["name"] == "ub_target" for t in self.terms)
    n_ub = len(p.get("ub_obs_joints", []))
    self.ub_rel = np.zeros(n_ub)   # upper body held at home
    self.ub_rate = np.zeros(n_ub)
    self.reset()

  def reset(self):
    self.prev = np.zeros(len(self.act_joints))
    self.hist: dict[str, deque] = {}

  def __call__(self, ang_vel, grav, q, qd, cmd) -> np.ndarray:
    frames = {"base_ang_vel": ang_vel, "projected_gravity": grav,
              "joint_pos": q - self.anchor_all, "joint_vel": qd,
              "actions": self.prev, "command": cmd,
              "ub_target": self.ub_rel, "ub_target_rate": self.ub_rate}
    out = []
    for t in self.terms:
      f = np.asarray(frames[t["name"]], float)
      h = int(t["history"])
      if h <= 1:
        out.append(f)
        continue
      buf = self.hist.get(t["name"])
      if buf is None:  # backfill with the first reading, never zeros
        buf = self.hist[t["name"]] = deque([f.copy()] * h, maxlen=h)
      else:
        buf.append(f)
      out.append(np.concatenate(buf))  # term-major, oldest -> newest
    obs = np.concatenate(out).astype(np.float32)
    assert obs.size == self.obs_dim, (obs.size, self.obs_dim)
    self.prev = self.sess.run(None, {self.in_name: obs[None]})[0][0].astype(float)
    target = self.anchor_all.copy()
    target[self.act_idx] += self.prev * self.scale
    return target


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", default="v5", help="folder name under 04_policies")
  ap.add_argument("--vx", type=float, default=0.0)
  ap.add_argument("--vy", type=float, default=0.0)
  ap.add_argument("--wz", type=float, default=0.0)
  ap.add_argument("--headless", action="store_true")
  ap.add_argument("--seconds", type=float, default=10.0)
  args = ap.parse_args()

  pol = Policy(ROOT / "04_policies" / args.policy)
  m = mujoco.MjModel.from_xml_path(str(SCENE_V11 if pol.v11 else SCENE))
  d = mujoco.MjData(m)
  act_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
                for a in range(m.nu)]
  assert act_joints == pol.all_joints, "scene actuator order != deploy.json joint order"
  jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in pol.all_joints]
  qadr = np.array([m.jnt_qposadr[j] for j in jid])
  vadr = np.array([m.jnt_dofadr[j] for j in jid])
  pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  gs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
  gadr = m.sensor_adr[gs]
  key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  dec = int(pol.cfg["control"]["decimation"])
  lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
  prev_ctrl = [None]
  cmd = np.array([args.vx, args.vy, args.wz])

  def clamp():
    for i, k in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
      cmd[i] = np.clip(cmd[i], *pol.ranges[k])

  def reset():
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    pol.reset()
    prev_ctrl[0] = None

  def control_step():
    R = d.xmat[pelvis].reshape(3, 3)
    ctrl = pol(d.sensordata[gadr:gadr + 3].copy(), R.T @ GRAVITY_W,
               d.qpos[qadr].copy(), d.qvel[vadr].copy(), cmd)
    if not pol.v11:                      # v1-v10: zero-order hold
      d.ctrl[:] = ctrl
      for _ in range(dec):
        mujoco.mj_step(m, d)
      return
    ctrl[pol.act_idx] = np.clip(ctrl[pol.act_idx], lo[pol.act_idx], hi[pol.act_idx])
    if prev_ctrl[0] is None:
      prev_ctrl[0] = ctrl.copy()
    for s in range(dec):                 # v11: linear interpolation
      d.ctrl[:] = prev_ctrl[0] + (s + 1) / dec * (ctrl - prev_ctrl[0])
      mujoco.mj_step(m, d)
    prev_ctrl[0] = ctrl

  clamp()
  reset()
  meta = pol.cfg["meta"]
  print(f"{args.policy}: run {meta['run']}, {meta['iterations']} iters, "
        f"obs {pol.obs_dim}, actions {len(pol.act_joints)}")
  print(f"trained command ranges: {pol.ranges}")

  if args.headless:
    n = int(args.seconds * 50)
    for _ in range(n):
      control_step()
    dist = float(np.linalg.norm(d.qpos[:2]))
    fell = d.qpos[2] < 0.45
    print(f"cmd {cmd.tolist()}: travelled {dist:.2f} m in {args.seconds:.0f} s "
          f"({dist / args.seconds:.2f} m/s), pelvis z {d.qpos[2]:.3f}, "
          f"{'FELL' if fell else 'standing'}")
    return


  def on_key(k):
    step = {265: (0, 0.1), 264: (0, -0.1), 263: (2, 0.1), 262: (2, -0.1),
            44: (1, 0.1), 46: (1, -0.1)}
    if k in step:
      i, s = step[k]
      cmd[i] += s
      clamp()
    elif k == 32:
      cmd[:] = 0
    elif k == 259:
      reset()
    print(f"\rcmd vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f}   ", end="", flush=True)

  with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
    v.cam.trackbodyid = pelvis
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.distance = 3.0
    while v.is_running():
      t0 = time.time()
      control_step()
      if d.qpos[2] < 0.45:
        reset()
      v.sync()
      time.sleep(max(0.0, 0.02 - (time.time() - t0)))


if __name__ == "__main__":
  main()
