#!/usr/bin/env python3
"""Run a trained HV1V1 policy in plain MuJoCo, outside the mjlab training loop.

What this actually checks: the observation vector and action mapping are
rebuilt BY HAND here from the ONNX metadata, rather than by mjlab's managers.
That is where deployment bugs live -- a term in the wrong order, a missing
default-pose offset, the wrong gravity frame, history stacked frame-major
instead of term-major. If the policy behaves here the way it did in training,
the hand-assembled pipeline is right.

The physics model is built from the same EntityCfg mjlab trains with, so the
plant (gains, armature, frictionloss, collisions, initial pose) is identical
and cannot confound the comparison. Physics runs on plain CPU MuJoCo at 200 Hz
with the policy at 50 Hz (decimation 4), matching the training config.

Everything about the policy is read from the ONNX metadata_props -- joint
order, default pose, per-joint action scale, observation term order and
per-term history length -- so this script works unchanged for v1 (no history)
and v3 (history 5).

  # look at it
  python mjlab_hv1v1/scripts/sim2sim.py --run v1
  # no window: step N policy steps and print what happened
  python mjlab_hv1v1/scripts/sim2sim.py --run v1 --headless 500
  # record
  python mjlab_hv1v1/scripts/sim2sim.py --run v1 --video /tmp/v1.mp4 --seconds 8

Viewer keys: W/S forward, A/D lateral, Q/E turn, X zero command, R reset.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort

LOG_ROOT = Path(__file__).resolve().parents[1] / "logs/rsl_rl/hv1v1_velocity"
SIM_DT = 0.005      # cfg.sim.mujoco.timestep
DECIMATION = 4      # -> 50 Hz policy
GRAVITY_W = np.array([0.0, 0.0, -1.0])  # mjlab's gravity_vec_w is a UNIT vector


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
class Policy:
  """ONNX policy plus the observation assembly described by its metadata."""

  def __init__(self, path: str):
    meta = {p.key: p.value for p in onnx.load(path).metadata_props}
    self.path = path
    self.joint_names = meta["joint_names"].split(",")
    self.default_qpos = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
    self.action_scale = np.array([float(x) for x in meta["action_scale"].split(",")])
    self.obs_names = meta["observation_names"].split(",")
    self.obs_hist = [int(float(x)) for x in meta["observation_terms_history_length"].split(",")]
    self.obs_scale = [float(x) for x in meta["observation_terms_scale"].split(",")]
    self.kp = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
    self.kd = np.array([float(x) for x in meta["joint_damping"].split(",")])

    self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    self.in_name = self.sess.get_inputs()[0].name
    self.in_dim = self.sess.get_inputs()[0].shape[-1]
    self.nu = len(self.joint_names)
    self.prev_action = np.zeros(self.nu)
    self._hist: dict[str, deque] = {}

  def term(self, name: str, mj, robot) -> np.ndarray:
    """One frame of a single observation term, in mjlab's own convention."""
    if name == "base_ang_vel":
      return robot.gyro()
    if name == "base_lin_vel":
      return robot.velocimeter()
    if name == "projected_gravity":
      # mjlab uses the ROOT BODY orientation (data.py:605), not the IMU site.
      return robot.root_rot().T @ GRAVITY_W
    if name == "joint_pos":
      return robot.qpos_joints() - self.default_qpos
    if name == "joint_vel":
      return robot.qvel_joints()
    if name == "actions":
      return self.prev_action
    if name == "command":
      return robot.command
    raise KeyError(f"sim2sim does not know how to build observation term {name!r}")

  def reset(self):
    self.prev_action = np.zeros(self.nu)
    self._hist.clear()

  def observe(self, mj, robot) -> np.ndarray:
    """Assemble the flat observation: term-major, oldest frame first."""
    out = []
    for name, h, scale in zip(self.obs_names, self.obs_hist, self.obs_scale):
      frame = np.asarray(self.term(name, mj, robot), dtype=np.float64) * scale
      if h <= 1:
        out.append(frame)
        continue
      buf = self._hist.get(name)
      if buf is None:
        # Backfill, do not zero-fill: mjlab fills every slot with the first
        # post-reset frame (circular_buffer.py:231). A zeroed history is
        # off-distribution and the first steps would be wrong.
        buf = deque([frame.copy() for _ in range(h)], maxlen=h)
        self._hist[name] = buf
      else:
        buf.append(frame)
      out.append(np.concatenate(list(buf)))  # oldest -> newest
    obs = np.concatenate(out)
    assert obs.size == self.in_dim, (
      f"assembled {obs.size} observations but the network takes {self.in_dim}. "
      "Term order or history stacking is wrong."
    )
    return obs

  def act(self, obs: np.ndarray) -> np.ndarray:
    a = self.sess.run(None, {self.in_name: obs.astype(np.float32)[None]})[0][0]
    self.prev_action = a.astype(np.float64)
    return self.prev_action

  def targets(self) -> np.ndarray:
    """mjlab's JointPositionAction: processed = raw * scale + default."""
    return self.default_qpos + self.prev_action * self.action_scale


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class Robot:
  def __init__(self, joint_names, for_video: bool):
    from mjlab.entity.entity import Entity
    from mjlab_hv1v1.hv1v1_constants import get_hv1v1_robot_cfg

    spec = Entity(get_hv1v1_robot_cfg()).spec
    spec.worldbody.add_geom(
      name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
      condim=3, friction=[0.8, 0.02, 0.01], rgba=[0.3, 0.32, 0.34, 1],
    )
    if for_video:
      spec.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.24, 0.3, 0.4], rgb2=[0.05, 0.06, 0.08], width=512, height=3072,
      )
    self.m = spec.compile()
    self.m.opt.timestep = SIM_DT
    self.d = mujoco.MjData(self.m)

    jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    assert all(i >= 0 for i in jid), "a policy joint name is missing from the model"
    self.qadr = np.array([self.m.jnt_qposadr[i] for i in jid])
    self.vadr = np.array([self.m.jnt_dofadr[i] for i in jid])
    # ctrl order must match the policy's joint order (sort_actuators=True)
    act_joint = [
      mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, self.m.actuator_trnid[a, 0])
      for a in range(self.m.nu)
    ]
    assert act_joint == list(joint_names), (
      "actuator order does not match the policy's joint order:\n"
      f"  model  {act_joint[:4]}...\n  policy {list(joint_names)[:4]}..."
    )
    self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    self._sens = {}
    for s in range(self.m.nsensor):
      self._sens[mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_SENSOR, s)] = (
        self.m.sensor_adr[s], self.m.sensor_dim[s])
    self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    self.command = np.zeros(3)

  def reset(self):
    mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
    mujoco.mj_forward(self.m, self.d)

  def _sensor(self, name):
    a, n = self._sens[name]
    return self.d.sensordata[a:a + n].copy()

  def gyro(self):
    return self._sensor("imu_ang_vel")

  def velocimeter(self):
    return self._sensor("imu_lin_vel")

  def root_rot(self):
    return self.d.xmat[self.pelvis].reshape(3, 3).copy()

  def qpos_joints(self):
    return self.d.qpos[self.qadr].copy()

  def qvel_joints(self):
    return self.d.qvel[self.vadr].copy()

  def fell(self):
    return self.d.qpos[2] < 0.45


# ---------------------------------------------------------------------------
def find_onnx(run: str) -> str:
  hits = sorted(glob.glob(str(LOG_ROOT / f"*_{run}" / "*.onnx")))
  if not hits:
    raise SystemExit(
      f"no .onnx for run {run!r} under {LOG_ROOT}\n"
      f"available: {sorted(p.name for p in LOG_ROOT.glob('*'))}"
    )
  return hits[-1]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default="v1", help="run-name suffix, e.g. v1 / v2 / v3")
  ap.add_argument("--onnx", default=None, help="explicit .onnx path (overrides --run)")
  ap.add_argument("--headless", type=int, default=0, help="policy steps, no window")
  ap.add_argument("--video", default=None, help="write an mp4 here instead of a window")
  ap.add_argument("--seconds", type=float, default=8.0, help="video length")
  ap.add_argument("--vx", type=float, default=0.5)
  ap.add_argument("--vy", type=float, default=0.0)
  ap.add_argument("--wz", type=float, default=0.0)
  args = ap.parse_args()

  onnx_path = args.onnx or find_onnx(args.run)
  pol = Policy(onnx_path)
  robot = Robot(pol.joint_names, for_video=bool(args.video))
  robot.command[:] = [args.vx, args.vy, args.wz]
  robot.reset()
  pol.reset()

  print(f"policy : {Path(onnx_path).name}")
  print(f"         obs {pol.in_dim}  actions {pol.nu}  "
        f"history {sorted(set(pol.obs_hist))}")
  print(f"         terms {pol.obs_names}")
  print(f"model  : {robot.m.nbody} bodies, {robot.m.nu} actuators, "
        f"dt {SIM_DT}s x{DECIMATION} -> {1/(SIM_DT*DECIMATION):.0f} Hz policy")
  print(f"command: vx {args.vx}  vy {args.vy}  wz {args.wz}")

  def policy_step():
    obs = pol.observe(robot.m, robot)
    pol.act(obs)
    robot.d.ctrl[:] = pol.targets()
    for _ in range(DECIMATION):
      mujoco.mj_step(robot.m, robot.d)

  # ---- headless ------------------------------------------------------------
  if args.headless:
    zs, feet, taus = [], [], []
    fell_at = None
    for i in range(args.headless):
      policy_step()
      zs.append(float(robot.d.qpos[2]))
      lo = min(float(robot.d.geom_xpos[g][2]) for g in range(robot.m.ngeom)
               if (nm := mujoco.mj_id2name(robot.m, mujoco.mjtObj.mjOBJ_GEOM, g))
               and nm.endswith("_collision") and "foot" in nm)
      feet.append(lo)
      taus.append(float(np.abs(robot.d.actuator_force).max()))
      if fell_at is None and robot.fell():
        fell_at = i * SIM_DT * DECIMATION
    print(f"\n{args.headless} policy steps "
          f"({args.headless*SIM_DT*DECIMATION:.1f} s)")
    print(f"  pelvis z   start {zs[0]:.4f}  end {zs[-1]:.4f}  min {min(zs):.4f}")
    print(f"  fell       {'at %.2f s' % fell_at if fell_at else 'no'}")
    print(f"  travelled  {np.linalg.norm(robot.d.qpos[:2]):.3f} m")
    print(f"  foot pad z min {min(feet)*1000:+.1f} mm  max {max(feet)*1000:+.1f} mm "
          f"(lift {1000*(max(feet)-min(feet)):.1f} mm)")
    print(f"  peak |tau| {max(taus):.1f} N.m")
    return

  # ---- video ---------------------------------------------------------------
  if args.video:
    import mediapy
    os.environ.setdefault("MUJOCO_GL", "egl")
    n = int(args.seconds / (SIM_DT * DECIMATION))
    rend = mujoco.Renderer(robot.m, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(robot.m, cam)
    cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 140
    frames = []
    for _ in range(n):
      policy_step()
      cam.lookat[:] = [robot.d.qpos[0], robot.d.qpos[1], 0.8]
      rend.update_scene(robot.d, cam)
      frames.append(rend.render())
    mediapy.write_video(args.video, frames, fps=1 / (SIM_DT * DECIMATION))
    print(f"\nwrote {args.video}  ({len(frames)} frames, {args.seconds:.1f} s)")
    return

  # ---- interactive ---------------------------------------------------------
  # NOTE: `from mujoco import viewer`, not `import mujoco.viewer` -- the latter
  # binds the name `mujoco` as a local of main() and the policy_step closure
  # then cannot see the module-level import.
  from mujoco import viewer as mj_viewer

  def key_cb(keycode):
    c = chr(keycode) if keycode < 256 else ""
    step = {"W": (0, +0.1), "S": (0, -0.1), "A": (1, +0.1), "D": (1, -0.1),
            "Q": (2, +0.1), "E": (2, -0.1)}
    if c.upper() in step:
      i, dv = step[c.upper()]
      robot.command[i] = float(np.clip(robot.command[i] + dv, -1.5, 1.5))
      print(f"command {np.round(robot.command, 2)}")
    elif c.upper() == "X":
      robot.command[:] = 0
      print("command zeroed")
    elif c.upper() == "R":
      robot.reset()
      pol.reset()
      print("reset")

  print("\nviewer keys: W/S forward  A/D lateral  Q/E turn  X zero  R reset")
  with mj_viewer.launch_passive(robot.m, robot.d, key_callback=key_cb) as v:
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.2, -12, 140
    import time
    while v.is_running():
      t0 = time.time()
      policy_step()
      v.cam.lookat[:] = [robot.d.qpos[0], robot.d.qpos[1], 0.8]
      v.sync()
      dt = SIM_DT * DECIMATION - (time.time() - t0)
      if dt > 0:
        time.sleep(dt)


if __name__ == "__main__":
  main()
