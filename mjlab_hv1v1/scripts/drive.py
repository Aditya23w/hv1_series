#!/usr/bin/env python3
"""Drive a trained HV1V1 policy around a lit MuJoCo scene, by gamepad or keyboard.

This is the FUN viewer, not the verification one. It reuses sim2sim.py's Policy
and its ONNX-metadata-driven observation assembly verbatim -- same joint order,
same per-term history, same anchor and action scale -- so what you steer here is
exactly what sim2sim.py validates. The only differences are cosmetic: lights, a
textured ground, a camera that follows the robot, and live input.

  python mjlab_hv1v1/scripts/drive.py --run v4run
  python mjlab_hv1v1/scripts/drive.py --run v4run --no-gamepad
  python mjlab_hv1v1/scripts/drive.py --onnx /path/to/policy.onnx

GAMEPAD (auto-detected at /dev/input/js0, read with no extra dependency):
  left stick    forward/back and strafe
  right stick X turn
  A / button 0  zero the command
  B / button 1  reset the robot
  START / 7     pause

KEYBOARD (always active, in the viewer window):
  W / S   forward / back        A / D   strafe left / right
  Q / E   turn left / right     X       zero command
  R       reset                 SPACE   pause
  1 / 2   slower / faster command step
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim2sim import DECIMATION, SIM_DT, Policy, find_onnx  # noqa: E402

CONTROL_DT = SIM_DT * DECIMATION


# ---------------------------------------------------------------------------
# Gamepad: the Linux joystick API directly, so there is nothing to install.
# ---------------------------------------------------------------------------
class Gamepad:
  """Non-blocking reader for /dev/input/jsN.

  Each event is 8 bytes: uint32 time, int16 value, uint8 type, uint8 number.
  type 0x01 = button, 0x02 = axis, 0x80 = the synthetic initial-state flag that
  the kernel replays on open (masked off below so a resting stick at open does
  not register as input).
  """

  EVENT = struct.Struct("IhBB")

  def __init__(self, path: str = "/dev/input/js0", deadzone: float = 0.12):
    self.deadzone = deadzone
    self.axes: dict[int, float] = {}
    self.pressed: set[int] = set()
    self.fd = None
    try:
      self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
      self.name = path
    except OSError:
      self.name = ""

  @property
  def ok(self) -> bool:
    return self.fd is not None

  def poll(self) -> set[int]:
    """Drain pending events. Returns buttons that went down this frame."""
    downs: set[int] = set()
    if self.fd is None:
      return downs
    while True:
      try:
        raw = os.read(self.fd, self.EVENT.size)
      except BlockingIOError:
        break
      except OSError:  # unplugged mid-run; degrade to keyboard rather than die
        os.close(self.fd)
        self.fd = None
        print("\n[gamepad disconnected -- keyboard still works]")
        break
      if len(raw) < self.EVENT.size:
        break
      _, value, ev_type, number = self.EVENT.unpack(raw)
      if ev_type & 0x80:  # initial-state replay on open
        continue
      if ev_type == 0x02:
        v = value / 32767.0
        self.axes[number] = 0.0 if abs(v) < self.deadzone else v
      elif ev_type == 0x01:
        if value:
          self.pressed.add(number)
          downs.add(number)
        else:
          self.pressed.discard(number)
    return downs

  def axis(self, number: int) -> float:
    return self.axes.get(number, 0.0)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
def build_scene(joint_names):
  """The robot plus a floor, a sky and two lights.

  Kept separate from sim2sim.Robot on purpose: that class is part of the
  deployment-verification path and its bare grey plane is deliberate, because
  visuals must never change what the physics does. Nothing added here touches
  dynamics -- textures, lights and the skybox are render-only, and the floor
  keeps sim2sim's exact condim/friction so the contact behaviour is identical.
  """
  from mjlab.entity.entity import Entity

  from mjlab_hv1v1.hv1v1_constants import get_hv1v1_robot_cfg

  spec = Entity(get_hv1v1_robot_cfg()).spec

  spec.add_texture(
    name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=[0.35, 0.45, 0.60], rgb2=[0.06, 0.07, 0.10], width=512, height=3072,
  )
  spec.add_texture(
    name="groundtex", type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=[0.22, 0.24, 0.27], rgb2=[0.30, 0.33, 0.37],
    mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.5, 0.55, 0.6],
    width=512, height=512,
  )
  ground = spec.add_material(
    name="groundmat", texrepeat=[8, 8], texuniform=True,
    reflectance=0.08, shininess=0.1, specular=0.2,
  )
  # textures[] is indexed by texture ROLE; 1 is the RGB role, i.e. what
  # `<material texture="...">` means in XML.
  ground.textures[1] = "groundtex"

  # Same condim and friction as sim2sim.py's floor -- only the look changes.
  spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
    condim=3, friction=[0.8, 0.02, 0.01], material="groundmat",
  )
  spec.worldbody.add_light(
    name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    pos=[0, 0, 6], dir=[-0.3, -0.4, -1.0], castshadow=True,
    diffuse=[0.75, 0.75, 0.72], specular=[0.25, 0.25, 0.25],
  )
  spec.worldbody.add_light(
    name="fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    pos=[-4, 4, 4], dir=[0.4, -0.4, -0.8], castshadow=False,
    diffuse=[0.28, 0.30, 0.35], specular=[0, 0, 0],
  )

  m = spec.compile()
  m.opt.timestep = SIM_DT
  # Softer headlight so the directional lights actually shape the robot.
  m.vis.headlight.ambient[:] = [0.30, 0.30, 0.32]
  m.vis.headlight.diffuse[:] = [0.18, 0.18, 0.18]
  m.vis.headlight.specular[:] = [0.05, 0.05, 0.05]
  m.vis.quality.shadowsize = 8192
  # shadowclip 2.0 leaves a visible streak of shadow acne on the ground plane.
  m.vis.map.shadowclip = 8.0
  m.vis.map.shadowscale = 0.8

  d = mujoco.MjData(m)

  jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
  assert all(i >= 0 for i in jid), "a policy joint name is missing from the model"
  act_joint = [
    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
    for a in range(m.nu)
  ]
  assert act_joint == list(joint_names), "actuator order != policy joint order"
  return m, d


class Rig:
  """The sensor/state surface Policy.observe() expects, over our own model."""

  def __init__(self, m, d, joint_names):
    self.m, self.d = m, d
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    self.qadr = np.array([m.jnt_qposadr[i] for i in jid])
    self.vadr = np.array([m.jnt_dofadr[i] for i in jid])
    self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    self._sens = {
      mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s): (m.sensor_adr[s], m.sensor_dim[s])
      for s in range(m.nsensor)
    }
    self.key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    self.command = np.zeros(3)

  def reset(self):
    if self.key >= 0:
      mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
    else:
      mujoco.mj_resetData(self.m, self.d)
    mujoco.mj_forward(self.m, self.d)

  def _sensor(self, name):
    a, n = self._sens[name]
    return self.d.sensordata[a : a + n].copy()

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


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default="v4run", help="run-name suffix")
  ap.add_argument("--onnx", default=None, help="explicit .onnx path (overrides --run)")
  ap.add_argument("--no-gamepad", action="store_true")
  ap.add_argument("--js", default="/dev/input/js0")
  ap.add_argument("--axes", default="1,0,3",
                  help="gamepad axis numbers for vx,vy,wz (default 1,0,3)")
  ap.add_argument("--vx-max", type=float, default=1.2)
  ap.add_argument("--vy-max", type=float, default=0.6)
  ap.add_argument("--wz-max", type=float, default=1.0)
  args = ap.parse_args()

  onnx_path = args.onnx or find_onnx(args.run)
  pol = Policy(onnx_path)
  m, d = build_scene(pol.joint_names)
  rig = Rig(m, d, pol.joint_names)
  rig.reset()
  pol.reset()

  ax_vx, ax_vy, ax_wz = (int(x) for x in args.axes.split(","))
  limits = np.array([args.vx_max, args.vy_max, args.wz_max])

  pad = None if args.no_gamepad else Gamepad(args.js)
  if pad is not None and not pad.ok:
    pad = None

  print(f"policy  : {Path(onnx_path).name}")
  print(f"          obs {pol.in_dim}  actions {pol.nu}  "
        f"history {sorted(set(pol.obs_hist))}")
  print(f"model   : {m.nbody} bodies, {m.nu} actuators, {m.nlight} lights, "
        f"{1 / CONTROL_DT:.0f} Hz policy")
  print(f"gamepad : {'ACTIVE on ' + args.js if pad else 'not used (keyboard only)'}")
  print("keys    : W/S fwd  A/D strafe  Q/E turn  X zero  R reset  SPACE pause  1/2 step")

  state = {"paused": False, "step": 0.1}

  def key_cb(keycode):
    c = chr(keycode).upper() if keycode < 256 else ""
    axis_step = {"W": (0, +1), "S": (0, -1), "A": (1, +1), "D": (1, -1),
                 "Q": (2, +1), "E": (2, -1)}
    if c in axis_step:
      i, sign = axis_step[c]
      rig.command[i] = float(
        np.clip(rig.command[i] + sign * state["step"], -limits[i], limits[i])
      )
    elif c == "X":
      rig.command[:] = 0.0
    elif c == "R":
      rig.reset()
      pol.reset()
    elif c == " ":
      state["paused"] = not state["paused"]
    elif c == "1":
      state["step"] = max(0.02, state["step"] / 2)
      print(f"command step {state['step']:.3f}")
    elif c == "2":
      state["step"] = min(0.5, state["step"] * 2)
      print(f"command step {state['step']:.3f}")

  from mujoco import viewer as mj_viewer

  with mj_viewer.launch_passive(m, d, key_callback=key_cb) as v:
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.6, -14, 135
    last_print = 0.0
    while v.is_running():
      t0 = time.time()

      if pad is not None:
        downs = pad.poll()
        if 0 in downs:
          rig.command[:] = 0.0
        if 1 in downs:
          rig.reset()
          pol.reset()
        if 7 in downs:
          state["paused"] = not state["paused"]
        if pad.ok:
          # Linux reports stick-up as negative, and +y is the robot's left, so
          # both translation axes are negated to match "push the way you go".
          rig.command[0] = -pad.axis(ax_vx) * limits[0]
          rig.command[1] = -pad.axis(ax_vy) * limits[1]
          rig.command[2] = -pad.axis(ax_wz) * limits[2]

      if not state["paused"]:
        pol.act(pol.observe(m, rig))
        d.ctrl[:] = pol.targets()
        for _ in range(DECIMATION):
          mujoco.mj_step(m, d)

      # Camera rides along; the free camera keeps its user-set angle and zoom.
      v.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.85]

      # Command arrow from the pelvis: where you are ASKING it to go.
      v.user_scn.ngeom = 0
      cmd_b = rig.command[:2]
      if np.linalg.norm(cmd_b) > 1e-3:
        R = rig.root_rot()
        world = R @ np.array([cmd_b[0], cmd_b[1], 0.0])
        base = np.array([d.qpos[0], d.qpos[1], 1.05])
        g = v.user_scn.geoms[v.user_scn.ngeom]
        mujoco.mjv_initGeom(
          g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
          np.array([0.25, 0.85, 1.0, 0.85], dtype=np.float32),
        )
        mujoco.mjv_connector(
          g, mujoco.mjtGeom.mjGEOM_ARROW, 0.022, base, base + world * 0.6
        )
        v.user_scn.ngeom += 1

      v.sync()

      now = time.time()
      if now - last_print > 0.5:
        vel = rig.root_rot().T @ d.qvel[:3]
        print(
          f"\rcmd vx {rig.command[0]:+.2f} vy {rig.command[1]:+.2f} wz {rig.command[2]:+.2f}"
          f"   actual vx {vel[0]:+.2f} vy {vel[1]:+.2f} wz {d.qvel[5]:+.2f}"
          f"   z {d.qpos[2]:.2f}{'  [PAUSED]' if state['paused'] else ''}"
          f"{'  [FELL - press R]' if rig.fell() else ''}        ",
          end="", flush=True,
        )
        last_print = now

      # Real time, so the gait looks like the gait.
      slack = CONTROL_DT - (time.time() - t0)
      if slack > 0:
        time.sleep(slack)


if __name__ == "__main__":
  main()
