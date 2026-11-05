#!/usr/bin/env python3
"""Run a trained HV1V2 policy in plain MuJoCo, outside the mjlab training loop.

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
  python mjlab_hv1v2/scripts/sim2sim.py --run v1
  # no window: step N policy steps and print what happened
  python mjlab_hv1v2/scripts/sim2sim.py --run v1 --headless 500
  # gait table at a fixed command: step length, swing time, cadence, flight
  python mjlab_hv1v2/scripts/sim2sim.py --onnx exported/deploy_v4/policy.onnx \
      --headless 750 --vx 1.0
  # record
  python mjlab_hv1v2/scripts/sim2sim.py --run v1 --video /tmp/v1.mp4 --seconds 8

Viewer keys: W/S forward, A/D lateral, Q/E turn, X zero command, R reset.
v11h (ONNX has `height_cmd`): Up/Down arrow = pelvis height target +-1 cm,
Page Up = back to home height; the applied height follows the target at
0.10 m/s, exactly as in training / merai. `--height 0.92` sets the start target.

Gamepad (pygame, auto-detected, hot-pluggable; keyboard keeps working):
  left stick  up/down -> vx, left/right -> vy     right stick left/right -> wz
  A / Cross   zero command                         B / Circle  reset
Sticks map linearly onto --vx-range / --vy-range / --wz-range, which default
to the v5 trained ranges. Anything outside them is off-distribution.

  python mjlab_hv1v2/scripts/sim2sim.py --run v5

Decoupled policies (v6+, ONNX carries `action_joint_names`): the policy drives
only those joints; every other joint is driven by an upper-body driver standing
in for the IK controller -- `--upper hold|random|wave`, amplitude
`--upper-level` (0..1 of the trained envelope), optional `--payload` kg per
hand. Keyboard U / gamepad Y cycles the mode, gamepad LB/RB changes the level.

  # the decoupling test: stand still while the arms move
  python mjlab_hv1v2/scripts/sim2sim.py --run v6 --headless 750 \
      --vx 0 --upper random --upper-level 1.0 --payload 2
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort

LOG_ROOT = Path(__file__).resolve().parents[1] / "logs/rsl_rl/hv1v2_velocity"
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
    # Which joints the actions drive. Written by Hv1v2OnPolicyRunner since the
    # decoupled task; absent in earlier ONNX, where actions cover every joint.
    self.action_joints = (
      meta["action_joint_names"].split(",") if "action_joint_names" in meta
      else list(self.joint_names)
    )
    self.act_idx = np.array([self.joint_names.index(n) for n in self.action_joints])
    self.decoupled = len(self.action_joints) < len(self.joint_names)
    scale = np.array([float(x) for x in meta["action_scale"].split(",")])
    self.action_scale = np.broadcast_to(scale, (len(self.action_joints),)).copy()
    self.obs_names = meta["observation_names"].split(",")
    self.obs_hist = [int(float(x)) for x in meta["observation_terms_history_length"].split(",")]
    self.obs_scale = [float(x) for x in meta["observation_terms_scale"].split(",")]
    self.kp = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
    self.kd = np.array([float(x) for x in meta["joint_damping"].split(",")])

    self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    self.in_name = self.sess.get_inputs()[0].name
    self.in_dim = self.sess.get_inputs()[0].shape[-1]
    self.nu = len(self.action_joints)
    self.prev_action = np.zeros(self.nu)
    self._hist: dict[str, deque] = {}
    # v11: the actor also sees the upper-body target (rel. home) and its rate,
    # for V11_UB_OBS_JOINTS, at history 1
    self.v11 = "ub_target" in self.obs_names
    # v11h: pelvis-height command (applied, slewed) minus home, history 1
    self.v11h = "height_cmd" in self.obs_names

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
    if name == "ub_target":
      return robot.ub_rel
    if name == "ub_target_rate":
      return robot.ub_rate
    if name == "height_cmd":
      return np.array([robot.height - robot.height_home])
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
    """mjlab's JointPositionAction: processed = raw * scale + default.

    Full 31-wide; joints outside the action hold their default here and are
    overwritten by the upper-body driver for a decoupled policy.
    """
    q = self.default_qpos.copy()
    q[self.act_idx] += self.prev_action * self.action_scale
    return q


def pelvis_height(robot) -> float:
  """Measured pelvis height above the floor (z = 0), as mdp_v11h.base_height."""
  return float(robot.d.qpos[2])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class Robot:
  def __init__(self, joint_names, for_video: bool, v11: bool = False):
    from mjlab.entity.entity import Entity
    from mjlab_hv1v2.hv1v2_constants import get_hv1v2_robot_cfg, get_hv1v2_v11_robot_cfg

    # v11 trains on its own plant (HOME_KEYFRAME, peak torque caps, waist yaw 400/24)
    spec = Entity(get_hv1v2_v11_robot_cfg() if v11 else get_hv1v2_robot_cfg()).spec
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
    self.ub_rel = np.zeros(0)
    self.ub_rate = np.zeros(0)
    self.jnt_lo = self.m.jnt_range[jid, 0].copy()
    self.jnt_hi = self.m.jnt_range[jid, 1].copy()

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
# gamepad
# ---------------------------------------------------------------------------
class Gamepad:
  """Polls the first SDL joystick through pygame. No pygame window is opened.

  SDL only delivers joystick events to a focused window unless told
  otherwise, and the MuJoCo viewer owns focus -- hence the env var.
  Axis 3 is the right stick X on both xpad (Xbox) and hid-sony (PS4/PS5);
  override with --turn-axis if your pad differs.
  """

  def __init__(self, turn_axis: int, deadzone: float = 0.1):
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    self.pg = pygame
    pygame.init()
    pygame.joystick.init()
    self.turn_axis = turn_axis
    self.deadzone = deadzone
    self.js = None
    self._open_first()

  def _open_first(self):
    if self.pg.joystick.get_count():
      self.js = self.pg.joystick.Joystick(0)
      self.js.init()
      print(f"gamepad: {self.js.get_name()} "
            f"({self.js.get_numaxes()} axes, {self.js.get_numbuttons()} buttons)")
    else:
      self.js = None
      print("gamepad: none found -- keyboard only (plug one in any time)")

  def _axis(self, i: int) -> float:
    if self.js is None or i >= self.js.get_numaxes():
      return 0.0
    v = float(self.js.get_axis(i))
    if abs(v) < self.deadzone:
      return 0.0
    # rescale so the stick leaves the deadzone at 0, not at 0.1
    return float(np.sign(v) * (abs(v) - self.deadzone) / (1.0 - self.deadzone))

  def poll(self):
    """Returns (sticks [-1,1]^3 as (fwd, left, ccw) or None, buttons pressed)."""
    pressed = set()
    for ev in self.pg.event.get():
      if ev.type == self.pg.JOYDEVICEADDED and self.js is None:
        self._open_first()
      elif ev.type == self.pg.JOYDEVICEREMOVED:
        print("gamepad: disconnected")
        self.js = None
        self.pg.joystick.quit()
        self.pg.joystick.init()
      elif ev.type == self.pg.JOYBUTTONDOWN:
        pressed.add(ev.button)
    if self.js is None:
      return None, pressed
    # SDL: stick up and stick left are NEGATIVE.
    return (-self._axis(1), -self._axis(0), -self._axis(self.turn_axis)), pressed


def _stick_to_range(x: float, lo: float, hi: float) -> float:
  """Map [-1, 1] onto [lo, hi] keeping 0 -> 0 (ranges may be asymmetric)."""
  return x * hi if x >= 0 else -x * lo


# ---------------------------------------------------------------------------
# upper body (decoupled policies)
# ---------------------------------------------------------------------------
class UpperBodyDriver:
  """Stand-in for the IK controller: drives every joint the policy does not.

  `random` is a numpy port of mdp.UpperBodyCommand (goal = home + amp*(u-home),
  amp = level*U^1.5, 15% quiet, 1-3 s resample, 0.3-2.0 rad/s slew), so the
  sim2sim disturbance matches training. `wave` is a deterministic sweep of the
  same envelope; `hold` keeps home. Ranges are the TRAINED envelope,
  UPPER_BODY_SAMPLE_RANGE -- deployed IK must stay inside it too.
  """

  MODES = ("hold", "random", "wave")
  # The joints the deployed teleop IK drives in WALK_TELEOP (merai_policy.yaml
  # walk_decoupled upper_body; left_wrist_roll is masked out of service). The
  # head, left_wrist_roll and waist roll/pitch stay at home on the robot -- and
  # v8 was trained that way, so moving them is off-distribution for it.
  IK_JOINTS = frozenset((
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  ))

  def __init__(self, pol: "Policy", mode: str, level: float, seed: int = 0,
               ik_only: bool = False):
    from mjlab_hv1v2.hv1v2_constants import UPPER_BODY_SAMPLE_RANGE

    self.idx = np.array([i for i in range(len(pol.joint_names))
                         if i not in set(pol.act_idx.tolist())])
    self.names = [pol.joint_names[i] for i in self.idx]
    lo, hi = [], []
    for n in self.names:
      hits = [r for pat, r in UPPER_BODY_SAMPLE_RANGE.items() if re.match(pat, n)]
      assert len(hits) == 1, f"{n} matched {len(hits)} UPPER_BODY_SAMPLE_RANGE keys"
      lo.append(hits[0][0])
      hi.append(hits[0][1])
    self.lo, self.hi = np.array(lo), np.array(hi)
    self.home = pol.default_qpos[self.idx].copy()
    if ik_only:  # everything outside the deployed IK set holds home
      for k, n in enumerate(self.names):
        if n not in self.IK_JOINTS:
          self.lo[k] = self.hi[k] = self.home[k]
    self.mode, self.level = mode, level
    self.rng = np.random.default_rng(seed)
    self.phase = self.rng.uniform(0, 2 * np.pi, len(self.idx))
    self.reset()

  def reset(self):
    self.t = 0.0
    self.target = self.home.copy()
    self.goal = self.home.copy()
    self.t_next = 0.0
    self.speed = 1.0

  def cycle(self):
    self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
    print(f"upper body: {self.mode}  level {self.level:.1f}")

  def step(self, dt: float) -> np.ndarray:
    self.t += dt
    if self.mode == "hold":
      self.goal = self.home
    elif self.mode == "wave":
      u = self.lo + (self.hi - self.lo) * (
        0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * self.t + self.phase))
      self.goal = self.home + self.level * (u - self.home)
    elif self.t >= self.t_next:  # random
      u = self.rng.uniform(self.lo, self.hi)
      amp = 0.0 if self.rng.random() < 0.15 else self.level * self.rng.random() ** 1.5
      self.goal = self.home + amp * (u - self.home)
      self.speed = self.rng.uniform(0.3, 2.0)
      self.t_next = self.t + self.rng.uniform(1.0, 3.0)
    lim = self.speed * dt
    self.target = self.target + np.clip(self.goal - self.target, -lim, lim)
    return self.target


def add_hand_payload(robot: "Robot", kg: float) -> None:
  """Point mass at each hand, as the training payload event adds it."""
  from mjlab_hv1v2.hv1v2_constants import HAND_BODY_NAMES

  for n in HAND_BODY_NAMES:
    robot.m.body_mass[mujoco.mj_name2id(robot.m, mujoco.mjtObj.mjOBJ_BODY, n)] += kg


# ---------------------------------------------------------------------------
# smoothness
# ---------------------------------------------------------------------------
def smoothness_report(pol: "Policy", A, T, V, TAU, dt: float) -> None:
  """Per-leg-joint jerk report; the numbers the v7 acceptance gates use.

  A raw policy action, T joint targets (rad), V joint velocity, TAU actuator
  torque -- each (steps, n_leg) over the leg joints, post warm-up.
  """
  legs = [i for i, n in enumerate(pol.joint_names)
          if any(k in n for k in ("hip", "knee", "ankle"))]
  names = [pol.joint_names[i].replace("_joint", "") for i in legs]
  A, T, V, TAU = (np.asarray(x) for x in (A, T, V, TAU))
  dA, d2A = np.abs(np.diff(A, axis=0)), np.abs(np.diff(A, 2, axis=0))
  dT = np.degrees(np.abs(np.diff(T, axis=0)))
  acc = np.abs(np.diff(V, axis=0)) / dt
  dtau = np.abs(np.diff(TAU, axis=0)) / dt
  f = np.fft.rfftfreq(len(V), dt)
  P = np.abs(np.fft.rfft(V - V.mean(0), axis=0)) ** 2
  band = P[(f >= 3) & (f <= 8)].sum(0) / np.maximum(P[f > 0.2].sum(0), 1e-12)
  print(f"  smoothness (legs; overall mean |da| {dA.mean():.3f}  |d2a| {d2A.mean():.3f}  "
        f"3-8 Hz share {band.mean():.2f})")
  print(f"    {'joint':20s} {'|da|':>6s} {'|d2a|':>6s} {'jump p99':>9s} {'max':>6s} "
        f"{'acc p95':>8s} {'dtau p95':>9s} {'3-8Hz':>6s}")
  for k, n in enumerate(names):
    print(f"    {n:20s} {dA[:, k].mean():6.3f} {d2A[:, k].mean():6.3f} "
          f"{np.percentile(dT[:, k], 99):8.1f}d {dT[:, k].max():5.1f}d "
          f"{np.percentile(acc[:, k], 95):8.1f} {np.percentile(dtau[:, k], 95):9.0f} "
          f"{band[k]:6.2f}")


# ---------------------------------------------------------------------------
# gait
# ---------------------------------------------------------------------------
class GaitStats:
  """Per-foot touchdown events, measured the way the training terms see them.

  Contact is any foot pad touching the floor (the sim analogue of the
  feet_ground_contact subtree sensor). Step length is the landing foot's site
  minus the other foot's site, in the yaw-only pelvis frame -- the same
  quantity step_length_excess prices in training. The first `warmup_s` are
  dropped so the start-from-standing transient does not pollute the averages.
  """

  SIDES = ("left", "right")

  def __init__(self, robot: "Robot", joint_names, warmup_s: float = 1.0):
    m = robot.m
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    self.floor = floor
    self.pads = []
    for side in self.SIDES:
      ids = {g for g in range(m.ngeom)
             if (nm := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
             and nm.startswith(f"{side}_foot") and nm.endswith("_collision")}
      assert ids, f"no {side} foot pads found"
      self.pads.append(ids)
    self.sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot")
                  for s in self.SIDES]
    assert all(i >= 0 for i in self.sites), "left_foot/right_foot sites missing"
    self.arm = np.array([i for i, n in enumerate(joint_names)
                         if "shoulder" in n or "elbow" in n])
    self.robot = robot
    self.warmup_steps = int(warmup_s / (SIM_DT * DECIMATION))
    self.n = 0
    self.prev = [True, True]
    self.lift_t = [None, None]
    self.dx, self.dxy, self.swing = [], [], []
    self.touchdowns = 0
    self.flight_steps = 0
    self.arm_speed = []

  def _in_contact(self):
    d = self.robot.d
    c = [False, False]
    for k in range(d.ncon):
      g1, g2 = d.contact[k].geom1, d.contact[k].geom2
      if self.floor not in (g1, g2):
        continue
      other = g2 if g1 == self.floor else g1
      for i in (0, 1):
        if other in self.pads[i]:
          c[i] = True
    return c

  def update(self, joint_vel):
    d = self.robot.d
    t = self.n * SIM_DT * DECIMATION
    c = self._in_contact()
    counting = self.n >= self.warmup_steps
    if counting:
      self.flight_steps += int(not c[0] and not c[1])
      self.arm_speed.append(float(np.sqrt(np.mean(joint_vel[self.arm] ** 2))))
    for i in (0, 1):
      if self.prev[i] and not c[i]:
        self.lift_t[i] = t
      elif not self.prev[i] and c[i]:
        if counting and self.lift_t[i] is not None:
          yaw = self.robot.root_rot()[:, 0]
          yaw = np.arctan2(yaw[1], yaw[0])
          cy, sy = np.cos(yaw), np.sin(yaw)
          delta = d.site_xpos[self.sites[i]][:2] - d.site_xpos[self.sites[1 - i]][:2]
          dx = cy * delta[0] + sy * delta[1]
          self.dx.append(abs(dx))
          self.dxy.append(float(np.linalg.norm(delta)))
          self.swing.append(t - self.lift_t[i])
          self.touchdowns += 1
    self.prev = c
    self.n += 1

  def report(self):
    span = (self.n - self.warmup_steps) * SIM_DT * DECIMATION
    if not self.touchdowns or span <= 0:
      print("  gait       no touchdowns after warm-up")
      return
    f = lambda xs: f"{np.mean(xs):.3f} (p90 {np.percentile(xs, 90):.3f})"
    print(f"  gait       over {span:.1f} s after {self.warmup_steps} warm-up steps")
    print(f"    step length, heading axis  {f(self.dx)} m")
    print(f"    step length, planar        {f(self.dxy)} m")
    print(f"    swing time                 {f(self.swing)} s")
    print(f"    cadence                    {self.touchdowns / span:.2f} steps/s")
    print(f"    double-flight fraction     "
          f"{self.flight_steps / (self.n - self.warmup_steps):.3f}")
    print(f"    arm joint RMS speed        {np.mean(self.arm_speed):.3f} rad/s")


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
  # Defaults are the v5 trained ranges (env_cfgs.py velocity_stages).
  ap.add_argument("--vx-range", type=float, nargs=2, default=(-0.8, 1.0))
  ap.add_argument("--vy-range", type=float, nargs=2, default=(-0.5, 0.5))
  ap.add_argument("--wz-range", type=float, nargs=2, default=(-0.7, 0.7))
  ap.add_argument("--no-gamepad", action="store_true", help="keyboard only")
  ap.add_argument("--turn-axis", type=int, default=3,
                  help="joystick axis for yaw rate (right stick X)")
  ap.add_argument("--upper", choices=UpperBodyDriver.MODES + ("human",), default="random",
                  help="decoupled policies: what drives the non-policy joints. "
                       "human = human_upper.py (leg-synced arm swing, waist, head); "
                       "U / gamepad Y toggles it on/off")
  ap.add_argument("--upper-level", type=float, default=0.5,
                  help="0..1 of the trained upper-body envelope")
  ap.add_argument("--payload", type=float, default=0.0, help="kg added per hand")
  ap.add_argument("--height", type=float, default=None,
                  help="v11h: pelvis height target (m), slewed to from home at 0.10 m/s")
  ap.add_argument("--upper-joints", choices=("all", "all-but-waist-rp", "ik"),
                  default="all",
                  help="ik: drive only the 14 deployed teleop-IK joints (head, "
                       "left_wrist_roll, waist roll/pitch hold home, as on the "
                       "robot) -- use for v8/v8c. all / all-but-waist-rp (same "
                       "thing: waist roll/pitch have a zero range, never move): "
                       "every other upper-body joint -- use for v7 and v8.")
  ap.add_argument("--interp", action="store_true",
                  help="v8+: interpolate every 50 Hz target linearly across the "
                       "physics substeps (merai_control timing). Off = zero-order "
                       "hold, which is what v1-v7 were trained with.")
  ap.add_argument("--smooth", action="store_true",
                  help="headless: per-leg-joint jerk report (|da|, |d2a|, target "
                       "jumps, accel, torque rate, 3-8 Hz share)")
  args = ap.parse_args()
  ranges = np.array([args.vx_range, args.vy_range, args.wz_range])

  onnx_path = args.onnx or find_onnx(args.run)
  pol = Policy(onnx_path)
  robot = Robot(pol.joint_names, for_video=bool(args.video), v11=pol.v11)
  if pol.v11:
    args.interp = True   # v11 trains with merai_control's interpolated targets
  robot.command[:] = [args.vx, args.vy, args.wz]
  if args.payload:
    add_hand_payload(robot, args.payload)
  robot.reset()
  pol.reset()
  if pol.v11h:
    from mjlab_hv1v2.hv1v2_constants import V11H_HEIGHT_RANGE, V11H_HOME_Z
    robot.height_home = V11H_HOME_Z
    robot.height_range = V11H_HEIGHT_RANGE
    robot.height = V11H_HOME_Z                 # applied (what the policy sees)
    robot.height_target = float(np.clip(args.height if args.height is not None else V11H_HOME_Z,
                                        *V11H_HEIGHT_RANGE))
  use_human = args.upper == "human"
  upper = (UpperBodyDriver(pol, "hold" if use_human else args.upper, args.upper_level,
                           ik_only=args.upper_joints == "ik")
           if pol.decoupled else None)
  if pol.v11:
    from mjlab_hv1v2.hv1v2_constants import V11_UB_OBS_JOINTS
    for k, n in enumerate(upper.names):          # v11: waist roll/pitch locked
      if n in ("waist_roll_joint", "waist_pitch_joint"):
        upper.lo[k] = upper.hi[k] = upper.home[k]
    ub_cols = np.array([upper.names.index(n) for n in V11_UB_OBS_JOINTS])
    robot.ub_rel = np.zeros(len(ub_cols))
    robot.ub_rate = np.zeros(len(ub_cols))

  human = None
  if upper is not None and use_human:
    from human_upper import HumanUpperBody
    human = HumanUpperBody(pol.joint_names, pol.default_qpos, upper.idx, SIM_DT * DECIMATION)
    human.reset(robot.qpos_joints())

  def toggle_human():
    nonlocal use_human
    if human is None:
      upper.cycle()
      return
    use_human = not use_human
    human.reset(robot.qpos_joints())
    upper.reset()
    upper.target = robot.qpos_joints()[upper.idx]   # 'hold' slews home from here
    print(f"upper body: {'human' if use_human else 'hold'}")

  print(f"policy : {Path(onnx_path).name}")
  print(f"         obs {pol.in_dim}  actions {pol.nu}  "
        f"history {sorted(set(pol.obs_hist))}")
  if upper is not None:
    ub_desc = ("human generator" if use_human
               else f"driver '{upper.mode}' level {upper.level}")
    print(f"         DECOUPLED: policy drives {pol.nu} joints, upper body "
          f"({len(upper.idx)}) by {ub_desc}, "
          f"payload {args.payload} kg/hand")
  print(f"         terms {pol.obs_names}")
  print(f"model  : {robot.m.nbody} bodies, {robot.m.nu} actuators, "
        f"dt {SIM_DT}s x{DECIMATION} -> {1/(SIM_DT*DECIMATION):.0f} Hz policy")
  print(f"command: vx {args.vx}  vy {args.vy}  wz {args.wz}")

  prev_ctrl = [None]

  last_ub = [None]   # v11: previous upper-body target, for ub_target_rate

  def policy_step():
    ub_tgt = None
    if pol.v11:
      # training order: the upper-body command updates, THEN the actor sees
      # that target (and its rate) and the legs act on it
      # (human mode: the generator's target is what the actor sees)
      if human is not None and use_human:
        ub_tgt = human.step(robot.qpos_joints(), robot.root_rot(),
                            robot.command)[upper.idx].copy()
      else:
        ub_tgt = upper.step(SIM_DT * DECIMATION).copy()
      prev = ub_tgt if last_ub[0] is None else last_ub[0]
      last_ub[0] = ub_tgt.copy()
      robot.ub_rel = ub_tgt[ub_cols] - upper.home[ub_cols]
      robot.ub_rate = (ub_tgt[ub_cols] - prev[ub_cols]) / (SIM_DT * DECIMATION)
    if pol.v11h:
      # training order (BaseHeightCommand): the applied height moves toward the
      # target by <= 0.10 m/s * 20 ms, THEN the actor observes it
      d = 0.10 * SIM_DT * DECIMATION
      robot.height += float(np.clip(robot.height_target - robot.height, -d, d))
    obs = pol.observe(robot.m, robot)
    pol.act(obs)
    ctrl = pol.targets()
    if pol.v11:
      ctrl[pol.act_idx] = np.clip(ctrl[pol.act_idx], robot.jnt_lo[pol.act_idx],
                                  robot.jnt_hi[pol.act_idx])   # as merai_control clips
      ctrl[upper.idx] = ub_tgt
    elif human is not None and use_human:
      ctrl[upper.idx] = human.step(robot.qpos_joints(), robot.root_rot(),
                                   robot.command)[upper.idx]
    elif upper is not None:
      ctrl[upper.idx] = upper.step(SIM_DT * DECIMATION)
    if not args.interp:
      robot.d.ctrl[:] = ctrl
      for _ in range(DECIMATION):
        mujoco.mj_step(robot.m, robot.d)
      return
    if prev_ctrl[0] is None:
      prev_ctrl[0] = ctrl.copy()
    for s in range(DECIMATION):
      a = (s + 1) / DECIMATION
      robot.d.ctrl[:] = prev_ctrl[0] + a * (ctrl - prev_ctrl[0])
      mujoco.mj_step(robot.m, robot.d)
    prev_ctrl[0] = ctrl

  # ---- headless ------------------------------------------------------------
  if args.headless:
    zs, feet, taus = [], [], []
    fell_at = None
    gait = GaitStats(robot, pol.joint_names)
    feet0 = pelvis0 = None
    tilt = []
    legs = [i for i, n in enumerate(pol.joint_names)
            if any(k in n for k in ("hip", "knee", "ankle"))]
    leg_cols = [list(pol.act_idx).index(i) for i in legs]
    sA, sT, sV, sTAU = [], [], [], []
    for i in range(args.headless):
      policy_step()
      if fell_at is None:
        gait.update(robot.qvel_joints())
      if i == gait.warmup_steps:
        feet0 = robot.d.site_xpos[gait.sites][:, :2].copy()
        pelvis0 = robot.d.qpos[:2].copy()
      if args.smooth and i >= gait.warmup_steps and fell_at is None:
        sA.append(pol.prev_action[leg_cols].copy())
        sT.append(robot.d.ctrl[legs].copy())
        sV.append(robot.qvel_joints()[legs])
        sTAU.append(robot.d.actuator_force[legs].copy())
      if i >= gait.warmup_steps and fell_at is None:
        g = robot.root_rot().T @ GRAVITY_W
        tilt.append(float(np.degrees(np.arccos(np.clip(-g[2], -1, 1)))))
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
    if pol.v11h:
      print(f"  height cmd target {robot.height_target:.4f}  applied {robot.height:.4f}  "
            f"measured {pelvis_height(robot):.4f}  (err {1000*(pelvis_height(robot)-robot.height):+.1f} mm)")
    print(f"  fell       {'at %.2f s' % fell_at if fell_at else 'no'}")
    print(f"  travelled  {np.linalg.norm(robot.d.qpos[:2]):.3f} m")
    print(f"  foot pad z min {min(feet)*1000:+.1f} mm  max {max(feet)*1000:+.1f} mm "
          f"(lift {1000*(max(feet)-min(feet)):.1f} mm)")
    print(f"  peak |tau| {max(taus):.1f} N.m")
    gait.report()
    if args.smooth and len(sA) > 10:
      smoothness_report(pol, sA, sT, sV, sTAU, SIM_DT * DECIMATION)
    if upper is not None and feet0 is not None and not robot.command.any():
      # The decoupling test: with a zero command, upper-body motion must not
      # move the lower body. Drift is end minus post-warm-up position.
      drift = np.linalg.norm(robot.d.site_xpos[gait.sites][:, :2] - feet0, axis=1)
      print(f"  disturbance (upper '{upper.mode}' level {upper.level}, "
            f"payload {args.payload} kg)")
      print(f"    foot drift      L {drift[0]*1000:.1f} mm  R {drift[1]*1000:.1f} mm")
      print(f"    pelvis drift    "
            f"{np.linalg.norm(robot.d.qpos[:2]-pelvis0)*1000:.1f} mm")
      print(f"    pelvis tilt     mean {np.mean(tilt):.2f} deg  max {np.max(tilt):.2f} deg")
      print(f"    steps taken     {gait.touchdowns}")
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
    if pol.v11h and keycode in (265, 264, 266):   # GLFW Up / Down / Page Up
      if keycode == 266:
        robot.height_target = robot.height_home
      else:
        robot.height_target = float(np.clip(robot.height_target + (0.01 if keycode == 265 else -0.01),
                                            *robot.height_range))
      print(f"height target {robot.height_target:.3f} m  (home {robot.height_home:.3f}; "
            f"measured now {pelvis_height(robot):.3f})")
      return
    c = chr(keycode) if keycode < 256 else ""
    step = {"W": (0, +0.1), "S": (0, -0.1), "A": (1, +0.1), "D": (1, -0.1),
            "Q": (2, +0.1), "E": (2, -0.1)}
    if c.upper() in step:
      i, dv = step[c.upper()]
      robot.command[i] = float(np.clip(robot.command[i] + dv, *ranges[i]))
      print(f"command {np.round(robot.command, 2)}")
    elif c.upper() == "X":
      robot.command[:] = 0
      print("command zeroed")
    elif c.upper() == "R":
      robot.reset()
      pol.reset()
      if upper is not None:
        upper.reset()
      if human is not None:
        human.reset(robot.qpos_joints())
      last_ub[0] = None
      if pol.v11h:
        robot.height = robot.height_home
      print("reset")
    elif c.upper() == "U" and upper is not None:
      toggle_human()

  pad = None if args.no_gamepad else Gamepad(args.turn_axis)
  stick_live = False  # keyboard owns the command until the stick moves

  def gamepad_step():
    nonlocal stick_live
    sticks, pressed = pad.poll()
    if 0 in pressed:
      robot.command[:] = 0
      print("command zeroed")
    if 1 in pressed:
      robot.reset()
      pol.reset()
      if upper is not None:
        upper.reset()
      if human is not None:
        human.reset(robot.qpos_joints())
      last_ub[0] = None
      print("reset")
    if upper is not None:
      if 3 in pressed:  # Y / Triangle
        toggle_human()
      for b, dl in ((4, -0.1), (5, +0.1)):  # LB / RB
        if b in pressed:
          upper.level = float(np.clip(upper.level + dl, 0.0, 1.0))
          print(f"upper body level {upper.level:.1f}")
    if sticks is None:
      return
    moved = any(s != 0.0 for s in sticks)
    if moved or stick_live:
      # Once the stick has been used it stays authoritative, so releasing
      # it returns the command to zero instead of freezing the last value.
      stick_live = moved
      robot.command[:] = [_stick_to_range(s, *r) for s, r in zip(sticks, ranges)]

  print("\nviewer keys: W/S forward  A/D lateral  Q/E turn  X zero  R reset")
  if pol.v11h:
    print(f"height     : Up/Down arrow +-1 cm, Page Up = home ({robot.height_home:.3f});  "
          f"range {robot.height_range[0]:.3f}..{robot.height_range[1]:.3f} m, slewed 0.10 m/s")
  if upper is not None:
    if human is not None:
      print("upper body : HUMAN on -- U (keyboard) / Y (pad) toggles human <-> home arms")
    else:
      print("upper body : U (keyboard) / Y (pad) cycles hold|random|wave, "
            "LB/RB level -/+0.1")
  print(f"command limits: vx {tuple(ranges[0])}  vy {tuple(ranges[1])}  "
        f"wz {tuple(ranges[2])}")
  with mj_viewer.launch_passive(robot.m, robot.d, key_callback=key_cb) as v:
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.2, -12, 140
    import time
    last_print = 0.0
    while v.is_running():
      t0 = time.time()
      if pad is not None:
        gamepad_step()
        if stick_live and t0 - last_print > 0.5:
          print(f"command {np.round(robot.command, 2)}", end="\r", flush=True)
          last_print = t0
      policy_step()
      if pol.v11h and t0 - last_print > 1.0 and not stick_live:
        print(f"height target {robot.height_target:.3f}  applied {robot.height:.3f}  "
              f"measured {pelvis_height(robot):.3f}   cmd {np.round(robot.command, 2)}   ",
              end="\r", flush=True)
        last_print = t0
      v.cam.lookat[:] = [robot.d.qpos[0], robot.d.qpos[1], 0.8]
      v.sync()
      dt = SIM_DT * DECIMATION - (time.time() - t0)
      if dt > 0:
        time.sleep(dt)


if __name__ == "__main__":
  main()
