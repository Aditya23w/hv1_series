#!/usr/bin/env python3
"""Human-like upper body for the decoupled HV1V2 walking policies (v7 first).

The decoupled policy drives the 12 leg joints only; everything above the
pelvis is "someone else's". This module is that someone else: a 50 Hz,
pure-numpy generator that makes the arms, waist yaw and head move the way a
walking human's do, slaved to what the LEGS are actually doing. No training,
no IK -- every motion here is a joint-space coupling, so it ports to the C++
runtime as a few filters and a rate limiter.

Signal: s = (q_hip_pitch_R - q_hip_pitch_L) / 2, the hips' scissor angle.
s > 0 <=> the left leg is in front (hip pitch + swings the leg BACK on HV1V2).
On v7 it is a clean 1.0 Hz sinusoid of amplitude 0.11 / 0.22 / 0.32 rad at
vx 0.3 / 0.6 / 1.0 and ~0.03 rad when stepping sideways or turning in place.
It is band-passed (0.3 Hz high-pass removes the stance asymmetry, 3 Hz
2nd-order low-pass removes policy jitter) and gated by a soft deadband on its
envelope so the arms hang perfectly still while the robot stands.

What each joint does (sign conventions verified by FK on the sim model:
shoulder pitch + = arm back, elbow + = extend (0 = 90 deg bend, ~1.55 straight),
waist yaw + = torso CCW from above, head pitch + = look down):

  shoulders   contralateral swing: left arm forward with the right leg.
              amplitude = ARM_GAIN x scissor angle, capped -- ~0.75 is the
              human shoulder/hip swing ratio (about 30 deg vs 40 deg pk-pk).
              The swing is centred slightly behind the rest angle, as in
              human gait (more extension than flexion).
  elbows      flex more on the forward swing, open on the back swing.
  shoulder roll
              rest abduction so the hands clear the hips, a touch more
              when walking.
  waist yaw   the thorax counter-rotates against the pelvis (right shoulder
              forward with the left leg) -- the counter-rotation that cancels
              the legs' yaw angular momentum. Small (<= 0.07 rad) because the
              deployed waist slew limit is 0.6 rad/s. Plus a lead into turns.
  head        gaze stabilisation: yaw cancels the pelvis' yaw wobble and the
              waist's counter-rotation, pitch cancels pelvis pitch bob; looks
              slightly down while walking, and turns into a commanded turn.
              (The head is virtual on the robot today -- sim only.)

Every output passes a per-joint velocity + acceleration limiter with the
deployed Ruckig limits (arms 2.0 rad/s / 8 rad/s^2, waist yaw 0.6 / 4) and is
clamped into UPPER_BODY_SAMPLE_RANGE, the envelope v7 was trained against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HumanUpperCfg:
  # walking arm posture ("relaxed" = arms hanging, elbows ~20 deg bent; "home" =
  # v7's anchor). STANDING always uses home: v7 balances with its arms there,
  # and the relaxed pose while standing slowed real-plant push recovery
  # (0.35 m/s: 0.98 -> 2.1 s). The posture blends home -> relaxed with the gate.
  rest: str = "relaxed"
  rest_shoulder_pitch: float = 0.12
  rest_shoulder_roll: float = 0.14      # abduction magnitude (sign per side)
  rest_elbow: float = 1.20
  # arm swing
  arm_gain: float = 0.75                # shoulder swing / hip scissor angle
  arm_cap: float = 0.20                 # rad, absolute cap on shoulder swing amplitude
  # ... and an acceleration cap: amplitude <= accel_margin * a_max / w^2 at the
  # estimated stride frequency, so the limiter never saturates (saturation adds
  # tens of degrees of lag: v11 strides at 1.28 Hz, where 0.2 rad needs 13 rad/s^2)
  accel_margin: float = 0.85
  arm_back_bias: float = 0.04           # rad, swing centre behind rest while walking
  elbow_gain: float = 0.55              # elbow change per rad of shoulder swing
  elbow_walk_flex: float = 0.10         # extra elbow bend while walking
  elbow_max: float = 1.45               # never lock the elbow straight
  roll_walk: float = 0.04               # extra abduction while walking
  # waist yaw
  # 0: v7's arm swing alone wags the 200/12 waist +-0.06 rad (3.4 deg) in the
  # human counter-rotation phase; an active +0.22 doubles it to an exaggerated
  # +-0.11 rad, -0.22 cancels it (stiff, robotic torso). On the robot's 400/24
  # waist the passive part is ~half, so a small + gain may be wanted there.
  waist_gain: float = 0.0
  waist_cap: float = 0.07
  waist_turn_lead: float = 0.12         # rad per rad/s of commanded yaw rate
  waist_turn_cap: float = 0.10
  # head
  head_stabilise: float = 0.9           # fraction of torso world-yaw / pelvis-pitch wobble cancelled
  head_lead_s: float = 0.05             # predict the torso yaw this far ahead (head PD + interpolation lag)
  head_turn_lead: float = 0.35
  head_turn_cap: float = 0.30
  head_walk_down: float = 0.10          # look slightly down while walking
  head_lateral_lead: float = 0.0        # look toward a sideways step (rad per m/s); 0.25 doubled vy+0.3 heading drift on v7
  # signal conditioning
  hp_hz: float = 0.3
  lp_hz: float = 4.0
  # Phase lead, recomputed from the ONLINE stride-frequency estimate: cancels
  # the band-pass phase plus a fixed delay (limiter + joint PD + interpolation).
  # Tuned on v7 (1.0 Hz); v11 strides at ~1.28 Hz.
  delay_s: float = 0.10
  lead_max_deg: float = 75.0
  f0_hz: float = 1.0                    # initial stride-frequency estimate
  f_range: tuple[float, float] = (0.55, 1.8)
  env_hz: float = 0.6                   # envelope low-pass
  deadband: tuple[float, float] = (0.012, 0.04)   # envelope -> activity 0..1
  heading_hz: float = 0.4               # gaze reference low-pass
  # command gate: swing only when told to walk forward/back. Without it,
  # push-recovery steps start the leg-slaved swing, which feeds back into the
  # legs (real plant, 0.35 m/s push: 3.5 -> 6 recovery steps); side-steps and
  # turns in place got a little off-phase swing too. Humans do neither.
  gate_vx: tuple[float, float] = (0.08, 0.25)     # |vx_cmd| -> gate 0..1
  gate_hz: float = 0.8                            # gate low-pass (fade in/out ~0.4 s)
  # limits (deployed Ruckig: arms v 2.0 / a 8, waist yaw 0.6 / 4)
  vmax: dict = field(default_factory=lambda: {"waist_yaw": 0.6, "head": 1.5, "default": 2.0})
  amax: dict = field(default_factory=lambda: {"waist_yaw": 4.0, "head": 8.0, "default": 8.0})
  joints: str = "all"                   # "all" | "ik" (head + left wrist roll hold)


class _LP1:
  def __init__(self, hz, dt):
    self.a = 1.0 - np.exp(-2 * np.pi * hz * dt)
    self.y = None

  def __call__(self, x):
    self.y = x if self.y is None else self.y + self.a * (x - self.y)
    return self.y


def _smoothstep(x):
  x = np.clip(x, 0.0, 1.0)
  return x * x * (3 - 2 * x)


class HumanUpperBody:
  """Per 50 Hz tick: step(q, R_pelvis, cmd) -> full-width joint targets
  (only the non-policy joints are meant to be used)."""

  IK_JOINTS = frozenset((
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  ))

  def __init__(self, joint_names, home, upper_idx, dt=0.02, cfg: HumanUpperCfg | None = None):
    from mjlab_hv1v2.hv1v2_constants import UPPER_BODY_SAMPLE_RANGE

    self.cfg = cfg or HumanUpperCfg()
    self.dt = dt
    self.names = list(joint_names)
    self.home = np.asarray(home, float).copy()
    self.idx = np.asarray(upper_idx)
    J = self.names.index
    self.j = {k: J(f"{k}_joint") for k in (
      "left_hip_pitch", "right_hip_pitch", "waist_yaw", "head_yaw", "head_pitch",
      "left_shoulder_pitch", "right_shoulder_pitch", "left_shoulder_roll",
      "right_shoulder_roll", "left_elbow", "right_elbow")}
    lo, hi, vm, am = [], [], [], []
    c = self.cfg
    for i in self.idx:
      n = self.names[i]
      r = [v for p, v in UPPER_BODY_SAMPLE_RANGE.items() if re.match(p, n)][0]
      if c.joints == "ik" and n not in self.IK_JOINTS:
        r = (self.home[i], self.home[i])
      lo.append(r[0]); hi.append(r[1])
      key = "waist_yaw" if n.startswith("waist_yaw") else "head" if n.startswith("head") else "default"
      vm.append(c.vmax[key]); am.append(c.amax[key])
    self.lo, self.hi = np.array(lo), np.array(hi)
    self.vmax, self.amax = np.array(vm), np.array(am)
    self.rest = self.home.copy()
    if c.rest == "relaxed":
      self.rest[self.j["left_shoulder_pitch"]] = self.rest[self.j["right_shoulder_pitch"]] = c.rest_shoulder_pitch
      self.rest[self.j["left_shoulder_roll"]] = +c.rest_shoulder_roll
      self.rest[self.j["right_shoulder_roll"]] = -c.rest_shoulder_roll
      self.rest[self.j["left_elbow"]] = self.rest[self.j["right_elbow"]] = c.rest_elbow
    self.reset()

  def reset(self, q=None):
    c, dt = self.cfg, self.dt
    self.hp = _LP1(c.hp_hz, dt)            # high-pass = x - LP(x)
    self.lp_a, self.lp_b = _LP1(c.lp_hz, dt), _LP1(c.lp_hz, dt)
    self.env = _LP1(c.env_hz, dt)
    self.gate_lp = _LP1(c.gate_hz, dt)
    self.yaw_ref = _LP1(c.heading_hz, dt)
    self.pitch_ref = _LP1(c.heading_hz, dt)
    self.yaw_unwrap = None
    self.q = (self.home if q is None else np.asarray(q, float))[self.idx].copy()
    self.v = np.zeros(len(self.idx))
    self.act = 0.0
    self.s = 0.0
    self.s_prev = None
    self.s_bp_prev = None
    self.waist_prev = None
    self.f_est = c.f0_hz
    self.t = 0.0
    self.t_cross = None
    self.goal = self.home.copy()

  # -- one tick -------------------------------------------------------------
  def step(self, q, R, cmd):
    """q: measured joint positions (full width), R: pelvis rotation (world),
    cmd: (vx, vy, wz) velocity command."""
    c, j = self.cfg, self.j
    q = np.asarray(q, float)
    # 1. leg phase signal (band-passed hip scissor angle) + activity gate
    raw = 0.5 * (q[j["right_hip_pitch"]] - q[j["left_hip_pitch"]])
    s = raw - self.hp(raw)
    s = self.lp_b(self.lp_a(s))
    env = self.env(abs(s)) * np.pi / 2              # mean|sin| -> amplitude
    # stride frequency from upward zero crossings (only while the legs swing)
    self.t += self.dt
    if self.s_bp_prev is not None and self.s_bp_prev < 0 <= s and env > c.deadband[0]:
      if self.t_cross is not None:
        T = self.t - self.t_cross
        if 1 / c.f_range[1] <= T <= 1 / c.f_range[0]:
          self.f_est += 0.3 * (1 / T - self.f_est)
      self.t_cross = self.t
    self.s_bp_prev = s
    # lead compensator s + tau ds/dt, unity gain at f_est: the arms would
    # otherwise trail the legs by the band-pass + limiter + joint PD delay
    f = self.f_est
    w0 = 2 * np.pi * f
    lead = (2 * np.degrees(np.arctan(f / c.lp_hz)) - np.degrees(np.arctan(c.hp_hz / f))
            + 360 * f * c.delay_s)
    self.lead_deg = float(np.clip(lead, 0.0, c.lead_max_deg))
    tau = np.tan(np.radians(self.lead_deg)) / w0
    ds = 0.0 if self.s_prev is None else (s - self.s_prev) / self.dt
    self.s_prev = s
    s = (s + tau * ds) / np.sqrt(1 + (w0 * tau) ** 2)
    self.s = s
    lo, hi = c.deadband
    g0, g1 = c.gate_vx
    gate = self.gate_lp(float(_smoothstep((abs(float(cmd[0])) - g0) / (g1 - g0))))
    self.act = float(_smoothstep((env - lo) / (hi - lo))) * gate
    a = self.act

    # 2. pelvis yaw / pitch wobble for the head (world gaze stabilisation)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    if self.yaw_unwrap is None:
      self.yaw_unwrap = yaw
    self.yaw_unwrap += (yaw - self.yaw_unwrap + np.pi) % (2 * np.pi) - np.pi
    yaw_dev = self.yaw_unwrap - self.yaw_ref(self.yaw_unwrap)
    g = R.T @ np.array([0.0, 0.0, -1.0])
    pitch_down = float(np.arctan2(g[0], -g[2]))    # + = pelvis nose down
    pitch_dev = pitch_down - self.pitch_ref(pitch_down)

    rest = self.home + gate * (self.rest - self.home)
    goal = rest.copy()
    # 3. arms: contralateral swing, swing centre slightly behind rest
    a_arm = c.amax.get("default", 8.0)
    cap = min(c.arm_cap, c.accel_margin * a_arm / (2 * np.pi * self.f_est) ** 2)
    u = float(np.clip(c.arm_gain * s, -cap, cap)) * a
    for side, sgn in (("left", +1.0), ("right", -1.0)):
      sp = rest[j[f"{side}_shoulder_pitch"]] + c.arm_back_bias * a + sgn * u
      goal[j[f"{side}_shoulder_pitch"]] = sp
      dev = sp - rest[j[f"{side}_shoulder_pitch"]]       # + = arm behind rest
      el = rest[j[f"{side}_elbow"]] - c.elbow_walk_flex * a + c.elbow_gain * dev
      goal[j[f"{side}_elbow"]] = min(el, c.elbow_max)
      roll_sgn = 1.0 if side == "left" else -1.0
      goal[j[f"{side}_shoulder_roll"]] = rest[j[f"{side}_shoulder_roll"]] + roll_sgn * c.roll_walk * a

    # 4. waist yaw: thorax counter-rotation + lead into turns
    wz = float(cmd[2]); vy = float(cmd[1])
    w = float(np.clip(c.waist_gain * s, -c.waist_cap, c.waist_cap)) * a
    w += float(np.clip(c.waist_turn_lead * wz, -c.waist_turn_cap, c.waist_turn_cap))
    goal[j["waist_yaw"]] = self.home[j["waist_yaw"]] + w

    # 5. head: cancel pelvis yaw wobble + measured waist yaw, look into turns
    hy_base = -c.head_stabilise * yaw_dev
    hy = hy_base + float(np.clip(c.head_turn_lead * wz + c.head_lateral_lead * vy,
                        -c.head_turn_cap, c.head_turn_cap))
    goal[j["head_yaw"]] = self.home[j["head_yaw"]] + hy
    goal[j["head_pitch"]] = (self.home[j["head_pitch"]] + c.head_walk_down * a
                             - c.head_stabilise * pitch_dev)
    self.goal = goal

    # 6. head yaw also cancels the torso's yaw relative to the pelvis: the
    #    MEASURED waist angle (mostly passive, driven by the arms' reaction),
    #    predicted head_lead_s ahead from its velocity
    wq = q[j["waist_yaw"]] - self.home[j["waist_yaw"]]
    wv = 0.0 if self.waist_prev is None else (wq - self.waist_prev) / self.dt
    self.waist_prev = wq
    goal[j["head_yaw"]] -= c.head_stabilise * (wq + c.head_lead_s * wv)

    # 7. clamp into the trained envelope, then v/a limit (Ruckig stand-in:
    #    braking-distance velocity cap, so it never overshoots)
    tgt = np.clip(goal[self.idx], self.lo, self.hi)
    err = tgt - self.q
    v_brake = np.sqrt(2 * self.amax * np.abs(err)) * np.sign(err)
    v_des = np.clip(np.clip(err / self.dt, -np.abs(v_brake) - 1e-9, np.abs(v_brake) + 1e-9),
                    -self.vmax, self.vmax)
    dv = np.clip(v_des - self.v, -self.amax * self.dt, self.amax * self.dt)
    self.v += dv
    self.q += self.v * self.dt
    out = self.home.copy()
    out[self.idx] = self.q
    return out
