"""v11 MDP terms (Mjlab-Velocity-Flat-HV1V2-Decoupled-v11), 2026-11-01.

A separate module so v7-v10 stay byte-identical. Plan:
~/.claude/plans/keen-pondering-fiddle.md. What lives here, by plan section:

  A  plant      V11JointPositionAction (interp + delay + hardware-limit clip +
                rare 5 ms leg dropout), v11_actuator_gains (one reset event for
                PD DR, four-bar stiffness and waist yaw), v11_dr_level
                (curriculum: DR ranges grow from 50% to 100%).
  B  upper body V11UpperBodyCommand (per-joint speeds with caps, 2% single-
                step hold, target rate), HandLoadCommand (smooth hand contact
                forces + mid-episode pick-ups as a ramped gravity wrench),
                TorsoPushCommand, hand_payload_v11 (mass + CoM offset +
                inertia, drawn at reset).
  C  obs        ub_target_rel / ub_target_rate (actor, 1 frame), IMU with
                per-env roll/pitch + gyro bias and rare freezes, privileged
                critic terms.
  D  rewards    upright_tight, torso_quiet_under_ub, loaded_sole_tilt
                (F_z-gated, walking and standing), pad_load_balance,
                ankle_roll_torque_stance (4 N.m deadband).

All tilt rewards read the simulator's TRUE orientation (robot.data), never the
biased/noisy observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from mjlab_hv1v2.mdp import (
  InterpolatedJointPositionAction,
  InterpolatedJointPositionActionCfg,
  StickyStandingVelocityCommandCfg,
  UpperBodyMixedCommand,
  UpperBodyMixedCommandCfg,
  _command_active,
)

_G = 9.81


# --- DR level curriculum ---------------------------------------------------------


def dr_level(env: ManagerBasedRlEnv) -> float:
  """Fraction of the full DR half-width in use: 0.5 early, 1.0 after the ramp."""
  return float(getattr(env, "_v11_dr_level", 1.0))


def scaled_range(lo: float, hi: float, center: float, level: float) -> tuple[float, float]:
  """Shrink (lo, hi) toward `center` to `level` of its half-widths."""
  return center - level * (center - lo), center + level * (hi - center)


def iters_equiv(env: ManagerBasedRlEnv, ref_num_envs: int | None = None,
                steps_per_iter: int = 24) -> float:
  """Training iterations so far (absolute, resume-safe), counted as SAMPLES at
  ref_num_envs envs per iteration when given."""
  it = env.common_step_counter / steps_per_iter
  return it * env.num_envs / ref_num_envs if ref_num_envs else it


class v11_dr_level:
  """Curriculum term: DR level from `lo` to 1.0 over (end_iter - start_iter)
  iterations (sample-keyed at ref_num_envs). The ramp starts at start_iter OR
  when walking first tracks (walk_tracking_error EMA under its threshold),
  whichever is LATER -- widening DR before there is a gait helps nothing
  (2026-11-01: v11_s42 stood still). Every v11 DR term that draws at reset
  reads env._v11_dr_level."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    env._v11_dr_level = float(cfg.params.get("lo", 0.5))
    self.t0: float | None = None

  def __call__(self, env, env_ids, start_iter: int, end_iter: int, lo: float = 0.5,
               steps_per_iter: int = 24, ref_num_envs: int | None = None,
               gate_on_tracking: bool = True, pause_when_closed: bool = False
               ) -> dict[str, torch.Tensor]:
    """pause_when_closed (v11d): the ramp only ADVANCES while the walking gate
    is open (and it >= start_iter), so it holds instead of widening DR while
    the gait collapses (v11c_s43 reached DR 1.0 at 0.00 m/s)."""
    del env_ids
    it = iters_equiv(env, ref_num_envs, steps_per_iter)
    ok = not gate_on_tracking or getattr(env, "_v11_track_ok", False)
    if pause_when_closed:
      prev = getattr(self, "_last_it", None)
      self._last_it = it
      self._prog = getattr(self, "_prog", 0.0)
      if prev is not None and ok and it >= start_iter:
        self._prog += max(it - prev, 0.0)
      a = min(self._prog / max(end_iter - start_iter, 1), 1.0)
      env._v11_dr_level = lo + a * (1.0 - lo)
      return {"level": torch.tensor(env._v11_dr_level), "gate_open": torch.tensor(float(ok))}
    if self.t0 is None and ok:
      self.t0 = max(it, float(start_iter))
    a = 0.0
    if self.t0 is not None:
      a = min(max((it - self.t0) / max(end_iter - start_iter, 1), 0.0), 1.0)
    env._v11_dr_level = lo + a * (1.0 - lo)
    return {"level": torch.tensor(env._v11_dr_level),
            "ramp_started": torch.tensor(float(self.t0 is not None))}


class v11_gated_clock:
  """Curriculum term (v11d, 2026-11-01): a training clock, in sample-keyed
  iterations, that only ADVANCES while walking tracks (walk_tracking_error's
  overall EMA < threshold). The posture and smoothness weight ramps read it
  (reward_weight_ramp clock="_v11_gclock") instead of the iteration count, so
  their prices rise only while the gait holds and freeze when it degrades
  (v11c_s43: the iteration-keyed ramps kept rising while walking died).
  It starts at the iteration of its first call, so a warm start from
  model_N continues the schedules from N (the gate history before N is
  assumed open)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.last: float | None = None

  def __call__(self, env, env_ids, threshold: float = 0.35,
               ref_num_envs: int | None = None, slow_threshold: float | None = None,
               rewind_threshold: float | None = None, rewind_slow: float | None = None,
               fast_threshold: float | None = None, rewind_fast: float | None = None,
               resume_clock: float | None = None) -> dict[str, torch.Tensor]:
    """v11e adds hysteresis: ADVANCE while overall error < threshold (and the
    slow-command error < slow_threshold), REWIND at the same rate while
    overall > rewind_threshold or slow > rewind_slow (the posture/smoothness
    prices ease off when tracking degrades), hold in between."""
    del env_ids
    it = iters_equiv(env, ref_num_envs)
    if self.last is None:
      env._v11_gclock = it
    e = getattr(env, "_v11_track_ema", None)
    es = getattr(env, "_v11_track_slow", None)
    ef = getattr(env, "_v11_track_fast", None)
    ok = e is None or (e < threshold and (slow_threshold is None or es is None or es < slow_threshold)
                       and (fast_threshold is None or ef is None or ef < fast_threshold))
    bad = e is not None and ((rewind_threshold is not None and e > rewind_threshold)
                             or (rewind_slow is not None and es is not None and es > rewind_slow)
                             or (rewind_fast is not None and ef is not None and ef > rewind_fast))
    state = 0.0
    if self.last is not None:
      d = max(it - self.last, 0.0)
      # A jump of > 50 iterations in one call is the runner restoring
      # common_step_counter from a checkpoint (the env was built, and this
      # term first called, at 0): count it, so the schedules resume at N.
      if d > 50.0:
        # resume_clock (v11f): the clock value saved with the checkpoint's
        # run (not stored in the checkpoint), else assume it ran throughout
        env._v11_gclock = float(resume_clock) if resume_clock is not None else env._v11_gclock + d
        state = 1.0
      elif ok:
        env._v11_gclock += d
        state = 1.0
      elif bad:
        env._v11_gclock = max(env._v11_gclock - d, 0.0)
        state = -1.0
    self.last = it
    return {"clock": torch.tensor(env._v11_gclock), "open": torch.tensor(float(ok)),
            "state": torch.tensor(state)}


def _ema(env, name: str, value: float, alpha: float) -> float:
  prev = getattr(env, name, None)
  v = value if prev is None else (1 - alpha) * prev + alpha * value
  setattr(env, name, v)
  return v


def walk_tracking_error(env: ManagerBasedRlEnv, command_name: str = "twist",
                        min_cmd: float = 0.2, ema: float = 0.002,
                        threshold: float = 0.30, dir_cmd: float = 0.15,
                        dir_threshold: float = 0.32, min_swing_s: float = 0.08,
                        sensor_name: str = "feet_ground_contact",
                        slow_range: tuple[float, float] = (0.15, 0.4),
                        slow_threshold: float | None = None,
                        fast_cmd: float = 0.7,
                        fast_threshold: float | None = None) -> torch.Tensor:
  """Metric, every step: the walking-maturity gate for the upper-body level and
  the DR ramp (2026-11-01, v11c). Each quantity is an EMA kept on the env:

    all   mean |v_cmd_xy - v_xy| (m/s, pelvis yaw frame) over envs commanded to
          translate (|v_cmd_xy| > min_cmd)                     < threshold
    back  the same over envs with vx_cmd < -dir_cmd           < dir_threshold
    lat   the same over envs with |vy_cmd| > dir_cmd          < dir_threshold
    swing mean swing (air) time of feet landing in envs commanded to move
                                                               > min_swing_s

  slow  the same over envs with slow_range[0] < |v_cmd_xy| < slow_range[1]
        (v11e: v11d_s43 dropped 0.3 m/s commands while 'all' stayed 0.30)
                                                               < slow_threshold
  fast  the same over envs with |v_cmd_xy| > fast_cmd (v11f: v11e_s43 stood
        still at >= 0.8 m/s commands)                          < fast_threshold
  env._v11_track_ok = all of them. Forward walking alone, or a short-stepping
  shuffle, cannot open the gate (v11b_s43: forward-dominated error passed 0.35
  with backward/lateral near 0 and 0.058 s swings). Direct measurements, not
  mjlab's episode-accumulated error_vel_xy. Returns 0 (logged only)."""
  cmd3 = env.command_manager.get_command(command_name)
  cmd = cmd3[:, :2]
  v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  err = torch.norm(cmd - v, dim=1)
  log = env.extras.setdefault("log", {})
  for key, mask in (("all", torch.norm(cmd, dim=1) > min_cmd),
                    ("back", cmd[:, 0] < -dir_cmd),
                    ("lat", cmd[:, 1].abs() > dir_cmd),
                    ("slow", (torch.norm(cmd, dim=1) > slow_range[0])
                     & (torch.norm(cmd, dim=1) < slow_range[1])),
                    ("fast", torch.norm(cmd, dim=1) > fast_cmd)):
    n = mask.sum()
    if n > 0:
      e = float((err * mask).sum() / n)
      log[f"Metrics/walk_track_err_{key}"] = torch.tensor(e)
      log[f"Metrics/walk_track_err_{key}_ema"] = torch.tensor(_ema(env, f"_v11_track_{key}", e, ema))
  sensor = env.scene[sensor_name]
  first = sensor.compute_first_contact(dt=env.step_dt).float()
  moving = ((torch.norm(cmd, dim=1) + cmd3[:, 2].abs()) > 0.1).float().unsqueeze(1)
  landed = first * moving
  if landed.sum() > 0:
    sw = float((sensor.data.last_air_time * landed).sum() / landed.sum())
    log["Metrics/walk_swing_ema"] = torch.tensor(_ema(env, "_v11_swing", sw, ema))
  e_all = getattr(env, "_v11_track_all", None)
  e_back = getattr(env, "_v11_track_back", None)
  e_lat = getattr(env, "_v11_track_lat", None)
  sw = getattr(env, "_v11_swing", None)
  e_slow = getattr(env, "_v11_track_slow", None)
  ok = (None not in (e_all, e_back, e_lat, sw) and e_all < threshold
        and e_back < dir_threshold and e_lat < dir_threshold and sw > min_swing_s)
  if slow_threshold is not None:
    ok = ok and e_slow is not None and e_slow < slow_threshold
  e_fast = getattr(env, "_v11_track_fast", None)
  if fast_threshold is not None:
    ok = ok and e_fast is not None and e_fast < fast_threshold
  env._v11_track_ok = bool(ok)
  env._v11_track_ema = e_all
  log["Metrics/walk_track_ok"] = torch.tensor(float(ok))
  if e_all is not None:
    log["Metrics/walk_track_err_ema"] = torch.tensor(e_all)
  return torch.zeros(env.num_envs, device=env.device)


class v11_upper_body_level:
  """mdp.upper_body_level, with two v11 gates (2026-11-01):
    * it only PROMOTES while env._v11_track_ok (walk_tracking_error: omni-
      directional tracking AND swing time) -- a robot that stands still or
      shuffles never falls, and v11_s42 reached full arm motion without a gait;
    * promotions are rate-limited to max_rate per 100 iterations (sample-keyed
      at ref_num_envs): 0.05 -> >= 1,800 iterations from 0.1 to 1.0 (v11b_s43
      went 0.0 -> 0.92 in 400).
  Demotion on falls is unchanged and immediate."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.last_promo: float | None = None

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str,
               window: int = 1024, step: float = 0.02, promote_below: float = 0.10,
               demote_above: float = 0.30, max_rate_per_100: float = 0.05,
               ref_num_envs: int | None = None) -> dict[str, torch.Tensor]:
    term = env.command_manager.get_term(command_name)
    finished = env.episode_length_buf[env_ids] > 0
    fell = env.termination_manager.terminated[env_ids] & finished
    term.window_episodes += int(finished.sum())
    term.window_falls += int(fell.sum())
    if term.window_episodes >= window:
      frac = term.window_falls / term.window_episodes
      it = iters_equiv(env, ref_num_envs)
      spacing = step / max_rate_per_100 * 100.0
      rate_ok = self.last_promo is None or it - self.last_promo >= spacing
      if frac < promote_below and getattr(env, "_v11_track_ok", False) and rate_ok:
        term.level = min(1.0, term.level + step)
        self.last_promo = it
      elif frac > demote_above:
        term.level = max(0.0, term.level - step)
      term.window_episodes = 0
      term.window_falls = 0
    return {"level": torch.tensor(term.level)}


# --- A. actuation ------------------------------------------------------------------


class V11JointPositionAction(InterpolatedJointPositionAction):
  """v8's interpolated, delayed leg action, plus:
    * targets clipped to the joint's HARD range (= hardware limits: ankle roll
      +-0.26, knee >= -0.174), as merai_control clips them -- the policy learns
      that asking for 0.38 rad of ankle roll buys nothing;
    * max delay scales with the DR level;
    * rare fault: for one 5 ms substep a leg's target is its measured position
      (no spring force), the closest cheap stand-in for a bus torque dropout.
  """

  cfg: "V11JointPositionActionCfg"

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    lim = self._entity.data.joint_pos_limits[0, self._target_ids]  # (A, 2), hard range
    self._lo, self._hi = lim[:, 0].clone(), lim[:, 1].clone()
    self._drop = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

  def process_actions(self, actions: torch.Tensor):
    first = not hasattr(self, "_p0")
    if not first:
      self._p2, self._p1 = self._p1, self._p0
    JointPositionAction.process_actions(self, actions)
    self._processed_actions = torch.clamp(self._processed_actions, self._lo, self._hi)
    self._p0 = self._processed_actions.clone()
    fresh = self._fresh.unsqueeze(1) | first
    self._p1 = torch.where(fresh, self._p0, self._p1)
    self._p2 = torch.where(fresh, self._p0, self._p2)
    self._fresh[:] = False
    self._substep = 0
    p = self.cfg.dropout_prob
    self._drop = (torch.rand(self.num_envs, 1, device=self.device) < p) if p > 0 else self._drop

  def apply_actions(self) -> None:
    self._substep += 1
    s, D, d = float(self._substep), float(self._D), self.delay
    tail = self._p2 + ((D - d + s) / D) * (self._p1 - self._p2)
    new = self._p1 + ((s - d) / D).clamp(0.0, 1.0) * (self._p0 - self._p1)
    target = torch.where(d >= s, tail, new)
    if self._substep == 1 and self.cfg.dropout_prob > 0:
      q = self._entity.data.joint_pos[:, self._target_ids]
      target = torch.where(self._drop, q, target)
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    lo, hi = self.cfg.delay_substeps_range
    hi_l = int(round(lo + dr_level(self._env) * (hi - lo)))
    k = self.num_envs if env_ids is None else len(env_ids)
    self.delay[ids] = torch.randint(lo, hi_l + 1, (k, 1), device=self.device).float()


@dataclass(kw_only=True)
class V11JointPositionActionCfg(InterpolatedJointPositionActionCfg):
  dropout_prob: float = 0.0   # per env per policy step

  def build(self, env: ManagerBasedRlEnv) -> V11JointPositionAction:
    return V11JointPositionAction(self, env)


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def v11_actuator_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  groups: dict[str, tuple[tuple[str, ...], tuple[float, float], float]],
) -> None:
  """Reset DR for every position actuator, from the DEFAULT gains (so it never
  stacks): per-joint kp/kd scale U(kp_range)/U(kd_range), times a per-group
  stiffness factor for the four-bars, drawn from that group's range shrunk to
  the DR level about its center. groups = {name: (joint regexes, range, center)}.
  ctrl index == joint index (sort_actuators=True). Stores the per-joint total
  kp factor in env._v11_kp_factor for the critic."""
  robot = env.scene["robot"]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  e = env_ids.to(env.device, dtype=torch.long)
  nu = robot.num_joints
  lvl = dr_level(env)
  kp = torch.empty(len(e), nu, device=env.device).uniform_(*scaled_range(*kp_range, 1.0, lvl))
  kd = torch.empty(len(e), nu, device=env.device).uniform_(*scaled_range(*kd_range, 1.0, lvl))
  for patterns, (lo, hi), center in groups.values():
    ids, _ = robot.find_joints(patterns)
    c = torch.tensor(ids, device=env.device, dtype=torch.long)
    s = torch.empty(len(e), 1, device=env.device).uniform_(*scaled_range(lo, hi, center, lvl))
    kp[:, c] *= s
    kd[:, c] *= s
  g0 = env.sim.get_default_field("actuator_gainprm")
  b0 = env.sim.get_default_field("actuator_biasprm")
  ctrl = robot.indexing.ctrl_ids
  m = env.sim.model
  m.actuator_gainprm[e[:, None], ctrl, 0] = g0[ctrl, 0] * kp
  m.actuator_biasprm[e[:, None], ctrl, 1] = b0[ctrl, 1] * kp
  m.actuator_biasprm[e[:, None], ctrl, 2] = b0[ctrl, 2] * kd
  if not hasattr(env, "_v11_kp_factor"):
    env._v11_kp_factor = torch.ones(env.num_envs, nu, device=env.device)
  env._v11_kp_factor[e] = kp


# --- B. upper body ------------------------------------------------------------------


class V11UpperBodyCommand(UpperBodyMixedCommand):
  """v8b's mixed upper-body generator, plus:
    * per-JOINT speeds (v7 used one speed for all 19), redrawn whenever the
      env's base speed is redrawn, each capped by speed_cap (waist yaw 1.0);
    * hold: with prob hold_prob per env per step the target does not advance
      (a dropped teleop packet on the robot);
    * ub_rate = (target_t - target_{t-1}) / step_dt, the exact quantity
      merai_policy computes from its own slewed ub_cmd.
  Apply order per step = robot causal chain: hold -> target -> rate; the
  +-0.1 rad/s noise is the observation term's.
  """

  cfg: "V11UpperBodyCommandCfg"

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n, j = self.num_envs, len(self.joint_names)
    cap = torch.full((j,), float("inf"), device=self.device)
    for name, v in cfg.speed_cap.items():
      if name in self.joint_names:
        cap[self.joint_names.index(name)] = v
    self.cap = cap
    self.jspeed = torch.ones(n, j, device=self.device)
    self._redraw_jspeed(torch.arange(n, device=self.device))
    self.prev_target = self.target.clone()
    self.ub_rate = torch.zeros(n, j, device=self.device)

  def _redraw_jspeed(self, ids: torch.Tensor) -> None:
    if len(ids) == 0:
      return
    lo, hi = self.cfg.speed_range
    s = lo + (hi - lo) * torch.rand(len(ids), len(self.joint_names), device=self.device)
    self.jspeed[ids] = torch.minimum(s, self.cap)

  def reset(self, env_ids):
    out = super().reset(env_ids)
    self.prev_target[env_ids] = self.target[env_ids]
    self.ub_rate[env_ids] = 0.0
    return out

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    before = self.speed[env_ids, 0].clone()
    super()._resample_command(env_ids)
    changed = env_ids[self.speed[env_ids, 0] != before]
    self._redraw_jspeed(changed)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is not None:
      return super()._update_command(env_ids)
    held = torch.rand(self.num_envs, device=self.device) < self.cfg.hold_prob
    t0, ph0 = self.target.clone(), self.phase.clone()
    # per-joint slew: the parent multiplies self.speed (n,1) by step_dt
    base = self.speed
    self.speed = self.jspeed
    super()._update_command(None)
    self.speed = base
    self.target = torch.where(held.unsqueeze(1), t0, self.target)
    self.phase = torch.where(held, ph0, self.phase)
    self.ub_rate = (self.target - self.prev_target) / self._env.step_dt
    self.prev_target = self.target.clone()


@dataclass(kw_only=True)
class V11UpperBodyCommandCfg(UpperBodyMixedCommandCfg):
  speed_cap: dict[str, float] = None  # type: ignore[assignment]
  hold_prob: float = 0.02

  def __post_init__(self):
    if self.speed_cap is None:
      self.speed_cap = {"waist_yaw_joint": 1.0}

  def build(self, env: ManagerBasedRlEnv) -> V11UpperBodyCommand:
    return V11UpperBodyCommand(self, env)


@requires_model_fields("body_mass", "body_ipos", "body_inertia",
                       recompute=RecomputeLevel.set_const)
def hand_payload_v11(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  body_names: tuple[str, ...],
  mass_range: tuple[float, float],
  offset_max: float,
  object_size: float = 0.10,
) -> None:
  """Reset: a payload of U(mass_range) kg per hand, as a box of side
  `object_size`, centred `offset` from the hand body's CoM (|offset| <=
  offset_max, uniform direction). Writes body mass, CoM (body_ipos) and the
  diagonal inertia (box + parallel axis), all from the DEFAULTS so draws never
  stack. Stores mass/offset in env._v11_payload for HandLoadCommand and the
  critic."""
  robot = env.scene["robot"]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  e = env_ids.to(env.device, dtype=torch.long)
  ids, _ = robot.find_bodies(body_names, preserve_order=True)
  b = robot.indexing.body_ids[torch.tensor(ids, device=env.device)]
  k, nb = len(e), len(ids)
  m0 = env.sim.get_default_field("body_mass")[b]            # (nb,)
  c0 = env.sim.get_default_field("body_ipos")[b]            # (nb, 3)
  i0 = env.sim.get_default_field("body_inertia")[b]         # (nb, 3)
  mp = torch.empty(k, nb, device=env.device).uniform_(*mass_range)
  d = torch.randn(k, nb, 3, device=env.device)
  d = d / torch.clamp(torch.norm(d, dim=-1, keepdim=True), min=1e-6)
  r = d * (offset_max * torch.rand(k, nb, 1, device=env.device))
  mt = m0 + mp
  com = c0 + (mp / mt).unsqueeze(-1) * r                    # combined CoM, body frame
  # parallel axis about the new CoM: hand moves by (com - c0), payload by (r - (com - c0))
  dh, dp = com - c0, r - (com - c0)
  box = (mp * object_size**2 / 6.0).unsqueeze(-1).expand(-1, -1, 3)

  def pa(m, v):  # diagonal of m(|v|^2 I - v v^T)
    return m.unsqueeze(-1) * (torch.sum(v * v, dim=-1, keepdim=True) - v * v)

  inertia = i0 + pa(m0.expand(k, nb), dh) + box + pa(mp, dp)
  mdl = env.sim.model
  mdl.body_mass[e[:, None], b] = mt
  mdl.body_ipos[e[:, None], b] = com
  mdl.body_inertia[e[:, None], b] = inertia
  # keyed by hand name: an eval may load one hand only (asymmetric payload)
  store = getattr(env, "_v11_payload_by_body", None)
  if store is None:
    store = env._v11_payload_by_body = {}
  for i, name in enumerate(body_names):
    if name not in store:
      store[name] = torch.zeros(env.num_envs, 4, device=env.device)
    store[name][e] = torch.cat([mp[:, i:i + 1], r[:, i]], dim=-1)


def payload_of(env: ManagerBasedRlEnv, body_names) -> torch.Tensor:
  """(B, len(body_names), 4) = [mass, offset xyz] per hand; zeros for a hand
  that carries no payload in this env."""
  store = getattr(env, "_v11_payload_by_body", {})
  z = torch.zeros(env.num_envs, 4, device=env.device)
  return torch.stack([store.get(n, z) for n in body_names], dim=1)


class _RampedWrench(CommandTerm):
  """Shared machinery: per env and body a trapezoid force pulse (rise, hold,
  fall), written to xfrc_applied every step (forces in world frame, at the
  body CoM). Subclasses add terms to the wrench before it is written."""

  cfg: "_RampedWrenchCfg"

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene["robot"]
    ids, _ = self.robot.find_bodies(cfg.body_names, preserve_order=True)
    self.body_ids = ids
    n, nb = self.num_envs, len(ids)
    self.F = torch.zeros(n, nb, 3, device=self.device)
    self.t = torch.full((n, nb), 1e9, device=self.device)        # time since pulse start
    self.rise = torch.ones(n, nb, device=self.device)
    self.hold = torch.zeros(n, nb, device=self.device)
    self.force = torch.zeros(n, nb, 3, device=self.device)
    self.torque = torch.zeros(n, nb, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.force.flatten(1)

  def _envelope(self) -> torch.Tensor:
    t, r, h = self.t, self.rise, self.hold
    up = torch.clamp(t / r, 0.0, 1.0)
    down = torch.clamp(1.0 - (t - r - h) / r, 0.0, 1.0)
    return torch.where(t < r + h, up, down)

  def reset(self, env_ids):
    self.t[env_ids] = 1e9
    self.F[env_ids] = 0.0
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    c = self.cfg
    busy = self.t[env_ids] < (self.rise[env_ids] * 2 + self.hold[env_ids])
    start = (torch.rand(len(env_ids), len(self.body_ids), device=self.device) < c.active_prob) & ~busy
    if not start.any():
      return
    lvl = dr_level(self._env)
    k, nb = len(env_ids), len(self.body_ids)
    F = (torch.rand(k, nb, 3, device=self.device) * 2 - 1) * c.force_max * lvl
    if c.opposite_prob > 0 and nb == 2:
      opp = torch.rand(k, device=self.device) < c.opposite_prob
      F[:, 1] = torch.where(opp.unsqueeze(1), -F[:, 0], F[:, 1])
    rise = c.rise_range[0] + (c.rise_range[1] - c.rise_range[0]) * torch.rand(k, nb, device=self.device)
    hold = c.hold_range[0] + (c.hold_range[1] - c.hold_range[0]) * torch.rand(k, nb, device=self.device)
    sel = start
    e2 = env_ids.unsqueeze(1).expand(k, nb)[sel]
    b2 = torch.arange(nb, device=self.device).unsqueeze(0).expand(k, nb)[sel]
    self.F[e2, b2] = F[sel]
    self.rise[e2, b2] = rise[sel]
    self.hold[e2, b2] = hold[sel]
    self.t[e2, b2] = 0.0

  def _extra_wrench(self) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.zeros_like(self.force)
    return z, z

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      self.t = self.t + self._env.step_dt
    f = self.F * self._envelope().unsqueeze(-1)
    fx, tx = self._extra_wrench()
    self.force = f + fx
    self.torque = tx
    self.robot.write_external_wrench_to_sim(self.force, self.torque, body_ids=self.body_ids)

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class _RampedWrenchCfg(CommandTermCfg):
  body_names: tuple[str, ...] = ()
  active_prob: float = 0.5
  force_max: float = 30.0
  rise_range: tuple[float, float] = (0.2, 1.0)
  hold_range: tuple[float, float] = (1.0, 4.0)
  opposite_prob: float = 0.0


class HandLoadCommand(_RampedWrench):
  """Hand contact forces (smooth trapezoids, +-force_max N per axis, sometimes
  equal and opposite on the two hands) PLUS mid-episode pick-ups/put-downs.

  A pick-up is a gravity wrench F = (0, 0, -m_x g) at the payload point,
  ramped linearly over pickup_ramp_s; m_x moves the hand's total to a new
  U(0, mass_max) kg. Changing body_mass mid-episode would need MuJoCo's
  set_const every step of the ramp, so mid-episode load changes carry the
  static load (force + moment), not the extra inertia; the reset payload
  (hand_payload_v11) carries real mass and inertia."""

  cfg: "HandLoadCommandCfg"

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n, nb = self.num_envs, len(self.body_ids)
    self.mx = torch.zeros(n, nb, device=self.device)        # current extra mass
    self.mx_from = torch.zeros(n, nb, device=self.device)
    self.mx_to = torch.zeros(n, nb, device=self.device)
    self.mx_t = torch.full((n, nb), 1e9, device=self.device)

  def reset(self, env_ids):
    self.mx[env_ids] = 0.0
    self.mx_from[env_ids] = 0.0
    self.mx_to[env_ids] = 0.0
    self.mx_t[env_ids] = 1e9
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    c = self.cfg
    k, nb = len(env_ids), len(self.body_ids)
    go = torch.rand(k, nb, device=self.device) < c.pickup_prob
    if not go.any():
      return
    m_reset = payload_of(self._env, self.cfg.body_names)[env_ids, :, 0]
    new_total = c.mass_max * torch.rand(k, nb, device=self.device)
    tgt = new_total - m_reset
    sel = go
    e2 = env_ids.unsqueeze(1).expand(k, nb)[sel]
    b2 = torch.arange(nb, device=self.device).unsqueeze(0).expand(k, nb)[sel]
    self.mx_from[e2, b2] = self.mx[e2, b2]
    self.mx_to[e2, b2] = tgt[sel]
    self.mx_t[e2, b2] = 0.0

  def _extra_wrench(self) -> tuple[torch.Tensor, torch.Tensor]:
    self.mx_t = self.mx_t + self._env.step_dt
    a = torch.clamp(self.mx_t / self.cfg.pickup_ramp_s, 0.0, 1.0)
    self.mx = self.mx_from + a * (self.mx_to - self.mx_from)
    F = torch.zeros_like(self.force)
    F[..., 2] = -self.mx * _G
    # moment about the body CoM of a load hung at the payload offset (body frame)
    base = payload_of(self._env, self.cfg.body_names)
    q = self.robot.data.body_link_quat_w[:, self.body_ids]
    r_w = quat_apply(q, base[..., 1:4])
    return F, torch.cross(r_w, F, dim=-1)


@dataclass(kw_only=True)
class HandLoadCommandCfg(_RampedWrenchCfg):
  pickup_prob: float = 0.3
  pickup_ramp_s: float = 0.2
  mass_max: float = 2.0

  def build(self, env: ManagerBasedRlEnv) -> HandLoadCommand:
    return HandLoadCommand(self, env)


@dataclass(kw_only=True)
class TorsoPushCommandCfg(_RampedWrenchCfg):
  """Short force pushes on the torso while the arms move (plus v7's velocity
  pushes, which stay)."""

  def build(self, env: ManagerBasedRlEnv) -> _RampedWrench:
    return _RampedWrench(self, env)


class ReversalVelocityCommand(UniformVelocityCommand):
  """Velocity command with explicit stops and reversals: at a resample, with
  prob reversal_prob the new command is the NEGATED previous one (the vy
  -0.5 -> +0.5 case that made v5ft clip ankle roll), with prob stop_prob it is
  zero; standing draws hold for stand_hold_range s."""

  cfg: "ReversalVelocityCommandCfg"

  def _resample(self, env_ids: torch.Tensor) -> None:
    prev = self.vel_command_b[env_ids].clone()
    super()._resample(env_ids)
    if len(env_ids) == 0:
      return
    u = torch.rand(len(env_ids), device=self.device)
    rev = (u < self.cfg.reversal_prob) & (prev.abs().sum(1) > 0.1)
    self.vel_command_b[env_ids[rev]] = -prev[rev]
    stop = (u >= self.cfg.reversal_prob) & (u < self.cfg.reversal_prob + self.cfg.stop_prob)
    self.vel_command_b[env_ids[stop]] = 0.0
    st = env_ids[self.is_standing_env[env_ids]]
    if len(st) > 0:
      lo, hi = self.cfg.stand_hold_range
      self.time_left[st] = lo + (hi - lo) * torch.rand(len(st), device=self.device)


@dataclass(kw_only=True)
class ReversalVelocityCommandCfg(StickyStandingVelocityCommandCfg):
  reversal_prob: float = 0.15
  stop_prob: float = 0.10

  def build(self, env: ManagerBasedRlEnv) -> ReversalVelocityCommand:
    return ReversalVelocityCommand(self, env)


# --- C. observations ------------------------------------------------------------------


def _ub_cols(env: ManagerBasedRlEnv, command_name: str, joint_names) -> torch.Tensor:
  key = ("_v11_ub_cols", command_name, tuple(joint_names))
  cache = getattr(env, "_v11_cache", None)
  if cache is None:
    cache = env._v11_cache = {}
  if key not in cache:
    term = env.command_manager.get_term(command_name)
    cache[key] = torch.tensor([term.joint_names.index(n) for n in joint_names],
                              device=env.device, dtype=torch.long)
  return cache[key]


def ub_target_rel(env: ManagerBasedRlEnv, command_name: str, joint_names) -> torch.Tensor:
  """Commanded upper-body target relative to home, for `joint_names` (in that order)."""
  term = env.command_manager.get_term(command_name)
  c = _ub_cols(env, command_name, joint_names)
  return term.target[:, c] - term.home[:, c]


def ub_target_rate(env: ManagerBasedRlEnv, command_name: str, joint_names) -> torch.Tensor:
  """(target_t - target_{t-1}) / step_dt for `joint_names`."""
  term = env.command_manager.get_term(command_name)
  return term.ub_rate[:, _ub_cols(env, command_name, joint_names)]


class _ImuState:
  """Per-env IMU bias (roll/pitch mounting error, gyro bias) and freeze faults,
  shared by the two IMU observation terms."""

  def __init__(self, env: ManagerBasedRlEnv):
    n = env.num_envs
    self.rp = torch.zeros(n, 2, device=env.device)
    self.gyro = torch.zeros(n, 3, device=env.device)
    self.freeze = torch.zeros(n, dtype=torch.long, device=env.device)
    self.counter = -1
    self.g_hold = torch.zeros(n, 3, device=env.device)
    self.w_hold = torch.zeros(n, 3, device=env.device)
    self.g_hold[:, 2] = -1.0


def _imu(env: ManagerBasedRlEnv) -> _ImuState:
  s = getattr(env, "_v11_imu", None)
  if s is None:
    s = env._v11_imu = _ImuState(env)
  return s


def encoder_bias_scaled(env: ManagerBasedRlEnv, env_ids, bias_max: float,
                        asset_cfg: SceneEntityCfg) -> None:
  """Reset event: encoder zero offset U(+-bias_max * DR level) on asset_cfg joints."""
  from mjlab.envs.mdp import dr

  b = bias_max * dr_level(env)
  dr.encoder_bias(env, env_ids, bias_range=(-b, b), asset_cfg=asset_cfg)


def imu_bias_v11(env: ManagerBasedRlEnv, env_ids, rp_max_rad: float, gyro_max: float) -> None:
  """Reset event: draw the IMU's roll/pitch mounting bias and gyro bias."""
  s = _imu(env)
  ids = slice(None) if env_ids is None else env_ids
  k = env.num_envs if env_ids is None else len(env_ids)
  lvl = dr_level(env)
  s.rp[ids] = (torch.rand(k, 2, device=env.device) * 2 - 1) * rp_max_rad * lvl
  s.gyro[ids] = (torch.rand(k, 3, device=env.device) * 2 - 1) * gyro_max * lvl
  s.freeze[ids] = 0


def _imu_step(env: ManagerBasedRlEnv, freeze_prob: float, freeze_steps: tuple[int, int]) -> _ImuState:
  s = _imu(env)
  if env.common_step_counter != s.counter:
    s.counter = env.common_step_counter
    s.freeze = torch.clamp(s.freeze - 1, min=0)
    if freeze_prob > 0:
      start = (torch.rand(env.num_envs, device=env.device) < freeze_prob) & (s.freeze == 0)
      if start.any():
        n = torch.randint(freeze_steps[0], freeze_steps[1] + 1, (int(start.sum()),), device=env.device)
        s.freeze[start] = n
  return s


def projected_gravity_v11(env: ManagerBasedRlEnv, freeze_prob: float = 0.0,
                          freeze_steps: tuple[int, int] = (1, 5)) -> torch.Tensor:
  """Projected gravity as the robot's IMU reports it: tilted by the per-env
  mounting bias, frozen during an IMU-stall fault."""
  s = _imu_step(env, freeze_prob, freeze_steps)
  g = env.scene["robot"].data.projected_gravity_b
  r, p = s.rp[:, 0], s.rp[:, 1]
  # small rotation (roll about x, then pitch about y) of the measured vector
  gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
  gy2 = gy * torch.cos(r) - gz * torch.sin(r)
  gz2 = gy * torch.sin(r) + gz * torch.cos(r)
  gx3 = gx * torch.cos(p) + gz2 * torch.sin(p)
  gz3 = -gx * torch.sin(p) + gz2 * torch.cos(p)
  out = torch.stack([gx3, gy2, gz3], dim=1)
  frozen = (s.freeze > 0).unsqueeze(1)
  out = torch.where(frozen, s.g_hold, out)
  s.g_hold = out.clone()
  return out


def base_ang_vel_v11(env: ManagerBasedRlEnv, sensor_name: str, freeze_prob: float = 0.0,
                     freeze_steps: tuple[int, int] = (1, 5)) -> torch.Tensor:
  """Gyro as the robot reports it: + per-env bias, frozen during an IMU stall."""
  from mjlab.envs import mdp as envs_mdp

  s = _imu_step(env, freeze_prob, freeze_steps)
  w = envs_mdp.builtin_sensor(env, sensor_name=sensor_name) + s.gyro
  frozen = (s.freeze > 0).unsqueeze(1)
  out = torch.where(frozen, s.w_hold, w)
  s.w_hold = out.clone()
  return out


def privileged_plant(env: ManagerBasedRlEnv, leg_joint_names) -> torch.Tensor:
  """Critic: leg kp factors, action delay, payload (mass + offset x2), hand
  load extra mass, IMU bias."""
  robot = env.scene["robot"]
  key = ("_v11_leg_ids", tuple(leg_joint_names))
  cache = getattr(env, "_v11_cache", None) or {}
  env._v11_cache = cache
  if key not in cache:
    ids, _ = robot.find_joints(leg_joint_names, preserve_order=True)
    cache[key] = torch.tensor(ids, device=env.device, dtype=torch.long)
  n = env.num_envs
  kpf = getattr(env, "_v11_kp_factor", torch.ones(n, robot.num_joints, device=env.device))[:, cache[key]]
  act = env.action_manager.get_term("joint_pos")
  delay = getattr(act, "delay", torch.zeros(n, 1, device=env.device)) / 4.0
  from mjlab_hv1v2.hv1v2_constants import HAND_BODY_NAMES

  pay = payload_of(env, HAND_BODY_NAMES).flatten(1)
  hl = env.command_manager.get_term("hand_load")
  imu = _imu(env)
  return torch.cat([kpf, delay, pay, hl.mx, imu.rp, imu.gyro], dim=1)


# --- D. rewards --------------------------------------------------------------------------


def upright_tight(env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """exp(-|g_xy|^2 / std^2) of the TRUE pelvis orientation. v7's std 0.447
  paid 0.96 at 5 deg of tilt; std 0.08 pays 0.95 / 0.83 / 0.31 at 1 / 2 / 5 deg."""
  robot = env.scene[asset_cfg.name]
  q = robot.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
  g = quat_apply_inverse(q, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(q.shape[0], 3))
  xy = torch.sum(torch.square(g[:, :2]), dim=1)
  env.extras.setdefault("log", {})["Metrics/pelvis_tilt_deg"] = torch.rad2deg(
    torch.asin(torch.clamp(torch.sqrt(xy), max=1.0))).mean()
  return torch.exp(-xy / std**2)


def torso_quiet_under_ub(
  env: ManagerBasedRlEnv,
  command_name: str,
  rate_ref: float,
  w_weight: float,
  asset_cfg: SceneEntityCfg,
  twist_name: str | None = None,
  walk_cmd: float = 0.05,
  walk_scale: float = 1.0,
) -> torch.Tensor:
  """gate * (|g_xy|^2 + w_weight * |w_xy|^2) of the TRUE pelvis, where gate =
  clamp(|commanded upper-body rate| / rate_ref, 0, 1): priced only WHILE the
  upper body moves, so it targets the response to the arms and cannot be
  satisfied by a slow lean at rest (v8's unconditional pelvis_quiet did that)."""
  robot = env.scene[asset_cfg.name]
  b = asset_cfg.body_ids[0]
  q = robot.data.body_link_quat_w[:, b]
  g = quat_apply_inverse(q, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(q.shape[0], 3))
  w = robot.data.body_link_ang_vel_w[:, b]
  w_b = quat_apply_inverse(q, w)
  term = env.command_manager.get_term(command_name)
  gate = torch.clamp(torch.norm(term.ub_rate, dim=1) / rate_ref, 0.0, 1.0)
  raw = torch.sum(torch.square(g[:, :2]), dim=1) + w_weight * torch.sum(torch.square(w_b[:, :2]), dim=1)
  if twist_name is not None:
    raw = raw * walk_cmd_scale(env, twist_name, walk_cmd, walk_scale)
  return gate * raw


def walk_cmd_scale(env: ManagerBasedRlEnv, command_name: str, walk_cmd: float,
                   walk_scale: float) -> torch.Tensor:
  """1 when |v_cmd_xy| < walk_cmd (standing / turning in place), walk_scale
  when commanded to translate (v11d)."""
  cmd = env.command_manager.get_command(command_name)[:, :2]
  walking = torch.norm(cmd, dim=1) >= walk_cmd
  return torch.where(walking, torch.full_like(cmd[:, 0], walk_scale), torch.ones_like(cmd[:, 0]))


def body_ang_vel_cmd_scaled(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg,
                            command_name: str = "twist", walk_cmd: float = 0.05,
                            walk_scale: float = 0.25) -> torch.Tensor:
  """mjlab body_angular_velocity_penalty (xy), x walk_scale while commanded to
  translate (v11d, 2026-11-01): walking needs ~0.3-0.6 rad/s of pelvis
  roll/pitch rate to shift weight; the full price stays on standing, where
  it targets the 1-2 Hz rocking."""
  asset = env.scene[asset_cfg.name]
  w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.sum(torch.square(w[:, :2]), dim=1) * walk_cmd_scale(env, command_name, walk_cmd, walk_scale)


def feet_air_time_tracked(env: ManagerBasedRlEnv, sensor_name: str, threshold_min: float = 0.05,
                          threshold_max: float = 0.5, command_name: str | None = None,
                          command_threshold: float = 0.5, track_var: float = 0.25) -> torch.Tensor:
  """mjlab feet_air_time x exp(-|v_xy - v_cmd_xy|^2 / track_var) (v11d,
  2026-11-01): lifting the feet pays only when the body actually moves as
  commanded, so marching in place under a walk command earns little
  (exp(-1) = 0.37 at 0.5 m/s short, exp(-4) = 0.02 at 1.0 m/s)."""
  from mjlab.tasks.velocity.mdp.rewards import feet_air_time
  r = feet_air_time(env, sensor_name, threshold_min, threshold_max, command_name, command_threshold)
  cmd = env.command_manager.get_command(command_name or "twist")[:, :2]
  v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  return r * torch.exp(-torch.sum(torch.square(v - cmd), dim=1) / track_var)


def _foot_fz(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """|F_z| per primary of a netforce contact sensor, (B, n_primary), in the
  sensor's primary_names order."""
  sensor: ContactSensor = env.scene[sensor_name]
  f = sensor.data.force
  assert f is not None, f"{sensor_name} needs fields=('found','force')"
  return f[..., 2].abs().reshape(env.num_envs, -1)


def _sensor_ids(env: ManagerBasedRlEnv, sensor_name: str, kind: str) -> torch.Tensor:
  """Robot body / joint indices matching the sensor's primaries, in its order.
  kind 'body': the primary bodies; 'roll_joint': '<side>_ankle_roll_joint'
  of each primary foot. Foot order is the SENSOR's (never assumed)."""
  key = ("_v11_sensor_ids", sensor_name, kind)
  cache = getattr(env, "_v11_cache", None)
  if cache is None:
    cache = env._v11_cache = {}
  if key not in cache:
    robot = env.scene["robot"]
    names = env.scene[sensor_name].primary_names
    if kind == "body":
      ids, got = robot.find_bodies([f"^{n}$" for n in names], preserve_order=True)
    else:
      jn = [n.replace("_ankle_roll_link", "_ankle_roll_joint") for n in names]
      ids, got = robot.find_joints([f"^{n}$" for n in jn], preserve_order=True)
    assert len(ids) == len(names), (sensor_name, kind, names, got)
    cache[key] = torch.tensor(ids, device=env.device, dtype=torch.long)
  return cache[key]


def loaded_sole_tilt(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  fz_min: float,
) -> torch.Tensor:
  """Sum over feet of sole tilt (rad) for every foot carrying > fz_min N --
  standing AND walking. v10's stand_sole_tilt only priced standing with both
  feet down; heel strike / toe-off stay unpriced by the force gate. Feet are
  taken in the sensor's own order."""
  data = env.scene["robot"].data
  q = data.body_link_quat_w[:, _sensor_ids(env, sensor_name, "body")]
  down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(q.shape[:-1] + (3,))
  g = quat_apply_inverse(q, down)
  tilt = torch.atan2(torch.norm(g[..., :2], dim=-1), -g[..., 2])
  loaded = (_foot_fz(env, sensor_name) > fz_min).float()
  n = torch.clamp(loaded.sum(), min=1.0)
  env.extras.setdefault("log", {})["Metrics/loaded_sole_tilt_deg"] = torch.rad2deg(
    torch.sum(tilt * loaded) / n)
  return torch.sum(tilt * loaded, dim=1)


def pad_load_balance(
  env: ManagerBasedRlEnv,
  pad_sensor_name: str,
  command_name: str,
  command_threshold: float,
  min_share: float,
  fz_min: float,
) -> torch.Tensor:
  """While commanded to stand: sum over the 8 sole pads of
  relu(min_share - pad_share)/min_share, pad_share = the pad's vertical load /
  its foot's (4 pads). 0 when every pad carries >= min_share of its foot.
  Prices standing on an edge (v7: 2 of 4 pads, 25 N.m roll preload). Logs
  Metrics/pads_loaded_frac (pads above min_share, standing, loaded feet)."""
  # group the 8 pads by foot from the sensor's own names
  key = ("_v11_pad_perm", pad_sensor_name)
  cache = getattr(env, "_v11_cache", None) or {}
  env._v11_cache = cache
  if key not in cache:
    names = env.scene[pad_sensor_name].primary_names
    order = [i for side in ("left", "right") for i, n in enumerate(names) if n.startswith(side)]
    assert len(order) == 8, names
    cache[key] = torch.tensor(order, device=env.device, dtype=torch.long)
  fz = _foot_fz(env, pad_sensor_name)[:, cache[key]].reshape(env.num_envs, 2, 4)
  foot = fz.sum(dim=2, keepdim=True)
  loaded = (foot > fz_min).float()
  share = fz / torch.clamp(foot, min=1e-6)
  cost = (torch.clamp(min_share - share, min=0.0) / min_share * loaded).sum(dim=(1, 2))
  standing = 1.0 - _command_active(env, command_name, command_threshold)
  ok = ((share >= min_share).float() * loaded).sum(dim=(1, 2))
  n = torch.clamp((loaded.sum(dim=(1, 2)) * 4 * standing).sum(), min=1.0)
  env.extras.setdefault("log", {})["Metrics/pads_loaded_frac"] = torch.sum(ok * standing) / n
  return cost * standing


def ankle_roll_torque_stance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  fz_min: float,
  deadband: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  single_support_scale: float = 1.0,
) -> torch.Tensor:
  """sum over feet carrying > fz_min N of relu(|tau_ankle_roll| - deadband)^2.
  The deadband (4 N.m) leaves the balancing authority an asymmetric payload
  needs and prices the steady preload v7 held (25 N.m). Each foot's own ankle
  roll joint is found from the sensor's foot names.

  single_support_scale (v11 fix, 2026-11-01): while WALKING (command active)
  with exactly one foot loaded, the cost is scaled by it (0.25) -- the stance
  ankle must shift the CoP to carry a single-leg stance; double support and
  standing stay at 1.0."""
  tau = env.scene["robot"].data.actuator_force[:, _sensor_ids(env, sensor_name, "roll_joint")]
  loaded = (_foot_fz(env, sensor_name) > fz_min).float()
  excess = torch.clamp(tau.abs() - deadband, min=0.0)
  n = torch.clamp(loaded.sum(), min=1.0)
  env.extras.setdefault("log", {})["Metrics/stance_ankle_roll_tau"] = torch.sum(tau.abs() * loaded) / n
  cost = torch.sum(torch.square(excess) * loaded, dim=1)
  if command_name is not None and single_support_scale != 1.0:
    single = (loaded.sum(dim=1) == 1).float() * _command_active(env, command_name, command_threshold)
    cost = cost * (1.0 - single * (1.0 - single_support_scale))
  return cost


# --- v11g: standing internal-force penalty ---------------------------------------


def stand_leg_torque(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  deadband: dict[str, float],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """v11g (2026-11-02): while commanded to stand, sum over the hip roll, hip
  pitch and knee joints of max(|tau| - deadband_joint, 0)^2 (N.m^2). The
  deadbands cover what holding the body up needs (sim2sim standing: hip roll
  <= 7, hip pitch <= 13, knee <= 21 N.m).

  Why: on the robot (2026-11-02 sessions) v11f held a SQUEEZE at zero command,
  either spreading (hip roll -55/+59 N.m) or scissoring (hip pitch +62/-44).
  The two legs' forces cancel through the planted feet, so no joint moves and
  the observation does not see it; the replay showed the policy simply holds
  its previous action. Only a torque price makes those equilibria costly.
  Logs Metrics/stand_leg_tau_excess."""
  robot = env.scene[asset_cfg.name]
  key = ("_v11g_tau_ids", tuple(sorted(deadband)))
  cache = getattr(env, "_v11_cache", None)
  if cache is None:
    cache = env._v11_cache = {}
  if key not in cache:
    ids, db = [], []
    for pat, d in deadband.items():
      jid, _ = robot.find_joints((pat,), preserve_order=True)
      ids += list(jid)
      db += [d] * len(jid)
    cache[key] = (torch.tensor(ids, device=env.device, dtype=torch.long),
                  torch.tensor(db, device=env.device))
  ids, db = cache[key]
  tau = robot.data.actuator_force[:, ids]   # actuators are joint-ordered (sort_actuators)
  excess = torch.clamp(tau.abs() - db, min=0.0)
  m = 1.0 - _command_active(env, command_name, command_threshold)
  cost = torch.sum(torch.square(excess), dim=1) * m
  n = torch.clamp(m.sum(), min=1.0)
  env.extras.setdefault("log", {})["Metrics/stand_leg_tau_excess"] = torch.sum(
    torch.sqrt(torch.sum(torch.square(excess), dim=1)) * m) / n
  return cost
