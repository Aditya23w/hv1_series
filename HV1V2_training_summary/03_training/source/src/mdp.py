"""HV1V2-specific MDP terms.

Out-of-tree, so nothing here can be imported from mjlab's own task configs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.string import resolve_expr
from mjlab_hv1v2.hv1v2_constants import CONTINUOUS_TORQUE

_LIMIT_CACHE: dict[int, torch.Tensor] = {}


def _continuous_limits(env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
  """Per-actuator continuous torque rating, in ctrl order. Shape ``(1, nu)``."""
  key = id(env)
  cached = _LIMIT_CACHE.get(key)
  if cached is not None:
    return cached

  asset = env.scene[asset_name]
  names = tuple(asset.joint_names)
  nu = asset.data.actuator_force.shape[1]
  assert len(names) == nu, (
    f"{len(names)} joints but {nu} actuators; the torque-duty metric assumes "
    "one actuator per joint in joint order (sort_actuators=True)"
  )
  vals = resolve_expr(CONTINUOUS_TORQUE, names, None)
  missing = [n for n, v in zip(names, vals) if v is None]
  assert not missing, f"no continuous torque rating for {missing}"

  limits = torch.tensor(
    [float(v) for v in vals], dtype=torch.float32, device=asset.data.actuator_force.device
  ).unsqueeze(0)
  _LIMIT_CACHE[key] = limits
  return limits


def torque_duty_frac(env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
  """Fraction of actuators currently above their CONTINUOUS torque rating.

  ``effort_limit`` in the plant is the vendor's PEAK torque, which is only a
  3-10 second rating (X12 at 300 N.m -> 3 s; X8 at 86 N.m -> 10 s). Nothing in
  the reward notices a gait that lives up there, so it gets measured instead.

  Returns:
    Per-environment scalar in [0, 1]. Shape ``(B,)``.
  """
  asset = env.scene[asset_name]
  tau = asset.data.actuator_force.abs()
  return (tau > _continuous_limits(env, asset_name)).float().mean(dim=1)


def torque_frac_of_continuous(
  env: ManagerBasedRlEnv, asset_name: str = "robot"
) -> torch.Tensor:
  """Worst actuator's torque as a fraction of its continuous rating.

  1.0 means some joint is exactly at its continuous limit.

  Returns:
    Per-environment scalar. Shape ``(B,)``.
  """
  asset = env.scene[asset_name]
  tau = asset.data.actuator_force.abs()
  return (tau / _continuous_limits(env, asset_name)).max(dim=1).values


def torque_frac_legs(env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
  """Same as above but restricted to the 12 leg joints, which carry the robot.

  Returns:
    Per-environment scalar. Shape ``(B,)``.
  """
  asset = env.scene[asset_name]
  names = tuple(asset.joint_names)
  idx = [
    i for i, n in enumerate(names)
    if any(k in n for k in ("hip", "knee", "ankle"))
  ]
  tau = asset.data.actuator_force[:, idx].abs()
  return (tau / _continuous_limits(env, asset_name)[:, idx]).max(dim=1).values


##
# Gait shaping (v5).
#
# v4, measured in sim2sim on exported/deploy_v4: 0.395 m steps at vx 1.0 and
# 0.35-0.53 s swings (longest at the SLOWEST commands), which is what mjlab's
# feet_air_time asks for -- it pays every step a foot spends inside its window,
# so a swing earns more the longer it lasts. These terms bound the swing from
# the other side. The window itself is narrowed in env_cfgs.py.
##


def _command_active(
  env: ManagerBasedRlEnv, command_name: str, command_threshold: float
) -> torch.Tensor:
  """1.0 where the commanded twist is large enough to call for stepping."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (total > command_threshold).float()


def feet_flight(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1.0 while BOTH feet are off the ground, i.e. a running flight phase.

  v4 measured 0.000 double-flight in sim2sim, so this is a fence, not a fix:
  narrowing the air_time window removes the only thing that paid for a flight
  phase, and nothing else would stop one appearing. Not command-gated -- a
  flight phase is never wanted.

  Returns:
    Shape ``(B,)``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.current_air_time
  assert air is not None, f"{sensor_name} needs track_air_time=True"
  return (air > 0).all(dim=1).float()


def air_time_excess(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  t_max: float,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Seconds each foot has overrun ``t_max`` in the current swing, summed.

  feet_air_time merely stops paying above its window; this makes overrunning
  it cost something, and the cost grows for as long as the foot stays up.

  Returns:
    Shape ``(B,)``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.current_air_time
  assert air is not None, f"{sensor_name} needs track_air_time=True"
  excess = torch.clamp(air - t_max, min=0.0).sum(dim=1)
  return excess * _command_active(env, command_name, command_threshold)


class step_length_excess:
  """Penalise touchdowns that land more than ``max_length`` ahead of/behind
  the other foot, measured along the robot's heading.

  Evaluated once per step, at first contact: landing foot site minus the other
  foot's site, rotated into the yaw-only base frame, x component. That is the
  same quantity ``scripts/sim2sim.py --headless`` reports as "step length,
  heading axis", so training and evaluation numbers are directly comparable.
  Lateral offset is deliberately excluded: the stance width alone is ~0.24 m.

  Also logs Metrics/step_length_mean and Metrics/swing_time_mean (over the
  command-active landings of this step), the same way upstream
  feet_swing_height logs peak_height_mean.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    assert asset_cfg.site_names is not None and len(asset_cfg.site_names) == 2, (
      "step_length_excess needs exactly two foot sites, in contact-sensor order"
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_length: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    asset = env.scene[asset_cfg.name]
    first_contact = sensor.compute_first_contact(dt=self.step_dt).float()  # [B, 2]
    last_air = sensor.data.last_air_time
    assert last_air is not None

    feet_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    # Foot i's offset from the OTHER foot. With exactly two feet that is
    # [left - right, right - left].
    delta = feet_xy - feet_xy.flip(1)
    yaw = asset.data.heading_w
    dx = torch.abs(
      torch.cos(yaw).unsqueeze(1) * delta[..., 0]
      + torch.sin(yaw).unsqueeze(1) * delta[..., 1]
    )  # [B, 2]

    active = _command_active(env, command_name, command_threshold)
    landed = first_contact * active.unsqueeze(1)
    cost = torch.sum(torch.clamp(dx - max_length, min=0.0) * landed, dim=1)

    n = torch.clamp(landed.sum(), min=1.0)
    env.extras["log"]["Metrics/step_length_mean"] = torch.sum(dx * landed) / n
    env.extras["log"]["Metrics/swing_time_mean"] = torch.sum(last_air * landed) / n
    return cost


def double_flight_frac(
  env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
  """Metric: 1.0 on steps with both feet airborne. Episode mean = fraction.

  Returns:
    Shape ``(B,)``.
  """
  return feet_flight(env, sensor_name)


def arm_joint_speed(
  env: ManagerBasedRlEnv, asset_name: str = "robot"
) -> torch.Tensor:
  """Metric: RMS joint speed over the shoulders and elbows, rad/s.

  Same definition as sim2sim's "arm joint RMS speed" (v4: 0.80 at vx 1.0).

  Returns:
    Shape ``(B,)``.
  """
  asset = env.scene[asset_name]
  idx = [
    i for i, n in enumerate(asset.joint_names) if "shoulder" in n or "elbow" in n
  ]
  return torch.sqrt(torch.mean(torch.square(asset.data.joint_vel[:, idx]), dim=1))


##
# Decoupled task (v6): the upper body is not the policy's.
#
# The leg policy acts on 12 joints. The other 19 still need a position target
# every step: mjlab zeroes joint_pos_target for EVERY joint on reset
# (entity/data.py clear_state) and nothing rewrites a joint no action covers,
# so without this term the elbows (home 0.9) would be driven to 0 rad.
#
# A command term is the right owner: command_manager.compute() runs after the
# physics substeps and after auto-resets, and the leg action writes only its own
# indices, so the targets written here hold through the next step's substeps.
##


class UpperBodyCommand(CommandTerm):
  """Random, smooth upper-body motion standing in for the manipulation/IK
  controller the legs will run beside on hardware.

  Every resample draws a goal pose
      goal = home + amp * (u - home),   u ~ U(range_lo, range_hi) per joint
  where amp = level * r**amp_exponent is skewed toward small motions, and a
  `quiet_prob` fraction of resamples keeps goal = home. The target then slews
  toward the goal at a per-env speed, and is written as the joint position
  target of the upper-body actuators. `level` is raised by the
  upper_body_level curriculum as the legs cope.

  `.command` is the current 19-wide target, for the critic only: the deployed
  actor never depends on IK internals, it sees the measured joint state.
  """

  cfg: "UpperBodyCommandCfg"

  def __init__(self, cfg: "UpperBodyCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.asset_name]
    ids, names = self.robot.find_joints(cfg.joint_names)
    self.joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self.joint_names = names
    _, matched, ranges = resolve_matching_names_values(cfg.sample_range, names)
    assert list(matched) == list(names), (
      f"sample_range must cover every upper-body joint; missing "
      f"{sorted(set(names) - set(matched))}"
    )
    r = torch.tensor(ranges, device=self.device, dtype=torch.float32)  # (J, 2)
    self.lo, self.hi = r[:, 0], r[:, 1]
    self.home = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
    outside = (self.home[0] < self.lo - 1e-6) | (self.home[0] > self.hi + 1e-6)
    assert not outside.any(), (
      "home pose outside sample_range for "
      f"{[n for n, o in zip(names, outside.tolist()) if o]}"
    )

    n = self.num_envs
    self.target = self.home.clone()
    self.goal = self.home.clone()
    self.speed = torch.ones(n, 1, device=self.device)
    self.amp = torch.zeros(n, device=self.device)
    self.level = float(cfg.initial_level)
    # Curriculum bookkeeping (see upper_body_level).
    self.window_episodes = 0
    self.window_falls = 0
    self.metrics["upper_body_level"] = torch.zeros(n, device=self.device)
    self.metrics["upper_body_amp"] = torch.zeros(n, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target

  def reset(self, env_ids):
    # Start every episode from home; super().reset() then resamples a goal and
    # the target slews there at the drawn speed.
    self.target[env_ids] = self.home[env_ids]
    self.goal[env_ids] = self.home[env_ids]
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    k = len(env_ids)
    u = self.lo + (self.hi - self.lo) * torch.rand(k, len(self.lo), device=self.device)
    amp = self.level * torch.rand(k, device=self.device) ** self.cfg.amp_exponent
    amp = torch.where(
      torch.rand(k, device=self.device) < self.cfg.quiet_prob,
      torch.zeros_like(amp), amp,
    )
    self.amp[env_ids] = amp
    home = self.home[env_ids]
    self.goal[env_ids] = home + amp.unsqueeze(1) * (u - home)
    lo, hi = self.cfg.speed_range
    self.speed[env_ids, 0] = lo + (hi - lo) * torch.rand(k, device=self.device)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:  # per-step path: advance; the reset path only writes
      step = self.speed * self._env.step_dt
      delta = torch.clamp(self.goal - self.target, -step, step)
      self.target += delta
    self.robot.set_joint_position_target(self.target, joint_ids=self.joint_ids)

  def _update_metrics(self) -> None:
    self.metrics["upper_body_level"][:] = self.level
    self.metrics["upper_body_amp"][:] = self.amp


@dataclass(kw_only=True)
class UpperBodyCommandCfg(CommandTermCfg):
  asset_name: str = "robot"
  joint_names: tuple[str, ...] = ()
  sample_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  speed_range: tuple[float, float] = (0.3, 2.0)  # rad/s, per env per resample
  quiet_prob: float = 0.15
  amp_exponent: float = 1.5  # amp = level * U(0,1)**this; mean level/2.5
  initial_level: float = 0.1

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyCommand:
    return UpperBodyCommand(self, env)


def upper_body_level(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  window: int = 1024,
  step: float = 0.02,
  promote_below: float = 0.10,
  demote_above: float = 0.30,
) -> dict[str, torch.Tensor]:
  """Performance-gated amplitude curriculum for UpperBodyCommand.

  Counts finished episodes and how many of them FELL (terminated, not timed
  out), then every `window` episodes moves the global level by `step`: up if
  the fall fraction is under `promote_below`, down if over `demote_above`.
  Reset batches are a handful of envs at a time, far too noisy to act on one
  by one. Runs before the scene reset, so episode_length_buf is still the
  finished episode's; length 0 is the initial env.reset(), not an episode.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, UpperBodyCommand)
  finished = env.episode_length_buf[env_ids] > 0
  fell = env.termination_manager.terminated[env_ids] & finished
  term.window_episodes += int(finished.sum())
  term.window_falls += int(fell.sum())
  if term.window_episodes >= window:
    frac = term.window_falls / term.window_episodes
    if frac < promote_below:
      term.level = min(1.0, term.level + step)
    elif frac > demote_above:
      term.level = max(0.0, term.level - step)
    term.window_episodes = 0
    term.window_falls = 0
  return {"level": torch.tensor(term.level)}


def stand_still_feet(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Summed horizontal foot-site speed, only while commanded to stand.

  The decoupled requirement in one term: when the operator asks for zero
  twist, upper-body motion must not move the feet -- no shuffling, no
  stepping. Also logs Metrics/stand_foot_speed (mean over standing envs), the
  headline number for "the lower body does not move".

  Returns:
    Shape ``(B,)``.
  """
  asset = env.scene[asset_cfg.name]
  speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  standing = 1.0 - _command_active(env, command_name, command_threshold)
  cost = speed.sum(dim=1) * standing
  n = torch.clamp(standing.sum(), min=1.0)
  env.extras["log"]["Metrics/stand_foot_speed"] = torch.sum(cost) / n
  return cost


def stand_feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """1.0 while commanded to stand with BOTH feet on the ground.

  Returns:
    Shape ``(B,)``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.current_air_time
  assert air is not None, f"{sensor_name} needs track_air_time=True"
  both_down = (air <= 0).all(dim=1).float()
  return both_down * (1.0 - _command_active(env, command_name, command_threshold))


##
# Smoothness (v7, Mjlab-Velocity-Flat-HV1V2-Decoupled-Smooth).
#
# v6, measured in sim2sim at vx 0.6 with the arms moving: mean |da| 0.122
# (v4 0.094), |d2a| 0.102 (v4 0.056), hip-yaw |d2a| 3.2x v4, hip/knee target
# jumps up to 19 deg, hip/knee torque rate ~2.4x v4, and a 3-8 Hz share of
# 0.26 while STANDING under arm motion (v4 0.04). Upstream already has
# action_rate_l2 / action_acc_l2 / joint_acc_l2 / joint_vel_l2; these fill the
# rest.
##


def _action_columns(env: ManagerBasedRlEnv, joint_names, action_name: str) -> torch.Tensor:
  """Indices of the action columns that drive `joint_names` (regexes)."""
  am = env.action_manager
  term = am.get_term(action_name)
  # Offset of this term inside the concatenated action vector (v8 adds a
  # zero-width UpperBodyActuation term after joint_pos).
  offset = 0
  for name in am.active_terms:
    if name == action_name:
      break
    offset += am.get_term(name).action_dim
  idx = [offset + i for i, n in enumerate(term.target_names)
         if any(re.match(p, n) for p in joint_names)]
  assert idx, f"{joint_names} match no action joint of {term.target_names}"
  return torch.tensor(idx, device=env.device, dtype=torch.long)


class action_rate_l2_joints:
  """action_rate_l2 restricted to the action columns of `joint_names`.

  Lets one joint be priced harder than the rest -- here hip yaw, the closest
  to buzzing -- without touching the 12-wide action_rate_l2.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.idx = _action_columns(
      env, cfg.params["joint_names"], cfg.params.get("action_name", "joint_pos")
    )

  def __call__(
    self, env: ManagerBasedRlEnv, joint_names, action_name: str = "joint_pos"
  ) -> torch.Tensor:
    am = env.action_manager
    d = am.action[:, self.idx] - am.prev_action[:, self.idx]
    return torch.sum(torch.square(d), dim=1)


class torque_rate_l2:
  """Squared rate of change of actuator torque, N.m/s, over `asset_cfg` joints.

  Measured once per policy step. Actuator i drives joint i (sort_actuators=True,
  asserted in _continuous_limits), so joint ids index actuator_force directly.

  An env's first sample after a reset is not a rate, and neither is the first
  sample after a gap: the reward manager SKIPS a term whose weight is 0, which
  is exactly what happens before reward_weight_ramp switches it on. Both
  cases return 0 instead of a spike. Also logs Metrics/leg_torque_rate.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.prev: torch.Tensor | None = None
    self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self.last_counter = -2
    self.step_dt = env.step_dt

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      self.fresh[:] = True
    else:
      self.fresh[env_ids] = True

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    tau = env.scene[asset_cfg.name].data.actuator_force[:, asset_cfg.joint_ids]
    if self.prev is None or env.common_step_counter != self.last_counter + 1:
      self.prev = tau.clone()
      self.fresh[:] = True
    rate = (tau - self.prev) / self.step_dt
    valid = (~self.fresh).float()
    cost = torch.sum(torch.square(rate), dim=1) * valid
    n = torch.clamp(valid.sum(), min=1.0)
    env.extras["log"]["Metrics/leg_torque_rate"] = (
      torch.sum(rate.abs().mean(dim=1) * valid) / n
    )
    self.prev = tau.clone()
    self.fresh[:] = False
    self.last_counter = env.common_step_counter
    return cost


class reward_weight_ramp:
  """Linearly move a reward's weight from `w0` to `w1` between `start_iter` and
  `end_iter` training iterations, counted from this term's FIRST call.

  Upstream reward_curriculum is stepwise and keyed on the absolute
  common_step_counter, which a resume restores -- an absolute schedule would
  then already be over. Counting from the first call keeps the ramp meaningful
  for a fresh run and a resumed one alike.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.term_cfg = env.reward_manager.get_term_cfg(cfg.params["reward_name"])
    self.start: int | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    w0: float,
    w1: float,
    start_iter: int,
    end_iter: int,
    steps_per_iter: int = 24,
    absolute: bool = False,
    ref_num_envs: int | None = None,
    clock: str | None = None,
  ) -> dict[str, torch.Tensor]:
    """absolute=True counts iterations from common_step_counter itself, which
    mjlab restores on resume -- so a resumed run keeps its ramped weights
    (v8). absolute=False counts from this term's first call (v7, which was
    resumed across tasks from v6). ref_num_envs (v11) keys the schedule to
    SAMPLES: iterations are counted as if the run had ref_num_envs envs.
    clock (v11d): read the schedule position from env.<clock> (already in
    sample-keyed iterations, e.g. mdp_v11.v11_gated_clock's _v11_gclock)."""
    del env_ids, reward_name
    if clock is not None:
      it = float(getattr(env, clock, 0.0))
      a = min(max((it - start_iter) / max(end_iter - start_iter, 1), 0.0), 1.0)
      self.term_cfg.weight = w0 + a * (w1 - w0)
      return {"weight": torch.tensor(self.term_cfg.weight)}
    if self.start is None:
      self.start = 0 if absolute else env.common_step_counter
    it = (env.common_step_counter - self.start) / steps_per_iter
    if ref_num_envs:
      it *= env.num_envs / ref_num_envs
    a = min(max((it - start_iter) / max(end_iter - start_iter, 1), 0.0), 1.0)
    self.term_cfg.weight = w0 + a * (w1 - w0)
    return {"weight": torch.tensor(self.term_cfg.weight)}


def leg_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Metric: mean |a_t - a_{t-1}| over the action (the 12 legs here)."""
  am = env.action_manager
  return (am.action - am.prev_action).abs().mean(dim=1)


def leg_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Metric: mean |a_t - 2 a_{t-1} + a_{t-2}|."""
  am = env.action_manager
  return (am.action - 2 * am.prev_action + am.prev_prev_action).abs().mean(dim=1)


_HIP_YAW_COLS: dict[int, torch.Tensor] = {}


def hip_yaw_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Metric: mean |da| over the hip-yaw action columns."""
  idx = _HIP_YAW_COLS.get(id(env))
  if idx is None:
    idx = _HIP_YAW_COLS[id(env)] = _action_columns(
      env, (r".*_hip_yaw_joint$",), "joint_pos"
    )
  am = env.action_manager
  return (am.action[:, idx] - am.prev_action[:, idx]).abs().mean(dim=1)


##
# v8: deployment timing. Real path (merai_control): the policy's 50 Hz target
# is linearly interpolated over the 20 ms period to 1 kHz, then joint PD. v7's
# sim held each 50 Hz target for the 4 substeps (zero-order hold). Here both
# legs and upper body are interpolated at the 200 Hz physics resolution (5 ms),
# and the legs additionally see a per-env 0-20 ms delay (sensing, bus, compute)
# before each new target starts its ramp.
##


def _interp_alpha(substep: int, decimation: int) -> float:
  """Blend factor for substep `substep` (1..decimation): the new 50 Hz target
  is reached exactly at the last substep."""
  return min(max(substep / decimation, 0.0), 1.0)


class InterpolatedJointPositionAction(JointPositionAction):
  """JointPositionAction whose target ramps linearly from the previous to the
  new 50 Hz value across the physics substeps, as merai_control does, with a
  per-env delay of `d` substeps (0..decimation) before the new ramp starts.

  With targets p[k-2], p[k-1], p[k] (p[k] = this step's action), substep
  s = 1..D of the current period applies
      s <= d : p[k-2] + (D - d + s)/D * (p[k-1] - p[k-2])   (previous ramp's tail)
      s >  d : p[k-1] + (s - d)/D     * (p[k]   - p[k-1])   (new ramp)
  d = 0 is plain interpolation; d = D is a full one-step delay. Still a
  JointPositionAction named "joint_pos": the 12-D interface, action scale /
  offset and the ONNX exporter are untouched.
  """

  cfg: "InterpolatedJointPositionActionCfg"

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._D = int(env.cfg.decimation)
    lo, hi = cfg.delay_substeps_range
    assert 0 <= lo <= hi <= self._D, "delay must be 0..decimation substeps"
    n, a = self.num_envs, self.action_dim
    self._p2 = torch.zeros(n, a, device=self.device)   # p[k-2]
    self._p1 = torch.zeros(n, a, device=self.device)   # p[k-1]
    self._fresh = torch.ones(n, dtype=torch.bool, device=self.device)
    self._substep = 0
    self.delay = torch.zeros(n, 1, device=self.device)

  def process_actions(self, actions: torch.Tensor):
    first = not hasattr(self, "_p0")
    if not first:
      self._p2, self._p1 = self._p1, self._p0
    super().process_actions(actions)
    self._p0 = self._processed_actions.clone()
    fresh = self._fresh.unsqueeze(1) | first
    self._p1 = torch.where(fresh, self._p0, self._p1)
    self._p2 = torch.where(fresh, self._p0, self._p2)
    self._fresh[:] = False
    self._substep = 0

  def apply_actions(self) -> None:
    self._substep += 1
    s, D, d = float(self._substep), float(self._D), self.delay
    tail = self._p2 + ((D - d + s) / D) * (self._p1 - self._p2)
    new = self._p1 + ((s - d) / D).clamp(0.0, 1.0) * (self._p0 - self._p1)
    target = torch.where(d >= s, tail, new)
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._fresh[ids] = True
    lo, hi = self.cfg.delay_substeps_range
    k = self.num_envs if env_ids is None else len(env_ids)
    self.delay[ids] = torch.randint(lo, hi + 1, (k, 1), device=self.device).float()


@dataclass(kw_only=True)
class InterpolatedJointPositionActionCfg(JointPositionActionCfg):
  # per-env delay before each new target starts its ramp, in physics substeps
  # (5 ms each), drawn uniformly at every reset; (0, 0) = no delay.
  delay_substeps_range: tuple[int, int] = (0, 0)

  def build(self, env: ManagerBasedRlEnv) -> InterpolatedJointPositionAction:
    return InterpolatedJointPositionAction(self, env)


def ankle_stiffness(env: ManagerBasedRlEnv, env_ids, pitch_range, roll_range):
  """Startup DR: scale ankle pitch / roll PD gains (kp and kd) per env.

  The real ankles are two-motor four-bars with per-motor PD; mapped to the
  joint, that is ~1.9x (pitch) / 1.4x (roll) the sim joint stiffness. Runs
  AFTER v7's pd_gains event and multiplies the CURRENT gains, so v7's +-10%
  gain DR is kept. ctrl index == joint index (sort_actuators=True).
  """
  robot = env.scene["robot"]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  e = env_ids.to(env.device, dtype=torch.long)
  m = env.sim.model
  for pattern, (lo, hi) in ((r".*_ankle_pitch_joint$", pitch_range),
                            (r".*_ankle_roll_joint$", roll_range)):
    ids, _ = robot.find_joints((pattern,))
    c = torch.tensor(ids, device=env.device, dtype=torch.long)
    s = torch.empty(len(e), 1, device=env.device).uniform_(lo, hi)
    m.actuator_gainprm[e[:, None], c, 0] *= s
    m.actuator_biasprm[e[:, None], c, 1] *= s
    m.actuator_biasprm[e[:, None], c, 2] *= s


class UpperBodyActuation(ActionTerm):
  """Zero-width action term: writes the upper-body command's targets every
  physics substep, interpolated from the previous to the current 50 Hz value.

  The policy has no say here (action_dim 0). It exists only because action
  terms are the one hook mjlab calls per substep. Reads the command term named
  cfg.command_name, which must expose .target and .joint_ids.
  """

  cfg: "UpperBodyActuationCfg"

  def __init__(self, cfg: "UpperBodyActuationCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._decimation = int(env.cfg.decimation)
    self._raw = torch.zeros(self.num_envs, 0, device=self.device)
    self._prev: torch.Tensor | None = None
    self._cur: torch.Tensor | None = None
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self._substep = 0

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def _cmd(self):
    return self._env.command_manager.get_term(self.cfg.command_name)

  def process_actions(self, actions: torch.Tensor):
    cmd = self._cmd()
    new = cmd.target.clone()
    if self._cur is None:
      self._prev, self._cur = new.clone(), new
    else:
      self._prev = torch.where(self._fresh.unsqueeze(1), new, self._cur)
      self._cur = new
    self._fresh[:] = False
    self._substep = 0

  def apply_actions(self) -> None:
    if self._cur is None:
      return
    self._substep += 1
    cmd = self._cmd()
    a = _interp_alpha(self._substep, self._decimation)
    target = self._prev + a * (self._cur - self._prev)
    robot = self._env.scene[cmd.cfg.asset_name]
    bias = robot.data.encoder_bias[:, cmd.joint_ids]
    robot.set_joint_position_target(target - bias, joint_ids=cmd.joint_ids)

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      self._fresh[:] = True
    else:
      self._fresh[env_ids] = True


@dataclass(kw_only=True)
class UpperBodyActuationCfg(ActionTermCfg):
  command_name: str = "upper_body"

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyActuation:
    return UpperBodyActuation(self, env)


class StickyStandingVelocityCommand(UniformVelocityCommand):
  """UniformVelocityCommand whose standing draws HOLD zero for longer.

  The base resamples every 3-8 s and redraws standing each time, so a long
  zero-command stretch only happens by chance. Here a draw that comes up
  standing keeps its zero command for U(stand_hold_range) seconds -- long
  enough to expose slow rocking and pelvis drift.
  """

  cfg: "StickyStandingVelocityCommandCfg"

  def _resample(self, env_ids: torch.Tensor) -> None:
    super()._resample(env_ids)
    if len(env_ids) == 0:
      return
    st = env_ids[self.is_standing_env[env_ids]]
    if len(st) > 0:
      lo, hi = self.cfg.stand_hold_range
      self.time_left[st] = lo + (hi - lo) * torch.rand(len(st), device=self.device)


@dataclass(kw_only=True)
class StickyStandingVelocityCommandCfg(UniformVelocityCommandCfg):
  stand_hold_range: tuple[float, float] = (10.0, 20.0)

  def build(self, env: ManagerBasedRlEnv) -> StickyStandingVelocityCommand:
    return StickyStandingVelocityCommand(self, env)


# --- v8 upper body: v7's random motion mixed with synthetic Ruckig clips -----


class UpperBodyMixedCommand(UpperBodyCommand):
  """v7's UpperBodyCommand, where each resample is, with prob. clip_prob, a
  clip from the offline Mink IK -> online Ruckig library (data/clips_v2)
  instead of a random goal.

  * random half: exactly v7 (goal = home + amp*(u-home), constant slew speed,
    quiet_prob, all scaled by the curriculum level).
  * clip half: the target first slews home at the env's v7 speed (clips start
    at home, so the clip never starts with a jump), then plays the clip
    start to finish at amplitude `level` about home -- scaling the size down
    keeps every Ruckig v/a/j limit. Standing clips normally; while the twist
    command is non-zero, a walking clip with prob. walking_prob. Resamples
    during a clip are ignored; when it ends the target holds its last value
    until the next resample.
  v7's upper_body_level curriculum drives `level` for both halves.

  write_targets=False: the targets are written per substep, interpolated, by
  UpperBodyActuation instead of once per step here.
  """

  cfg: "UpperBodyMixedCommandCfg"

  def __init__(self, cfg: "UpperBodyMixedCommandCfg", env: ManagerBasedRlEnv):
    import numpy as np

    n = env.num_envs
    dev = env.device
    # clip state must exist before super().__init__ (it may resample)
    self.clip = torch.full((n,), -1, dtype=torch.long, device=dev)
    self.pending = torch.zeros(n, dtype=torch.bool, device=dev)
    self.hold_left = torch.zeros(n, device=dev)    # v8b: hold after a clip
    self.still_time = torch.zeros(n, device=dev)   # s the target has not moved
    self._prev_target = None
    self.phase = torch.zeros(n, device=dev)
    self.rate = torch.ones(n, device=dev)
    super().__init__(cfg, env)

    qs, starts, lens, peak, walking = [], [], [], [], []
    clip_joints, off = None, 0
    for family in cfg.families:
      d = np.load(f"{cfg.clip_dir}/{family}.npz")
      jn = [str(x) for x in d["joint_names"]]
      clip_joints = clip_joints or jn
      assert jn == clip_joints, f"{family}: joint order differs"
      assert abs(float(d["dt"]) - env.step_dt) < 1e-6, f"{family}: dt != step_dt"
      qs.append(d["q"])
      starts.append(d["start"] + off)
      lens.append(d["length"])
      peak.append(d["peak_speed"])
      walking.append(d["walking"])
      off += len(d["q"])
    assert clip_joints is not None, "no clip families"
    self.frames = torch.tensor(np.concatenate(qs), device=dev, dtype=torch.float32)
    self.clip_start = torch.tensor(np.concatenate(starts), device=dev)
    self.clip_len = torch.tensor(np.concatenate(lens), device=dev)
    self.clip_peak = torch.tensor(np.concatenate(peak), device=dev, dtype=torch.float32)
    is_walk = torch.tensor(np.concatenate(walking), device=dev)
    self.stand_pool = (~is_walk).nonzero().flatten()
    self.walk_pool = is_walk.nonzero().flatten()
    self.clip_cols = torch.tensor([self.joint_names.index(j) for j in clip_joints],
                                  device=dev, dtype=torch.long)
    self.metrics["upper_clip_frac"] = torch.zeros(n, device=dev)

  def reset(self, env_ids):
    self.clip[env_ids] = -1
    self.pending[env_ids] = False
    self.hold_left[env_ids] = 0.0
    self.still_time[env_ids] = 0.0
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    busy = (self.clip[env_ids] >= 0) | self.pending[env_ids] | (self.hold_left[env_ids] > 0)
    env_ids = env_ids[~busy]
    if len(env_ids) == 0:
      return
    use_clip = torch.rand(len(env_ids), device=self.device) < self.cfg.clip_prob
    rnd, clp = env_ids[~use_clip], env_ids[use_clip]
    if len(rnd) > 0:
      super()._resample_command(rnd)
    if len(clp) > 0 and self.cfg.clip_quiet_prob > 0:
      # v8b: a quiet clip draw = go home at the env's speed and stay there
      quiet = torch.rand(len(clp), device=self.device) < self.cfg.clip_quiet_prob
      q_ids, clp = clp[quiet], clp[~quiet]
      if len(q_ids) > 0:
        self.goal[q_ids] = self.home[q_ids]
        self.amp[q_ids] = 0.0
        lo, hi = self.cfg.speed_range
        self.speed[q_ids, 0] = lo + (hi - lo) * torch.rand(len(q_ids), device=self.device)
    if len(clp) > 0:
      k = len(clp)
      twist = self._env.command_manager.get_command(self.cfg.twist_name)
      moving = (torch.norm(twist[clp, :2], dim=1) + twist[clp, 2].abs()) > 0.1
      walk = moving & (torch.rand(k, device=self.device) < self.cfg.walking_prob)
      if len(self.walk_pool) == 0:
        walk[:] = False
      pick = torch.empty(k, dtype=torch.long, device=self.device)
      for flag, pool in ((False, self.stand_pool), (True, self.walk_pool)):
        m = walk == flag
        if m.any():
          pick[m] = pool[torch.randint(len(pool), (int(m.sum()),), device=self.device)]
      self.clip[clp] = pick
      self.pending[clp] = True
      self.phase[clp] = 0.0
      self.goal[clp] = self.home[clp]
      self.amp[clp] = self.level
      lo, hi = self.cfg.speed_range
      self.speed[clp, 0] = lo + (hi - lo) * torch.rand(k, device=self.device)
      peak = torch.clamp(self.clip_peak[pick] * max(self.level, 1e-3), min=1e-3)
      self.rate[clp] = torch.clamp(self.cfg.vmax / peak, max=1.0)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is None:
      # random envs and envs returning home before a clip: v7's slew
      playing = (self.clip >= 0) & ~self.pending
      step = self.speed * self._env.step_dt
      delta = torch.clamp(self.goal - self.target, -step, step)
      self.target = torch.where(playing.unsqueeze(1), self.target, self.target + delta)
      arrived = self.pending & ((self.target - self.home).abs().amax(dim=1) < 1e-4)
      self.pending &= ~arrived
      active = ((self.clip >= 0) & ~self.pending).nonzero().flatten()
      if len(active) > 0:
        c = self.clip[active]
        ph = self.phase[active]
        last = self.clip_len[c] - 1
        i0 = torch.minimum(ph.long(), last)
        i1 = torch.minimum(i0 + 1, last)
        w = (ph - i0.float()).unsqueeze(1)
        q = (1 - w) * self.frames[self.clip_start[c] + i0] + w * self.frames[self.clip_start[c] + i1]
        home = self.home[active]
        sub = self.target[active]
        sub[:, self.clip_cols] = home[:, self.clip_cols] + self.level * (
          q - home[:, self.clip_cols])
        self.target[active] = sub
        self.phase[active] += self.rate[active]
        done = active[self.phase[active] >= last.float()]
        if len(done) > 0:
          self.clip[done] = -1
          self.goal[done] = self.target[done]   # hold until the next resample
          lo, hi = self.cfg.hold_after_clip_range
          self.hold_left[done] = lo + (hi - lo) * torch.rand(len(done), device=self.device)
      self.hold_left = torch.clamp(self.hold_left - self._env.step_dt, min=0.0)
      if self._prev_target is not None:
        moved = (self.target - self._prev_target).abs().amax(dim=1) > 1e-6
        self.still_time = torch.where(moved, torch.zeros_like(self.still_time),
                                      self.still_time + self._env.step_dt)
      self._prev_target = self.target.clone()
    if self.cfg.write_targets:
      self.robot.set_joint_position_target(self.target, joint_ids=self.joint_ids)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["upper_clip_frac"][:] = ((self.clip >= 0) & ~self.pending).float()


@dataclass(kw_only=True)
class UpperBodyMixedCommandCfg(UpperBodyCommandCfg):
  clip_dir: str = ""
  families: tuple[str, ...] = ("scripted", "random", "walking")
  clip_prob: float = 0.5       # P(clip | resample); else v7's random goal
  walking_prob: float = 0.5    # P(walking clip | clip, twist command non-zero)
  vmax: float = 2.0            # playback-rate cap (= Ruckig arm v limit)
  twist_name: str = "twist"
  write_targets: bool = True   # False when UpperBodyActuation writes them
  # v8b: after a clip ends, hold its last pose for U(range) s (resamples are
  # ignored meanwhile); (0, 0) = v8 behaviour.
  hold_after_clip_range: tuple[float, float] = (0.0, 0.0)
  # v8b: P(a clip draw becomes "go home and hold"), like v7's quiet_prob
  clip_quiet_prob: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyMixedCommand:
    return UpperBodyMixedCommand(self, env)


# --- activity gate, anti-rocking rewards ---------------------------------------


class _UpperActivity:
  """||qd_upper|| + c * ||qdd_upper||, from the MEASURED upper-body state;
  qdd by finite difference over one policy step. Computed once per step."""

  def __init__(self, env: ManagerBasedRlEnv, joint_names, c: float):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(joint_names)
    self.ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self.c = c
    self.prev_qd = None
    self.counter = -1
    self.value = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if env.common_step_counter != self.counter:
      qd = env.scene["robot"].data.joint_vel[:, self.ids]
      if self.prev_qd is None:
        self.prev_qd = qd.clone()
      qdd = (qd - self.prev_qd) / env.step_dt
      fresh = env.episode_length_buf <= 1
      qdd[fresh] = 0.0
      self.value = torch.norm(qd, dim=1) + self.c * torch.norm(qdd, dim=1)
      self.prev_qd = qd.clone()
      self.counter = env.common_step_counter
    return self.value


_ACTIVITY: dict[int, _UpperActivity] = {}


def upper_activity_gate(env: ManagerBasedRlEnv, joint_names, c: float, ref: float):
  """Bounded gate in [1, 2]: 1 with a still upper body, 2 at/above `ref`."""
  a = _ACTIVITY.get(id(env))
  if a is None:
    a = _ACTIVITY[id(env)] = _UpperActivity(env, joint_names, c)
  return 1.0 + torch.clamp(a(env) / ref, 0.0, 1.0)


def pelvis_quiet(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  lin_weight: float,
  upper_joint_names,
  activity_c: float,
  activity_ref: float,
) -> torch.Tensor:
  """standing x gate x (pelvis roll-rate^2 + pitch-rate^2 + lin_weight * xy-speed^2).

  No tilt-ANGLE term (user: no large direct tilt penalty). Rates and drift are
  what rocking and creeping look like; the gate makes the price higher while
  the upper body is actually moving. Also logs Metrics/pelvis_rate_standing.
  """
  robot = env.scene["robot"]
  w = robot.data.root_link_ang_vel_b
  v = robot.data.root_link_lin_vel_b
  standing = 1.0 - _command_active(env, command_name, command_threshold)
  raw = w[:, 0] ** 2 + w[:, 1] ** 2 + lin_weight * (v[:, 0] ** 2 + v[:, 1] ** 2)
  gate = upper_activity_gate(env, upper_joint_names, activity_c, activity_ref)
  n = torch.clamp(standing.sum(), min=1.0)
  env.extras["log"]["Metrics/pelvis_rate_standing"] = (
    torch.sum(torch.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2) * standing) / n
  )
  return raw * standing * gate


def leg_motion_standing(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  deadband: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """standing x sum(max(0, |qd_leg| - deadband)^2): leg motion above the
  standing noise floor while commanded to stand."""
  qd = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  standing = 1.0 - _command_active(env, command_name, command_threshold)
  return torch.sum(torch.clamp(qd.abs() - deadband, min=0.0) ** 2, dim=1) * standing


class standing_steps:
  """Metric: UNNECESSARY steps -- real steps taken while settled in a stand.

  A touchdown counts only if all of these hold:
    * the command has been zero for > settle_s (excludes the stop transient);
    * the foot was airborne >= min_air_s (excludes contact flicker: a pad
      breaking contact for one 10 ms tick -- v7 with the arms at home made ~160 of
      those per 128 envs x 10 s with ZERO real steps);
    * the foot moved >= min_disp_m between lift-off and touchdown;
    * the last push was > push_grace_s ago (push_robot recoveries are legal
      recovery steps, not unnecessary ones; pushes are stamped by
      push_and_stamp). 3 s: on v8b every push-related counted step landed
      2.0-2.4 s after the push, so a 2 s grace let them through.
  reduce="sum" gives unnecessary steps per episode.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    n = env.num_envs
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    robot = env.scene["robot"]
    # Foot sites in the contact sensor's primary order (left/right by name).
    prim = list(getattr(sensor, "primary_names", None) or
                ["left_ankle_roll_link", "right_ankle_roll_link"])
    sites = ["left_foot" if p.startswith("left") else "right_foot" for p in prim]
    self.site_ids = robot.find_sites(sites, preserve_order=True)[0]
    self.lift_xy = torch.zeros(n, len(sites), 2, device=env.device)
    self.stand_time = torch.zeros(n, device=env.device)
    if not hasattr(env, "_hv1v2_push_step"):
      env._hv1v2_push_step = torch.full((n,), -10**9, dtype=torch.long, device=env.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self.stand_time[env_ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv, sensor_name: str, command_name: str,
               settle_s: float = 1.0, min_air_s: float = 0.10,
               min_disp_m: float = 0.02, push_grace_s: float = 3.0) -> torch.Tensor:
    standing = (1.0 - _command_active(env, command_name, 0.1)).bool()
    self.stand_time = torch.where(standing, self.stand_time + env.step_dt,
                                  torch.zeros_like(self.stand_time))
    sensor: ContactSensor = env.scene[sensor_name]
    xy = env.scene["robot"].data.site_pos_w[:, self.site_ids, :2]
    lift = sensor.compute_first_air(dt=env.step_dt)
    self.lift_xy = torch.where(lift.unsqueeze(-1), xy, self.lift_xy)
    td = sensor.compute_first_contact(dt=env.step_dt)
    real = td & (sensor.data.last_air_time >= min_air_s) & (
      torch.norm(xy - self.lift_xy, dim=-1) >= min_disp_m)
    since_push = (env.common_step_counter - env._hv1v2_push_step) * env.step_dt
    settled = (self.stand_time > settle_s) & (since_push > push_grace_s)
    step = (real.any(dim=1) & settled).float()
    return step


def push_and_stamp(env: ManagerBasedRlEnv, env_ids, velocity_range, asset_cfg=None):
  """push_robot, plus a per-env timestamp so standing_steps can tell a push
  recovery (legal) from an unnecessary step."""
  from mjlab.envs import mdp as envs_mdp

  kw = {} if asset_cfg is None else {"asset_cfg": asset_cfg}
  envs_mdp.push_by_setting_velocity(env, env_ids, velocity_range, **kw)
  if not hasattr(env, "_hv1v2_push_step"):
    env._hv1v2_push_step = torch.full((env.num_envs,), -10**9, dtype=torch.long,
                                      device=env.device)
  ids = slice(None) if env_ids is None else env_ids
  env._hv1v2_push_step[ids] = env.common_step_counter


# --- v8b: arms-still sway, narrow yaw tracking -------------------------------


def _still_standing(env: ManagerBasedRlEnv, command_name: str, upper_name: str,
                    min_still_s: float) -> torch.Tensor:
  standing = 1.0 - _command_active(env, command_name, 0.1)
  ub = env.command_manager.get_term(upper_name)
  return standing * (ub.still_time > min_still_s).float()


def still_sway(env: ManagerBasedRlEnv, command_name: str = "twist",
               upper_name: str = "upper_body", min_still_s: float = 1.0) -> torch.Tensor:
  """Metric: pelvis roll/pitch rate (rad/s) of envs standing with the upper
  body still for > min_still_s -- v8's self-sway. Logs the mean over those
  envs as Metrics/still_sway (and their share as Metrics/still_stand_frac);
  returns 0 so the episode metric is not diluted."""
  m = _still_standing(env, command_name, upper_name, min_still_s)
  w = env.scene["robot"].data.root_link_ang_vel_b
  rate = torch.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2)
  n = m.sum()
  log = env.extras.setdefault("log", {})
  log["Metrics/still_sway"] = torch.sum(rate * m) / torch.clamp(n, min=1.0)
  log["Metrics/still_stand_frac"] = n / env.num_envs
  return torch.zeros(env.num_envs, device=env.device)


def still_stand_quiet(env: ManagerBasedRlEnv, command_name: str = "twist",
                      upper_name: str = "upper_body", min_still_s: float = 1.0) -> torch.Tensor:
  """Penalty (use a negative weight): pelvis roll^2 + pitch^2 rate, ONLY while
  standing with the upper body still for > min_still_s. Never active during
  arm motion, so it cannot soften the response to the arms (what pelvis_quiet
  did to the deleted v8). v8b contingency C1, weight 0 unless enabled."""
  m = _still_standing(env, command_name, upper_name, min_still_s)
  w = env.scene["robot"].data.root_link_ang_vel_b
  return (w[:, 0] ** 2 + w[:, 1] ** 2) * m


class track_yaw_bias:
  """exp(-e^2 / std^2), e = yaw-rate tracking error low-passed over ~tau s.

  Heading drift is a small *bias* (v8: +35 deg in 20 s = 0.03 rad/s), while
  the instantaneous yaw rate swings +-0.1-0.2 rad/s every stride. A narrow
  kernel on the instantaneous rate would mostly price that gait wobble; the
  low-passed error is what integrates into heading drift. Yaw only: mjlab's
  track_angular_velocity folds roll/pitch rates into its kernel, which at a
  narrow std would also price the pelvis' response to arm motion.
  The filter restarts on reset. Command changes pass through the filter
  like any other error (tau ~1 s), so transients cost a little briefly.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.e = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      self.e[:] = 0.0
    else:
      self.e[env_ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv, std: float, tau: float = 1.0,
               command_name: str = "twist") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    wz = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    a = min(env.step_dt / tau, 1.0)
    self.e = (1 - a) * self.e + a * (cmd[:, 2] - wz)
    log = env.extras.setdefault("log", {})
    log["Metrics/yaw_bias_abs"] = self.e.abs().mean()
    return torch.exp(-torch.square(self.e) / std**2)


# --- v10 stance terms ----------------------------------------------------------
# v7 stands with the right foot on its outer edge (2 of 4 pads, 25 N.m of
# ankle-roll preload, targets 3.6-5.4 deg past where the floor lets the foot
# go) and leaves the feet 6-10 deg apart in yaw after a push: nothing priced a
# flat foot, a relaxed stance or a tidy-up step (logs/eval/v7_stance). All three
# terms act only while commanded to stand. Foot order does not matter to any of
# them (each is symmetric in the two feet), so the contact sensor's slot order
# and the asset's body order need not agree.


def _foot_quat_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
  data = env.scene[asset_cfg.name].data
  return data.body_link_quat_w[:, asset_cfg.body_ids], data.body_link_pos_w[:, asset_cfg.body_ids]


def _yaw(q: torch.Tensor) -> torch.Tensor:
  """Yaw of wxyz quaternions."""
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _both_feet_down(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.current_air_time
  assert air is not None, f"{sensor_name} needs track_air_time=True"
  return (air <= 0).all(dim=1).float()


def stand_sole_tilt(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Summed sole tilt (rad, |roll, pitch| of each ankle_roll_link against the
  floor) while commanded to stand with both feet down. Linear, not squared: at
  v7's 1-3 deg a squared term is too small to matter. The pads are axis-aligned
  in ankle_roll_link, so its frame IS the sole's. Logs Metrics/stand_sole_tilt
  in degrees, mean over the envs it applies to."""
  q, _ = _foot_quat_pos(env, asset_cfg)
  down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(q.shape[:-1] + (3,))
  g = quat_apply_inverse(q, down)
  tilt = torch.atan2(torch.norm(g[..., :2], dim=-1), -g[..., 2]).sum(dim=1)
  m = (1.0 - _command_active(env, command_name, command_threshold)) * _both_feet_down(env, sensor_name)
  n = torch.clamp(m.sum(), min=1.0)
  env.extras.setdefault("log", {})["Metrics/stand_sole_tilt_deg"] = torch.rad2deg(torch.sum(tilt * m) / n / 2)
  return tilt * m


def stand_leg_action_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  """Sum of squared raw leg actions while commanded to stand. On HOME_KEYFRAME
  zero action is a flat, level stance, so this prices standing AWAY from it --
  in particular v7's habit of holding targets the floor will not let the joints
  reach (a static preload). Logs Metrics/stand_leg_action_abs."""
  a = env.action_manager.get_term(action_name).raw_action
  m = 1.0 - _command_active(env, command_name, command_threshold)
  n = torch.clamp(m.sum(), min=1.0)
  env.extras.setdefault("log", {})["Metrics/stand_leg_action_abs"] = torch.sum(a.abs().mean(dim=1) * m) / n
  return torch.sum(torch.square(a), dim=1) * m


def stand_stance_shape(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg,
  nominal_width: float,
  yaw_weight: float,
  width_weight: float,
) -> torch.Tensor:
  """yaw_weight * |foot yaw difference| + width_weight * |stance width -
  nominal| (rad, m) while commanded to stand. Width is the feet's lateral
  separation in the pelvis heading frame. Together with stand_still_feet this
  makes one corrective step cheaper than holding a pivoted stance. Logs
  Metrics/stand_foot_yaw_diff_deg and Metrics/stand_width_err."""
  q, p = _foot_quat_pos(env, asset_cfg)
  yaw = _yaw(q)
  dyaw = torch.atan2(torch.sin(yaw[:, 0] - yaw[:, 1]), torch.cos(yaw[:, 0] - yaw[:, 1])).abs()
  py = _yaw(env.scene[asset_cfg.name].data.root_link_quat_w)
  rel = p[:, 0, :2] - p[:, 1, :2]
  width = (-torch.sin(py) * rel[:, 0] + torch.cos(py) * rel[:, 1]).abs()
  werr = (width - nominal_width).abs()
  m = 1.0 - _command_active(env, command_name, command_threshold)
  n = torch.clamp(m.sum(), min=1.0)
  log = env.extras.setdefault("log", {})
  log["Metrics/stand_foot_yaw_diff_deg"] = torch.rad2deg(torch.sum(dyaw * m) / n)
  log["Metrics/stand_width_err"] = torch.sum(werr * m) / n
  return (yaw_weight * dyaw + width_weight * werr) * m
