"""v11h MDP terms (Mjlab-Velocity-Flat-HV1V2-Decoupled-v11h), 2026-11-04.

v11g plus an operator pelvis-height command, 0.89-0.979 m (crouch only).
Plan: ~/.claude/plans/now-i-have-a-robust-lecun.md.

  step 1  HeightPoseTable     height -> reference leg pose + static torque
  step 2  BaseHeightCommand   per-env height target, slew-limited like merai
          base_height         measured pelvis height above the terrain
          v11h_height_range   curriculum: the low end widens 0.95 -> 0.89
  step 3  height_cmd_obs      actor + critic: applied command - home (1)
          base_height_obs     critic only: measured height - home (1)
  step 4  track_base_height   reward: exp(-(h - h_cmd)^2 / sigma^2)
          height_posture      v11 pose reward, reference = table pose at h_cmd
          stand_leg_action_ref_l2   v10/v11g standing-action price, about a_ref(h)
          stand_leg_torque_h  v11g squeeze price, deadband raised by the crouch

Every height-referenced term reduces EXACTLY to its v11g form at the home
height (the table's top row is HOME_KEYFRAME; offsets are relative to home).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import variable_posture

from mjlab_hv1v2.hv1v2_constants import V11H_HEIGHT_RANGE, V11H_HEIGHT_TABLE, V11H_HOME_Z
from mjlab_hv1v2.mdp import _command_active
from mjlab_hv1v2.mdp_v11 import _ema, iters_equiv


class HeightPoseTable:
  """Pelvis height -> (reference leg pose, static leg torque), on the GPU.

  Wraps V11H_HEIGHT_TABLE (ascending pelvis_z, top row = HOME_KEYFRAME).
  `joint_names` picks and orders the columns, e.g. the action term's
  target_names, so callers never depend on the table's own order.

    tab = HeightPoseTable(term.target_names, device)
    q_ref, tau = tab(h)          # h: (N,) metres -> (N, J), (N, J)

  h is clamped to [z_min, z_max] first, so a 0.980 command lands exactly on
  the pinned HOME row (0.97939), then interpolated linearly between the two
  bracketing rows (torch.searchsorted; the rows are not evenly spaced in z).
  """

  def __init__(self, joint_names: Sequence[str], device: str | torch.device,
               table: dict = V11H_HEIGHT_TABLE):
    names = list(table["joint_names"])
    missing = [n for n in joint_names if n not in names]
    assert not missing, f"not in the height table: {missing}"
    cols = [names.index(n) for n in joint_names]
    z = torch.tensor(table["pelvis_z"], dtype=torch.float32, device=device)
    assert bool(torch.all(z[1:] > z[:-1])), "pelvis_z must be strictly ascending"
    self.joint_names = tuple(joint_names)
    self.z = z
    self.q = torch.tensor(table["q"], dtype=torch.float32, device=device)[:, cols]
    self.tau = torch.tensor(table["tau_static"], dtype=torch.float32, device=device)[:, cols]
    self.z_min = float(z[0])
    self.z_max = float(z[-1])
    self.q_home = self.q[-1]
    self.tau_home = self.tau[-1]

  def weights(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(lower row index, upper row index, upper weight) for each height."""
    h = torch.clamp(h, self.z_min, self.z_max)
    hi = torch.searchsorted(self.z, h.contiguous(), right=False).clamp(1, len(self.z) - 1)
    lo = hi - 1
    w = ((h - self.z[lo]) / (self.z[hi] - self.z[lo])).clamp(0.0, 1.0)
    return lo, hi, w

  def __call__(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lo, hi, w = self.weights(h)
    w = w.unsqueeze(-1)
    q = self.q[lo] * (1 - w) + self.q[hi] * w
    tau = self.tau[lo] * (1 - w) + self.tau[hi] * w
    return q, tau


# --- step 2: the height command ------------------------------------------------------


def base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(N,) measured pelvis height above the terrain (m), from the simulator.
  Flat task: terrain at the env origin's z. Same definition as the table
  (pelvis z with the lowest sole face at 0), up to ~1-2 mm contact
  penetration."""
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


class BaseHeightCommand(CommandTerm):
  """Per-env pelvis-height target, slew-limited exactly as merai_policy will be.

    target   drawn at each resample (cfg.resampling_time_range): with prob
             home_prob the home height, else U(low, home). `low` is set by
             the v11h_height_range curriculum (cfg.initial_low until then).
    command  (N, 1) the APPLIED height: moves toward target by at most
             slew_rate * step_dt per step. This is what the policy observes
             and what the rewards track.

  Reset: applied height = home (the robot resets to HOME_KEYFRAME), held for
  reset_hold_s before slewing toward the freshly drawn target.
  """

  cfg: "BaseHeightCommandCfg"

  def __init__(self, cfg: "BaseHeightCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n = self.num_envs
    self.home = float(cfg.home)
    self.high = float(cfg.high)
    resume = os.environ.get("V11H_RESUME_LOW")
    self.low = float(resume) if resume else float(cfg.initial_low)
    self.target = torch.full((n,), self.home, device=self.device)
    self.height = torch.full((n,), self.home, device=self.device)
    self.hold_left = torch.zeros(n, device=self.device)
    self.metrics["height_err"] = torch.zeros(n, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.height.unsqueeze(1)

  def reset(self, env_ids):
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    self.height[env_ids] = self.home
    self.hold_left[env_ids] = self.cfg.reset_hold_s
    return super().reset(env_ids)              # -> _resample_command (new target)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    k = len(env_ids)
    u = torch.rand(k, device=self.device)
    lo = min(self.low, self.high)
    draw = lo + (self.high - lo) * torch.rand(k, device=self.device)
    self.target[env_ids] = torch.where(u < self.cfg.home_prob,
                                       torch.full_like(draw, self.home), draw)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is not None:                    # reset path: state already set
      return
    dt = self._env.step_dt
    held = self.hold_left > 0
    self.hold_left = torch.clamp(self.hold_left - dt, min=0.0)
    step = torch.clamp(self.target - self.height, -self.cfg.slew_rate * dt, self.cfg.slew_rate * dt)
    self.height = torch.where(held, self.height, self.height + step)

  def _update_metrics(self) -> None:
    err = (base_height(self._env) - self.height).abs()
    # per-env episode mean (logged at reset as Metrics/height/height_err)
    steps = self._env.episode_length_buf.clamp(min=1).float()
    m = self.metrics["height_err"]
    self.metrics["height_err"] = m + (err - m) / steps
    # global EMA for the range curriculum; skip each episode's first second
    # (the reset settle) so drops/early falls don't dominate it
    live = self._env.episode_length_buf.float() * self._env.step_dt > 1.0
    if bool(live.any()):
      e = float(err[live].mean())
      _ema(self._env, "_v11h_height_err", e, self.cfg.ema_alpha)


@dataclass(kw_only=True)
class BaseHeightCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (4.0, 10.0)
  home: float = V11H_HOME_Z
  high: float = V11H_HEIGHT_RANGE[1]
  initial_low: float = 0.95
  home_prob: float = 0.30
  slew_rate: float = 0.10          # m/s, identical in merai_policy
  reset_hold_s: float = 0.5
  ema_alpha: float = 0.002

  def build(self, env: ManagerBasedRlEnv) -> BaseHeightCommand:
    return BaseHeightCommand(self, env)


class v11h_height_range:
  """Curriculum term: widen the height command's low end from its initial
  value to `final_low` by `step` m at most once per `every_iters` iterations
  (sample-keyed at ref_num_envs), and only while
    * v11's walking gate is open (env._v11_track_ok: omnidirectional
      tracking + swing time, set by the walk_track metric), and
    * the height-error EMA (BaseHeightCommand) is under err_ok.
  It never narrows: falling back would undo a range the policy already
  covers. The low end is NOT in checkpoints; on a resume set
  V11H_RESUME_LOW=<logged Curriculum/height_range/low>."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.last_widen: float | None = None

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str,
               final_low: float = V11H_HEIGHT_RANGE[0], step: float = 0.01,
               every_iters: float = 200.0, err_ok: float = 0.025,
               ref_num_envs: int | None = None) -> dict[str, torch.Tensor]:
    del env_ids
    term: BaseHeightCommand = env.command_manager.get_term(command_name)
    it = iters_equiv(env, ref_num_envs)
    prev = getattr(self, "_last_it", None)
    self._last_it = it
    # first call, or the runner restoring common_step_counter from a checkpoint
    # (a > 50-iteration jump, as in v11_gated_clock): restart the interval, so
    # the first widening comes one full interval after start/resume
    if self.last_widen is None or (prev is not None and it - prev > 50.0):
      self.last_widen = it
    e = getattr(env, "_v11h_height_err", None)
    ok = bool(getattr(env, "_v11_track_ok", False)) and e is not None and e < err_ok
    if ok and term.low > final_low + 1e-9 and it - self.last_widen >= every_iters:
      term.low = max(final_low, round(term.low - step, 4))
      self.last_widen = it
    return {"low": torch.tensor(term.low),
            "err_ema": torch.tensor(float("nan") if e is None else e),
            "open": torch.tensor(float(ok))}


# --- step 3: observations ----------------------------------------------------------------


def height_cmd_obs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(N, 1) the APPLIED (slewed) height command minus home: 0 at home,
  -0.089 at the bottom of the range. merai_policy builds the identical value
  from its own clamped, slewed operator height. No noise: like the twist
  command, it is the operator's input, known exactly."""
  return env.command_manager.get_command(command_name) - V11H_HOME_Z


def base_height_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(N, 1) measured pelvis height minus home. Critic (privileged) only:
  the robot has no height sensor; the actor infers height from its joints."""
  return (base_height(env) - V11H_HOME_Z).unsqueeze(1)


# --- step 4: rewards -------------------------------------------------------------------


def _h_cmd(env: ManagerBasedRlEnv, height_name: str) -> torch.Tensor:
  return env.command_manager.get_command(height_name)[:, 0]


def _table_for(env: ManagerBasedRlEnv, key: str, joint_names) -> HeightPoseTable:
  """One HeightPoseTable per (term, joint order), cached on the env."""
  cache = getattr(env, "_v11h_tables", None)
  if cache is None:
    cache = env._v11h_tables = {}
  k = (key, tuple(joint_names))
  if k not in cache:
    cache[k] = HeightPoseTable(joint_names, env.device)
  return cache[k]


def track_base_height(env: ManagerBasedRlEnv, height_name: str, twist_name: str,
                      std_standing: float = 0.020, std_walking: float = 0.035,
                      command_threshold: float = 0.1) -> torch.Tensor:
  """exp(-(h - h_cmd)^2 / sigma^2) on the measured pelvis height. sigma is
  wider while the twist command asks for stepping (the gait bobs the pelvis
  by ~1-2 cm). Logs Metrics/height_track_err_{stand,walk} (m)."""
  err = base_height(env) - _h_cmd(env, height_name)
  walk = _command_active(env, twist_name, command_threshold)
  std = std_standing + (std_walking - std_standing) * walk
  log = env.extras.setdefault("log", {})
  for key, m in (("stand", 1.0 - walk), ("walk", walk)):
    n = m.sum()
    if n > 0:
      log[f"Metrics/height_track_err_{key}"] = (err.abs() * m).sum() / n
  return torch.exp(-torch.square(err) / std**2)


class height_posture(variable_posture):
  """mjlab's variable_posture with the reference pose taken from the height
  table at the APPLIED height command instead of default_joint_pos. Legs only
  (v11's pose term is legs-only); at home it is the stock term exactly."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    asset = env.scene[cfg.params["asset_cfg"].name]
    # same lookup (and so the same column order) as variable_posture's std vectors
    ids, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self.ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self.tab = HeightPoseTable(names, env.device)

  def __call__(self, env: ManagerBasedRlEnv, std_standing, std_walking, std_running,
               asset_cfg: SceneEntityCfg, command_name: str, height_name: str = "height",
               walking_threshold: float = 0.5, running_threshold: float = 1.5) -> torch.Tensor:
    del std_standing, std_walking, std_running
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = (speed < walking_threshold).float().unsqueeze(1)
    walking = ((speed >= walking_threshold) & (speed < running_threshold)).float().unsqueeze(1)
    running = (speed >= running_threshold).float().unsqueeze(1)
    std = self.std_standing * standing + self.std_walking * walking + self.std_running * running
    q_ref, _ = self.tab(_h_cmd(env, height_name))
    err2 = torch.square(asset.data.joint_pos[:, self.ids] - q_ref)
    return torch.exp(-torch.mean(err2 / std**2, dim=1))


def stand_leg_action_ref_l2(env: ManagerBasedRlEnv, command_name: str, command_threshold: float,
                            height_name: str = "height", action_name: str = "joint_pos"
                            ) -> torch.Tensor:
  """v10's stand_leg_action_l2 (sum of squared raw leg actions while
  commanded to stand), measured about the raw action that holds the table
  pose at the commanded height:

    a_ref = (q_ref(h) - anchor + (tau(h) - tau(home)) / kp) / scale

  tau / kp is the PD sag that carries the static load; it is taken RELATIVE
  to home, so a_ref(home) = 0 and the term is v11g's exactly there (v11g's
  standing already includes the home sag). kp is the NOMINAL gain (no DR):
  the policy cannot observe the per-env draw. Logs
  Metrics/stand_leg_action_abs (about a_ref, comparable with v11g's)."""
  term = env.action_manager.get_term(action_name)
  names = term.target_names
  tab = _table_for(env, "action", names)
  if not hasattr(term, "_v11h_kp"):
    robot = env.scene["robot"]
    ctrl = robot.indexing.ctrl_ids[term.target_ids]     # ctrl index == joint index
    term._v11h_kp = env.sim.get_default_field("actuator_gainprm")[ctrl, 0].to(env.device)
  q_ref, tau = tab(_h_cmd(env, height_name))
  a_ref = (q_ref - term.offset + (tau - tab.tau_home) / term._v11h_kp) / term.scale
  d = term.raw_action - a_ref
  m = 1.0 - _command_active(env, command_name, command_threshold)
  n = torch.clamp(m.sum(), min=1.0)
  env.extras.setdefault("log", {})["Metrics/stand_leg_action_abs"] = torch.sum(d.abs().mean(dim=1) * m) / n
  return torch.sum(torch.square(d), dim=1) * m


def stand_leg_torque_h(env: ManagerBasedRlEnv, command_name: str, command_threshold: float,
                       deadband: dict[str, float], asset_cfg: SceneEntityCfg,
                       height_name: str = "height") -> torch.Tensor:
  """v11g's stand_leg_torque, (|tau| - deadband)^2 on hip roll / hip pitch /
  knee while standing, with each joint's deadband raised by the extra static
  torque the commanded crouch needs:

    deadband_j(h) = deadband_j + max(0, |tau_j(h)| - |tau_j(home)|)

  = v11g's 10 / 15 / 30 N.m at home; the knee's ~80 N.m at 0.89 m. Logs
  Metrics/stand_leg_tau_excess."""
  robot = env.scene[asset_cfg.name]
  key = ("_v11h_tau_ids", tuple(sorted(deadband)))
  cache = getattr(env, "_v11_cache", None)
  if cache is None:
    cache = env._v11_cache = {}
  if key not in cache:
    ids, db, names = [], [], []
    for pat, dv in deadband.items():
      jid, jn = robot.find_joints((pat,), preserve_order=True)
      ids += list(jid); names += list(jn); db += [dv] * len(jid)
    cache[key] = (torch.tensor(ids, device=env.device, dtype=torch.long),
                  torch.tensor(db, device=env.device), tuple(names))
  ids, db, names = cache[key]
  tab = _table_for(env, "torque", names)
  _, tau_h = tab(_h_cmd(env, height_name))
  db_h = db + torch.clamp(tau_h.abs() - tab.tau_home.abs(), min=0.0)
  tau = robot.data.actuator_force[:, ids]
  excess = torch.clamp(tau.abs() - db_h, min=0.0)
  m = 1.0 - _command_active(env, command_name, command_threshold)
  cost = torch.sum(torch.square(excess), dim=1) * m
  n = torch.clamp(m.sum(), min=1.0)
  env.extras.setdefault("log", {})["Metrics/stand_leg_tau_excess"] = torch.sum(
    torch.sqrt(torch.sum(torch.square(excess), dim=1)) * m) / n
  return cost
