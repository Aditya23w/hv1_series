"""HV1V1-specific MDP terms.

Out-of-tree, so nothing here can be imported from mjlab's own task configs.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.string import resolve_expr
from mjlab_hv1v1.hv1v1_constants import CONTINUOUS_TORQUE

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


# ---------------------------------------------------------------------------
# Push curriculum.
#
# mjlab has NO built-in event curriculum: every shipped curriculum touches a
# command (commands_vel), the terrain (terrain_levels_vel), a reward term or a
# termination term. This is the first one that drives an EVENT, so the mechanism
# it relies on is spelled out and was re-verified against the local checkout:
#
#  1. EventManager.__init__ does `self.cfg = deepcopy(cfg)`
#     (managers/event_manager.py:155), so mutating `env.cfg.events[...]` would
#     silently do NOTHING. The manager owns its own copies.
#  2. get_term_cfg() returns `self._mode_term_cfgs[mode][index]`
#     (event_manager.py:209) -- the live object that apply() reads, not a copy.
#     There is no set_term_cfg anywhere in mjlab; in-place mutation is the
#     sanctioned route.
#  3. The interval branch re-reads `term_cfg.params` and
#     `term_cfg.interval_range_s` on EVERY fire (event_manager.py:269-292), so a
#     write here takes effect at the next push with no further plumbing.
#  4. `_interval_term_time_left` is a list positionally parallel to
#     `_mode_term_cfgs["interval"]`, so event terms must NEVER be added or
#     removed at runtime. This function only mutates fields.
#
# The curriculum manager splats **params with no signature validation
# (managers/manager_base.py), so the keys of CurriculumTermCfg.params must match
# this signature exactly or the first reset raises TypeError. compute() runs once
# per reset batch, not per step.
#
# Shape follows tasks/velocity/mdp/curriculums.py:commands_vel exactly: read
# env.common_step_counter, apply every stage whose step has passed (later stages
# win), return 0-d tensors so the ramp is logged as Curriculum/push_vel/*.
#
# WHY A RAMP AT ALL. HV1V2's v4 trained with the full +-0.5 m/s push from
# iteration 0 and no ramp. Two measured consequences: early on the pushes land on
# a policy that cannot stand, where they are noise rather than curriculum; and
# at convergence nothing in training ever asked for more than +-0.5, so v4
# survives 2x in the tested directions and falls at 3x forward. This starts at
# half strength, reaches v4's level by iteration 3000, then goes past it.
#
# This raises DISTURBANCE EXPOSURE. It does not add ankle torque and it does not
# change the static CoM: what strategy the policy learns from the exposure --
# capture stepping, hip motion, arm use -- is not something a curriculum can
# guarantee.

_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def push_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  event_name: str,
  stages: list[dict],
) -> dict[str, torch.Tensor]:
  """Ramp an interval push event's velocity_range by training step count.

  Args:
    env: the environment.
    env_ids: resetting envs; unused, the range is global.
    event_name: the interval event term to drive, e.g. ``"push_robot"``.
    stages: ``[{"step": int, "velocity_range": {axis: (lo, hi)}}, ...]``.
      Applied in order, every stage whose ``step`` has passed, so later stages
      overwrite earlier ones and the list must be sorted by ``step``.

  Returns:
    The six axis magnitudes actually in force, for logging.
  """
  del env_ids  # the push range is global, not per-env
  term_cfg = env.event_manager.get_term_cfg(event_name)
  assert term_cfg.mode == "interval", (
    f"push_stages drives an interval event; {event_name!r} is mode {term_cfg.mode!r}"
  )
  steps = [s["step"] for s in stages]
  assert steps == sorted(steps), f"push_stages: stages must be sorted by step, got {steps}"

  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      # Replace the dict wholesale rather than mutating it in place: the same
      # dict object may also be referenced by env.cfg, and rebinding keeps this
      # function's effect confined to the manager's live copy.
      term_cfg.params["velocity_range"] = dict(stage["velocity_range"])

  rng = term_cfg.params["velocity_range"]
  out: dict[str, torch.Tensor] = {}
  for axis in _SE3_KEYS:
    lo, hi = rng.get(axis, (0.0, 0.0))
    out[f"{axis}_max"] = torch.tensor(float(hi))
    out[f"{axis}_min"] = torch.tensor(float(lo))
  return out
