"""Left/right mirror maps for v11 (rsl_rl symmetry extension), 2026-11-01.

The mirror is DERIVED from the MJCF joint axes, never typed by hand. Mirroring
the world about the sagittal (x-z) plane is S = diag(1, -1, 1); a hinge about
axis a by q becomes a hinge about S a by -q (S is a reflection). So with the
partner joint p (left <-> right, or itself on the midline):

    S a_j = +a_p  ->  q_p' = -q_j        (roll / yaw-type joints)
    S a_j = -a_p  ->  q_p' = +q_j        (pitch-type joints)

i.e. sign_j = -dot(S a_j, a_p), asserted to be +-1. That holds for the canted
hip-pitch axes too ([0, .866, +-.5]). tests/test_symmetry.py checks the result
by forward kinematics (mirrored config -> reflected body CoMs), which an
involution test alone cannot: negating twice is the identity for ANY sign.

Base terms: angular velocity (wx, wy, wz) -> (-wx, wy, -wz); gravity
(gx, gy, gz) -> (gx, -gy, gz); command (vx, vy, wz) -> (vx, -vy, -wz).

The model is mirror-symmetric to < 1 mm (CoM) on legs, pelvis, torso and
shoulders; the wrist links and head yaw link are off by <= 9 mm (frame
placement in the export, equal masses) -- a property of the model, not of the
map.

Used as mirror LOSS only (rsl_rl Symmetry use_mirror_loss): the augmentation
function must also return a critic observation, which is passed through
unmirrored -- the actor alone is evaluated on the mirrored batch.
"""

from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np
import torch

from mjlab_hv1v2.hv1v2_constants import HV1V2_XML

_S = np.diag([1.0, -1.0, 1.0])


def partner(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[5:]
  if name.startswith("right_"):
    return "left_" + name[6:]
  return name


@lru_cache(maxsize=1)
def _joint_axes() -> dict[str, np.ndarray]:
  m = mujoco.MjModel.from_xml_path(str(HV1V2_XML))
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  return {m.joint(j).name: d.xaxis[j].copy() for j in range(m.njnt)
          if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}


def joint_mirror(names: list[str] | tuple[str, ...]) -> tuple[list[int], list[float]]:
  """(perm, sign) with mirrored[i] = sign[i] * x[perm[i]]."""
  axes = _joint_axes()
  idx = {n: i for i, n in enumerate(names)}
  perm, sign = [], []
  for n in names:
    p = partner(n)
    assert p in idx, f"{n}: partner {p} not in the vector"
    dot = float(_S @ axes[p] @ axes[n])
    assert abs(abs(dot) - 1.0) < 1e-3, f"{n}/{p}: axes not mirror images (dot {dot:.4f})"
    perm.append(idx[p])
    sign.append(-round(dot))
  return perm, sign


_BASE = {
  "angvel": ([0, 1, 2], [-1.0, 1.0, -1.0]),
  "gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),
  "cmd": ([0, 1, 2], [1.0, -1.0, -1.0]),
}


class ActorMirror:
  """Mirror for a flat actor observation built term-major with per-term
  history (mjlab ObservationManager layout). `terms` = [(kind, width,
  history)], kind one of angvel / gravity / cmd or a tuple of joint names."""

  def __init__(self, terms, device):
    perm, sign, off = [], [], 0
    for kind, width, hist in terms:
      if isinstance(kind, str):
        p, s = _BASE[kind]
      else:
        p, s = joint_mirror(kind)
      assert len(p) == width, (kind, width, len(p))
      for h in range(max(hist, 1)):
        base = off + h * width
        perm += [base + i for i in p]
        sign += s
      off += width * max(hist, 1)
    self.width = off
    self.perm = torch.tensor(perm, device=device, dtype=torch.long)
    self.sign = torch.tensor(sign, device=device, dtype=torch.float32)

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == self.width, (x.shape, self.width)
    return x[..., self.perm] * self.sign


_TERM_KIND = {
  "base_ang_vel": "angvel",
  "projected_gravity": "gravity",
  "command": "cmd",
}

_CACHE: dict[int, tuple[ActorMirror, ActorMirror]] = {}


def build_v11_mirrors(env) -> tuple[ActorMirror, ActorMirror]:
  """(observation mirror, action mirror) for the env's actor group."""
  from mjlab_hv1v2.hv1v2_constants import V11_UB_OBS_JOINTS

  u = env.unwrapped if hasattr(env, "unwrapped") else env
  key = id(u)
  if key in _CACHE:
    return _CACHE[key]
  om = u.observation_manager
  robot = u.scene["robot"]
  joint_names = tuple(robot.joint_names)
  leg_names = tuple(u.action_manager.get_term("joint_pos").target_names)
  terms = []
  cfgs = om._group_obs_term_cfgs["actor"]
  for name, cfg, dim in zip(om.active_terms["actor"], cfgs, om.group_obs_term_dim["actor"]):
    hist = max(cfg.history_length, 1)
    width = int(np.prod(dim)) // hist
    if name in _TERM_KIND:
      kind = _TERM_KIND[name]
    elif name in ("joint_pos", "joint_vel"):
      kind = joint_names
    elif name == "actions":
      kind = leg_names
    elif name in ("ub_target", "ub_target_rate"):
      kind = V11_UB_OBS_JOINTS
    else:
      raise KeyError(f"no mirror rule for actor term {name!r}")
    terms.append((kind, width, cfg.history_length))
  obs_m = ActorMirror(terms, u.device)
  act_m = ActorMirror([(leg_names, len(leg_names), 0)], u.device)
  _CACHE[key] = (obs_m, act_m)
  return obs_m, act_m


def mirror_v11(env, obs=None, actions=None):
  """rsl_rl data_augmentation_func: returns ([x; M x] obs, [a; M a] actions).
  Critic observations are duplicated unmirrored (mirror loss only)."""
  obs_m, act_m = build_v11_mirrors(env)
  out_obs = out_act = None
  if obs is not None:
    mo = obs.clone()
    mo["actor"] = obs_m(obs["actor"])
    out_obs = torch.cat([obs, mo], dim=0)
  if actions is not None:
    out_act = torch.cat([actions, act_m(actions)], dim=0)
  return out_obs, out_act
