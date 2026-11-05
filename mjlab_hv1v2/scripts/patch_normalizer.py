#!/usr/bin/env python3
"""Patch a checkpoint's observation normalizers for joints that never moved.

Why: rsl-rl's EmpiricalNormalization keeps a running mean/var over ~1e9
samples. A joint that sat at home for the whole run ends up with std ~0.006
rad, and the moment it moves the network sees (x - mean)/(std + 0.01) ~ 50
sigma -- v8c fell within 1.4-2.0 s whenever head_yaw, head_pitch or
left_wrist_roll moved. Warm-starting v8d (which moves them) from v8c without
this patch repeats the stage-1 collapse; with it, those inputs arrive at their
real scale from step one.

What it does (never touches the source file):
  * builds the target task's env (4 envs) only to read the observation layout
    -- term order, per-term width, history length -- by NAME, nothing
    hard-coded;
  * for each patched joint, sets mean/var/std of
      actor : joint_pos and joint_vel, every history frame
      critic: joint_pos, joint_vel, and the joint's column of the upper_body
              command term (absolute target)
    from the clip library the env will actually replay, plus the actor's
    observation-noise variance (uniform noise: (max-min)^2/12);
  * leaves every other channel, the count, and all network weights untouched.

  python mjlab_hv1v2/scripts/patch_normalizer.py \
      --ckpt logs/rsl_rl/hv1v2_velocity/<v8c>/model_16600.pt \
      --task Mjlab-Velocity-Flat-HV1V2-Decoupled-v8 \
      --out  logs/rsl_rl/hv1v2_velocity/<v8c>/model_16600_v8dpatch.pt
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

PATCH_JOINTS = ("head_yaw_joint", "head_pitch_joint", "left_wrist_roll_joint")


def library_stats(clip_dir: str, joints, idle_frac: float = 0.15):
  """mean/var of q-home and qd per joint over every clip family, with the
  idle-at-home gaps the env inserts between clips (~idle_frac of the time)."""
  q, qd, home, names = [], [], None, None
  for f in sorted(glob.glob(f"{clip_dir}/*.npz")):
    d = np.load(f)
    names = [str(x) for x in d["joint_names"]]
    home = d["home"]
    q.append(d["q"])
    qd.append(d["qd"])
  q, qd = np.concatenate(q), np.concatenate(qd)
  out = {}
  for j in joints:
    k = names.index(j)
    dq = q[:, k] - home[k]
    v = qd[:, k]
    # mixture with idle (dq = 0, v = 0)
    m_q = (1 - idle_frac) * dq.mean()
    var_q = (1 - idle_frac) * (dq ** 2).mean() - m_q ** 2
    m_v = (1 - idle_frac) * v.mean()
    var_v = (1 - idle_frac) * (v ** 2).mean() - m_v ** 2
    m_abs = (1 - idle_frac) * q[:, k].mean() + idle_frac * home[k]
    out[j] = dict(home=float(home[k]), q_mean=m_q, q_var=var_q, v_mean=m_v, v_var=var_v,
                  abs_mean=m_abs, abs_var=var_q)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-HV1V2-Decoupled-v8")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  assert Path(args.out).resolve() != Path(args.ckpt).resolve(), "never overwrite the source"

  import mjlab_hv1v2  # noqa: F401  (registers tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device="cpu")
  om = env.observation_manager
  joint_names = list(env.scene["robot"].joint_names)
  nj = len(joint_names)
  ub = env.command_manager.get_term("upper_body")
  stats = library_stats(cfg.commands["upper_body"].clip_dir, PATCH_JOINTS)

  ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
  report = []
  for group, sd_key in (("actor", "actor_state_dict"), ("critic", "critic_state_dict")):
    sd = ck[sd_key]
    mean = sd["obs_normalizer._mean"].reshape(-1).clone()
    var = sd["obs_normalizer._var"].reshape(-1).clone()
    names = om.active_terms[group]
    dims = [int(np.prod(d)) for d in om.group_obs_term_dim[group]]
    assert sum(dims) == mean.numel(), f"{group}: layout {sum(dims)} != normalizer {mean.numel()}"
    off = 0
    for name, dim in zip(names, dims):
      tcfg = om.get_term_cfg(group, name)
      noise_var = 0.0
      n = getattr(tcfg, "noise", None)
      if group == "actor" and n is not None and hasattr(n, "n_min"):
        noise_var = (float(n.n_max) - float(n.n_min)) ** 2 / 12.0
      if name in ("joint_pos", "joint_vel"):
        frames = dim // nj
        assert frames * nj == dim, f"{group}/{name}: {dim} not a multiple of {nj}"
        for j in PATCH_JOINTS:
          s = stats[j]
          mu, vv = ((s["q_mean"], s["q_var"]) if name == "joint_pos" else (s["v_mean"], s["v_var"]))
          for f in range(frames):  # term-major, oldest -> newest
            idx = off + f * nj + joint_names.index(j)
            before = float(np.sqrt(var[idx]))
            mean[idx], var[idx] = float(mu), float(vv + noise_var)
            if f == frames - 1:
              report.append(f"{group:6s} {name:9s} {j:22s} std {before:.4f} -> {np.sqrt(vv + noise_var):.4f}")
      if name == "upper_body":
        assert dim == len(ub.joint_names)
        for j in PATCH_JOINTS:
          idx = off + ub.joint_names.index(j)
          before = float(np.sqrt(var[idx]))
          mean[idx], var[idx] = float(stats[j]["abs_mean"]), float(stats[j]["abs_var"])
          report.append(f"{group:6s} {name:9s} {j:22s} std {before:.4f} -> {np.sqrt(stats[j]['abs_var']):.4f}")
      off += dim
    shape = sd["obs_normalizer._mean"].shape
    sd["obs_normalizer._mean"] = mean.reshape(shape)
    sd["obs_normalizer._var"] = var.reshape(shape)
    sd["obs_normalizer._std"] = torch.sqrt(var).reshape(shape)
  env.close()
  torch.save(ck, args.out)
  print("\n".join(report))
  print(f"wrote {args.out} (iter {ck.get('iter')}); source untouched")


if __name__ == "__main__":
  main()
