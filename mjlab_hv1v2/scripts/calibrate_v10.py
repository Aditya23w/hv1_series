#!/usr/bin/env python3
"""Calibrate v10's stance weights on rollouts of a checkpoint in the v10 env.

Each new term is priced at --frac of the per-step reward that standing envs
earn from everything ELSE (the new terms' own contribution is subtracted, so
the placeholder weights do not bias the answer):

  weight = -frac * R_stand_base / (mean raw term value over standing envs * dt)

Run it on the policy v10 warm-starts from (v7), and paste the printed weights
into V10_WEIGHTS in env_cfgs.py.

  python mjlab_hv1v2/scripts/calibrate_v10.py \
      --ckpt logs/rsl_rl/hv1v2_velocity/2026-10-06_17-58-29_v7/model_29999.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_hv1v2 import mdp as hv1v2_mdp

TASK = "Mjlab-Velocity-Flat-HV1V2-Decoupled-v10"
TERMS = ("stand_sole_tilt", "stand_leg_action_l2", "stand_stance_shape")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--num-envs", type=int, default=2048)
  ap.add_argument("--seconds", type=float, default=30.0)
  ap.add_argument("--frac", type=float, default=0.02)
  args = ap.parse_args()

  configure_torch_backends()
  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")

  dt = env.step_dt
  cfgs = {n: env.reward_manager.get_term_cfg(n) for n in TERMS}
  raw = {n: 0.0 for n in TERMS}
  base, n_stand = 0.0, 0
  obs = venv.get_observations()
  with torch.inference_mode():
    for _ in range(int(args.seconds / dt)):
      obs, rew, dones, _ = venv.step(policy(obs))
      m = (1.0 - hv1v2_mdp._command_active(env, "twist", 0.1)) * (~dones.bool()).float()
      own = torch.zeros_like(rew)
      for n, c in cfgs.items():
        v = c.func(env, **c.params)
        raw[n] += float((v * m).sum())
        own += v * c.weight * dt
      base += float(((rew - own) * m).sum())
      n_stand += int(m.sum())

  r_base = base / n_stand
  print(f"standing env-steps {n_stand}, base reward per standing step {r_base:.4f} (dt {dt})")
  for n in TERMS:
    mean = raw[n] / n_stand
    w = -args.frac * r_base / (mean * dt) if mean > 0 else float("nan")
    print(f"  {n:22s} mean raw {mean:.5f}  -> weight {w:.3f}  (current {cfgs[n].weight})")
  env.close()


if __name__ == "__main__":
  main()
