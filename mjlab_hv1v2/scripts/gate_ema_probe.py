#!/usr/bin/env python3
"""Measure the walk-gate EMAs (all/back/lat/slow, swing) a checkpoint
produces under the normal random command stream (play env, arms level as
given), to calibrate gate thresholds (2026-11-02, v11e).
  python scripts/gate_ema_probe.py --run v11d_s43 --iters 2400 6000"""
import argparse, glob, os
from dataclasses import asdict
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "rsl_rl", "hv1v2_velocity")
TASK = "Mjlab-Velocity-Flat-HV1V2-Decoupled-v11"

def probe(ckpt, ub_level, steps, num_envs):
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.commands["upper_body"].initial_level = ub_level
  agent = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")
  obs = venv.get_observations()
  acc, n = {}, 0
  with torch.inference_mode():
    for k in range(steps):
      obs, *_ = venv.step(pol(obs))
      if k >= steps // 3:
        for key in ("all", "back", "lat", "slow", "fast"):
          v = getattr(env, f"_v11_track_{key}", None)
          if v is not None:
            acc[key] = acc.get(key, 0.0) + v
        acc["swing"] = acc.get("swing", 0.0) + (getattr(env, "_v11_swing", 0.0) or 0.0)
        n += 1
  env.close()
  return {k: round(v / n, 3) for k, v in acc.items()}

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--iters", type=int, nargs="+")
ap.add_argument("--ub-level", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--num-envs", type=int, default=2048)
a = ap.parse_args()
d = sorted(glob.glob(os.path.join(ROOT, f"*_{a.run}")))[-1]
for it in a.iters:
  print(f"RESULT {a.run} model_{it} ub {a.ub_level}:", probe(os.path.join(d, f"model_{it}.pt"), a.ub_level, a.steps, a.num_envs), flush=True)
