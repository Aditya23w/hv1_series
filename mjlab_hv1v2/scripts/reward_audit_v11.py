#!/usr/bin/env python3
"""Per-term reward audit (2026-11-01, after v11c_s43 collapsed into marching in
place). Rolls checkpoints in the PLAY env (= 100%-ramp weights) at a fixed
command and prints each term's mean weighted per-step reward (raw*weight, no
dt) over steps 100..N, plus achieved velocity, so walking and marching in
place can be compared under the final reward budget.

  python scripts/reward_audit_v11.py --run v11c_s43 --iters 800 5000 --cmd 0.5 0 0
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import asdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "rsl_rl", "hv1v2_velocity")


def audit(task, ckpt, cmd, ub_level=None, no_load=False, num_envs=256, steps=500):
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 30.0
  tw = cfg.commands["twist"]
  tw.resampling_time_range = (1e9, 1e9)
  tw.reversal_prob = tw.stop_prob = 0.0
  tw.rel_standing_envs = 0.0
  tw.rel_heading_envs = 0.0
  if ub_level is not None:
    cfg.commands["upper_body"].initial_level = ub_level
  if no_load:
    if "hand_payload" in cfg.events:
      cfg.events["hand_payload"].params["mass_range"] = (0.0, 0.0)
    if "hand_load" in cfg.commands:
      cfg.commands["hand_load"].active_prob = cfg.commands["hand_load"].pickup_prob = 0.0
  agent = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")
  term = env.command_manager.get_term("twist")
  rm = env.reward_manager
  robot = env.scene["robot"]
  c = torch.tensor(cmd, device=env.device)
  acc = torch.zeros(len(rm._term_names), device=env.device)
  vel = torch.zeros(3, device=env.device)
  n, falls = 0, 0
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(steps):
      term.vel_command_b[:] = c
      if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
      obs, _, dones, _ = venv.step(pol(obs))
      if k >= 100:
        acc += rm._step_reward.mean(0)
        v = robot.data.root_link_lin_vel_b
        vel += torch.stack([v[:, 0].mean(), v[:, 1].mean(), robot.data.root_link_ang_vel_b[:, 2].mean()])
        falls += int(dones.sum())
        n += 1
  env.close()
  return dict(zip(rm._term_names, (acc / n).tolist())), (vel / n).tolist(), falls


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True)
  ap.add_argument("--iters", type=int, nargs="+", required=True)
  ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
  ap.add_argument("--ub-level", type=float, default=None)
  ap.add_argument("--no-load", action="store_true")
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-HV1V2-Decoupled-v11")
  a = ap.parse_args()
  run_dir = sorted(glob.glob(os.path.join(ROOT, f"*_{a.run}")))[-1]
  res = {it: audit(a.task, os.path.join(run_dir, f"model_{it}.pt"), a.cmd, a.ub_level, a.no_load) for it in a.iters}
  names = list(next(iter(res.values()))[0])
  print(f"\n{os.path.basename(run_dir)}  task {a.task}  cmd {a.cmd}  (weighted reward per step, play = 100% weights)  ub_level {a.ub_level}  no_load {a.no_load}")
  print(f"{'term':28s}" + "".join(f"{'it ' + str(it):>12s}" for it in a.iters))
  for nm in sorted(names, key=lambda x: -abs(res[a.iters[0]][0][x] - res[a.iters[-1]][0][x])):
    print(f"{nm:28s}" + "".join(f"{res[it][0][nm]:12.4f}" for it in a.iters))
  print(f"{'TOTAL':28s}" + "".join(f"{sum(res[it][0].values()):12.4f}" for it in a.iters))
  for it in a.iters:
    v = res[it][1]
    print(f"  it {it}: vx {v[0]:+.3f} vy {v[1]:+.3f} wz {v[2]:+.3f}  resets {res[it][2]}")


if __name__ == "__main__":
  main()
