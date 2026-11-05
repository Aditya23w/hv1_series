#!/usr/bin/env python3
"""Push-recovery sweep: fall rate vs push size, per pelvis height (v11h) or at
home (v11 / v11g).

Each env settles for --settle s (v11h: the height command slews there from
home at 0.10 m/s), then gets ONE horizontal velocity kick of size m (m/s, the
same kind of push as training's push_robot) in a random direction, and is
watched for --watch s. A fall = the env's fell_over termination fires during
the watch window. Also reports the pre-push height and the peak pelvis tilt
of the envs that survived.

  # v11g at home
  python scripts/push_sweep.py --ckpt .../v11g_s43/model_19000.pt
  # v11h at four heights
  python scripts/push_sweep.py --ckpt .../v11h_s43/model_26999.pt \\
      --task Mjlab-Velocity-Flat-HV1V2-Decoupled-v11h --heights 0.97939 0.95 0.92 0.89
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-HV1V2-Decoupled-v11")
  ap.add_argument("--heights", type=float, nargs="*", default=[])
  ap.add_argument("--mags", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
  ap.add_argument("--per", type=int, default=48, help="envs per (height, push size)")
  ap.add_argument("--mode", choices=("stand", "walk"), default="stand")
  ap.add_argument("--vx", type=float, default=0.5, help="walk mode forward speed")
  ap.add_argument("--ub-level", type=float, default=0.0, help="arm motion level (0 = arms home)")
  ap.add_argument("--payload", type=float, default=0.0, help="kg in each hand")
  ap.add_argument("--settle", type=float, default=4.0)
  ap.add_argument("--watch", type=float, default=5.0)
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args()
  import mjlab_hv1v2  # noqa: F401
  from mjlab_hv1v2 import mdp_v11h as H
  from mjlab_hv1v2.hv1v2_constants import HAND_BODY_NAMES

  heights = a.heights or [None]
  G, M = len(heights), len(a.mags)
  n = G * M * a.per
  cfg = load_env_cfg(a.task, play=True)
  cfg.scene.num_envs = n
  cfg.episode_length_s = 120.0
  tw = cfg.commands["twist"]
  tw.resampling_time_range = (1e9, 1e9)
  tw.reversal_prob = tw.stop_prob = 0.0
  tw.rel_heading_envs = 0.0
  vx = 0.0 if a.mode == "stand" else a.vx
  tw.ranges.lin_vel_x, tw.ranges.lin_vel_y, tw.ranges.ang_vel_z = (vx, vx), (0.0, 0.0), (0.0, 0.0)
  tw.rel_standing_envs = 1.0 if a.mode == "stand" else 0.0
  ub = cfg.commands["upper_body"]
  ub.initial_level = a.ub_level
  if a.ub_level > 0:
    ub.quiet_prob = 0.0
  pay = cfg.events["hand_payload"].params
  pay["body_names"], pay["mass_range"] = HAND_BODY_NAMES, (a.payload, a.payload)
  hl = cfg.commands["hand_load"]
  hl.active_prob = hl.pickup_prob = 0.0
  cfg.events["imu_bias"].params.update(rp_max_rad=0.0, gyro_max=0.0)
  cfg.events.pop("push_robot", None)
  if heights[0] is not None:
    cfg.commands["height"].resampling_time_range = (1e9, 1e9)

  agent = load_rl_cfg(a.task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(a.task)(venv, asdict(agent), device="cuda:0")
  runner.load(a.ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")
  robot = env.scene["robot"]
  dev = env.device

  # env layout: [height][mag][per]
  gi = torch.arange(n, device=dev) // (M * a.per)
  mi = (torch.arange(n, device=dev) // a.per) % M
  mag = torch.tensor(a.mags, device=dev)[mi]
  hc = None
  if heights[0] is not None:
    hc = env.command_manager.get_term("height")
    h_t = torch.tensor(heights, device=dev)[gi]
  g = torch.Generator(device=dev).manual_seed(a.seed)
  ang = torch.rand(n, generator=g, device=dev) * 2 * math.pi
  kick = torch.zeros(n, 6, device=dev)
  kick[:, 0], kick[:, 1] = mag * torch.cos(ang), mag * torch.sin(ang)

  dt = env.step_dt
  k_push = int(a.settle / dt)
  k_end = k_push + int(a.watch / dt)
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  fell_before = torch.zeros(n, dtype=torch.bool, device=dev)
  h_pre, tilt_peak = [], torch.zeros(n, device=dev)
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(k_end):
      if hc is not None:
        hc.target[:] = h_t
      if k == k_push:
        v = robot.data.root_link_vel_w.clone() + kick
        robot.write_root_link_velocity_to_sim(v)
      obs, _, _, _ = venv.step(pol(obs))
      term = env.termination_manager.terminated
      if k < k_push:
        fell_before |= term
        if k >= k_push - int(1.0 / dt):
          h_pre.append(H.base_height(env).clone())
      else:
        fell |= term
        gr = robot.data.projected_gravity_b
        t = torch.rad2deg(torch.asin(torch.clamp(torch.norm(gr[:, :2], dim=1), max=1.0)))
        tilt_peak = torch.where(fell, tilt_peak, torch.maximum(tilt_peak, t))
  env.close()

  hp = torch.stack(h_pre).mean(0).cpu().numpy()
  fell, fb, tp = fell.cpu().numpy(), fell_before.cpu().numpy(), tilt_peak.cpu().numpy()
  gi, mi = gi.cpu().numpy(), mi.cpu().numpy()
  print(f"\npush sweep  {a.ckpt.split('/')[-2]}/{a.ckpt.split('/')[-1]}  mode {a.mode}"
        f"{f' vx {a.vx}' if a.mode == 'walk' else ''}  arms {a.ub_level}  payload {a.payload} kg"
        f"  ({a.per} envs per cell, random direction)")
  head = "  height cmd / meas   | " + " ".join(f"{m:>5.1f}" for m in a.mags) + "  m/s kick -> fall %"
  print(head)
  for i, h in enumerate(heights):
    sel = gi == i
    row = []
    for j in range(M):
      c = sel & (mi == j)
      row.append(100.0 * fell[c].mean())
    lab = "home(v11)" if h is None else f"{h:.3f}"
    print(f"  {lab:>9s} / {hp[sel].mean():.4f} | " + " ".join(f"{r:5.0f}" for r in row)
          + f"   pre-push falls {int(fb[sel].sum())}")
  print("  peak tilt of survivors (deg, mean):")
  for i, h in enumerate(heights):
    sel = gi == i
    row = []
    for j in range(M):
      c = sel & (mi == j) & ~fell
      row.append(tp[c].mean() if c.any() else float("nan"))
    lab = "home(v11)" if h is None else f"{h:.3f}"
    print(f"  {lab:>9s}            | " + " ".join(f"{r:5.1f}" for r in row))


if __name__ == "__main__":
  main()
