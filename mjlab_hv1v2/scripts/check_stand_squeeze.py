#!/usr/bin/env python3
"""v11g gate (2026-11-02): the standing SQUEEZE seen on the robot with v11f.
Each env walks (random 0.3-0.5 m/s, random direction) for 3 s, then is
commanded to stand for 25 s; over the last 15 s it measures per-leg torque on
hip roll / hip pitch / knee. Floor friction fixed per run (1.0 and 2.0).

  PASS: mean |hip roll tau| <= 10 N.m per side, |hip pitch R - L| <= 15 N.m.
  (Robot, v11f model_12000: hip roll 55/59 N.m; scissor hip pitch +62/-44.)

  python scripts/check_stand_squeeze.py --ckpt logs/.../model_N.pt
"""
from __future__ import annotations
import argparse, math
from dataclasses import asdict
import numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
TASK = "Mjlab-Velocity-Flat-HV1V2-Decoupled-v11"
J = ("right_hip_roll_joint", "left_hip_roll_joint", "right_hip_pitch_joint", "left_hip_pitch_joint",
     "right_knee_joint", "left_knee_joint")

def run(ckpt, friction, n=128, walk_s=3.0, stand_s=25.0, ub_level=0.0, task=TASK, height=None):
  cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = n; cfg.episode_length_s = 60.0
  tw = cfg.commands["twist"]; tw.resampling_time_range = (1e9, 1e9); tw.reversal_prob = tw.stop_prob = 0.0
  tw.rel_standing_envs = 0.0; tw.rel_heading_envs = 0.0
  cfg.events["foot_friction"].params["ranges"] = (friction, friction)
  cfg.commands["upper_body"].initial_level = ub_level
  if "push_robot" in cfg.events: cfg.events.pop("push_robot")
  agent = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  for g in range(env.sim.mj_model.ngeom):          # floor at the same friction (contact = max of pair)
    if env.sim.mj_model.geom_bodyid[g] == 0: env.sim.model.geom_friction[:, g, 0] = friction
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(task)(venv, asdict(agent), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")
  hc = env.command_manager.get_term("height") if height is not None else None
  term = env.command_manager.get_term("twist"); robot = env.scene["robot"]
  ids = [robot.find_joints((j,), preserve_order=True)[0][0] for j in J]
  g = torch.Generator(device=env.device).manual_seed(0)
  ang = torch.rand(n, generator=g, device=env.device) * 2 * math.pi
  spd = 0.3 + 0.2 * torch.rand(n, generator=g, device=env.device)
  walk = torch.stack([spd * torch.cos(ang), 0.6 * spd * torch.sin(ang), torch.zeros(n, device=env.device)], 1)
  dt = env.step_dt; tau = []; fell = 0
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int((walk_s + stand_s) / dt)):
      standing = k * dt >= walk_s
      term.vel_command_b[:] = 0.0 if standing else walk
      if hc is not None: hc.target[:] = height
      if hasattr(term, "is_standing_env"): term.is_standing_env[:] = standing
      obs, _, dones, _ = venv.step(pol(obs))
      fell += int(env.termination_manager.terminated.sum())
      if k * dt >= walk_s + stand_s - 15.0:
        tau.append(robot.data.actuator_force[:, ids].cpu().numpy())
  env.close()
  T = np.stack(tau)                       # (steps, envs, 6)
  m = T.mean(0)                           # per-env mean torque
  hr = np.abs(m[:, :2]).mean(0); hp_diff = np.abs(m[:, 2] - m[:, 3]); kn = m[:, 4:6].mean(0)
  return dict(hip_roll=hr, hip_roll_p90=np.percentile(np.abs(m[:, :2]), 90, axis=0),
              hip_pitch_diff=hp_diff.mean(), hip_pitch_diff_p90=np.percentile(hp_diff, 90),
              hip_pitch=m[:, 2:4].mean(0), knee=kn, falls=fell)

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
  ap.add_argument("--frictions", type=float, nargs="+", default=[1.0, 2.0])
  ap.add_argument("--task", default=TASK)
  ap.add_argument("--height", type=float, default=None, help="v11h only: pelvis-height command (m)")
  a = ap.parse_args()
  import mjlab_hv1v2  # noqa: F401
  print(f"\nstanding-squeeze gate for {a.ckpt}  task {a.task}  height {a.height}")
  allok = True
  for f in a.frictions:
    r = run(a.ckpt, f, task=a.task, height=a.height)
    ok = r["hip_roll"].max() <= 10.0 and r["hip_pitch_diff"] <= 15.0
    allok &= ok
    print(f"  friction {f:.1f}: {'PASS' if ok else 'FAIL'}  |hip roll tau| R/L {r['hip_roll'][0]:.1f}/{r['hip_roll'][1]:.1f}"
          f" (p90 {r['hip_roll_p90'][0]:.1f}/{r['hip_roll_p90'][1]:.1f})  |hip pitch R-L| {r['hip_pitch_diff']:.1f}"
          f" (p90 {r['hip_pitch_diff_p90']:.1f})  hip pitch R/L {r['hip_pitch'][0]:+.1f}/{r['hip_pitch'][1]:+.1f}"
          f"  knee R/L {r['knee'][0]:+.1f}/{r['knee'][1]:+.1f}  falls {r['falls']}")
  print(f"  overall: {'PASS' if allok else 'FAIL'}  (targets: hip roll <= 10 N.m per side, hip pitch R-L <= 15 N.m;"
        " robot v11f: 55/59, scissor 106)")
