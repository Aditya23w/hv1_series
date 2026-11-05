#!/usr/bin/env python3
"""Early go/no-go check for a v11 run (2026-11-01, after v11_s42 learned to
stand still). For checkpoint model_<IT>.pt of a run:

  (a) upper_body_level stays low (<= 0.15) until walking forms
      (walk_track_ok), from tensorboard;
  (b) swing time > 0.08 s and step length >= 0.10 m, tensorboard mean over
      the 50 iterations up to IT;
  (c) achieved forward speed at a 0.5 m/s command >= 0.4 m/s (IT >= 800),
      measured: the checkpoint pinned at fixed commands in the play env.

  python mjlab_hv1v2/scripts/check_v11_early.py --run v11b_s42 --iter 400
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import asdict

TASK = "Mjlab-Velocity-Flat-HV1V2-Decoupled-v11"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "rsl_rl", "hv1v2_velocity")


def tb(run_dir: str, it: int) -> dict[str, float]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  e = EventAccumulator(sorted(glob.glob(os.path.join(run_dir, "events*")))[-1],
                       size_guidance={"scalars": 0})
  e.Reload()
  out = {}
  for t in ("Curriculum/upper_body_level/level", "Metrics/swing_time_mean",
            "Metrics/step_length_mean", "Metrics/walk_track_err_ema", "Metrics/walk_track_ok",
            "Metrics/walk_track_err_back_ema", "Metrics/walk_track_err_lat_ema", "Metrics/walk_swing_ema",
            "Metrics/stance_ankle_roll_tau",
            "Metrics/pelvis_tilt_deg", "Episode_Termination/fell_over",
            "Curriculum/dr_level/level"):
    if t in e.Tags()["scalars"]:
      s = [x.value for x in e.Scalars(t) if it - 50 < x.step <= it]
      if s:
        out[t.split("/")[-1] if "level" not in t else t.split("/")[1]] = sum(s) / len(s)
  return out


def speeds(ckpt: str, num_envs: int = 60, steps: int = 300):
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 30.0
  tw = cfg.commands["twist"]
  tw.resampling_time_range = (1e9, 1e9)
  tw.reversal_prob = tw.stop_prob = 0.0
  tw.rel_standing_envs = 0.0
  tw.rel_heading_envs = 0.0
  agent = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")
  term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  cmds = torch.tensor([[0.3, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.5]],
                      device=env.device)
  grp = torch.arange(env.num_envs, device=env.device) % len(cmds)
  acc = torch.zeros(env.num_envs, 3, device=env.device)
  n = 0
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(steps):
      term.vel_command_b[:] = cmds[grp]
      if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
      obs, *_ = venv.step(pol(obs))
      if k >= 100:
        v = robot.data.root_link_lin_vel_b
        acc += torch.stack([v[:, 0], v[:, 1], robot.data.root_link_ang_vel_b[:, 2]], 1)
        n += 1
  env.close()
  return [(cmds[i].tolist(), (acc[grp == i] / n).mean(0).tolist()) for i in range(len(cmds))]


def standing(ckpt: str, num_envs: int = 64, steps: int = 500):
  """Zero command, upper body held at home, no payload/forces/bias: true
  pelvis tilt (deg) and stance ankle-roll |tau| (N.m) per foot, Fz > 20 N."""
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab_hv1v2 import mdp_v11

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 30.0
  tw = cfg.commands["twist"]
  tw.resampling_time_range = (1e9, 1e9)
  tw.reversal_prob = tw.stop_prob = 0.0
  r = tw.ranges
  r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
  tw.rel_standing_envs = 1.0
  tw.rel_heading_envs = 0.0
  ub = cfg.commands["upper_body"]
  ub.initial_level = 0.0
  cfg.events["hand_payload"].params["mass_range"] = (0.0, 0.0)
  hl = cfg.commands["hand_load"]
  hl.active_prob = hl.pickup_prob = 0.0
  cfg.events["imu_bias"].params.update(rp_max_rad=0.0, gyro_max=0.0)
  agent = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")
  robot = env.scene["robot"]
  rid = mdp_v11._sensor_ids(env, "feet_ground_contact", "roll_joint")
  names = env.scene["feet_ground_contact"].primary_names
  tilt, tau = [], {n: [] for n in names}
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(steps):
      obs, *_ = venv.step(pol(obs))
      if k < 100:
        continue
      g = robot.data.projected_gravity_b
      tilt.append(torch.rad2deg(torch.asin(torch.clamp(torch.norm(g[:, :2], dim=1), max=1.0))))
      t = robot.data.actuator_force[:, rid].abs()
      loaded = mdp_v11._foot_fz(env, "feet_ground_contact") > 20.0
      for i, n in enumerate(names):
        tau[n].append(t[:, i][loaded[:, i]])
  env.close()
  tl = torch.cat(tilt)
  side = {("R" if n.startswith("right") else "L"): float(torch.cat(v).mean()) if v else float("nan")
          for n, v in tau.items()}
  return float(tl.mean()), float(tl.max()), side


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True, help="run-name suffix, e.g. v11b_s42")
  ap.add_argument("--iter", type=int, required=True)
  args = ap.parse_args()
  run_dir = sorted(glob.glob(os.path.join(ROOT, f"*_{args.run}")))[-1]
  ckpt = os.path.join(run_dir, f"model_{args.iter}.pt")
  m = tb(run_dir, args.iter)
  sp = speeds(ckpt)
  vx05 = sp[1][1][0]
  st = standing(ckpt)
  print(f"\n{os.path.basename(run_dir)}  model_{args.iter}.pt")
  print("  tensorboard (mean over the last 50 iterations):",
        {k: round(v, 3) for k, v in m.items()})
  for c, a in sp:
    print(f"  cmd {[round(x, 2) for x in c]} -> achieved vx {a[0]:+.3f}  vy {a[1]:+.3f}  wz {a[2]:+.3f}")
  print(f"  standing (arms home, no load): pelvis tilt mean {st[0]:.2f} / max {st[1]:.2f} deg; "
        f"stance ankle-roll |tau| R {st[2].get('R', float('nan')):.1f} / L {st[2].get('L', float('nan')):.1f} N.m")
  print(f"  curricula: upper_body_level {m.get('upper_body_level', float('nan')):.2f}, "
        f"dr_level {m.get('dr_level', float('nan')):.3f}, walk_track_err EMA "
        f"{m.get('walk_track_err_ema', float('nan')):.3f} / back {m.get('walk_track_err_back_ema', float('nan')):.3f}"
        f" / lat {m.get('walk_track_err_lat_ema', float('nan')):.3f}, swing EMA {m.get('walk_swing_ema', float('nan')):.3f} s"
        f" (gate open {m.get('walk_track_ok', 0):.2f})")
  ub, track_ok = m.get("upper_body_level", 0.0), m.get("walk_track_ok", 0.0)
  checks = [
    ("(a) upper-body level low until walking tracks",
     ub <= 0.15 or track_ok > 0.5, f"level {ub:.2f}, walk_track_ok {track_ok:.2f}, "
                                    f"err EMA {m.get('walk_track_err_ema', float('nan')):.3f}"),
    ("(b) swing time > 0.08 s", m.get("swing_time_mean", 0) > 0.08, f"{m.get('swing_time_mean', 0):.3f} s"),
    ("(b) step length >= 0.10 m", m.get("step_length_mean", 0) >= 0.10, f"{m.get('step_length_mean', 0):.3f} m"),
  ]
  if args.iter >= 800:
    checks.append(("(c) vx at 0.5 m/s command >= 0.4", vx05 >= 0.4, f"{vx05:.3f} m/s"))
  else:
    checks.append(("(c) vx at 0.5 m/s command (gate at 800)", True, f"{vx05:.3f} m/s (info)"))
  for name, ok, val in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {val}")


if __name__ == "__main__":
  main()
