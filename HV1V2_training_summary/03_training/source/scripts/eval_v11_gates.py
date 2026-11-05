#!/usr/bin/env python3
"""Phase-2 sim gates for a v11 checkpoint (plan: keen-pondering-fiddle.md).

Plays the checkpoint in the v11 play env (no corruption, no pushes, IMU bias 0,
DR at its full width) in five scenarios and prints each gate against v7's
numbers:

  stand    zero command, full-envelope upper-body motion (level 1.0), 2 kg in
           each hand: max / p99 pelvis tilt, pads loaded (4/4), stance
           ankle-roll |tau|, still-standing (unnecessary steps), raw ankle-roll
           target inside +-0.26 rad, pelvis gyro 3-8 Hz share.
  walk     vx 0.5 with the same sweeps: max / p99 pelvis tilt.
  asym_r / asym_l   2 kg in ONE hand, standing, sweeps: per-side stance
           ankle-roll |tau|; the mirrored L/R delta must be < 3 N.m.
  track    vx 1.0, arms still: achieved forward speed.

Latency margin (no fall to 40 ms) is NOT here: the leg action delay in the
v11 env stops at one control step. Run the corrected
AMP_mjlab/scripts/plant_latency_test_hv1v2.py for it once that harness builds
the v11 observation.

  python mjlab_hv1v2/scripts/eval_v11_gates.py --ckpt logs/rsl_rl/.../model_N.pt
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_hv1v2 import mdp_v11
from mjlab_hv1v2.hv1v2_constants import HAND_BODY_NAMES

TASK = "Mjlab-Velocity-Flat-HV1V2-Decoupled-v11"


def make_env(scenario: str, num_envs: int):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = 30.0
  tw = cfg.commands["twist"]
  tw.resampling_time_range = (1e9, 1e9)
  tw.reversal_prob = tw.stop_prob = 0.0
  r = tw.ranges
  vx = {"stand": 0.0, "asym_r": 0.0, "asym_l": 0.0, "walk": 0.5, "track": 1.0}[scenario]
  r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (vx, vx), (0.0, 0.0), (0.0, 0.0)
  tw.rel_standing_envs = 1.0 if vx == 0.0 else 0.0
  tw.rel_heading_envs = 0.0
  ub = cfg.commands["upper_body"]
  ub.initial_level = 0.0 if scenario == "track" else 1.0
  ub.quiet_prob = 0.0
  pay = cfg.events["hand_payload"].params
  if scenario == "asym_r":
    pay["body_names"], pay["mass_range"] = ("right_wrist_yaw_link",), (2.0, 2.0)
  elif scenario == "asym_l":
    pay["body_names"], pay["mass_range"] = ("left_wrist_yaw_link",), (2.0, 2.0)
  elif scenario == "track":
    pay["mass_range"] = (0.0, 0.0)
  else:
    pay["body_names"], pay["mass_range"] = HAND_BODY_NAMES, (2.0, 2.0)
  hl = cfg.commands["hand_load"]
  hl.active_prob = hl.pickup_prob = 0.0
  cfg.events["imu_bias"].params.update(rp_max_rad=0.0, gyro_max=0.0)
  return cfg


def run(ckpt: str, scenario: str, num_envs: int, seconds: float):
  cfg = make_env(scenario, num_envs)
  agent = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  robot = env.scene["robot"]
  act = env.action_manager.get_term("joint_pos")
  roll_cols = [i for i, n in enumerate(act.target_names) if "ankle_roll" in n]
  dt = env.step_dt
  tilt, gyro, pads, tau_r, tau_l, vxs, raw_ok, steps, fell = [], [], [], [], [], [], [], 0.0, 0
  rid = mdp_v11._sensor_ids(env, "feet_ground_contact", "roll_joint")
  names = env.scene["feet_ground_contact"].primary_names
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int(seconds / dt)):
      a = policy(obs)
      raw = a[:, roll_cols] * act._scale[:, roll_cols] + act._offset[:, roll_cols]
      obs, rew, dones, extras = venv.step(a)
      fell += int(env.termination_manager.terminated.sum())
      if k < int(2.0 / dt):
        continue
      g = robot.data.projected_gravity_b
      tilt.append(torch.rad2deg(torch.asin(torch.clamp(torch.norm(g[:, :2], dim=1), max=1.0))).cpu())
      gyro.append(robot.data.root_link_ang_vel_b[:, :2].cpu())
      raw_ok.append(((raw.abs() <= 0.26).float().mean()).item())
      fz = mdp_v11._foot_fz(env, "feet_ground_contact")
      tau = robot.data.actuator_force[:, rid].abs()
      loaded = fz > 20.0
      for i, n in enumerate(names):
        (tau_r if n.startswith("right") else tau_l).append(tau[:, i][loaded[:, i]].cpu())
      pf = mdp_v11._foot_fz(env, "foot_pads")
      pads.append(((pf > 0.1 * pf.reshape(-1, 2, 4).sum(-1).repeat_interleave(4, dim=1))
                   .reshape(-1, 2, 4).all(-1).float().mean()).item())
      vxs.append(robot.data.root_link_lin_vel_b[:, 0].mean().item())
      steps += float(env.metrics_manager._step_buffer.get("standing_steps", torch.zeros(1)).sum()) \
        if hasattr(env.metrics_manager, "_step_buffer") else 0.0
  env.close()
  t = torch.cat(tilt).numpy()
  gy = torch.stack(gyro).numpy()  # (T, B, 2)
  spec = np.abs(np.fft.rfft(gy - gy.mean(0), axis=0)) ** 2
  f = np.fft.rfftfreq(gy.shape[0], d=dt)
  share = float(spec[(f >= 3) & (f <= 8)].sum() / max(spec[f > 0.2].sum(), 1e-12))
  cat = lambda x: torch.cat(x).numpy() if x else np.zeros(1)
  return dict(tilt_max=float(t.max()), tilt_p99=float(np.percentile(t, 99)), tilt_mean=float(t.mean()),
              band_3_8=share, pads_4of4=float(np.mean(pads)), tau_r=float(cat(tau_r).mean()),
              tau_l=float(cat(tau_l).mean()), roll_target_ok=float(np.mean(raw_ok)),
              vx=float(np.mean(vxs)), falls=fell)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--only", default="stand,walk,asym_r,asym_l,track")
  args = ap.parse_args()
  configure_torch_backends()
  R = {s: run(args.ckpt, s, args.num_envs, args.seconds) for s in args.only.split(",")}
  gates = []
  def gate(name, ok, val, target, v7):
    gates.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {val:<22s} target {target:<14s} v7 {v7}")
  print(f"\nv11 gates for {args.ckpt}")
  if "stand" in R:
    s = R["stand"]
    gate("stand tilt max (sweeps, 2 kg each hand)", s["tilt_max"] <= 1.5, f"{s['tilt_max']:.2f} deg (p99 {s['tilt_p99']:.2f})", "<= 1.5 deg", "3.8-5.0 deg")
    gate("pads 4/4 loaded, standing", s["pads_4of4"] >= 0.95, f"{100 * s['pads_4of4']:.1f}%", ">= 95%", "2/4 pads")
    tau = max(s["tau_r"], s["tau_l"])
    gate("stance ankle-roll |tau|", tau < 5.0, f"R {s['tau_r']:.1f} / L {s['tau_l']:.1f} N.m", "< 5 N.m", "25 N.m")
    gate("raw ankle-roll target inside +-0.26", s["roll_target_ok"] >= 0.999, f"{100 * s['roll_target_ok']:.2f}%", ">= 99.9%", "clips on reversals")
    gate("pelvis gyro 3-8 Hz share, standing", s["band_3_8"] <= 0.10, f"{s['band_3_8']:.3f}", "<= 0.10", "0.13 (gate missed)")
    gate("no falls standing", s["falls"] == 0, f"{s['falls']}", "0", "-")
  if "walk" in R:
    w = R["walk"]
    gate("walk 0.5 m/s tilt max (sweeps, 2 kg)", w["tilt_max"] <= 2.5, f"{w['tilt_max']:.2f} deg (p99 {w['tilt_p99']:.2f})", "<= 2.5 deg", "-")
  if "asym_r" in R and "asym_l" in R:
    a, b = R["asym_r"], R["asym_l"]
    d = max(abs(a["tau_r"] - b["tau_l"]), abs(a["tau_l"] - b["tau_r"]))
    gate("asymmetric payload: mirrored roll-|tau| delta", d < 3.0, f"{d:.2f} N.m", "< 3 N.m", "-")
    gate("asymmetric payload: pads 4/4", min(a["pads_4of4"], b["pads_4of4"]) >= 0.95,
         f"{100 * min(a['pads_4of4'], b['pads_4of4']):.1f}%", ">= 95%", "-")
  if "track" in R:
    gate("velocity tracking at 1.0 m/s", R["track"]["vx"] >= 0.9, f"{R['track']['vx']:.3f} m/s", ">= 0.9 m/s", "0.77 m/s")
  print(f"\n  {sum(gates)}/{len(gates)} gates passed (latency gate: see docstring)")


if __name__ == "__main__":
  main()
