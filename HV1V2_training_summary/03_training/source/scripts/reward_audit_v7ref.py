#!/usr/bin/env python3
"""v7 reference for the v11 reward budget (2026-11-01). v7 model_29999 walks
in the v10 env (its obs layout) at a fixed command; prints the v10 env's own
weighted per-step terms (shared with v11: tracking, gait, v7 smoothness) and
the v11-only terms evaluated on the same states at v11's FINAL weights
(weighted, no dt), i.e. what a mature smooth walker pays under v11.

  python scripts/reward_audit_v7ref.py --cmd 0.5 0 0 --ub-level 0.2 --no-load
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_hv1v2 import env_cfgs as E
from mjlab_hv1v2 import mdp_v11
from mjlab_hv1v2.hv1v2_constants import FOOT_GEOM_RE

TASK = "Mjlab-Velocity-Flat-HV1V2-Decoupled-v10"
CKPT = "logs/rsl_rl/hv1v2_velocity/2026-10-06_17-58-29_v7/model_29999.pt"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default=CKPT)
  ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
  ap.add_argument("--ub-level", type=float, default=None)
  ap.add_argument("--no-load", action="store_true")
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=500)
  a = ap.parse_args()
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = a.num_envs
  cfg.episode_length_s = 30.0
  tw = cfg.commands["twist"]
  tw.resampling_time_range = (1e9, 1e9)
  for k in ("reversal_prob", "stop_prob"):
    if hasattr(tw, k):
      setattr(tw, k, 0.0)
  tw.rel_standing_envs = 0.0
  tw.rel_heading_envs = 0.0
  if a.ub_level is not None:
    cfg.commands["upper_body"].initial_level = a.ub_level
  if a.no_load:
    for k in ("hand_payload", "hand_force"):
      cfg.events.pop(k, None)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (ContactSensorCfg(
    name="foot_pads", primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_RE, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1),)
  agent = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent), device="cuda:0")
  runner.load(a.ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")
  term = env.command_manager.get_term("twist")
  rm = env.reward_manager
  robot = env.scene["robot"]
  dt = env.step_dt
  pelvis = SceneEntityCfg("robot", body_names=("pelvis",)); pelvis.resolve(env.scene)
  ankles = SceneEntityCfg("robot", joint_names=(r".*_ankle_pitch_joint$", r".*_ankle_roll_joint$"))
  ankles.resolve(env.scene)
  ub = env.command_manager.get_term("upper_body")
  W = E.V11_WEIGHTS
  sm = dict(E.V11_SMOOTH_EXTRA)
  c = torch.tensor(a.cmd, device=env.device)
  base = torch.zeros(len(rm._term_names), device=env.device)
  extra: dict[str, float] = {}
  vel = torch.zeros(3, device=env.device)
  prev_t, prev_qd, prev_tau, n, resets = ub.target.clone(), None, None, 0, 0
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(a.steps):
      term.vel_command_b[:] = c
      if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
      obs, _, dones, _ = venv.step(pol(obs))
      ub.ub_rate = (ub.target - prev_t) / dt
      prev_t = ub.target.clone()
      qd = robot.data.joint_vel[:, ankles.joint_ids]
      tau = robot.data.actuator_force[:, ankles.joint_ids]
      v = {}
      if prev_qd is not None:
        v["ankle_joint_acc"] = sm["ankle_joint_acc"][1] * torch.sum(torch.square((qd - prev_qd) / dt), 1)
        v["ankle_torque_rate"] = sm["ankle_torque_rate"][1] * torch.sum(torch.square((tau - prev_tau) / dt), 1)
      prev_qd, prev_tau = qd.clone(), tau.clone()
      if k < 100:
        continue
      v["torso_quiet_under_ub"] = W["torso_quiet_under_ub"] * mdp_v11.torso_quiet_under_ub(
        env, "upper_body", rate_ref=1.0, w_weight=0.1, asset_cfg=pelvis,
        twist_name="twist", walk_cmd=E.V11_WALK_CMD, walk_scale=E.V11_WALK_SCALE)
      v["loaded_sole_tilt"] = W["loaded_sole_tilt"] * mdp_v11.loaded_sole_tilt(env, "feet_ground_contact", fz_min=20.0)
      v["ankle_roll_torque_stance"] = W["ankle_roll_torque_stance"] * mdp_v11.ankle_roll_torque_stance(
        env, "feet_ground_contact", fz_min=20.0, deadband=4.0, command_name="twist",
        command_threshold=0.1, single_support_scale=E.V11_SINGLE_SUPPORT_SCALE)
      v["upright_wide(v11)"] = E.V11_UPRIGHT_WIDE[1] * mdp_v11.upright_tight(env, E.V11_UPRIGHT_WIDE[0], pelvis)
      v["upright_precise(v11)"] = E.V11_UPRIGHT_PRECISE[1] * mdp_v11.upright_tight(env, E.V11_UPRIGHT_PRECISE[0], pelvis)
      w = robot.data.root_link_ang_vel_b
      v["body_ang_vel(v11)"] = E.V11_BODY_ANG_VEL[1] * mdp_v11.body_ang_vel_cmd_scaled(
        env, pelvis, "twist", E.V11_WALK_CMD, E.V11_WALK_SCALE)
      ati = rm._term_names.index("air_time")
      lv = robot.data.root_link_lin_vel_b[:, :2]
      trk = torch.exp(-torch.sum(torch.square(lv - c[:2]), 1) / E.V11_AIR_TRACK_VAR)
      v["air_time(v11d gated) minus v10 air_time"] = rm._step_reward[:, ati] * (trk - 1.0)
      for nm, x in v.items():
        extra[nm] = extra.get(nm, 0.0) + float(x.mean())
      base += rm._step_reward.mean(0)
      lv = robot.data.root_link_lin_vel_b
      vel += torch.stack([lv[:, 0].mean(), lv[:, 1].mean(), w[:, 2].mean()])
      resets += int(dones.sum())
      n += 1
  env.close()
  print(f"\nv7 in {TASK}  cmd {a.cmd}  ub_level {a.ub_level}  no_load {a.no_load}")
  print("v10-env terms (weighted per step, no dt):")
  for nm, x in sorted(zip(rm._term_names, (base / n).tolist()), key=lambda t: -abs(t[1])):
    print(f"  {nm:28s}{x:10.4f}")
  print("v11-only terms at v11 final weights:")
  for nm, x in extra.items():
    print(f"  {nm:28s}{x / n:10.4f}")
  vv = (vel / n).tolist()
  print(f"  vx {vv[0]:+.3f} vy {vv[1]:+.3f} wz {vv[2]:+.3f}  resets {resets}")


if __name__ == "__main__":
  main()
