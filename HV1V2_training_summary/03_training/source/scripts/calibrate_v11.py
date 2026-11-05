#!/usr/bin/env python3
"""Calibrate v11's new reward weights on v7 rollouts.

A fresh v11 policy cannot be rolled out yet, and v7's actor (403 inputs)
cannot run in the v11 env (837), so -- as calibrate_v10.py did -- v7 is rolled
out in the v10 env (v7's observation layout, v8 timing, real ankle stiffness)
and every new v11 term is evaluated on that behaviour by calling its function
directly. Each weight is set so the term costs --frac (per term, below) of
v7's per-step reward there:

    weight = -frac * R_step / (mean raw value per step * dt)

The gate of torso_quiet_under_ub is computed from the upper-body target's
finite difference (the v10 command has no ub_rate). Prints the weights to
paste into V11_WEIGHTS / V11_SMOOTH_EXTRA in env_cfgs.py.

  python mjlab_hv1v2/scripts/calibrate_v11.py \
      --ckpt logs/rsl_rl/hv1v2_velocity/2026-10-06_17-58-29_v7/model_29999.pt
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
from mjlab.utils.torch import configure_torch_backends
from mjlab_hv1v2 import mdp as hv1v2_mdp
from mjlab_hv1v2 import mdp_v11
from mjlab_hv1v2.hv1v2_constants import FOOT_GEOM_RE

TASK = "Mjlab-Velocity-Flat-HV1V2-Decoupled-v10"
# term -> target share of v7's per-step reward
FRAC = {
  "torso_quiet_under_ub": 0.04,
  "loaded_sole_tilt": 0.03,
  "pad_load_balance": 0.02,
  "ankle_roll_torque_stance": 0.03,
  "still_stand_quiet": 0.01,
  "ankle_joint_acc": 0.03,
  "ankle_torque_rate": 0.03,
}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--seconds", type=float, default=20.0)
  args = ap.parse_args()

  configure_torch_backends()
  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (ContactSensorCfg(
    name="foot_pads",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_RE, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1),)
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")

  dt = env.step_dt
  robot = env.scene["robot"]
  pelvis = SceneEntityCfg("robot", body_names=("pelvis",))
  pelvis.resolve(env.scene)
  ankles = SceneEntityCfg("robot", joint_names=(r".*_ankle_pitch_joint$", r".*_ankle_roll_joint$"))
  ankles.resolve(env.scene)
  ub = env.command_manager.get_term("upper_body")
  prev_t, prev_tau, prev_qd = ub.target.clone(), None, None
  raw = {k: 0.0 for k in FRAC}
  extra = {"upright_v7": 0.0, "upright_v11": 0.0, "tilt_deg": 0.0}
  total_rew, n = 0.0, 0
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int(args.seconds / dt)):
      obs, rew, dones, _ = venv.step(policy(obs))
      alive = (~dones.bool()).float()
      ub.ub_rate = (ub.target - prev_t) / dt          # what V11UpperBodyCommand exposes
      prev_t = ub.target.clone()
      v = {
        "torso_quiet_under_ub": mdp_v11.torso_quiet_under_ub(
          env, "upper_body", rate_ref=1.0, w_weight=0.1, asset_cfg=pelvis),
        "loaded_sole_tilt": mdp_v11.loaded_sole_tilt(env, "feet_ground_contact", fz_min=20.0),
        "pad_load_balance": mdp_v11.pad_load_balance(
          env, "foot_pads", "twist", 0.1, min_share=0.10, fz_min=20.0),
        "ankle_roll_torque_stance": mdp_v11.ankle_roll_torque_stance(
          env, "feet_ground_contact", fz_min=20.0, deadband=4.0),
        "still_stand_quiet": hv1v2_mdp.still_stand_quiet(env, "twist", "upper_body", 1.0),
      }
      qd = robot.data.joint_vel[:, ankles.joint_ids]
      tau = robot.data.actuator_force[:, ankles.joint_ids]
      if prev_qd is not None and k > 2:
        v["ankle_joint_acc"] = torch.sum(torch.square((qd - prev_qd) / dt), dim=1)
        v["ankle_torque_rate"] = torch.sum(torch.square((tau - prev_tau) / dt), dim=1)
      else:
        v["ankle_joint_acc"] = torch.zeros_like(rew)
        v["ankle_torque_rate"] = torch.zeros_like(rew)
      prev_qd, prev_tau = qd.clone(), tau.clone()
      if k < 50:          # skip the reset transient
        continue
      for name in FRAC:
        raw[name] += float((v[name] * alive).sum())
      extra["upright_v11"] += float((mdp_v11.upright_tight(env, 0.08, pelvis) * alive).sum())
      extra["upright_v7"] += float((mdp_v11.upright_tight(env, 0.4472, pelvis) * alive).sum())
      extra["tilt_deg"] += float(env.extras["log"]["Metrics/pelvis_tilt_deg"]) * float(alive.sum())
      total_rew += float((rew * alive).sum())
      n += int(alive.sum())

  r_step = total_rew / n
  print(f"env-steps {n}; v7 reward per step {r_step:.4f} (rew already x dt={dt})")
  print(f"pelvis tilt mean {extra['tilt_deg'] / n:.2f} deg; upright v7 kernel {extra['upright_v7'] / n:.3f} "
        f"vs v11 kernel {extra['upright_v11'] / n:.3f}")
  for name, frac in FRAC.items():
    mean = raw[name] / n
    w = -frac * r_step / (mean * dt) if mean > 0 else float("nan")
    print(f"  {name:26s} mean raw {mean:.6g}  -> weight {w:.4g}  ({100 * frac:.0f}% of reward)")
  env.close()


if __name__ == "__main__":
  main()
