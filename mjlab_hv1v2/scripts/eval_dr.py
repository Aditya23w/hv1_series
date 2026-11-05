#!/usr/bin/env python3
"""Population evaluation of a whole-body checkpoint in the TRAINING env, on GPU.

Thousands of envs with everything training randomises: domain randomisation
(friction, CoM, PD gains, armature, effort limits), pushes, observation noise,
random velocity commands, 20 s episodes. It answers what sim2sim cannot: how
often the policy falls and how well it tracks across the whole training
distribution, not at a handful of hand-picked commands.

The plant is the CURRENT model; the env's anchor (init_state) is set per
checkpoint with --keyframe, because the action offset and the joint_pos
observation are both taken from it. v1-v8 trained on HOME_KEYFRAME_V1_V8.

  python mjlab_hv1v2/scripts/eval_dr.py --ckpt exported/deploy_v5/model_47998.pt \
      --keyframe v1_v8 --num-envs 4096 --seconds 60 --seed 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_hv1v2 import hv1v2_constants as C

TASK = "Mjlab-Velocity-Flat-HV1V2"
KEYFRAMES = {"current": C.HOME_KEYFRAME, "v1_v8": C.HOME_KEYFRAME_V1_V8}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--keyframe", choices=KEYFRAMES, required=True)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default=None)
  args = ap.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)
  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg.seed = args.seed
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.scene.entities["robot"].init_state = KEYFRAMES[args.keyframe]
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(venv, asdict(agent_cfg), device="cuda:0")
  runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")

  robot = env.scene["robot"]
  n_steps = int(args.seconds / env.step_dt)
  episodes = falls = 0
  err_xy = err_wz = 0.0
  err_xy_moving = 0.0
  n_moving = 0
  n_samples = 0
  obs = venv.get_observations()
  with torch.inference_mode():
    for _ in range(n_steps):
      cmd = env.command_manager.get_command("twist").clone()
      obs, _, dones, _ = venv.step(policy(obs))
      # Tracking on the post-step state against the command that was acted on,
      # excluding envs that just reset (their state is the new episode's).
      live = ~dones.bool()
      v = robot.data.root_link_lin_vel_b[:, :2]
      w = robot.data.root_link_ang_vel_b[:, 2]
      exy = torch.linalg.norm(v - cmd[:, :2], dim=1)[live]
      ewz = (w - cmd[:, 2]).abs()[live]
      moving = (torch.linalg.norm(cmd[:, :2], dim=1) > 0.1)[live]
      err_xy += float(exy.sum())
      err_wz += float(ewz.sum())
      err_xy_moving += float(exy[moving].sum())
      n_moving += int(moving.sum())
      n_samples += int(live.sum())
      term = env.termination_manager.terminated
      episodes += int(dones.sum())
      falls += int(term.sum())

  res = dict(
    ckpt=args.ckpt, keyframe=args.keyframe, seed=args.seed, num_envs=args.num_envs,
    seconds=args.seconds, episodes_ended=episodes, falls=falls,
    falls_per_env_minute=falls / (args.num_envs * args.seconds / 60),
    mean_err_xy=err_xy / n_samples, mean_err_xy_moving=err_xy_moving / max(1, n_moving),
    mean_err_wz=err_wz / n_samples,
  )
  print(json.dumps(res))
  if args.out:
    Path(args.out).write_text(json.dumps(res, indent=1))
  env.close()


if __name__ == "__main__":
  main()
