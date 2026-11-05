#!/usr/bin/env python3
"""Widen a checkpoint's observation inputs for a task that APPENDS terms
(v11h: v11g model_19000, actor 837 -> 838, critic 171 -> 173), so the new
policy starts out acting exactly like the old one.

What it does (never touches the source file):
  * builds both tasks' envs (4 envs) only to read the observation layouts
    by NAME, and asserts the new layout = the old one + terms appended LAST
    (term names and widths, actor and critic). Nothing is hard-coded;
  * first layer (mlp.0.weight) of actor and critic: inserts ZERO columns for
    the new inputs, so their output is unchanged whatever the new inputs are;
  * observation normaliser: the new inputs get mean 0, std --new-std (plus
    any observation-noise variance). Mean 0 keeps "home" (input 0) exactly
    neutral; the count is untouched, so rsl_rl's running stats move these
    entries only slowly, from a sane scale;
  * Adam state for those two weights: exp_avg padded with 0, exp_avg_sq
    padded with the row-wise mean of the existing columns, so the 837 / 171
    pre-trained columns keep their momentum and scaling and the new columns
    start at a typical per-row step size;
  * everything else (other layers, the action std, iteration, env_state,
    i.e. common_step_counter for the curricula) is copied as is.

Verification (--check, default on): loads the old checkpoint into the old
task's runner and the new one into the new task's runner, feeds the same
observations (new inputs set to random values AND to 0) and asserts actor
means and critic values agree to 1e-6.

  python mjlab_hv1v2/scripts/expand_obs.py \\
      --ckpt logs/rsl_rl/hv1v2_velocity/2026-11-02_13-24-36_v11g_s43/model_19000.pt \\
      --old-task Mjlab-Velocity-Flat-HV1V2-Decoupled-v11 \\
      --new-task Mjlab-Velocity-Flat-HV1V2-Decoupled-v11h \\
      --out logs/rsl_rl/hv1v2_velocity/2026-11-02_13-24-36_v11g_s43/model_19000_v11h.pt
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

GROUPS = (("actor", "actor_state_dict"), ("critic", "critic_state_dict"))


def layout(task: str, device: str):
  """{group: [(term, width, noise_var)]} for the task's training env."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  om = env.observation_manager
  out = {}
  for g, _ in GROUPS:
    rows = []
    for name, tcfg, dim in zip(om.active_terms[g], om._group_obs_term_cfgs[g], om.group_obs_term_dim[g]):
      nv = 0.0
      n = getattr(tcfg, "noise", None)
      if n is not None and hasattr(n, "n_min"):
        nv = (float(n.n_max) - float(n.n_min)) ** 2 / 12.0
      rows.append((name, int(np.prod(dim)), nv))
    out[g] = rows
  env.close()
  return out


def expand(ck: dict, old: dict, new: dict, new_std: float) -> tuple[dict, list[str]]:
  ck = copy.deepcopy(ck)
  report = []
  opt = ck["optimizer_state_dict"]
  for g, sd_key in GROUPS:
    o, n = old[g], new[g]
    assert [(a, b) for a, b, _ in n[:len(o)]] == [(a, b) for a, b, _ in o], \
      f"{g}: the new layout does not start with the old one -- only APPENDED terms are supported"
    w_old = sum(w for _, w, _ in o)
    added = n[len(o):]
    k = sum(w for _, w, _ in added)
    sd = ck[sd_key]
    W = sd["mlp.0.weight"]
    assert W.shape[1] == w_old, f"{g}: checkpoint has {W.shape[1]} inputs, old layout {w_old}"
    if k == 0:
      continue
    # first layer: zero columns at the end
    sd["mlp.0.weight"] = torch.cat([W, torch.zeros(W.shape[0], k, dtype=W.dtype)], dim=1)
    # normaliser
    mean = sd["obs_normalizer._mean"].reshape(-1)
    var = sd["obs_normalizer._var"].reshape(-1)
    add_var = []
    for name, w, nv in added:
      add_var += [new_std ** 2 + nv] * w
      report.append(f"{g:6s} + {name:12s} width {w}  normaliser mean 0, std {math.sqrt(new_std**2 + nv):.4f}")
    av = torch.tensor(add_var, dtype=var.dtype)
    sd["obs_normalizer._mean"] = torch.cat([mean, torch.zeros(k, dtype=mean.dtype)]).reshape(1, -1)
    sd["obs_normalizer._var"] = torch.cat([var, av]).reshape(1, -1)
    sd["obs_normalizer._std"] = torch.sqrt(sd["obs_normalizer._var"])
    # Adam state of that weight, found by shape (exactly one match per group)
    hits = [i for i, s in opt["state"].items()
            if "exp_avg" in s and tuple(s["exp_avg"].shape) == tuple(W.shape)]
    assert len(hits) == 1, f"{g}: {len(hits)} optimizer states shaped {tuple(W.shape)}"
    s = opt["state"][hits[0]]
    s["exp_avg"] = torch.cat([s["exp_avg"], torch.zeros(W.shape[0], k, dtype=W.dtype)], dim=1)
    sq = s["exp_avg_sq"]
    s["exp_avg_sq"] = torch.cat([sq, sq.mean(dim=1, keepdim=True).expand(-1, k)], dim=1)
    report.append(f"{g:6s}   mlp.0.weight {tuple(W.shape)} -> {tuple(sd['mlp.0.weight'].shape)}; "
                  f"Adam state #{hits[0]} padded")
  return ck, report


def check(old_task, new_task, old_ckpt, new_ckpt, old_l, new_l, device) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  def runner(task, ckpt):
    cfg = load_env_cfg(task, play=False)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent = load_rl_cfg(task)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    r = load_runner_cls(task)(venv, asdict(agent), device=device)
    r.load(str(ckpt), map_location=device)          # full load: actor, critic, optimizer
    return r, env

  r_old, e_old = runner(old_task, old_ckpt)
  r_new, e_new = runner(new_task, new_ckpt)
  a_old, a_new = r_old.alg.actor, r_new.alg.actor
  c_old, c_new = r_old.alg.critic, r_new.alg.critic
  g = torch.Generator(device=device).manual_seed(0)
  B = 256
  worst = {}
  with torch.inference_mode():
    for grp, m_old, m_new in (("actor", a_old, a_new), ("critic", c_old, c_new)):
      w_old = sum(w for _, w, _ in old_l[grp])
      w_new = sum(w for _, w, _ in new_l[grp])
      x = torch.randn(B, w_old, device=device, generator=g)
      for label, extra in (("random", torch.randn(B, w_new - w_old, device=device, generator=g) * 0.1),
                           ("zero", torch.zeros(B, w_new - w_old, device=device))):
        y_old = _forward(m_old, x, grp)
        y_new = _forward(m_new, torch.cat([x, extra], 1), grp)
        worst[f"{grp}/{label}"] = (y_old - y_new).abs().max().item()
  e_old.close(); e_new.close()
  for k, v in worst.items():
    print(f"  check {k:14s} max |old - new| = {v:.2e}")
  assert all(v < 1e-6 for v in worst.values()), "expanded checkpoint does not reproduce the original"
  # the optimizer state loaded and matches the new shapes
  for i, s in r_new.alg.optimizer.state.items():
    assert s["exp_avg"].shape == i.shape and s["exp_avg_sq"].shape == i.shape
  print("  check optimizer state shapes match the new parameters")


def _forward(model, x, grp):
  """Deterministic output (rsl_rl MLPModel.forward defaults to
  stochastic_output=False): the actor's mean action, the critic's value."""
  from tensordict import TensorDict
  return model(TensorDict({k: x for k in model.obs_groups}, batch_size=[x.shape[0]]))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--old-task", default="Mjlab-Velocity-Flat-HV1V2-Decoupled-v11")
  ap.add_argument("--new-task", default="Mjlab-Velocity-Flat-HV1V2-Decoupled-v11h")
  ap.add_argument("--out", required=True)
  ap.add_argument("--new-std", type=float, default=0.05,
                  help="normaliser std for the new inputs (v11h height: range 0.089 m)")
  ap.add_argument("--no-check", action="store_true")
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  import mjlab_hv1v2  # noqa: F401  (registers the tasks)

  src, out = Path(args.ckpt), Path(args.out)
  assert src.resolve() != out.resolve(), "refusing to overwrite the source checkpoint"
  old_l = layout(args.old_task, args.device)
  new_l = layout(args.new_task, args.device)
  ck = torch.load(src, map_location="cpu", weights_only=False)
  new_ck, report = expand(ck, old_l, new_l, args.new_std)
  print(f"{src.name} ({args.old_task}) -> {out.name} ({args.new_task})")
  for line in report:
    print("  " + line)
  print(f"  iter {new_ck.get('iter')}, env_state {new_ck.get('infos', {}).get('env_state')}")
  torch.save(new_ck, out)
  print(f"  wrote {out}")
  if not args.no_check:
    check(args.old_task, args.new_task, src, out, old_l, new_l, args.device)
    print("PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
