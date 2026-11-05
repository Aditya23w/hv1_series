#!/usr/bin/env python3
"""Side-by-side gait / tracking / effort table for whole-body ONNX policies.

Built on eval_plant.Sim, so the plant (`--plant train|real`) and the deployed
command timing (targets interpolated over the 4 substeps) are eval_plant's.
Unlike eval_plant, each episode starts at the POLICY'S OWN anchor
(default_joint_pos from its ONNX) plus +-0.01 rad seeded noise, so a policy
trained on an older home pose is not handicapped by the current HOME_KEYFRAME.

Per (policy, plant, command, seed): stand 2 s, then hold the command 10 s and
measure the last 8 s.

  python mjlab_hv1v2/scripts/compare_gait.py \
      --onnx v5=exported/deploy_v5/policy.onnx v9=exported/deploy_v9/policy.onnx \
      --plant train real --seeds 5 --out /tmp/gait.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_plant as E  # noqa: E402
import sim2sim as S  # noqa: E402

DT = E.DT
COMMANDS = {
  "stand": (0.0, 0.0, 0.0),
  "vx0.3": (0.3, 0.0, 0.0),
  "vx0.6": (0.6, 0.0, 0.0),
  "vx1.0": (1.0, 0.0, 0.0),
  "vx-0.5": (-0.5, 0.0, 0.0),
  "vy+0.3": (0.0, 0.3, 0.0),
  "vy-0.3": (0.0, -0.3, 0.0),
  "wz+0.5": (0.0, 0.0, 0.5),
  "wz-0.5": (0.0, 0.0, -0.5),
  "vx0.5_wz0.4": (0.5, 0.0, 0.4),
}
STAND_S, CMD_S, SKIP_S = 2.0, 10.0, 2.0
LEG = ("hip", "knee", "ankle")


def yaw_of(R):
  return float(np.arctan2(R[1, 0], R[0, 0]))


def episode(job):
  name, onnx, plant, cname, seed = job
  cmd = np.array(COMMANDS[cname])
  sim = E.Sim(onnx, plant, False, seed)
  rob, pol = sim.rob, sim.pol
  rng = np.random.default_rng(seed)
  rob.reset()
  rob.d.qpos[rob.qadr] = pol.default_qpos + rng.uniform(-0.01, 0.01, len(rob.qadr))
  mujoco.mj_forward(rob.m, rob.d)
  pol.reset()

  m, d = rob.m, rob.d
  leg = np.array([i for i, n in enumerate(pol.joint_names) if any(k in n for k in LEG)])
  from mjlab.utils.string import resolve_expr
  from mjlab_hv1v2.hv1v2_constants import CONTINUOUS_TORQUE
  cont = np.array(resolve_expr(CONTINUOUS_TORQUE, pol.joint_names))
  gait = S.GaitStats(rob, pol.joint_names, warmup_s=STAND_S + SKIP_S)

  for _ in range(int(STAND_S / DT)):
    sim.step((0, 0, 0))
    gait.update(rob.qvel_joints())
  rest = np.median(np.array(sim.fz)[-25:], axis=0)

  xy, yaw, acts, tau, qd, rp = [], [], [], [], [], []
  for k in range(int(CMD_S / DT)):
    sim.step(cmd)
    gait.update(rob.qvel_joints())
    if k * DT < SKIP_S:
      continue
    xy.append(d.qpos[:2].copy())
    yaw.append(yaw_of(rob.root_rot()))
    acts.append(pol.prev_action.copy())
    t = d.qfrc_actuator[rob.vadr].copy()
    if rob.ankle_pd is not None:
      t += d.qfrc_applied[rob.vadr]
    tau.append(t)
    qd.append(rob.qvel_joints())
    rp.append(sim.rp[-1])
    if sim.fell is not None:
      break

  out = dict(policy=name, plant=plant, cmd=cname, seed=seed, fell=sim.fell is not None)
  if sim.fell is not None or len(xy) < 50:
    return out
  xy, yaw = np.array(xy), np.unwrap(np.array(yaw))
  acts, tau, qd, rp = map(np.array, (acts, tau, qd, rp))
  span = (len(xy) - 1) * DT
  # Body-frame velocity: integrate world displacement rotated into the
  # instantaneous heading, so turning commands are measured correctly.
  dxy = np.diff(xy, axis=0)
  c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
  vx = float((c * dxy[:, 0] + s * dxy[:, 1]).sum() / span)
  vy = float((-s * dxy[:, 0] + c * dxy[:, 1]).sum() / span)
  wz = float((yaw[-1] - yaw[0]) / span)
  ratio = np.abs(tau) / cont
  power = np.abs(tau[:, leg] * qd[:, leg]).sum(axis=1)

  peaks = []
  fz = np.array(sim.fz)[-len(xy):]
  for f in (0, 1):
    h = (fz[:, f] - rest[f]) * 1000
    up, st = h >= E.LIFT_MM, None
    for i in range(1, len(up)):
      if up[i] and not up[i - 1]:
        st = i
      elif not up[i] and up[i - 1] and st is not None:
        peaks.append(h[st:i].max())
        st = None
  gspan = (gait.n - gait.warmup_steps) * DT
  out.update(
    vx=vx, vy=vy, wz=wz,
    err_xy=float(np.hypot(vx - cmd[0], vy - cmd[1])), err_wz=abs(wz - cmd[2]),
    step_len=float(np.mean(gait.dx)) if gait.dx else 0.0,
    swing=float(np.mean(gait.swing)) if gait.swing else 0.0,
    cadence=gait.touchdowns / gspan if gspan > 0 else 0.0,
    flight=gait.flight_steps / max(1, gait.n - gait.warmup_steps),
    lift_med=float(np.median(peaks)) if peaks else 0.0,
    n_lifts=len(peaks),
    torque_mean_ratio=float(ratio[:, leg].mean()),
    torque_worst_joint_ratio=float(ratio[:, leg].mean(axis=0).max()),
    above_cont=float((ratio[:, leg] > 1).mean()),
    leg_power_w=float(power.mean()),
    cot=float(power.mean() / (m.body_mass.sum() * 9.81 * max(np.hypot(vx, vy), 1e-3)))
        if np.hypot(*cmd[:2]) > 0 else float("nan"),
    da=float(np.abs(np.diff(acts, axis=0)).mean()),
    d2a=float(np.abs(np.diff(acts, n=2, axis=0)).mean()),
    tilt_rms=float(np.sqrt((rp ** 2).sum(axis=1).mean())),
  )
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", nargs="+", required=True, help="name=path ...")
  ap.add_argument("--plant", nargs="+", default=["train"], choices=("train", "real"))
  ap.add_argument("--cmds", nargs="+", default=list(COMMANDS))
  ap.add_argument("--seeds", type=int, default=5)
  ap.add_argument("--workers", type=int, default=16)
  ap.add_argument("--out", default=None)
  args = ap.parse_args()
  pols = dict(a.split("=", 1) for a in args.onnx)
  jobs = [(n, p, pl, c, s) for pl in args.plant for c in args.cmds
          for n, p in pols.items() for s in range(args.seeds)]
  with ProcessPoolExecutor(args.workers) as ex:
    rows = list(ex.map(episode, jobs))
  if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1))

  keys = ("vx", "vy", "wz", "err_xy", "err_wz", "step_len", "swing", "cadence",
          "lift_med", "torque_worst_joint_ratio", "above_cont", "leg_power_w",
          "da", "tilt_rms")
  for pl in args.plant:
    print(f"\n=== plant: {pl}  (mean over {args.seeds} seeds; falls = fell/seeds) ===")
    print(f"{'cmd':12s} {'pol':4s} falls " + " ".join(f"{k[:9]:>9s}" for k in keys))
    for c in args.cmds:
      for n in pols:
        rs = [r for r in rows if r["plant"] == pl and r["cmd"] == c and r["policy"] == n]
        ok = [r for r in rs if not r["fell"]]
        vals = " ".join(f"{np.mean([r[k] for r in ok]):9.3f}" if ok else f"{'-':>9s}"
                        for k in keys)
        print(f"{c:12s} {n:4s} {sum(r['fell'] for r in rs)}/{len(rs)}  {vals}")


if __name__ == "__main__":
  main()
