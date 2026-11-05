#!/usr/bin/env python3
"""Where does a policy SETTLE after a mild push, and how does it get there?

Built on eval_plant.Sim (plant, deployed command timing). Each episode starts
at the policy's own anchor, stands 3 s, takes a pelvis-velocity push, then
stands 8 s with a zero command. Recorded:

  stance at rest  ankle roll / hip roll / ankle pitch minus anchor (deg),
                  sole roll against the floor, foot spacing and relative yaw
                  in the pelvis heading frame, pelvis roll / pitch
  loading         per-foot inner/outer and heel/toe share of the normal force
                  (pads 1/3 sit at -y of the ankle frame, 2/4 at +y; 1/2 heel)
                  and the lateral ground force on each foot (opposite signs =
                  the legs squeezing or spreading against each other)
  recovery        steps, peak sole roll (edge balancing), peak hip-roll swing,
                  and whether the final stance was still drifting at 4 s

  python mjlab_hv1v2/scripts/probe_stance.py --onnx exported/deploy_v7/policy.onnx \
      --plant real --ankle-motor 180 180 10.8 10.8 --out /tmp/v7_stance.json
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

DT = E.DT
SIDES = ("left", "right")


def setup(onnx, plant, ankle_motor, waist_factor, seed):
  if ankle_motor:
    E.ANKLE_MOTOR_GAINS = tuple(ankle_motor)
  if waist_factor:
    E.REAL_GAIN_FACTOR["waist_roll"] = (waist_factor[0],) * 2
    E.REAL_GAIN_FACTOR["waist_pitch"] = (waist_factor[1],) * 2
  sim = E.Sim(onnx, plant, False, seed)
  rob, pol = sim.rob, sim.pol
  rng = np.random.default_rng(seed)
  rob.reset()
  rob.d.qpos[rob.qadr] = pol.default_qpos + rng.uniform(-0.01, 0.01, len(rob.qadr))
  mujoco.mj_forward(rob.m, rob.d)
  pol.reset()
  return sim


class Probe:
  def __init__(self, sim):
    self.sim = sim
    m, pol = sim.rob.m, sim.pol
    jn = pol.joint_names
    self.ix = {f"{s}_{j}": jn.index(f"{s}_{j}_joint")
               for s in SIDES for j in ("ankle_roll", "ankle_pitch", "hip_roll")}
    self.foot = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
                 for s in SIDES]
    self.pads = [[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot{i}_collision")
                  for i in (1, 2, 3, 4)] for s in SIDES]
    self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    self.pelvis = sim.rob.pelvis

  def sole_roll(self, f):
    R = self.sim.rob.d.xmat[self.foot[f]].reshape(3, 3)
    return float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))

  def pad_forces(self):
    """[foot][pad] normal force (N) and per-foot world lateral force."""
    m, d = self.sim.rob.m, self.sim.rob.d
    fn = np.zeros((2, 4))
    fw = np.zeros((2, 3))
    buf = np.zeros(6)
    for k in range(d.ncon):
      c = d.contact[k]
      g = c.geom2 if c.geom1 == self.floor else c.geom1 if c.geom2 == self.floor else None
      if g is None:
        continue
      for f in (0, 1):
        if g in self.pads[f]:
          mujoco.mj_contactForce(m, d, k, buf)
          fn[f, self.pads[f].index(g)] += buf[0]
          # contact frame -> world; sign so the force is ON the foot
          w = c.frame.reshape(3, 3).T @ buf[:3]
          fw[f] += w if c.geom2 == g else -w
    return fn, fw

  def stance(self):
    rob, pol = self.sim.rob, self.sim.pol
    q = rob.qpos_joints() - pol.default_qpos
    d = rob.d
    yaw = np.arctan2(d.xmat[self.pelvis][3], d.xmat[self.pelvis][0])
    c, s = np.cos(yaw), np.sin(yaw)
    p = [d.xpos[b][:2] for b in self.foot]
    rel = p[0] - p[1]
    fy = [np.arctan2(d.xmat[b][3], d.xmat[b][0]) for b in self.foot]
    fn, fw = self.pad_forces()
    out = {k: float(np.degrees(q[i])) for k, i in self.ix.items()}
    out.update(
      left_sole_roll=self.sole_roll(0), right_sole_roll=self.sole_roll(1),
      width=float(-s * rel[0] + c * rel[1]), fore_aft=float(c * rel[0] + s * rel[1]),
      foot_yaw_diff=float(np.degrees(np.angle(np.exp(1j * (fy[0] - fy[1]))))),
      pelvis_roll=float(self.sim.rp[-1][0]), pelvis_pitch=float(self.sim.rp[-1][1]),
      fy_left=float(fw[0, 1]), fy_right=float(fw[1, 1]),
    )
    # Inner side: left foot's -y pads (1, 3), right foot's +y pads (2, 4).
    for f, side, inner in ((0, "left", (0, 2)), (1, "right", (1, 3))):
      tot = fn[f].sum()
      out[f"{side}_load"] = float(tot)
      out[f"{side}_inner_share"] = float(fn[f, list(inner)].sum() / tot) if tot > 1 else float("nan")
      out[f"{side}_heel_share"] = float(fn[f, :2].sum() / tot) if tot > 1 else float("nan")
      out[f"{side}_pads_loaded"] = int((fn[f] > 0.02 * max(tot, 1)).sum())
    tau = rob.d.qfrc_actuator[rob.vadr] + rob.d.qfrc_applied[rob.vadr]
    out["tau_left_ankle_roll"] = float(tau[self.ix["left_ankle_roll"]])
    out["tau_right_ankle_roll"] = float(tau[self.ix["right_ankle_roll"]])
    return out


def avg(dicts):
  return {k: float(np.nanmean([x[k] for x in dicts])) for k in dicts[0]}


def episode(job):
  onnx, plant, ankle_motor, waist_factor, v, deg, seed = job
  sim = setup(onnx, plant, ankle_motor, waist_factor, seed)
  pr = Probe(sim)
  pre = []
  for k in range(int(3.0 / DT)):
    sim.step((0, 0, 0))
    if k >= int(2.5 / DT):
      pre.append(pr.stance())
  pre = avg(pre)
  rest = np.median(np.array(sim.fz)[-25:], axis=0)
  if v > 0:
    sim.push(v, deg)
  peak_sole, peak_hip, mid, late = 0.0, 0.0, [], []
  n = int(8.0 / DT)
  for k in range(n):
    sim.step((0, 0, 0))
    st = pr.stance()
    peak_sole = max(peak_sole, abs(st["left_sole_roll"] - pre["left_sole_roll"]),
                    abs(st["right_sole_roll"] - pre["right_sole_roll"]))
    peak_hip = max(peak_hip, abs(st["left_hip_roll"] - pre["left_hip_roll"]),
                   abs(st["right_hip_roll"] - pre["right_hip_roll"]))
    if int(3.5 / DT) <= k < int(4.0 / DT):
      mid.append(st)
    if k >= n - int(1.0 / DT):
      late.append(st)
  t, _, _, fz, _ = sim.arrays()
  after = t >= 3.0 + DT
  post, midv = avg(late), avg(mid)
  return dict(
    plant=plant, v=v, deg=deg, seed=seed, fell=sim.fell is not None,
    steps=E.foot_lifts(fz, rest, after), pre=pre, post=post,
    peak_sole_roll=peak_sole, peak_hip_roll=peak_hip,
    drift_4_to_8s=max(abs(post[k] - midv[k]) for k in
                      ("left_ankle_roll", "right_ankle_roll", "left_hip_roll", "right_hip_roll")),
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--plant", choices=("train", "real"), default="real")
  ap.add_argument("--ankle-motor", type=float, nargs=4, default=None)
  ap.add_argument("--waist-factor", type=float, nargs=2, default=None,
                  help="real plant: waist roll / pitch joint-gain factor")
  ap.add_argument("--speeds", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
  ap.add_argument("--seeds", type=int, default=3)
  ap.add_argument("--workers", type=int, default=16)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  jobs = []
  for v in args.speeds:
    dirs = [0] if v == 0 else range(0, 360, 45)
    for deg in dirs:
      for s in range(args.seeds):
        jobs.append((args.onnx, args.plant, args.ankle_motor, args.waist_factor, v, deg, s))
  with ProcessPoolExecutor(args.workers) as ex:
    rows = list(ex.map(episode, jobs))
  Path(args.out).write_text(json.dumps(rows, indent=1))
  print(f"wrote {len(rows)} episodes to {args.out}")


if __name__ == "__main__":
  main()
