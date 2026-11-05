#!/usr/bin/env python3
"""Solve the HV1V2 standing pose, and print it as an InitialStateCfg.

Why this needs solving at all: the hip-pitch joint axes are canted ~30 degrees
out of pure-Y (right world axis [0, 0.866, 0.5], left [0, 0.866, -0.5]) because
of URDF origin rpy that the hip-roll origin cancels. A hip-pitch rotation theta
therefore carries a 0.5*theta yaw component of OPPOSITE SIGN per leg. The legs
are not mirror images, and a symmetric crouch does not put the feet flat: it
drifts each foot ~24 mm laterally and leaves a 9 mm height mismatch. G1's and
H1-V1's crouch numbers cannot be transplanted.

  python mjlab_hv1v2/scripts/solve_init_pose.py            # solve and print
  python mjlab_hv1v2/scripts/solve_init_pose.py --check    # verify the committed pose
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from mjlab.entity.entity import Entity
from mjlab_hv1v2.hv1v2_constants import (
  FOOT_GEOM_NAMES,
  HOME_KEYFRAME,
  get_hv1v2_robot_cfg,
)

LEG_JOINTS = tuple(
  f"{side}_{j}_joint"
  for side in ("left", "right")
  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)

# Non-leg joints are set analytically, not solved. They only enter through the
# CoM residual, and the arms are ~20 kg so they do move it.
ARM_POSE = {
  ".*_shoulder_pitch_joint": 0.2,
  "left_shoulder_roll_joint": 0.12,
  "right_shoulder_roll_joint": -0.12,
  # At the URDF zero pose the forearms point FORWARD, so the elbow angle
  # rotates them down and then backward. The upper arm already hangs
  # down-and-slightly-back, so pi/2 makes forearm and upper arm nearly
  # COLLINEAR -- 155 deg interior, which reads as a dead-straight arm.
  # 0.9 rad gives ~141 deg interior with the forearm forward of vertical:
  # a natural relaxed bend that leaves room to swing either way.
  ".*_elbow_joint": 0.9,
}

STANCE_WIDTH = 0.24304   # URDF-nominal ankle separation, 2 x 0.12152
KNEE_PRIOR = 0.35        # H1-V1's validated crouch depth
SOFT_LIMIT_FACTOR = 0.9  # must match EntityArticulationInfoCfg

# Residual weights. The 10 geometric residuals are constraints we want met
# exactly; the knee depth and CoM placement are preferences. With 12 unknowns
# and 10 hard constraints only 2 degrees of freedom remain, which the knee
# prior consumes -- so the CoM terms must be weak or they trade against foot
# flatness and leave a millimetre of height mismatch.
W_GEOM = 100.0
W_KNEE = 0.1             # without it, straight legs is an exact solution
W_COM_X = 0.05
W_COM_Y = 0.05


class Model:
  def __init__(self):
    robot = Entity(get_hv1v2_robot_cfg())
    self.m = robot.spec.compile()
    self.d = mujoco.MjData(self.m)
    self.qadr = {n: self.m.jnt_qposadr[self._j(n)] for n in LEG_JOINTS}
    self.feet = {s: self._b(f"{s}_ankle_roll_link") for s in ("left", "right")}
    self.sites = {s: self._s(f"{s}_foot") for s in ("left", "right")}
    self.pads = [self._g(n) for n in FOOT_GEOM_NAMES]

  def _j(self, n):
    return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)

  def _b(self, n):
    return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)

  def _s(self, n):
    return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, n)

  def _g(self, n):
    return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)

  def soft_bounds(self):
    lo, hi = [], []
    for n in LEG_JOINTS:
      a, b = self.m.jnt_range[self._j(n)]
      mid, half = (a + b) / 2, (b - a) / 2
      lo.append(mid - SOFT_LIMIT_FACTOR * half)
      hi.append(mid + SOFT_LIMIT_FACTOR * half)
    return np.array(lo), np.array(hi)

  def fk(self, q_leg, extra=None):
    """Pelvis pinned at the origin, identity orientation."""
    self.d.qpos[:] = 0.0
    self.d.qpos[3] = 1.0
    for n, v in zip(LEG_JOINTS, q_leg):
      self.d.qpos[self.qadr[n]] = v
    for pat, val in (extra or {}).items():
      for i in range(self.m.njnt):
        nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and nm != "floating_base" and __import__("re").match(pat, nm):
          self.d.qpos[self.m.jnt_qposadr[i]] = val
    mujoco.mj_forward(self.m, self.d)

    return dict(
      foot_pos={s: self.d.xpos[b].copy() for s, b in self.feet.items()},
      foot_rot={s: self.d.xmat[b].reshape(3, 3).copy() for s, b in self.feet.items()},
      site={s: self.d.site_xpos[i].copy() for s, i in self.sites.items()},
      com=self.d.subtree_com[1].copy(),
    )

  def lowest_pad_z(self):
    """Lowest world z over the eight pad bottom faces."""
    lo = np.inf
    for g in self.pads:
      half = self.m.geom_size[g]
      Rg = self.d.geom_xmat[g].reshape(3, 3)
      p = self.d.geom_xpos[g]
      for sx in (-1, 1):
        for sy in (-1, 1):
          for sz in (-1, 1):
            c = p + Rg @ (np.array([sx, sy, sz]) * half)
            lo = min(lo, float(c[2]))
    return lo


def residuals(q, M):
  st = M.fk(q, ARM_POSE)
  r = []
  for s in ("left", "right"):
    # feet flat AND square (all three rotation components)
    r += list(W_GEOM * R.from_matrix(st["foot_rot"][s]).as_rotvec())
  pl, pr = st["foot_pos"]["left"], st["foot_pos"]["right"]
  r.append(W_GEOM * (pl[2] - pr[2]))            # feet level
  r.append(W_GEOM * (pl[1] - STANCE_WIDTH / 2)) # stance width, left
  r.append(W_GEOM * (pr[1] + STANCE_WIDTH / 2)) # stance width, right
  r.append(W_GEOM * (pl[0] - pr[0]))            # no fore/aft stagger
  # CoM over the centroid of the contact patches, not over the ankles: the
  # sole runs -0.108..+0.179 about the ankle, so centring on the ankle would
  # leave only ~108 mm of rear margin.
  target_x = 0.5 * (st["site"]["left"][0] + st["site"]["right"][0])
  r.append(W_COM_X * (st["com"][0] - target_x))
  r.append(W_COM_Y * st["com"][1])
  for s in ("left", "right"):
    r.append(W_KNEE * (q[LEG_JOINTS.index(f"{s}_knee_joint")] - KNEE_PRIOR))
  return np.array(r)


def report(M, q, pelvis_z, label):
  st = M.fk(q, ARM_POSE)
  M.d.qpos[2] = pelvis_z
  mujoco.mj_forward(M.m, M.d)
  st = dict(
    foot_pos={s: M.d.xpos[b].copy() for s, b in M.feet.items()},
    foot_rot={s: M.d.xmat[b].reshape(3, 3).copy() for s, b in M.feet.items()},
    com=M.d.subtree_com[1].copy(),
  )
  print(f"\n--- {label} ---")
  print(f"  pelvis z              {pelvis_z:.5f}")
  ok = True
  for s in ("left", "right"):
    tilt = np.degrees(np.linalg.norm(R.from_matrix(st["foot_rot"][s]).as_rotvec()))
    print(f"  {s:5s} foot pos {np.round(st['foot_pos'][s],5)}  tilt {tilt:.4f} deg")
    ok &= tilt < 0.5
  pl, pr = st["foot_pos"]["left"], st["foot_pos"]["right"]
  print(f"  stance width          {pl[1]-pr[1]:.5f}  (target {STANCE_WIDTH})")
  print(f"  fore/aft stagger      {abs(pl[0]-pr[0])*1000:.3f} mm")
  print(f"  foot height mismatch  {abs(pl[2]-pr[2])*1000:.3f} mm")
  lo = M.lowest_pad_z()
  print(f"  lowest pad face z     {lo*1000:+.3f} mm")
  ok &= abs(lo) < 1e-3
  ok &= abs(pl[0] - pr[0]) < 1e-3 and abs(pl[2] - pr[2]) < 1e-3

  # support polygon from the pad corners
  xs, ys = [], []
  for g in M.pads:
    half = M.m.geom_size[g]
    Rg = M.d.geom_xmat[g].reshape(3, 3)
    p = M.d.geom_xpos[g]
    for sx in (-1, 1):
      for sy in (-1, 1):
        c = p + Rg @ (np.array([sx, sy, -1]) * half)
        xs.append(c[0]); ys.append(c[1])
  com = st["com"]
  print(f"  CoM                   {np.round(com,5)}")
  print(f"  support x {min(xs):+.4f}..{max(xs):+.4f}   margin fore {max(xs)-com[0]:.4f} "
        f"aft {com[0]-min(xs):.4f}")
  ok &= min(xs) + 0.08 < com[0] < max(xs) - 0.08
  ok &= min(ys) < com[1] < max(ys)

  # soft limits
  lob, hib = M.soft_bounds()
  slack = min(min(v - a, b - v) for v, a, b in zip(q, lob, hib))
  worst_j = min(zip(LEG_JOINTS, q, lob, hib), key=lambda t: min(t[1] - t[2], t[3] - t[1]))[0]
  print(f"  soft-limit slack      {slack:+.4f} rad (tightest: {worst_j})")
  ok &= slack > 0

  # self collision
  ncon = M.d.ncon
  print(f"  self contacts         {ncon}")
  ok &= ncon == 0
  print(f"  => {'PASS' if ok else 'FAIL'}")
  return ok


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--check", action="store_true",
                  help="verify the pose committed in hv1v2_constants.py")
  args = ap.parse_args()
  M = Model()

  if args.check:
    q = []
    for n in LEG_JOINTS:
      v = HOME_KEYFRAME.joint_pos.get(n)
      assert v is not None, f"{n} is not listed per-joint in HOME_KEYFRAME"
      q.append(v)
    ok = report(M, np.array(q), HOME_KEYFRAME.pos[2], "committed pose")
    return 0 if ok else 1

  lo, hi = M.soft_bounds()
  x0 = np.zeros(len(LEG_JOINTS))
  for s in ("left", "right"):
    x0[LEG_JOINTS.index(f"{s}_knee_joint")] = KNEE_PRIOR
  sol = least_squares(residuals, x0, args=(M,), bounds=(lo, hi),
                      xtol=1e-14, ftol=1e-14, gtol=1e-14)
  print(f"solve: max|residual| = {np.abs(sol.fun).max():.3e}   cost {sol.cost:.3e}")

  M.fk(sol.x, ARM_POSE)
  pelvis_z = round(-M.lowest_pad_z(), 5)
  ok = report(M, sol.x, pelvis_z, "solved pose")

  print("\n# paste into hv1v2_constants.py")
  print(f'  pos=(0.0, 0.0, {pelvis_z}),')
  print("  joint_pos={")
  for n, v in zip(LEG_JOINTS, sol.x):
    print(f'    "{n}": {v:+.5f},')
  # emit from ARM_POSE so the block can never drift from what was solved against
  for pat, val in ARM_POSE.items():
    key = f'r"{pat}$"' if pat.startswith(".*") else f'"{pat}"'
    print(f"    {key}: {val},")
  print("  },")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
