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
  python mjlab_hv1v2/scripts/solve_init_pose.py --table    # v11h height table
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


##
# v11h height table (2026-11-04). Plan: ~/.claude/plans/now-i-have-a-robust-lecun.md
##

TABLE_KNEES = tuple(round(0.35 + 0.05 * i, 2) for i in range(14))   # 0.35 .. 1.00
TABLE_W_KNEE = 5.0   # strong prior: the knee is the height knob, the solver places the rest
G = 9.81


def static_torques(M, q):
  """Signed leg ACTUATOR torque (N.m, LEG_JOINTS order) that holds pose q in
  quasi-static double support: gravity on every body, balanced by a vertical
  ground reaction of m*g/2 at each sole centre site (the CoM is solved over
  the sole centroid). tau = qfrc_bias - J^T F, so a PD target that produces it
  is q + tau / kp. Also returns the floating-base residual |f|, |m| (should be
  ~0 if the pose really is balanced)."""
  M.fk(q, ARM_POSE)
  m, d = M.m, M.d
  mass = float(m.body_subtreemass[1])
  tau = d.qfrc_bias.copy()                     # qvel = 0 -> gravity only
  F = np.array([0.0, 0.0, mass * G / 2])
  for s in ("left", "right"):
    sid = M.sites[s]
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, d.site_xpos[sid], m.site_bodyid[sid])
    tau -= jp.T @ F
  leg = np.array([tau[m.jnt_dofadr[M._j(n)]] for n in LEG_JOINTS])
  return leg, float(np.linalg.norm(tau[0:3])), float(np.linalg.norm(tau[3:6]))


def solve_table(M):
  """Rows (pelvis_z, q[12], tau[12]) for TABLE_KNEES. The knee-0.35 row is
  HOME_KEYFRAME verbatim (pinned: a re-solve with the strong prior lands a few
  mrad away, which would move v11g's pose / stand-action reference at home)."""
  global KNEE_PRIOR, W_KNEE
  lo, hi = M.soft_bounds()
  home_q = np.array([HOME_KEYFRAME.joint_pos[n] for n in LEG_JOINTS])
  rows, ok_all = [], True
  saved = (KNEE_PRIOR, W_KNEE)
  try:
    W_KNEE = TABLE_W_KNEE
    x0 = home_q.copy()
    for k in TABLE_KNEES:
      if abs(k - 0.35) < 1e-9:
        q, z, label = home_q.copy(), float(HOME_KEYFRAME.pos[2]), "knee 0.35 = HOME_KEYFRAME (pinned)"
      else:
        KNEE_PRIOR = k
        xi = x0.copy()
        for s in ("left", "right"):
          xi[LEG_JOINTS.index(f"{s}_knee_joint")] = k
        xi = np.clip(xi, lo + 1e-6, hi - 1e-6)
        sol = least_squares(residuals, xi, args=(M,), bounds=(lo, hi),
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)
        q = sol.x
        M.fk(q, ARM_POSE)
        z = round(-M.lowest_pad_z(), 5)
        label = f"knee {k:.2f}"
      x0 = q.copy()
      ok_all &= report(M, q, z, label)
      tau, rf, rm = static_torques(M, q)
      print(f"  base residual         |f| {rf:.2f} N  |m| {rm:.2f} N.m")
      rows.append((z, q, tau))
  finally:
    KNEE_PRIOR, W_KNEE = saved
  rows.sort(key=lambda r: r[0])                # ascending pelvis_z
  return rows, ok_all


def print_table(rows):
  f = lambda v: ", ".join(f"{x:+.5f}" for x in v)
  print("\n# paste into hv1v2_constants.py (generated by solve_init_pose.py --table)")
  print("V11H_HEIGHT_TABLE = {")
  print(f"  \"joint_names\": {LEG_JOINTS!r},")
  print("  \"pelvis_z\": (" + ", ".join(f"{z:.5f}" for z, _, _ in rows) + "),")
  print("  \"q\": (")
  for z, q, _ in rows:
    print(f"    ({f(q)}),   # z {z:.5f}")
  print("  ),")
  print("  \"tau_static\": (")
  for z, _, t in rows:
    print(f"    ({', '.join(f'{x:+.2f}' for x in t)}),   # z {z:.5f}")
  print("  ),")
  print("}")


##
# v11i 2-D table (2026-11-05): pelvis height x waist pitch.
# Plan: ~/.claude/plans/now-i-have-a-robust-lecun.md, step 1.
##

TABLE2D_NEW_Z = (0.88, 0.87, 0.86)                      # below v11h's 0.88885 row
TABLE2D_WAIST = (-0.09, 0.0, 0.1, 0.2, 0.3, 0.44)       # rad, + = forward lean
W_HEIGHT = 100.0
W_COM_2D = 30.0       # CoM over the sole centre, firmly: the lean is what moves it
W_KNEE_SYM = 1.0      # weak L = R knee prior, keeps the solution unique


def _pose_w(w: float) -> dict:
  return {**ARM_POSE, "waist_pitch_joint": float(w)}


def residuals_hw(q, M, z_t, w):
  """Feet flat/square/level/no stagger, stance width, CoM over the sole centre,
  pelvis at height z_t, with waist pitch fixed at w (pelvis level)."""
  st = M.fk(q, _pose_w(w))
  r = []
  for s in ("left", "right"):
    r += list(W_GEOM * R.from_matrix(st["foot_rot"][s]).as_rotvec())
  pl, pr = st["foot_pos"]["left"], st["foot_pos"]["right"]
  r.append(W_GEOM * (pl[2] - pr[2]))
  r.append(W_GEOM * (pl[1] - STANCE_WIDTH / 2))
  r.append(W_GEOM * (pr[1] + STANCE_WIDTH / 2))
  r.append(W_GEOM * (pl[0] - pr[0]))
  tx = 0.5 * (st["site"]["left"][0] + st["site"]["right"][0])
  r.append(W_COM_2D * (st["com"][0] - tx))
  r.append(W_COM_2D * st["com"][1])
  r.append(W_HEIGHT * (-M.lowest_pad_z() - z_t))
  r.append(W_KNEE_SYM * (q[LEG_JOINTS.index("left_knee_joint")] - q[LEG_JOINTS.index("right_knee_joint")]))
  return np.array(r)


def static_torques_w(M, q, w):
  """static_torques() with waist pitch at w. Returns (leg tau[12], waist tau,
  floating-base residual norm)."""
  M.fk(q, _pose_w(w))
  m, d = M.m, M.d
  mass = float(m.body_subtreemass[1])
  tau = d.qfrc_bias.copy()
  F = np.array([0.0, 0.0, mass * G / 2])
  for s in ("left", "right"):
    sid = M.sites[s]
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, d.site_xpos[sid], m.site_bodyid[sid])
    tau -= jp.T @ F
  leg = np.array([tau[m.jnt_dofadr[M._j(n)]] for n in LEG_JOINTS])
  return leg, float(tau[m.jnt_dofadr[M._j("waist_pitch_joint")]]), float(np.linalg.norm(tau[0:6]))


def cell_check(M, q, z_t, w):
  """(ok, info) for one table cell: flat feet, height, CoM over the soles with
  >= 80 mm margin, soft-limit slack, no self contact."""
  lo, hi = M.soft_bounds()
  st = M.fk(q, _pose_w(w))
  tilt = max(np.degrees(np.linalg.norm(R.from_matrix(st["foot_rot"][s]).as_rotvec()))
             for s in ("left", "right"))
  z = -M.lowest_pad_z()
  M.d.qpos[2] = z
  mujoco.mj_forward(M.m, M.d)
  xs = []
  for g in M.pads:
    half, Rg, p = M.m.geom_size[g], M.d.geom_xmat[g].reshape(3, 3), M.d.geom_xpos[g]
    for sx in (-1, 1):
      for sy in (-1, 1):
        xs.append(float((p + Rg @ (np.array([sx, sy, -1]) * half))[0]))
  com_x = float(M.d.subtree_com[1][0])
  margin = min(max(xs) - com_x, com_x - min(xs))
  slack = float(min(np.min(q - lo), np.min(hi - q)))
  ncon = int(M.d.ncon)
  ok = tilt < 0.5 and abs(z - z_t) < 1e-3 and margin > 0.08 and slack > 0 and ncon == 0
  return ok, dict(tilt=tilt, dz=z - z_t, margin=margin, slack=slack, ncon=ncon)


def _solve_cell(M, x0, z_t, w):
  """Warm-started solve; if the warm start stalls (it can, on the non-smooth
  lowest-pad height), retry from knee guesses and keep the converged solution
  CLOSEST to the warm start (branch continuity)."""
  lo, hi = M.soft_bounds()
  def run(x):
    s = least_squares(residuals_hw, np.clip(x, lo + 1e-6, hi - 1e-6), args=(M, z_t, w),
                      bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return s, np.abs(s.fun).max() < 0.05
  sol, conv = run(x0)
  if conv:
    return sol.x, "warm"
  cands = []
  for kg in (0.35, 0.7, 1.0, 1.2, 1.4):
    x = x0.copy()
    for s in ("left", "right"):
      x[LEG_JOINTS.index(f"{s}_knee_joint")] = kg
    s2, c2 = run(x)
    if c2:
      cands.append(s2.x)
  if not cands:
    raise RuntimeError(f"no converged solution at z {z_t:.4f}, waist {w:+.3f}")
  return min(cands, key=lambda c: np.linalg.norm(c - x0)), "restart"


def solve_table2d(M):
  """V11I_POSE_TABLE. waist = 0 column: v11h's rows VERBATIM (bit-identical),
  extended downward by warm-started solves; every w != 0 cell warm-starts from
  its own height's w = 0 pose."""
  from mjlab_hv1v2.hv1v2_constants import V11H_HEIGHT_TABLE as T
  assert tuple(T["joint_names"]) == LEG_JOINTS, "v11h table joint order differs"
  W = list(TABLE2D_WAIST)
  w0 = W.index(0.0)
  col0 = {}                                   # z -> (q, tau) at w = 0
  for z, q, tau in zip(T["pelvis_z"], T["q"], T["tau_static"]):
    col0[float(z)] = (np.array(q, dtype=float), np.array(tau, dtype=float), "v11h")
  x = np.array(T["q"][0], dtype=float)        # lowest v11h row, warm start downward
  for z in TABLE2D_NEW_Z:
    q, how = _solve_cell(M, x, z, 0.0)
    tau, _, _ = static_torques_w(M, q, 0.0)
    col0[z] = (q, tau, how)
    x = q
  Z = sorted(col0)
  assert all(a < b for a, b in zip(Z, Z[1:]))
  Q = np.zeros((len(Z), len(W), 12)); TAU = np.zeros_like(Q)
  info, ok_all = {}, True
  for i, z in enumerate(Z):
    q0, t0, how0 = col0[z]
    Q[i, w0], TAU[i, w0] = q0, t0
    ok, inf = cell_check(M, q0, z, 0.0)
    info[(i, w0)] = (how0, inf, *static_torques_w(M, q0, 0.0)[1:])
    ok_all &= ok or how0 == "v11h"            # v11h rows were checked when made
    for j, w in enumerate(W):
      if j == w0:
        continue
      q, how = _solve_cell(M, q0, z, w)
      tl, tw, base = static_torques_w(M, q, w)
      Q[i, j], TAU[i, j] = q, tl
      ok, inf = cell_check(M, q, z, w)
      info[(i, j)] = (how, inf, tw, base)
      ok_all &= ok
  return Z, W, Q, TAU, info, ok_all


def print_table2d(Z, W, Q, TAU, info):
  ix = LEG_JOINTS.index
  k, h, a = ix("left_knee_joint"), ix("left_hip_pitch_joint"), ix("left_ankle_pitch_joint")
  kr = ix("right_knee_joint")
  print("\n   z      waist |  knee  hipP  ankP | tau knee L/R    hipP   ankP  waist | ankP slack  CoM margin  base | how")
  for i, z in enumerate(Z):
    for j, w in enumerate(W):
      how, inf, tw, base = info[(i, j)]
      q, t = Q[i, j], TAU[i, j]
      print(f"  {z:.5f} {w:+.2f} | {q[k]:5.2f} {q[h]:+5.2f} {q[a]:+5.2f} | {t[k]:+6.1f}/{t[kr]:+6.1f} "
            f"{t[h]:+6.1f} {t[a]:+6.1f} {tw:+6.1f} | {q[a] - ANKLE_PITCH_SOFT_LO:+.3f}     "
            f"{inf['margin']*1000:5.1f} mm  {base:4.2f} | {how}")
    print()
  f = lambda v: ", ".join(f"{x:+.5f}" for x in v)
  print("# paste into hv1v2_constants.py (generated by solve_init_pose.py --table2d)")
  print("V11I_POSE_TABLE = {")
  print(f"  \"joint_names\": {LEG_JOINTS!r},")
  print("  \"pelvis_z\": (" + ", ".join(f"{z:.5f}" for z in Z) + "),")
  print("  \"waist\": (" + ", ".join(f"{w:+.2f}" for w in W) + "),")
  for key, A, fmt in (("q", Q, f), ("tau_static", TAU, lambda v: ", ".join(f"{x:+.2f}" for x in v))):
    print(f"  \"{key}\": (   # [pelvis_z][waist][joint]")
    for i, z in enumerate(Z):
      print(f"    (   # z {z:.5f}")
      for j, w in enumerate(W):
        print(f"      ({fmt(A[i, j])}),   # waist {w:+.2f}")
      print("    ),")
    print("  ),")
  print("}")


ANKLE_PITCH_SOFT_LO = None   # set in main from the model


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--check", action="store_true",
                  help="verify the pose committed in hv1v2_constants.py")
  ap.add_argument("--table", action="store_true",
                  help="solve the v11h height table (knee 0.35..1.00)")
  ap.add_argument("--table2d", action="store_true",
                  help="solve the v11i height x waist-pitch table")
  args = ap.parse_args()
  M = Model()

  if args.table2d:
    global ANKLE_PITCH_SOFT_LO
    lo, _ = M.soft_bounds()
    ANKLE_PITCH_SOFT_LO = float(lo[LEG_JOINTS.index("left_ankle_pitch_joint")])
    Z, W, Q, TAU, info, ok = solve_table2d(M)
    print_table2d(Z, W, Q, TAU, info)
    # smoothness along waist: largest joint step between neighbouring cells
    dq = np.abs(np.diff(Q, axis=1)).max()
    dz = np.abs(np.diff(Q, axis=0)).max()
    print(f"\n  max |dq| between waist neighbours {dq:.3f} rad, between height neighbours {dz:.3f} rad")
    print(f"=> table2d {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

  if args.table:
    rows, ok = solve_table(M)
    print("\n  pelvis_z   knee(L/R)        hipP(L/R)        ankP(L/R)        "
          "tau knee(L/R)   tau hipP(L/R)   tau ankP(L/R)")
    ix = lambda n: LEG_JOINTS.index(n)
    for z, q, t in rows:
      g = lambda a, j: f"{a[ix('left_' + j)]:+.3f}/{a[ix('right_' + j)]:+.3f}"
      h = lambda a, j: f"{a[ix('left_' + j)]:+6.1f}/{a[ix('right_' + j)]:+6.1f}"
      print(f"  {z:.5f}  {g(q,'knee_joint')}  {g(q,'hip_pitch_joint')}  {g(q,'ankle_pitch_joint')}  "
            f"{h(t,'knee_joint')}  {h(t,'hip_pitch_joint')}  {h(t,'ankle_pitch_joint')}")
    print_table(rows)
    print(f"\n=> table {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

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
