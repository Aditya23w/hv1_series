#!/usr/bin/env python3
"""Solve the HV1V1 standing pose, and print it as an InitialStateCfg.

Why this needs solving at all: the hip-pitch joint axes are canted ~30 degrees
out of pure-Y (right world axis [0, 0.866, 0.5], left [0, 0.866, -0.5]) because
of URDF origin rpy that the hip-roll origin cancels. A hip-pitch rotation theta
therefore carries a 0.5*theta yaw component of OPPOSITE SIGN per leg. The legs
are not mirror images, and a symmetric crouch does not put the feet flat: it
drifts each foot ~24 mm laterally and leaves a 9 mm height mismatch. G1's and
H1-V1's crouch numbers cannot be transplanted.

  python mjlab_hv1v1/scripts/solve_init_pose.py            # solve and print
  python mjlab_hv1v1/scripts/solve_init_pose.py --check    # verify the committed pose
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from mjlab.entity.entity import Entity
from mjlab_hv1v1.hv1v1_constants import (
  FOOT_GEOM_NAMES,
  HOME_KEYFRAME,
  get_hv1v1_robot_cfg,
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

# Where the CoM goes, expressed as a FRACTION ALONG THE SOLE rather than as an
# offset from the ankle -- because the ankle is at a different place in each
# robot's foot, and it is the foot-relative position that a policy's foot
# placement is expressed against.
#
#   HV1V2: ankle at 36.5% of the sole, CoM at 50.0%  -> +35.6 mm, 37.9% measured duty
#   HV1V1: ankle at 22.0% of the sole
#
# TWO EARLIER ARGUMENTS FOR A MORE FORWARD CoM WERE WRONG. Both are recorded
# here because they are easy to re-derive and both are wrong for subtle reasons.
#
# 1. "Static ankle torque = m*g*x, so a forward CoM costs ankle capacity."
#    FALSE for a robot standing on flat feet. Measured with mj_inverse at the
#    init pose, the required ankle_pitch torque is -1.59 N.m on HV1V1 and
#    -0.52 N.m on HV1V2 -- about 60x BELOW what that formula predicts. The
#    ground reaction under the sole supplies the moment by shifting its CoP
#    anywhere inside the support polygon; the ankle does not have to. The
#    formula only becomes relevant once the CoP is pinned at a sole edge.
#
# 2. "34.5% is 2 mm from the configuration that needed the 2.5 deg offset on
#    hardware, so it is disqualified." FALSE -- this conflates a CoM POSITION
#    with a MISMATCH between two positions. The offset was needed because a
#    policy trained at 50% of ITS sole was run on a robot sitting at 34.6% of
#    THIS sole. A policy trained from scratch at 34.4% has no mismatch to
#    correct. The position was never the problem; the disagreement was.
#
# What actually discriminates is the OPEN-LOOP CREEP RATE: every anchor pose is
# a PD-held inverted pendulum and none is passively stable, so what matters is
# how fast it diverges relative to the pose that is known to train. Measured
# through one code path (creep fitted over 0.5-2.5 s, torque taken only while
# |dz| < 15 mm, so the fall is excluded):
#
#   HV1V2 committed, 50.0%   54.6 N.m (64% cont)   -0.83 mm/s   falls  3.91 s
#   HV1V1  34.4% (+34.7 mm)  31.1 N.m (36% cont)   -0.21 mm/s   falls  6.24 s
#   HV1V1  37.8% (+44.5 mm)  52.2 N.m (61% cont)   -9.47 mm/s   falls  1.91 s
#   HV1V1  41.3% (+54.4 mm)  49.2 N.m (57% cont)  -19.85 mm/s   falls  1.35 s
#   HV1V1  44.8% (+64.2 mm)  46.0 N.m (53% cont)  -31.12 mm/s   falls  1.10 s
#   HV1V1  49.8% (+78.3 mm)  43.9 N.m (51% cont)  -47.32 mm/s   falls  0.89 s
#
# 34.4% is the ONLY HV1V1 option at or better than HV1V2's proven creep, and it
# is better on all three: 4x slower creep, 1.6x longer hold, and 36% of
# continuous against HV1V2's 64%. It also has the LARGEST front margin
# (185.1 mm vs 141.5 at the sole centre), which is the right way round -- v4
# survived 3x backward and lateral pushes and fell only forward. Its cost is the
# smallest rear margin, 96.7 mm, which is still ample.
#
# Creep grows monotonically with the offset and HV1V2's own value is the
# reference: require |creep| <= 1.0 mm/s. Do NOT move the CoM forward of 34.4%
# without a measurement that beats this table.
COM_SOLE_FRACTION = 0.345

# ---------------------------------------------------------------------------
# ANKLE PITCH BIAS -- the thing that actually makes this robot walk.
#
# The 12 leg joints below are solved with the feet FLAT. The bias is then ADDED
# to the two ankle-pitch values afterwards, so the committed anchor sits 2.7 deg
# PLANTARFLEXED (toes-down) from feet-flat. That is deliberate and it must not be
# "fixed" by re-solving.
#
# WHY. On hardware this robot does not balance without it: it loads its heels and
# drives backwards. An HV1V2 policy plus a 2.7 deg ankle offset walks flawlessly
# on it; HV1V1's own from-scratch policy without the offset does not balance at
# all. So the offset is a property of the ROBOT, not -- as this file previously
# claimed -- an artifact of running an HV1V2-trained policy on an HV1V1 foot.
#
# MECHANISM. It is a TORQUE bias, not a pose change. With the foot planted flat,
# 2.7 deg of error between the PD target and the actual joint angle holds a
# constant plantarflexion torque, which drives the centre of pressure toward the
# toe -- the "weight on the toes" seen on hardware -- and accelerates the body
# forward, cancelling the backward drive.
#
# SIGN. Positive is toes-down here. Verified by FK: at +2.7 deg the heel pad sits
# at 17.1 mm and the toe pad at 8.5 mm, i.e. the heel lifts. Note that with the
# foot FREE to tilt this moves the CoM backwards (34.4% -> 19.4% of the sole),
# which is the opposite of the loaded case and is why a static CoM analysis gets
# this backwards. Do not re-derive the sign from CoM geometry.
#
# Training with the bias in the anchor beats applying it on the robot: HV1V1's
# policy plus a HARDWARE offset balanced but walked poorly, because its actions
# were learned for an unbiased operating point and then shifted underneath it.
#
# DEPLOY: the robot's own offset must be ZERO, or it doubles to 5.4 deg.
ANKLE_PITCH_BIAS_DEG = 2.7
ANKLE_PITCH_BIAS = np.radians(ANKLE_PITCH_BIAS_DEG)


def apply_ankle_bias(q_leg):
  """Add the plantarflexion bias to the two ankle-pitch entries of a leg vector."""
  out = np.asarray(q_leg, dtype=float).copy()
  for side in ("left", "right"):
    out[LEG_JOINTS.index(f"{side}_ankle_pitch_joint")] += ANKLE_PITCH_BIAS
  return out


def remove_ankle_bias(q_leg):
  """Inverse of apply_ankle_bias, for checking a committed pose's geometry."""
  out = np.asarray(q_leg, dtype=float).copy()
  for side in ("left", "right"):
    out[LEG_JOINTS.index(f"{side}_ankle_pitch_joint")] -= ANKLE_PITCH_BIAS
  return out
COM_FRACTION_TOL = 0.005      # achieved fraction must land within +-0.5%
# OBSERVATION, NOT A GATE. Open-loop creep is how fast the pose sags when a PD
# holds it with no policy. It does NOT predict hardware success -- HV1V2's own
# proven anchor manages only 3.9 s before toppling, because the POLICY does the
# stabilising, not the pose. Using it as a gate is what moved this pose from 50%
# of the sole to 34.4%, and the hardware then needed 2.7 deg of ankle bias to get
# the load back onto the toes. Report it; do not decide on it.
MAX_CREEP_MM_S = 1.0          # HV1V2 measures -0.83
ANKLE_DUTY_TARGET = 0.188   # HV1V2's static value; diagnostic reference only
ANKLE_CONTINUOUS_NM = 172.0   # 2 ankles x 2 x X8-120 rated 43 N.m
MIN_REAR_MARGIN = 0.090       # m, CoM ahead of the heel edge
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


def com_x_for_ankle_duty(mass_kg: float, duty: float = ANKLE_DUTY_TARGET) -> float:
  """CoM offset from the ankle, in the foot frame, giving `duty` ankle load.

  tau = m * g * x  over both ankles, against ANKLE_CONTINUOUS_NM.
  """
  return duty * ANKLE_CONTINUOUS_NM / (mass_kg * 9.80665)


class Model:
  def __init__(self):
    robot = Entity(get_hv1v1_robot_cfg())
    self.m = robot.spec.compile()
    self.d = mujoco.MjData(self.m)
    self.qadr = {n: self.m.jnt_qposadr[self._j(n)] for n in LEG_JOINTS}
    self.feet = {s: self._b(f"{s}_ankle_roll_link") for s in ("left", "right")}
    self.sites = {s: self._s(f"{s}_foot") for s in ("left", "right")}
    self.pads = [self._g(n) for n in FOOT_GEOM_NAMES]
    self.total_mass = float(sum(self.m.body_mass))
    # Sole extent in the FOOT body frame, from the pad geoms (rigidly attached,
    # so this is a constant of the model).
    xs = []
    for g in self.pads:
      if self.m.geom_bodyid[g] != self.feet["left"]:
        continue
      half = self.m.geom_size[g]
      for sx in (-1, 1):
        xs.append(float(self.m.geom_pos[g][0] + sx * half[0]))
    self.sole_x0, self.sole_x1 = min(xs), max(xs)
    self.sole_centre_x = 0.5 * (self.sole_x0 + self.sole_x1)
    self.sole_len = self.sole_x1 - self.sole_x0
    # The target, and the inverse map used to report what was achieved.
    self.com_target_x = self.sole_x0 + COM_SOLE_FRACTION * self.sole_len

  def sole_fraction(self, x_foot: float) -> float:
    """Where an x in the foot frame sits along the sole, 0 = heel, 1 = toe."""
    return (x_foot - self.sole_x0) / self.sole_len

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
  # CoM at COM_SOLE_FRACTION along the sole, in the FOOT frame (the frame fixed
  # to the ground). NOT the sole centre -- see the note at the top of the file.
  target_x = M.com_target_x
  Rf = st["foot_rot"]["left"]
  tf = st["foot_pos"]["left"]
  com_foot_x = float((Rf.T @ (st["com"] - tf))[0])
  r.append(W_COM_X * (com_foot_x - target_x))
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
  Rf = st["foot_rot"]["left"]; tf = st["foot_pos"]["left"]
  com_foot_x = float((Rf.T @ (com - tf))[0])
  duty = M.total_mass * 9.80665 * com_foot_x / ANKLE_CONTINUOUS_NM
  tgt = M.com_target_x
  frac = M.sole_fraction(com_foot_x)
  print(f"  CoM                   {np.round(com,5)}")
  print(f"  CoM in foot frame     {com_foot_x*1000:+.2f} mm  (target {tgt*1000:+.2f})")
  # The gate is on the ACHIEVED fraction, not on having set the constant.
  print(f"  CoM along sole        {100*frac:.2f}%  "
        f"(target {100*COM_SOLE_FRACTION:.1f} +-{100*COM_FRACTION_TOL:.1f}%, "
        f"ankle at {100*M.sole_fraction(0.0):.1f}%, HV1V2 50.0%)")
  # Reported only. This formula is NOT the ankle's requirement -- measured with
  # mj_inverse the real static need is ~1.6 N.m, ~60x less. See the note above.
  print(f"  m*g*x formula         {100*duty:.1f}% of continuous  "
        f"(NOT the real requirement; mj_inverse says ~1.6 N.m. The gate is the "
        f"open-loop creep rate, |creep| <= {MAX_CREEP_MM_S} mm/s -- see hold test)")
  ok &= abs(frac - COM_SOLE_FRACTION) <= COM_FRACTION_TOL
  ok &= com_foot_x >= 0
  print(f"  support x {min(xs):+.4f}..{max(xs):+.4f}   margin fore {max(xs)-com[0]:.4f} "
        f"aft {com[0]-min(xs):.4f}")
  ok &= (com[0] - min(xs)) >= MIN_REAR_MARGIN
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
                  help="verify the pose committed in hv1v1_constants.py")
  ap.add_argument("--com-fraction", type=float, default=None,
                  help="override COM_SOLE_FRACTION for a sweep (does not commit)")
  args = ap.parse_args()
  if args.com_fraction is not None:
    global COM_SOLE_FRACTION
    COM_SOLE_FRACTION = args.com_fraction
  M = Model()

  if args.check:
    q = []
    for n in LEG_JOINTS:
      v = HOME_KEYFRAME.joint_pos.get(n)
      assert v is not None, f"{n} is not listed per-joint in HOME_KEYFRAME"
      q.append(v)
    q = np.array(q)
    # The committed pose carries the ankle bias; the geometric constraints
    # (feet flat, CoM fraction) are only meaningful on the unbiased solve.
    q_flat = remove_ankle_bias(q)
    M.fk(q_flat, ARM_POSE)
    pelvis_flat = round(-M.lowest_pad_z(), 5)
    geom_ok = report(M, q_flat, pelvis_flat, "committed pose, ankle bias removed")
    # And confirm the bias really is toes-down on the committed pose.
    M.fk(q, ARM_POSE)
    pz = round(-M.lowest_pad_z(), 5)
    M.d.qpos[2] = pz
    mujoco.mj_forward(M.m, M.d)
    toe = min(float(M.d.geom_xpos[g][2]) for g in M.pads
              if M.m.geom_bodyid[g] == M.feet["left"] and M.m.geom_pos[g][0] > 0.05)
    heel = min(float(M.d.geom_xpos[g][2]) for g in M.pads
               if M.m.geom_bodyid[g] == M.feet["left"] and M.m.geom_pos[g][0] < 0.05)
    bias_ok = heel > toe
    print(f"\n--- ankle bias {ANKLE_PITCH_BIAS_DEG:+.2f} deg ---")
    print(f"  committed pelvis z    {HOME_KEYFRAME.pos[2]:.5f}   (bias-applied solve: {pz:.5f})")
    print(f"  left toe pad z        {1000*toe:+.1f} mm")
    print(f"  left heel pad z       {1000*heel:+.1f} mm")
    print(f"  heel above toe        {'YES -- toes-down, correct' if bias_ok else 'NO -- SIGN IS WRONG'}")
    ok = geom_ok and bias_ok and abs(HOME_KEYFRAME.pos[2] - pz) < 1e-4
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

  lo, hi = M.soft_bounds()
  x0 = np.zeros(len(LEG_JOINTS))
  for s in ("left", "right"):
    x0[LEG_JOINTS.index(f"{s}_knee_joint")] = KNEE_PRIOR
  sol = least_squares(residuals, x0, args=(M,), bounds=(lo, hi),
                      xtol=1e-14, ftol=1e-14, gtol=1e-14)
  print(f"solve: max|residual| = {np.abs(sol.fun).max():.3e}   cost {sol.cost:.3e}")

  M.fk(sol.x, ARM_POSE)
  pelvis_z_flat = round(-M.lowest_pad_z(), 5)
  ok = report(M, sol.x, pelvis_z_flat, "solved pose (feet flat, before ankle bias)")

  # Now bias the ankles and re-seat the robot on the floor: with the heel lifted
  # the lowest contact point moves to the toe, so the pelvis rises.
  q_biased = apply_ankle_bias(sol.x)
  M.fk(q_biased, ARM_POSE)
  pelvis_z = round(-M.lowest_pad_z(), 5)
  print(f"\nankle bias {ANKLE_PITCH_BIAS_DEG:+.2f} deg applied: "
        f"pelvis {pelvis_z_flat:.5f} -> {pelvis_z:.5f}")

  print("\n# paste into hv1v1_constants.py")
  print(f'  pos=(0.0, 0.0, {pelvis_z}),')
  print("  joint_pos={")
  for n, v in zip(LEG_JOINTS, q_biased):
    print(f'    "{n}": {v:+.5f},')
  # emit from ARM_POSE so the block can never drift from what was solved against
  for pat, val in ARM_POSE.items():
    key = f'r"{pat}$"' if pat.startswith(".*") else f'"{pat}"'
    print(f"    {key}: {val},")
  print("  },")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
