#!/usr/bin/env python3
"""Characterise the HV1V2 plant before any RL is run.

Three parts:

  UNLOADED   pelvis welded to the world. Step response per actuator group with
             only the limb's own inertia. This is the actuator behaving well.

  LOADED     free base, feet on a floor, PD holding the init pose. Each leg
             joint now carries the body above it, which ACTUATORS.txt S5 puts
             at 3x to 1400x the free-swing inertia.

  SENSORS    signs and frames of the three sensors the velocity task reads.

The loaded test is a CHARACTERISATION, not a pass/fail gate. The robot is
EXPECTED to fall: toppling about the ankles needs m*g*z_com ~= 841 N.m/rad and
two ankle-pitch joints at kp 400 supply 800, so the stance is 95% of critical
and open-loop stability is not on offer at these gains. ACTUATORS.txt S7
records the same for H1-V1: "no pose holds the robot up without a policy".
What we check is that the fall is slow and symmetric, the torques are inside
duty, and nothing goes numerically wrong.

  python mjlab_hv1v2/scripts/plant_report.py
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from mjlab.entity.entity import Entity
from mjlab.utils.string import resolve_expr
from mjlab_hv1v2.hv1v2_constants import (
  CONTINUOUS_TORQUE,
  HOME_KEYFRAME,
  get_hv1v2_robot_cfg,
  resolve_joint_groups,
)

DT = 0.002
REPRESENTATIVE = {
  "X12-320": "left_knee_joint",
  "X8-120": "left_shoulder_pitch_joint",
  "X8-120x2": "left_ankle_pitch_joint",
  "X6-60": "left_wrist_roll_joint",
  "X4-36": "left_wrist_pitch_joint",
}
FAILS: list[str] = []


def check(cond, msg):
  if not cond:
    FAILS.append(msg)
  return cond


def _spec():
  return Entity(get_hv1v2_robot_cfg()).spec


def _init_ctrl(m, d):
  """Actuator targets equal to the init pose."""
  ctrl = np.zeros(m.nu)
  for a in range(m.nu):
    j = m.actuator_trnid[a, 0]
    ctrl[a] = d.qpos[m.jnt_qposadr[j]]
  return ctrl


def _apply_init(m, d):
  """Init pose, whether or not the model still has a floating base."""
  k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if k >= 0:
    mujoco.mj_resetDataKeyframe(m, d, k)
  else:
    mujoco.mj_resetData(m, d)
    jnames = tuple(
      n for i in range(m.njnt)
      if (n := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)) and n != "floating_base"
    )
    vals = resolve_expr(HOME_KEYFRAME.joint_pos, jnames, 0.0)
    for n, v in zip(jnames, vals):
      d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
  mujoco.mj_forward(m, d)


# ---------------------------------------------------------------------------
# 1. unloaded
# ---------------------------------------------------------------------------
def unloaded():
  print("\n=== UNLOADED: base pinned, limb inertia only ===")
  # The base is pinned by resetting the free joint every step, NOT by deleting
  # it. Deleting the freejoint welds the pelvis into the world body, and
  # MuJoCo's parent-child contact filter has an explicit exception for the
  # world - so the pelvis would immediately start colliding with its own hip
  # and waist children and the sim explodes. Pinning keeps the filtering right.
  m = _spec().compile()
  m.opt.timestep = DT
  d = mujoco.MjData(m)
  _apply_init(m, d)
  base_q = d.qpos[:7].copy()

  def hold_base():
    d.qpos[:7] = base_q
    d.qvel[:6] = 0.0

  print(f"  {'joint':26s} {'group':10s} {'step':>6s} {'rise10-90':>10s} "
        f"{'overshoot':>10s} {'sse':>9s}")
  for g, jn in REPRESENTATIVE.items():
    _apply_init(m, d)
    hold_base()
    a = next(a for a in range(m.nu)
             if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]) == jn)
    qadr = m.jnt_qposadr[m.actuator_trnid[a, 0]]
    ctrl = _init_ctrl(m, d)
    d.ctrl[:] = ctrl

    # settle first: gravity sags each limb against its PD, and that offset is
    # a property of the pose, not of the step we are about to command.
    for _ in range(int(1.0 / DT)):
      mujoco.mj_step(m, d)
      hold_base()
    q0 = float(d.qpos[qadr])

    step = 0.10
    ctrl[a] += step
    d.ctrl[:] = ctrl
    traj = []
    for _ in range(int(2.0 / DT)):
      mujoco.mj_step(m, d)
      hold_base()
      traj.append(float(d.qpos[qadr]))
    traj = np.array(traj)
    prog = (traj - q0) / step
    above10 = np.flatnonzero(prog > 0.1)
    above90 = np.flatnonzero(prog > 0.9)
    rise = ((above90[0] - above10[0]) * DT
            if len(above10) and len(above90) and above90[0] > above10[0] else float("nan"))
    over = max(0.0, float(prog.max() - 1.0)) * 100
    sse = abs(float(traj[-1] - (q0 + step)))
    print(f"  {jn:26s} {g:10s} {step:6.2f} {rise:9.3f}s {over:9.1f}% {sse:9.4f}")
    check(sse < 0.02, f"{jn}: unloaded steady-state error {sse:.4f} rad is large")
    check(over < 60.0, f"{jn}: unloaded overshoot {over:.0f}% is large")


# ---------------------------------------------------------------------------
# 2. loaded
# ---------------------------------------------------------------------------
def loaded():
  print("\n=== LOADED: free base, feet on floor, PD holding the init pose ===")
  spec = _spec()
  spec.worldbody.add_geom(
    name="_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
    condim=3, friction=[0.8, 0.02, 0.01],
  )
  m = spec.compile()
  m.opt.timestep = DT
  d = mujoco.MjData(m)
  _apply_init(m, d)
  z0 = float(d.qpos[2])
  d.ctrl[:] = _init_ctrl(m, d)
  tgt = d.ctrl.copy()

  jnames = tuple(
    n for i in range(m.njnt)
    if (n := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)) and n != "floating_base"
  )
  groups = resolve_joint_groups(jnames)
  cont = dict(zip(jnames, resolve_expr(CONTINUOUS_TORQUE, jnames, None)))

  # Torque is measured over the STEADY HOLD window only. Once the robot starts
  # toppling every actuator saturates, and reporting that as "torque required
  # to stand" would be nonsense.
  HOLD_S = 10.0
  WINDOW_S = 2.0
  peak_tau = np.zeros(m.nu)
  t_fall = None
  zs = []
  n_steps = int(HOLD_S / DT)
  for i in range(n_steps):
    mujoco.mj_step(m, d)
    if i * DT <= WINDOW_S:
      peak_tau = np.maximum(peak_tau, np.abs(d.actuator_force))
    zs.append(float(d.qpos[2]))
    if t_fall is None and d.qpos[2] < z0 - 0.15:
      t_fall = i * DT

  err = np.array([d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] - tgt[a]
                  for a in range(m.nu)])
  print(f"  pelvis z {z0:.4f} -> {d.qpos[2]:.4f}   xy drift {np.linalg.norm(d.qpos[:2]):.4f} m")
  print(f"  time to fall: {'%.2f s' % t_fall if t_fall else 'did not fall in %.0f s' % HOLD_S}")
  print(f"  max |joint tracking error| {np.abs(err).max():.4f} rad "
        f"({np.degrees(np.abs(err).max()):.2f} deg)")

  print(f"\n  peak torque over the first {WINDOW_S:.0f} s (steady hold, before the topple):")
  print(f"  {'joint':26s} {'group':10s} {'peak Nm':>9s} {'cont Nm':>8s} {'% cont':>7s}")
  rows = []
  for a in range(m.nu):
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
    c = cont[jn]
    rows.append((100 * peak_tau[a] / c, jn, groups[jn], peak_tau[a], c))
  for pct, jn, g, t, c in sorted(rows, reverse=True)[:6]:
    print(f"  {jn:26s} {g:10s} {t:9.2f} {c:8.1f} {pct:6.1f}%")
  worst = max(r[0] for r in rows)
  print(f"\n  worst joint reaches {worst:.1f}% of its CONTINUOUS rating while holding")
  check(worst < 100.0,
        f"a joint exceeds its continuous rating just holding the pose ({worst:.0f}%)")

  # the fall must be slow and roughly symmetric
  zs = np.array(zs)
  ts = np.arange(len(zs)) * DT
  print("  pelvis z trace: " + "  ".join(
    f"{t:.0f}s {zs[int(t/DT)-1]:.4f}" for t in (1, 2, 4, 6, 8, 10) if int(t/DT) <= len(zs)))
  if t_fall:
    check(t_fall > 0.3, f"fall is too fast ({t_fall:.2f} s) - suspect a real bug")
  roll = abs(float(2 * np.arcsin(np.clip(d.qpos[4], -1, 1))))
  print(f"  final base roll {np.degrees(roll):.1f} deg "
        f"(a large roll with small pitch would mean L/R asymmetry)")
  check(not np.any(np.isnan(d.qpos)), "NaN in qpos")
  nwarn = sum(int(w.number) for w in d.warning)
  print(f"  mujoco warnings: {nwarn}")
  check(nwarn == 0, f"{nwarn} mujoco warnings during the hold")


# ---------------------------------------------------------------------------
# 3. sensors
# ---------------------------------------------------------------------------
def sensors():
  print("\n=== SENSORS ===")
  m = _spec().compile()
  d = mujoco.MjData(m)
  adr = {}
  for s in range(m.nsensor):
    adr[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s)] = (
      m.sensor_adr[s], m.sensor_dim[s])
  for need in ("imu_lin_vel", "imu_ang_vel", "root_angmom"):
    check(need in adr, f"missing sensor {need}")
  print(f"  present: {sorted(adr)}")

  site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
  check(site >= 0, "imu_in_pelvis site missing")
  body = m.site_bodyid[site]
  bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
  print(f"  imu_in_pelvis is on body '{bname}'")
  check(bname == "pelvis", f"IMU site is on '{bname}', expected the floating base")

  s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "root_angmom")
  amb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.sensor_objid[s])
  print(f"  root_angmom is on body '{amb}'")
  check(amb == "pelvis", f"root_angmom on '{amb}', expected the floating base")

  # sign/frame: tilt the base and read gravity in the IMU frame
  print(f"\n  {'base tilt':>18s} {'gravity in IMU frame':>34s}")
  for lbl, rpy in (("level", (0, 0, 0)),
                   ("+10 deg roll", (np.deg2rad(10), 0, 0)),
                   ("+10 deg pitch", (0, np.deg2rad(10), 0))):
    _apply_init(m, d)
    from scipy.spatial.transform import Rotation as R
    q = R.from_euler("xyz", rpy).as_quat()
    d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
    mujoco.mj_forward(m, d)
    Rs = d.site_xmat[site].reshape(3, 3)
    g_local = Rs.T @ np.array([0, 0, -1.0])
    print(f"  {lbl:>18s} {np.array2string(g_local, precision=4, suppress_small=True):>34s}")
    if lbl == "level":
      check(abs(g_local[2] + 1) < 1e-6, "level: gravity should be [0,0,-1] in IMU frame")
    if lbl == "+10 deg roll":
      check(g_local[1] < -0.1, "+roll should tip gravity toward -y in the IMU frame")
    if lbl == "+10 deg pitch":
      check(g_local[0] > 0.1, "+pitch should tip gravity toward +x in the IMU frame")
  print("\n  NOTE: these are the SIM conventions. Confirm against the deployed"
        "\n  state estimator's hardware.yaml before trusting a transferred policy.")


def main() -> int:
  unloaded()
  loaded()
  sensors()
  print()
  if FAILS:
    print(f"ISSUES ({len(FAILS)}):")
    for f in FAILS:
      print("  -", f)
    return 1
  print("PLANT REPORT OK (the loaded fall is expected, see the module docstring)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
