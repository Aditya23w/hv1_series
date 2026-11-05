#!/usr/bin/env python3
"""Stop / push / step-clearance evaluation, on the training plant or on a plant
that matches the REAL HV1V2 (tahiti-d1-002).

Why a second plant. Fitting measured joint torque against joint error on the
robot (session 2026-10-10T14-32-23, R^2 1.00, hips/knees x1.00 as a check):

  * the four-bar joints apply more joint-space gain than commanded, because
    merai passes kp/kd to the two motors unchanged and the linkage amplifies:
      ankle pitch kp x2.25 kd x2.30   ankle roll kp x1.60 kd x1.65
      waist roll  kp x4.36 kd x4.34   waist pitch kp x4.90 kd x4.78
  * training caps the ankle JOINTS at +-43 N.m and hips/knees at +-85 N.m (MJCF
    actuatorfrcrange); the real drives deliver ~235/199 N.m at the ankle
    (four-bar envelope |tp|/1.956 + |tr|/1.66 <= 120 per motor) and ~300 N.m
    at the X12 joints. On the robot, ankle pitch is above 43 N.m 24-29 % of
    standing time.

`--plant real` applies both. The ANKLES are modelled in motor space (see
ANKLE_J below): each motor runs PD on its own angle with its joint slot's gain,
so the joint sees TS * J^T diag(k_a, k_b) J, cross-coupling included and
mirrored left/right. That model matches the robot at both gain sets tried
(400/400 and 178/400) to ~5-10 %; the per-joint factors above are only used for
the waist. `--ankle-motor KA KB DA DB` sets the two motor gains directly;
`--ankle-div` is shorthand for the deployed 2026-10-10 runtime fix, ankle pitch
slot 178 / 10.5 (motor a), roll slot 400 / 24 (motor b).

Command timing is the deployed one: every 50 Hz target interpolated linearly
over the 4 physics substeps (merai_control's 1 kHz interpolation).

Suites (metrics are computed the way scripts/stand-analysis.py computes them on
robot sessions, so sim and robot numbers are comparable):
  stop       walk 4 s at a command, step it to zero, stand 8 s
             -> sway pk-pk / frequency over 2-8 s, settle time, foot lifts
  push       stand 3 s, set the pelvis velocity (0.2/0.35/0.5 m/s, 8 dirs)
             -> fell, recovery steps, peak tilt, recovery time
  clearance  walk 8 s at vx 0.3 / 0.5 and vy 0.3
             -> per-swing peak foot height (median, p10), per foot
  torque     stand 6 s + walk 6 s: ankle joint |tau| p99, share above 43 N.m

  conda run -n hv1v2_mjlab python mjlab_hv1v2/scripts/eval_plant.py \
      --onnx exported/deploy_v8/policy.onnx --plant real --suite all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim2sim as S  # noqa: E402

DT = S.SIM_DT * S.DECIMATION

# Measured on tahiti-d1-002, 2026-10-10 (see module docstring): (kp, kd) factors.
REAL_GAIN_FACTOR = {
  "ankle_pitch": (2.25, 2.30),
  "ankle_roll": (1.60, 1.65),
  "waist_roll": (4.36, 4.34),
  "waist_pitch": (4.90, 4.78),
}
# Joint torque capacity of the real drives (N.m).
REAL_TORQUE_CAP = {"ankle_pitch": 235.0, "ankle_roll": 199.0,
                   "waist_roll": 199.0, "waist_pitch": 235.0,
                   "hip": 300.0, "knee": 300.0}
ANKLE_DIV = {"ankle_pitch": (178.0, 10.5)}   # commanded (kp, kd) for --ankle-div

LIFT_MM = 3.0          # a foot lift: site >= 3 mm above its standing height
SETTLE_RATE = 0.06     # rad/s, pelvis roll/pitch rate, held 1 s


def _kind(name: str):
  for k in ("ankle_pitch", "ankle_roll", "waist_roll", "waist_pitch", "hip", "knee"):
    if k in name:
      return k
  return None


class BiasedRobot(S.Robot):
  """The policy's IMU reads the pelvis rotated by a fixed (roll, pitch) offset,
  as a mis-mounted or mis-calibrated IMU would."""

  def __init__(self, joint_names, bias_deg):
    super().__init__(joint_names, for_video=False)
    r, p = np.radians(bias_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    self._bias = Rx @ Ry

  def root_rot(self):
    return super().root_rot() @ self._bias


# Four-bar ankle, motor-space model. merai gives motor a the PITCH slot's kp/kd and
# motor b the ROLL slot's, and the drive runs PD on its own motor angle, so the
# joint sees K = TS * J^T diag(k_a, k_b) J (cross term included). J at the
# standing pose from config/params/tahiti_d1/ankle.yaml reference samples; TS is
# the X8-120 torque scale measured 2026-09-29 (passthrough X8s read x1.13).
# Checked against the robot: (400,400) -> 889/598 vs measured 860-930/610-670;
# (178,400) -> 568/382 vs measured 570-630/380-400 (N.m/rad, pitch/roll).
ANKLE_J = np.array([[0.994, 0.813], [-0.993, 0.809]])
ANKLE_TS = 1.13
ANKLE_MOTOR_TAU = 129.0     # X8-120 max_torque_nm (merai config_types.cpp)
ANKLE_MOTOR_GAINS = None    # (ka, kb, da, db) motor kp/kd; None = joint slot values
MIRROR_LEFT = True          # --no-mirror: same cross-coupling sign on both ankles

IMU_BIAS_DEG = (0.0, 0.0)   # set from --imu-bias
COM_SHIFT_MM = (0.0, 0.0)   # set from --com-shift: torso CoM offset, x / y


def make(onnx: str, plant: str, ankle_div: bool, payload: float = 0.0):
  pol = S.Policy(onnx)
  rob = (BiasedRobot(pol.joint_names, IMU_BIAS_DEG) if any(IMU_BIAS_DEG)
         else S.Robot(pol.joint_names, for_video=False, v11=pol.v11))
  m = rob.m
  if payload:
    S.add_hand_payload(rob, payload)
  if any(COM_SHIFT_MM):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert b >= 0, "torso_link missing"
    m.body_ipos[b, :2] += np.array(COM_SHIFT_MM) / 1000.0
  rob.ankle_pd = None
  for a in range(m.nu):
    j = m.actuator_trnid[a, 0]
    k = _kind(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
    kp, kd = m.actuator_gainprm[a, 0], -m.actuator_biasprm[a, 2]
    if ankle_div and k in ANKLE_DIV:
      kp, kd = ANKLE_DIV[k]
    if plant == "real" and k in ("ankle_pitch", "ankle_roll"):
      # Driven by the motor-space PD in Sim.step instead; the actuator goes slack.
      m.actuator_gainprm[a, 0] = 0.0
      m.actuator_biasprm[a, 1:3] = 0.0
      m.jnt_actfrclimited[j] = 0
      continue
    if plant == "real":
      fkp, fkd = REAL_GAIN_FACTOR.get(k, (1.0, 1.0))
      kp, kd = kp * fkp, kd * fkd
      if k in REAL_TORQUE_CAP:
        cap = REAL_TORQUE_CAP[k]
        m.jnt_actfrcrange[j] = (-cap, cap)
        lo, hi = m.actuator_forcerange[a]
        m.actuator_forcerange[a] = (min(lo, -cap), max(hi, cap))
    m.actuator_gainprm[a, 0] = kp
    m.actuator_biasprm[a, 1] = -kp
    m.actuator_biasprm[a, 2] = -kd
  if plant == "real":
    if ANKLE_MOTOR_GAINS is not None:
      ka, kb, da, db = ANKLE_MOTOR_GAINS
    else:   # what merai does: each motor takes its joint slot's gain
      kp_p, kd_p = ANKLE_DIV["ankle_pitch"] if ankle_div else (400.0, 24.0)
      ka, kb, da, db = kp_p, 400.0, kd_p, 24.0
    sides = []
    for s in ("right", "left"):
      ip = pol.joint_names.index(f"{s}_ankle_pitch_joint")
      ir = pol.joint_names.index(f"{s}_ankle_roll_joint")
      # The left ankle is the mirror image: roll flips sign, so with unequal
      # motor gains the pitch-roll cross stiffness is opposite on the two sides.
      J = ANKLE_J if s == "right" or not MIRROR_LEFT else ANKLE_J @ np.diag([1.0, -1.0])
      sides.append((ip, ir, rob.qadr[[ip, ir]], rob.vadr[[ip, ir]], J))
    rob.ankle_pd = (np.diag([ka, kb]) * ANKLE_TS, np.diag([da, db]) * ANKLE_TS, sides)
  return pol, rob


def apply_ankle_pd(rob, ctrl):
  """Motor-space PD for both ankles: tau_j = J^T clip(Kp J e + Kd J de)."""
  Kp, Kd, sides = rob.ankle_pd
  d = rob.d
  for ip, ir, qa, va, J in sides:
    e = ctrl[[ip, ir]] - d.qpos[qa]
    de = -d.qvel[va]
    tau_m = Kp @ (J @ e) + Kd @ (J @ de)
    tau_m = np.clip(tau_m, -ANKLE_MOTOR_TAU, ANKLE_MOTOR_TAU)
    d.qfrc_applied[va] = J.T @ tau_m


class Sim:
  """One episode: policy at 50 Hz, interpolated targets, per-tick recording."""

  def __init__(self, onnx, plant, ankle_div, seed=0, payload=0.0):
    self.pol, self.rob = make(onnx, plant, ankle_div, payload)
    rng = np.random.default_rng(seed)
    self.rob.reset()
    self.rob.d.qpos[self.rob.qadr] += rng.uniform(-0.01, 0.01, len(self.rob.qadr))
    mujoco.mj_forward(self.rob.m, self.rob.d)
    self.pol.reset()
    self.prev = None
    self.sites = [mujoco.mj_name2id(self.rob.m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot")
                  for s in ("left", "right")]
    self.ank = [self.pol.joint_names.index(n) for n in
                ("right_ankle_pitch_joint", "left_ankle_pitch_joint",
                 "right_ankle_roll_joint", "left_ankle_roll_joint")]
    self.vadr_ank = self.rob.vadr[self.ank]
    self.t, self.rp, self.w, self.fz, self.tau = [], [], [], [], []
    self.fell = None
    self.k = 0

  def step(self, cmd):
    rob, pol, m, d = self.rob, self.pol, self.rob.m, self.rob.d
    rob.command[:] = cmd
    pol.act(pol.observe(m, rob))
    new = pol.targets()             # upper body holds the anchor = upper_body.home
    if self.prev is None:
      self.prev = new.copy()
    for s in range(S.DECIMATION):
      d.ctrl[:] = self.prev + (s + 1) / S.DECIMATION * (new - self.prev)
      if rob.ankle_pd is not None:
        apply_ankle_pd(rob, d.ctrl)
      mujoco.mj_step(m, d)
    self.prev = new
    g = rob.root_rot().T @ S.GRAVITY_W
    self.t.append(self.k * DT)
    self.rp.append([np.degrees(np.arctan2(g[1], -g[2])), np.degrees(np.arctan2(-g[0], -g[2]))])
    self.w.append(rob.gyro()[:2].copy())
    self.fz.append([d.site_xpos[s][2] for s in self.sites])
    src = d.qfrc_applied if rob.ankle_pd is not None else d.qfrc_actuator
    self.tau.append(src[self.vadr_ank].copy())
    if self.fell is None and rob.fell():
      self.fell = self.k * DT
    self.k += 1

  def push(self, v, deg):
    a = np.radians(deg)
    self.rob.d.qvel[0] += v * np.cos(a)
    self.rob.d.qvel[1] += v * np.sin(a)

  def arrays(self):
    return (np.array(self.t), np.array(self.rp), np.array(self.w),
            np.array(self.fz), np.array(self.tau))


def zc_freq(x):
  x = np.asarray(x) - np.mean(x)
  z = np.count_nonzero(np.diff(np.sign(x)) != 0)
  return z / 2 / (len(x) * DT) if len(x) else float("nan")


def settle_time(t, w, t0):
  gn = np.abs(w).max(axis=1)
  n = int(1.0 / DT)
  i0 = int(np.searchsorted(t, t0))
  for i in range(i0, len(gn) - n):
    if gn[i:i + n].max() < SETTLE_RATE:
      return float(t[i] - t0)
  return float("nan")


def foot_lifts(fz, rest, mask):
  """Rising edges of (foot height above its standing height) >= LIFT_MM."""
  out = 0
  for f in (0, 1):
    up = ((fz[:, f] - rest[f]) * 1000 >= LIFT_MM) & mask
    out += int(np.count_nonzero(np.diff(up.astype(int)) == 1))
  return out


# ---------------------------------------------------------------------------
def suite_stop(args):
  cmds = [(0.4, 0, 0), (-0.3, 0, 0), (0.2, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0, 0, 0.7)]
  rows = []
  for cmd in cmds:
    for seed in range(args.seeds):
      sim = Sim(args.onnx, args.plant, args.ankle_div, seed)
      for _ in range(int(1.0 / DT)):
        sim.step((0, 0, 0))
      rest = np.median(np.array(sim.fz)[-25:], axis=0)
      for _ in range(int(4.0 / DT)):
        sim.step(cmd)
      t_stop = sim.k * DT
      for _ in range(int(8.0 / DT)):
        sim.step((0, 0, 0))
      t, rp, w, fz, _ = sim.arrays()
      win = (t >= t_stop + 2) & (t < t_stop + 8)
      late = t >= t_stop + 1.5
      rows.append(dict(
        cmd=list(cmd), seed=seed, fell=sim.fell,
        pitch_pkpk=float(np.ptp(rp[win, 1])), roll_pkpk=float(np.ptp(rp[win, 0])),
        pitch_hz=zc_freq(rp[win, 1]), roll_hz=zc_freq(rp[win, 0]),
        roll_mean=float(rp[win, 0].mean()), pitch_mean=float(rp[win, 1].mean()),
        late_pitch_pkpk=float(np.ptp(rp[t >= t_stop + 6, 1])),
        settle_s=settle_time(t, w, t_stop),
        lifts_after_1p5s=foot_lifts(fz, rest, late)))
  agg = lambda k: float(np.nanmedian([r[k] for r in rows]))
  return dict(runs=rows, falls=sum(r["fell"] is not None for r in rows),
              **{f"median_{k}": agg(k) for k in
                 ("pitch_pkpk", "roll_pkpk", "pitch_hz", "roll_hz", "roll_mean",
                  "pitch_mean", "late_pitch_pkpk",
                  "settle_s", "lifts_after_1p5s")})


def suite_push(args):
  res = {}
  for v in (0.2, 0.35, 0.5):
    rows = []
    for deg in range(0, 360, 45):
      for seed in range(args.seeds):
        sim = Sim(args.onnx, args.plant, args.ankle_div, seed)
        for _ in range(int(3.0 / DT)):
          sim.step((0, 0, 0))
        rest = np.median(np.array(sim.fz)[-25:], axis=0)
        t_push = sim.k * DT
        sim.push(v, deg)
        for _ in range(int(5.0 / DT)):
          sim.step((0, 0, 0))
        t, rp, w, fz, _ = sim.arrays()
        after = t >= t_push
        tilt = np.linalg.norm(rp - rp[int(t_push / DT) - 1], axis=1)
        rows.append(dict(deg=deg, seed=seed, fell=sim.fell,
                         steps=foot_lifts(fz, rest, after),
                         peak_tilt=float(tilt[after].max()),
                         recover_s=settle_time(t, w, t_push)))
    ok = [r for r in rows if r["fell"] is None]
    res[str(v)] = dict(
      success=len(ok) / len(rows),
      stepped=float(np.mean([r["steps"] > 0 for r in rows])),
      median_steps=float(np.median([r["steps"] for r in rows])),
      median_peak_tilt=float(np.median([r["peak_tilt"] for r in rows])),
      median_recover_s=float(np.nanmedian([r["recover_s"] for r in ok])) if ok else float("nan"),
      failed_dirs=sorted({r["deg"] for r in rows if r["fell"] is not None}))
  return res


def suite_shuttle(args):
  """Side-to-side weight-shift loop: after a push, how long does the policy keep
  swinging its hip-roll targets? Measured as the (R - L)/2 hip-roll TARGET
  pk-pk in 2 s windows, the quantity that shuttled 12-14 deg for 18 s on
  tahiti-d1-002 (session 2026-10-10T15-44-00)."""
  rows = []
  for v, deg in ((0.35, 0), (0.35, 180), (0.35, 90), (0.35, 270), (0.2, 90)):
    for seed in range(args.seeds):
      sim = Sim(args.onnx, args.plant, args.ankle_div, seed)
      n = sim.pol.joint_names
      iR, iL = n.index("right_hip_roll_joint"), n.index("left_hip_roll_joint")
      for _ in range(int(3.0 / DT)):
        sim.step((0, 0, 0))
      sim.push(v, deg)
      hr = []
      for _ in range(int(12.0 / DT)):
        sim.step((0, 0, 0))
        t = sim.pol.targets()
        hr.append(np.degrees((t[iR] - t[iL]) / 2))
      hr = np.array(hr)
      pk = [float(np.ptp(hr[int(a / DT):int((a + 2) / DT)])) for a in range(0, 12, 2)]
      rows.append(dict(v=v, deg=deg, seed=seed, fell=sim.fell, pkpk=pk))
  late = [np.mean(r["pkpk"][2:]) for r in rows]     # 4-12 s after the push
  return dict(runs=rows, falls=sum(r["fell"] is not None for r in rows),
              median_late_hiproll_pkpk=float(np.median(late)),
              max_late_hiproll_pkpk=float(np.max(late)))


def suite_clearance(args):
  res = {}
  for cmd in ((0.3, 0, 0), (0.5, 0, 0), (0, 0.3, 0)):
    sim = Sim(args.onnx, args.plant, args.ankle_div, 0)
    for _ in range(int(1.0 / DT)):
      sim.step((0, 0, 0))
    rest = np.median(np.array(sim.fz)[-25:], axis=0)
    xy = []
    for _ in range(int(8.0 / DT)):
      sim.step(cmd)
      xy.append(sim.rob.d.qpos[:2].copy())
    t, _, _, fz, _ = sim.arrays()
    sel = t >= 2.0
    xy = np.array(xy)[int(1.0 / DT):]             # skip the first second of walking
    disp = xy[-1] - xy[0]
    out = dict(speed=float(np.linalg.norm(disp) / ((len(xy) - 1) * DT)),
               heading_deg=float(np.degrees(np.arctan2(disp[1], disp[0]))))
    for f, side in enumerate(("left", "right")):
      h = (fz[sel, f] - rest[f]) * 1000
      up = h >= LIFT_MM
      peaks, s = [], None
      for i in range(1, len(up)):
        if up[i] and not up[i - 1]:
          s = i
        elif not up[i] and up[i - 1] and s is not None:
          peaks.append(h[s:i].max()); s = None
      out[side] = dict(n=len(peaks),
                       median_mm=float(np.median(peaks)) if peaks else 0.0,
                       p10_mm=float(np.percentile(peaks, 10)) if peaks else 0.0)
    res[f"vx{cmd[0]}_vy{cmd[1]}"] = dict(fell=sim.fell, **out)
  return res


def suite_torque(args):
  sim = Sim(args.onnx, args.plant, args.ankle_div, 0)
  for _ in range(int(6.0 / DT)):
    sim.step((0, 0, 0))
  n_stand = sim.k
  for _ in range(int(6.0 / DT)):
    sim.step((0.4, 0, 0))
  _, _, _, _, tau = sim.arrays()
  out = {}
  for phase, sl in (("stand", slice(int(1.0 / DT), n_stand)), ("walk", slice(n_stand + 50, None))):
    a = np.abs(tau[sl])
    out[phase] = dict(ankle_pitch_p99=float(np.percentile(a[:, :2], 99)),
                      ankle_roll_p99=float(np.percentile(a[:, 2:], 99)),
                      ankle_pitch_share_above_43=float((a[:, :2] > 43).mean()))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--plant", choices=("train", "real"), default="real")
  ap.add_argument("--ankle-div", action="store_true",
                  help="command ankle pitch at 178/10.5 (the runtime fix)")
  ap.add_argument("--suite", choices=("stop", "push", "shuttle", "clearance", "torque", "all"),
                  default="all")
  ap.add_argument("--seeds", type=int, default=3)
  ap.add_argument("--imu-bias", type=float, nargs=2, default=(0.0, 0.0), metavar=("ROLL", "PITCH"),
                  help="deg; the policy's IMU reads the pelvis rotated by this")
  ap.add_argument("--out", default=None)
  ap.add_argument("--com-shift", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                  help="mm; torso_link CoM offset")
  ap.add_argument("--ankle-motor", type=float, nargs=4, default=None,
                  metavar=("KA", "KB", "DA", "DB"),
                  help="real plant: ankle motor kp/kd (a = pitch slot, b = roll slot)")
  ap.add_argument("--no-mirror", action="store_true",
                  help="do not mirror the ankle cross-coupling on the left side")
  args = ap.parse_args()
  global IMU_BIAS_DEG, COM_SHIFT_MM, ANKLE_MOTOR_GAINS, MIRROR_LEFT
  MIRROR_LEFT = not args.no_mirror
  ANKLE_MOTOR_GAINS = tuple(args.ankle_motor) if args.ankle_motor else None
  IMU_BIAS_DEG = tuple(args.imu_bias)
  COM_SHIFT_MM = tuple(args.com_shift)
  suites = (("torque", "stop", "shuttle", "push", "clearance") if args.suite == "all"
            else (args.suite,))
  fn = dict(stop=suite_stop, push=suite_push, shuttle=suite_shuttle,
            clearance=suite_clearance, torque=suite_torque)
  res = dict(onnx=args.onnx, plant=args.plant, ankle_div=args.ankle_div)
  for s in suites:
    res[s] = fn[s](args)
    summary = {k: v for k, v in res[s].items() if k != "runs"}
    print(f"[{s}] {json.dumps(summary, default=float)}", flush=True)
  if args.out:
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
  main()
