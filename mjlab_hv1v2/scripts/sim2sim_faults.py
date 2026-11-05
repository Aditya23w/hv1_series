#!/usr/bin/env python3
"""sim2sim with the sensor faults found on tahiti-d1-002, to see what they do to a policy.

  conda run -n hv1v2_mjlab python mjlab_hv1v2/scripts/sim2sim_faults.py --run v9
  ... --scenario knee_measured --seconds 30 --vx 0
  ... --list

A JOINT ZERO ERROR delta (deg, true = read + delta) is injected the way it acts on
the robot: the encoder reports qpos - delta, so the policy observes that, and the
drive servos the READING to the target, so the plant's position target becomes
target + delta. An IMU PITCH BIAS b (deg, + = reads further forward than true)
only changes the projected gravity the policy observes.

Everything else -- plant, observation assembly, actuators -- is sim2sim.py's,
imported unchanged, so a baseline run here is a plain sim2sim run.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402

# Shares from merai_sdk/scripts/lean_rca.py T11 (raw gap 6.75 deg) and the
# 2026-10-21 hard-stop readings (right 12.19 / left 9 against a 10 deg stop).
SCENARIOS = {
  "baseline": dict(),
  "knee_measured": dict(joints={"right_knee_joint": +2.19, "left_knee_joint": -1.00}),
  "knee_measured_flipped": dict(joints={"right_knee_joint": -2.19, "left_knee_joint": +1.00}),
  "knee_1": dict(joints={"right_knee_joint": 1.0, "left_knee_joint": 1.0}),
  "knee_2": dict(joints={"right_knee_joint": 2.0, "left_knee_joint": 2.0}),
  "knee_3": dict(joints={"right_knee_joint": 3.0, "left_knee_joint": 3.0}),
  "knee_4": dict(joints={"right_knee_joint": 4.0, "left_knee_joint": 4.0}),
  "knee_6": dict(joints={"right_knee_joint": 6.0, "left_knee_joint": 6.0}),
  "chain_fit_imu_ok": dict(joints={  # T11 at IMU bias 0: hip 2.0 / knee 4.0 / ankle 0.7 deg of chain
    "right_hip_pitch_joint": 2.0 / 0.8660254, "left_hip_pitch_joint": 2.0 / 0.8660254,
    "right_knee_joint": 4.0, "left_knee_joint": 4.0,
    "right_ankle_pitch_joint": 0.7, "left_ankle_pitch_joint": 0.7}),
  "imu_only": dict(imu=-6.75),
  "imu_4p5_knee": dict(imu=-4.5, joints={"right_knee_joint": 1.1, "left_knee_joint": 1.1,
                                          "right_ankle_pitch_joint": 1.5, "left_ankle_pitch_joint": 1.5}),
}


LABELS = {
  "baseline": "v9, no error",
  "chain_fit_imu_ok": "v9 + chain error: hip 2.3, knee 4.0, ankle 0.7 deg",
  "knee_measured": "v9 + measured knees: R +2.19, L -1.0 deg",
  "knee_4": "v9 + knee 4 deg both",
  "imu_only": "v9 + IMU 6.75 deg nose-back",
}


def rot_y(a):
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class FaultyRobot(S.Robot):
  def __init__(self, joint_names, delta_deg: dict, imu_bias_deg: float, for_video: bool = False):
    super().__init__(joint_names, for_video=for_video)
    self.delta = np.array([math.radians(delta_deg.get(n, 0.0)) for n in joint_names])
    # Body-frame rotation that makes asin(g_x) read imu_bias_deg higher (checked below).
    self.R_bias = rot_y(math.radians(imu_bias_deg))
    self.names = list(joint_names)

  # what the encoders report
  def qpos_joints(self):
    return self.d.qpos[self.qadr].copy() - self.delta

  # what the IMU reports
  def root_rot(self):
    return self.d.xmat[self.pelvis].reshape(3, 3).copy() @ self.R_bias

  def true_pitch_deg(self):
    g = self.d.xmat[self.pelvis].reshape(3, 3).T @ S.GRAVITY_W
    return math.degrees(math.asin(np.clip(g[0], -1, 1)))

  def seen_pitch_deg(self):
    g = self.root_rot().T @ S.GRAVITY_W
    return math.degrees(math.asin(np.clip(g[0], -1, 1)))


def run(onnx_path, sc, seconds, vx, settle, walk=None):
  """walk = (vx, seconds): walk at vx first, then measure `seconds` at the
  command `vx` (normally 0) -- does the policy settle after a walk?"""
  pol = S.Policy(onnx_path)
  rb = FaultyRobot(pol.joint_names, sc.get("joints", {}), sc.get("imu", 0.0))
  rb.command[:] = [vx, 0.0, 0.0]
  rb.reset()
  pol.reset()
  m, d = rb.m, rb.d
  names = pol.joint_names
  J = {n: i for i, n in enumerate(names)}
  foot = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("right", "left")}
  mass = float(m.body_subtreemass[0]) if m.body_subtreemass[0] > 1 else float(sum(m.body_mass))
  pel_x0 = d.qpos[0]
  heading = None
  rows = []
  pre = int(walk[1] / (S.SIM_DT * S.DECIMATION)) if walk else 0
  steps = pre + int(seconds / (S.SIM_DT * S.DECIMATION))
  if walk:
    settle = settle + walk[1]
  foot_z0 = {s: d.xpos[foot[s]][2] for s in foot}
  lifts = {s: 0 for s in foot}
  up = {s: False for s in foot}
  fell_at = None
  for k in range(steps):
    rb.command[:] = [walk[0] if (walk and k < pre) else vx, 0.0, 0.0]
    obs = pol.observe(m, rb)
    pol.act(obs)
    ctrl = pol.targets() + rb.delta            # drive servos the reading to the target
    d.ctrl[:] = ctrl
    for _ in range(S.DECIMATION):
      mujoco.mj_step(m, d)
    t = (k + 1) * S.SIM_DT * S.DECIMATION
    if rb.fell():
      fell_at = t
      break
    if t < settle:
      continue
    if heading is None:
      R = d.xmat[rb.pelvis].reshape(3, 3)
      heading = math.atan2(R[1, 0], R[0, 0])
      x_start = d.qpos[0:2].copy()
    for s in foot:
      h = d.xpos[foot[s]][2] - foot_z0[s]
      if h > 0.012 and not up[s]:
        up[s] = True; lifts[s] += 1
      elif h < 0.006:
        up[s] = False
    tau = d.actuator_force.copy()
    rows.append((t, rb.true_pitch_deg(), rb.seen_pitch_deg(),
                 tau[J["right_ankle_pitch_joint"]] + tau[J["left_ankle_pitch_joint"]],
                 tau[J["right_knee_joint"]], tau[J["left_knee_joint"]],
                 max(abs(tau[J["right_ankle_pitch_joint"]]), abs(tau[J["left_ankle_pitch_joint"]]))))
  A = np.array(rows) if rows else np.zeros((1, 7))
  if heading is not None:
    dxy = d.qpos[0:2] - x_start
    fwd = float(dxy[0] * math.cos(heading) + dxy[1] * math.sin(heading))
  else:
    fwd = float("nan")
  T = A[-1, 0] - A[0, 0] if len(A) > 1 else 0.0
  return dict(fell_at=fell_at, fwd_m=fwd, fwd_speed=fwd / T if T > 0 else float("nan"),
              true_pitch=float(A[:, 1].mean()), seen_pitch=float(A[:, 2].mean()),
              com_x_mm=float(A[:, 3].mean() / (mass * 9.81) * 1000),
              knee_tau=(float(A[:, 4].mean()), float(A[:, 5].mean())),
              ankle_peak=float(np.percentile(A[:, 6], 95)),
              steps=sum(lifts.values()), seconds=T)


def video(onnx_path, names, path, walk, seconds, vx):
  """Side-by-side mp4: one panel per scenario, same command schedule, fixed side
  camera per panel with a marker where the walk command ends."""
  import mediapy
  from PIL import Image, ImageDraw, ImageFont
  os.environ.setdefault("MUJOCO_GL", "egl")
  W, H = 640, 480
  sims = []
  for n in names:
    sc = SCENARIOS[n]
    pol = S.Policy(onnx_path)
    rb = FaultyRobot(pol.joint_names, sc.get("joints", {}), sc.get("imu", 0.0), for_video=True)
    rb.reset(); pol.reset()
    rb.m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    rb.m.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    fl = mujoco.mj_name2id(rb.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fl >= 0:
      rb.m.geom_rgba[fl] = [0.55, 0.58, 0.62, 1.0]
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(rb.m, cam)
    cam.distance, cam.elevation, cam.azimuth = 3.7, -8, 90      # side view: pitch and fore-aft drift
    sims.append(dict(name=LABELS.get(n, n), pol=pol, rb=rb, cam=cam, rend=mujoco.Renderer(rb.m, H, W), mark=None,
                     x0=rb.d.qpos[0:2].copy()))
  dt = S.SIM_DT * S.DECIMATION
  pre = int(walk[1] / dt) if walk else 0
  total = pre + int(seconds / dt)
  try:
    font = ImageFont.truetype("DejaVuSans.ttf", 18); small = ImageFont.truetype("DejaVuSans.ttf", 15)
  except OSError:
    font = small = ImageFont.load_default()
  frames = []
  for k in range(total):
    cmd = walk[0] if (walk and k < pre) else vx
    panels = []
    for sm in sims:
      rb, pol = sm["rb"], sm["pol"]
      rb.command[:] = [cmd, 0.0, 0.0]
      if not rb.fell():
        pol.act(pol.observe(rb.m, rb))
        rb.d.ctrl[:] = pol.targets() + rb.delta
        for _ in range(S.DECIMATION):
          mujoco.mj_step(rb.m, rb.d)
      if k == pre and sm["mark"] is None:
        sm["mark"] = rb.d.qpos[0:2].copy()
      if sm["mark"] is None:
        sm["cam"].lookat[:] = [rb.d.qpos[0], rb.d.qpos[1], 0.85]
      else:                                                    # freeze the camera at the stop point
        sm["cam"].lookat[:] = [sm["mark"][0], sm["mark"][1], 0.85]
      sm["rend"].update_scene(rb.d, sm["cam"])
      sc = sm["rend"].scene
      for i in range(-12, 41):                                 # floor ticks every 0.25 m along the walk
        if sc.ngeom >= sc.maxgeom:
          break
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.004, 0.35 if i % 4 else 0.6, 0.001]),
                            np.array([sm["x0"][0] + 0.25 * i, sm["x0"][1], 0.001]), np.eye(3).flatten(),
                            np.array([0.15, 0.17, 0.2, 1.0], dtype=np.float32))
        sc.ngeom += 1
      if sm["mark"] is not None:
        if sc.ngeom < sc.maxgeom:
          g = sc.geoms[sc.ngeom]
          mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.012, 0.7, 0.002]),
                              np.array([sm["mark"][0], sm["mark"][1], 0.003]), np.eye(3).flatten(),
                              np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32))
          sc.ngeom += 1
      img = Image.fromarray(sm["rend"].render())
      dr = ImageDraw.Draw(img)
      t = (k + 1) * dt
      travel = (rb.d.qpos[0] - sm["mark"][0]) if sm["mark"] is not None else 0.0
      dr.rectangle([0, 0, W, 58], fill=(12, 14, 18))
      dr.text((12, 6), sm["name"], fill=(240, 240, 240), font=font)
      dr.text((12, 32), f"t {t:5.1f} s   cmd vx {cmd:+.1f} m/s   pelvis pitch {rb.true_pitch_deg():+5.1f} deg"
              + (f"   moved since stop {travel:+.2f} m" if sm["mark"] is not None else ""), fill=(200, 205, 210), font=small)
      if sm["mark"] is not None:
        dr.text((12, H - 26), "yellow line = where the walk command stopped", fill=(230, 200, 60), font=small)
      panels.append(np.asarray(img))
    frames.append(np.concatenate(panels, axis=1))
  mediapy.write_video(path, frames, fps=1 / dt)
  print(f"wrote {path}  ({len(frames)} frames, {total * dt:.1f} s, panels: {', '.join(names)})")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default="v9")
  ap.add_argument("--onnx", default=None)
  ap.add_argument("--scenario", default="all")
  ap.add_argument("--seconds", type=float, default=30.0)
  ap.add_argument("--settle", type=float, default=2.0, help="ignore the first N s")
  ap.add_argument("--vx", type=float, default=0.0)
  ap.add_argument("--list", action="store_true")
  ap.add_argument("--video", default=None, help="write a side-by-side mp4 of --scenario (comma list)")
  ap.add_argument("--walk-first", type=float, nargs=2, default=None, metavar=("VX", "SECONDS"),
                  help="walk at VX for SECONDS before the measured part")
  args = ap.parse_args()
  if args.list:
    for k, v in SCENARIOS.items():
      print(f"{k:24s} {v}")
    return
  onnx_path = args.onnx or S.find_onnx(args.run)
  if args.video:
    video(onnx_path, args.scenario.split(","), args.video, args.walk_first, args.seconds, args.vx)
    return
  # sign check of the IMU bias: a +1 deg bias must read +1 deg
  chk = FaultyRobot(S.Policy(onnx_path).joint_names, {}, 1.0); chk.reset()
  assert abs(chk.seen_pitch_deg() - chk.true_pitch_deg() - 1.0) < 1e-6, "IMU bias sign"
  names = list(SCENARIOS) if args.scenario == "all" else args.scenario.split(",")
  pre = f"walk {args.walk_first[0]} m/s for {args.walk_first[1]:.0f} s, then " if args.walk_first else ""
  print(f"policy {os.path.basename(onnx_path)}   {pre}vx {args.vx} for {args.seconds:.0f} s (first {args.settle:.0f} s of that ignored)")
  print(f"{'scenario':24s} {'fell':>6s} {'travel m':>9s} {'m/s':>6s} {'steps':>6s} {'true pitch':>11s} {'seen pitch':>11s}"
        f" {'CoM mm':>7s} {'knee R/L Nm':>13s} {'ankle p95':>9s}")
  for n in names:
    r = run(onnx_path, SCENARIOS[n], args.seconds, args.vx, args.settle, args.walk_first)
    fell = f"{r['fell_at']:.1f}s" if r["fell_at"] else "no"
    print(f"{n:24s} {fell:>6s} {r['fwd_m']:+9.3f} {r['fwd_speed']:+6.3f} {r['steps']:6d} {r['true_pitch']:+11.2f}"
          f" {r['seen_pitch']:+11.2f} {r['com_x_mm']:+7.1f} {r['knee_tau'][0]:+6.1f}/{r['knee_tau'][1]:+6.1f} {r['ankle_peak']:9.1f}")


if __name__ == "__main__":
  main()
