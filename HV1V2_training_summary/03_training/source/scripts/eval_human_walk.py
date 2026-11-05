#!/usr/bin/env python3
"""A/B: decoupled policy (v7) with its upper body held at home vs driven by the
human-like generator in human_upper.py.

Built on eval_plant.Sim, so the plant (`--plant train real`) and the deployed
command timing (targets interpolated over the 4 substeps; real-plant ankles in
motor space) are eval_plant's. The upper-body targets go through the same
interpolation as the legs.

Suites
  gait   per command (compare_gait's set): stand 2 s, command 10 s, measure the
         last 8 s -> tracking, heading drift, tilt, 1-2 Hz pelvis rocking, gait,
         plus the human-likeness numbers: arm swing amplitude, arm/leg phase,
         thorax counter-rotation, head world-yaw wobble, self-collisions
  start  stand 2 s -> vx 0.3 / vy -0.3: does it start walking (moved > 0.25 m)?
  push   stand 3 s, 0.35 / 0.5 m/s pelvis push in 8 directions
  tour   one scripted walk (stand, accelerate, turn, stop) -> metrics and,
         with --video, a side-by-side mp4 (home | human)

  conda run -n hv1v2_mjlab python mjlab_hv1v2/scripts/eval_human_walk.py \
      --onnx mjlab_hv1v2/exported/deploy_v7/policy.onnx --suite all \
      --out mjlab_hv1v2/logs/eval/human_walk_v7/result.json \
      --video mjlab_hv1v2/logs/eval/human_walk_v7/tour.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_plant as E  # noqa: E402
import sim2sim as S  # noqa: E402
from human_upper import HumanUpperBody, HumanUpperCfg  # noqa: E402

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
TOUR = [(2.0, (0, 0, 0)), (3.0, (0.3, 0, 0)), (4.0, (0.7, 0, 0)), (3.0, (1.0, 0, 0)),
        (5.0, (0.5, 0, 0.4)), (3.0, (0.5, 0, -0.4)), (3.0, (0, 0.3, 0)), (4.0, (0, 0, 0))]
LEG = ("hip", "knee", "ankle")
CFG_OVERRIDES: dict = {}


def yaw_of(R):
  return float(np.arctan2(R[1, 0], R[0, 0]))


class HumanSim(E.Sim):
  """eval_plant.Sim whose upper body is home (mode 'home') or human_upper."""

  def __init__(self, onnx, plant, mode, seed=0, cfg=None):
    super().__init__(onnx, plant, False, seed)
    pol, rob = self.pol, self.rob
    rng = np.random.default_rng(seed)       # start at the policy's own anchor
    rob.reset()
    rob.d.qpos[rob.qadr] = pol.default_qpos + rng.uniform(-0.01, 0.01, len(rob.qadr))
    mujoco.mj_forward(rob.m, rob.d)
    pol.reset()
    self.mode = mode
    self.upper_idx = np.array([i for i in range(len(pol.joint_names))
                               if i not in set(pol.act_idx.tolist())])
    self.gen = None
    if mode == "human":
      self.gen = HumanUpperBody(pol.joint_names, pol.default_qpos, self.upper_idx, DT,
                                cfg or HumanUpperCfg(**CFG_OVERRIDES))
      self.gen.reset(rob.qpos_joints())
    self.ub_log = []
    # v11: the actor sees the upper-body target (rel. home) and its rate
    self.v11 = pol.v11
    if self.v11:
      from mjlab_hv1v2.hv1v2_constants import V11_UB_OBS_JOINTS
      self.ub_cols = np.array([pol.joint_names.index(n) for n in V11_UB_OBS_JOINTS])
      rob.ub_rel = np.zeros(len(self.ub_cols))
      rob.ub_rate = np.zeros(len(self.ub_cols))
      self.last_ub = None

  def step(self, cmd):
    rob, pol, m, d = self.rob, self.pol, self.rob.m, self.rob.d
    rob.command[:] = cmd
    ub = None
    if self.gen is not None:
      ub = self.gen.step(rob.qpos_joints(), rob.root_rot(), np.asarray(cmd, float))
    if self.v11:   # training order: target updates, then the actor sees it
      full = ub if ub is not None else pol.default_qpos
      cur = full[self.ub_cols]
      prev = cur if self.last_ub is None else self.last_ub
      rob.ub_rel = cur - pol.default_qpos[self.ub_cols]
      rob.ub_rate = (cur - prev) / DT
      if os.environ.get("HUMAN_BLIND_UB"):   # diagnostic: actor sees a still target
        rob.ub_rel = np.zeros_like(rob.ub_rel); rob.ub_rate = np.zeros_like(rob.ub_rate)
      self.last_ub = cur.copy()
    pol.act(pol.observe(m, rob))
    new = pol.targets()
    if self.v11:   # as merai_control clips
      new[pol.act_idx] = np.clip(new[pol.act_idx], rob.jnt_lo[pol.act_idx], rob.jnt_hi[pol.act_idx])
    if ub is not None:
      new[self.upper_idx] = ub[self.upper_idx]
    if self.prev is None:
      self.prev = new.copy()
    for s in range(S.DECIMATION):
      d.ctrl[:] = self.prev + (s + 1) / S.DECIMATION * (new - self.prev)
      if rob.ankle_pd is not None:
        E.apply_ankle_pd(rob, d.ctrl)
      mujoco.mj_step(m, d)
    self.prev = new
    self.ub_log.append(new[self.upper_idx].copy())
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


# ---------------------------------------------------------------------------
_SELF_PAIRS = {}


def self_collisions(m, d):
  """Contacts between an arm geom and the torso / pelvis / legs / head."""
  key = id(m)
  if key not in _SELF_PAIRS:
    arm = set(); body = set()
    for g in range(m.ngeom):
      b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
      if any(k in b for k in ("shoulder", "elbow", "wrist")):
        arm.add(g)
      elif b and b != "world":
        body.add(g)
    _SELF_PAIRS[key] = (arm, body)
  arm, body = _SELF_PAIRS[key]
  n = 0
  for c in range(d.ncon):
    g1, g2 = d.contact[c].geom1, d.contact[c].geom2
    if (g1 in arm and g2 in body) or (g2 in arm and g1 in body):
      n += 1
  return n


def band_rms(x, lo, hi):
  x = np.asarray(x) - np.mean(x, axis=0)
  f = np.fft.rfftfreq(len(x), DT)
  P = 2 * np.abs(np.fft.rfft(x, axis=0)) ** 2 / len(x) ** 2
  return np.sqrt(P[(f >= lo) & (f <= hi)].sum(0))


def phase_lag_deg(a, b):
  """Phase of b relative to a at a's dominant frequency (+ = b lags)."""
  a = a - a.mean(); b = b - b.mean()
  A, B = np.fft.rfft(a), np.fft.rfft(b)
  k = 1 + int(np.abs(A[1:]).argmax())
  return float(np.degrees(np.angle(A[k] / B[k] if abs(B[k]) > 1e-9 else 1)))


def amp(x):
  return float((np.percentile(x, 95) - np.percentile(x, 5)) / 2)


# ---------------------------------------------------------------------------
def gait_episode(job):
  mode, onnx, plant, cname, seed = job
  cmd = np.array(COMMANDS[cname])
  sim = HumanSim(onnx, plant, mode, seed)
  rob, pol = sim.rob, sim.pol
  m, d = rob.m, rob.d
  J = pol.joint_names.index
  leg = np.array([i for i, n in enumerate(pol.joint_names) if any(k in n for k in LEG)])
  gait = S.GaitStats(rob, pol.joint_names, warmup_s=4.0)
  torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
  head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head_pitch_link")
  for _ in range(int(2.0 / DT)):
    sim.step((0, 0, 0))
    gait.update(rob.qvel_joints())
  rec = {k: [] for k in ("xy", "yaw", "rp", "w", "acts", "scis", "shL", "shR", "waist",
                         "torso_yaw", "head_yaw", "head_pitch", "coll")}
  for k in range(int(10.0 / DT)):
    sim.step(cmd)
    gait.update(rob.qvel_joints())
    if sim.fell is not None:
      break
    if k * DT < 2.0:
      continue
    q = rob.qpos_joints()
    rec["xy"].append(d.qpos[:2].copy())
    rec["yaw"].append(yaw_of(rob.root_rot()))
    rec["rp"].append(sim.rp[-1]); rec["w"].append(sim.w[-1])
    rec["acts"].append(pol.prev_action.copy())
    rec["scis"].append(0.5 * (q[J("right_hip_pitch_joint")] - q[J("left_hip_pitch_joint")]))
    rec["shL"].append(q[J("left_shoulder_pitch_joint")]); rec["shR"].append(q[J("right_shoulder_pitch_joint")])
    rec["waist"].append(q[J("waist_yaw_joint")])
    rec["torso_yaw"].append(yaw_of(d.xmat[torso].reshape(3, 3)))
    hR = d.xmat[head].reshape(3, 3)
    rec["head_yaw"].append(yaw_of(hR))
    rec["head_pitch"].append(float(np.degrees(np.arcsin(np.clip(-hR[2, 0], -1, 1)))))
    rec["coll"].append(self_collisions(m, d))
    rec.setdefault("shL_cmd", []).append(sim.prev[J("left_shoulder_pitch_joint")])
  out = dict(mode=mode, plant=plant, cmd=cname, seed=seed, fell=sim.fell is not None)
  if sim.fell is not None or len(rec["xy"]) < 50:
    return out
  xy = np.array(rec["xy"]); yaw = np.unwrap(rec["yaw"])
  span = (len(xy) - 1) * DT
  dxy = np.diff(xy, axis=0); c, s_ = np.cos(yaw[:-1]), np.sin(yaw[:-1])
  vx = float((c * dxy[:, 0] + s_ * dxy[:, 1]).sum() / span)
  vy = float((-s_ * dxy[:, 0] + c * dxy[:, 1]).sum() / span)
  wz = float((yaw[-1] - yaw[0]) / span)
  rp, w, acts = np.array(rec["rp"]), np.array(rec["w"]), np.array(rec["acts"])
  detrend = lambda y: y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
  tyaw, hyaw = np.unwrap(rec["torso_yaw"]), np.unwrap(rec["head_yaw"])
  scis = np.array(rec["scis"]); shL = np.array(rec["shL"])
  out.update(
    vx=vx, vy=vy, wz=wz,
    err_xy=float(np.hypot(vx - cmd[0], vy - cmd[1])), err_wz=abs(wz - cmd[2]),
    heading_drift_deg=float(np.degrees(yaw[-1] - yaw[0])) if cmd[2] == 0 else float("nan"),
    tilt_rms=float(np.sqrt((rp ** 2).sum(axis=1).mean())),
    roll_mean=float(rp[:, 0].mean()), pitch_mean=float(rp[:, 1].mean()),
    rock_1_2=float(np.linalg.norm(band_rms(w, 1, 2))),
    w_rms=float(np.sqrt((w ** 2).sum(1).mean())),
    step_len=float(np.mean(gait.dx)) if gait.dx else 0.0,
    swing=float(np.mean(gait.swing)) if gait.swing else 0.0,
    cadence=gait.touchdowns / max(1e-6, (gait.n - gait.warmup_steps) * DT),
    da=float(np.abs(np.diff(acts, axis=0)).mean()),
    # human-likeness
    scissor_amp=amp(scis),
    arm_amp=amp(shL),
    arm_vs_leg_lag_deg=phase_lag_deg(scis, shL) if amp(shL) > 0.02 else float("nan"),
    arm_cmd_lag_deg=phase_lag_deg(scis, np.array(rec["shL_cmd"])) if amp(shL) > 0.02 else float("nan"),
    pelvis_yaw_wobble_deg=float(np.degrees(amp(detrend(yaw)))),
    torso_yaw_wobble_deg=float(np.degrees(amp(detrend(tyaw)))),
    head_yaw_wobble_deg=float(np.degrees(amp(detrend(hyaw)))),
    head_pitch_wobble_deg=amp(detrend(np.array(rec["head_pitch"]))),
    waist_amp=amp(np.array(rec["waist"])),
    self_contacts=float(np.mean(np.array(rec["coll"]) > 0)),
  )
  return out


def start_episode(job):
  mode, onnx, plant, cname, seed = job
  cmd = {"vx0.3": (0.3, 0, 0), "vy-0.3": (0, -0.3, 0), "vx0.15": (0.15, 0, 0)}[cname]
  sim = HumanSim(onnx, plant, mode, seed)
  for _ in range(int(2.0 / DT)):
    sim.step((0, 0, 0))
  p0 = sim.rob.d.qpos[:2].copy()
  for _ in range(int(5.0 / DT)):
    sim.step(cmd)
  moved = float(np.linalg.norm(sim.rob.d.qpos[:2] - p0))
  return dict(mode=mode, plant=plant, cmd=cname, seed=seed, fell=sim.fell is not None,
              moved=moved, started=moved > 0.25)


def push_episode(job):
  mode, onnx, plant, v, deg, seed = job
  sim = HumanSim(onnx, plant, mode, seed)
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
  return dict(mode=mode, plant=plant, v=v, deg=deg, seed=seed, fell=sim.fell is not None,
              steps=E.foot_lifts(fz, rest, after), peak_tilt=float(tilt[after].max()),
              recover_s=E.settle_time(t, w, t_push))


def tour(onnx, plant, mode, video=False):
  sim = HumanSim(onnx, plant, mode, 0)
  frames, rend, cam = [], None, None
  if video:
    m = sim.rob.m
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 960
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.65
    m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")] = [0.55, 0.57, 0.6, 1]
    rend = mujoco.Renderer(m, 600, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.elevation, cam.azimuth = 2.6, -8, 70
  coll = 0
  yaw_cam = None
  for dur, cmd in TOUR:
    for _ in range(int(dur / DT)):
      sim.step(cmd)
      coll += self_collisions(sim.rob.m, sim.rob.d) > 0
      if video:
        cam.lookat[:] = [sim.rob.d.qpos[0], sim.rob.d.qpos[1], 0.95]
        # side view that follows the heading (low-passed so the gait wobble
        # does not shake the camera)
        yaw_cam = yaw_of(sim.rob.root_rot()) if yaw_cam is None else (
          yaw_cam + 0.03 * ((yaw_of(sim.rob.root_rot()) - yaw_cam + np.pi) % (2 * np.pi) - np.pi))
        cam.azimuth = np.degrees(yaw_cam) + 90
        rend.update_scene(sim.rob.d, cam)
        frames.append(rend.render())
  ub = np.array(sim.ub_log)
  return dict(fell=sim.fell, self_contact_ticks=int(coll),
              ub_peak_rate=float(np.abs(np.diff(ub, axis=0)).max() / DT)), frames


# ---------------------------------------------------------------------------
def agg(rows, keys):
  out = {}
  for k in keys:
    vals = [r[k] for r in rows if k in r and r[k] is not None and np.isfinite(r[k])]
    out[k] = float(np.median(vals)) if vals else float("nan")
  out["falls"] = int(sum(r["fell"] for r in rows))
  return out


GAIT_KEYS = ("vx", "vy", "wz", "err_xy", "err_wz", "heading_drift_deg", "tilt_rms", "roll_mean",
             "pitch_mean", "rock_1_2", "w_rms", "step_len", "swing", "cadence", "da",
             "scissor_amp", "arm_amp", "arm_vs_leg_lag_deg", "arm_cmd_lag_deg", "pelvis_yaw_wobble_deg",
             "torso_yaw_wobble_deg", "head_yaw_wobble_deg", "head_pitch_wobble_deg",
             "waist_amp", "self_contacts")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", default=str(Path(__file__).resolve().parents[1]
                                        / "exported/deploy_v7/policy.onnx"))
  ap.add_argument("--suite", default="all", choices=("gait", "start", "push", "tour", "all"))
  ap.add_argument("--plant", nargs="+", default=["train", "real"])
  ap.add_argument("--modes", nargs="+", default=["home", "human"])
  ap.add_argument("--seeds", type=int, default=3)
  ap.add_argument("--workers", type=int, default=os.cpu_count())
  ap.add_argument("--cmds", nargs="+", default=list(COMMANDS), help="gait suite commands")
  ap.add_argument("--cfg", default="{}", help="JSON overrides for HumanUpperCfg")
  ap.add_argument("--out", default=None)
  ap.add_argument("--video", default=None, help="tour mp4 (home | human side by side)")
  args = ap.parse_args()
  CFG_OVERRIDES.update(json.loads(args.cfg))
  res = {"cfg": CFG_OVERRIDES}
  suites = ("gait", "start", "push", "tour") if args.suite == "all" else (args.suite,)
  with ProcessPoolExecutor(args.workers, initializer=CFG_OVERRIDES.update,
                           initargs=(CFG_OVERRIDES,)) as ex:
    if "gait" in suites:
      jobs = [(mo, args.onnx, p, c, s) for mo in args.modes for p in args.plant
              for c in args.cmds for s in range(args.seeds)]
      rows = list(ex.map(gait_episode, jobs))
      res["gait"] = {f"{p}/{c}/{mo}": agg([r for r in rows if (r["mode"], r["plant"], r["cmd"]) == (mo, p, c)], GAIT_KEYS)
                     for p in args.plant for c in args.cmds for mo in args.modes}
    if "start" in suites:
      jobs = [(mo, args.onnx, p, c, s) for mo in args.modes for p in args.plant
              for c in ("vx0.15", "vx0.3", "vy-0.3") for s in range(5)]
      rows = list(ex.map(start_episode, jobs))
      res["start"] = {f"{p}/{c}/{mo}": dict(
        started=sum(r["started"] for r in rows if (r["mode"], r["plant"], r["cmd"]) == (mo, p, c)),
        of=5, falls=sum(r["fell"] for r in rows if (r["mode"], r["plant"], r["cmd"]) == (mo, p, c)))
        for p in args.plant for c in ("vx0.15", "vx0.3", "vy-0.3") for mo in args.modes}
    if "push" in suites:
      jobs = [(mo, args.onnx, p, v, deg, s) for mo in args.modes for p in args.plant
              for v in (0.35, 0.5) for deg in range(0, 360, 45) for s in range(2)]
      rows = list(ex.map(push_episode, jobs))
      res["push"] = {}
      for p in args.plant:
        for v in (0.35, 0.5):
          for mo in args.modes:
            rr = [r for r in rows if (r["mode"], r["plant"], r["v"]) == (mo, p, v)]
            ok = [r for r in rr if not r["fell"]]
            res["push"][f"{p}/{v}/{mo}"] = dict(
              success=len(ok) / len(rr),
              median_steps=float(np.median([r["steps"] for r in rr])),
              median_peak_tilt=float(np.median([r["peak_tilt"] for r in rr])),
              median_recover_s=float(np.nanmedian([r["recover_s"] for r in ok])) if ok else float("nan"))
  if "tour" in suites:
    res["tour"] = {}
    vids = {}
    for p in args.plant:
      for mo in args.modes:
        r, fr = tour(args.onnx, p, mo, video=bool(args.video) and p == args.plant[-1])
        res["tour"][f"{p}/{mo}"] = r
        if fr:
          vids[mo] = fr
    if args.video and vids:
      import mediapy
      n = min(len(v) for v in vids.values())
      frames = [np.concatenate([vids[mo][i] for mo in args.modes if mo in vids], axis=1)
                for i in range(n)]
      Path(args.video).parent.mkdir(parents=True, exist_ok=True)
      mediapy.write_video(args.video, frames, fps=1 / DT)
      print(f"wrote {args.video}")

  txt = json.dumps(res, indent=1, default=lambda x: None if x is None else float(x))
  if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt)
  print_table(res, args)


def print_table(res, args):
  if "gait" in res:
    cols = ("err_xy", "err_wz", "heading_drift_deg", "tilt_rms", "rock_1_2", "step_len",
            "arm_amp", "arm_vs_leg_lag_deg", "arm_cmd_lag_deg", "torso_yaw_wobble_deg", "head_yaw_wobble_deg",
            "self_contacts", "falls")
    print(f"{'plant/cmd/mode':28s}" + "".join(f"{c[:11]:>12s}" for c in cols))
    for k, v in res["gait"].items():
      print(f"{k:28s}" + "".join(f"{v[c]:12.3f}" for c in cols))
  for s in ("start", "push", "tour"):
    if s in res:
      print(f"\n[{s}]")
      for k, v in res[s].items():
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
  main()
