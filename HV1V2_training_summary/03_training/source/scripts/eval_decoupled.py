#!/usr/bin/env python3
"""Deterministic evaluation suite for the decoupled HV1V2 policies (v7, v8, ...).

Plain CPU MuJoCo, same plant as training (built from the EntityCfg), with the
DEPLOYED command timing: every 50 Hz target -- legs from the policy, upper body
from the Mink -> Ruckig clips -- is linearly interpolated across the 4 physics
substeps, as merai_control interpolates to 1 kHz.

Suites:
  primary    stand -> left hand -> right hand -> both -> waist yaw -> home
  sweep      the primary sequence at 25/50/75/100 % amplitude, N seeds each
  walking    the primary sequence while standing, at vx 0.3 and vx 0.5
  aggressive full deployment range at <= 2.0 rad/s (waist 0.6) + 2 kg/hand;
             a recovery step is legal, a fall is not

  python mjlab_hv1v2/scripts/eval_decoupled.py --onnx exported/deploy_v7/policy.onnx \
      --suite all --out /tmp/v7_eval.json

Metrics (per run): pelvis drift, roll/pitch RMS + peak, roll/pitch-rate RMS +
peak, leg qd RMS, foot displacement, steps (touchdowns while commanded to stand,
after a 1 s settle), upper-body tracking error, per-segment settling time
(after motion stops: pelvis rate < 0.05 rad/s and tilt within 0.2 deg of its
final value, held 0.5 s), 1-2 Hz / 3-8 Hz energy share (pelvis rates, each leg
joint), and -- reported only -- CoP, CoM projection and support margin.
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

LAB = Path(__file__).resolve().parents[1]
CLIPS = LAB / "data/clips_v2/scripted.npz"
DT = S.SIM_DT * S.DECIMATION
PRIMARY = ("left_fwd_slow", "right_fwd_slow", "both_fwd_slow", "waist_yaw_+0.8")
SETTLE_HOLD_S = 3.0           # quiet time after each segment (settling window)
LEGS = ("hip", "knee", "ankle")


# ---------------------------------------------------------------------------
def load_clips(path=None):
  d = np.load(path or CLIPS)
  names = [str(x) for x in d["names"]]
  clips = {n: d["q"][s:s + L] for n, s, L in zip(names, d["start"], d["length"])}
  return clips, [str(x) for x in d["joint_names"]], d["home"]


def build_upper_stream(pol, clip_names, amp, clips, clip_joints, lead_s=2.0):
  """Full-width upper targets per 50 Hz tick + segment bookkeeping."""
  home = pol.default_qpos.copy()
  cols = [pol.joint_names.index(j) for j in clip_joints]
  ticks, segs = [], []
  def hold(sec):
    for _ in range(int(sec / DT)):
      ticks.append(home.copy())
  hold(lead_s)
  for n in clip_names:
    q = clips[n]
    t0 = len(ticks)
    for row in q:
      t = home.copy()
      t[cols] = home[cols] + amp * (row - home[cols])
      ticks.append(t)
    # "Motion stops" = the last tick the command still moves, not the clip end:
    # clips finish with their own home hold, which would count as settling.
    moving = np.nonzero(np.abs(np.diff(q, axis=0)).max(axis=1) > 1e-4)[0]
    stop = t0 + (int(moving[-1]) + 2 if len(moving) else len(q))
    segs.append(dict(name=n, start=t0, stop=stop))
    hold(SETTLE_HOLD_S)
  return np.array(ticks), segs


def aggressive_stream(pol, seconds=12.0, upper_set="ik"):
  """Full trained range sweep, rate-limited to the deployed limits."""
  from mjlab_hv1v2.hv1v2_constants import UPPER_BODY_SAMPLE_RANGE
  import re
  home = pol.default_qpos.copy()
  # Only the joints the deployed teleop IK drives move (head, left_wrist_roll,
  # waist roll/pitch hold home, as on the robot).
  idx = [i for i in range(len(pol.joint_names)) if i not in set(pol.act_idx.tolist())
         and (upper_set == "all" or pol.joint_names[i] in S.UpperBodyDriver.IK_JOINTS)]
  lo, hi, vmax = [], [], []
  for i in idx:
    n = pol.joint_names[i]
    r = [v for p, v in UPPER_BODY_SAMPLE_RANGE.items() if re.match(p, n)][0]
    lo.append(r[0]); hi.append(r[1])
    vmax.append(0.6 if n.startswith("waist_yaw") else 2.0)
  lo, hi, vmax = map(np.array, (lo, hi, vmax))
  ph = np.random.default_rng(0).uniform(0, 2 * np.pi, len(idx))
  cur = home[idx].copy()
  ticks = []
  for _ in range(int(2.0 / DT)):
    ticks.append(home.copy())
  for k in range(int(seconds / DT)):
    goal = lo + (hi - lo) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * k * DT + ph))
    cur = cur + np.clip(goal - cur, -vmax * DT, vmax * DT)
    t = home.copy(); t[idx] = cur; ticks.append(t)
  for _ in range(int(3.0 / DT)):          # slew home
    cur = cur + np.clip(home[idx] - cur, -vmax * DT, vmax * DT)
    t = home.copy(); t[idx] = cur; ticks.append(t)
  return np.array(ticks), [dict(name="aggressive", start=int(2.0 / DT),
                                stop=int(2.0 / DT) + int(seconds / DT))]


# ---------------------------------------------------------------------------
def band_share(x, lo, hi):
  x = np.asarray(x) - np.mean(x, axis=0)
  f = np.fft.rfftfreq(len(x), DT)
  P = np.abs(np.fft.rfft(x, axis=0)) ** 2
  tot = P[f > 0.2].sum(0)
  return P[(f >= lo) & (f <= hi)].sum(0) / np.maximum(tot, 1e-12)


def band_rms(x, lo, hi):
  """Absolute RMS of the lo..hi Hz content of x (per column). Unlike
  band_share it does not go down when OTHER frequencies go up."""
  x = np.asarray(x) - np.mean(x, axis=0)
  f = np.fft.rfftfreq(len(x), DT)
  P = 2 * np.abs(np.fft.rfft(x, axis=0)) ** 2 / len(x) ** 2
  return np.sqrt(P[(f >= lo) & (f <= hi)].sum(0))


def hull(pts):
  pts = sorted(map(tuple, pts))
  if len(pts) < 3:
    return np.array(pts)
  def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  lower, upper = [], []
  for p in pts:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
      lower.pop()
    lower.append(p)
  for p in reversed(pts):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
      upper.pop()
    upper.append(p)
  return np.array(lower[:-1] + upper[:-1])


def support_margin(p, poly):
  """Signed distance from point p to polygon edge (+ inside)."""
  if len(poly) < 3:
    return float("nan")
  d = np.inf
  inside = True
  for i in range(len(poly)):
    a, b = poly[i], poly[(i + 1) % len(poly)]
    e = b - a
    t = np.clip(np.dot(p - a, e) / max(np.dot(e, e), 1e-12), 0, 1)
    d = min(d, np.linalg.norm(p - (a + t * e)))
    if e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) < 0:
      inside = False
  return d if inside else -d


def run(onnx, upper_ticks, segs, vx=0.0, payload=0.0, seed=0):
  pol = S.Policy(onnx)
  rob = S.Robot(pol.joint_names, for_video=False)
  if payload:
    S.add_hand_payload(rob, payload)
  rng = np.random.default_rng(seed)
  rob.reset()
  rob.d.qpos[rob.qadr] += rng.uniform(-0.01, 0.01, len(rob.qadr))
  mujoco.mj_forward(rob.m, rob.d)
  pol.reset()
  rob.command[:] = [vx, 0.0, 0.0]
  m, d = rob.m, rob.d
  upper_idx = np.array([i for i in range(len(pol.joint_names))
                        if i not in set(pol.act_idx.tolist())])
  leg_idx = [i for i, n in enumerate(pol.joint_names) if any(k in n for k in LEGS)]
  gait = S.GaitStats(rob, pol.joint_names, warmup_s=1.0)
  floor = gait.floor
  pads = gait.pads[0] | gait.pads[1]
  prev_ctrl = None
  rec = {k: [] for k in ("xy", "rp", "w", "qd_leg", "track", "cop", "com", "margin",
                         "feet")}
  rec["leg_names"] = [pol.joint_names[i] for i in leg_idx]
  fell = None
  for k, upper_t in enumerate(upper_ticks):
    pol.act(pol.observe(m, rob))
    new = pol.targets()
    new[upper_idx] = upper_t[upper_idx]
    if prev_ctrl is None:
      prev_ctrl = new.copy()
    for s in range(S.DECIMATION):                 # merai_control-style interpolation
      a = (s + 1) / S.DECIMATION
      d.ctrl[:] = prev_ctrl + a * (new - prev_ctrl)
      mujoco.mj_step(m, d)
    prev_ctrl = new
    if fell is None and rob.fell():
      fell = k * DT
    if fell is not None:
      continue
    gait.update(rob.qvel_joints())
    R = rob.root_rot()
    g = R.T @ S.GRAVITY_W
    rec["xy"].append(d.qpos[:2].copy())
    rec["rp"].append([np.degrees(np.arctan2(g[1], -g[2])), np.degrees(np.arctan2(-g[0], -g[2]))])
    rec["w"].append(rob.gyro()[:2].copy())
    rec["qd_leg"].append(rob.qvel_joints()[leg_idx])
    rec["track"].append(np.abs(rob.qpos_joints()[upper_idx] - upper_t[upper_idx]).mean())
    rec["feet"].append(d.site_xpos[gait.sites][:, :2].copy())
    # CoP / CoM / support margin (report only)
    fz_sum, cop, pts = 0.0, np.zeros(2), []
    f6 = np.zeros(6)
    for c in range(d.ncon):
      con = d.contact[c]
      if floor not in (con.geom1, con.geom2):
        continue
      if (con.geom2 if con.geom1 == floor else con.geom1) not in pads:
        continue
      mujoco.mj_contactForce(m, d, c, f6)
      fz = abs(f6[0])
      fz_sum += fz
      cop += fz * con.pos[:2]
      pts.append(con.pos[:2].copy())
    com = d.subtree_com[1][:2].copy()
    rec["com"].append(com)
    rec["cop"].append(cop / fz_sum if fz_sum > 0 else com)
    rec["margin"].append(support_margin(com, hull(pts)) if len(pts) >= 3 else np.nan)
  return summarize(rec, segs, gait, fell, vx)


def settling_times(rp, w, segs):
  out = []
  rate = np.linalg.norm(w, axis=1)
  tilt = np.linalg.norm(rp, axis=1)
  hold_n = int(0.5 / DT)
  for sgm in segs:
    a = sgm["stop"]
    b = min(a + int(SETTLE_HOLD_S / DT), len(rate), len(tilt))
    if b - a < hold_n + 2:
      out.append(float("nan")); continue
    final = tilt[b - hold_n:b].mean()
    ok = (rate[a:b] < 0.05) & (np.abs(tilt[a:b] - final) < 0.2)
    t = float("nan")
    for i in range(0, b - a - hold_n + 1):
      if ok[i:i + hold_n].all():
        t = i * DT; break
    out.append(t)
  return out


def summarize(rec, segs, gait, fell, vx):
  if len(rec["xy"]) < 10:
    return dict(fell=fell)
  xy, rp, w = np.array(rec["xy"]), np.array(rec["rp"]), np.array(rec["w"])
  qd = np.array(rec["qd_leg"]); feet = np.array(rec["feet"])
  w0 = int(1.0 / DT)
  base_rp = rp[w0:w0 + int(0.5 / DT)].mean(0)
  rp_dev = rp[w0:] - base_rp
  disp = xy[w0:] - xy[w0]
  extra = {}
  if vx > 0:  # walking: lateral drift only (in the start heading)
    drift = float(np.abs(disp[:, 1]).max())
    T = (len(xy) - w0) * DT
    extra = dict(speed=float(np.linalg.norm(disp[-1]) / max(T, 1e-6)),
                 heading_deg=float(np.degrees(np.arctan2(disp[-1, 1], disp[-1, 0]))))
  else:
    drift = float(np.linalg.norm(disp, axis=1).max())
  foot_disp = float(np.linalg.norm(feet[w0:] - feet[w0], axis=2).max()) if vx == 0 else float("nan")
  # Same full timeline for tilt and rate: segment indices count from tick 0.
  st = settling_times(rp - base_rp, w, segs) if vx == 0 else []
  b12_p = band_share(w[w0:], 1, 2); b38_p = band_share(w[w0:], 3, 8)
  b12_l = band_share(qd[w0:], 1, 2); b38_l = band_share(qd[w0:], 3, 8)
  margin = np.array(rec["margin"], float)
  ln = rec["leg_names"]
  col = lambda pat: [i for i, n in enumerate(ln) if pat in n]
  ww, qq = w[w0:], qd[w0:]
  per_axis = dict(
    # per axis, absolute (rad/s RMS in band) + share; w[:, 0] roll, w[:, 1] pitch
    pitch_1_2_rms=float(band_rms(ww[:, 1], 1, 2)), roll_1_2_rms=float(band_rms(ww[:, 0], 1, 2)),
    pitch_lt1_rms=float(band_rms(ww[:, 1], 0.2, 1)), roll_lt1_rms=float(band_rms(ww[:, 0], 0.2, 1)),
    pitch_1_2_share=float(band_share(ww[:, 1], 1, 2)), roll_1_2_share=float(band_share(ww[:, 0], 1, 2)),
    hip_pitch_1_2_rms=float(band_rms(qq[:, col("hip_pitch")], 1, 2).mean()),
    hip_roll_1_2_rms=float(band_rms(qq[:, col("hip_roll")], 1, 2).mean()),
    knee_1_2_rms=float(band_rms(qq[:, col("knee")], 1, 2).mean()),
    ankle_pitch_1_2_rms=float(band_rms(qq[:, col("ankle_pitch")], 1, 2).mean()),
  )
  return dict(
    **per_axis,
    fell=fell,
    # REAL steps only (>= 100 ms airborne); sub-100 ms touchdowns are contact
    # flicker (a pad unloading for a tick), reported separately.
    steps=int((np.array(gait.swing) >= 0.10).sum()) if vx == 0 else None,
    flicker=int((np.array(gait.swing) < 0.10).sum()) if vx == 0 else None,
    pelvis_drift_mm=1000 * drift,
    rp_rms_deg=float(np.sqrt((rp_dev ** 2).mean())),
    rp_peak_deg=float(np.abs(rp_dev).max()),
    w_rms=float(np.sqrt((w[w0:] ** 2).mean())),
    w_peak=float(np.abs(w[w0:]).max()),
    leg_qd_rms=float(np.sqrt((qd[w0:] ** 2).mean())),
    foot_disp_mm=1000 * foot_disp,
    upper_track_err=float(np.mean(rec["track"])),
    settle_s=st,
    settle_median_s=float(np.nanmedian(st)) if st and not np.all(np.isnan(st)) else float("nan"),
    band_1_2_pelvis=float(b12_p.mean()), band_3_8_pelvis=float(b38_p.mean()),
    band_1_2_legs=float(b12_l.mean()), band_3_8_legs=float(b38_l.mean()),
    cop_com_offset_mm=float(1000 * np.nanmean(np.linalg.norm(
      np.array(rec["cop"]) - np.array(rec["com"]), axis=1))),
    # p5 / median, not min: a single instant with a few near-collinear contact
    # points gives a degenerate hull and a meaningless extreme.
    support_margin_p5_mm=float(1000 * np.nanpercentile(margin, 5)) if np.isfinite(margin).any() else float("nan"),
    support_margin_median_mm=float(1000 * np.nanmedian(margin)) if np.isfinite(margin).any() else float("nan"),
    **extra,
  )


# ---------------------------------------------------------------------------
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--suite", default="primary",
                  choices=("primary", "sweep", "walking", "aggressive", "upper_all", "all"))
  ap.add_argument("--seeds", type=int, default=10)
  ap.add_argument("--out", default=None)
  ap.add_argument("--clips", default=None,
                  help="scripted clip npz (default data/clips_v2/scripted.npz)")
  ap.add_argument("--upper-set", choices=("ik", "all"), default="ik",
                  help="aggressive test: move the 14 deployed IK joints, or all non-leg "
                       "joints (waist roll/pitch stay home: zero range)")
  args = ap.parse_args()
  clips, cj, _ = load_clips(args.clips)
  pol = S.Policy(args.onnx)
  res = {}
  suites = (("primary", "sweep", "walking", "aggressive")
            + (("upper_all",) if "head_look" in clips else ())
            if args.suite == "all" else (args.suite,))

  if "primary" in suites:
    up, segs = build_upper_stream(pol, PRIMARY, 1.0, clips, cj)
    res["primary"] = run(args.onnx, up, segs)
  if "sweep" in suites:
    res["sweep"] = {}
    for amp in (0.25, 0.5, 0.75, 1.0):
      up, segs = build_upper_stream(pol, PRIMARY, amp, clips, cj)
      runs = [run(args.onnx, up, segs, seed=s) for s in range(args.seeds)]
      res["sweep"][str(amp)] = dict(
        fall_rate=float(np.mean([r.get("fell") is not None for r in runs])),
        step_prob=float(np.mean([(r.get("steps") or 0) > 0 for r in runs])),
        **{k: float(np.nanmean([r[k] for r in runs if k in r]))
           for k in ("rp_rms_deg", "w_rms", "leg_qd_rms", "pelvis_drift_mm")})
  if "walking" in suites:
    res["walking"] = {}
    for vx in (0.0, 0.3, 0.5, 1.0):
      for arms in ("moving", "home"):
        up, segs = build_upper_stream(pol, PRIMARY, 1.0 if arms == "moving" else 0.0, clips, cj)
        r = run(args.onnx, up, segs, vx=vx)
        res["walking"][f"{vx}" if arms == "moving" else f"{vx}_arms_home"] = r
  if "upper_all" in suites:
    # head look-around + left wrist roll: the joints v8 never trained
    seq = PRIMARY + ("head_look", "left_wrist_twist_1.2")
    up, segs = build_upper_stream(pol, seq, 1.0, clips, cj)
    res["upper_all"] = run(args.onnx, up, segs)
  if "aggressive" in suites:
    up, segs = aggressive_stream(pol, upper_set=args.upper_set)
    res["aggressive"] = run(args.onnx, up, segs, payload=2.0)

  print(json.dumps(res, indent=1, default=lambda x: None if x is None else float(x)))
  if args.out:
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
  main()
