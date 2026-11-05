#!/usr/bin/env python3
"""Offline upper-body clip library for the v8 decoupled task.

    hand-space target stream -> Mink ArmIK (the deployed teleop IK) -> Ruckig
    -> q / qdot / qddot at 50 Hz -> data/clips/<family>.npz

Nothing here runs inside RL training. The RL env only replays the stored
tensors (mjlab_hv1v2.mdp.UpperBodyMixedCommand).

Why each stage is the deployed one:
  * IK: merai_teleop.ik.ArmIK built by walk_teleop_config(load_config()) --
    the WALK_TELEOP contract from merai_policy.yaml: 14 IK joints (waist_yaw +
    arms, left_wrist_roll masked out of service), ranges = trained_range
    intersected with the hardware margins, IK speed capped at the slew limits.
    It solves in the PELVIS frame of the same MJCF the sim uses, so its joint
    coordinates are sim coordinates -- no sign/offset mapping.
  * Mink runs at 100 Hz like the teleop loop (two IK steps per 50 Hz tick).
  * Ruckig runs ONCE, online, at 50 Hz, exactly as the runtime will
    (merai_policy upper_body_input.hpp): every tick the latest IK output is the
    target position (target velocity/acceleration 0), per-joint independent
    limits (Synchronization.No -- the per-joint slew it replaces was
    independent too). Limits: v 2.0 / a 8 / j 40, waist_yaw 0.6 / 4 / 20.
    The C++ port must use the same mode for the parity test.

Families (one npz each):
  scripted   Type A, deterministic: left / right / both reaches, holds,
             returns, figure-8s, slow + fast, waist yaw. Also the evaluation
             sequences (eval_* clips).
  random     Type A, randomised in hand space.
  walking    Type B: world-fixed hand targets seen from a moving pelvis.

  .venv_clips/bin/python mjlab_hv1v2/scripts/gen_ik_clips.py --n-random 2000 --n-walking 600
  .venv_clips/bin/python mjlab_hv1v2/scripts/gen_ik_clips.py --v2   # v8 library

--v2 (v8, -> data/clips_v2): every upper-body joint except the locked waist
roll/pitch moves -- 17 columns:
  * the IK uses its full 15 joints (load_config().ik_joints), i.e. ALSO
    left_wrist_roll, which walk_teleop_config masks out of service on the robot;
  * head_yaw / head_pitch are not IK joints (and are virtual on the robot): an
    independent gaze stream -- random look targets held 0.5-3 s, through the
    same online Ruckig -- in ~60% of clips, home in the rest.
v8 trained with these three joints frozen and fell the moment one moved (its
normalizer had a 0.006 rad spread for them); v8d trains them.

Run with the clip venv (Python 3.12, mink, ruckig, merai_teleop), not the
training env: MERAI_TELEOP_MODEL / meshdir are pointed at this repo's MJCF.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MERAI_TELEOP_MODEL", str(ROOT / "mjcf/hv1v2.xml"))

import mink  # noqa: E402
from ruckig import InputParameter, OutputParameter, Ruckig, Synchronization  # noqa: E402

from merai_teleop.config import load_config, walk_teleop_config  # noqa: E402
from merai_teleop.ik import ArmIK  # noqa: E402
from merai_teleop.model import build_ik_model  # noqa: E402

OUT = ROOT / "mjlab_hv1v2/data/clips"
OUT_V2 = ROOT / "mjlab_hv1v2/data/clips_v2"
# Head gaze envelope = UPPER_BODY_SAMPLE_RANGE in hv1v2_constants.py (this venv
# does not import mjlab_hv1v2); the MJCF home of both head joints is 0.
HEAD = {"head_yaw_joint": (-1.0, 1.0), "head_pitch_joint": (-0.6, 0.4)}
DT = 0.02          # 50 Hz policy / Ruckig tick
IK_SUB = 2         # 100 Hz Mink, like teleop
RUCKIG = {"default": (2.0, 8.0, 40.0), "waist_yaw_joint": (0.6, 4.0, 20.0)}


# ---------------------------------------------------------------------------
class Pipeline:
  def __init__(self, v2: bool = False):
    base = load_config()
    cfg = walk_teleop_config(base)
    if v2:   # un-mask left_wrist_roll: the full 15-joint IK
      cfg = replace(cfg, ik_joints=list(base.ik_joints))
    cfg = replace(cfg, meshdir=ROOT / "meshes")
    self.v2 = v2
    self.cfg = cfg
    self.ik_model = build_ik_model(cfg)
    self.joints = list(self.ik_model.joints)
    self.home = np.array([cfg.ready_pose[cfg.index(n)] for n in self.joints])
    self.lo, self.hi = self.ik_model.lower.copy(), self.ik_model.upper.copy()
    self.ik = ArmIK(self.ik_model, cfg.section("ik"), q_posture=self.home)
    self.home_pose = {s: self.ik.hand_pose(s) for s in ("right", "left")}
    self.wy = self.joints.index("waist_yaw_joint")
    self.head_joints = list(HEAD) if v2 else []
    self.out_joints = self.joints + self.head_joints
    self.out_home = np.concatenate([self.home, np.zeros(len(self.head_joints))])
    self.head_lo = np.array([HEAD[n][0] for n in self.head_joints])
    self.head_hi = np.array([HEAD[n][1] for n in self.head_joints])

  def head_stream(self, n: int, rng, p_move: float = 0.6) -> np.ndarray:
    """Random gaze: piecewise-constant look targets (Ruckig smooths them)."""
    t = np.zeros((n, len(self.head_joints)))
    if not self.v2 or rng.random() >= p_move:
      return t
    k = int(rng.uniform(0.3, 1.0) / DT)          # start looking at home
    while k < n:
      hold = int(rng.uniform(0.5, 3.0) / DT)
      t[k:k + hold] = rng.uniform(self.head_lo, self.head_hi)
      k += hold
    t[-int(0.6 / DT):] = 0.0                       # end at home
    return t

  def hand_home(self, side: str) -> mink.SE3:
    return self.home_pose[side]

  def run(self, targets, waist=None, head=None) -> dict:
    """targets: list over 50 Hz ticks of {side: SE3 in pelvis frame}.
    waist: optional per-tick waist-yaw posture target (rad).
    head: v2 only, (n, 2) head yaw/pitch targets (default: home)."""
    ik = self.ik
    ik.reset(self.home, q_posture=self.home)
    n = len(targets)
    q_ik = np.empty((n, len(self.joints)))
    fails = 0
    for k, tgt in enumerate(targets):
      for side, pose in tgt.items():
        ik.set_target(side, pose)
      if waist is not None:
        post = self.home.copy()
        post[self.wy] = waist[k]
        ik.posture.set_target(np.clip(post, self.lo, self.hi))
      for _ in range(IK_SUB):
        fails += 0 if ik.step(DT / IK_SUB).ok else 1
      q_ik[k] = ik.q
    q, qd, qdd = ruckig_online(q_ik, self.joints, self.home, self.lo, self.hi)
    if self.v2:
      h = np.zeros((n, len(self.head_joints))) if head is None else head
      hq, hqd, hqdd = ruckig_online(h, self.head_joints, np.zeros(len(self.head_joints)),
                                    self.head_lo, self.head_hi)
      q, qd, qdd = (np.concatenate(p, axis=1) for p in ((q, hq), (qd, hqd), (qdd, hqdd)))
    return dict(q=q, qd=qd, qdd=qdd, ik_fail_frac=fails / max(n * IK_SUB, 1))


def ruckig_online(q_target: np.ndarray, joints, q0, lo, hi):
  """Online Ruckig at 50 Hz: new target every tick, independent per joint."""
  dof = len(joints)
  otg = Ruckig(dof, DT)
  inp, out = InputParameter(dof), OutputParameter(dof)
  lim = np.array([RUCKIG.get(n, RUCKIG["default"]) for n in joints])
  inp.max_velocity, inp.max_acceleration, inp.max_jerk = (
    lim[:, 0].tolist(), lim[:, 1].tolist(), lim[:, 2].tolist())
  inp.synchronization = Synchronization.No
  inp.current_position = q0.tolist()
  inp.current_velocity = [0.0] * dof
  inp.current_acceleration = [0.0] * dof
  n = len(q_target)
  q, qd, qdd = (np.empty((n, dof)) for _ in range(3))
  for k in range(n):
    inp.target_position = np.clip(q_target[k], lo, hi).tolist()   # runtime clamp
    inp.target_velocity = [0.0] * dof
    inp.target_acceleration = [0.0] * dof
    otg.update(inp, out)
    q[k], qd[k], qdd[k] = out.new_position, out.new_velocity, out.new_acceleration
    out.pass_to_input(inp)
  return q, qd, qdd


# ---------------------------------------------------------------------------
# hand-space target streams (pelvis frame, relative to each hand's READY pose)
def min_jerk(s):
  s = np.clip(s, 0.0, 1.0)
  return 10 * s**3 - 15 * s**4 + 6 * s**5


def offset_pose(base: mink.SE3, dp, drot=(0.0, 0.0, 0.0)) -> mink.SE3:
  t = mink.SE3.from_translation(np.asarray(dp, float))
  r = mink.SE3.from_rotation(mink.SO3.exp(np.asarray(drot, float)))
  return t @ base @ r


def reach_segments(P: Pipeline, sides, dp, drot, t_reach, t_hold, t_home=0.6):
  """home -> reach -> hold -> return -> home, min-jerk in hand space."""
  ticks = []
  def seg(alpha_fn, dur):
    for k in range(max(int(round(dur / DT)), 1)):
      a = alpha_fn((k + 1) * DT / dur)
      ticks.append({s: offset_pose(P.hand_home(s), np.array(dp[s]) * a,
                                   np.array(drot[s]) * a) for s in sides})
  seg(lambda s: 0.0, t_home)
  seg(min_jerk, t_reach)
  seg(lambda s: 1.0, t_hold)
  seg(lambda s: 1.0 - min_jerk(s), t_reach)
  seg(lambda s: 0.0, t_home)
  return ticks


def figure8(P: Pipeline, sides, center, rx, rz, period, cycles):
  ticks = []
  n_in = int(0.8 / DT)
  for k in range(n_in):                    # ease into the start point
    a = min_jerk((k + 1) / n_in)
    ticks.append({s: offset_pose(P.hand_home(s), np.array(center[s]) * a) for s in sides})
  n = int(period * cycles / DT)
  for k in range(n):
    th = 2 * np.pi * k * DT / period
    ticks.append({s: offset_pose(P.hand_home(s), np.array(center[s]) + np.array(
      [0.0, rx * np.sin(th), rz * np.sin(2 * th)])) for s in sides})
  for k in range(n_in):
    a = 1.0 - min_jerk((k + 1) / n_in)
    ticks.append({s: offset_pose(P.hand_home(s), np.array(center[s]) * a) for s in sides})
  return ticks


SIDE_Y = {"right": -1.0, "left": 1.0}   # outward sign in the pelvis frame


def scripted(P: Pipeline):
  """Deterministic Type A set; also the evaluation building blocks."""
  clips = []
  fwd = {"right": [0.25, -0.05, 0.15], "left": [0.25, 0.05, 0.15]}
  side_out = {s: [0.05, 0.20 * SIDE_Y[s], 0.10] for s in SIDE_Y}
  up = {s: [0.15, 0.0, 0.30] for s in SIDE_Y}
  zero_rot = {s: [0.0, 0.0, 0.0] for s in SIDE_Y}
  for tag, dp in (("fwd", fwd), ("side", side_out), ("up", up)):
    for sides, sname in ((("left",), "left"), (("right",), "right"),
                         (("right", "left"), "both")):
      for speed, t_reach in (("slow", 2.0), ("fast", 0.7)):
        clips.append((f"{sname}_{tag}_{speed}", sides, dict(dp=dp, drot=zero_rot,
                      t_reach=t_reach, t_hold=1.5), "reach"))
  clips.append(("both_fig8", ("right", "left"), dict(
    center={s: [0.20, 0.0, 0.10] for s in SIDE_Y}, rx=0.08, rz=0.06,
    period=2.5, cycles=2), "fig8"))
  out = []
  for name, sides, kw, kind in clips:
    ticks = (reach_segments(P, sides, **kw) if kind == "reach"
             else figure8(P, sides, **kw))
    out.append((name, P.run(ticks), dict(walking=False)))
  # waist yaw: arms at home, torso turns and returns
  n_hold = int(0.6 / DT)
  for amp, t in ((0.4, 1.5), (0.8, 2.5)):
    ticks = [{} for _ in range(int(2 * t / DT) + 2 * n_hold)]
    w = np.concatenate([np.zeros(n_hold),
                        amp * min_jerk(np.linspace(0, 1, int(t / DT))),
                        amp * (1 - min_jerk(np.linspace(0, 1, int(t / DT)))),
                        np.zeros(n_hold)])
    for sgn in (1, -1):
      out.append((f"waist_yaw_{'+' if sgn > 0 else '-'}{amp}",
                  P.run([{} for _ in w], waist=sgn * w), dict(walking=False)))
  if P.v2:
    # head look-around: arms home, gaze left / right / up / down / home
    seq = [(0.0, 0.0), (0.8, 0.0), (-0.8, 0.0), (0.0, 0.3), (0.0, -0.5), (0.6, -0.4), (0.0, 0.0)]
    h = np.concatenate([np.tile(np.array(a), (int(1.2 / DT), 1)) for a in seq])
    out.append(("head_look", P.run([{} for _ in h], head=h), dict(walking=False)))
    # left wrist twist: left hand rotates about its own axis (IK uses wrist roll)
    for amp in (0.6, 1.2):
      ticks = reach_segments(P, ("left",), {"left": [0.10, 0.05, 0.05]},
                             {"left": [amp, 0.0, 0.0]}, t_reach=1.2, t_hold=1.0)
      out.append((f"left_wrist_twist_{amp}", P.run(ticks), dict(walking=False)))
  return out


def randomized(P: Pipeline, n: int, rng):
  out = []
  for i in range(n):
    two = rng.random() < 0.4
    sides = ("right", "left") if two else (rng.choice(["right", "left"]),)
    dp, drot = {}, {}
    for s in ("right", "left"):
      dp[s] = [rng.uniform(0.0, 0.35), SIDE_Y[s] * rng.uniform(-0.10, 0.25),
               rng.uniform(-0.10, 0.35)]
      drot[s] = list(rng.uniform(-0.6, 0.6, 3))
    amp = rng.uniform(0.3, 1.0)
    dp = {s: list(np.array(v) * amp) for s, v in dp.items()}
    t_reach = rng.uniform(0.5, 3.0)
    t_hold = rng.uniform(0.0, 3.0)
    ticks = reach_segments(P, sides, dp, drot, t_reach, t_hold,
                           t_home=rng.uniform(0.3, 1.0))
    if two and rng.random() < 0.5:        # left/right phase offset
      lag = int(rng.uniform(0.1, 0.8) / DT)
      left = [t.get("left") for t in ticks]
      left = [P.hand_home("left")] * lag + left[:-lag]
      ticks = [{**t, "left": l} for t, l in zip(ticks, left)]
    waist = None
    if rng.random() < 0.4:
      wa = rng.uniform(-0.8, 0.8)
      m = len(ticks)
      prof = np.concatenate([min_jerk(np.linspace(0, 1, m // 2)),
                             1 - min_jerk(np.linspace(0, 1, m - m // 2))])
      waist = wa * prof
    out.append((f"random_{i:05d}", P.run(ticks, waist=waist,
                                          head=P.head_stream(len(ticks), rng)),
                dict(walking=False, amp=amp, t_reach=t_reach)))
  return out


def walking(P: Pipeline, n: int, rng):
  """Type B: world-fixed hand target, pelvis moving at vx / yaw-rate."""
  out = []
  for i in range(n):
    vx = rng.uniform(0.3, 0.8)
    wz = rng.uniform(-0.3, 0.3)
    side = rng.choice(["right", "left"])
    sides = ("right", "left") if rng.random() < 0.3 else (side,)
    t_walk0, t_reach, t_hold, t_ret = (rng.uniform(0.5, 1.5), rng.uniform(0.6, 1.5),
                                       rng.uniform(0.3, 1.5), rng.uniform(0.6, 1.5))
    total = t_walk0 + t_reach + t_hold + t_ret + 0.6
    n_t = int(total / DT)
    # pelvis pose over time (world frame, planar)
    ts = np.arange(n_t) * DT
    yaw = wz * ts
    xy = np.zeros((n_t, 2))
    for k in range(1, n_t):
      xy[k] = xy[k - 1] + vx * DT * np.array([np.cos(yaw[k - 1]), np.sin(yaw[k - 1])])
    def T_pelvis(k):
      return mink.SE3.from_rotation_and_translation(
        mink.SO3.from_z_radians(yaw[k]), np.array([xy[k, 0], xy[k, 1], 0.0]))
    # world target: where a forward reach would put the hand at grasp time
    k_grasp = int((t_walk0 + t_reach) / DT)
    reach = {s: [rng.uniform(0.15, 0.35), SIDE_Y[s] * rng.uniform(0.0, 0.2),
                 rng.uniform(0.0, 0.25)] for s in ("right", "left")}
    world = {s: T_pelvis(k_grasp) @ offset_pose(P.hand_home(s), reach[s]) for s in sides}
    ticks = []
    for k in range(n_t):
      t = k * DT
      if t < t_walk0:
        a = 0.0
      elif t < t_walk0 + t_reach:
        a = min_jerk((t - t_walk0) / t_reach)
      elif t < t_walk0 + t_reach + t_hold:
        a = 1.0
      else:
        a = 1.0 - min_jerk((t - t_walk0 - t_reach - t_hold) / t_ret)
      tgt = {}
      for s in sides:
        seen = T_pelvis(k).inverse() @ world[s]       # world target, pelvis frame
        home = P.hand_home(s)
        # blend home -> seen target (translation), keep orientation from seen
        p = (1 - a) * home.translation() + a * seen.translation()
        tgt[s] = mink.SE3.from_rotation_and_translation(seen.rotation(), p)
      ticks.append(tgt)
    out.append((f"walking_{i:05d}", P.run(ticks, head=P.head_stream(len(ticks), rng)),
                dict(walking=True, vx=vx, wz=wz)))
  return out


def save(family: str, clips, P: Pipeline):
  out_dir = OUT_V2 if P.v2 else OUT
  out_dir.mkdir(parents=True, exist_ok=True)
  qs = [c[1]["q"] for c in clips]
  starts = np.cumsum([0] + [len(q) for q in qs])[:-1]
  meta = [c[2] for c in clips]
  peak_v = np.array([np.abs(c[1]["qd"]).max() for c in clips])
  np.savez_compressed(
    out_dir / f"{family}.npz",
    q=np.concatenate(qs).astype(np.float32),
    qd=np.concatenate([c[1]["qd"] for c in clips]).astype(np.float32),
    qdd=np.concatenate([c[1]["qdd"] for c in clips]).astype(np.float32),
    start=starts.astype(np.int64),
    length=np.array([len(q) for q in qs], dtype=np.int64),
    names=np.array([c[0] for c in clips]),
    walking=np.array([m.get("walking", False) for m in meta]),
    amplitude=np.array([m.get("amp", 1.0) for m in meta], dtype=np.float32),
    peak_speed=peak_v.astype(np.float32),
    ik_fail_frac=np.array([c[1]["ik_fail_frac"] for c in clips], dtype=np.float32),
    joint_names=np.array(P.out_joints),
    home=P.out_home.astype(np.float32),
    dt=np.float32(DT),
    family=np.array(family),
  )
  dur = sum(len(q) for q in qs) * DT
  print(f"  {family:9s} {len(clips):5d} clips  {dur/60:6.1f} min  "
        f"peak |qd| max {peak_v.max():.2f} rad/s  "
        f"IK fail {np.mean([c[1]['ik_fail_frac'] for c in clips]):.3%}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n-random", type=int, default=2000)
  ap.add_argument("--n-walking", type=int, default=600)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--only", default=None, help="scripted | random | walking")
  ap.add_argument("--v2", action="store_true",
                  help="v8 library: 15-joint IK (+left_wrist_roll) + head gaze -> clips_v2")
  args = ap.parse_args()
  rng = np.random.default_rng(args.seed)
  P = Pipeline(v2=args.v2)
  print(f"clip joints ({len(P.out_joints)}): {P.out_joints}")
  print(f"writing {OUT_V2 if args.v2 else OUT}")
  if args.only in (None, "scripted"):
    save("scripted", scripted(P), P)
  if args.only in (None, "random"):
    save("random", randomized(P, args.n_random, rng), P)
  if args.only in (None, "walking"):
    save("walking", walking(P, args.n_walking, rng), P)


if __name__ == "__main__":
  main()
