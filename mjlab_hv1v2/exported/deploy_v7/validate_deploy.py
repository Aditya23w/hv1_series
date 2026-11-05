#!/usr/bin/env python3
"""Drive the policy using ONLY this bundle, and check it still walks.

Nothing here reads mjlab_hv1v2 or the training config -- every policy parameter
(joint order, anchor, action scale, observation layout, history length) comes
from deploy.json. If this walks, the bundle is complete and this file is the
reference implementation to port to the robot.

MuJoCo is used only to supply physics and fake sensors in place of the robot.

  python validate_deploy.py              # walk at 1.0 m/s
  python validate_deploy.py --vx 0.0     # stand
  python validate_deploy.py --vx 0.0 --wave 1.0   # stand while the arms sweep

Decoupled bundles (an `upper_body` section in deploy.json): the policy drives
only `action.joint_names`; this script plays the IK controller for the rest --
holding `upper_body.home`, or with --wave sweeping inside
`upper_body.trained_range` at <= max_slew_rad_s, exactly the limits deploy.json
tells the real IK to respect.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
GRAVITY_W = np.array([0.0, 0.0, -1.0])


class DeployedPolicy:
  """Exactly what the robot's runtime has to implement."""

  def __init__(self, bundle_dir: Path):
    self.cfg = json.loads((bundle_dir / "deploy.json").read_text())
    self.jn = [j["name"] for j in self.cfg["joints"]]
    self.anchor = np.array([self.cfg["anchor"]["default_joint_pos"][n] for n in self.jn])
    # The joints the policy drives. Older bundles act on every joint.
    act = self.cfg.get("action", {})
    self.aj = act.get("joint_names", self.jn)
    self.aidx = np.array([self.jn.index(n) for n in self.aj])
    self.scale = np.array([self.cfg["action_scale"][n] if "action_scale" in self.cfg
                           else act["scale"][n] for n in self.aj])
    self.terms = self.cfg["policy"]["observation"]
    self.dt = self.cfg["control"]["control_dt"]
    self.sess = ort.InferenceSession(str(bundle_dir / self.cfg["policy"]["file"]),
                                     providers=["CPUExecutionProvider"])
    self.in_name = self.cfg["policy"]["input_name"]
    self.obs_dim = self.cfg["policy"]["obs_dim"]
    self.prev_action = np.zeros(self.cfg["policy"]["action_dim"])
    self.hist: dict[str, deque] = {}

  def observe(self, *, ang_vel, proj_gravity, q, qdot, command) -> np.ndarray:
    frames = {
      "base_ang_vel": np.asarray(ang_vel, float),
      "projected_gravity": np.asarray(proj_gravity, float),
      "joint_pos": np.asarray(q, float) - self.anchor,
      "joint_vel": np.asarray(qdot, float),
      "actions": self.prev_action,
      "command": np.asarray(command, float),
    }
    out = []
    for t in self.terms:
      f = frames[t["name"]]
      assert f.size == t["dim"], f"{t['name']}: got {f.size}, spec says {t['dim']}"
      h = t["history"]
      if h <= 1:
        out.append(f)
        continue
      buf = self.hist.get(t["name"])
      if buf is None:                       # backfill, never zero-fill
        buf = deque([f.copy()] * h, maxlen=h)
        self.hist[t["name"]] = buf
      else:
        buf.append(f)
      out.append(np.concatenate(list(buf)))  # oldest -> newest
    obs = np.concatenate(out)
    assert obs.size == self.obs_dim, f"assembled {obs.size}, graph takes {self.obs_dim}"
    return obs

  def step(self, **sensors) -> np.ndarray:
    obs = self.observe(**sensors)
    self.prev_action = self.sess.run(
      None, {self.in_name: obs.astype(np.float32)[None]})[0][0].astype(float)
    target = self.anchor.copy()  # joints outside the action: caller overwrites
    target[self.aidx] += self.prev_action * self.scale
    return target


class UpperBodyStandIn:
  """Plays the IK controller using ONLY deploy.json's upper_body section."""

  def __init__(self, cfg: dict, jn: list[str], wave: float):
    ub = cfg["upper_body"]
    self.idx = np.array([jn.index(n) for n in ub["joints"]])
    self.home = np.array([ub["home"][n] for n in ub["joints"]])
    r = np.array([ub["trained_range"][n] for n in ub["joints"]])
    self.lo, self.hi = r[:, 0], r[:, 1]
    self.slew = ub["trained_disturbance"]["max_slew_rad_s"]
    self.wave = wave
    self.phase = np.random.default_rng(0).uniform(0, 2 * np.pi, len(self.idx))
    self.target = self.home.copy()
    self.t = 0.0

  def step(self, dt: float) -> np.ndarray:
    self.t += dt
    u = self.lo + (self.hi - self.lo) * (
      0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * self.t + self.phase))
    goal = self.home + self.wave * (u - self.home)
    goal = np.clip(goal, self.lo, self.hi)               # IK must clamp ...
    lim = self.slew * dt                                  # ... and rate-limit
    self.target += np.clip(goal - self.target, -lim, lim)
    return self.target


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--vx", type=float, default=1.0)
  ap.add_argument("--vy", type=float, default=0.0)
  ap.add_argument("--wz", type=float, default=0.0)
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--wave", type=float, default=0.0,
                  help="decoupled bundles: upper-body sweep, 0..1 of trained_range")
  args = ap.parse_args()

  pol = DeployedPolicy(HERE)
  print(f"bundle : {pol.cfg['meta']['run']}  ({pol.cfg['meta']['iterations']} iters)")
  print(f"         obs {pol.obs_dim}  actions {len(pol.aj)}  @ {1/pol.dt:.0f} Hz")
  upper = (UpperBodyStandIn(pol.cfg, pol.jn, args.wave)
           if "upper_body" in pol.cfg else None)
  if upper is not None:
    print(f"         decoupled: upper body {len(upper.idx)} joints "
          f"{'held at home' if not args.wave else f'sweeping at {args.wave}'}")

  # --- physics stand-in for the robot --------------------------------------
  import sys
  sys.path.insert(0, str(HERE.parents[1] / "scripts"))
  from sim2sim import Robot  # physics + fake sensors only
  rob = Robot(pol.jn, for_video=False)
  rob.command[:] = [args.vx, args.vy, args.wz]
  rob.reset()

  dec = pol.cfg["control"]["decimation"]
  n = int(args.seconds / pol.dt)
  lifts = []
  feet = [mujoco.mj_name2id(rob.m, mujoco.mjtObj.mjOBJ_SITE, s)
          for s in ("left_foot", "right_foot")]
  feet0 = None
  for i in range(n):
    if i == int(1.0 / pol.dt):
      feet0 = rob.d.site_xpos[feet][:, :2].copy()
    ctrl = pol.step(
      ang_vel=rob.gyro(),
      proj_gravity=rob.root_rot().T @ GRAVITY_W,
      q=rob.qpos_joints(),
      qdot=rob.qvel_joints(),
      command=rob.command,
    )
    if upper is not None:
      ctrl[upper.idx] = upper.step(pol.dt)
    rob.d.ctrl[:] = ctrl
    for _ in range(dec):
      mujoco.mj_step(rob.m, rob.d)
    lifts.append(min(float(rob.d.geom_xpos[g][2]) for g in range(rob.m.ngeom)
                     if (nm := mujoco.mj_id2name(rob.m, mujoco.mjtObj.mjOBJ_GEOM, g))
                     and "foot" in nm and nm.endswith("_collision")))

  dist = float(np.linalg.norm(rob.d.qpos[:2]))
  print(f"\ncommand vx {args.vx} vy {args.vy} wz {args.wz}, {args.seconds:.0f} s")
  print(f"  travelled  {dist:.2f} m  ->  {dist/args.seconds:.2f} m/s")
  print(f"  pelvis z   {rob.d.qpos[2]:.4f}   fell: {'YES' if rob.fell() else 'no'}")
  print(f"  foot lift  {1000*(max(lifts)-min(lifts)):.1f} mm")
  if feet0 is not None and not any((args.vx, args.vy, args.wz)):
    drift = np.linalg.norm(rob.d.site_xpos[feet][:, :2] - feet0, axis=1) * 1000
    print(f"  foot drift L {drift[0]:.1f} mm  R {drift[1]:.1f} mm  (standing, after 1 s)")
  ok = not rob.fell()
  print(f"\n  {'BUNDLE OK - policy runs from deploy.json alone' if ok else 'FAILED - robot fell'}")
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
