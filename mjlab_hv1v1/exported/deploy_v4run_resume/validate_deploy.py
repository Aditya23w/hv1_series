#!/usr/bin/env python3
"""Drive the policy using ONLY this bundle, and check it still walks.

Nothing here reads mjlab_hv1v1 or the training config -- every policy parameter
(joint order, anchor, action scale, observation layout, history length) comes
from deploy.json. If this walks, the bundle is complete and this file is the
reference implementation to port to the robot.

MuJoCo is used only to supply physics and fake sensors in place of the robot.

  python validate_deploy.py              # walk at 1.0 m/s
  python validate_deploy.py --vx 0.0     # stand
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
    self.scale = np.array([self.cfg["action_scale"][n] if "action_scale" in self.cfg
                           else self.cfg["action"]["scale"][n] for n in self.jn])
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
    return self.anchor + self.prev_action * self.scale


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--vx", type=float, default=1.0)
  ap.add_argument("--vy", type=float, default=0.0)
  ap.add_argument("--wz", type=float, default=0.0)
  ap.add_argument("--seconds", type=float, default=8.0)
  args = ap.parse_args()

  pol = DeployedPolicy(HERE)
  print(f"bundle : {pol.cfg['meta']['run']}  ({pol.cfg['meta']['iterations']} iters)")
  print(f"         obs {pol.obs_dim}  actions {len(pol.jn)}  @ {1/pol.dt:.0f} Hz")

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
  for _ in range(n):
    rob.d.ctrl[:] = pol.step(
      ang_vel=rob.gyro(),
      proj_gravity=rob.root_rot().T @ GRAVITY_W,
      q=rob.qpos_joints(),
      qdot=rob.qvel_joints(),
      command=rob.command,
    )
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
  ok = not rob.fell()
  print(f"\n  {'BUNDLE OK - policy runs from deploy.json alone' if ok else 'FAILED - robot fell'}")
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
