#!/usr/bin/env python3
"""Build a self-contained deployment bundle for a trained HV1V1 policy.

Everything a runtime needs to drive the robot: the ONNX graph, the joint order,
the anchor pose its actions offset from, the READY pose to servo into before
handing over, the PD gains, the action scales, the observation layout, and the
limits. Nothing here is hand-typed -- it is read from the compiled model, the
ONNX metadata and the source URDF, so the bundle cannot drift from what was
trained.

  python mjlab_hv1v1/scripts/export_deploy.py --run v4
  python mjlab_hv1v1/scripts/export_deploy.py --run v4 --out /tmp/deploy_v4
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = Path(__file__).resolve().parents[1] / "logs/rsl_rl/hv1v1_velocity"
# HV1V1's own export. NOT ROOT/"urdf/..." -- that directory holds HV1V2's URDF,
# and this line was previously HV1V2's path with the robot name string-replaced,
# which pointed at a file that has never existed. The directory is called
# URDF_HV1V2_26092026 even though it is the OLD-FOOT robot; the foot geometry
# identifies it, not the filename.
URDF = ROOT / "URDF_HV1V2_26092026/urdf/URDF_HV1V2_26092026.urdf"
assert URDF.exists(), f"HV1V1 URDF not found at {URDF}"

SIM_DT = 0.005
DECIMATION = 4

# Where each observation term comes from on the robot. This is the part a
# runtime cannot infer from the graph.
OBS_SOURCE = {
  "base_ang_vel": "imu:angular_velocity (pelvis frame, rad/s)",
  "base_lin_vel": "state_estimator:base_linear_velocity (pelvis frame, m/s)",
  "projected_gravity": "imu:attitude -> R_pelvis^T @ [0,0,-1] (unit vector)",
  "joint_pos": "encoders: q - anchor.default_joint_pos (rad)",
  "joint_vel": "encoders: qdot (rad/s)",
  "actions": "previous RAW policy output (before action_scale)",
  "command": "operator twist [vx, vy, wz]",
  "height_scan": "terrain perception - NOT AVAILABLE on hardware",
}

# Four-bar amplification MEASURED ON H1-V1, not on HV1V1. Reference only.
H1V1_FOURBAR_REFERENCE = {
  "ankle_pitch": 2.41, "ankle_roll": 1.23, "waist_roll": 4.00, "waist_pitch": 4.81,
}


def find_run(run: str) -> Path:
  hits = sorted(LOG_ROOT.glob(f"*_{run}"))
  if not hits:
    raise SystemExit(f"no run {run!r} under {LOG_ROOT}")
  return hits[-1]


def urdf_limits() -> dict:
  r = ET.parse(URDF).getroot()
  out = {}
  for j in r.findall("joint"):
    lim = j.find("limit")
    if lim is None:
      continue
    out[j.get("name")] = dict(
      lower=float(lim.get("lower")), upper=float(lim.get("upper")),
      velocity=float(lim.get("velocity")),
    )
  return out


def sha256(p: Path) -> str:
  return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default="v4")
  ap.add_argument("--out", default=None)
  args = ap.parse_args()

  run_dir = find_run(args.run)
  onnx_src = sorted(run_dir.glob("*.onnx"))[-1]
  out = Path(args.out) if args.out else ROOT / "mjlab_hv1v1/exported" / f"deploy_{args.run}"
  out.mkdir(parents=True, exist_ok=True)

  # ---- read the policy ----------------------------------------------------
  g = onnx.load(str(onnx_src))
  meta = {p.key: p.value for p in g.metadata_props}
  jn = meta["joint_names"].split(",")
  fl = lambda k: [float(x) for x in meta[k].split(",")]  # noqa: E731
  anchor = fl("default_joint_pos")
  scale = fl("action_scale")
  kp, kd = fl("joint_stiffness"), fl("joint_damping")
  obs_names = meta["observation_names"].split(",")
  obs_hist = [int(x) for x in fl("observation_terms_history_length")]
  obs_dim_in = g.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
  act_dim = g.graph.output[0].type.tensor_type.shape.dim[-1].dim_value

  # ---- read the model the policy was trained against ----------------------
  from mjlab.entity.entity import Entity
  from mjlab.utils.string import resolve_expr
  from mjlab_hv1v1.hv1v1_constants import (
    ARMATURE_PROVENANCE, CONTINUOUS_TORQUE, HOME_KEYFRAME,
    get_hv1v1_robot_cfg, resolve_joint_groups,
  )

  m = Entity(get_hv1v1_robot_cfg()).spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "init_state"))
  mujoco.mj_forward(m, d)

  ctrl_joint = [
    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
    for a in range(m.nu)
  ]
  assert ctrl_joint == jn, "model actuator order != policy joint order"

  # The ONNX metadata prints default_joint_pos to 3 decimals, which is a 0.026
  # deg error at worst. Harmless for control, but the compiled init_state
  # keyframe is the ground truth and costs nothing, so prefer it -- and assert
  # they agree, which catches a bundle built against the wrong checkpoint.
  anchor_exact = [
    float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]])
    for n in jn
  ]
  drift = max(abs(a - b) for a, b in zip(anchor_exact, anchor))
  assert drift < 1e-3, (
    f"anchor disagrees with the model keyframe by {drift:.5f} rad -- the bundle "
    "is being built against a different init pose than the policy trained with"
  )
  anchor = anchor_exact

  groups = resolve_joint_groups(tuple(jn))
  cont = dict(zip(jn, resolve_expr(CONTINUOUS_TORQUE, tuple(jn), None)))
  ulim = urdf_limits()
  effort = {n: float(m.actuator_forcerange[a, 1]) for a, n in enumerate(ctrl_joint)}

  # per-observation-term dims, in the policy's own order
  term_dim = {"base_ang_vel": 3, "base_lin_vel": 3, "projected_gravity": 3,
              "joint_pos": len(jn), "joint_vel": len(jn), "actions": len(jn),
              "command": 3}
  obs_terms = []
  for name, h in zip(obs_names, obs_hist):
    dim = term_dim.get(name)
    assert dim is not None, f"unknown observation term {name!r}; add it to term_dim"
    obs_terms.append(dict(name=name, dim=dim, history=max(h, 1),
                          contributes=dim * max(h, 1), source=OBS_SOURCE.get(name, "?")))
  total = sum(t["contributes"] for t in obs_terms)
  assert total == obs_dim_in, f"observation spec sums to {total}, graph takes {obs_dim_in}"

  # ---- READY pose: is the trained anchor safe to servo into? --------------
  pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot{i}_collision")
          for s in ("left", "right") for i in range(1, 5)]
  xs = []
  for gi in pads:
    half, R, p = m.geom_size[gi], d.geom_xmat[gi].reshape(3, 3), d.geom_xpos[gi]
    for sx in (-1, 1):
      for sy in (-1, 1):
        xs.append(float((p + R @ (np.array([sx, sy, -1]) * half))[0]))
  sole_c = 0.5 * (min(xs) + max(xs))
  com = d.subtree_com[1]
  com_off_mm = float((com[0] - sole_c) * 1000)

  # ---- assemble -----------------------------------------------------------
  try:
    sha = subprocess.check_output(
      ["git", "-C", str(ROOT / "mjlab_src"), "rev-parse", "--short", "HEAD"],
      text=True).strip()
  except Exception:
    sha = "unknown"
  ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))

  bundle = {
    "meta": {
      "robot": "HV1V1", "run": run_dir.name, "task": "Mjlab-Velocity-Flat-HV1V1",
      "iterations": int(ckpts[-1].stem.split("_")[1]) if ckpts else None,
      "exported_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
      "mjlab_commit": sha, "mujoco": mujoco.__version__,
      "total_mass_kg": float(sum(m.body_mass)),
      "armature_provenance": ARMATURE_PROVENANCE,
    },
    "control": {
      "sim_timestep": SIM_DT, "decimation": DECIMATION,
      "control_dt": SIM_DT * DECIMATION, "control_hz": 1.0 / (SIM_DT * DECIMATION),
      "root_body": "pelvis",
    },
    "policy": {
      "file": "policy.onnx", "input_name": g.graph.input[0].name,
      "output_name": g.graph.output[0].name,
      "obs_dim": obs_dim_in, "action_dim": act_dim,
      "observation_layout": "term-major, oldest-first; each term's full history "
                            "is contiguous, then terms concatenate in this order. "
                            "The LAST dim values of each term's slice are the "
                            "current frame.",
      "history_init": "backfill every slot with the first reading at t=0; do NOT zero-fill",
      "observation": obs_terms,
    },
    "anchor": {
      "note": "action mapping is  target[i] = anchor[i] + raw_action[i] * action_scale[i]",
      "default_joint_pos": {n: round(v, 6) for n, v in zip(jn, anchor)},
      "pelvis_height_m": round(float(HOME_KEYFRAME.pos[2]), 5),
    },
    "ready_pose": {
      "note": "Servo here with joint PD before handing over to the policy. "
              "Identical to the anchor. NOTE this is NOT the sole-centre rule "
              "HV1V2 uses: HV1V1's ankle sits 22.0% along its sole instead of "
              "36.5%, so centring the CoM would put it 78.7 mm ahead of the "
              "ankle and the pose diverges open-loop in 0.89 s. This pose is at "
              "34.4% of the sole (+34.7 mm), which measured the slowest "
              "open-loop creep of any candidate, -0.21 mm/s against HV1V2's "
              "-0.83 mm/s.",
      "joint_pos": {n: round(v, 6) for n, v in zip(jn, anchor)},
      "pelvis_height_m": round(float(HOME_KEYFRAME.pos[2]), 5),
      "com_offset_from_sole_centre_mm": round(com_off_mm, 2),
      "feet": "flat, stance width 0.24304 m",
      "WARNING": "This pose does NOT stand open-loop. Holding it with PD and no "
                 "policy the robot topples: ankle stiffness is ~95% of the m*g*h "
                 "the inverted pendulum demands. Measured hold is 6.2 s to a 50 mm "
                 "drop (HV1V2's own anchor manages 3.9 s), so this is NOT a "
                 "standing pose for either robot. Keep the policy running.",
    },
    "gains": {
      "note": "Simulation joint-space PD, exactly what the policy was trained "
              "against. See four_bar_warning before writing these to hardware.",
      "kp": {n: round(v, 3) for n, v in zip(jn, kp)},
      "kd": {n: round(v, 3) for n, v in zip(jn, kd)},
    },
    "action": {
      "scale": {n: round(v, 6) for n, v in zip(jn, scale)},
      "pinned_zero_scale": [n for n, v in zip(jn, scale) if v == 0.0],
      "live": [n for n, v in zip(jn, scale) if v != 0.0],
    },
    "joints": [
      {
        "index": i, "name": n, "actuator": groups[n],
        "lower": round(ulim[n]["lower"], 5), "upper": round(ulim[n]["upper"], 5),
        "velocity_limit": round(ulim[n]["velocity"], 4),
        "effort_peak_nm": round(effort[n], 1),
        "effort_continuous_nm": round(float(cont[n]), 1),
      }
      for i, n in enumerate(jn)
    ],
    "commands": {
      "order": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
      "trained_ranges": {"lin_vel_x": [-1.2, 1.5], "lin_vel_y": [-1.0, 1.0],
                         "ang_vel_z": [-0.7, 0.7]},
      "WARNING": "Do not command outside these. The policy has never seen it.",
    },
    "four_bar_warning": {
      "joints": ["left_ankle_pitch_joint", "right_ankle_pitch_joint",
                 "left_ankle_roll_joint", "right_ankle_roll_joint",
                 "waist_roll_joint", "waist_pitch_joint"],
      "issue": "These six are parallel four-bars driven by two X8-120s each. The "
               "linkage AMPLIFIES commanded joint gain, so writing the sim kp "
               "(400) straight to the motors will over-stiffen them. Measure the "
               "amplification on HV1V1 and divide before deploying.",
      "h1v1_reference_only": H1V1_FOURBAR_REFERENCE,
      "h1v1_caveat": "Those factors were measured on H1-V1 hardware, NOT on HV1V1. "
                     "They are a starting point for a measurement, not a value to use.",
      "waist_yaw": "passthrough, not part of any differential - no correction",
    },
  }

  shutil.copy2(onnx_src, out / "policy.onnx")
  (out / "deploy.json").write_text(json.dumps(bundle, indent=2) + "\n")
  if ckpts:
    shutil.copy2(ckpts[-1], out / ckpts[-1].name)
  (out / "checksums.txt").write_text(
    "".join(f"{sha256(p)}  {p.name}\n" for p in sorted(out.iterdir()) if p.is_file()
            and p.name != "checksums.txt")
  )

  print(f"wrote {out}")
  for p in sorted(out.iterdir()):
    print(f"   {p.name:34s} {p.stat().st_size/1024:9.1f} KiB")
  print(f"\n  obs {obs_dim_in} = " + " + ".join(
    f"{t['name']}({t['dim']}x{t['history']})" for t in obs_terms))
  print(f"  actions {act_dim}   live {len(bundle['action']['live'])}   "
        f"pinned {len(bundle['action']['pinned_zero_scale'])}")
  print(f"  ready pose CoM {com_off_mm:+.2f} mm from sole centre")


if __name__ == "__main__":
  main()
