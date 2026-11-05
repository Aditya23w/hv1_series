#!/usr/bin/env python3
"""Build a self-contained deployment bundle for a trained HV1V2 policy.

Everything a runtime needs to drive the robot: the ONNX graph, the joint order,
the anchor pose its actions offset from, the READY pose to servo into before
handing over, the PD gains, the action scales, the observation layout, and the
limits. Nothing here is hand-typed -- it is read from the compiled model, the
ONNX metadata and the source URDF, so the bundle cannot drift from what was
trained.

  python mjlab_hv1v2/scripts/export_deploy.py --run v5
  python mjlab_hv1v2/scripts/export_deploy.py --run v6 \
      --task Mjlab-Velocity-Flat-HV1V2-Decoupled
  python mjlab_hv1v2/scripts/export_deploy.py --run v4 --out /tmp/deploy_v4
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
LOG_ROOT = Path(__file__).resolve().parents[1] / "logs/rsl_rl/hv1v2_velocity"
URDF = ROOT / "urdf/HV1V2_URDF_NEW_FOOT_V1_220926.urdf"

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
  "ub_target": "teleop upper-body target as merai_policy commands it (clamped, "
               "slewed ub_cmd), minus anchor, for policy.ub_obs_joints (rad)",
  "ub_target_rate": "(ub_cmd_t - ub_cmd_{t-1}) * 50 Hz for policy.ub_obs_joints, "
                    "updated BEFORE the observation is built (rad/s)",
  "height_scan": "terrain perception - NOT AVAILABLE on hardware",
  "height_cmd": "v11h: operator pelvis-height command AS APPLIED (clamped to "
                "height.range_m, slewed at height.slew_m_s), minus "
                "height.home_m (m; 0 at home, negative when crouched)",
}

# Four-bar amplification MEASURED ON H1-V1, not on HV1V2. Reference only.
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


def trained_command_ranges(task: str) -> dict:
  """The command ranges the policy finished training on, read from the config.

  The twist ranges are overwritten by the command_vel curriculum, so the
  answer is the base ranges with every stage applied in order -- the last stage
  is what the policy saw at the end. Hand-typing these let v4's (-1.2, 1.5)
  survive into a config that had changed.
  """
  import mjlab_hv1v2  # noqa: F401  (registers the tasks)
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(task)
  r = cfg.commands["twist"].ranges
  out = {"lin_vel_x": list(r.lin_vel_x), "lin_vel_y": list(r.lin_vel_y),
         "ang_vel_z": list(r.ang_vel_z)}
  cur = cfg.curriculum.get("command_vel")
  for stage in (cur.params["velocity_stages"] if cur else []):
    for k in out:
      if stage.get(k) is not None:
        out[k] = list(stage[k])
  return out


# Tasks whose upper body was driven half by v7's random generator and half by
# the offline Mink -> Ruckig clip library, with per-substep interpolation of
# every target, a 0-20 ms leg delay and ankle stiffness DR (v8).
RUCKIG_TRAINED_TASKS = (
  "Mjlab-Velocity-Flat-HV1V2-Decoupled-v8",
  "Mjlab-Velocity-Flat-HV1V2-Decoupled-v11",
  "Mjlab-Velocity-Flat-HV1V2-Decoupled-v11h",
)
RUCKIG_LIMITS = {"default": [2.0, 8.0, 40.0], "waist_yaw_joint": [0.6, 4.0, 20.0]}


def upper_body_section(jn, aj, anchor, kp, kd, task: str = "") -> dict:
  """What the IK side must know: which joints are its, where they rest, the
  gains they were simulated with, and the envelope the leg policy was trained
  to tolerate. Read from the same constants the training generator used."""
  import re

  from mjlab_hv1v2.hv1v2_constants import HAND_BODY_NAMES, UPPER_BODY_SAMPLE_RANGE

  names = [n for n in jn if n not in set(aj)]
  rng = {}
  for n in names:
    hits = [r for pat, r in UPPER_BODY_SAMPLE_RANGE.items() if re.match(pat, n)]
    assert len(hits) == 1, f"{n} matched {len(hits)} UPPER_BODY_SAMPLE_RANGE keys"
    rng[n] = [float(hits[0][0]), float(hits[0][1])]
  i = {n: jn.index(n) for n in names}
  out = {
    "note": "NOT driven by the policy. The manipulation/IK controller writes "
            "these position targets every cycle, through the same joint PD. "
            "With no IK running, hold home -- never leave them unwritten.",
    "joints": names,
    "joint_indices": [i[n] for n in names],
    "home": {n: round(anchor[i[n]], 6) for n in names},
    "kp": {n: round(kp[i[n]], 3) for n in names},
    "kd": {n: round(kd[i[n]], 3) for n in names},
    "trained_range": rng,
    "locked_at_home": [n for n, (lo, hi) in rng.items() if lo == hi],
    "WARNING": "The leg policy was trained against upper-body targets inside "
               "trained_range, slewing at <= 2.0 rad/s, with 0-2 kg per hand "
               "and +-20 N hand forces. IK must clamp to trained_range and rate-"
               "limit to 2.0 rad/s; joints in locked_at_home must stay at home.",
    "trained_disturbance": {
      "max_slew_rad_s": 2.0,
      "hand_bodies": list(HAND_BODY_NAMES),
      "payload_kg_per_hand": [0.0, 2.0],
      "hand_force_n_per_axis": [-20.0, 20.0],
    },
  }
  if task in RUCKIG_TRAINED_TASKS:
    moving = [n for n in names if n not in out["locked_at_home"]]
    out["trained_disturbance"].update({
      "shaping": "half the motion: online Ruckig at 50 Hz on the IK target "
                 "(target velocity/acceleration 0, joints independent), applied "
                 "ONCE; the other half: v7's random goals at a constant "
                 "0.3-2.0 rad/s slew",
      "ruckig_limits_v_a_j": {n: RUCKIG_LIMITS.get(n, RUCKIG_LIMITS["default"])
                              for n in moving},
      "moving_joints": moving,
      "command_timing": "every 50 Hz target (legs AND upper body) linearly "
                        "interpolated to the next over the control period -- "
                        "merai_control's 1 kHz interpolation; trained at 5 ms "
                        "resolution, legs with a 0-20 ms delay before each ramp",
      "ankle_stiffness_dr": "ankle pitch kp/kd x U(1.0, 1.9), roll x U(1.0, 1.4) "
                            "on top of +-10% gain DR -- the real four-bar's "
                            "joint stiffness",
    })
    out["WARNING"] = (
      "Trained against upper-body motion inside trained_range: half shaped by "
      "online Ruckig (v <= 2.0, a <= 8, j <= 40; waist_yaw 0.6 / 4 / 20), half "
      "v7's random constant-speed slews (<= 2.0 rad/s), with 0-2 kg "
      "per hand and +-20 N hand forces. The merai runtime today only clamps "
      "and velocity-slews (no acceleration/jerk limit): that is HARSHER than "
      "training. Joints in locked_at_home must stay at home.")
  if task in ("Mjlab-Velocity-Flat-HV1V2-Decoupled-v11", "Mjlab-Velocity-Flat-HV1V2-Decoupled-v11h"):
    td = out["trained_disturbance"]
    td.update({
      "shaping": "clips (Mink -> Ruckig) + random goals at PER-JOINT speeds "
                 "U(0.3, 2.0) rad/s (waist_yaw <= 1.0), abrupt starts/stops, "
                 "holds; 2% single-step target hold (packet loss), +-0.1 rad/s "
                 "noise on the observed rate",
      "payload": "0-2 kg per hand, own inertia, CoM 0-10 cm from the wrist, "
                 "per episode; mid-episode pick-ups ramped over 0.2 s",
      "hand_force_n": "smooth ramps (rise 0.2-1 s, hold 1-4 s), |F| <= 30 N, "
                      "sometimes both hands opposite",
      "torso_push_n": "<= 60 N on torso_link, 0.05-0.3 s, while the arms move",
      "ankle_stiffness_dr": "ankle pitch x U(0.8, 1.45), roll x U(0.5, 1.1), "
                            "waist roll/pitch x U(2.5, 4.5) of the 400/24 joint "
                            "PD (real four-bar at motor_gain 0.45 / 0.75), "
                            "kp/kd x U(0.85, 1.15) on every joint",
    })
    td["hand_force_n_per_axis"] = [-30.0, 30.0]
    out["WARNING"] = (
      "v11 SEES the upper-body target and its rate (policy.ub_obs_joints): "
      "merai_policy must feed ub_cmd - home and (ub_cmd_t - ub_cmd_{t-1}) * 50 "
      "from the SAME clamped, slewed target it commands. Trained inside "
      "trained_range, per-joint speeds <= 2.0 rad/s (waist_yaw <= 1.0), 0-2 kg "
      "per hand, <= 30 N hand forces. Joints in locked_at_home must stay at home.")
  return out


def height_section(task: str, jn) -> dict:
  """v11h: the operator pelvis-height input -- range, slew, reset rule, and
  the reference pose/torque table the policy was rewarded against. Read from
  the task's command config and V11H_HEIGHT_TABLE."""
  import mjlab_hv1v2  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg
  from mjlab_hv1v2.hv1v2_constants import V11H_HEIGHT_RANGE, V11H_HEIGHT_TABLE, V11H_HOME_Z

  c = load_env_cfg(task, play=True).commands["height"]
  t = V11H_HEIGHT_TABLE
  knee = [i for i, n in enumerate(t["joint_names"]) if n.endswith("knee_joint")]
  return {
    "note": "Operator pelvis height (m, pelvis origin above the sole plane). The "
            "policy bends knee + hip pitch + ankle pitch itself; the runtime only "
            "feeds the command. Observation height_cmd = h_applied - home_m.",
    "range_m": [round(float(V11H_HEIGHT_RANGE[0]), 5), round(float(V11H_HEIGHT_RANGE[1]), 5)],
    "home_m": round(float(V11H_HOME_Z), 5),
    "slew_m_s": float(c.slew_rate),
    "runtime_rule": "h_target = clamp(operator, range_m); h_applied moves toward "
                    "h_target by at most slew_m_s * control_dt per tick, BEFORE the "
                    "observation is built; observe h_applied - home_m.",
    "on_activation": f"h_applied = h_target = home_m (obs 0) and hold home for "
                     f"{float(c.reset_hold_s)} s before following the operator -- "
                     "exactly the training reset.",
    "trained": "while standing AND walking (all trained twist ranges), arms at "
               "full upper-body envelope; resampled every "
               f"{list(c.resampling_time_range)} s, {int(100 * c.home_prob)}% of "
               "targets at home",
    "reference_table": {
      "note": "static double-support solve (feet flat, CoM over sole centre, "
              "92.7 kg). Reference only -- NOT a feedforward to send.",
      "joint_names": list(t["joint_names"]),
      "pelvis_z": [round(z, 5) for z in t["pelvis_z"]],
      "q": [[round(v, 5) for v in r] for r in t["q"]],
      "tau_static": [[round(v, 2) for v in r] for r in t["tau_static"]],
    },
    "knee_static_torque_nm": {"at_min": round(abs(t["tau_static"][0][knee[0]]), 1),
                              "at_home": round(abs(t["tau_static"][-1][knee[0]]), 1)},
    "WARNING": "At 0.89 m each knee holds ~65 N.m static (sim, model_26999: "
               "R 61 / L 71 N.m) against an 85 N.m continuous rating. Long low "
               "crouches/walks are thermally limited: watch knee temperature, "
               "especially the left. Never feed a value outside range_m or "
               "an unslewed step.",
  }


def sha256(p: Path) -> str:
  return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", default="v4")
  ap.add_argument("--out", default=None)
  ap.add_argument("--task", default="Mjlab-Velocity-Flat-HV1V2",
                  help="the task the run was trained on (sets command ranges)")
  args = ap.parse_args()

  run_dir = find_run(args.run)
  onnx_src = sorted(run_dir.glob("*.onnx"))[-1]
  out = Path(args.out) if args.out else ROOT / "mjlab_hv1v2/exported" / f"deploy_{args.run}"
  out.mkdir(parents=True, exist_ok=True)

  # ---- read the policy ----------------------------------------------------
  g = onnx.load(str(onnx_src))
  meta = {p.key: p.value for p in g.metadata_props}
  jn = meta["joint_names"].split(",")
  fl = lambda k: [float(x) for x in meta[k].split(",")]  # noqa: E731
  anchor = fl("default_joint_pos")
  # The joints the actions drive. Recorded by Hv1v2OnPolicyRunner since the
  # decoupled task; earlier ONNX act on every joint, in joint order.
  aj = meta["action_joint_names"].split(",") if "action_joint_names" in meta else jn
  scale = list(np.broadcast_to(fl("action_scale"), (len(aj),)))
  decoupled = len(aj) < len(jn)
  kp, kd = fl("joint_stiffness"), fl("joint_damping")
  obs_names = meta["observation_names"].split(",")
  obs_hist = [int(x) for x in fl("observation_terms_history_length")]
  obs_dim_in = g.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
  act_dim = g.graph.output[0].type.tensor_type.shape.dim[-1].dim_value

  # ---- read the model the policy was trained against ----------------------
  from mjlab.entity.entity import Entity
  from mjlab.utils.string import resolve_expr
  from mjlab_hv1v2.hv1v2_constants import (
    ARMATURE_PROVENANCE, CONTINUOUS_TORQUE, HOME_KEYFRAME,
    get_hv1v2_robot_cfg, resolve_joint_groups,
  )

  m = Entity(get_hv1v2_robot_cfg()).spec.compile()
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
  assert len(aj) == act_dim, f"{len(aj)} action joints but the graph outputs {act_dim}"

  groups = resolve_joint_groups(tuple(jn))
  cont = dict(zip(jn, resolve_expr(CONTINUOUS_TORQUE, tuple(jn), None)))
  ulim = urdf_limits()
  effort = {n: float(m.actuator_forcerange[a, 1]) for a, n in enumerate(ctrl_joint)}

  # per-observation-term dims, in the policy's own order
  from mjlab_hv1v2.hv1v2_constants import V11_UB_OBS_JOINTS

  term_dim = {"base_ang_vel": 3, "base_lin_vel": 3, "projected_gravity": 3,
              "joint_pos": len(jn), "joint_vel": len(jn), "actions": act_dim,
              "command": 3,
              # v11: the actor observes the teleop target and its rate
              "ub_target": len(V11_UB_OBS_JOINTS),
              "ub_target_rate": len(V11_UB_OBS_JOINTS),
              # v11h: pelvis-height command, centred on home
              "height_cmd": 1}
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
      "robot": "HV1V2", "run": run_dir.name, "task": args.task,
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
      **({"ub_obs_joints": list(V11_UB_OBS_JOINTS)}
         if any(t["name"].startswith("ub_") for t in obs_terms) else {}),
    },
    "anchor": {
      "note": "action mapping is  target[j] = anchor[j] + raw_action[k] * action.scale[j] "
              "for j = action.joint_names[k]",
      "default_joint_pos": {n: round(v, 6) for n, v in zip(jn, anchor)},
      "pelvis_height_m": round(float(HOME_KEYFRAME.pos[2]), 5),
    },
    "ready_pose": {
      "note": "Servo here with joint PD before handing over to the policy. "
              "Identical to the anchor, which is safe here because the trained "
              "pose was solved with the CoM over the sole centre.",
      "joint_pos": {n: round(v, 6) for n, v in zip(jn, anchor)},
      "pelvis_height_m": round(float(HOME_KEYFRAME.pos[2]), 5),
      "com_offset_from_sole_centre_mm": round(com_off_mm, 2),
      "feet": "flat, stance width 0.24304 m",
      "WARNING": "This pose does NOT stand open-loop. Holding it with PD and no "
                 "policy, the robot topples in ~3.8 s: ankle stiffness is ~95% of "
                 "the m*g*h the inverted pendulum demands. Keep the policy running.",
    },
    "gains": {
      "note": "Simulation joint-space PD, exactly what the policy was trained "
              "against. See four_bar_warning before writing these to hardware.",
      "kp": {n: round(v, 3) for n, v in zip(jn, kp)},
      "kd": {n: round(v, 3) for n, v in zip(jn, kd)},
    },
    "action": {
      "joint_names": list(aj),
      "joint_indices": [jn.index(n) for n in aj],
      "scale": {n: round(float(v), 6) for n, v in zip(aj, scale)},
      "pinned_zero_scale": [n for n, v in zip(aj, scale) if v == 0.0],
      "live": [n for n, v in zip(aj, scale) if v != 0.0],
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
      "trained_ranges": trained_command_ranges(args.task),
      "WARNING": "Do not command outside these. The policy has never seen it.",
    },
    "four_bar_warning": {
      "joints": ["left_ankle_pitch_joint", "right_ankle_pitch_joint",
                 "left_ankle_roll_joint", "right_ankle_roll_joint",
                 "waist_roll_joint", "waist_pitch_joint"],
      "issue": "These six are parallel four-bars driven by two X8-120s each. The "
               "linkage AMPLIFIES commanded joint gain, so writing the sim kp "
               "(400) straight to the motors will over-stiffen them. Measure the "
               "amplification on HV1V2 and divide before deploying.",
      "h1v1_reference_only": H1V1_FOURBAR_REFERENCE,
      "h1v1_caveat": "Those factors were measured on H1-V1 hardware, NOT on HV1V2. "
                     "They are a starting point for a measurement, not a value to use.",
      "waist_yaw": "passthrough, not part of any differential - no correction",
    },
  }

  if any(t["name"] == "height_cmd" for t in obs_terms):
    bundle["height"] = height_section(args.task, jn)

  if decoupled:
    bundle["upper_body"] = upper_body_section(jn, aj, anchor, kp, kd, task=args.task)

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
  if decoupled:
    ub = bundle["upper_body"]
    print(f"  upper body {len(ub['joints'])} externally commanded "
          f"({len(ub['locked_at_home'])} locked at home)")
  print(f"  ready pose CoM {com_off_mm:+.2f} mm from sole centre")


if __name__ == "__main__":
  main()
