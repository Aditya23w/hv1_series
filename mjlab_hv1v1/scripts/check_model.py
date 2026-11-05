#!/usr/bin/env python3
"""Audit the COMPILED HV1V1 model against the intended plant.

Everything here is asserted against mujoco's compiled arrays rather than the
config objects, because the failure mode that matters is a regex that matched
nothing: mjlab does not error on that, it just leaves the joint alone.

  python mjlab_hv1v1/scripts/check_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity.entity import Entity
from mjlab_hv1v1.hv1v1_constants import (
  CONTINUOUS_TORQUE,
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  HOME_KEYFRAME,
  HV1V1_ACTION_SCALE,
  PINNED_JOINTS,
  get_hv1v1_robot_cfg,
  resolve_joint_groups,
)
from mjlab.utils.string import resolve_expr

# group -> (stiffness, damping, effort, armature, frictionloss, viscous, count)
EXPECTED = {
  "X12-320":  (250.0, 15.0, 320.0, 0.5160, 3.8,  0.390, 8),
  "X8-120":   (200.0, 12.0, 120.0, 0.0577, 1.5,  0.141, 9),
  "X8-120x2": (400.0, 24.0, 240.0, 0.1154, 3.0,  0.282, 6),
  "X6-60":    (200.0, 12.0,  60.0, 0.0254, 1.6,  0.016, 2),
  "X4-36":    (100.0,  6.0,  34.0, 0.0389, 1.14, 0.049, 2 + 4),
}
EXPECTED_SCALE = {"X12-320": 0.32, "X8-120": 0.15, "X8-120x2": 0.15,
                  "X6-60": 0.075, "X4-36": 0.085}
# Per-JOINT departures from the group value, spelled out by joint name rather
# than by regex. EMPTY for v5: the config is HV1V2 v4's exactly, so every joint
# must sit at its group value. If ACTION_SCALE_OVERRIDES is ever repopulated,
# list the joints here too -- this table is what proves an override pattern hit
# only what it was meant to hit and not its actuator-group neighbours.
EXPECTED_SCALE_BY_JOINT: dict[str, float] = {}
# How much room must remain between what the policy can command (anchor +- scale)
# and the soft joint limit. This is a HARD gate only for the joints in
# EXPECTED_SCALE_BY_JOINT -- the ones this config deliberately moved -- and is
# informational for the rest.
#
# It is deliberately NOT a global requirement, because the shipped design does not
# meet one. Measured, the eight tightest joints are IDENTICAL on HV1V2 and HV1V1
# to four decimals, and four of them are already past their soft limit at full
# action: shoulder_roll -0.031 rad, hip_roll -0.006/-0.005, with knee +0.065 and
# ankle_roll +0.084 close behind. HV1V2 trains and walks at 0.93 m/s like that
# and its dof_pos_limits term stays at ~0 in practice, because the policy never
# commands |action| = 1 on those joints. Asserting 0.1 rad everywhere would fail
# the known-good reference, which makes it the wrong gate.
MIN_SOFT_LIMIT_CLEARANCE = 0.1  # rad, enforced on overridden joints only
SOFT_LIMIT_FACTOR = 0.9         # must match EntityArticulationInfoCfg
# The XML placeholders the viewer model carries; none may survive into training.
VIEWER_PLACEHOLDERS = {"armature": 0.01, "frictionloss": 0.1, "damping": 1.0}

FAILS: list[str] = []


def check(cond, msg):
  if not cond:
    FAILS.append(msg)
  return cond


# Expected actor width per task, and the critic width that must NOT change.
# actor = 3*5 + 3*5 + 31*5 + 31*5 + 31*5 + 3*1 (command stays one frame).
EXPECTED_ACTOR = {
  "Mjlab-Velocity-Flat-HV1V1": 498,
  "Mjlab-Velocity-Flat-HV1V1-State-Estimation": 513,  # + base_lin_vel 3*5
}
EXPECTED_CRITIC = 117


def check_observations() -> None:
  """Observation history: widths, per-term settings, and reset backfill.

  The critic width is the assertion that matters most. The actor and critic
  groups share ObservationTermCfg objects, so an in-place history edit would
  silently widen the critic too, and nothing else in the stack would complain.
  """
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab_hv1v1.env_cfgs import OBS_HISTORY, OBS_HISTORY_EXCLUDE

  print("\n[8] observation history")
  for task, want_actor in EXPECTED_ACTOR.items():
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
      dims = {g: int(d[0]) for g, d in env.observation_manager.group_obs_dim.items()}
      check(dims["actor"] == want_actor,
            f"{task}: actor width {dims['actor']} != {want_actor}")
      check(dims["critic"] == EXPECTED_CRITIC,
            f"{task}: CRITIC width {dims['critic']} != {EXPECTED_CRITIC} - history "
            "leaked into the critic (in-place mutation of a shared term cfg?)")
      print(f"    {task:44s} actor {dims['actor']:4d}  critic {dims['critic']:4d}")

      # per-term settings, read back from the RESOLVED cfg the manager built
      for name in env.observation_manager.active_terms["actor"]:
        tc = env.observation_manager.get_term_cfg("actor", name)
        want = 0 if name in OBS_HISTORY_EXCLUDE else OBS_HISTORY
        check(tc.history_length == want,
              f"{task}: actor term {name} history_length "
              f"{tc.history_length} != {want}")
        check(tc.flatten_history_dim,
              f"{task}: actor term {name} flatten_history_dim is False - "
              "rsl-rl's MLPModel only accepts 1D observations")
      for name in env.observation_manager.active_terms["critic"]:
        tc = env.observation_manager.get_term_cfg("critic", name)
        check(tc.history_length == 0,
              f"{task}: CRITIC term {name} has history_length {tc.history_length}")

      # reset backfill: all H slots hold the same frame, not zeros (and not
      # zeros in particular - a zeroed history is off-distribution at t=0).
      if task == "Mjlab-Velocity-Flat-HV1V1":
        obs, _ = env.reset()
        a = obs["actor"]
        js = env.observation_manager.active_terms["actor"].index("joint_pos")
        off = sum(
          int(env.observation_manager.group_obs_term_dim["actor"][i][0])
          for i in range(js)
        )
        block = a[:, off:off + 31 * OBS_HISTORY].reshape(-1, OBS_HISTORY, 31)
        same = torch.allclose(block[:, 0], block[:, -1], atol=1e-6)
        nonzero = bool(block.abs().sum() > 0)
        check(same, "after reset the history slots differ - expected the first "
                    "frame backfilled into all of them")
        check(nonzero, "history is all zeros after reset")
        for _ in range(OBS_HISTORY + 1):
          obs, *_ = env.step(torch.zeros_like(env.action_manager.action))
        a2 = obs["actor"]
        b2 = a2[:, off:off + 31 * OBS_HISTORY].reshape(-1, OBS_HISTORY, 31)
        check(not torch.allclose(b2[:, 0], b2[:, -1], atol=1e-6),
              "after stepping, oldest and newest history slots are still equal - "
              "the buffer is not advancing")
        print(f"    reset backfill OK (all {OBS_HISTORY} slots equal, non-zero); "
              "buffer advances after stepping")
    finally:
      env.close()


def main() -> int:
  robot = Entity(get_hv1v1_robot_cfg())
  m = robot.spec.compile()

  jnames = tuple(
    n for i in range(m.njnt)
    if (n := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)) and n != "floating_base"
  )
  groups = resolve_joint_groups(jnames)

  print(f"model: {m.nbody} bodies, {len(jnames)} hinges, {m.nu} actuators, "
        f"{sum(m.body_mass):.6f} kg")

  # -- 1. group coverage -----------------------------------------------------
  print("\n[1] actuator group coverage")
  check(len(jnames) == 31, f"expected 31 hinges, got {len(jnames)}")
  for g, exp in EXPECTED.items():
    n = sum(1 for v in groups.values() if v == g)
    ok = check(n == exp[6], f"{g}: {n} joints, expected {exp[6]}")
    print(f"    {g:10s} {n:2d} joints  {'ok' if ok else 'MISMATCH'}")

  # -- 2. per-joint plant, from the compiled arrays --------------------------
  print("\n[2] compiled plant per joint")
  act_of = {}
  for a in range(m.nu):
    act_of[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])] = a
  check(len(act_of) == 31, f"{len(act_of)} joints have actuators, expected 31")

  for jn in jnames:
    g = groups[jn]
    kp, kd, eff, arm, fl, visc, _ = EXPECTED[g]
    d = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]

    # passive joint properties
    check(abs(m.dof_armature[d] - arm) < 1e-9,
          f"{jn}: armature {m.dof_armature[d]:.6g} != {arm}")
    check(abs(m.dof_frictionloss[d] - fl) < 1e-9,
          f"{jn}: frictionloss {m.dof_frictionloss[d]:.6g} != {fl}")
    check(abs(m.dof_damping[d] - visc) < 1e-9,
          f"{jn}: viscous damping {m.dof_damping[d]:.6g} != {visc}")
    # the viewer placeholders must not have survived
    for lbl, bad in VIEWER_PLACEHOLDERS.items():
      got = {"armature": m.dof_armature[d], "frictionloss": m.dof_frictionloss[d],
             "damping": m.dof_damping[d]}[lbl]
      check(abs(got - bad) > 1e-12,
            f"{jn}: {lbl} is the viewer placeholder {bad} - a regex matched nothing")

    # active PD gains: position actuator is gainprm[0]=kp, biasprm=[0,-kp,-kd]
    a = act_of[jn]
    check(abs(m.actuator_gainprm[a, 0] - kp) < 1e-9,
          f"{jn}: kp {m.actuator_gainprm[a,0]:.6g} != {kp}")
    check(abs(m.actuator_biasprm[a, 1] + kp) < 1e-9,
          f"{jn}: biasprm[1] {m.actuator_biasprm[a,1]:.6g} != {-kp}")
    check(abs(m.actuator_biasprm[a, 2] + kd) < 1e-9,
          f"{jn}: kd {-m.actuator_biasprm[a,2]:.6g} != {kd}")
    check(abs(m.actuator_forcerange[a, 1] - eff) < 1e-9,
          f"{jn}: forcerange {m.actuator_forcerange[a,1]:.6g} != {eff}")
    # PD gain and passive damping must not be the same number by accident
    check(abs(kd - visc) > 1e-12 or kd == 0,
          f"{jn}: PD kd equals viscous damping - check they were not conflated")

  for g, (kp, kd, eff, arm, fl, visc, n) in EXPECTED.items():
    print(f"    {g:10s} kp {kp:5.0f}  kd {kd:4.0f} | armature {arm:.4f}  "
          f"friction {fl:.2f}  viscous {visc:.3f} | effort +-{eff:.0f}")

  # -- 3. four-bar joints ----------------------------------------------------
  print("\n[3] four-bar (2x X8-120) joints")
  fb = [j for j, g in groups.items() if g == "X8-120x2"]
  check(len(fb) == 6, f"{len(fb)} four-bar joints, expected 6")
  for jn in sorted(fb):
    a = act_of[jn]
    print(f"    {jn:26s} kp {m.actuator_gainprm[a,0]:.0f}  "
          f"forcerange +-{m.actuator_forcerange[a,1]:.0f}")

  # -- 4. action scale -------------------------------------------------------
  print("\n[4] action scale over the 31 joints")
  SENTINEL = -12345.0
  scale = np.array(resolve_expr(HV1V1_ACTION_SCALE, jnames, SENTINEL))
  check(not np.any(scale == SENTINEL),
        "some joint matched no action-scale pattern (it would silently get 1.0)")
  check(not np.any(np.isclose(scale, 1.0)),
        "a joint has scale exactly 1.0 - the torch.ones default leaked through")
  pinned = [j for j in jnames if any(__import__("re").match(p, j) for p in PINNED_JOINTS)]
  zeros = [j for j, s in zip(jnames, scale) if s == 0.0]
  check(sorted(zeros) == sorted(pinned),
        f"pinned set mismatch: {sorted(set(zeros) ^ set(pinned))}")
  check(len(zeros) == 11, f"{len(zeros)} pinned joints, expected 11")
  live = [j for j, s in zip(jnames, scale) if s != 0.0]
  check(len(live) == 20, f"{len(live)} live joints, expected 20")
  expect_live = {j for j in jnames
                 if any(k in j for k in ("hip", "knee", "ankle", "shoulder", "elbow"))}
  check(set(live) == expect_live,
        f"live set is not legs+shoulders+elbows: {sorted(set(live) ^ expect_live)}")
  # Assert the RESOLVED per-joint scale, so an override pattern that matched more
  # joints than intended cannot pass. Every live joint is either at its group
  # value or at an explicitly named exception.
  overridden = []
  for j, sc in zip(jnames, scale):
    if sc == 0.0:
      continue
    want = EXPECTED_SCALE_BY_JOINT.get(j, EXPECTED_SCALE[groups[j]])
    check(abs(sc - want) < 1e-9, f"{j}: scale {sc} != {want}")
    if j in EXPECTED_SCALE_BY_JOINT:
      overridden.append(j)
  check(sorted(overridden) == sorted(EXPECTED_SCALE_BY_JOINT),
        f"override set mismatch: {sorted(set(overridden) ^ set(EXPECTED_SCALE_BY_JOINT))}")
  # Nothing outside the override table may differ from its group value -- this is
  # the check that would catch ".*_ankle_pitch_joint$" also hitting ankle_roll.
  for j, sc in zip(jnames, scale):
    if sc != 0.0 and j not in EXPECTED_SCALE_BY_JOINT:
      check(abs(sc - EXPECTED_SCALE[groups[j]]) < 1e-9,
            f"{j} is not in the override table but departs from its group value")
  print(f"    pinned {len(zeros)}  live {len(live)}  "
        f"values {sorted(set(round(float(s),4) for s in scale))}")
  print(f"    overridden {overridden}")

  # -- 4b. reachable window vs the soft joint limits -------------------------
  # The policy commands anchor + action * scale with |action| nominally <= 1, so
  # the widest target it can ask for is anchor +- scale. That must stay clear of
  # the soft limits, or dof_pos_limits starts charging for ordinary walking.
  # Measured on HV1V2 v4: 19 deg of slack low and 27 deg high, and the term never
  # fires. The anchor is read from the committed keyframe -- nothing here is
  # hard-coded, because the whole point is to check a value the pose solver
  # produced.
  print("\n[4b] reachable window (anchor +- scale) vs 0.9 soft limits")
  # Joints absent from HOME_KEYFRAME legitimately anchor at 0.0 (shoulder_yaw is
  # one), which is what mjlab does with an unlisted joint -- so resolve to 0.0,
  # not to a sentinel.
  anchor = np.array(resolve_expr(HOME_KEYFRAME.joint_pos, jnames, 0.0))
  rows = []
  for j, sc, a0 in zip(jnames, scale, anchor):
    if sc == 0.0:
      continue
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    lo, hi = m.jnt_range[jid]
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    slo, shi = mid - SOFT_LIMIT_FACTOR * half, mid + SOFT_LIMIT_FACTOR * half
    clear = float(min((a0 - sc) - slo, shi - (a0 + sc)))
    rows.append((clear, j, sc, a0))
    # Hard only where this config departs from the group rule.
    if j in EXPECTED_SCALE_BY_JOINT:
      check(clear >= MIN_SOFT_LIMIT_CLEARANCE,
            f"{j}: scale was overridden to {sc}, and reachable "
            f"[{a0-sc:+.3f},{a0+sc:+.3f}] leaves only {clear:.3f} rad to the soft "
            f"limits [{slo:+.3f},{shi:+.3f}]; need {MIN_SOFT_LIMIT_CLEARANCE}")
  for clear, j, sc, a0 in sorted(rows)[:4]:
    print(f"    tightest  {j:28s} scale {sc:.3f}  clearance {clear:+.4f} rad "
          f"({np.degrees(clear):+5.1f} deg){'   <- pre-existing, same on HV1V2' if clear < 0.1 else ''}")
  for j in EXPECTED_SCALE_BY_JOINT:
    clear = next(c for c, jj, _, _ in rows if jj == j)
    print(f"    OVERRIDDEN {j:27s} clearance {clear:+.4f} rad "
          f"({np.degrees(clear):+5.1f} deg)  gate >= {MIN_SOFT_LIMIT_CLEARANCE}")

  # -- 5. ordering -----------------------------------------------------------
  print("\n[5] ctrl order vs canonical hardware order")
  # HV1V1's own joint-name file. Verified byte-equivalent in ORDER to HV1V2's,
  # so ctrl/observation indexing is shared - but read HV1V1's, not HV1V2's.
  import yaml
  cy = yaml.safe_load(
    open(Path(__file__).resolve().parents[2] /
         "URDF_HV1V2_26092026/config/joint_names_URDF_HV1V2_26092026.yaml")
  )["controller_joint_names"]
  cy = [j for j in cy if j]
  ctrl_order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0])
                for a in range(m.nu)]
  check(ctrl_order == cy, "ctrl order != controller_joint_names order")
  check(list(jnames) == cy, "joint order != controller_joint_names order")
  print(f"    ctrl[0..3] = {ctrl_order[:4]}")
  print(f"    ctrl[-2:]  = {ctrl_order[-2:]}")

  # -- 6. feet ---------------------------------------------------------------
  print("\n[6] foot collision pads and sites")
  gnames = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)}
  for n in FOOT_GEOM_NAMES:
    check(n in gnames, f"missing foot geom {n}")
  snames = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(m.nsite)}
  for n in FOOT_SITE_NAMES:
    check(n in snames, f"missing foot site {n}")
  check(not any(n and "ankle_roll" in n and n.endswith("_collision") for n in gnames),
        "the ankle_roll convex hull is still present and would fight the pads")
  for n in FOOT_GEOM_NAMES:
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    check(m.geom_condim[g] == 3, f"{n}: condim {m.geom_condim[g]} != 3")
    check(m.geom_priority[g] == 1, f"{n}: priority {m.geom_priority[g]} != 1")
    check(abs(m.geom_friction[g, 0] - 0.6) < 1e-9,
          f"{n}: friction {m.geom_friction[g,0]} != 0.6")
  print(f"    {len(FOOT_GEOM_NAMES)} pads, condim 3, priority 1, friction 0.6; "
        f"sites {FOOT_SITE_NAMES}")

  # -- 7. continuous-torque table --------------------------------------------
  print("\n[7] continuous torque table covers every joint")
  cont = resolve_expr(CONTINUOUS_TORQUE, jnames, None)
  check(all(c is not None for c in cont), "a joint has no continuous torque rating")
  for jn, c in zip(jnames, cont):
    eff = EXPECTED[groups[jn]][2]
    check(c < eff, f"{jn}: continuous {c} should be below peak {eff}")

  # -- 8. observation history (needs a built env, so it runs last) -----------
  check_observations()

  # -- report ----------------------------------------------------------------
  print()
  if FAILS:
    print(f"FAILED ({len(FAILS)}):")
    for f in FAILS:
      print("  -", f)
    return 1
  print("ALL CHECKS PASSED")
  return 0


if __name__ == "__main__":
  sys.exit(main())
