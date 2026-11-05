"""HV1V2 velocity-tracking environment configurations.

Built by calling mjlab's generic ``make_velocity_env_cfg()`` and then filling in
every blank it leaves for the robot. Structure follows
``mjlab/tasks/velocity/config/g1/env_cfgs.py``; the HV1V2-specific deviations
are commented where they occur.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_hv1v2 import mdp as hv1v2_mdp
from mjlab_hv1v2.hv1v2_constants import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  HAND_BODY_NAMES,
  HOME_KEYFRAME,
  HV1V2_ACTION_SCALE,
  LEG_ACTION_SCALE,
  LEG_JOINTS,
  UPPER_BODY_JOINTS,
  UPPER_BODY_SAMPLE_RANGE,
  get_hv1v2_robot_cfg,
)

# HV1V2 is 92.682 kg against G1's ~35 kg. Angular momentum scales with mass and
# the reward term squares it, so reusing G1's -0.02 would price rotation ~7x
# harder here; H1-V1 measured 100% fell_over at ~20 steps doing exactly that.
_MASS_RATIO = 35.0 / 92.682
ANGULAR_MOMENTUM_WEIGHT = -0.02 * _MASS_RATIO**2  # ~= -0.0029

##
# Observation history.
#
# mjlab's velocity task ships a memoryless actor: one frame, no past. That is
# the default carried forward, not a considered choice, and it is a poor fit
# here for two concrete reasons.
#
# 1. This env redraws mass, friction, PD gains, armature, payload, push and CoM
#    offset fresh every episode. A memoryless actor cannot tell which draw it is
#    on, so it cannot be right for the plant it is standing on -- only safe for
#    all of them at once, and it pays for every range we widen.
# 2. Base linear velocity appears in NO actor term in the primary task (it comes
#    from a velocimeter the robot does not have). One frame cannot form it;
#    `actions` against `joint_pos` one step later can.
#
# 5 frames at 50 Hz control is 100 ms, matching H1-V1. It is NOT claimed optimal
# -- 100 ms is short of the ~312 ms pendulum time constant, and 10 is the named
# follow-up if the velocity argument turns out to be the binding one.
#
# COST: only the actor's first layer grows, 512x102 -> 512x498. Microseconds at
# a 20 ms control period. But it also makes every earlier HV1V2 checkpoint
# shape-incompatible -- set OBS_HISTORY = 1 to restore the old layout exactly.
##

OBS_HISTORY = 5
# Terms that stay at one frame:
#   command     constant within a resample window; 5 copies = 12 wasted inputs.
#   height_scan 187 numbers per frame on the rough task. Stacking it would add
#               748 inputs and take the rough actor from 700 to 1448, for a
#               terrain map that moves by about one 0.1 m grid cell over the
#               100 ms window. The little forward-motion signal it does carry is
#               already available from joint_pos/actions history. (Not an issue
#               on the flat tasks, which delete height_scan outright.)
OBS_HISTORY_EXCLUDE = ("command", "height_scan")


def _add_obs_history(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Give the ACTOR a short observation history. Critic is left alone.

  Two things here are load-bearing and neither is obvious:

  * **Per-term, not the group-level override.** Setting
    ``ObservationGroupCfg.history_length`` would clobber EVERY term's
    ``history_length`` *and* ``flatten_history_dim``
    (``observation_manager.py:461-463``), so ``command`` could not stay at one
    frame. mjlab's own docs claim the opposite precedence
    (``observations.rst:110``: "Term-level settings override group-level") --
    that is wrong; ``tests/test_observation_history.py:233`` asserts the code.

  * **dataclasses.replace, never in-place.** ``make_velocity_env_cfg`` builds
    ``critic_terms = {**actor_terms, ...}``, a shallow copy, so the actor and
    critic groups SHARE term objects. ``term.history_length = 5`` would widen
    the critic from 117 to 585 without any error. check_model.py asserts the
    critic width for exactly this reason.

  ``command`` is excluded: it is constant within a resample window, so five
  copies would be twelve wasted inputs.
  """
  actor = cfg.observations["actor"]
  terms = dict(actor.terms)
  for name, term in terms.items():
    if name in OBS_HISTORY_EXCLUDE:
      continue
    # flatten_history_dim must stay True: rsl-rl's MLPModel raises
    # "only supports 1D observations" on an unflattened (B, H, D) tensor.
    terms[name] = dataclasses.replace(
      term, history_length=OBS_HISTORY, flatten_history_dim=True
    )
  cfg.observations = dict(cfg.observations)
  cfg.observations["actor"] = dataclasses.replace(actor, terms=terms)
  return cfg


def hv1v2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """HV1V2 rough-terrain velocity tracking."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  # G1's 70 is not enough here: each of the 8 foot pads is a box, and a
  # box-against-terrain contact generates up to 4 points, so the feet alone can
  # ask for ~32 before any link hull touches the ground. Measured overflow at
  # 104 on rough terrain; 256 leaves headroom for an unlucky seed.
  cfg.sim.nconmax = 256

  cfg.scene.entities = {"robot": get_hv1v2_robot_cfg()}

  # --- sensors -------------------------------------------------------------
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # Not in the base cfg: without it, air_time / foot_swing_height / foot_slip /
  # soft_landing and the critic's foot terms all fail to resolve.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Mandatory here: 6 shoulders and 2 elbows are live, so the arms can reach
  # the torso and the legs.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # --- actions -------------------------------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # 11 of the 31 entries are 0.0 (waist, wrists, head); see PINNED_JOINTS.
  # The locked waist is a v5 REQUIREMENT now, not just an inherited H1-V1
  # setting: the PD holds it at the init pose (kp 400 roll/pitch, 200 yaw).
  joint_pos_action.scale = HV1V2_ACTION_SCALE

  # --- viewer / commands ---------------------------------------------------
  cfg.viewer.body_name = "torso_link"
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.30  # head reaches 1.841 m

  # --- events / domain randomisation ---------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  )
  # Wider than H1-V1's (0.8, 1.2) on purpose. Our armature is vendor
  # J*ratio^2, which assumes the published inertia is motor-side -- something
  # the vendor never states (ACTUATOR_DATASHEET.txt note 2). Until a pendulum
  # decay measurement settles it, be robust across the uncertainty rather than
  # fitted to the middle of it.
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "ranges": (0.6, 1.4),
      "operation": "scale",
    },
  )
  # Never train against more torque than the hardware has.
  cfg.events["effort_limits"] = EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "effort_limit_range": (0.85, 1.0),
      "operation": "scale",
    },
  )
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  cfg.events["push_robot"].interval_range_s = (2.0, 4.0)

  # --- command curriculum --------------------------------------------------
  # mjlab's default ladder is step-scheduled, not performance-gated, and tops
  # out at lin_vel_x (-2.0, 3.0). v3 measured ~0.78 m/s actual against a 1.0
  # command, so from iteration 10000 almost every command was unreachable.
  # v4 laddered up to (-1.2, 1.5).
  #
  # v5 caps the range for the short-step gait: speed = step length x cadence,
  # and step_length_excess caps the first at 0.30 m, so 1.0 m/s already needs
  # ~3.3 steps/s. ONE stage on purpose: v5 fine-tunes from v4, and resuming
  # restores common_step_counter, so a ladder would start on its last rung.
  # lin_vel_y was never set before and ran at mjlab's +-1.0 default.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.8, 1.0),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.7, 0.7),
    },
  ]

  # --- rewards -------------------------------------------------------------
  # Tracking above the 2.0 default: at 2.0 against pose 1.0, a 93 kg humanoid
  # maximises reward by standing still and collecting upright + pose.
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].weight = 3.0

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms are deliberately TIGHT (v5 requirement: arms rarely used in
    # locomotion). v4 loosened shoulder pitch to 0.4 to buy counter-swing and
    # measured 0.80 rad/s arm RMS speed at vx 1.0. The arms stay live (not
    # pinned) so they can still help recover from a push; arm_joint_vel
    # prices using them.
    r".*shoulder_pitch.*": 0.1,
    r".*shoulder_roll.*": 0.08,
    r".*shoulder_yaw.*": 0.08,
    # Elbow stays tight so it holds the 0.9 rad bend rather than straightening.
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.3,
    # HV1V2 has a head; H1-V1 welds it and G1 has none. A joint matched by no
    # pattern would resolve to None here.
    r".*head.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
    r".*head.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = ANGULAR_MOMENTUM_WEIGHT

  # air_time must stay ON for a robot this heavy. mjlab's base cfg and the G1
  # config both leave the weight at 0.0, and a 2600-iteration run at 4096 envs
  # showed exactly what that buys: peak_height_mean sat at 3 mm against an 80 mm
  # clearance target and slip_velocity_mean *rose* 0.061 -> 0.074. With nothing
  # rewarding lifting a foot, sliding is a valid optimum.
  #
  # But it pays +1 per foot for EVERY step spent inside its window, so the
  # window also sets how long a swing wants to be. v4 ran H1-V1's 0.75 and
  # 0.25-0.6 s and measured 0.35-0.53 s swings in sim2sim (longest at the
  # slowest commands). v5 wants short, frequent steps: a lower weight and a
  # window that stops paying at 0.35 s, with air_time_excess pricing overruns.
  cfg.rewards["air_time"].weight = 0.3
  # The base command_threshold of 0.5 would switch the term off across most of
  # our commanded range.
  cfg.rewards["air_time"].params["threshold_min"] = 0.10
  cfg.rewards["air_time"].params["threshold_max"] = 0.35
  cfg.rewards["air_time"].params["command_threshold"] = 0.1

  # Bound the swing from the other side (see mdp.py "Gait shaping").
  cfg.rewards["feet_flight"] = RewardTermCfg(
    func=hv1v2_mdp.feet_flight,
    weight=-1.0,
    params={"sensor_name": feet_ground_cfg.name},
  )
  cfg.rewards["air_time_excess"] = RewardTermCfg(
    func=hv1v2_mdp.air_time_excess,
    weight=-2.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "t_max": 0.40,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  # 0.30 m = 0.75 x v4's 0.395 m measured at vx 1.0 (sim2sim, heading axis).
  cfg.rewards["step_length_excess"] = RewardTermCfg(
    func=hv1v2_mdp.step_length_excess,
    weight=-2.0,
    params={
      "sensor_name": feet_ground_cfg.name,
      "max_length": 0.30,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  )
  # Arms rarely used: price arm motion directly, on top of the tight pose std.
  cfg.rewards["arm_joint_vel"] = RewardTermCfg(
    func=envs_mdp.joint_vel_l2,
    weight=-0.02,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(r".*_shoulder_.*_joint$", r".*_elbow_joint$")
      )
    },
  )

  cfg.rewards["foot_clearance"].params["target_height"] = 0.08
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.08

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  # Price torque, so the policy is not free to live at the 3-10 second peak
  # rating. Squared term, so it needs the same mass rescaling as the angular
  # momentum term did.
  cfg.rewards["joint_torques"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=-2.0e-6 * _MASS_RATIO**2,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  # --- metrics -------------------------------------------------------------
  # effort_limit is PEAK; these say whether the learned gait is one the
  # hardware could actually hold. Read them against the vendor stall table.
  cfg.metrics["torque_duty_frac"] = MetricsTermCfg(func=hv1v2_mdp.torque_duty_frac)
  cfg.metrics["torque_frac_of_continuous"] = MetricsTermCfg(
    func=hv1v2_mdp.torque_frac_of_continuous
  )
  cfg.metrics["torque_frac_legs"] = MetricsTermCfg(func=hv1v2_mdp.torque_frac_legs)
  # Gait shape, comparable with scripts/sim2sim.py --headless. Step length and
  # swing time are logged by step_length_excess as Metrics/*_mean.
  cfg.metrics["double_flight_frac"] = MetricsTermCfg(
    func=hv1v2_mdp.double_flight_frac,
    params={"sensor_name": feet_ground_cfg.name},
  )
  cfg.metrics["arm_joint_speed"] = MetricsTermCfg(func=hv1v2_mdp.arm_joint_speed)

  # --- play ----------------------------------------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={}
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  # Applied last, and only here: hv1v2_flat_env_cfg derives from this function
  # and only *deletes* actor terms afterwards, so the survivors keep their
  # history and all three registered tasks stay consistent by construction.
  cfg = _add_obs_history(cfg)

  return cfg


def hv1v2_flat_env_cfg(
  play: bool = False, state_estimation: bool = False
) -> ManagerBasedRlEnvCfg:
  """HV1V2 flat-terrain velocity tracking.

  Args:
    play: evaluation mode (no corruption, no pushes, infinite episode).
    state_estimation: whether the ACTOR may observe base linear velocity.
      Defaults to False, because on hardware that signal comes from a base
      state estimator HV1V2 does not have. The critic keeps it either way.
  """
  cfg = hv1v2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No terrain to scan on a plane.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if not state_estimation:
    # base_lin_vel is fed by the MJCF velocimeter. There is no counterpart on
    # the robot without a base state estimator, so an actor trained on it
    # cannot be deployed as-is. It stays in the critic (asymmetric
    # actor-critic), which costs nothing at deploy time.
    del cfg.observations["actor"].terms["base_lin_vel"]

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # Matches the trained range; v5 never sees anything faster.
    twist_cmd.ranges.lin_vel_x = (-0.8, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def _legs_only(std: dict[str, float]) -> dict[str, float]:
  return {k: v for k, v in std.items() if any(t in k for t in ("hip", "knee", "ankle"))}


# v7 smoothness shaping: reward -> (weight at the start of the ramp, final
# weight). Ramped linearly over SMOOTH_RAMP_ITERS, so the policy first learns
# to stand at v6's weights; full smoothness pressure from iteration 0 rewards a
# robot that barely moves. joint_acc / torque_rate finals are CALIBRATED: each
# costs ~3% of v6's per-step reward when evaluated at v6's behaviour (see
# mjlab_hv1v2/README.md "v7"), because their raw scale (qacc includes substep
# contact spikes) is not something to guess.
HIP_KNEE_JOINTS_V7 = (
  r".*_hip_pitch_joint$", r".*_hip_roll_joint$", r".*_hip_yaw_joint$", r".*_knee_joint$",
)
HIP_YAW_JOINTS = (r".*_hip_yaw_joint$",)
SMOOTH_RAMP_ITERS = (1000, 5000)
#
# Calibration (v6 policy rolled out in this env, upper-body level 1.0, every
# new term at weight 1; v6's per-step reward from its own terms = 7.20):
#
#   term                   raw/step   final weight   share of v6 reward
#   action_rate_l2           1.016       -0.4           5.6%  (v6: -0.1, 1.4%)
#   action_acc_l2            1.77        -0.2           4.9%
#   hip_knee_joint_acc       5489        -4e-5          3.0%
#   hip_knee_torque_rate     8.33e6      -2.5e-8        2.9%
#   hip_yaw_action_rate      0.058       -3.5           2.8%
#   hip_yaw_vel              0.77        -0.14          1.5%  (half: turning
#                                                              needs hip yaw)
# The guessed torque-rate weight (-1e-6) would have cost 116% of the reward.
SMOOTH_WEIGHTS: dict[str, tuple[float, float]] = {
  "action_rate_l2": (-0.1, -0.4),       # target mean |da| <= 0.08 (v6 0.122)
  "action_acc_l2": (0.0, -0.2),         # |d2a| -> v4's 0.056; the 3-8 Hz wobble
  "hip_knee_joint_acc": (0.0, -4.0e-5),
  "hip_knee_torque_rate": (0.0, -2.5e-8),
  "hip_yaw_action_rate": (0.0, -3.5),   # extra, on top of action_rate_l2
  "hip_yaw_vel": (0.0, -0.14),
}


def hv1v2_decoupled_env_cfg(
  play: bool = False, smooth: bool = False
) -> ManagerBasedRlEnvCfg:
  """v6: the policy drives the 12 leg joints ONLY; the upper body is someone
  else's (a manipulation/IK controller on hardware) and is a disturbance.

  Built on the flat task, so the plant, DR, observation history and the v5
  short-step gait shaping all carry over. What changes:

  * action: 12 legs. The other 19 joints get their targets from
    UpperBodyCommand -- without it mjlab would drive them to 0 rad.
  * every term that charged the legs for upper-body motion they cannot control
    is re-scoped to the legs/pelvis or removed.
  * standing still while the upper body moves is rewarded explicitly, and the
    upper body carries a payload and takes hand forces.

  Waist roll/pitch are locked at home through a zero-width sample range; waist
  yaw, arms, wrists and head move (UPPER_BODY_SAMPLE_RANGE).
  """
  cfg = hv1v2_flat_env_cfg(play=play)

  # --- actions -------------------------------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = LEG_JOINTS
  joint_pos_action.scale = LEG_ACTION_SCALE

  # --- commands ------------------------------------------------------------
  cfg.commands["upper_body"] = hv1v2_mdp.UpperBodyCommandCfg(
    resampling_time_range=(1.0, 3.0),
    joint_names=UPPER_BODY_JOINTS,
    sample_range=UPPER_BODY_SAMPLE_RANGE,
    # play: a fixed, clearly visible amount of motion; no curriculum runs.
    initial_level=0.5 if play else 0.1,
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Manipulation mostly happens standing, and standing still under upper-body
  # motion is the requirement, so it gets 3x the flat task's share.
  twist_cmd.rel_standing_envs = 0.3

  # --- observations ----------------------------------------------------------
  # Actor: unchanged terms. It already sees all 31 joint positions/velocities,
  # i.e. the measured upper-body state, which is the disturbance signal; only
  # `actions` shrinks to 12. Critic: also the upper-body target (privileged).
  cfg.observations["critic"].terms["upper_body"] = ObservationTermCfg(
    func=envs_mdp.generated_commands, params={"command_name": "upper_body"}
  )

  # --- rewards ---------------------------------------------------------------
  legs = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
  pose = cfg.rewards["pose"]
  pose.params["asset_cfg"] = legs
  for key in ("std_walking", "std_running"):
    pose.params[key] = _legs_only(pose.params[key])
  # torso_link sits above the (moving) waist; the pelvis is what the legs hold.
  cfg.rewards["upright"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("pelvis",)
  )
  cfg.rewards["body_ang_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("pelvis",)
  )
  # Whole-body subtree momentum is mostly arms and torso -- not the legs' to fix.
  cfg.rewards.pop("angular_momentum")
  cfg.rewards.pop("arm_joint_vel")
  cfg.rewards["joint_torques"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LEG_JOINTS
  )
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=LEG_JOINTS
  )

  cfg.rewards["stand_still_feet"] = RewardTermCfg(
    func=hv1v2_mdp.stand_still_feet,
    weight=-1.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  )
  cfg.rewards["stand_feet_contact"] = RewardTermCfg(
    func=hv1v2_mdp.stand_feet_contact,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )

  # Self-collision: only leg against leg is the policy's doing. A hand
  # brushing a hip (1.2% of uniformly sampled poses) is the upper body's.
  sensors = []
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "self_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor = dataclasses.replace(
        sensor,
        primary=ContactMatch(mode="subtree", pattern="left_hip_pitch_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="right_hip_pitch_link", entity="robot"),
      )
    sensors.append(sensor)
  cfg.scene.sensors = tuple(sensors)

  # --- events: what manipulation does to the legs ----------------------------
  # Payload: a point mass at each hand, redrawn every episode. 0-2 kg per hand
  # (user decision); `add` is applied to the DEFAULT mass, so it never stacks.
  cfg.events["hand_payload"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES),
      "ranges": (0.0, 2.0),
      "operation": "add",
    },
  )
  # Contact with the world: a constant force at each hand, redrawn every
  # 2-5 s (it never expires on its own, so the interval IS its duration).
  cfg.events["hand_force"] = EventTermCfg(
    mode="interval",
    interval_range_s=(2.0, 5.0),
    func=envs_mdp.apply_external_force_torque,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES),
      "force_range": (-20.0, 20.0),
      "torque_range": (0.0, 0.0),
    },
  )

  # --- curriculum --------------------------------------------------------------
  if not play:
    cfg.curriculum["upper_body_level"] = CurriculumTermCfg(
      func=hv1v2_mdp.upper_body_level,
      params={"command_name": "upper_body"},
    )

  if smooth:
    _add_smoothness(cfg, play)

  return cfg


def _add_smoothness(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """v7: price jerk on the legs. See SMOOTH_WEIGHTS for the numbers."""
  hip_knee = SceneEntityCfg("robot", joint_names=HIP_KNEE_JOINTS_V7)
  cfg.rewards["action_acc_l2"] = RewardTermCfg(func=envs_mdp.action_acc_l2, weight=0.0)
  cfg.rewards["hip_knee_joint_acc"] = RewardTermCfg(
    func=envs_mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": hip_knee}
  )
  cfg.rewards["hip_knee_torque_rate"] = RewardTermCfg(
    func=hv1v2_mdp.torque_rate_l2, weight=0.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_KNEE_JOINTS_V7)},
  )
  cfg.rewards["hip_yaw_action_rate"] = RewardTermCfg(
    func=hv1v2_mdp.action_rate_l2_joints, weight=0.0,
    params={"joint_names": HIP_YAW_JOINTS},
  )
  cfg.rewards["hip_yaw_vel"] = RewardTermCfg(
    func=envs_mdp.joint_vel_l2, weight=0.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_YAW_JOINTS)},
  )
  start, end = SMOOTH_RAMP_ITERS
  for name, (w0, w1) in SMOOTH_WEIGHTS.items():
    # play: no curriculum runs, so evaluate at the final weights.
    cfg.rewards[name].weight = w1 if play else w0
    if not play:
      cfg.curriculum[f"ramp_{name}"] = CurriculumTermCfg(
        func=hv1v2_mdp.reward_weight_ramp,
        params={"reward_name": name, "w0": w0, "w1": w1,
                "start_iter": start, "end_iter": end},
      )

  cfg.metrics["leg_action_rate"] = MetricsTermCfg(func=hv1v2_mdp.leg_action_rate)
  cfg.metrics["leg_action_acc"] = MetricsTermCfg(func=hv1v2_mdp.leg_action_acc)
  cfg.metrics["hip_yaw_action_rate"] = MetricsTermCfg(func=hv1v2_mdp.hip_yaw_action_rate)


##
# v8 (Mjlab-Velocity-Flat-HV1V2-Decoupled-v8): v7, trained from scratch with
# v7's exact recipe (rewards, weights, smoothness ramps, commands, DR, obs
# 403/117, 12-D action, PPO), plus:
#   1. upper-body motion: each v7 resample is, half the time, a synthetic
#      Mink IK -> online Ruckig clip (data/clips_v2, 17 joints, waist
#      roll/pitch locked) instead of v7's random constant-speed goal;
#   2. robot timing: legs and upper body linearly interpolated across the
#      substeps as merai_control does, legs with a 0-20 ms delay;
#   3. ankle stiffness DR up to the real four-bar's 1.9x (pitch) / 1.4x
#      (roll) joint stiffness.
# Why 2 and 3: v7 does not rock by itself in sim (arms still: 0 in every
# condition); its 1-2 Hz hip/pelvis rocking is the response to arm motion,
# and interpolation (+41%), stiffer ankles (+44%) and delay (+19%) each
# amplify it -- 3.3x together. v7 never trained on any of them.
# No reward changes: no pelvis_quiet / leg_motion_standing (the deleted v8
# used them and traded rocking for slow sway and lean).
##

CLIP_DIR_V2 = str(Path(__file__).resolve().parents[2] / "data" / "clips_v2")
V8_CLIP_PROB = 0.5                   # P(clip | upper-body resample)
V8_DELAY_SUBSTEPS = (0, 4)           # 0-20 ms in 5 ms substeps
V8_ANKLE_PITCH_K = (1.0, 1.9)        # x sim joint kp/kd, after v7's pd_gains DR
V8_ANKLE_ROLL_K = (1.0, 1.4)
# Trained at 8192 envs x 15k iterations = v7's 4096 x 30k samples (5.7 h vs
# 7.8 h on the remote). v7's smoothness ramps (iters 1000-5000 at 4096 envs)
# are keyed on iterations, so they are halved to the same sample counts. The
# upper_body_level curriculum counts episodes, already sample-based.
V8_SMOOTH_RAMP_ITERS = (500, 2500)


def hv1v2_decoupled_v8_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = hv1v2_decoupled_env_cfg(play=play, smooth=True)   # v7
  for name, term in cfg.curriculum.items():
    if name.startswith("ramp_"):
      term.params["start_iter"], term.params["end_iter"] = V8_SMOOTH_RAMP_ITERS

  # --- 1. upper body: v7's generator + synthetic Ruckig clips -----------------
  ub = cfg.commands["upper_body"]
  cfg.commands["upper_body"] = hv1v2_mdp.UpperBodyMixedCommandCfg(
    **{f.name: getattr(ub, f.name) for f in dataclasses.fields(ub)},
    clip_dir=CLIP_DIR_V2,
    clip_prob=V8_CLIP_PROB,
    write_targets=False,          # UpperBodyActuation writes them per substep
  )

  # --- 2. robot timing ----------------------------------------------------------
  old = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = hv1v2_mdp.InterpolatedJointPositionActionCfg(
    **{f.name: getattr(old, f.name) for f in dataclasses.fields(old)},
    delay_substeps_range=V8_DELAY_SUBSTEPS,
  )
  cfg.actions["upper_body_actuation"] = hv1v2_mdp.UpperBodyActuationCfg(
    entity_name="robot", command_name="upper_body"
  )

  # --- 3. ankle stiffness (startup; inserted after v7's pd_gains) ---------------
  cfg.events["ankle_stiffness"] = EventTermCfg(
    mode="startup", func=hv1v2_mdp.ankle_stiffness,
    params={"pitch_range": V8_ANKLE_PITCH_K, "roll_range": V8_ANKLE_ROLL_K},
  )

  # --- logging only (no effect on training) -------------------------------------
  # push_and_stamp = v7's push + a timestamp, so standing_steps can skip push
  # recoveries.
  if "push_robot" in cfg.events:   # play mode has no pushes
    cfg.events["push_robot"].func = hv1v2_mdp.push_and_stamp
  cfg.metrics["standing_steps"] = MetricsTermCfg(
    func=hv1v2_mdp.standing_steps, reduce="sum",
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
  )
  return cfg


##
# v8b (Mjlab-Velocity-Flat-HV1V2-Decoupled-v8b): v8 + fixes for its arms-still
# self-sway and heading drift. Warm-started from v8's final checkpoint.
#   * v8 trained with the upper body completely still only 35% of the time
#     (v7: 68%) and sways at 0.73 Hz when the arms are held (v7: none). Clips
#     now hold their final pose for U(1, 4) s, and 15% of clip draws are
#     "go home and hold" (v7's quiet_prob) -> arms-still share back near v7.
#   * heading: a yaw-BIAS term (low-passed yaw-rate error, std 0.05 rad/s);
#     v7's wide kernel (std 0.71) barely prices the 0.03 rad/s bias that made
#     v8 turn 35 deg in 20 s.
#   * still_stand_quiet (contingency C1) is wired at weight 0; Metrics/
#     still_sway tracks the self-sway during training.
#   * smoothness weights at their finals (a warm start would re-run v8's ramp
#     from v6's weights).
##

V8B_HOLD_AFTER_CLIP_S = (4.0, 16.0)
V8B_CLIP_QUIET_PROB = 0.15
# 0.5 x (1 - exp(-(0.03/0.05)^2)) = 0.15/step for v8's 0.03 rad/s bias, ~2% of
# the ~6.8 per-step reward; v7's wide term charges 3 x (1 - exp(-0.03^2/0.5))
# = 0.005 for the same bias.
V8B_YAW_BIAS = dict(weight=0.5, std=0.05, tau=1.0)
# C1, weight 0 unless enabled. Candidate if needed: calibrate to ~2% of reward
# on v8b's own still-standing sway.
V8B_STILL_STAND_WEIGHT = 0.0


def hv1v2_decoupled_v8b_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = hv1v2_decoupled_v8_env_cfg(play=play)

  # warm start: smoothness straight at v7's final weights, no ramp
  for name, (_, w1) in SMOOTH_WEIGHTS.items():
    cfg.rewards[name].weight = w1
    cfg.curriculum.pop(f"ramp_{name}", None)

  ub = cfg.commands["upper_body"]
  assert isinstance(ub, hv1v2_mdp.UpperBodyMixedCommandCfg)
  ub.hold_after_clip_range = V8B_HOLD_AFTER_CLIP_S
  ub.clip_quiet_prob = V8B_CLIP_QUIET_PROB

  cfg.rewards["track_yaw_bias"] = RewardTermCfg(
    func=hv1v2_mdp.track_yaw_bias, weight=V8B_YAW_BIAS["weight"],
    params={"std": V8B_YAW_BIAS["std"], "tau": V8B_YAW_BIAS["tau"],
            "command_name": "twist"},
  )
  cfg.rewards["still_stand_quiet"] = RewardTermCfg(
    func=hv1v2_mdp.still_stand_quiet, weight=V8B_STILL_STAND_WEIGHT,
    params={"command_name": "twist", "upper_name": "upper_body", "min_still_s": 1.0},
  )
  cfg.metrics["still_sway"] = MetricsTermCfg(
    func=hv1v2_mdp.still_sway,
    params={"command_name": "twist", "upper_name": "upper_body", "min_still_s": 1.0},
  )
  return cfg


##
# v10 (Mjlab-Velocity-Flat-HV1V2-Decoupled-v10): v8b + a flat, relaxed,
# re-aligned stance, warm-started from v7's final checkpoint.
#
# Why (logs/eval/v7_stance, probe_stance.py): v7 stands with its right foot on
# the outer edge (2 of 4 pads, 25 N.m steady ankle-roll torque -- its target sits
# 3.6-5.4 deg past where the floor lets the foot go), leans ~1 deg in roll (its
# anchor was solved against the mis-modelled left knee: on the real geometry the
# left foot is 3.8 mm low), and leaves the feet 6-10 deg apart in yaw after a
# push because no term pays for a tidy-up step. Changes:
#   1. HOME_KEYFRAME (corrected knee) -- asserted, see below.
#   2. stand_sole_tilt: price a tilted sole while standing.
#   3. stand_leg_action_l2: price leg targets away from the (now consistent)
#      anchor while standing, i.e. the preload.
#   4. stand_stance_shape: price foot-yaw difference and width error while
#      standing, so one corrective step pays for itself.
#   5. pose std_standing 0.02 rad on ankle roll (others stay 0.05).
#   6. ankle stiffness DR covers the hardware as it runs today: motor_gain 0.45
#      gives ~1.0x pitch / 0.67-0.72x roll (merai hardware.yaml, 2026-10-24).
# Weights are calibrated on v7 rollouts in this env: each term priced at
# V10_TARGET_FRAC of the per-step reward of standing envs (see
# scripts/calibrate_v10.py, which prints the numbers below).
##

V10_ANKLE_PITCH_K = (0.9, 1.9)
V10_ANKLE_ROLL_K = (0.65, 1.4)
V10_NOMINAL_WIDTH = 0.243     # m, ankle_roll_link lateral separation at HOME_KEYFRAME
V10_STANCE_YAW_W = 1.0        # per rad of foot-yaw difference, inside stand_stance_shape
V10_STANCE_WIDTH_W = 2.0      # per m of width error
# Calibrated 2026-10-26 on v7 model_29999 in this env (2048 envs x 30 s, 922k
# standing env-steps): base reward 0.149 per standing step; mean raw values
# sole tilt 0.0256 rad (both feet), leg action 3.51 (sum a^2, i.e. v7 holds
# ~0.54 per joint at rest), stance shape 0.119. Each priced at 2 %.
V10_WEIGHTS = dict(
  stand_sole_tilt=-5.8,
  stand_leg_action_l2=-0.042,
  stand_stance_shape=-1.25,
)
_FEET = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))


def hv1v2_decoupled_v10_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = hv1v2_decoupled_v8b_env_cfg(play=play)

  # 1. anchor: a v1-v8 pose here would re-introduce the 3.8 mm foot-height error
  init = cfg.scene.entities["robot"].init_state
  assert init.joint_pos == HOME_KEYFRAME.joint_pos and init.pos == HOME_KEYFRAME.pos, (
    "v10 must train on HOME_KEYFRAME (corrected knee), not HOME_KEYFRAME_V1_V8"
  )
  # warm start from a fully trained v7: start the upper-body curriculum at the
  # top instead of re-climbing from 0.1 (it can still demote).
  cfg.commands["upper_body"].initial_level = 1.0

  standing = {"command_name": "twist", "command_threshold": 0.1}
  cfg.rewards["stand_sole_tilt"] = RewardTermCfg(
    func=hv1v2_mdp.stand_sole_tilt, weight=V10_WEIGHTS["stand_sole_tilt"],
    params={"sensor_name": "feet_ground_contact", "asset_cfg": _FEET, **standing},
  )
  cfg.rewards["stand_leg_action_l2"] = RewardTermCfg(
    func=hv1v2_mdp.stand_leg_action_l2, weight=V10_WEIGHTS["stand_leg_action_l2"],
    params=dict(standing),
  )
  cfg.rewards["stand_stance_shape"] = RewardTermCfg(
    func=hv1v2_mdp.stand_stance_shape, weight=V10_WEIGHTS["stand_stance_shape"],
    params={"asset_cfg": _FEET, "nominal_width": V10_NOMINAL_WIDTH,
            "yaw_weight": V10_STANCE_YAW_W, "width_weight": V10_STANCE_WIDTH_W, **standing},
  )

  # 5. resolve_matching_names_values rejects overlapping patterns, so the
  # five non-roll leg joints are listed instead of a catch-all.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_hip_pitch_joint$": 0.05, r".*_hip_roll_joint$": 0.05,
    r".*_hip_yaw_joint$": 0.05, r".*_knee_joint$": 0.05,
    r".*_ankle_pitch_joint$": 0.05, r".*_ankle_roll_joint$": 0.02,
  }

  # 6.
  cfg.events["ankle_stiffness"].params["pitch_range"] = V10_ANKLE_PITCH_K
  cfg.events["ankle_stiffness"].params["roll_range"] = V10_ANKLE_ROLL_K
  return cfg


##
# v11 (Mjlab-Velocity-Flat-HV1V2-Decoupled-v11), 2026-11-01. FRESH training
# (no warm start). Plan: ~/.claude/plans/keen-pondering-fiddle.md; terms in
# mdp_v11.py, mirror maps in symmetry.py.
#
# v7 tilted with every arm/head/waist move and held one ankle rolled. Why, in
# v7's own config: tilt was nearly free (upright std 0.447 pays 0.96 at 5 deg;
# body_ang_vel -0.05), the actor never saw the upper-body target, nothing
# priced a flat foot while walking, and the plant had no latency, ZOH targets,
# no four-bar stiffness, continuous-only torque and waist yaw at 200/12.
# v11 = v8 (timing, clips, stiffness DR) + v8b (holds, yaw bias) + v10 (stance
# terms) as FRESH-training settings, plus everything below. Decisions (user):
# waist roll/pitch stay locked; payload 0-2 kg per hand.
#
# Plant spec (Phase 0 without the robot; hv1v2_constants.py "v11 plant"):
# ankle pitch/roll/waist-rp stiffness centred on the measured 1.0x / 0.7x /
# 3.5x with ranges wide enough to absorb the unresolved X8 1.16x question.
#
# Deviation from the plan, documented: an observation delay of 0-10 ms is not
# representable (mjlab delays observations in whole 20 ms policy steps); it is
# covered by the 0-20 ms leg action delay instead.
##

V11_SMOOTH_RAMP_ITERS = (500, 2500)        # v7's 1000-5000 at 4096 envs, at 8192
V11_DR_RAMP_ITERS = (1500, 6000)           # DR half-widths 50% -> 100%
V11_ANKLE_PITCH_K = (0.8, 1.45)
V11_ANKLE_ROLL_K = (0.5, 1.1)
V11_WAIST_RP_K = (2.5, 4.5)
V11_ENCODER_BIAS_LEGS = 0.0524             # 3 deg
V11_IMU_RP_BIAS = 0.0131                   # 0.75 deg
V11_HISTORY = 10                            # proprioception frames (200 ms)
V11_FZ_MIN = 20.0                           # N, a foot / pad "carrying load"
# Reward weights for the new terms, calibrated 2026-11-01 by
# scripts/calibrate_v11.py: v7 model_29999 rolled out in the v10 env (v7's obs
# layout, v8 timing, real ankle stiffness), 1024 envs x 20 s = 971,771
# env-steps; v7 earns 0.1309 per step there. Each term is priced at the share
# of that reward shown:
#   term                      mean raw on v7   share   weight
#   torso_quiet_under_ub      0.00235          4%      -111.4
#   loaded_sole_tilt          0.0315 rad       3%      -6.24
#   pad_load_balance          0.784            2%      -0.167
#   ankle_roll_torque_stance  2274 (N.m)^2     3%      -8.64e-5
#   still_stand_quiet         0.00380          1%      -17.2
#   ankle_joint_acc           2258             3%      -8.7e-5
#   ankle_torque_rate         2.95e6           3%      -6.66e-8
# Same rollouts: v7's pelvis tilt averages 2.08 deg; v7's upright kernel paid
# 0.990 for it, the v11 kernel (std 0.08) pays 0.801. body_ang_vel 10x v7's.
V11_WEIGHTS = dict(
  upright=1.0,
  body_ang_vel=-0.5,
  torso_quiet_under_ub=-111.4,
  loaded_sole_tilt=-6.24,
  pad_load_balance=-0.167,
  ankle_roll_torque_stance=-8.64e-5,
  still_stand_quiet=-17.2,
)
V11_UPRIGHT_STD = 0.08   # (v11_s42 only; superseded by the split below)
#
# 2026-11-01 FIX after v11_s42 (stopped at iteration 2000): with every new
# posture/stance penalty at full weight from iteration 0, the fresh policy
# learned to stand and turn in place and never translate (0.00 m/s at 0.3-1.0
# m/s commands, 4 cm steps, 0.04 s swings; v7 at the same samples: 13 cm,
# 0.10 s). Standing still was the cheapest way to avoid them, and the
# fall-gated upper-body level then reached 1.0 at iteration ~1000. Now:
#   * those penalties ramp from 5% to 100% over iterations 1000-4000 [v11c: 1500-5000] (keyed to
#     SAMPLES at 8192 envs), body_ang_vel -0.05 -> -0.5 over the same span;
#   * upright = wide kernel (std 0.35, weight 0.7) from the start + precision
#     kernel (std 0.06, weight 0 -> 0.3) on the same ramp, so the CoM shift that
#     lifts a foot is not punished before a gait exists;
#   * stance ankle-roll cost x0.25 in walking single support;
#   * the upper-body level only promotes, and the DR ramp only starts, once
#     walking tracks (v11b: walk_track_err EMA < 0.35 m/s).
# v11c (same day): v11b_s43 opened that gate at iteration ~600 on forward
# walking alone (backward -0.07 / lateral 0.04 m/s at 0.3 commands, 0.058 s
# swing, 8.6 cm steps), then the arm level went 0 -> 0.92 in 400 iterations.
# Now the gate needs ALL of: overall error EMA < 0.30, backward and lateral
# error EMAs < 0.32, swing-time EMA > 0.08 s; the level rises at most 0.05 per
# 100 iterations; the posture ramp moves to 1500-5000.
# (Error scale: finished v7 scores 0.25 on the overall error in the v10 env --
# it includes the lag after each command change -- stuck v11_s42 0.54.)
V11_REF_ENVS = 8192
V11_POSTURE_RAMP_ITERS = (1500, 5000)   # v11c (was 1000-4000 in v11b)
V11_POSTURE_RAMP_START = 0.05
V11_POSTURE_RAMPED = ("torso_quiet_under_ub", "pad_load_balance", "ankle_roll_torque_stance",
                      "loaded_sole_tilt", "stand_leg_action_l2", "still_stand_quiet")
V11_BODY_ANG_VEL = (-0.05, -0.5)
V11_UPRIGHT_WIDE = (0.35, 0.7)       # (std, weight), constant
V11_UPRIGHT_PRECISE = (0.06, 0.3)    # (std, final weight), ramped from 0
V11_SINGLE_SUPPORT_SCALE = 0.25
V11_TRACK_OK = 0.30   # v11c (v11b: 0.35, forward-dominated -> opened with no backward/lateral gait)
V11_TRACK_DIR_OK = 0.32   # backward (vx_cmd < -0.15) and lateral (|vy_cmd| > 0.15) EMAs
V11_MIN_SWING_S = 0.08
V11_UB_MAX_RATE = 0.05    # upper_body_level per 100 iterations (8192-env samples)
V11_SMOOTH_EXTRA: dict[str, tuple[float, float]] = {
  "ankle_joint_acc": (0.0, -8.7e-5 / 4),      # v11d: 1/4 (v11c -8.7e-5)
  "ankle_torque_rate": (0.0, -6.66e-8 / 2),   # v11d: 1/2 (v11c -6.66e-8)
}
#
# v11d (2026-11-01), after v11c_s43 walked at iteration 800 (0.44 m/s at a 0.5
# command, -0.36 backward) and then stood still from ~3000 (0.00 m/s at 5000
# and 9600; posture fine: tilt 1.2 deg, ankle-roll 4 N.m). Reward audit at
# 100% weights, 0.5 m/s command, arms level 0.2, no load
# (scripts/reward_audit_v11.py, reward_audit_v7ref.py), per step:
#   model_800 (walks, jerky)  -5.03: ankle_joint_acc -6.13, torso_quiet -2.08,
#                                    loaded_sole_tilt -0.71, track_lin +1.86
#   model_5000 (stands)       +6.45: track_lin +1.11, track_ang +3.00
#   v7 (mature walker)        ~+7.3: ankle_joint_acc -0.08, torso_quiet -0.03,
#                                    sole_tilt -0.12, track_lin +2.82
# i.e. the full budget lets a MATURE walker beat standing, but the price of an
# immature gait (ankle acceleration 75x v7's) makes stopping the gradient
# direction while the ramps rise on iteration count. Changes:
#   * ankle_joint_acc 1/4, ankle_torque_rate 1/2, both on the POSTURE schedule;
#   * every weight ramp (posture + all smoothness) runs on a clock that only
#     advances while the walking error EMA < V11_POSTURE_GATE (v11_gated_clock);
#   * the DR ramp only advances while the walking gate is open;
#   * torso_quiet_under_ub and body_ang_vel x0.25 when |v_cmd_xy| >= 0.05;
#   * air_time x exp(-|v_xy - v_cmd_xy|^2 / 0.25).
V11_POSTURE_GATE = 0.35
V11_WALK_CMD = 0.05
V11_WALK_SCALE = 0.25
V11_AIR_TRACK_VAR = 0.25
V11_GATED_SMOOTH = ("ankle_joint_acc", "ankle_torque_rate")   # ride the posture schedule
#
# v11e (2026-11-02), after v11d_s43 (resumed from v11c model_800) held its
# gait to 6000 but gave up SLOW commands once the posture ramp completed: at
# 0.3 m/s forward 0.22 (it 2400) -> 0.10 (4800) -> 0.03 (6000), lateral 0.22
# -> 0.10; 0.5 m/s 0.41 -> 0.32. Posture reached the targets (standing tilt
# 1.1 deg mean, ankle roll 4-5 N.m). The overall error EMA stayed at 0.30
# throughout, so no gate saw it. Cause: track_linear_velocity's kernel (std^2
# 0.25) pays standing still exp(-0.09/0.25) = 70% of full reward at a 0.3
# command. Changes (resumed from v11d_s43 model_2400):
#   * linear tracking = wide kernel (std^2 0.25, weight 1) + precise kernel
#     (std^2 0.0625, weight 2): standing at 0.3 m/s earns 39% (was 70%), at
#     0.5 m/s 14% (was 37%); total weight unchanged (3);
#   * gate: a slow-command error EMA (0.15 < |v_cmd_xy| < 0.4) joins the
#     walking gate and the weight clock;
#   * the clock REWINDS (weights ease off) when the overall error EMA > 0.30 or
#     the slow one > V11_SLOW_REWIND; advances below 0.28 / V11_SLOW_OK.
#     (probe, arms at 1.0: overall 0.269 at model_2400 vs 0.316 at model_6000)
V11_TRACK_LIN_WIDE = (0.25, 3.0)      # (std^2, weight); v11e 1.0, v11f 3.0
V11_TRACK_LIN_PRECISE = (0.05, 2.0)   # v11e std^2 0.0625, v11f 0.05
V11_CLOCK_OK = 0.28
V11_CLOCK_REWIND = 0.30
V11_SLOW_RANGE = (0.15, 0.4)
V11_SLOW_OK = 0.19       # gate_ema_probe.py: v11d model_2400 0.183, model_6000 0.210
V11_SLOW_REWIND = 0.20
#
# v11f (2026-11-02), after v11e_s43 kept slow commands but from iteration
# ~7000 STOOD STILL at >= 0.8 m/s commands (1.0 m/s: 0.72 at 5000 -> 0.04 at
# 8000 -> 0.00 at 17000; it walks to 0.7). Audit at 0.9 m/s (100% weights):
# walking at 0.59 earned only 1.02 of linear tracking (wide weight 1) vs
# standing 0.05, while standing collected ~0.6 more yaw-tracking/pose reward
# and paid no gait costs -> total 5.32 standing vs 4.82 walking. The
# overall EMA sat at 0.29 (hold band), so the clock froze at ~3000 (posture
# at ~45%) from iteration 10000 on. Changes (user-specified), resumed from
# v11e model_6000 (clock 5922, posture already 100%):
#   * wide tracking kernel weight 1 -> 3, precise kernel std^2 0.05, weight 2;
#   * a fast-command EMA (|v_cmd_xy| > 0.7) in the gate and the clock.
V11_FAST_CMD = 0.7
V11_FAST_OK = 0.50      # gate_ema_probe.py: v11e model_6000 0.458, model_8000 0.668, model_17000 0.643
V11_FAST_REWIND = 0.55
# The clock is not saved in checkpoints: on a resume, V11_RESUME_CLOCK=<value>
# in the environment restores it (v11f: 5922 at v11e model_6000).
V11_RESUME_CLOCK = float(os.environ["V11_RESUME_CLOCK"]) if os.environ.get("V11_RESUME_CLOCK") else None
#
# v11g (2026-11-02), after v11f model_12000 ran on the robot: level torso, flat
# feet, but at zero command it held a SQUEEZE -- hip roll -55/+59 N.m (legs
# pushed apart) or a scissor (hip pitch +62/-44 N.m) for 40 s. Offline replay
# of the sessions: merai's obs match training (0.6 deg mean), and zeroing the
# prev-action history removes the squeeze -- the policy holds its own last
# action while the planted feet cancel the forces, invisible to its inputs. In
# sim2sim the same policy drifts into a +-30 N.m squeeze, worse at friction 2.
# stand_leg_action_l2 (-0.042) priced it at ~1% of reward. Changes (resumed
# from v11f's last checkpoint):
#   * stand_leg_torque (new): (|tau| - deadband)^2 on hip roll/pitch + knee
#     while standing; deadbands 10 / 15 / 30 N.m cover holding the body up;
#     the robot's squeeze costs ~1.6/step (spread), ~0.9 (scissor);
#   * stand_leg_action_l2 x5 (-0.042 -> -0.21);
#   * foot friction DR 0.3-1.2 -> 0.3-2.0 (contact uses the max of the pair, so
#     with the floor at ~1 the old range never exceeded 1.2; the robot's pads grip);
#   * both standing terms ramp in over 500 clock iterations from the resume.
V11G_STAND_TAU_DEADBAND = {r".*_hip_roll_joint$": 10.0, r".*_hip_pitch_joint$": 15.0, r".*_knee_joint$": 30.0}
V11G_STAND_TAU_W = -3.0e-4
V11G_STAND_ACTION_W = -0.21
V11G_FRICTION = (0.3, 2.0)
V11G_RAMP_ITERS = 500
_ANKLES = (r".*_ankle_pitch_joint$", r".*_ankle_roll_joint$")


def hv1v2_decoupled_v11_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  from mjlab_hv1v2 import mdp_v11
  from mjlab_hv1v2.hv1v2_constants import (
    FOOT_GEOM_RE,
    V11_UB_OBS_JOINTS,
    get_hv1v2_v11_robot_cfg,
  )
  from mjlab.utils.noise import UniformNoiseCfg as Unoise

  cfg = hv1v2_decoupled_env_cfg(play=play, smooth=True)   # v7, fresh

  # --- A. plant ----------------------------------------------------------------
  cfg.scene.entities = {"robot": get_hv1v2_v11_robot_cfg()}
  init = cfg.scene.entities["robot"].init_state
  assert init.joint_pos == HOME_KEYFRAME.joint_pos, "v11 trains on HOME_KEYFRAME"

  old = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = mdp_v11.V11JointPositionActionCfg(
    **{f.name: getattr(old, f.name) for f in dataclasses.fields(old)},
    delay_substeps_range=(0, 4),     # 0-20 ms, widened with the DR level
    dropout_prob=0.0 if play else 0.001,
  )
  cfg.actions["upper_body_actuation"] = hv1v2_mdp.UpperBodyActuationCfg(
    entity_name="robot", command_name="upper_body"
  )

  cfg.events.pop("pd_gains", None)
  cfg.events["actuator_gains"] = EventTermCfg(
    mode="reset", func=mdp_v11.v11_actuator_gains,
    params={
      "kp_range": (0.85, 1.15), "kd_range": (0.85, 1.15),
      "groups": {
        "ankle_pitch": ((r".*_ankle_pitch_joint$",), V11_ANKLE_PITCH_K, 1.0),
        "ankle_roll": ((r".*_ankle_roll_joint$",), V11_ANKLE_ROLL_K, 0.7),
        "waist_rp": ((r"waist_roll_joint$", r"waist_pitch_joint$"), V11_WAIST_RP_K, 3.5),
      },
    },
  )
  cfg.events["encoder_bias_legs"] = EventTermCfg(
    mode="reset", func=mdp_v11.encoder_bias_scaled,
    params={"bias_max": V11_ENCODER_BIAS_LEGS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
  )
  cfg.events["imu_bias"] = EventTermCfg(
    mode="reset", func=mdp_v11.imu_bias_v11,
    params={"rp_max_rad": V11_IMU_RP_BIAS, "gyro_max": 0.05},
  )
  # Link mass +-10% (mass and inertia together) and CoM +-2 cm on every link
  # except the torso (base_com keeps v7's wider torso range) and the hands
  # (the payload event rewrites them from the defaults every reset).
  _links = SceneEntityCfg("robot", body_names=(
    r"^(?!torso_link$|left_wrist_yaw_link$|right_wrist_yaw_link$).*$",))
  cfg.events["link_mass"] = EventTermCfg(
    mode="startup", func=dr.pseudo_inertia,
    params={"asset_cfg": _links, "alpha_range": (-0.05, 0.05)},
  )
  cfg.events["link_com"] = EventTermCfg(
    mode="startup", func=dr.body_com_offset,
    params={"asset_cfg": _links, "ranges": (-0.02, 0.02), "operation": "add"},
  )

  # --- B. upper body, loads, pushes -------------------------------------------
  ub = cfg.commands["upper_body"]
  cfg.commands["upper_body"] = mdp_v11.V11UpperBodyCommandCfg(
    **{f.name: getattr(ub, f.name) for f in dataclasses.fields(ub)},
    clip_dir=CLIP_DIR_V2,
    clip_prob=V8_CLIP_PROB,
    write_targets=False,
    hold_after_clip_range=V8B_HOLD_AFTER_CLIP_S,
    clip_quiet_prob=V8B_CLIP_QUIET_PROB,
    speed_cap={"waist_yaw_joint": 1.0},
    hold_prob=0.0 if play else 0.02,
  )
  cfg.events.pop("hand_force", None)
  cfg.events["hand_payload"] = EventTermCfg(
    mode="reset", func=mdp_v11.hand_payload_v11,
    params={"body_names": HAND_BODY_NAMES, "mass_range": (0.0, 2.0), "offset_max": 0.10},
  )
  cfg.commands["hand_load"] = mdp_v11.HandLoadCommandCfg(
    resampling_time_range=(1.0, 3.0), body_names=HAND_BODY_NAMES,
    active_prob=0.35, force_max=30.0, rise_range=(0.2, 1.0), hold_range=(1.0, 4.0),
    opposite_prob=0.3, pickup_prob=0.2, pickup_ramp_s=0.2, mass_max=2.0,
  )
  cfg.commands["torso_push"] = mdp_v11.TorsoPushCommandCfg(
    resampling_time_range=(1.0, 4.0), body_names=("torso_link",),
    active_prob=0.0 if play else 0.15, force_max=60.0,
    rise_range=(0.05, 0.2), hold_range=(0.05, 0.3),
  )
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].func = hv1v2_mdp.push_and_stamp

  # --- E. commands --------------------------------------------------------------
  tw = cfg.commands["twist"]
  cfg.commands["twist"] = mdp_v11.ReversalVelocityCommandCfg(
    **{f.name: getattr(tw, f.name) for f in dataclasses.fields(tw)},
    stand_hold_range=(5.0, 15.0), reversal_prob=0.15, stop_prob=0.10,
  )
  cfg.commands["twist"].rel_standing_envs = 0.35
  stage = {"step": 0, "lin_vel_x": (-0.6, 1.0), "lin_vel_y": (-0.5, 0.5),
           "ang_vel_z": (-0.8, 0.8)}
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [stage]
  if play:
    r = cfg.commands["twist"].ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = stage["lin_vel_x"], stage["lin_vel_y"], stage["ang_vel_z"]

  # --- sensors: per-pad contact for pad_load_balance ------------------------------
  pads = ContactSensorCfg(
    name="foot_pads",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_RE, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (pads,)

  # --- C. observations -------------------------------------------------------------
  actor = cfg.observations["actor"]
  terms = dict(actor.terms)
  terms["base_ang_vel"] = dataclasses.replace(
    terms["base_ang_vel"], func=mdp_v11.base_ang_vel_v11,
    params={"sensor_name": "robot/imu_ang_vel",
            "freeze_prob": 0.0 if play else 0.002, "freeze_steps": (1, 5)})
  terms["projected_gravity"] = dataclasses.replace(
    terms["projected_gravity"], func=mdp_v11.projected_gravity_v11,
    params={"freeze_prob": 0.0 if play else 0.002, "freeze_steps": (1, 5)})
  for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"):
    terms[name] = dataclasses.replace(terms[name], history_length=V11_HISTORY,
                                      flatten_history_dim=True)
  ub_params = {"command_name": "upper_body", "joint_names": V11_UB_OBS_JOINTS}
  terms["ub_target"] = ObservationTermCfg(func=mdp_v11.ub_target_rel, params=dict(ub_params))
  terms["ub_target_rate"] = ObservationTermCfg(
    func=mdp_v11.ub_target_rate, params=dict(ub_params),
    noise=Unoise(n_min=-0.1, n_max=0.1))
  cfg.observations = dict(cfg.observations)
  cfg.observations["actor"] = dataclasses.replace(actor, terms=terms)

  critic = cfg.observations["critic"]
  cterms = dict(critic.terms)
  cterms["ub_target_rate"] = ObservationTermCfg(func=mdp_v11.ub_target_rate, params=dict(ub_params))
  cterms["plant"] = ObservationTermCfg(
    func=mdp_v11.privileged_plant, params={"leg_joint_names": LEG_JOINTS})
  cterms["hand_load"] = ObservationTermCfg(
    func=envs_mdp.generated_commands, params={"command_name": "hand_load"})
  cterms["torso_push"] = ObservationTermCfg(
    func=envs_mdp.generated_commands, params={"command_name": "torso_push"})
  cfg.observations["critic"] = dataclasses.replace(critic, terms=cterms)

  # --- D. rewards ------------------------------------------------------------------
  pelvis = SceneEntityCfg("robot", body_names=("pelvis",))
  cfg.rewards["upright"] = RewardTermCfg(
    func=mdp_v11.upright_tight, weight=V11_UPRIGHT_WIDE[1],
    params={"std": V11_UPRIGHT_WIDE[0], "asset_cfg": pelvis})
  cfg.rewards["upright_precise"] = RewardTermCfg(
    func=mdp_v11.upright_tight, weight=V11_UPRIGHT_PRECISE[1],
    params={"std": V11_UPRIGHT_PRECISE[0], "asset_cfg": pelvis})
  cfg.rewards["body_ang_vel"] = RewardTermCfg(
    func=mdp_v11.body_ang_vel_cmd_scaled, weight=V11_BODY_ANG_VEL[1],
    params={"asset_cfg": pelvis, "command_name": "twist", "walk_cmd": V11_WALK_CMD,
            "walk_scale": V11_WALK_SCALE})
  cfg.rewards["torso_quiet_under_ub"] = RewardTermCfg(
    func=mdp_v11.torso_quiet_under_ub, weight=V11_WEIGHTS["torso_quiet_under_ub"],
    params={"command_name": "upper_body", "rate_ref": 1.0, "w_weight": 0.1, "asset_cfg": pelvis,
            "twist_name": "twist", "walk_cmd": V11_WALK_CMD, "walk_scale": V11_WALK_SCALE})
  # v11e: linear tracking split into a wide + a precise kernel
  import math as _m
  from mjlab.tasks.velocity import mdp as vel_mdp
  cfg.rewards["track_linear_velocity"].weight = V11_TRACK_LIN_WIDE[1]
  cfg.rewards["track_linear_velocity"].params["std"] = _m.sqrt(V11_TRACK_LIN_WIDE[0])
  cfg.rewards["track_linear_velocity_precise"] = RewardTermCfg(
    func=vel_mdp.track_linear_velocity, weight=V11_TRACK_LIN_PRECISE[1],
    params={**cfg.rewards["track_linear_velocity"].params,
            "std": _m.sqrt(V11_TRACK_LIN_PRECISE[0])})
  # v11g: price the standing squeeze (see V11G_*)
  cfg.rewards["stand_leg_torque"] = RewardTermCfg(
    func=mdp_v11.stand_leg_torque, weight=V11G_STAND_TAU_W,
    params={"command_name": "twist", "command_threshold": 0.1,
            "deadband": V11G_STAND_TAU_DEADBAND, "asset_cfg": SceneEntityCfg("robot")})
  cfg.events["foot_friction"].params["ranges"] = V11G_FRICTION
  # v11d: air time pays only while the body moves as commanded
  cfg.rewards["air_time"] = dataclasses.replace(
    cfg.rewards["air_time"], func=mdp_v11.feet_air_time_tracked,
    params={**cfg.rewards["air_time"].params, "track_var": V11_AIR_TRACK_VAR})
  cfg.rewards["loaded_sole_tilt"] = RewardTermCfg(
    func=mdp_v11.loaded_sole_tilt, weight=V11_WEIGHTS["loaded_sole_tilt"],
    params={"sensor_name": "feet_ground_contact", "fz_min": V11_FZ_MIN})
  cfg.rewards["pad_load_balance"] = RewardTermCfg(
    func=mdp_v11.pad_load_balance, weight=V11_WEIGHTS["pad_load_balance"],
    params={"pad_sensor_name": "foot_pads", "command_name": "twist", "command_threshold": 0.1,
            "min_share": 0.10, "fz_min": V11_FZ_MIN})
  cfg.rewards["ankle_roll_torque_stance"] = RewardTermCfg(
    func=mdp_v11.ankle_roll_torque_stance, weight=V11_WEIGHTS["ankle_roll_torque_stance"],
    params={"sensor_name": "feet_ground_contact", "fz_min": V11_FZ_MIN, "deadband": 4.0,
            "command_name": "twist", "command_threshold": 0.1,
            "single_support_scale": V11_SINGLE_SUPPORT_SCALE})
  # v10 stance terms
  standing = {"command_name": "twist", "command_threshold": 0.1}
  cfg.rewards["stand_leg_action_l2"] = RewardTermCfg(
    func=hv1v2_mdp.stand_leg_action_l2, weight=V10_WEIGHTS["stand_leg_action_l2"],
    params=dict(standing))
  cfg.rewards["stand_stance_shape"] = RewardTermCfg(
    func=hv1v2_mdp.stand_stance_shape, weight=V10_WEIGHTS["stand_stance_shape"],
    params={"asset_cfg": _FEET, "nominal_width": V10_NOMINAL_WIDTH,
            "yaw_weight": V10_STANCE_YAW_W, "width_weight": V10_STANCE_WIDTH_W, **standing})
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_hip_pitch_joint$": 0.05, r".*_hip_roll_joint$": 0.05,
    r".*_hip_yaw_joint$": 0.05, r".*_knee_joint$": 0.05,
    r".*_ankle_pitch_joint$": 0.05, r".*_ankle_roll_joint$": 0.02,
  }
  # v8b
  cfg.rewards["track_yaw_bias"] = RewardTermCfg(
    func=hv1v2_mdp.track_yaw_bias, weight=V8B_YAW_BIAS["weight"],
    params={"std": V8B_YAW_BIAS["std"], "tau": V8B_YAW_BIAS["tau"], "command_name": "twist"})
  cfg.rewards["still_stand_quiet"] = RewardTermCfg(
    func=hv1v2_mdp.still_stand_quiet, weight=V11_WEIGHTS["still_stand_quiet"],
    params={"command_name": "twist", "upper_name": "upper_body", "min_still_s": 1.0})
  # ankle smoothness (v7 priced hip/knee only), ramped like v7's six
  ankles = SceneEntityCfg("robot", joint_names=_ANKLES)
  cfg.rewards["ankle_joint_acc"] = RewardTermCfg(
    func=envs_mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": ankles})
  cfg.rewards["ankle_torque_rate"] = RewardTermCfg(
    func=hv1v2_mdp.torque_rate_l2, weight=0.0, params={"asset_cfg": ankles})
  smooth = {**SMOOTH_WEIGHTS, **V11_SMOOTH_EXTRA}
  for name, (w0, w1) in smooth.items():
    cfg.rewards[name].weight = w1 if play else w0
    cfg.curriculum.pop(f"ramp_{name}", None)
    if not play:
      cfg.curriculum[f"ramp_{name}"] = CurriculumTermCfg(
        func=hv1v2_mdp.reward_weight_ramp,
        params={"reward_name": name, "w0": w0, "w1": w1, "absolute": True,
                "start_iter": (V11_POSTURE_RAMP_ITERS if name in V11_GATED_SMOOTH
                               else V11_SMOOTH_RAMP_ITERS)[0],
                "end_iter": (V11_POSTURE_RAMP_ITERS if name in V11_GATED_SMOOTH
                             else V11_SMOOTH_RAMP_ITERS)[1],
                "ref_num_envs": V11_REF_ENVS, "clock": "_v11_gclock"})
  # posture / stance terms: 5% -> 100% over 1000-4000 (play: final weights)
  posture = {n: (V11_POSTURE_RAMP_START * cfg.rewards[n].weight, cfg.rewards[n].weight)
             for n in V11_POSTURE_RAMPED}
  # v11g: stand_leg_action_l2 is in V11_POSTURE_RAMPED (done at clock 5000); its
  # new weight and stand_leg_torque ramp in separately below
  posture.pop("stand_leg_action_l2", None)
  posture["body_ang_vel"] = V11_BODY_ANG_VEL
  posture["upright_precise"] = (0.0, V11_UPRIGHT_PRECISE[1])
  for name, (w0, w1) in posture.items():
    cfg.rewards[name].weight = w1 if play else w0
    if not play:
      cfg.curriculum[f"ramp_{name}"] = CurriculumTermCfg(
        func=hv1v2_mdp.reward_weight_ramp,
        params={"reward_name": name, "w0": w0, "w1": w1, "absolute": True,
                "start_iter": V11_POSTURE_RAMP_ITERS[0], "end_iter": V11_POSTURE_RAMP_ITERS[1],
                "ref_num_envs": V11_REF_ENVS, "clock": "_v11_gclock"})

  g0 = V11_RESUME_CLOCK or 0.0
  v10_sla = cfg.rewards["stand_leg_action_l2"].weight
  for name, (w0, w1) in {"stand_leg_torque": (0.0, V11G_STAND_TAU_W),
                         "stand_leg_action_l2": (v10_sla, V11G_STAND_ACTION_W)}.items():
    cfg.rewards[name].weight = w1 if play else w0
    if not play:
      cfg.curriculum[f"ramp_{name}"] = CurriculumTermCfg(
        func=hv1v2_mdp.reward_weight_ramp,
        params={"reward_name": name, "w0": w0, "w1": w1, "absolute": True,
                "start_iter": int(g0), "end_iter": int(g0) + V11G_RAMP_ITERS,
                "ref_num_envs": V11_REF_ENVS, "clock": "_v11_gclock"})

  # --- curricula / metrics ------------------------------------------------------------
  if not play:
    # v11d: the gated clock runs FIRST, so the ramps above read this step's value
    cfg.curriculum = {"gated_clock": CurriculumTermCfg(
      func=mdp_v11.v11_gated_clock,
      params={"threshold": V11_CLOCK_OK, "ref_num_envs": V11_REF_ENVS,
              "slow_threshold": V11_SLOW_OK, "rewind_threshold": V11_CLOCK_REWIND,
              "rewind_slow": V11_SLOW_REWIND, "fast_threshold": V11_FAST_OK,
              "rewind_fast": V11_FAST_REWIND, "resume_clock": V11_RESUME_CLOCK}), **cfg.curriculum}
    cfg.curriculum["dr_level"] = CurriculumTermCfg(
      func=mdp_v11.v11_dr_level,
      params={"start_iter": V11_DR_RAMP_ITERS[0], "end_iter": V11_DR_RAMP_ITERS[1], "lo": 0.5,
              "ref_num_envs": V11_REF_ENVS, "gate_on_tracking": True, "pause_when_closed": True})
    cfg.curriculum["upper_body_level"] = CurriculumTermCfg(
      func=mdp_v11.v11_upper_body_level,
      params={"command_name": "upper_body", "max_rate_per_100": V11_UB_MAX_RATE,
              "ref_num_envs": V11_REF_ENVS})
  cfg.metrics["walk_track"] = MetricsTermCfg(
    func=mdp_v11.walk_tracking_error,
    params={"command_name": "twist", "min_cmd": 0.2, "threshold": V11_TRACK_OK,
            "dir_cmd": 0.15, "dir_threshold": V11_TRACK_DIR_OK, "min_swing_s": V11_MIN_SWING_S,
            "slow_range": V11_SLOW_RANGE, "slow_threshold": V11_SLOW_OK,
            "fast_cmd": V11_FAST_CMD, "fast_threshold": V11_FAST_OK})
  cfg.metrics["standing_steps"] = MetricsTermCfg(
    func=hv1v2_mdp.standing_steps, reduce="sum",
    params={"sensor_name": "feet_ground_contact", "command_name": "twist"})
  cfg.metrics["still_sway"] = MetricsTermCfg(
    func=hv1v2_mdp.still_sway,
    params={"command_name": "twist", "upper_name": "upper_body", "min_still_s": 1.0})
  return cfg


##
# v11h (Mjlab-Velocity-Flat-HV1V2-Decoupled-v11h), 2026-11-04: v11g + an
# operator pelvis-height command, 0.89-0.979 m (crouch only), fine-tuned from
# v11g model_19000. Plan: ~/.claude/plans/now-i-have-a-robust-lecun.md; terms in
# mdp_v11h.py, table in hv1v2_constants.py (V11H_HEIGHT_TABLE).
#
#   step 2  commands["height"] (BaseHeightCommand) + curriculum["height_range"]
#   step 3  actor  += height_cmd (1, history 1, LAST term)       837 -> 838
#           critic += height_cmd, base_height (1 + 1, LAST terms)
#   step 4  rewards: + track_base_height (weight 1.0 = ~11% of v11g's 8.78/s,
#           measured with model_19000 at final weights, arms level 1.0);
#           pose / stand_leg_action_l2 / stand_leg_torque re-referenced to the
#           table at h_cmd (each equals v11g's term at home). v11g's two
#           standing ramps (keyed to V11_RESUME_CLOCK) are PINNED at their
#           final weights: on a v11h resume they would otherwise ramp in from
#           0 again and switch the squeeze fix off for ~500 iterations.
##

V11H_RANGE_START = 0.95      # low end of the height range at the start of v11h
V11H_RANGE_EVERY = 200.0     # iterations (8192-env samples) between 0.01 m widenings
V11H_HEIGHT_ERR_OK = 0.025   # m, height-error EMA under which the range may widen
V11H_TRACK_W = 1.0           # track_base_height weight
V11H_TRACK_STD = (0.020, 0.035)   # m, standing / walking kernel width


def hv1v2_decoupled_v11h_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  from mjlab_hv1v2 import mdp_v11h
  from mjlab_hv1v2.hv1v2_constants import V11H_HEIGHT_RANGE

  cfg = hv1v2_decoupled_v11_env_cfg(play=play)   # v11g, exactly

  # --- step 2: command + range curriculum -------------------------------------------
  cfg.commands["height"] = mdp_v11h.BaseHeightCommandCfg(
    initial_low=V11H_HEIGHT_RANGE[0] if play else V11H_RANGE_START)
  if not play:
    cfg.curriculum["height_range"] = CurriculumTermCfg(
      func=mdp_v11h.v11h_height_range,
      params={"command_name": "height", "final_low": V11H_HEIGHT_RANGE[0], "step": 0.01,
              "every_iters": V11H_RANGE_EVERY, "err_ok": V11H_HEIGHT_ERR_OK,
              "ref_num_envs": V11_REF_ENVS})

  # --- step 3: observations (appended LAST: expand_obs.py inserts zero columns there)
  hparams = {"command_name": "height"}
  cfg.observations = dict(cfg.observations)
  actor = cfg.observations["actor"]
  aterms = dict(actor.terms)
  aterms["height_cmd"] = ObservationTermCfg(func=mdp_v11h.height_cmd_obs, params=dict(hparams))
  cfg.observations["actor"] = dataclasses.replace(actor, terms=aterms)
  critic = cfg.observations["critic"]
  cterms = dict(critic.terms)
  cterms["height_cmd"] = ObservationTermCfg(func=mdp_v11h.height_cmd_obs, params=dict(hparams))
  cterms["base_height"] = ObservationTermCfg(func=mdp_v11h.base_height_obs)
  cfg.observations["critic"] = dataclasses.replace(critic, terms=cterms)

  # --- step 4: rewards ----------------------------------------------------------------
  cfg.rewards["track_base_height"] = RewardTermCfg(
    func=mdp_v11h.track_base_height, weight=V11H_TRACK_W,
    params={"height_name": "height", "twist_name": "twist",
            "std_standing": V11H_TRACK_STD[0], "std_walking": V11H_TRACK_STD[1],
            "command_threshold": 0.1})
  pose = cfg.rewards["pose"]
  cfg.rewards["pose"] = dataclasses.replace(
    pose, func=mdp_v11h.height_posture, params={**pose.params, "height_name": "height"})
  sla = cfg.rewards["stand_leg_action_l2"]
  cfg.rewards["stand_leg_action_l2"] = dataclasses.replace(
    sla, func=mdp_v11h.stand_leg_action_ref_l2,
    params={**sla.params, "height_name": "height"}, weight=V11G_STAND_ACTION_W)
  slt = cfg.rewards["stand_leg_torque"]
  cfg.rewards["stand_leg_torque"] = dataclasses.replace(
    slt, func=mdp_v11h.stand_leg_torque_h,
    params={**slt.params, "height_name": "height"}, weight=V11G_STAND_TAU_W)
  for name in ("stand_leg_torque", "stand_leg_action_l2"):
    cfg.curriculum.pop(f"ramp_{name}", None)
  return cfg
