"""HV1V1 velocity-tracking environment configurations.

Built by calling mjlab's generic ``make_velocity_env_cfg()`` and then filling in
every blank it leaves for the robot. Structure follows
``mjlab/tasks/velocity/config/g1/env_cfgs.py``; the HV1V1-specific deviations
are commented where they occur.
"""

from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
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

from mjlab_hv1v1 import mdp as hv1v1_mdp
from mjlab_hv1v1.hv1v1_constants import (
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  HV1V1_ACTION_SCALE,
  get_hv1v1_robot_cfg,
)

# HV1V1 is 94.309 kg against G1's ~35 kg. Angular momentum scales with mass and
# the reward term squares it, so reusing G1's -0.02 would price rotation ~7x
# harder here; H1-V1 measured 100% fell_over at ~20 steps doing exactly that.
_MASS_RATIO = 35.0 / 94.3088   # HV1V1 total mass
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
# a 20 ms control period. But it also makes every earlier HV1V1 checkpoint
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


def hv1v1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """HV1V1 rough-terrain velocity tracking."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  # G1's 70 is not enough here: each of the 8 foot pads is a box, and a
  # box-against-terrain contact generates up to 4 points, so the feet alone can
  # ask for ~32 before any link hull touches the ground. Measured overflow at
  # 104 on rough terrain; 256 leaves headroom for an unlucky seed.
  cfg.sim.nconmax = 256

  cfg.scene.entities = {"robot": get_hv1v1_robot_cfg()}

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
  joint_pos_action.scale = HV1V1_ACTION_SCALE

  # --- viewer / commands ---------------------------------------------------
  cfg.viewer.body_name = "torso_link"
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.30  # head reaches 1.841 m
  # resampling_time_range stays at mjlab's (3.0, 8.0). v2 used (2.0, 6.0) for more
  # walk->stand transitions, but resampling_time_range[1] is ALSO the divisor of
  # Metrics/twist/error_vel_xy (velocity_command.py:_update_metrics, which
  # accumulates per step and divides by range[1]/step_dt). Shortening it inflated
  # that metric 1.33x for identical physical error, and since it accumulates it
  # also scales with episode length -- so HV1V1's longer episodes made it look 2x
  # worse when a fixed-command rollout showed tracking was identical (0.6478 vs
  # 0.6474 at iteration 1600). That artifact cost a killed run. Keeping the
  # default keeps every comparison against v4 honest by default.

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
  # command, so from iteration 10000 almost every command was unreachable:
  # error_vel_xy climbed 0.71 -> 2.31 and the tracking reward fell from its
  # peak at iteration ~4000. Unreachable commands do not stretch the policy,
  # they just dilute the signal. Cap the ladder near what the robot achieves.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.8, 1.0), "ang_vel_z": (-0.5, 0.5)},
    {"step": 5000 * 24, "lin_vel_x": (-1.0, 1.3), "ang_vel_z": (-0.7, 0.7)},
    {"step": 15000 * 24, "lin_vel_x": (-1.2, 1.5)},
  ]

  # --- rewards -------------------------------------------------------------
  # Tracking above the 2.0 default: at 2.0 against pose 1.0, a 93 kg humanoid
  # maximises reward by standing still and collecting upright + pose.
  cfg.rewards["track_linear_velocity"].weight = 3.0
  # NOTE: std is left at mjlab's sqrt(0.25). v1 tried sqrt(0.15) to attack the
  # low-speed dead zone (any command <= 0.20 m/s produces 0.001 m/s) and it was
  # a GRADIENT-VANISHING BUG. The term is exp(-e/std^2), so narrowing std flattens
  # it wherever the error is large -- which is everywhere early training lives.
  # Gradient ratio, sqrt(0.15) against sqrt(0.25):
  #
  #   err 0.04 (the dead zone being fixed)   1.50x   helps
  #   err 0.48 (v4 at iteration 500)         0.46x
  #   err 1.15 (v1 at iteration 500)         0.078x
  #   err 2.07 (v1 at iteration 5400)        0.007x
  #
  # Measured consequence: v1's error_vel_xy ran 2.4x v4's from iteration 500 --
  # before the push ramp at 3000 and before the command ladder at 5000, so no
  # curriculum explains it -- and then climbed, 1.49 -> 2.07 over iterations
  # 4800-5600, while v4 sat flat at ~0.65. High error gives almost no gradient,
  # so the error stays high: a self-inflicted plateau. Run v1 was killed for it.
  #
  # The dead zone is a CONVERGED-POLICY problem and a global reward-shape change
  # is the wrong tool for it. Revisit after this robot walks, either by not
  # sampling the ambiguous |vx| band at all or by ramping std DOWN late, once the
  # error is already small enough for the sharpening to bite.
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
    # Shoulder pitch is deliberately LOOSE. `pose` penalises deviation from
    # the default, so a tight std (G1 uses 0.15) prices arm swing as an error
    # and the policy holds the arms still -- which is what v3 did. The
    # counter-swing is worth having: it cancels the angular momentum the legs
    # generate, which is exactly what the `angular_momentum` term rewards.
    # 0.4 sits between G1's walking 0.15 and its running 0.5.
    r".*shoulder_pitch.*": 0.4,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    # Elbow stays tight so it holds the 0.9 rad bend rather than straightening.
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
    # HV1V1 has a head; H1-V1 welds it and G1 has none. A joint matched by no
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
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
    r".*head.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = ANGULAR_MOMENTUM_WEIGHT

  # air_time must be ON for a robot this heavy. mjlab's base cfg and the G1
  # config both leave the weight at 0.0, and a 2600-iteration run at 4096 envs
  # showed exactly what that buys: reward climbed to 27.9 and episode length to
  # 518 steps while peak_height_mean sat at 3 mm against an 80 mm clearance
  # target and slip_velocity_mean *rose* 0.061 -> 0.074. With nothing rewarding
  # flight time, sliding is a valid optimum and tracking+upright+pose pay for
  # it. These are H1-V1's values, same mass class and same actuators.
  cfg.rewards["air_time"].weight = 0.75
  # The base defaults (0.05 / 0.5 / command_threshold 0.5) are tuned for a much
  # lighter robot; a 0.5 command threshold would also switch the term off across
  # most of our commanded range.
  # 0.25, HV1V2 v4's proven value. Run v2 tried 0.30 to lower the 1.26 Hz cadence
  # and it KILLED the term outright -- air_time sat at exactly 0.0000 for 2700
  # iterations while v4's climbed 0.0035 -> 0.0207 over the same window.
  #
  # The term is a BINARY INDICATOR, not a graded one
  # (tasks/velocity/mdp/rewards.py:222):
  #     in_range = (current_air_time > threshold_min) & (< threshold_max)
  #     reward   = sum(in_range.float())
  # so threshold_min is a CLIFF with no partial credit and no gradient below it.
  # Neither run's air_time_mean ever crosses its own threshold -- v4 earns the
  # reward from the TAIL of the flight-time distribution, since current_air_time
  # grows through the swing. v4's tail clears 0.25; v2's never cleared 0.30, and
  # v2's air_time_mean was 3.3x lower at iteration 2650 (0.031 vs 0.103) and
  # barely growing, because the only reward for leaving the ground was gated
  # behind a bar it could not reach. No reward -> no gradient -> no flight.
  #
  # DO NOT raise this to chase cadence. Cadence needs a lever with a gradient --
  # threshold_max, or a separate step-frequency term -- not a higher binary floor.
  cfg.rewards["air_time"].params["threshold_min"] = 0.25
  cfg.rewards["air_time"].params["threshold_max"] = 0.6
  cfg.rewards["air_time"].params["command_threshold"] = 0.1


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
  cfg.metrics["torque_duty_frac"] = MetricsTermCfg(func=hv1v1_mdp.torque_duty_frac)
  cfg.metrics["torque_frac_of_continuous"] = MetricsTermCfg(
    func=hv1v1_mdp.torque_frac_of_continuous
  )
  cfg.metrics["torque_frac_legs"] = MetricsTermCfg(func=hv1v1_mdp.torque_frac_legs)

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

  # Applied last, and only here: hv1v1_flat_env_cfg derives from this function
  # and only *deletes* actor terms afterwards, so the survivors keep their
  # history and all three registered tasks stay consistent by construction.
  cfg = _add_obs_history(cfg)

  return cfg


def hv1v1_flat_env_cfg(
  play: bool = False, state_estimation: bool = False
) -> ManagerBasedRlEnvCfg:
  """HV1V1 flat-terrain velocity tracking.

  Args:
    play: evaluation mode (no corruption, no pushes, infinite episode).
    state_estimation: whether the ACTOR may observe base linear velocity.
      Defaults to False, because on hardware that signal comes from a base
      state estimator HV1V1 does not have. The critic keeps it either way.
  """
  cfg = hv1v1_rough_env_cfg(play=play)

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
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
