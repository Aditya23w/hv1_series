# v11 — `2026-11-02_13-24-36_v11g19k`

Generated from `params/env.yaml` and `params/agent.yaml` by `tools/build_summary.py`.

## Simulation

- num_envs: 8192
- physics dt: 0.005 s, decimation 4 → policy at 50 Hz
- episode length: 20.0 s

## Rewards

| term | function | weight | params |
|---|---|---|---|
| `track_linear_velocity` | track_linear_velocity | 3.0 | {'command_name': 'twist', 'std': 0.5} |
| `track_angular_velocity` | track_angular_velocity | 3.0 | {'command_name': 'twist', 'std': 0.7071067811865476} |
| `upright` | upright_tight | 0.7 | {'std': 0.35, 'asset_cfg': {'body_names': ['pelvis']}} |
| `pose` | variable_posture | 1.0 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$', '.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']}, 'command_name': 'twist', 'std_standing': {'.*_hip_p... |
| `body_ang_vel` | body_ang_vel_cmd_scaled | -0.05 | {'asset_cfg': {'body_names': ['pelvis']}, 'command_name': 'twist', 'walk_cmd': 0.05, 'walk_scale': 0.25} |
| `dof_pos_limits` | joint_pos_limits | -1.0 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$', '.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']}} |
| `action_rate_l2` | action_rate_l2 | -0.1 | {} |
| `air_time` | feet_air_time_tracked | 0.3 | {'sensor_name': 'feet_ground_contact', 'threshold_min': 0.1, 'threshold_max': 0.35, 'command_name': 'twist', 'command_threshold': 0.1, 'track_var': 0.25} |
| `foot_clearance` | feet_clearance | -2.0 | {'target_height': 0.08, 'height_sensor_name': 'foot_height_scan', 'command_name': 'twist', 'command_threshold': 0.05, 'asset_cfg': {'site_names': ['left_foot', 'right_foot']}} |
| `foot_swing_height` | feet_swing_height | -0.25 | {'sensor_name': 'feet_ground_contact', 'height_sensor_name': 'foot_height_scan', 'target_height': 0.08, 'command_name': 'twist', 'command_threshold': 0.05} |
| `foot_slip` | feet_slip | -0.1 | {'sensor_name': 'feet_ground_contact', 'command_name': 'twist', 'command_threshold': 0.05, 'asset_cfg': {'site_names': ['left_foot', 'right_foot']}} |
| `soft_landing` | soft_landing | -1e-05 | {'sensor_name': 'feet_ground_contact', 'command_name': 'twist', 'command_threshold': 0.05} |
| `feet_flight` | feet_flight | -1.0 | {'sensor_name': 'feet_ground_contact'} |
| `air_time_excess` | air_time_excess | -2.0 | {'sensor_name': 'feet_ground_contact', 't_max': 0.4, 'command_name': 'twist', 'command_threshold': 0.1} |
| `step_length_excess` | step_length_excess | -2.0 | {'sensor_name': 'feet_ground_contact', 'max_length': 0.3, 'command_name': 'twist', 'command_threshold': 0.1, 'asset_cfg': {'site_names': ['left_foot', 'right_foot']}} |
| `self_collisions` | self_collision_cost | -1.0 | {'sensor_name': 'self_collision', 'force_threshold': 10.0} |
| `joint_torques` | joint_torques_l2 | -2.8521692314650633e-07 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$', '.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']}} |
| `stand_still_feet` | stand_still_feet | -1.0 | {'command_name': 'twist', 'command_threshold': 0.1, 'asset_cfg': {'site_names': ['left_foot', 'right_foot']}} |
| `stand_feet_contact` | stand_feet_contact | 0.5 | {'sensor_name': 'feet_ground_contact', 'command_name': 'twist', 'command_threshold': 0.1} |
| `action_acc_l2` | action_acc_l2 | 0.0 | {} |
| `hip_knee_joint_acc` | joint_acc_l2 | 0.0 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$']}} |
| `hip_knee_torque_rate` | torque_rate_l2 | 0.0 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$']}} |
| `hip_yaw_action_rate` | action_rate_l2_joints | 0.0 | {'joint_names': ['.*_hip_yaw_joint$']} |
| `hip_yaw_vel` | joint_vel_l2 | 0.0 | {'asset_cfg': {'joint_names': ['.*_hip_yaw_joint$']}} |
| `upright_precise` | upright_tight | 0.0 | {'std': 0.06, 'asset_cfg': {'body_names': ['pelvis']}} |
| `torso_quiet_under_ub` | torso_quiet_under_ub | -5.57 | {'command_name': 'upper_body', 'rate_ref': 1.0, 'w_weight': 0.1, 'asset_cfg': {'body_names': ['pelvis']}, 'twist_name': 'twist', 'walk_cmd': 0.05, 'walk_scale': 0.25} |
| `track_linear_velocity_precise` | track_linear_velocity | 2.0 | {'command_name': 'twist', 'std': 0.22360679774997896} |
| `stand_leg_torque` | stand_leg_torque | 0.0 | {'command_name': 'twist', 'command_threshold': 0.1, 'deadband': {'.*_hip_roll_joint$': 10.0, '.*_hip_pitch_joint$': 15.0, '.*_knee_joint$': 30.0}} |
| `loaded_sole_tilt` | loaded_sole_tilt | -0.31200000000000006 | {'sensor_name': 'feet_ground_contact', 'fz_min': 20.0} |
| `pad_load_balance` | pad_load_balance | -0.008350000000000002 | {'pad_sensor_name': 'foot_pads', 'command_name': 'twist', 'command_threshold': 0.1, 'min_share': 0.1, 'fz_min': 20.0} |
| `ankle_roll_torque_stance` | ankle_roll_torque_stance | -4.32e-06 | {'sensor_name': 'feet_ground_contact', 'fz_min': 20.0, 'deadband': 4.0, 'command_name': 'twist', 'command_threshold': 0.1, 'single_support_scale': 0.25} |
| `stand_leg_action_l2` | stand_leg_action_l2 | -0.042 | {'command_name': 'twist', 'command_threshold': 0.1} |
| `stand_stance_shape` | stand_stance_shape | -1.25 | {'asset_cfg': {'body_names': ['left_ankle_roll_link', 'right_ankle_roll_link']}, 'nominal_width': 0.243, 'yaw_weight': 1.0, 'width_weight': 2.0, 'command_name': 'twist', 'command_threshold': 0.1} |
| `track_yaw_bias` | track_yaw_bias | 0.5 | {'std': 0.05, 'tau': 1.0, 'command_name': 'twist'} |
| `still_stand_quiet` | still_stand_quiet | -0.86 | {'command_name': 'twist', 'upper_name': 'upper_body', 'min_still_s': 1.0} |
| `ankle_joint_acc` | joint_acc_l2 | 0.0 | {'asset_cfg': {'joint_names': ['.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']}} |
| `ankle_torque_rate` | torque_rate_l2 | 0.0 | {'asset_cfg': {'joint_names': ['.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']}} |

## Observations — actor

concatenate=True, corruption=True, history=None

| term | function | noise | history | params |
|---|---|---|---|---|
| `base_ang_vel` | base_ang_vel_v11 | {'operation': 'add', 'n_min': -0.2, 'n_max': 0.2} | 10 | {'sensor_name': 'robot/imu_ang_vel', 'freeze_prob': 0.002, 'freeze_steps': [1, 5]} |
| `projected_gravity` | projected_gravity_v11 | {'operation': 'add', 'n_min': -0.05, 'n_max': 0.05} | 10 | {'freeze_prob': 0.002, 'freeze_steps': [1, 5]} |
| `joint_pos` | joint_pos_rel | {'operation': 'add', 'n_min': -0.01, 'n_max': 0.01} | 10 | {'biased': True} |
| `joint_vel` | joint_vel_rel | {'operation': 'add', 'n_min': -1.5, 'n_max': 1.5} | 10 | {} |
| `actions` | last_action | None | 10 | {} |
| `command` | generated_commands | None | 0 | {'command_name': 'twist'} |
| `ub_target` | ub_target_rel | None | 0 | {'command_name': 'upper_body', 'joint_names': ['waist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitc... |
| `ub_target_rate` | ub_target_rate | {'operation': 'add', 'n_min': -0.1, 'n_max': 0.1} | 0 | {'command_name': 'upper_body', 'joint_names': ['waist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitc... |

## Observations — critic

concatenate=True, corruption=False, history=None

| term | function | noise | history | params |
|---|---|---|---|---|
| `base_lin_vel` | builtin_sensor | {'operation': 'add', 'n_min': -0.5, 'n_max': 0.5} | 0 | {'sensor_name': 'robot/imu_lin_vel'} |
| `base_ang_vel` | builtin_sensor | {'operation': 'add', 'n_min': -0.2, 'n_max': 0.2} | 0 | {'sensor_name': 'robot/imu_ang_vel'} |
| `projected_gravity` | projected_gravity | {'operation': 'add', 'n_min': -0.05, 'n_max': 0.05} | 0 | {} |
| `joint_pos` | joint_pos_rel | None | 0 | {} |
| `joint_vel` | joint_vel_rel | {'operation': 'add', 'n_min': -1.5, 'n_max': 1.5} | 0 | {} |
| `actions` | last_action | None | 0 | {} |
| `command` | generated_commands | None | 0 | {'command_name': 'twist'} |
| `foot_height` | foot_height | None | 0 | {'sensor_name': 'foot_height_scan'} |
| `foot_air_time` | foot_air_time | None | 0 | {'sensor_name': 'feet_ground_contact'} |
| `foot_contact` | foot_contact | None | 0 | {'sensor_name': 'feet_ground_contact'} |
| `foot_contact_forces` | foot_contact_forces | None | 0 | {'sensor_name': 'feet_ground_contact'} |
| `upper_body` | generated_commands | None | 0 | {'command_name': 'upper_body'} |
| `ub_target_rate` | ub_target_rate | None | 0 | {'command_name': 'upper_body', 'joint_names': ['waist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitc... |
| `plant` | privileged_plant | None | 0 | {'leg_joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$', '.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']} |
| `hand_load` | generated_commands | None | 0 | {'command_name': 'hand_load'} |
| `torso_push` | generated_commands | None | 0 | {'command_name': 'torso_push'} |

## Actions

### `joint_pos`

```
actuator_names:
- .*_hip_pitch_joint$
- .*_hip_roll_joint$
- .*_hip_yaw_joint$
- .*_knee_joint$
- .*_ankle_pitch_joint$
- .*_ankle_roll_joint$
clip: null
delay_substeps_range:
- 0
- 4
dropout_prob: 0.001
entity_name: ...
```

scale: `.*_hip_pitch_joint$` 0.32, `.*_hip_roll_joint$` 0.32, `.*_hip_yaw_joint$` 0.32, `.*_knee_joint$` 0.32, `.*_ankle_pitch_joint$` 0.15, `.*_ankle_roll_joint$` 0.15
### `upper_body_actuation`

```
clip: null
command_name: upper_body
entity_name: robot

```

## Events (resets, domain randomisation, pushes)

| event | function | mode | params |
|---|---|---|---|
| `reset_base` | reset_root_state_uniform | reset  | {'pose_range': {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [0.01, 0.05], 'yaw': [-3.14, 3.14]}} |
| `reset_robot_joints` | reset_joints_by_offset | reset  | {'position_range': [-0.05, 0.05], 'velocity_range': [-0.1, 0.1], 'asset_cfg': {'joint_names': ['.*']}} |
| `push_robot` | push_and_stamp | interval [2.0, 4.0] | {'velocity_range': {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [-0.4, 0.4], 'roll': [-0.52, 0.52], 'pitch': [-0.52, 0.52], 'yaw': [-0.78, 0.78]}} |
| `foot_friction` | geom_friction | startup  | {'asset_cfg': {'geom_names': ['left_foot1_collision', 'left_foot2_collision', 'left_foot3_collision', 'left_foot4_collision', 'right_foot1_collision', 'right_foot2_collision', 'right_foot3_collision', 'right_foot4_col... |
| `encoder_bias` | encoder_bias | startup  | {'bias_range': [-0.015, 0.015]} |
| `base_com` | body_com_offset | startup  | {'asset_cfg': {'body_names': ['torso_link']}, 'operation': 'add', 'ranges': {0: [-0.025, 0.025], 1: [-0.025, 0.025], 2: [-0.03, 0.03]}} |
| `joint_armature` | joint_armature | startup  | {'ranges': [0.6, 1.4], 'operation': 'scale'} |
| `effort_limits` | effort_limits | startup  | {'effort_limit_range': [0.85, 1.0], 'operation': 'scale'} |
| `hand_payload` | hand_payload_v11 | reset  | {'body_names': ['left_wrist_yaw_link', 'right_wrist_yaw_link'], 'mass_range': [0.0, 2.0], 'offset_max': 0.1} |
| `actuator_gains` | v11_actuator_gains | reset  | {'kp_range': [0.85, 1.15], 'kd_range': [0.85, 1.15], 'groups': {'ankle_pitch': [['.*_ankle_pitch_joint$'], [0.8, 1.45], 1.0], 'ankle_roll': [['.*_ankle_roll_joint$'], [0.5, 1.1], 0.7], 'waist_rp': [['waist_roll_joint$... |
| `encoder_bias_legs` | encoder_bias_scaled | reset  | {'bias_max': 0.0524, 'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$', '.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']}} |
| `imu_bias` | imu_bias_v11 | reset  | {'rp_max_rad': 0.0131, 'gyro_max': 0.05} |
| `link_mass` | pseudo_inertia | startup  | {'asset_cfg': {'body_names': ['^(?!torso_link$\|left_wrist_yaw_link$\|right_wrist_yaw_link$).*$']}, 'alpha_range': [-0.05, 0.05]} |
| `link_com` | body_com_offset | startup  | {'asset_cfg': {'body_names': ['^(?!torso_link$\|left_wrist_yaw_link$\|right_wrist_yaw_link$).*$']}, 'ranges': [-0.02, 0.02], 'operation': 'add'} |

## Commands

### `twist`

```
entity_name: robot
heading_command: true
heading_control_stiffness: 0.5
init_velocity_prob: 0.0
ranges:
  ang_vel_z: [-0.8, 0.8]
  heading: [-3.141592653589793, 3.141592653589793]
  lin_vel_x: [-0.6, 1.0]
  lin_vel_y: [-0.5, 0.5]
rel_forward_envs: 0.2
rel_heading_envs: 0.3
rel_standing_envs: 0.35
rel_world_envs: 0.0
resampling_time_range: [3.0, 8.0]
reversal_prob: 0.15
stand_hold_range: [5.0, 15.0]
stop_prob: 0.1

```

### `upper_body`

```
amp_exponent: 1.5
asset_name: robot
clip_dir: /home/admin/HV1V2/mjlab_hv1v2/data/clips_v2
clip_prob: 0.5
clip_quiet_prob: 0.15
families: [scripted, random, walking]
hold_after_clip_range: [4.0, 16.0]
hold_prob: 0.02
initial_level: 0.1
joint_names: [waist_yaw_joint$, waist_roll_joint$, waist_pitch_joint$, .*_shoulder_pitch_joint$, .*_shoulder_roll_joint$,
  .*_shoulder_yaw_joint$, .*_elbow_joint$, .*_wrist_roll_joint$, .*_wrist_pitch_joint$, .*_wrist_yaw_joint$, head_yaw_joint$,
  head_pitch_joint$]
quiet_prob: 0.15
resampling_time_range: [1.0, 3.0]
sample_range:
  .*_elbow_joint$: [0.0, 2.2]
  .*_shoulder_pitch_joint$: [-1.6, 0.8]
  .*_shoulder_yaw_joint$: [-1.0, 1.0]
  .*_wrist_pitch_joint$: [-0.45, 0.45]
  .*_wrist_roll_joint$: [-1.5, 1.5]
  .*_wrist_yaw_joint$: [-0.9, 0.9]
  head_pitch_joint: [-0.6, 0.4]
  head_yaw_joint: [-1.0, 1.0]
  left_shoulder_roll_joint: [0.0, 1.2]
  right_shoulder_roll_joint: [-1.2, 0.0]
  waist_pitch_joint: [0.0, 0.0]
  waist_roll_joint: [0.0, 0.0]
  waist_yaw_joint: [-1.0, 1.0]
speed_cap: {waist_yaw_joint: 1.0}
speed_range: [0.3, 2.0]
twist_name: twist
vmax: 2.0
walking_prob: 0.5
write_targets: false

```

### `hand_load`

```
active_prob: 0.35
body_names: [left_wrist_yaw_link, right_wrist_yaw_link]
force_max: 30.0
hold_range: [1.0, 4.0]
mass_max: 2.0
opposite_prob: 0.3
pickup_prob: 0.2
pickup_ramp_s: 0.2
resampling_time_range: [1.0, 3.0]
rise_range: [0.2, 1.0]

```

### `torso_push`

```
active_prob: 0.15
body_names: [torso_link]
force_max: 60.0
hold_range: [0.05, 0.3]
opposite_prob: 0.0
resampling_time_range: [1.0, 4.0]
rise_range: [0.05, 0.2]

```

## Terminations

- `time_out`: time_out {} (time-out)
- `fell_over`: bad_orientation {'limit_angle': 1.2217304763960306}

## Curriculum

- `gated_clock`: v11_gated_clock {'threshold': 0.28, 'ref_num_envs': 8192, 'slow_threshold': 0.19, 'rewind_threshold': 0.3, 'rewind_slow': 0.2, 'fast_threshold': 0.5, 'rewind_fast': 0.55, 'resume_clock': 14249.6}
- `command_vel`: commands_vel {'command_name': 'twist', 'velocity_stages': [{'step': 0, 'lin_vel_x': [-0.6, 1.0], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_z': [-0.8, 0.8]}]}
- `upper_body_level`: v11_upper_body_level {'command_name': 'upper_body', 'max_rate_per_100': 0.05, 'ref_num_envs': 8192}
- `ramp_action_rate_l2`: reward_weight_ramp {'reward_name': 'action_rate_l2', 'w0': -0.1, 'w1': -0.4, 'absolute': True, 'start_iter': 500, 'end_iter': 2500, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_action_acc_l2`: reward_weight_ramp {'reward_name': 'action_acc_l2', 'w0': 0.0, 'w1': -0.2, 'absolute': True, 'start_iter': 500, 'end_iter': 2500, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_hip_knee_joint_acc`: reward_weight_ramp {'reward_name': 'hip_knee_joint_acc', 'w0': 0.0, 'w1': -4e-05, 'absolute': True, 'start_iter': 500, 'end_iter': 2500, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_hip_knee_torque_rate`: reward_weight_ramp {'reward_name': 'hip_knee_torque_rate', 'w0': 0.0, 'w1': -2.5e-08, 'absolute': True, 'start_iter': 500, 'end_iter': 2500, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_hip_yaw_action_rate`: reward_weight_ramp {'reward_name': 'hip_yaw_action_rate', 'w0': 0.0, 'w1': -3.5, 'absolute': True, 'start_iter': 500, 'end_iter': 2500, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_hip_yaw_vel`: reward_weight_ramp {'reward_name': 'hip_yaw_vel', 'w0': 0.0, 'w1': -0.14, 'absolute': True, 'start_iter': 500, 'end_iter': 2500, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_ankle_joint_acc`: reward_weight_ramp {'reward_name': 'ankle_joint_acc', 'w0': 0.0, 'w1': -2.175e-05, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_ankle_torque_rate`: reward_weight_ramp {'reward_name': 'ankle_torque_rate', 'w0': 0.0, 'w1': -3.33e-08, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_torso_quiet_under_ub`: reward_weight_ramp {'reward_name': 'torso_quiet_under_ub', 'w0': -5.57, 'w1': -111.4, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_pad_load_balance`: reward_weight_ramp {'reward_name': 'pad_load_balance', 'w0': -0.008350000000000002, 'w1': -0.167, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_ankle_roll_torque_stance`: reward_weight_ramp {'reward_name': 'ankle_roll_torque_stance', 'w0': -4.32e-06, 'w1': -8.64e-05, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_loaded_sole_tilt`: reward_weight_ramp {'reward_name': 'loaded_sole_tilt', 'w0': -0.31200000000000006, 'w1': -6.24, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_still_stand_quiet`: reward_weight_ramp {'reward_name': 'still_stand_quiet', 'w0': -0.86, 'w1': -17.2, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_body_ang_vel`: reward_weight_ramp {'reward_name': 'body_ang_vel', 'w0': -0.05, 'w1': -0.5, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_upright_precise`: reward_weight_ramp {'reward_name': 'upright_precise', 'w0': 0.0, 'w1': 0.3, 'absolute': True, 'start_iter': 1500, 'end_iter': 5000, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_stand_leg_torque`: reward_weight_ramp {'reward_name': 'stand_leg_torque', 'w0': 0.0, 'w1': -0.0003, 'absolute': True, 'start_iter': 14249, 'end_iter': 14749, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `ramp_stand_leg_action_l2`: reward_weight_ramp {'reward_name': 'stand_leg_action_l2', 'w0': -0.042, 'w1': -0.21, 'absolute': True, 'start_iter': 14249, 'end_iter': 14749, 'ref_num_envs': 8192, 'clock': '_v11_gclock'}
- `dr_level`: v11_dr_level {'start_iter': 1500, 'end_iter': 6000, 'lo': 0.5, 'ref_num_envs': 8192, 'gate_on_tracking': True, 'pause_when_closed': True}

## Logged metrics

`mean_action_acc`, `torque_duty_frac`, `torque_frac_of_continuous`, `torque_frac_legs`, `double_flight_frac`, `arm_joint_speed`, `leg_action_rate`, `leg_action_acc`, `hip_yaw_action_rate`, `walk_track`, `standing_steps`, `still_sway`

## PPO (rsl_rl)

- actor [512, 256, 128] elu, critic [512, 256, 128], obs normalisation True
- init action std 1.0
- steps/env/iter 24, max_iterations 40000, save every 200
- resume: True from `2026-11-02_09-10-54_v11f_s43` / `model_14400\.pt`

| param | value |
|---|---|
| num_learning_epochs | 5 |
| num_mini_batches | 4 |
| learning_rate | 0.001 |
| schedule | adaptive |
| gamma | 0.99 |
| lam | 0.95 |
| entropy_coef | 0.005 |
| desired_kl | 0.01 |
| max_grad_norm | 1.0 |
| value_loss_coef | 1.0 |
| use_clipped_value_loss | True |
| clip_param | 0.2 |
| normalize_advantage_per_mini_batch | False |
| optimizer | adam |
| share_cnn_encoders | False |
| class_name | PPO |
| symmetry_cfg | {'use_data_augmentation': False, 'use_mirror_loss': True, 'mirror_loss_coeff': 1.0, 'data_augmentation_func': 'mjlab_hv1v2.symmetry:mirror_v11'} |
