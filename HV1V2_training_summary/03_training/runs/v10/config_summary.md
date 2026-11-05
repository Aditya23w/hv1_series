# v10 — `2026-10-26_23-21-21_v10`

Generated from `params/env.yaml` and `params/agent.yaml` by `tools/build_summary.py`.

## Simulation

- num_envs: 4096
- physics dt: 0.005 s, decimation 4 → policy at 50 Hz
- episode length: 20.0 s

## Rewards

| term | function | weight | params |
|---|---|---|---|
| `track_linear_velocity` | track_linear_velocity | 3.0 | {'command_name': 'twist', 'std': 0.5} |
| `track_angular_velocity` | track_angular_velocity | 3.0 | {'command_name': 'twist', 'std': 0.7071067811865476} |
| `upright` | upright | 1.0 | {'std': 0.4472135954999579, 'asset_cfg': {'body_names': ['pelvis']}} |
| `pose` | variable_posture | 1.0 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$', '.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']}, 'command_name': 'twist', 'std_standing': {'.*_hip_p... |
| `body_ang_vel` | body_angular_velocity_penalty | -0.05 | {'asset_cfg': {'body_names': ['pelvis']}} |
| `dof_pos_limits` | joint_pos_limits | -1.0 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$', '.*_ankle_pitch_joint$', '.*_ankle_roll_joint$']}} |
| `action_rate_l2` | action_rate_l2 | -0.4 | {} |
| `air_time` | feet_air_time | 0.3 | {'sensor_name': 'feet_ground_contact', 'threshold_min': 0.1, 'threshold_max': 0.35, 'command_name': 'twist', 'command_threshold': 0.1} |
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
| `action_acc_l2` | action_acc_l2 | -0.2 | {} |
| `hip_knee_joint_acc` | joint_acc_l2 | -4e-05 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$']}} |
| `hip_knee_torque_rate` | torque_rate_l2 | -2.5e-08 | {'asset_cfg': {'joint_names': ['.*_hip_pitch_joint$', '.*_hip_roll_joint$', '.*_hip_yaw_joint$', '.*_knee_joint$']}} |
| `hip_yaw_action_rate` | action_rate_l2_joints | -3.5 | {'joint_names': ['.*_hip_yaw_joint$']} |
| `hip_yaw_vel` | joint_vel_l2 | -0.14 | {'asset_cfg': {'joint_names': ['.*_hip_yaw_joint$']}} |
| `track_yaw_bias` | track_yaw_bias | 0.5 | {'std': 0.05, 'tau': 1.0, 'command_name': 'twist'} |
| `still_stand_quiet` | still_stand_quiet | 0.0 | {'command_name': 'twist', 'upper_name': 'upper_body', 'min_still_s': 1.0} |
| `stand_sole_tilt` | stand_sole_tilt | -5.8 | {'sensor_name': 'feet_ground_contact', 'asset_cfg': {'body_names': ['left_ankle_roll_link', 'right_ankle_roll_link']}, 'command_name': 'twist', 'command_threshold': 0.1} |
| `stand_leg_action_l2` | stand_leg_action_l2 | -0.042 | {'command_name': 'twist', 'command_threshold': 0.1} |
| `stand_stance_shape` | stand_stance_shape | -1.25 | {'asset_cfg': {'body_names': ['left_ankle_roll_link', 'right_ankle_roll_link']}, 'nominal_width': 0.243, 'yaw_weight': 1.0, 'width_weight': 2.0, 'command_name': 'twist', 'command_threshold': 0.1} |

## Observations — actor

concatenate=True, corruption=True, history=None

| term | function | noise | history | params |
|---|---|---|---|---|
| `base_ang_vel` | builtin_sensor | {'operation': 'add', 'n_min': -0.2, 'n_max': 0.2} | 5 | {'sensor_name': 'robot/imu_ang_vel'} |
| `projected_gravity` | projected_gravity | {'operation': 'add', 'n_min': -0.05, 'n_max': 0.05} | 5 | {} |
| `joint_pos` | joint_pos_rel | {'operation': 'add', 'n_min': -0.01, 'n_max': 0.01} | 5 | {'biased': True} |
| `joint_vel` | joint_vel_rel | {'operation': 'add', 'n_min': -1.5, 'n_max': 1.5} | 5 | {} |
| `actions` | last_action | None | 5 | {} |
| `command` | generated_commands | None | 0 | {'command_name': 'twist'} |

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
entity_name: robot
preserve_order...
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
| `pd_gains` | pd_gains | startup  | {'kp_range': [0.9, 1.1], 'kd_range': [0.9, 1.1], 'operation': 'scale'} |
| `joint_armature` | joint_armature | startup  | {'ranges': [0.6, 1.4], 'operation': 'scale'} |
| `effort_limits` | effort_limits | startup  | {'effort_limit_range': [0.85, 1.0], 'operation': 'scale'} |
| `hand_payload` | body_mass | reset  | {'asset_cfg': {'body_names': ['left_wrist_yaw_link', 'right_wrist_yaw_link']}, 'ranges': [0.0, 2.0], 'operation': 'add'} |
| `hand_force` | apply_external_force_torque | interval [2.0, 5.0] | {'asset_cfg': {'body_names': ['left_wrist_yaw_link', 'right_wrist_yaw_link']}, 'force_range': [-20.0, 20.0], 'torque_range': [0.0, 0.0]} |
| `ankle_stiffness` | ankle_stiffness | startup  | {'pitch_range': [0.9, 1.9], 'roll_range': [0.65, 1.4]} |

## Commands

### `twist`

```
entity_name: robot
heading_command: true
heading_control_stiffness: 0.5
init_velocity_prob: 0.0
ranges:
  ang_vel_z: [-0.7, 0.7]
  heading: [-3.141592653589793, 3.141592653589793]
  lin_vel_x: [-0.8, 1.0]
  lin_vel_y: [-0.5, 0.5]
rel_forward_envs: 0.2
rel_heading_envs: 0.3
rel_standing_envs: 0.3
rel_world_envs: 0.0
resampling_time_range: [3.0, 8.0]

```

### `upper_body`

```
amp_exponent: 1.5
asset_name: robot
clip_dir: /home/aditya/Desktop/HV1V2/mjlab_hv1v2/data/clips_v2
clip_prob: 0.5
clip_quiet_prob: 0.15
families: [scripted, random, walking]
hold_after_clip_range: [4.0, 16.0]
initial_level: 1.0
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
speed_range: [0.3, 2.0]
twist_name: twist
vmax: 2.0
walking_prob: 0.5
write_targets: false

```

## Terminations

- `time_out`: time_out {} (time-out)
- `fell_over`: bad_orientation {'limit_angle': 1.2217304763960306}

## Curriculum

- `command_vel`: commands_vel {'command_name': 'twist', 'velocity_stages': [{'step': 0, 'lin_vel_x': [-0.8, 1.0], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_z': [-0.7, 0.7]}]}
- `upper_body_level`: upper_body_level {'command_name': 'upper_body'}

## Logged metrics

`mean_action_acc`, `torque_duty_frac`, `torque_frac_of_continuous`, `torque_frac_legs`, `double_flight_frac`, `arm_joint_speed`, `leg_action_rate`, `leg_action_acc`, `hip_yaw_action_rate`, `standing_steps`, `still_sway`

## PPO (rsl_rl)

- actor [512, 256, 128] elu, critic [512, 256, 128], obs normalisation True
- init action std 1.0
- steps/env/iter 24, max_iterations 8000, save every 200
- resume: True from `2026-10-06_17-58-29_v7$` / `model_29999\.pt`

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
