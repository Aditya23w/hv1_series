"""HV1V2 humanoid for mjlab.

mjlab imports this module automatically through the ``mjlab.tasks`` entry
point declared in pyproject.toml, which is when the registrations below run.
Nothing is vendored into the mjlab checkout.

Debugging note: a failure during this import is swallowed by mjlab's
_import_registered_packages() into a [WARN] traceback on stderr, and the tasks
simply never appear. Run ``python -c "import mjlab_hv1v2"`` to get it fatally.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_hv1v2.env_cfgs import (
  hv1v2_decoupled_env_cfg,
  hv1v2_decoupled_v8_env_cfg,
  hv1v2_decoupled_v8b_env_cfg,
  hv1v2_decoupled_v10_env_cfg,
  hv1v2_decoupled_v11_env_cfg,
  hv1v2_flat_env_cfg,
  hv1v2_rough_env_cfg,
)
from mjlab_hv1v2.rl_cfg import (
  Hv1v2OnPolicyRunner,
  hv1v2_ppo_runner_cfg,
  hv1v2_v11_ppo_runner_cfg,
)

# Primary task: the actor sees only what the real robot can measure, so a
# trained policy is deployable without a base state estimator.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V2",
  env_cfg=hv1v2_flat_env_cfg(),
  play_env_cfg=hv1v2_flat_env_cfg(play=True),
  rl_cfg=hv1v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Baseline/ablation: the actor additionally observes base linear velocity,
# which comes from the MJCF velocimeter and has no hardware counterpart.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V2-State-Estimation",
  env_cfg=hv1v2_flat_env_cfg(state_estimation=True),
  play_env_cfg=hv1v2_flat_env_cfg(play=True, state_estimation=True),
  rl_cfg=hv1v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-HV1V2",
  env_cfg=hv1v2_rough_env_cfg(),
  play_env_cfg=hv1v2_rough_env_cfg(play=True),
  rl_cfg=hv1v2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# v6: the policy drives the 12 leg joints only; waist yaw, arms, wrists and head
# belong to a manipulation/IK controller and are randomised as a disturbance
# (waist roll/pitch held at home). Its runner writes `action_joint_names` into
# the ONNX, because 12 actions over 31 joints is no longer self-describing.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V2-Decoupled",
  env_cfg=hv1v2_decoupled_env_cfg(),
  play_env_cfg=hv1v2_decoupled_env_cfg(play=True),
  rl_cfg=hv1v2_ppo_runner_cfg(),
  runner_cls=Hv1v2OnPolicyRunner,
)

# v7: the decoupled task plus smoothness shaping (action rate/acc, hip+knee
# joint acceleration and torque rate, hip-yaw rate/velocity), ramped in over
# iterations 1000-5000. Same 403/12 layout as v6.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V2-Decoupled-Smooth",
  env_cfg=hv1v2_decoupled_env_cfg(smooth=True),
  play_env_cfg=hv1v2_decoupled_env_cfg(play=True, smooth=True),
  rl_cfg=hv1v2_ppo_runner_cfg(),
  runner_cls=Hv1v2OnPolicyRunner,
)

# v8: v7 from scratch with v7's recipe, plus synthetic Mink -> Ruckig
# upper-body clips (half of the upper-body motion; 17 joints, waist roll/pitch
# locked), robot timing (interpolated targets, 0-20 ms leg delay) and ankle
# stiffness DR up to the real four-bar's 1.9x / 1.4x. No reward changes.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V2-Decoupled-v8",
  env_cfg=hv1v2_decoupled_v8_env_cfg(),
  play_env_cfg=hv1v2_decoupled_v8_env_cfg(play=True),
  rl_cfg=hv1v2_ppo_runner_cfg(),
  runner_cls=Hv1v2OnPolicyRunner,
)

# v8b: v8 + arms-still time restored to v7's level (clip holds, quiet draws),
# a yaw-bias tracking term against heading drift, and Metrics/still_sway.
# Warm start from v8's final checkpoint.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V2-Decoupled-v8b",
  env_cfg=hv1v2_decoupled_v8b_env_cfg(),
  play_env_cfg=hv1v2_decoupled_v8b_env_cfg(play=True),
  rl_cfg=hv1v2_ppo_runner_cfg(),
  runner_cls=Hv1v2OnPolicyRunner,
)

# v10: v8b + a flat, relaxed, re-aligned stance (sole tilt, standing leg
# action, stance shape, tighter standing ankle roll), ankle stiffness DR down
# to the hardware's 0.67x roll, on HOME_KEYFRAME. Warm start from v7's final
# checkpoint.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V2-Decoupled-v10",
  env_cfg=hv1v2_decoupled_v10_env_cfg(),
  play_env_cfg=hv1v2_decoupled_v10_env_cfg(play=True),
  rl_cfg=hv1v2_ppo_runner_cfg(),
  runner_cls=Hv1v2OnPolicyRunner,
)

# v11: fresh decoupled policy (plan: ~/.claude/plans/keen-pondering-fiddle.md)
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V2-Decoupled-v11",
  env_cfg=hv1v2_decoupled_v11_env_cfg(),
  play_env_cfg=hv1v2_decoupled_v11_env_cfg(play=True),
  rl_cfg=hv1v2_v11_ppo_runner_cfg(),
  runner_cls=Hv1v2OnPolicyRunner,
)
