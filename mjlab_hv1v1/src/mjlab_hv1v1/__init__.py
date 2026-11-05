"""HV1V1 humanoid for mjlab.

mjlab imports this module automatically through the ``mjlab.tasks`` entry
point declared in pyproject.toml, which is when the registrations below run.
Nothing is vendored into the mjlab checkout.

Debugging note: a failure during this import is swallowed by mjlab's
_import_registered_packages() into a [WARN] traceback on stderr, and the tasks
simply never appear. Run ``python -c "import mjlab_hv1v1"`` to get it fatally.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_hv1v1.env_cfgs import hv1v1_flat_env_cfg, hv1v1_rough_env_cfg
from mjlab_hv1v1.rl_cfg import hv1v1_ppo_runner_cfg

# Primary task: the actor sees only what the real robot can measure, so a
# trained policy is deployable without a base state estimator.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V1",
  env_cfg=hv1v1_flat_env_cfg(),
  play_env_cfg=hv1v1_flat_env_cfg(play=True),
  rl_cfg=hv1v1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Baseline/ablation: the actor additionally observes base linear velocity,
# which comes from the MJCF velocimeter and has no hardware counterpart.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-HV1V1-State-Estimation",
  env_cfg=hv1v1_flat_env_cfg(state_estimation=True),
  play_env_cfg=hv1v1_flat_env_cfg(play=True, state_estimation=True),
  rl_cfg=hv1v1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-HV1V1",
  env_cfg=hv1v1_rough_env_cfg(),
  play_env_cfg=hv1v1_rough_env_cfg(play=True),
  rl_cfg=hv1v1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
