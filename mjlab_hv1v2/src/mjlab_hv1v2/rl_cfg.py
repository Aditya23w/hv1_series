"""PPO runner configuration for the HV1V2 velocity tasks."""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def hv1v2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Below G1's 0.01, following H1-V1. Of the 31 actions, 11 are pinned to
      # scale 0.0 and 8 more drive arms that carry no load; high entropy buys
      # arm flailing rather than exploration that matters.
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="hv1v2_velocity",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=40_000,
  )


class Hv1v2OnPolicyRunner(VelocityOnPolicyRunner):
  """VelocityOnPolicyRunner that also records WHICH joints the actions drive.

  mjlab's exporter writes joint_names / default_joint_pos / kp / kd for all 31
  joints but action_scale only for the action term, and never the mapping
  between them. With a 31-wide action that was implicit; the decoupled task
  acts on 12, so the ONNX states `action_joint_names` explicitly. Consumers
  (sim2sim, export_deploy) fall back to "all joints" when it is absent, so
  every earlier ONNX keeps working.
  """

  def save(self, path: str, infos=None):
    super().save(path, infos)
    _, _, onnx_path = self._get_export_paths(path)
    try:
      term = self.env.unwrapped.action_manager.get_term("joint_pos")
      attach_metadata_to_onnx(
        str(onnx_path), {"action_joint_names": list(term.target_names)}
      )
    except Exception as e:  # never kill training over metadata
      print(f"[WARN] could not attach action_joint_names to ONNX: {e}")


##
# v11: same PPO, plus the left/right mirror loss (rsl_rl Symmetry extension).
##

from dataclasses import dataclass, field


@dataclass
class RslRlPpoSymmetryAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """RslRlPpoAlgorithmCfg + rsl_rl's symmetry_cfg (mjlab's dataclass has no
  such field; rsl_rl's PPO takes it as a keyword)."""

  symmetry_cfg: dict | None = None


def hv1v2_v11_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = hv1v2_ppo_runner_cfg()
  a = cfg.algorithm
  cfg.algorithm = RslRlPpoSymmetryAlgorithmCfg(
    **{f: getattr(a, f) for f in a.__dataclass_fields__},
    # Mirror LOSS only: data augmentation would also need the critic's
    # privileged terms (contacts, payload, ...) mirrored. The loss alone makes
    # the actor treat left and right alike, which is what removes one-sided
    # habits (v7's rolled right ankle, right-leg loading).
    symmetry_cfg={
      "use_data_augmentation": False,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0,
      "data_augmentation_func": "mjlab_hv1v2.symmetry:mirror_v11",
    },
  )
  cfg.experiment_name = "hv1v2_velocity"
  cfg.run_name = "v11"
  cfg.max_iterations = 50_000
  return cfg


def hv1v2_v11h_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """v11's PPO + mirror loss, unchanged; only the run name differs."""
  cfg = hv1v2_v11_ppo_runner_cfg()
  cfg.run_name = "v11h"
  return cfg
