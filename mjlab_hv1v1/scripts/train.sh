#!/usr/bin/env bash
# Train an HV1V1 velocity policy.
#
#   ./train.sh                                  # primary: deployable actor
#   ./train.sh Mjlab-Velocity-Rough-HV1V1       # rough terrain
#   RUN=smoke ITERS=200 NUM_ENVS=512 ./train.sh
#
# Both robots share ONE conda env (named hv1v2_mjlab for historical reasons;
# it hosts mjlab plus both robot packages). Override with MJLAB_ENV.
set -euo pipefail
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-Mjlab-Velocity-Flat-HV1V1}"
shift || true
# --log-root is required: mjlab's default is the RELATIVE path logs/rsl_rl and
# `conda run` does not change the working directory.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run --no-capture-output -n "${MJLAB_ENV:-hv1v2_mjlab}" \
  train "$TASK" \
    --log-root "$LAB/logs/rsl_rl" \
    --env.scene.num-envs "${NUM_ENVS:-4096}" \
    --agent.max-iterations "${ITERS:-40000}" \
    --agent.save-interval 200 \
    --agent.logger tensorboard \
    --video False \
    --agent.run-name "${RUN:-v1}" "$@"
