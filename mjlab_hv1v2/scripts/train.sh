#!/usr/bin/env bash
# Train an HV1V2 velocity policy.
#
#   ./train.sh                                  # primary: deployable actor
#   ./train.sh Mjlab-Velocity-Rough-HV1V2       # rough terrain
#   RUN=smoke ITERS=200 NUM_ENVS=512 ./train.sh
#
# Note this lab is conda-managed here, not uv-managed like g1_lab/c1_lab. Using
# `uv run` would silently build a second virtualenv. The remote PC instead has
# a plain venv at ../.venv, which is picked up below when present.
set -euo pipefail
LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-Mjlab-Velocity-Flat-HV1V2}"
shift || true
# --log-root is required: mjlab's default is the RELATIVE path logs/rsl_rl and
# `conda run` does not change the working directory.
# Environment: a uv venv at ../.venv if there is one (the remote training PC),
# otherwise the hv1v2_mjlab conda env (this laptop).
if [[ -x "$LAB/../.venv/bin/train" ]]; then
  RUNNER=("$LAB/../.venv/bin/train")
else
  RUNNER=(conda run --no-capture-output -n hv1v2_mjlab train)
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${RUNNER[@]}" "$TASK" \
    --log-root "$LAB/logs/rsl_rl" \
    --env.scene.num-envs "${NUM_ENVS:-4096}" \
    --agent.max-iterations "${ITERS:-40000}" \
    --agent.save-interval 200 \
    --agent.logger tensorboard \
    --video False \
    --agent.run-name "${RUN:-v1}" "$@"
