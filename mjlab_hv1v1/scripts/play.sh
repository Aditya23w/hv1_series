#!/usr/bin/env bash
# Watch a trained HV1V1 policy.
#   ./play.sh /abs/path/to/model_XXXX.pt [task_id]
set -euo pipefail
CKPT="${1:?usage: play.sh /abs/path/to/model_XXXX.pt [task_id]}"
TASK="${2:-Mjlab-Velocity-Flat-HV1V1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run --no-capture-output -n "${MJLAB_ENV:-hv1v2_mjlab}" \
  play "$TASK" --checkpoint-file "$CKPT" --num-envs "${NUM_ENVS:-1}"
