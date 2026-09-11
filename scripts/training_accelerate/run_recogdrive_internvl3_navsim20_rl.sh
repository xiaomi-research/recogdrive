#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FAMILY=internvl3 TRAINING_TARGET=waypoint TRAIN_STAGE=rl \
  NAVSIM_TREE=2.0 \
  GRPO_NUM_ITERATIONS="${GRPO_NUM_ITERATIONS:-4}" \
  bash "${SCRIPT_DIR}/run_recogdrive_accel_variant.sh"
