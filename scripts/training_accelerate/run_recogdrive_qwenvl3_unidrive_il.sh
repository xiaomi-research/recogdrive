#!/usr/bin/env bash
set -euo pipefail
# UniDriveVLA-Base Stage1 as Qwen3-VL backbone. Override VLM_PATH if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FAMILY=qwenvl3 TRAINING_TARGET=waypoint TRAIN_STAGE=il \
  VLM_PATH="${VLM_PATH:-/path/to/UniDriveVLA_Nusc_Base_Stage1}" \
  bash "${SCRIPT_DIR}/run_recogdrive_accel_variant.sh"
