#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WAYMOE2E="${WAYMOE2E:-True}"
export MODEL_FAMILY="${MODEL_FAMILY:-internvl3}"
export TRAINING_TARGET="${TRAINING_TARGET:-waypoint}"
export TRAIN_STAGE=il
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_waymoe2e_stage2_il}"
export LOG_FILE="${LOG_FILE:-train_recogdrive_waymoe2e_stage2_il.txt}"

bash "${SCRIPT_DIR}/run_recogdrive_accel_variant.sh"
