#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_LOADER="${DATA_LOADER:-physicalai}"
export PHYSICALAI_ROOT="${PHYSICALAI_ROOT:-${PHYSICALAI_CACHE_PATH:-}}"
if [[ -z "${PHYSICALAI_ROOT}" ]]; then
  echo "Set PHYSICALAI_ROOT or PHYSICALAI_CACHE_PATH to a cache built by python -m recogdrive.data.physicalai" >&2
  exit 2
fi

export MODEL_FAMILY="${MODEL_FAMILY:-internvl3}"
export TRAINING_TARGET="${TRAINING_TARGET:-waypoint}"
export TRAIN_STAGE=il
export SAMPLING_METHOD="${SAMPLING_METHOD:-ddim}"
export OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adamw}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_${MODEL_FAMILY}_${SAMPLING_METHOD}_${TRAINING_TARGET}_${OPTIMIZER_TYPE}_il_physicalai}"
export LOG_FILE="${LOG_FILE:-train_recogdrive_${MODEL_FAMILY}_${SAMPLING_METHOD}_${TRAINING_TARGET}_${OPTIMIZER_TYPE}_il_physicalai.txt}"

# PhysicalAI-AV converter produces a 5 s future horizon (10 poses at 0.5 s).
PHYSICALAI_ARGS=(
  "data_loader=${DATA_LOADER}"
  "physicalai.root=${PHYSICALAI_ROOT}"
  "agent.trajectory_sampling.time_horizon=5"
)

# Multi-view training: set PAI_VIEWS=multi to use every mapped camera with the standard
# front 960 / side 480 resolution; keep front-only by default to match single-view baselines.
if [[ "${PAI_VIEWS:-single}" == "multi" ]]; then
  PHYSICALAI_ARGS+=(
    "agent.cam_type=multi"
    "agent.front_short_side=${PAI_FRONT_SHORT_SIDE:-960}"
    "agent.side_short_side=${PAI_SIDE_SHORT_SIDE:-480}"
    "agent.vlm_max_length=8704"
  )
fi

bash "${SCRIPT_DIR}/run_recogdrive_train.sh" \
  "${PHYSICALAI_ARGS[@]}" \
  "$@"
