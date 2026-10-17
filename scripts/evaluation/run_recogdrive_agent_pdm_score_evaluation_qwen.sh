#!/usr/bin/env bash
set -euo pipefail

# NAVSIM 1.1 PDMS evaluation for a Qwen3-VL / UniDriveVLA-backed ReCogDrive checkpoint.
# Multi-view by default (Exp5 layout); override CAM_TYPE=single for a single-view model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export NAVSIM_TREE="${NAVSIM_TREE:-1.1}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/navsim${NAVSIM_TREE}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/path/to/NAVSIM/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/path/to/NAVSIM/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/NAVSIM/dataset}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
CHECKPOINT="${CHECKPOINT:-/path/to/recogdrive_qwen.ckpt}"
VLM_PATH="${VLM_PATH:-/path/to/UniDriveVLA_Nusc_Base_Stage1}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/path/to/metric_cache}"

CAM_TYPE="${CAM_TYPE:-multi}"
FRONT_SHORT_SIDE="${FRONT_SHORT_SIDE:-960}"
SIDE_SHORT_SIDE="${SIDE_SHORT_SIDE:-480}"
VLM_MAX_LENGTH="${VLM_MAX_LENGTH:-8704}"
SAMPLING_METHOD="${SAMPLING_METHOD:-ddim}"
GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPUS}}"

MASTER_PORT="${MASTER_PORT:-63669}"
export MASTER_PORT

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=0

AGENT_OVERRIDES=(
  "agent.cam_type=${CAM_TYPE}"
)
if [[ "${CAM_TYPE}" == "multi" ]]; then
  AGENT_OVERRIDES+=(
    "agent.front_short_side=${FRONT_SHORT_SIDE}"
    "agent.side_short_side=${SIDE_SHORT_SIDE}"
    "agent.vlm_max_length=${VLM_MAX_LENGTH}"
  )
fi

torchrun \
  --nproc_per_node="${GPUS_PER_NODE}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  agent=recogdrive_agent \
  "agent.checkpoint_path=${CHECKPOINT}" \
  "agent.vlm_path=${VLM_PATH}" \
  "agent.vlm_type=qwen" \
  "agent.grpo=False" \
  "agent.cache_hidden_state=False" \
  "agent.dit_type=small" \
  "agent.vlm_size=small" \
  "agent.sampling_method=${SAMPLING_METHOD}" \
  "experiment_name=recogdrive_qwen_agent_eval_${TRAIN_TEST_SPLIT}" \
  "metric_cache_path=${METRIC_CACHE_PATH}" \
  "${AGENT_OVERRIDES[@]}" \
  "$@"
