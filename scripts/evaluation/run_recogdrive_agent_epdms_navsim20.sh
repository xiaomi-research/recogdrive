#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export NAVSIM_TREE=2.0
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/navsim2.0}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# One-stage navtest EPDMS (per-frame + adjacent-frame EC). Not navhard two-stage.
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}"
CHECKPOINT="${CHECKPOINT:-/path/to/recogdrive.ckpt}"
VLM_PATH="${VLM_PATH:-/path/to/internvl3_pretrain_model}"

CAM_TYPE="${CAM_TYPE:-single}"
FRONT_SHORT_SIDE="${FRONT_SHORT_SIDE:-960}"
SIDE_SHORT_SIDE="${SIDE_SHORT_SIDE:-480}"
VLM_MAX_LENGTH="${VLM_MAX_LENGTH:-8704}"
SAMPLING_METHOD="${SAMPLING_METHOD:-ddim}"

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

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  agent=recogdrive_agent \
  "agent.checkpoint_path=${CHECKPOINT}" \
  "agent.vlm_path=${VLM_PATH}" \
  "agent.grpo=False" \
  "agent.cache_hidden_state=False" \
  "agent.vlm_type=internvl" \
  "agent.dit_type=small" \
  "agent.vlm_size=small" \
  "agent.sampling_method=${SAMPLING_METHOD}" \
  "experiment_name=recogdrive_agent_epdms_navsim20_${TRAIN_TEST_SPLIT}" \
  "metric_cache_path=${CACHE_PATH}" \
  "${AGENT_OVERRIDES[@]}" \
  "$@"
