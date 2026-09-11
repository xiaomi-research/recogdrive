#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export NAVSIM_TREE=2.0
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/navsim2.0}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}"
CHECKPOINT="${CHECKPOINT:-/path/to/recogdrive.ckpt}"
VLM_PATH="${VLM_PATH:-/path/to/internvl3_pretrain_model}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  agent=recogdrive_agent \
  agent.checkpoint_path="${CHECKPOINT}" \
  agent.vlm_path="${VLM_PATH}" \
  agent.cam_type=single \
  agent.grpo=False \
  agent.cache_hidden_state=False \
  agent.vlm_type=internvl \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  experiment_name=recogdrive_agent_epdms_navsim20 \
  metric_cache_path="${CACHE_PATH}"
