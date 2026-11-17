#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export NAVSIM_TREE=2.0
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/navsim2.0}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Two-stage navhard EPDMS (synthetic stage-two + endpoint-weighted product).
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"
CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT:?set CACHE_PATH or NAVSIM_EXP_ROOT}/metric_cache}"
if [[ -n "${TRAINING_CONFIG:-}" ]]; then
  : "${CHECKPOINT:?set CHECKPOINT to the trained weights}"
  : "${OUTPUT_DIR:?set OUTPUT_DIR to a fresh evaluation directory}"
  exec python "${REPO_ROOT}/scripts/verify/navsim_eval.py" \
    --training-config "${TRAINING_CONFIG}" --checkpoint "${CHECKPOINT}" \
    --split "${TRAIN_TEST_SPLIT}" --metric-cache "${CACHE_PATH}" --output-dir "${OUTPUT_DIR}" -- "$@"
fi
CHECKPOINT="${CHECKPOINT:-/path/to/recogdrive.ckpt}"
VLM_PATH="${VLM_PATH:-/path/to/internvl3_pretrain_model}"

exec python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" \
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
  experiment_name=recogdrive_agent_epdms_navsim20_two_stage \
  metric_cache_path="${CACHE_PATH}" \
  "$@"
