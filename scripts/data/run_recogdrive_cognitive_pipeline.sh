#!/usr/bin/env bash
# Hierarchical NAVSIM cognitive VQA (generation, refinement, quality control).
# VLM calls read OPENAI_API_KEY, OPENAI_BASE_URL, and optional COGNITIVE_VLM_MODEL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAVSIM_TREE="${NAVSIM_TREE:-1.1}"
case "${NAVSIM_TREE}" in
  1.1|2.0) ;;
  *) echo "NAVSIM_TREE must be 1.1 or 2.0, got ${NAVSIM_TREE}" >&2; exit 1 ;;
esac

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/path/to/NAVSIM/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/path/to/NAVSIM/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/navsim${NAVSIM_TREE}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/NAVSIM/dataset}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-train}"
CAM_TYPE="${CAM_TYPE:-single}"
PROMPT_TYPE="${PROMPT_TYPE:-base}"
# Concurrent scenes. Each scene issues several VLM calls; lower this if the endpoint rate-limits.
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT="${OUT:-${NAVSIM_EXP_ROOT}/cognitive_vqa/vqa.jsonl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cognitive_vqa}"

args=(
  --config-path "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/config/training"
  --config-name "${CONFIG_NAME:-default_training}"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "experiment_name=${EXPERIMENT_NAME}"
  "split=${SPLIT}"
  "cam_type=${CAM_TYPE}"
  "prompt_type=${PROMPT_TYPE}"
  "batch_size=${BATCH_SIZE}"
  "out=${OUT}"
)
if [[ -n "${TRAJ_RESULTS:-}" ]]; then
  args+=("traj_results=${TRAJ_RESULTS}")
fi

python -m recogdrive.data.cognitive "${args[@]}" "$@"
