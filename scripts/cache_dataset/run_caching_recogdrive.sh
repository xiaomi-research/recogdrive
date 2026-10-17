#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAVSIM_TREE="${NAVSIM_TREE:-1.1}"
case "${NAVSIM_TREE}" in
  1.1|2.0) ;;
  *) echo "NAVSIM_TREE must be 1.1 or 2.0, got ${NAVSIM_TREE}" >&2; exit 1 ;;
esac

case "${CACHE_HIDDEN_STATE:-}" in
  True|true) CACHE_HIDDEN_STATE=True ;;
  False|false) CACHE_HIDDEN_STATE=False ;;
  *) echo "Use run_caching_recogdrive_hidden_state.sh or run_caching_recogdrive_no_hidden_state.sh" >&2; exit 1 ;;
esac

MODEL_FAMILY="${MODEL_FAMILY:-internvl3}"
case "${MODEL_FAMILY}" in
  internvl3)
    VLM_TYPE="${VLM_TYPE:-internvl}"
    VLM_PATH="${VLM_PATH:-/path/to/internvl3_pretrain_model}"
    ;;
  qwenvl3)
    VLM_TYPE="${VLM_TYPE:-qwen}"
    VLM_PATH="${VLM_PATH:-/path/to/qwenvl3_pretrain_model}"
    ;;
  *) echo "Unsupported MODEL_FAMILY=${MODEL_FAMILY}. Use internvl3 or qwenvl3." >&2; exit 1 ;;
esac

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/path/to/NAVSIM/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/path/to/NAVSIM/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}/navsim${NAVSIM_TREE}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/NAVSIM/dataset}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
if [[ "${NAVSIM_TREE}" == "1.1" ]]; then
  DEFAULT_GPUS=8
else
  DEFAULT_GPUS=1
fi
GPUS="${GPUS:-${DEFAULT_GPUS}}"
if ! [[ "${GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS must be a positive integer, got ${GPUS}" >&2
  exit 1
fi
CACHE_ROOT="${CACHE_ROOT:-${NAVSIM_EXP_ROOT}/recogdrive_agent_cache}"
if [[ "${CACHE_HIDDEN_STATE}" == "True" ]]; then
  CACHE_SUFFIX="_${MODEL_FAMILY}"
else
  CACHE_SUFFIX="_${MODEL_FAMILY}_no_hidden_state"
fi
CACHE_PATH="${CACHE_PATH:-${CACHE_ROOT}${CACHE_SUFFIX}}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/caching_dataset.txt}"
mkdir -p "$(dirname "${LOG_FILE}")"

if [[ ! -f "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/config/training/default_training.yaml" ]]; then
  echo "NAVSIM training config not found under NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}" >&2
  exit 1
fi

if [[ "${NAVSIM_TREE}" == "1.1" ]]; then
  GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
  if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS_PER_NODE must be a positive integer, got ${GPUS_PER_NODE}" >&2
    exit 1
  fi
  if (( GPUS_PER_NODE > GPUS )); then
    GPUS_PER_NODE="${GPUS}"
  fi
  NNODES="${NNODES:-$(( (GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))}"
  if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NNODES must be a positive integer, got ${NNODES}" >&2
    exit 1
  fi
  NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
  if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
    echo "NODE_RANK must be between 0 and NNODES-1, got ${NODE_RANK} for NNODES=${NNODES}" >&2
    exit 1
  fi
  MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
  MASTER_PORT="${MASTER_PORT:-63669}"
  CACHE_SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py"
  if [[ ! -f "${CACHE_SCRIPT}" ]]; then
    echo "NAVSIM 1.1 distributed caching entrypoint not found: ${CACHE_SCRIPT}" >&2
    exit 1
  fi
  COMMAND=(torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}"
           --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" "${CACHE_SCRIPT}")
else
  if [[ "${GPUS}" != "1" ]]; then
    echo "NAVSIM 2.0's upstream run_dataset_caching.py is single-process; set GPUS=1 (no torchrun sharding is provided)." >&2
    exit 1
  fi
  CACHE_SCRIPT="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching.py"
  if [[ ! -f "${CACHE_SCRIPT}" ]]; then
    echo "NAVSIM 2.0 caching entrypoint not found: ${CACHE_SCRIPT}" >&2
    exit 1
  fi
  # The 2.0 entry has no rank partitioning; run one worker so VLM/cache writes stay single-process.
  COMMAND=(python "${CACHE_SCRIPT}" worker=sequential)
fi

echo "NAVSIM_TREE=${NAVSIM_TREE} CACHE_HIDDEN_STATE=${CACHE_HIDDEN_STATE} MODEL_FAMILY=${MODEL_FAMILY}"
echo "VLM_PATH=${VLM_PATH} CACHE_PATH=${CACHE_PATH} GPUS=${GPUS} TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT}"

"${COMMAND[@]}" \
  agent=recogdrive_agent \
  experiment_name=recogdrive_agent_cache \
  agent.cam_type=single \
  agent.cache_hidden_state="${CACHE_HIDDEN_STATE}" \
  agent.cache_mode=True \
  agent.vlm_type="${VLM_TYPE}" \
  agent.vlm_path="${VLM_PATH}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  "$@" \
  >"${LOG_FILE}" 2>&1
