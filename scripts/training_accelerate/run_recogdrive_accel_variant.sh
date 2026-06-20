#!/usr/bin/env bash
set -euo pipefail

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/path/to/NAVSIM/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/path/to/NAVSIM/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/path/to/NAVSIM/navsim-main}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/path/to/NAVSIM/dataset}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
MODEL_FAMILY="${MODEL_FAMILY:-internvl3}"       # internvl3 | qwenvl3
TRAINING_TARGET="${TRAINING_TARGET:-waypoint}"  # waypoint | delta
TRAIN_STAGE="${TRAIN_STAGE:-il}"                # il | rl
WAYMOE2E="${WAYMOE2E:-False}"
WAYMOE2E_TRAIN_SPLIT="${WAYMOE2E_TRAIN_SPLIT:-training}"
WAYMOE2E_VAL_SPLIT="${WAYMOE2E_VAL_SPLIT:-val}"

case "${MODEL_FAMILY}" in
  internvl3)
    VLM_TYPE="${VLM_TYPE:-internvl}"
    VLM_PATH="${VLM_PATH:-/path/to/internvl3_pretrain_model}"
    VLM_HIDDEN_SIZE="${VLM_HIDDEN_SIZE:-null}"
    ;;
  qwenvl3)
    VLM_TYPE="${VLM_TYPE:-qwen}"
    VLM_PATH="${VLM_PATH:-/path/to/qwenvl3_pretrain_model}"
    VLM_HIDDEN_SIZE="${VLM_HIDDEN_SIZE:-null}"
    ;;
  *)
    echo "Unsupported MODEL_FAMILY=${MODEL_FAMILY}. Use internvl3 or qwenvl3." >&2
    exit 1
    ;;
esac

case "${TRAIN_STAGE}" in
  il)
    GRPO="False"
    MAX_EPOCHS="${MAX_EPOCHS:-100}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    CHECKPOINT_ARGS=()
    ;;
  rl)
    GRPO="True"
    MAX_EPOCHS="${MAX_EPOCHS:-10}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    CHECKPOINT="${CHECKPOINT:-/path/to/${MODEL_FAMILY}_${TRAINING_TARGET}_il.ckpt}"
    METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/path/to/metric_cache_train}"
    CHECKPOINT_ARGS=(
      "agent.checkpoint_path=${CHECKPOINT}"
      "agent.metric_cache_path=${METRIC_CACHE_PATH}"
      "agent.reference_policy_checkpoint=${CHECKPOINT}"
    )
    ;;
  *)
    echo "Unsupported TRAIN_STAGE=${TRAIN_STAGE}. Use il or rl." >&2
    exit 1
    ;;
esac

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-$((GPUS / GPUS_PER_NODE))}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63669}}"

SAMPLING_METHOD="${SAMPLING_METHOD:-ddim}"
FLOW_NOISE_LEVEL="${FLOW_NOISE_LEVEL:-0.7}"
FLOW_SDE_TYPE="${FLOW_SDE_TYPE:-sde}"
DIT_TYPE="${DIT_TYPE:-small}"
VLM_SIZE="${VLM_SIZE:-small}"
LR="${LR:-1e-4}"
CACHE_ROOT="${CACHE_ROOT:-/path/to/recogdrive_agent_cache}"
if [[ "${WAYMOE2E,,}" == "true" ]]; then
  CACHE_PATH="${CACHE_PATH:-${WAYMOE2E_CACHE_PATH:-/path/to/waymoe2e_recogdrive_cache}}"
else
  CACHE_PATH="${CACHE_PATH:-${CACHE_ROOT}_${MODEL_FAMILY}}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_${MODEL_FAMILY}_${TRAINING_TARGET}_${TRAIN_STAGE}}"
LOG_FILE="${LOG_FILE:-train_recogdrive_${MODEL_FAMILY}_${TRAINING_TARGET}_${TRAIN_STAGE}.txt}"

echo "MODEL_FAMILY=${MODEL_FAMILY}"
echo "TRAINING_TARGET=${TRAINING_TARGET}"
echo "TRAIN_STAGE=${TRAIN_STAGE}"
echo "VLM_TYPE=${VLM_TYPE}"
echo "VLM_PATH=${VLM_PATH}"
echo "CACHE_PATH=${CACHE_PATH}"
echo "WAYMOE2E=${WAYMOE2E}"
echo "GPUS=${GPUS}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_recogdrive_accelerate.py" \
  agent=recogdrive_agent \
  agent.lr="${LR}" \
  agent.grpo="${GRPO}" \
  agent.vlm_path="${VLM_PATH}" \
  agent.cam_type=single \
  agent.cache_hidden_state=True \
  agent.vlm_type="${VLM_TYPE}" \
  agent.dit_type="${DIT_TYPE}" \
  agent.vlm_size="${VLM_SIZE}" \
  agent.vlm_hidden_size="${VLM_HIDDEN_SIZE}" \
  agent.sampling_method="${SAMPLING_METHOD}" \
  agent.training_target="${TRAINING_TARGET}" \
  agent.flow_noise_level="${FLOW_NOISE_LEVEL}" \
  agent.flow_sde_type="${FLOW_SDE_TYPE}" \
  agent.train_backbone=False \
  "${CHECKPOINT_ARGS[@]}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  trainer.params.num_nodes="${NNODES}" \
  trainer.params.devices="${GPUS_PER_NODE}" \
  use_deepspeed=True \
  deepspeed.zero_optimization.stage="${DEEPSPEED_STAGE:-1}" \
  deepspeed.bf16.enabled="${DEEPSPEED_BF16:-False}" \
  deepspeed.fp16.enabled="${DEEPSPEED_FP16:-True}" \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  waymoe2e="${WAYMOE2E}" \
  waymoe2e_train_split="${WAYMOE2E_TRAIN_SPLIT}" \
  waymoe2e_val_split="${WAYMOE2E_VAL_SPLIT}" \
  use_cache_without_dataset=True \
  force_cache_computation=False > "${LOG_FILE}" 2>&1
