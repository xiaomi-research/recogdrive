set -x

TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
if (( GPUS < GPUS_PER_NODE )); then
  GPUS_PER_NODE=$GPUS
fi
NODES=$(( (GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=0

CHECKPOINT="/path/to/recogdrive.ckpt"
VLM_PATH="${VLM_PATH:-/path/to/ReCogDrive-VLM-2B}"
CAM_TYPE="${CAM_TYPE:-single}"
FRONT_SHORT_SIDE="${FRONT_SHORT_SIDE:-960}"
SIDE_SHORT_SIDE="${SIDE_SHORT_SIDE:-480}"
VLM_MAX_LENGTH="${VLM_MAX_LENGTH:-8704}"

AGENT_OVERRIDES=(
  agent.cam_type="${CAM_TYPE}"
)
if [[ "${CAM_TYPE}" == "multi" ]]; then
  AGENT_OVERRIDES+=(
    agent.front_short_side="${FRONT_SHORT_SIDE}"
    agent.side_short_side="${SIDE_SHORT_SIDE}"
    agent.vlm_max_length="${VLM_MAX_LENGTH}"
  )
fi

# 1. Set NAVSIM dataset and related environment variables
# 2. Configure torchrun (e.g., single machine: --nproc_per_node=8; adjust for multi-node)
# 3. Set agent.vlm_path and agent.checkpoint_path CHECKPOINT


torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path="${VLM_PATH}" \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    experiment_name=recogdrive_agent_eval \
    "${AGENT_OVERRIDES[@]}" \
    "$@"
