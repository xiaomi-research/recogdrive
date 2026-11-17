#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NAVSIM_TREE="${NAVSIM_TREE:-1.1}"
export WAYMOE2E=True
export MODEL_FAMILY="${MODEL_FAMILY:-internvl3}"
export CACHE_HIDDEN_STATE=False
export TRAINING_TARGET="${TRAINING_TARGET:-waypoint}"
export SAMPLING_METHOD="${SAMPLING_METHOD:-ddim}"
export TRAIN_STAGE=rl
DEV_ARGS=()
case "${WAYMO_DEV_RFS_RL:-0}" in
  0) WAYMO_DEV_RFS_RL=0 ;;
  1)
    if [[ "${WAYMOE2E_TRAIN_SPLIT:-val}" != val || "${WAYMOE2E_VAL_SPLIT:-val}" != val ||
          "${WAYMO_GRPO_REWARD:-waymo_rfs}" != waymo_rfs ]]; then
      echo "WAYMO_DEV_RFS_RL=1 requires train and score split val and WAYMO_GRPO_REWARD=waymo_rfs." >&2
      exit 2
    fi
    export WAYMOE2E_TRAIN_SPLIT=val WAYMOE2E_VAL_SPLIT=val
    export SKIP_VALIDATION=True
    WAYMO_GRPO_REWARD=waymo_rfs
    export EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_waymoe2e_stage3_dev_rfs}"
    export LOG_FILE="${LOG_FILE:-train_recogdrive_waymoe2e_stage3_dev_rfs.txt}"
    DEV_ARGS=(agent.grpo_waymo_dev_rfs=true evaluator.name=waymoe2e)
    echo "Waymo developer experiment: RFS trains on val; scores on the same split are development-set results."
    ;;
  *) echo "WAYMO_DEV_RFS_RL must be 0 or 1." >&2; exit 2 ;;
esac
export WAYMOE2E_TRAIN_SPLIT="${WAYMOE2E_TRAIN_SPLIT:-training}"
export WAYMOE2E_VAL_SPLIT="${WAYMOE2E_VAL_SPLIT:-val}"
WAYMO_GRPO_REWARD="${WAYMO_GRPO_REWARD:-waymo_ade}"

case "${WAYMOE2E_TRAIN_SPLIT,,}" in
  val|valid|validation|test|testing|eval|evaluation)
    if [[ "${WAYMO_DEV_RFS_RL}" != 1 ]]; then
      echo "WAYMOE2E_TRAIN_SPLIT must name a training cache split; RFS on val requires WAYMO_DEV_RFS_RL=1." >&2
      exit 2
    fi
    ;;
esac

export CACHE_PATH="${CACHE_PATH:-${WAYMOE2E_CACHE_PATH:-}}"
if [[ -z "${CACHE_PATH}" ]]; then
  echo "Set CACHE_PATH or WAYMOE2E_CACHE_PATH to a WOD-E2E training cache." >&2
  exit 2
fi

export CHECKPOINT="${STAGE2_CHECKPOINT:-${CHECKPOINT:-}}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Set STAGE2_CHECKPOINT to a stage-2 checkpoint trained with a 5 s Waymo horizon." >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Stage-2 checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

REWARD_ARGS=(
  "agent.grpo_reward=${WAYMO_GRPO_REWARD}"
  "agent.grpo_reward_cache_path=${CACHE_PATH}"
  "agent.grpo_reward_split=${WAYMOE2E_TRAIN_SPLIT}"
)
case "${WAYMO_GRPO_REWARD}" in
  waymo_ade)
    WAYMOE2E_LABELLED_ONLY=False
    if [[ ! -d "${CACHE_PATH}/${WAYMOE2E_TRAIN_SPLIT}" ]] || \
       ! find "${CACHE_PATH}/${WAYMOE2E_TRAIN_SPLIT}" -mindepth 2 -maxdepth 2 -type f -name targets.gz -size +0c \
         -exec sh -c 'test -s "${1%/*}/features.gz"' sh {} \; -print -quit | grep -q .; then
      echo "No nonempty WOD-E2E training features/targets pairs found under ${CACHE_PATH}/${WAYMOE2E_TRAIN_SPLIT}." >&2
      exit 2
    fi
    ;;
  waymo_rfs)
    WAYMOE2E_LABELLED_ONLY=True
    if [[ ! -d "${CACHE_PATH}/${WAYMOE2E_TRAIN_SPLIT}" ]] || \
       ! find "${CACHE_PATH}/${WAYMOE2E_TRAIN_SPLIT}" -mindepth 2 -maxdepth 2 -type f -name eval.gz -size +0c -print -quit | grep -q .; then
      echo "No rater-labelled WOD-E2E training frames found under ${CACHE_PATH}/${WAYMOE2E_TRAIN_SPLIT}." >&2
      echo "Convert rater-labelled frames for the selected Waymo RFS split first." >&2
      exit 2
    fi
    ;;
  *)
    echo "WAYMO_GRPO_REWARD must be waymo_ade or waymo_rfs, got ${WAYMO_GRPO_REWARD}." >&2
    exit 2
    ;;
esac

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_waymoe2e_stage3_rl}"
export LOG_FILE="${LOG_FILE:-train_recogdrive_waymoe2e_stage3_rl.txt}"

bash "${SCRIPT_DIR}/run_recogdrive_train.sh" \
  "${REWARD_ARGS[@]}" \
  "${DEV_ARGS[@]}" \
  "waymoe2e_labelled_only=${WAYMOE2E_LABELLED_ONLY}" \
  agent.trajectory_sampling.time_horizon=5 \
  "$@"
