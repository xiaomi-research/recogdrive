#!/usr/bin/env bash
# Alternating VQA/planning batches through one VLM. Source weights set the task ratio.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VQA_TRAIN_JSONL:?set VQA_TRAIN_JSONL}"
: "${VQA_VAL_JSONL:?set VQA_VAL_JSONL}"
case "${MODEL_FAMILY:-internvl3}" in
  qwenvl3) DEFAULT_CHECKPOINT_LAYER=Qwen3VLTextDecoderLayer ;;
  *) DEFAULT_CHECKPOINT_LAYER=Qwen2DecoderLayer ;;
esac
TRAIN_STAGE="${TRAIN_STAGE:-il}" LR_SCHEDULE=step CACHE_HIDDEN_STATE=False bash "${SCRIPT_DIR}/run_recogdrive_train.sh" \
  data_loader=mixture mixture_batch_by_task=true \
  "mixture=[{loader:navsim,weight:${PLANNING_WEIGHT:-1}},{loader:vqa,weight:${VQA_WEIGHT:-1}}]" \
  "vqa.train_jsonl=${VQA_TRAIN_JSONL}" "vqa.val_jsonl=${VQA_VAL_JSONL}" \
  "vqa.image_root=${VQA_IMAGE_ROOT:-null}" "vqa.max_length=${VQA_MAX_LENGTH:-8704}" \
  "agent.lora_rank=${LORA_RANK:-16}" train.strategy=ddp train.precision=fp32 train.find_unused_parameters=true \
  "train.activation_checkpointing=[${ACTIVATION_CHECKPOINT_LAYER:-${DEFAULT_CHECKPOINT_LAYER}}]" "$@"
