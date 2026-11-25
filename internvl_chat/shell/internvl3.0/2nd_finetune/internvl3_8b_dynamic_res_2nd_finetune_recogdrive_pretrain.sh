#!/usr/bin/env bash
set -e -o pipefail
set -x

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

OUTPUT_DIR='work_dirs/ReCogDrive_pretrain/internvl3_8b_finetune_full_recogdrive_pretrain'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes="${NNODES:-8}" \
  --node_rank="${MLP_ROLE_INDEX:-0}" \
  --master_addr="${MLP_WORKER_0_HOST:-127.0.0.1}" \
  --nproc_per_node=${GPUS} \
  --master_port="${MLP_WORKER_0_PORT:-$MASTER_PORT}" \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${MODEL_PATH:-/path/to/ckpt/InternVL3-8B}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data_info/recogdrive_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch "${MAX_DYNAMIC_PATCH:-16}" \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 32 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 10 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length "${MAX_SEQ_LENGTH:-12288}" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail "${USE_THUMBNAIL:-True}" \
  --ps_version 'v2' \
  --deepspeed "${ZERO_CONFIG:-zero_stage1_config.json}" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
