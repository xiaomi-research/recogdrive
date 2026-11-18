# Waymo E2E stage-3 RL

The stage-3 launcher defaults to `WAYMO_GRPO_REWARD=waymo_ade`, which works with the public WOD-E2E training split. `WaymoADEReward` reads only the logged future waypoint tensor in each training sample's `targets.gz` and returns the negative mean XY distance over the first ten 0.5-second poses (5 seconds). This is an explicit research reward based on logged trajectories; it is not the official Waymo RFS score and does not claim to optimize the leaderboard metric.

The public WOD-E2E validation split provides Rater Feedback labels, while the training split provides logged future trajectories. By default stage 3 keeps validation out of training. `WAYMO_GRPO_REWARD=waymo_rfs` on a regular training split requires separately supplied rater labels. The explicit development experiment below instead trains on official `val`. Both paths use the same per-frame RFS formula; neither reproduces the official scenario-cluster aggregate.

Both rewards require a stage-2 checkpoint trained for the same 5-second horizon, model family, and sampling method used in stage 3. The stage-2 launcher and stage-3 launcher default to DDIM:

```bash
WAYMOE2E_CACHE_PATH=/path/to/waymoe2e_recogdrive_cache \
GPUS=4 bash scripts/train/run_recogdrive_waymoe2e_stage2_il.sh \
  agent.trajectory_sampling.time_horizon=5
```

Run stage 3 with the public logged-waypoint reward:

```bash
STAGE2_CHECKPOINT=/path/to/waymo_stage2_5s.ckpt \
WAYMOE2E_CACHE_PATH=/path/to/waymoe2e_recogdrive_cache \
WAYMOE2E_TRAIN_SPLIT=training GPUS=4 \
bash scripts/train/run_recogdrive_waymoe2e_stage3_rl.sh
```

To explicitly request RFS, provide a separate labelled training split:

```bash
STAGE2_CHECKPOINT=/path/to/waymo_stage2_5s.ckpt \
WAYMOE2E_CACHE_PATH=/path/to/custom_labelled_training_cache \
WAYMOE2E_TRAIN_SPLIT=training WAYMO_GRPO_REWARD=waymo_rfs GPUS=4 \
bash scripts/train/run_recogdrive_waymoe2e_stage3_rl.sh
```

Sources: [Waymo's released per-frame RFS implementation](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py), [WOD-E2E data proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto), [official metrics proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_metrics.proto), and [official E2E dataset description](https://waymo.com/open/data/e2e/).

### Explicit RFS development experiment on official val (473)

Convert the public training and official validation TFRecords with all views, then run 5-second IL on `training`. Use the same Bash session for both stages so they share the model, camera, LoRA, and sampling settings. Add any action-normalization overrides to the shared arguments.

```bash
export WAYMOE2E_CACHE_PATH=/path/to/waymo_multi_view_cache
PYTHONPATH=navsim1.1:. python -m recogdrive.data.waymoe2e \
  --out "$WAYMOE2E_CACHE_PATH" --split training --views all /path/to/wod_e2e/training_*.tfrecord-*
PYTHONPATH=navsim1.1:. python -m recogdrive.data.waymoe2e \
  --out "$WAYMOE2E_CACHE_PATH" --split val --views all /path/to/wod_e2e/val_*.tfrecord-*

export NAVSIM_TREE=1.1 MODEL_FAMILY=qwenvl3 OPTIMIZER_TYPE=muon CACHE_HIDDEN_STATE=False GPUS=4
export VLM_PATH=/path/to/UniDriveVLA_Nusc_Base_Stage1
export CACHE_PATH="$WAYMOE2E_CACHE_PATH"
export SAMPLING_METHOD=ddim  # Use the same selected sampler for both stages.
WAYMO_ARGS=(agent.cam_type=multi agent.vlm_max_length=8704 agent.lora_rank=128
            agent.lora_alpha=16 train.ema_decay=0.999 'train.replicate_frozen=[]'
            agent.trajectory_sampling.time_horizon=5
            'agent.qwen_image_short_sides=[960,480]' agent.qwen_max_input_tokens=8704)

WAYMOE2E_TRAIN_SPLIT=training WAYMOE2E_VAL_SPLIT=val MAX_EPOCHS=30 \
EXPERIMENT_NAME=exp5_waymo_il LOG_FILE=exp5_waymo_il.log \
bash scripts/train/run_recogdrive_waymoe2e_stage2_il.sh "${WAYMO_ARGS[@]}" evaluator.name=waymoe2e
```

Set `STAGE2_CHECKPOINT` to the checkpoint from that IL run. Stage 3 uses RFS from the labelled validation frames.

```bash
WAYMO_DEV_RFS_RL=1 WAYMOE2E_TRAIN_SPLIT=val WAYMOE2E_VAL_SPLIT=val WAYMO_GRPO_REWARD=waymo_rfs \
STAGE2_CHECKPOINT=/path/to/waymo_stage2_5s_qwenvl3.ckpt \
EXPERIMENT_NAME=exp5_waymo_val_rfs_rl LOG_FILE=exp5_waymo_val_rfs_rl.log \
bash scripts/train/run_recogdrive_waymoe2e_stage3_rl.sh \
  "${WAYMO_ARGS[@]}"
```

This switch selects `val` and `waymo_rfs` for training, enables the Waymo evaluator on `val`, and skips the ordinary validation loss. Final metrics are written under `eval/waymoe2e/final/waymoe2e_dev_metrics.json` with keys such as `dev_rfs`. Scores on this split after RL are development-set results; use separate, unseen data for an independent evaluation. The switch does not permit `test` or other held-out splits as training inputs.

For a small run on one RTX 3090, convert a few real records per split, with rater labels in `val`. Use `GPUS=1 BATCH_SIZE=1 MAX_EPOCHS=1` for both launchers, including replacing the inline `MAX_EPOCHS=30` in the IL command. Append the following to `WAYMO_ARGS` before launching either stage. Finishing the epoch on the small cache writes the IL checkpoint used by stage 3.

```bash
WAYMO_ARGS+=(train.strategy=ddp train.precision=bf16 train.compile=false
             'train.activation_checkpointing=[Qwen3VLTextDecoderLayer]')
```
