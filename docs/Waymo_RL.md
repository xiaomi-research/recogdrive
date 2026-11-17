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

Convert the official `val` TFRecords into the `val` cache, with all views if using the multi-view recipe. Stage 3 samples only frames with cached rater feedback and uses their RFS as the RL reward. Start from the matching 5-second Stage 2 checkpoint and use the same backbone, action normalization and camera settings. For the multi-view Qwen recipe:

```bash
WAYMO_DEV_RFS_RL=1 STAGE2_CHECKPOINT=/path/to/waymo_stage2_5s_qwenvl3.ckpt \
WAYMOE2E_CACHE_PATH=/path/to/waymoe2e_recogdrive_cache \
MODEL_FAMILY=qwenvl3 VLM_PATH=/path/to/UniDriveVLA_Stage1 GPUS=4 \
bash scripts/train/run_recogdrive_waymoe2e_stage3_rl.sh \
  agent.cam_type=multi agent.vlm_max_length=8704
```

This switch selects `val` and `waymo_rfs` for training, enables the Waymo evaluator on `val`, and skips the ordinary validation loss. Final metrics are written under `eval/waymoe2e/final/waymoe2e_dev_metrics.json` with keys such as `dev_rfs`. Scores on this split after RL are development-set results; use separate, unseen data for an independent evaluation. The switch does not permit `test` or other held-out splits as training inputs.
