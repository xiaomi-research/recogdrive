# Exp4 input recipe and Navhard checkpoint evaluation

## Exp4: fixed InternVL tiles

Build an online image-path cache with all the selected cameras. Use the matching InternVL weights and pass the exact tile counts, in addition to the 960/480 resize sizes. A front budget of 3 is two crops plus one global thumbnail; a budget of 1 for each other camera is one whole-view tile. An eight-camera sample has 10 tiles. If an explicit budget is set, a missing camera fails instead of silently changing the input.

```bash
EXP4=(agent.cam_type=multi agent.front_short_side=960 agent.side_short_side=480
      agent.front_num_patches=3 agent.side_num_patches=1 agent.vlm_max_length=8704
      agent.lora_rank=128 train.ema_decay=0.999)

CACHE_HIDDEN_STATE=False CACHE_PATH=/path/to/multi_view_cache VLM_PATH=/path/to/internvl \
MAX_EPOCHS=30 GPUS=4 SAMPLING_METHOD=ddim \
bash scripts/train/run_recogdrive_internvl3_waypoint_il_muon.sh "${EXP4[@]}"
```

On a single 24 GiB RTX 3090, use batch size 1 and enable decoder activation checkpointing:

```bash
CACHE_HIDDEN_STATE=False CACHE_PATH=/path/to/multi_view_cache VLM_PATH=/path/to/internvl \
MAX_EPOCHS=30 GPUS=1 BATCH_SIZE=1 SAMPLING_METHOD=ddim \
bash scripts/train/run_recogdrive_internvl3_waypoint_il_muon.sh "${EXP4[@]}" \
  train.strategy=ddp train.precision=bf16 train.compile=false \
  'train.activation_checkpointing=[Qwen2DecoderLayer]'
```

These single-card settings passed one real eight-camera NAVSIM IL update and a fresh checkpoint reload with LoRA rank 128 and the 8704-token limit. Peak allocated GPU memory was 8.36 GiB; the run without activation checkpointing exhausted GPU memory before the first update. This check does not establish memory usage for larger batches or for RL.

For RL, reuse the same overrides and cache, set `OPTIMIZER_TYPE=muon`, `CHECKPOINT` to the matching IL checkpoint and `METRIC_CACHE_PATH` to the training metric cache, and use `scripts/train/run_recogdrive_internvl3_waypoint_rl.sh`. Tile counts default to unset for other runs and do not apply to Qwen.

## Navhard: use the saved training configuration

Use the checkpoint and Hydra config from the same run. The NAVSIM 2.0 two-stage launcher keeps the trained agent's model family, camera settings, patch counts, LoRA, sampling method and action normalization while disabling training-only modes for inference.

```bash
TRAINING_CONFIG=/path/to/train/run/code/hydra/config.yaml \
CHECKPOINT=/path/to/train/run/last.ckpt \
CACHE_PATH=/path/to/navhard_two_stage_metric_cache \
OUTPUT_DIR=/path/to/new/navhard_eval \
bash scripts/evaluation/run_recogdrive_agent_epdms_navsim20_two_stage.sh
```

Supply the matching NAVSIM 2.0 test logs, sensor blobs and maps. The launcher writes `standalone_result.json` and the scorer log under `eval/standalone/eval.log`. A failed scorer or missing score makes the launcher return a nonzero exit. Scorer options such as `worker=sequential` can follow the command. Runs without `TRAINING_CONFIG` still accept explicit Hydra overrides after the original InternVL/single-view/DDIM defaults.
