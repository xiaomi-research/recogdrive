# ReCogDrive October 17 experiment recipes

Run these examples from the repository root with Bash. They use the launchers shipped on `update-cache`, with online image encoding, waypoint targets, Muon, and EMA.

| EXP | VLM | Views | IL epochs | Sampling |
| --- | --- | --- | --- | --- |
| 1 | frozen InternVL | front | 100 | DDIM |
| 2 | frozen InternVL | front | 100 | Flow |
| 3 | InternVL LoRA rank 128 | front | 30 | explicitly selected DDIM/Flow |
| 4 | InternVL LoRA rank 128 | surround | 30 | same selected method |
| 5 | Qwen-VL LoRA rank 128 | surround | 30 | same selected method |

The commands set EMA decay to 0.999 and LoRA alpha to 16. They use 30 IL epochs for Exp3–5 and the launcher default of 10 epochs for RL. Save each run's resolved Hydra configuration together with its checkpoint.

## NAVSIM 1.1: start with Exp1/2

Set the dataset, map, and output paths for your environment. Keep the arrays below in the same Bash session for the following NAVSIM examples. [Training setup](Train_Eval.md)

```bash
export NAVSIM_TREE=1.1 GPUS=4 CACHE_HIDDEN_STATE=False OPTIMIZER_TYPE=muon
export VLM_PATH=/path/to/matching_internvl
export CACHE_PATH=/path/to/navsim_front_image_cache
bash scripts/cache_dataset/run_caching_recogdrive_no_hidden_state.sh

FROZEN=(agent.lora_rank=0 train.ema_decay=0.999)
LORA128=(agent.lora_rank=128 agent.lora_alpha=16 train.ema_decay=0.999 'train.replicate_frozen=[]')
MULTI=(agent.cam_type=multi agent.vlm_max_length=8704)

SAMPLING_METHOD=ddim MAX_EPOCHS=100 EXPERIMENT_NAME=exp1_navsim11_il LOG_FILE=exp1_navsim11_il.log \
bash scripts/train/run_recogdrive_internvl3_waypoint_il_muon.sh "${FROZEN[@]}"
SAMPLING_METHOD=flow MAX_EPOCHS=100 EXPERIMENT_NAME=exp2_navsim11_il LOG_FILE=exp2_navsim11_il.log \
bash scripts/train/run_recogdrive_internvl3_waypoint_il_muon.sh "${FROZEN[@]}"

# Set this to the method selected from Exp1/2.
SAMPLE=ddim
SAMPLING_METHOD="$SAMPLE" MAX_EPOCHS=30 EXPERIMENT_NAME=exp3_navsim11_il LOG_FILE=exp3_navsim11_il.log \
bash scripts/train/run_recogdrive_internvl3_waypoint_il_muon.sh "${LORA128[@]}"
```

Use the same initialization, split, and budget for Exp1 and Exp2. Set `SAMPLE` to the selected `ddim` or `flow` for Exp3–5. Prefix an individual training command with `MAX_STEPS=1 SKIP_VALIDATION=True` for a one-step smoke check.

## Exp4: exact InternVL image budgets

```bash
export CACHE_PATH=/path/to/navsim_multi_image_cache
bash scripts/cache_dataset/run_caching_recogdrive_no_hidden_state.sh agent.cam_type=multi
EXP4=("${LORA128[@]}" "${MULTI[@]}" agent.front_short_side=960 agent.side_short_side=480
      agent.front_num_patches=3 agent.side_num_patches=1)

GPUS=1 BATCH_SIZE=1 SAMPLING_METHOD="$SAMPLE" MAX_EPOCHS=30 \
EXPERIMENT_NAME=exp4_navsim11_il LOG_FILE=exp4_navsim11_il.log \
bash scripts/train/run_recogdrive_internvl3_waypoint_il_muon.sh "${EXP4[@]}" \
  train.strategy=ddp train.precision=bf16 train.compile=false \
  'train.activation_checkpointing=[Qwen2DecoderLayer]'
```

`agent.front_num_patches=3` means two aspect-ratio-selected 448×448 crops plus one global thumbnail. `agent.side_num_patches=1` means the whole view resized to one 448×448 image. Counts include the global thumbnail. The defaults are null and continue using dynamic tiling. Explicit budgets reject missing selected cameras rather than silently training a front-only cache as Exp4.

The example uses the single-3090 settings verified for one IL update and reload. Both NAVSIM trees expose the tile fields and training evaluators inherit the agent config. Use the saved config for evaluation where supported, or pass the same model, camera, tile, LoRA, and sampling settings to the standalone evaluator. [Exp4 settings and memory check](Exp4_Navhard.md)

## Exp5: Qwen visual tokenization

```bash
SAMPLING_METHOD="$SAMPLE" MAX_EPOCHS=30 EXPERIMENT_NAME=exp5_navsim11_il LOG_FILE=exp5_navsim11_il.log \
VLM_PATH=/path/to/UniDriveVLA_Nusc_Base_Stage1 \
CACHE_PATH=/path/to/navsim_multi_image_cache \
bash scripts/train/run_recogdrive_qwenvl3_unidrive_il.sh "${LORA128[@]}" "${MULTI[@]}"
```

Exp5 uses Qwen's native visual tokenizer and the surround-view image paths. On published commit `f257fe7`, the 960/480 resize settings apply to the InternVL transform and Qwen uses its processor's sizing. An Exp5 comparison with strict per-view resizing requires a revision that forwards those sizes to the Qwen processor.

The plain Qwen3-VL-2B-Instruct weights available on the server are sufficient for a smoke check, but are not a substitute for the requested UniDriveVLA Stage1 weights in the final Exp5 comparison.

## RL and other datasets

```bash
# Exp4 RL. Use the same sampler, weights, and image cache as the matching IL run.
SAMPLING_METHOD="$SAMPLE" CHECKPOINT=/path/to/exp4_il.ckpt \
METRIC_CACHE_PATH=/path/to/navtrain_metric_cache CACHE_PATH=/path/to/navsim_multi_image_cache \
EXPERIMENT_NAME=exp4_navsim11_rl LOG_FILE=exp4_navsim11_rl.log \
bash scripts/train/run_recogdrive_internvl3_waypoint_rl.sh "${EXP4[@]}"

# NAVSIM 2.0 cache and Exp4 IL. Use the matching 2.0 data and map paths.
NAVSIM_TREE=2.0 GPUS=1 CACHE_PATH=/path/to/navsim20_multi_image_cache \
bash scripts/cache_dataset/run_caching_recogdrive_no_hidden_state.sh agent.cam_type=multi
NAVSIM_TREE=2.0 CACHE_PATH=/path/to/navsim20_multi_image_cache \
SAMPLING_METHOD="$SAMPLE" MAX_EPOCHS=30 EXPERIMENT_NAME=exp4_navsim20_il LOG_FILE=exp4_navsim20_il.log \
bash scripts/train/run_recogdrive_internvl3_waypoint_il_muon.sh "${EXP4[@]}"

# PhysicalAI multi-view cache and 5-second IL.
PYTHONPATH=navsim1.1:. python -m recogdrive.data.physicalai \
  --src /path/to/physicalai_download --out /path/to/physicalai_cache --views all
PAI_VIEWS=multi MODEL_FAMILY=qwenvl3 SAMPLING_METHOD="$SAMPLE" MAX_EPOCHS=30 \
PHYSICALAI_ROOT=/path/to/physicalai_cache VLM_PATH=/path/to/UniDriveVLA_Nusc_Base_Stage1 \
EXPERIMENT_NAME=exp5_physicalai_il LOG_FILE=exp5_physicalai_il.log \
bash scripts/train/run_recogdrive_physicalai_il.sh "${LORA128[@]}"
```

For Exp1/2 RL use `FROZEN`, the front cache, and that experiment's sampler. For Exp3 RL use `LORA128` with the front cache. In each case supply the corresponding IL checkpoint and a distinct experiment name and log. Plan the RL batch size separately from the single-card IL check.

Waymo and PhysicalAI use 5-second targets. Validate the action normalization against the 5-second trajectories and reuse it for training and evaluation. For Waymo, the paired multi-view IL and `WAYMO_DEV_RFS_RL=1` commands share the same model settings. RFS on the same official val used for RL is a development-set result. [Waymo IL and RL](Waymo_RL.md)

Run HUGSIM with the matching saved Hydra config and trained checkpoint, the official simulator environment, and scene assets. [Closed-loop evaluation](Closedloop_Evaluation.md)
