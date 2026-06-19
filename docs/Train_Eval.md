# ReCogDrive Training and Evaluation

## Stage 1: Vision-Language Models Driving Pretraining

First, you need to download **13 QA datasets** (e.g., *DriveLM*, *LingoQA*, etc.) as mentioned in the paper.  
Due to dataset privacy policies, we are currently unable to release the JSON files. These files may be released later if permission is granted by the dataset authors. Once obtained, you should configure the corresponding JSON files under `./internvl_chat/shell/data_info`.

You can also generate the **ReCogDrive dataset on NAVSIM** following the steps below:

```bash
cd ./scripts
sh generate_dataset/generate_internvl_dataset.sh              # trajectory dataset
sh generate_dataset/generate_internvl_dataset_pipeline.sh     # auto-labeled dataset with pipeline
```
Note: Before running the pipeline script, you need to deploy the corresponding VLM using vllm or Sglang for automatic generation.

Next, download the pretrained VLM weights from HuggingFace. ReCogDrive supports InternVL checkpoints up to InternVL3, including earlier InternVL versions, and QwenVL3 checkpoints. For InternVL3, examples include:
👉 [InternVL3-2B Weights](https://huggingface.co/OpenGVLab/InternVL3-2B)
👉 [InternVL3-8B Weights](https://huggingface.co/OpenGVLab/InternVL3-8B)

For QwenVL3, set `agent.vlm_type=qwen` and point `agent.vlm_path` to the QwenVL3 checkpoint. QwenVL3 requires a Transformers version with Qwen3-VL support and `qwen-vl-utils`. ReCogDrive will infer the Qwen hidden size from the checkpoint config when possible; if you use precomputed hidden-state caches from a custom model, set `agent.vlm_hidden_size` to match the cached feature dimension.

After downloading, go to `./internvl_chat/shell/internvl3.0/2nd_finetune` and configure the training script.  
You can launch the pretraining process with the following commands:

```bash
cd /path/to/internvl_chat
sh ./shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh
```


## Stage 2: Diffusion Planner Imitation Learning

You can download our pretrained **ReCogDrive VLM** from [ReCogDrive VLM](https://huggingface.co/collections/owl10/recogdrive-68bafa143de172bab8de5752).  

For the diffusion planner training, the first step is to **cache datasets for faster training**.  
Since DiT training converges relatively slowly, training VLM and DiT jointly can be very time-consuming. To accelerate, we cache the hidden states output by the VLM, which enables much faster training.  
> ⚠️ Note: Caching requires approximately **1–2 TB of disk space**. We are also working on faster training methods.  


### Step 1: Cache hidden states
```bash
# cache dataset for training
sh cache_dataset/run_caching_recogdrive_hidden_state.sh
```

### Step 2: Configure and run training

Configure the script `training/run_recogdrive_train_multi_node_2b.sh` and then start training:

```bash
sh training/run_recogdrive_train_multi_node_2b.sh
```

By default, the diffusion planner trains with waypoint targets. You can also train with delta targets:

```bash
sh training/run_recogdrive_train_multi_node_2b.sh agent.training_target=delta
```

In delta mode, the target is the per-step trajectory velocity computed from NAVSIM waypoints. The delta normalization constants are built into the planner and were recomputed from `Navsim_Traj/dataset_navsim_traj.jsonl` in [ReCogDrive_Pretraining](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Navsim_Traj), so you do not need to pass `delta_norm_min` or `delta_norm_max`.

For accelerate training, we provide eight entry scripts covering pretrained VLM family, target type, and training stage:

| VLM | Target | Stage | Script |
| :---: | :---: | :---: | :--- |
| InternVL3 | waypoint | IL | `scripts/training_accelerate/run_recogdrive_internvl3_waypoint_il.sh` |
| InternVL3 | waypoint | RL | `scripts/training_accelerate/run_recogdrive_internvl3_waypoint_rl.sh` |
| InternVL3 | delta | IL | `scripts/training_accelerate/run_recogdrive_internvl3_delta_il.sh` |
| InternVL3 | delta | RL | `scripts/training_accelerate/run_recogdrive_internvl3_delta_rl.sh` |
| QwenVL3 | waypoint | IL | `scripts/training_accelerate/run_recogdrive_qwenvl3_waypoint_il.sh` |
| QwenVL3 | waypoint | RL | `scripts/training_accelerate/run_recogdrive_qwenvl3_waypoint_rl.sh` |
| QwenVL3 | delta | IL | `scripts/training_accelerate/run_recogdrive_qwenvl3_delta_il.sh` |
| QwenVL3 | delta | RL | `scripts/training_accelerate/run_recogdrive_qwenvl3_delta_rl.sh` |

These wrappers call `scripts/training_accelerate/run_recogdrive_accel_variant.sh`. Override paths with environment variables such as `VLM_PATH`, `CACHE_PATH`, `CHECKPOINT`, and `METRIC_CACHE_PATH`.

### WaymoE2E Stage 2 / Stage 3 Training

Following the RAP-style WaymoE2E cache format, ReCogDrive can train from cached samples without porting the full raw Waymo parser into this repository. The cache must be generated with the ReCogDrive agent feature/target builders, because the trainer expects keys such as `history_trajectory`, `high_command_one_hot`, `status_feature`, and `last_hidden_state`. The expected cache layout is:

```text
${WAYMOE2E_CACHE_PATH}/training/<token>/features.gz
${WAYMOE2E_CACHE_PATH}/training/<token>/targets.gz
${WAYMOE2E_CACHE_PATH}/val/<token>/features.gz
${WAYMOE2E_CACHE_PATH}/val/<token>/targets.gz
```

Run WaymoE2E stage 2 imitation learning:

```bash
WAYMOE2E_CACHE_PATH=/path/to/waymoe2e_recogdrive_cache \
VLM_PATH=/path/to/internvl3_or_qwenvl3 \
sh scripts/training_accelerate/run_recogdrive_waymoe2e_stage2_il.sh
```

Run WaymoE2E stage 3 RL from the stage 2 checkpoint:

```bash
WAYMOE2E_CACHE_PATH=/path/to/waymoe2e_recogdrive_cache \
CHECKPOINT=/path/to/waymoe2e_stage2_il.ckpt \
METRIC_CACHE_PATH=/path/to/waymoe2e_metric_cache \
VLM_PATH=/path/to/internvl3_or_qwenvl3 \
sh scripts/training_accelerate/run_recogdrive_waymoe2e_stage3_rl.sh
```

The stage 3 script reuses the existing ReCogDrive RL/PDM reward path. Therefore `METRIC_CACHE_PATH` must contain metric cache metadata and token names compatible with the WaymoE2E cache tokens. You can override `MODEL_FAMILY=qwenvl3`, `TRAINING_TARGET=delta`, `WAYMOE2E_TRAIN_SPLIT`, and `WAYMOE2E_VAL_SPLIT` when needed.

You can also enable **EMA (Exponential Moving Average)** during training for faster convergence. Note that this may lead to very slight performance degradation.

```bash
sh training/run_recogdrive_train_multi_node_ema_2b.sh
```

### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh
```

This will evaluate your trained agent using **PDM scores** on the navtest.




## Stage 3: Diffusion Planner Reinforcement Learning Training

In this stage, we perform **reinforcement learning (RL) training** on the Diffusion Planner  to further improve planning performance.

### Step 1: Metric Caching

First, you need to cache metrics for the training and test sets, which will be used for evaluation during RL training.

> ⚠️ **Note:** As mentioned in [Issue #10](https://github.com/xiaomi-research/recogdrive/issues/10#issuecomment-3344730681), you **must use NumPy version 1.26.4 or above** to avoid potential errors during metric caching.

```bash
# cache metrics for navtrain
sh cache_dataset/run_metric_caching_train.sh

# cache metrics for navtest
sh cache_dataset/run_metric_caching.sh
```


### Step 2: Configure and Launch RL Training

After caching metrics, configure the RL training script and launch training:

```bash
# Example path to the RL training script
sh training/run_recogdrive_train_multi_node_rl_2b.sh
```

For flow-matching RL, set the planner to flow sampling and enable GRPO:

```bash
sh training/run_recogdrive_train_multi_node_rl_2b.sh agent.sampling_method=flow agent.grpo=True
```

This uses a Flow-GRPO-style SDE rollout to obtain per-step log-probabilities, then follows the original ReCogDrive RL objective with `-log_prob * advantage` instead of a clipped policy ratio. The main tunable parameters are `agent.flow_noise_level` and `agent.flow_sde_type` (`sde` or `cps`).

Before running, modify the script parameters as needed  according to your hardware and training requirements. This command will start RL training immediately after configuration.


### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh
```
This will evaluate your trained agent using **PDM scores** on the navtest.

