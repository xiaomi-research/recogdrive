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

### Driving VQA evaluation (DriveLM, LingoQA, DriveBench)

`recogdrive/eval/vqa` answers the questions with lmdeploy (one worker per visible GPU, resumable) and writes the benchmark's prediction files plus `score.json`. It needs `pip install lmdeploy`; DriveBench language metrics also need `language-evaluation`, and `--gpt` reads `OPENAI_API_KEY`. The default TurboMind engine matches the reported numbers; `--backend pytorch` runs lmdeploy's PyTorch engine where the TurboMind build does not match the installed torch.

```bash
# DriveLM: output.json and submission.json for the DriveLM server
python -m recogdrive.eval.vqa drivelm --model /path/to/vlm --data v1_1_val_nus_q_only.json --image-root /path/to/nuscenes --out out/drivelm
# LingoQA: Lingo-Judge accuracy
python -m recogdrive.eval.vqa lingoqa --model /path/to/vlm --data /path/to/LingoQA/val.parquet --image-root /path/to/LingoQA/images/val --out out/lingoqa
# DriveBench: clean or --corruption Fog / NoImage / ...
python -m recogdrive.eval.vqa drivebench --model /path/to/vlm --data drivebench-test-final.json --image-root /path/to/toolkit --out out/drivebench
```

The same benchmarks run as training evaluators, e.g. `evaluator.name=lingoqa evaluator.vqa.data=... evaluator.vqa.image_root=...`; they score the `*_vlm` export when the backbone is trained, otherwise `agent.vlm_path`.


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

The training cache always stores waypoint targets. In delta mode, targets are converted after loading the cache, so waypoint and delta training can share the same `CACHE_PATH`. Delta targets are per-step trajectory velocities computed from NAVSIM waypoints. The delta normalization constants are built into the planner and were recomputed from `Navsim_Traj/dataset_navsim_traj.jsonl` in [ReCogDrive_Pretraining](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Navsim_Traj), so you do not need to pass `delta_norm_min` or `delta_norm_max`.

The `recogdrive` trainer (FSDP2, torchrun) has eight entry scripts covering pretrained VLM family, target type, and training stage:

| VLM | Target | Stage | Script |
| :---: | :---: | :---: | :--- |
| InternVL3 | waypoint | IL | `scripts/train/run_recogdrive_internvl3_waypoint_il.sh` |
| InternVL3 | waypoint | RL | `scripts/train/run_recogdrive_internvl3_waypoint_rl.sh` |
| InternVL3 | delta | IL | `scripts/train/run_recogdrive_internvl3_delta_il.sh` |
| InternVL3 | delta | RL | `scripts/train/run_recogdrive_internvl3_delta_rl.sh` |
| QwenVL3 | waypoint | IL | `scripts/train/run_recogdrive_qwenvl3_waypoint_il.sh` |
| QwenVL3 | waypoint | RL | `scripts/train/run_recogdrive_qwenvl3_waypoint_rl.sh` |
| QwenVL3 | delta | IL | `scripts/train/run_recogdrive_qwenvl3_delta_il.sh` |
| QwenVL3 | delta | RL | `scripts/train/run_recogdrive_qwenvl3_delta_rl.sh` |

These wrappers call `scripts/train/run_recogdrive_train.sh`. Override paths with environment variables such as `VLM_PATH`, `CACHE_PATH`, `CHECKPOINT`, and `METRIC_CACHE_PATH`; any extra arguments are passed to Hydra, for example:

```bash
sh scripts/train/run_recogdrive_internvl3_waypoint_il.sh \
  train.activation_checkpointing=[LightningDiTBlock] \
  evaluator.name=navsim evaluator.metric_cache_path=/path/to/metric_cache
```

The `train.*` block of `default_training.yaml` sets parallelism and precision (`strategy`, `precision`, `reshard_after_forward`, `hsdp_shard_size`, `replicate_frozen`, `compile`, `activation_checkpointing`, `activation_checkpointing_layers`, `resume`). `data_loader` picks a registered dataset loader (`navsim`, `waymoe2e`, `nuscenes`, `mixture`); a new dataset registers a loader (see below) and returns `(features, targets, token)` samples. The full ReCogDrive model is kept in three places with identical code apart from import lines: `recogdrive/models/recogdrive/` + `recogdrive/adapters/navsim/` (used by the training launchers via `agent._target_`), and `navsim1.1/navsim/agents/recogdrive/` and `navsim2.0/navsim/agents/recogdrive/` (used by the NAVSIM evaluation scripts through `agent=recogdrive_agent`). Change all three together; checkpoints load in either. The agent's `worker_transform()` (prompt, image path, InternVL tiles) runs in the dataloader workers for every data loader. `evaluator.*` scores checkpoints with the NAVSIM tree's own PDMS/EPDMS entry. Periodic evaluation (`evaluator.every_n_epochs`) generates the trajectories with the training model on the training GPUs and runs the official scoring on CPU in the background, so it never competes with training for GPUs; `evaluator.split_overrides` can restrict it to a scene subset. The final evaluation runs the trained agent after training has released the GPUs. Other GPU evaluators (VQA) wait for the end of training unless `evaluator.gpus` names GPUs training does not use. `optimizer_type=muon` works with FSDP2: each matrix is orthogonalized once, by one rank. Its orthogonalized updates are scaled by `0.2 * sqrt(max(rows, cols))` to AdamW's update RMS (Moonlight's calibration, used by DMuon / WALL-OSS), so Muon runs with the learning rate and weight decay tuned for AdamW. Muon does not need fp32 parameters: as in DMuon (WALL-OSS), a parameter kept in bf16 gets an fp32 master copy in the optimizer, which takes the momentum, weight decay and update and is written back to the bf16 weight. So `train.strategy=ddp train.precision=bf16` trains Muon with bf16 weights (gradients are averaged in fp32, the EMA is kept in fp32); optimizers without master copies are refused there, and FSDP keeps fp32 master shards for every optimizer. The training log reports `step_ms`, `data_wait_ms`, and `samples/s`. The learning rate warms up (3 epochs for imitation learning, none for RL) and then follows a cosine to its minimum over the actual training: the trainer passes the length in optimizer steps (steps per epoch × `trainer.params.max_epochs`, or `max_steps` when that is shorter) and the schedule advances every optimizer step.

### Adapters, data sources, benchmarks, monitoring and inference

**Adapters.** A model plugs in as a policy adapter (contract in `recogdrive/adapters/__init__.py`): `trajectory_sampling`, `forward` (training: an object with `.loss`; eval: `{"pred_traj": (B, poses, 3)}`), `compute_loss`, `get_optimizers`, `get_target_builders`, and optionally `worker_transform(image_augment=None)`. Data sources and benchmarks only see this contract, so adding either leaves the model alone. `recogdrive/adapters/template.py` (`agent=recogdrive_template`) is a minimal ego-state MLP planner to copy for a new model; it trains on every data source and every open-loop benchmark scores it.

**Data sources.** Samples follow the contract in `recogdrive/data/registry.py`: 4 history poses at 0.5 s, the 8-dim ego status, `camera_paths` (every camera the source has, front first) and the future trajectory at the model's horizon, all in the current ego frame. A new source is a function decorated with `@register("name")`, in a module listed in `plugins=[my_pkg.my_source]` when it lives outside this repository. Several sources train together with per-source weights (each source's share of the draws follows its weight; validation concatenates the sources):

```bash
... data_loader=mixture \
  'mixture=[{loader: navsim, weight: 1.0}, {loader: waymoe2e, weight: 0.5, overrides: {cache_path: /path/to/wod_cache}}]'
```

Training-split augmentation is off by default: `augment.history_dropout` and `augment.ego_status_dropout` zero the ego history / velocity and acceleration with the given probability (before the prompt is built), and `augment.color_jitter.p` (with `brightness`, `contrast`, `saturation`, `hue`), `augment.grayscale_p` and `augment.blur_p` are photometric image augmentations; geometric ones would move the scene against the labels.

**Horizon and camera views.** `agent.trajectory_sampling.time_horizon` sets the prediction horizon (4 s = 8 poses by default, 5 s for the WOD-E2E metrics) and every data source cuts its targets to it. For 5 s widen the planner's output range, which by default stops at 65 m forward: `agent.action_norm_min=[-2.0,-40.0,-3.2] agent.action_norm_max=[160.0,40.0,3.2]`. `agent.cam_type=multi` feeds every surround view the source has (`agent.cameras` selects them), the front view rescaled to a 960-pixel short side and the others to 480; the padded prompt length `agent.vlm_max_length` defaults to 8704 for multi-view (WOD-E2E's 8 views take about 8240 tokens, nuScenes' 6 about 6450) and stays 2800 for the single-view recipe. NAVSIM multi-view reads scenes, or a cache rebuilt with `agent.cam_type=multi`; WOD-E2E needs a cache converted with `--views all`.

**Benchmarks.** `evaluator.name` takes one benchmark or a list, `evaluator.<name>.*` overrides the shared keys for one of them, and results go to `eval/<benchmark>/<tag>/` and into the metrics as `eval/<benchmark>/<metric>`.

| `evaluator.name` | Metrics | How it runs |
| :--- | :--- | :--- |
| `navsim` | PDMS (NAVSIM 1.1), EPDMS (2.0) on `evaluator.split`: `navtest`, or on 2.0 `navhard_two_stage` / `navsafe_two_stage` with the official two-stage entry | periodic: trajectories from the training model, official scoring on CPU; final: the trained agent |
| `nuscenes` | L2 and collision rate at 1 / 2 / 3 s, ST-P3 (average up to t) and UniAD (at t) conventions | in the training process |
| `waymoe2e` | Rater Feedback Score, ADE at 3 / 5 s (needs a 5 s horizon) | in the training process, on the rater-labelled frames |
| `drivelm`, `lingoqa`, `drivebench` | VQA scores | child process |

Two-stage splits read the synthetic second-stage scenes, and their metric cache must be built for the split. Point both the trajectory generation and the scoring entry at them; Hydra list elements containing `=` or `/` need quotes:

```bash
S="'synthetic_scenes_path=/data/navhard_two_stage/synthetic_scene_pickles','synthetic_sensor_path=/data/navhard_two_stage/sensor_blobs'"
... evaluator.name=navsim evaluator.split=navhard_two_stage evaluator.metric_cache_path=/path/to/navhard_metric_cache \
  "evaluator.split_overrides=[$S]" "evaluator.overrides=[$S]"
```

**Monitoring.** Every run writes `metrics.csv` (step, epoch, key, value: training loss, gradient norm, learning rate, step time, data wait, throughput, validation loss, benchmark metrics) and redraws `loss_curve.png` after each epoch. `monitor.wandb=online` or `offline` also logs to Weights & Biases (`pip install wandb`; a resumed run continues the same W&B run); without the package the run goes on with the CSV.

**Inference.** `scripts/infer/run_recogdrive_infer.sh` runs a checkpoint on any registered data source and split and writes `predictions.json` (token to ego-frame poses) plus `vis/<token>.png` (front camera and bird's-eye view of history, ground truth and prediction) for the first `VISUALIZE` samples; `SPLIT=test` with the NAVSIM loader runs every scene of `train_test_split`:

```bash
CHECKPOINT=/path/to/epoch_0010.ckpt VLM_PATH=/path/to/vlm GPUS=2 \
  bash scripts/infer/run_recogdrive_infer.sh data_loader=nuscenes nuscenes.root=/path/to/nuscenes nuscenes.version=v1.0-trainval
```

### WaymoE2E Stage 2 / Stage 3 Training

Build the cache from the raw WOD-E2E TFRecords (front camera center-cropped to NAVSIM's 16:9 so the VLM prompt keeps its 9-patch budget, 0.5 s history spacing and targets up to 5 s in the current ego frame, intent as the driving command). `--views all` also writes the other seven cameras for multi-view training, and the rater-labelled validation frames get an `eval.gz` with the rater trajectories the Rater Feedback Score needs:

```bash
PYTHONPATH=navsim1.1:. python -m recogdrive.data.waymoe2e --out ${WAYMOE2E_CACHE_PATH} --split training /path/to/wod_e2e/training_*.tfrecord-*
PYTHONPATH=navsim1.1:. python -m recogdrive.data.waymoe2e --out ${WAYMOE2E_CACHE_PATH} --split val /path/to/wod_e2e/val_*.tfrecord-*
```

The resulting layout (RAP-style caches with the same keys also work):

```text
${WAYMOE2E_CACHE_PATH}/training/<token>/features.gz
${WAYMOE2E_CACHE_PATH}/training/<token>/targets.gz
${WAYMOE2E_CACHE_PATH}/val/<token>/features.gz
${WAYMOE2E_CACHE_PATH}/val/<token>/targets.gz
${WAYMOE2E_CACHE_PATH}/val/<token>/eval.gz      # rater-labelled frames only
```

Run WaymoE2E stage 2 imitation learning:

```bash
WAYMOE2E_CACHE_PATH=/path/to/waymoe2e_recogdrive_cache \
VLM_PATH=/path/to/internvl3_or_qwenvl3 \
sh scripts/train/run_recogdrive_waymoe2e_stage2_il.sh
```

Run WaymoE2E stage 3 RL from the stage 2 checkpoint:

```bash
WAYMOE2E_CACHE_PATH=/path/to/waymoe2e_recogdrive_cache \
CHECKPOINT=/path/to/waymoe2e_stage2_il.ckpt \
METRIC_CACHE_PATH=/path/to/waymoe2e_metric_cache \
VLM_PATH=/path/to/internvl3_or_qwenvl3 \
sh scripts/train/run_recogdrive_waymoe2e_stage3_rl.sh
```

The stage 3 script reuses the existing ReCogDrive RL/PDM reward path. Therefore `METRIC_CACHE_PATH` must contain metric cache metadata and token names compatible with the WaymoE2E cache tokens. You can override `MODEL_FAMILY=qwenvl3`, `TRAINING_TARGET=delta`, `WAYMOE2E_TRAIN_SPLIT`, and `WAYMOE2E_VAL_SPLIT` when needed.

You can also enable **EMA (Exponential Moving Average)** during training for faster convergence. Note that this may lead to very slight performance degradation. Validation then runs on the averaged weights, every checkpoint gets an `*-EMA.ckpt` twin, evaluators score the EMA twin, and the average is part of the resume state. `train.ema_decay` is the per-step decay at global batch `train.ema_reference_batch` (default 128); other batch sizes use `decay ** (batch / reference)`, which keeps the averaging horizon the same number of samples.

```bash
sh scripts/train/run_recogdrive_internvl3_waypoint_il.sh train.ema_decay=0.999
```

**LoRA on the VLM.** `agent.lora_rank` (with `agent.lora_alpha`, `agent.lora_dropout`, `agent.lora_targets`) adds peft LoRA adapters to the VLM's language model, by default its attention and MLP projections, and trains them together with the planner while the VLM's own weights stay frozen. `agent.train_backbone=true` fine-tunes the whole VLM instead; the two are exclusive. The VLM now back-propagates, so checkpoint its decoder layers' activations (`Qwen2DecoderLayer` for InternVL3). Each checkpointed layer runs its forward again in backward; `train.activation_checkpointing_layers=N` recomputes only the first N of the matched layers and keeps the others' activations. With 4 samples of 2800 tokens per RTX 3090, every InternVL3-2B decoder layer left out saves about 24 ms per step and costs about 1.2 GiB: all 28 layers take 3745 ms and 9.1 GiB, 20 layers 3557 ms and 18.4 GiB, 18 layers 3510 ms and 20.7 GiB, and 16 no longer fit in 24 GiB. Set the smallest count that still leaves memory to spare; larger GPUs can drop checkpointing altogether. The adapters are saved in the planner checkpoint (no `_vlm` export) and re-injected when an agent with the same `lora_*` settings loads it, so NAVSIM evaluation and inference need nothing else; the VQA benchmarks load a HuggingFace VLM and refuse LoRA runs until the adapters are merged.

```bash
sh scripts/train/run_recogdrive_internvl3_waypoint_il.sh agent.lora_rank=16 'train.activation_checkpointing=[Qwen2DecoderLayer]'
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

