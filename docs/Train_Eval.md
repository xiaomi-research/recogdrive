# ReCogDrive Training and Evaluation

## Stage 1: Vision-Language Models Driving Pretraining

The training annotations that have been released are in the [ReCogDrive pretraining collection](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining). Obtain the source images from the corresponding datasets and configure their paths under \`internvl_chat/shell/data_info\`. The collection contains training subsets. For validation or testing, use the splits supplied by each benchmark.

You can also generate the **ReCogDrive dataset on NAVSIM** following the steps below:

```bash
cd ./scripts
sh generate_dataset/generate_internvl_dataset.sh              # trajectory dataset
sh generate_dataset/generate_internvl_dataset_pipeline.sh     # auto-labeled dataset with pipeline
```
Note: Before running the pipeline script, you need to deploy the corresponding VLM using vllm or Sglang for automatic generation.

Next, download the **InternVL pretrained weights** from HuggingFace:  
👉 [InternVL3-2B Weights](https://huggingface.co/OpenGVLab/InternVL3-2B)
👉 [InternVL3-8B Weights](https://huggingface.co/OpenGVLab/InternVL3-8B)

After downloading, go to `./internvl_chat/shell/internvl3.0/2nd_finetune` and configure the training script.  
You can launch the pretraining process with the following commands:

```bash
cd /path/to/internvl_chat
bash ./shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh
```


### Reducing Stage 1 activation memory

The published InternVL3-8B launcher uses a sequence limit of 12,288 tokens, a dynamic tile budget of 16, 448-pixel tiles and thumbnails. For multiple views, the loader divides the tile budget among the images, with at least one tile each. A thumbnail can add a tile per view that needs multiple tiles. At the default image and patch size, each tile contributes 256 visual tokens. A shorter sequence limit alone can cut off image tokens or the supervised answer.

For a one-node debugging run, reduce the tile budget first and turn off thumbnails while keeping the six views together. Check that the input still contains the answer within the sequence limit. The values below are a starting point for measuring memory. Actual capacity depends on the samples and the DeepSpeed setup.

```bash
NNODES=1 GPUS=8 PER_DEVICE_BATCH_SIZE=1 MAX_DYNAMIC_PATCH=6 USE_THUMBNAIL=False MAX_SEQ_LENGTH=4096 \
  MODEL_PATH=/path/to/InternVL3-8B \
  bash ./shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh
```

Set `ZERO_CONFIG` to your DeepSpeed configuration if needed. The launcher uses ZeRO-1 by default. Gradient checkpointing remains enabled.

### Score the Stage 1 VLM on NAVSIM

Use the [Stage 1 VLM weights](https://huggingface.co/owl10/ReCogDrive-VLM-2B) with `internvl_agent`. This path decodes the eight waypoints as text using LMDeploy. Use the native trajectory prompt and a compatible LMDeploy/Transformers installation for comparisons.

```bash
torchrun --standalone --nproc-per-node=1 --module navsim.planning.script.run_pdm_score_internvl \
  train_test_split=navtest agent=internvl_agent agent.cam_type=single agent.prompt_type=base \
  agent.checkpoint_path=/path/to/ReCogDrive-VLM-2B metric_cache_path=/path/to/navtest_metric_cache \
  experiment_name=internvl_stage1 worker=sequential
```

Report the backend and precision alongside the score. The Stage 1 LMDeploy entry uses BF16. The diffusion-planner checkpoints and scores use the separate Stage 2/3 evaluation command below.

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

If the caching job receives SIGHUP, run it in a job allocation or a persistent session and retry with the same split, VLM and cache directory. Add `force_cache_computation=False` to skip complete feature/target pairs. The writer publishes a file only after the gzip stream is complete. If an interrupted run with older code left a nonempty but unreadable gzip, rebuild the affected scenes in a fresh cache directory.

### Step 2: Configure and run training

Configure the script `training/run_recogdrive_train_multi_node_2b.sh` and then start training:

```bash
sh training/run_recogdrive_train_multi_node_2b.sh
```

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

### Check inputs and visualize one NAVSIM scene

From the repository root, set the standard NAVSIM dataset and experiment environment variables and pick a token from a score CSV or the `navtest` scene filter. Use the same VLM family and size as in training. The published ReCogDrive planner checkpoints use `agent.dit_type=small`. For an 8B VLM, also set `agent.vlm_size=large`.

```bash
TOKEN=your_scene_token
EVAL_ARGS=(
  train_test_split=navtest
  agent=recogdrive_agent agent.cam_type=single agent.cache_hidden_state=false
  agent.vlm_path=/path/to/ReCogDrive-VLM-2B
  agent.checkpoint_path=/path/to/ReCogDrive-2B-IL.ckpt
  metric_cache_path=/path/to/navtest_metric_cache
  experiment_name=recogdrive_one_scene worker=sequential
  "train_test_split.scene_filter.tokens=[$TOKEN]"
)

python -m navsim.planning.script.run_pdm_score_recogdrive "${EVAL_ARGS[@]}" +preflight=check
torchrun --standalone --nproc-per-node=1 --module navsim.planning.script.run_pdm_score_recogdrive \
  "${EVAL_ARGS[@]}" visualize=true
```

The CPU check lists logs outside the chosen split separately from missing logs. It reports missing scenes, front-camera files, and metric-cache entries. It compares the checkpoint's saved Hydra model and camera settings when available. Use `preflight.checkpoint_config=/path/to/config.yaml` if you moved the weights. Confirmed missing inputs or mismatches produce a nonzero exit.

The score command regenerates features from images. To audit a feature cache for the same scenes, add `preflight.feature_cache_path=/path/to/feature_cache` to the check. The report inventories feature/target pairs and samples one hidden-state payload. Pass the original build config with `preflight.feature_cache_config=/path/to/config.yaml` if it is stored outside the cache. Older artifacts without saved configs are reported as having unknown model and view provenance. Supply only trusted logs, feature caches and checkpoints.

The scorer saves the exact trajectory it scored as a plot beside the front camera image under `<output_dir>/vis/<token>.png`. If you know the log name, add `"train_test_split.scene_filter.log_names=[$LOG]"` with `LOG` set to the filename stem. Omit the token filter to check the full configured split.





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

Before running, modify the script parameters as needed  according to your hardware and training requirements. This command will start RL training immediately after configuration.


### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh
```
This will evaluate your trained agent using **PDM scores** on the navtest.



## Benchmark and deployment notes

- **Bench2Drive.** The ReCogDrive Bench2Drive training and closed-loop evaluation pipeline remains unreleased. ReCogDrive Table 2 records the original result over 220 routes. For a reference implementation, see the UniDriveVLA training configurations and CARLA evaluation commands. [UniDriveVLA Bench2Drive guide](https://github.com/xiaomi-research/unidrivevla/blob/main/docs/train_eval_bench2drive.md) [ReCogDrive paper, Section 5](https://arxiv.org/html/2506.08052v2#S5)
- **DriveBench.** The values in Table 5 match the clean-input columns of Appendix Table 7. LLaVA-1.5 and InternVL2 refer to the 7B and 8B DriveBench baseline rows. The DriveLM column has its own GPT-Score. The DriveBench evaluator exposes accuracy and language metrics as well as an optional GPT judge. See the [DriveBench reproduction notes](../vqa_evaluation/DriveBench/README.md).
- **Text guidance.** The released diffusion planner takes the VLM's final hidden states directly. The text-guidance experiments in Appendix C describe research ablations. The public planner has no runtime switch for conditioning on a generated chain of thought. Use the Stage 1 agent for the text-waypoint baseline. [Paper, Appendix C](https://arxiv.org/html/2506.08052v2#A3)
- **NAVSIM reward.** The released DiffGRPO reward uses NAVSIM 1.1 PDMS. Set `agent.grpo_bc_coeff` in Hydra or `GRPO_BC_COEFF` for the RL scripts to tune the reference-policy imitation term. The released code keeps its existing default of `0.1`. The NAVSIM 2.0 adjacent-frame Extended Comfort metric needs trajectories from neighboring frames. A larger imitation weight can constrain policy changes, so evaluate EC separately with the official scorer. [NAVSIM discussion](https://github.com/xiaomi-research/recogdrive/issues/74#issuecomment-4189302283)
- **Deployment.** The supported open-source scoring path uses PyTorch with a BF16 VLM. The front-camera demo can serve as a reference when evaluating a deployment backend. This release has no validated ONNX export, INT8 recipe or end-to-end TensorRT engine for the VLM and planner.
