# ReCogDrive 改造与验收

本轮以 `update-cache / 3743d73` 为起点。原 NAVSIM 1.1 配方以 `main / d544047` 为行为基线；新增能力通过显式配置启用。训练与 NAVSIM 评测现在都使用 `recogdrive/models/recogdrive` 和 `recogdrive/adapters/navsim` 的同一实现，两个 NAVSIM 树保留旧导入路径作为兼容入口。

## 安装与默认配方

```bash
pip install -e .                       # NAVSIM 1.1 + shared ReCogDrive
NAVSIM_TREE=2.0 pip install -e .       # NAVSIM 2.0 + shared ReCogDrive
pip install -e '.[lora,vqa]'           # optional PEFT training and lmdeploy VQA evaluation
pip install -e '.[test]'               # regression and wheel checks in a fresh environment
```

原配方默认 `agent.grpo_objective=main`、`agent.grpo_num_iterations=1`、`agent.lr_schedule=main`。GRPO 使用原来的直接 log-probability 目标及 BC；IL 学习率按原 200 epoch 配方（3 epoch warmup，最低 1e-6），RL 按原 10 epoch 配方，每 epoch 调度一次。`agent.grpo_objective=ppo` 明确切换为 clipped ratio，可开启多轮 rollout 更新和 KL；`agent.lr_schedule=step` 明确切换为按实际 optimizer steps 调度。Flow 的随机 GRPO rollout 是新增能力，不能声明与 main 的历史 Flow RL 等价。

数据来源保持独立注册：NAVSIM、nuScenes、WOD-E2E、PhysicalAI-AV、VQA。planning 来源可按原 `mixture` 权重混合；非缓存 NAVSIM 经 adapter 补齐 token。混合数据不会静默丢弃不一致的样本列。预测时域由 `trajectory_sampling` 决定，缓存标签不足时直接报错。

main 默认同时保留 `agent.cache_hidden_state=True` 和末尾不满 batch 的样本；采样器接收配置中的 seed。通用训练入口默认读取 hidden-state cache，`CACHE_HIDDEN_STATE=False` 显式选择在线 VLM 和 image-path cache；VQA launcher 与 WOD launcher 已显式选择在线模式。可训练 VLM 必须使用在线输入。缓存训练中的新场景 NAVSIM 评估由独立在线模型执行；在线训练则使用当前模型生成 replay，结束时不再重复加载第二个模型。

## 支持范围

下表区分实现路径、已跑证据与仍需真实数据或官方服务的验证。最新整体验收记录见 [Repo Completion](Repo_Completion.md)。

| 路径 | 代码能力 | 已验证 | 仍需真实任务验收 |
| --- | --- | --- | --- |
| NAVSIM 1.1 / 2.0 | IL、NAVSIM PDM GRPO、各树官方 PDMS / EPDMS 入口 | 每树 118 项测试通过、0 跳过；mini 模板策略训练/导出/推理各有 59 个验证 token | 完整 ReCogDrive 训练、NAVSIM 1.1 navtest 和 2.0 两阶段正式成绩 |
| nuScenes | IL、混合 planning、开环 L2 / collision | mini 数据模板策略训练/导出/推理；GPU smoke 读取 mini CAM_FRONT | 全量 trainval 模型训练与 benchmark |
| WOD-E2E | IL、混合 planning、显式 logged-target ADE 或带独立训练标签的逐帧 RFS GRPO、开环 RFS/ADE | reward、split/来源校验及 launcher 路径纳入定向测试；细节见 [Waymo RL](Waymo_RL.md) | 真实 TFRecord 训练、Waymo 官方场景聚合成绩 |
| PhysicalAI-AV | 数据转换、IL、开环 L2/ADE/FDE | 合成视频/egomotion 转换和协议回归 | 真实授权数据训练及完整指标 |
| VQA | JSONL answer-only loss；共享 VLM 与 planning IL/GRPO 交替更新；LoRA 导出 | mixed-batch、answer mask、rank 路由和 GRPO 新入口定向回归；真实 Qwen GPU IL 混合 smoke 是此前单独一轮记录 | 真实 VQA+GRPO 数据训练和 lmdeploy 基准题集评分 |
| NeuroNCAP / HUGSIM / AlpaSim | policy 服务、训练后及独立 checkpoint 评估编排、失败传播、结果解析与清理 | fake 外部进程覆盖就绪、超时、失败、清理和有限结果读取；见 [闭环评估](Closedloop_Evaluation.md) | 真实 NeuRAD / 3DGS / NuRec 资产与官方闭环成绩 |

默认 GRPO 奖励读取 NAVSIM metric cache，其 planning 来源必须是 NAVSIM；同任务批次下可以插入 VQA 答案监督。显式 Waymo GRPO 的 planning 来源必须与奖励使用同一训练 cache/split；ADE 读取公开训练未来轨迹，RFS 读取额外提供的训练反馈标签。nuScenes/PhysicalAI 暂无内置 GRPO 奖励，不能混入这些 planning rollout。WOD rater labels 不能替代 NAVSIM PDM cache。nuScenes、PhysicalAI、WOD 和 VQA loader 当前接受 `train` / `val`；未支持的 `test` split 明确报错。

Waymo 配方与数据边界见 [Waymo RL](Waymo_RL.md)；三个外部模拟器的评估入口见 [闭环评估](Closedloop_Evaluation.md)。这些入口会记录失败并使请求的评估返回失败，不把协议检查当作官方出分。VQA 与 NAVSIM PDM planning 的 IL/GRPO 交替训练命令见 [Train_Eval](Train_Eval.md#vqa--planning-交替批次)。

## VQA / planning 交替批次

使用同一个 VLM，按任务权重交替抽取同任务 global batch；同一步所有 rank 使用同一任务，planning 批次仍可混合多个 planning 数据源。VQA 只监督 assistant 的答案 token，prompt、图像 token 和 padding 不参与语言 loss；token 长度按 batch 动态左补齐。

标准 VQA JSONL 每行例如：

```json
{"id":"scene-001","images":["front.jpg"],"question":"What should the ego vehicle do?","answer":"Slow down and yield."}
```

图片路径相对 `vqa.image_root`，未设置时相对各 split 的 JSONL 目录。`images` 支持多图；也接受 `image`、`image_path`、`image_paths`。训练/验证分别设置文件，避免把训练答案用于验证。

```bash
VLM_PATH=/path/to/internvl3 \
CACHE_PATH=/path/to/navsim_image_path_cache \
VQA_TRAIN_JSONL=/path/to/vqa/train.jsonl \
VQA_VAL_JSONL=/path/to/vqa/val.jsonl \
GPUS=4 bash scripts/train/run_recogdrive_vqa_planning_il.sh
```

此新配方默认 LoRA rank 16、DDP fp32、unused-parameter 检测以及 step scheduler。`PLANNING_WEIGHT`、`VQA_WEIGHT` 控制任务比例，`VQA_MAX_LENGTH` 默认 8704，仅作为上限。可用 `MODEL_FAMILY=qwenvl3` 选择 Qwen-VL；预设为 Qwen3-VL 选择 Qwen3VLTextDecoderLayer、InternVL3-2B 选择 Qwen2DecoderLayer；其他 checkpoint 可用 ACTIVATION_CHECKPOINT_LAYER 或 Hydra 参数覆盖。若采用全量微调，显式传 `agent.lora_rank=0 agent.train_backbone=true`。普通纯 planning 配方的默认 FSDP2 设置保持不变。

LoRA VQA 评测在独立 CPU 子进程加载 checkpoint 中的 adapter、合并到 base VLM 后 HF 导出，再调用原 DriveLM/LingoQA/DriveBench evaluator，不修改训练模型。该临时导出会占用完整 VLM 的 CPU 内存与磁盘；真实 lmdeploy 模型验证仍需可用评测 GPU及基准数据。

## 训练状态

验证分别记录 `val/task/planning/loss` 和 `val/task/vqa/loss`：planning 按有效样本数加权，VQA 按监督答案 token 数加权。混合任务的 `val/loss` 按 mixture 来源权重合并后的任务权重聚合，用于 best/top-k 选择；不同 loss 的量纲仍需由使用者设置任务权重。各 rank 的尾部占位 forward 只维持执行同步，不计入验证统计。`perf/samples_per_s` 计实际输入样本，GRPO 多次复用 rollout 不增加它；`perf/optimizer_steps_s` 单独报告更新频率。

推理 checkpoint 保留普通 backbone、外部投影层和 LoRA 权重；全量 HF VLM 的 side export 同时保存 processor/tokenizer。冻结 teacher 的 alias 与 buffer 只保留在训练恢复状态中。新 checkpoint 的 `vlm_export` 标记决定是否使用配套目录：禁止把旧目录误当当前冻结/LoRA基座，全量导出目录缺失时明确报错。旧 checkpoint 继续兼容 `_vlm` 目录约定。周期评测提交独立快照后再执行 top-k 清理。

异步 checkpoint 只有所有 rank 的 DCP 写入与元数据保存均成功后才发布原子完成标记。前台安全边界进行成功确认，回调线程不会与评测共享 Gloo collective。恢复记录 batch 游标及 rank RNG；梯度累积尾部完成剩余更新。GRPO 只在完整 rollout 和 clean accumulation 边界停下/保存，因此 `max_steps` 可能向该边界取整。当前不会保存 persistent DataLoader worker 的内部随机状态；开启随机 worker 增强时跨重启不保证逐样本 bitwise 相同，默认关闭增强的 main 配方不受该限制。

## 验证

本机导出未修改的 main 源码，再把代码快照和该参考目录一并传至新的服务器测试目录。GPU 数值和混合训练的复跑方式见 [GPU 验证记录](GPU_Validation.md)。

```bash
python scripts/verify/export_main_reference.py /path/to/main_reference
```

服务器在选定 NAVSIM 环境中：

```bash
export CUDA_VISIBLE_DEVICES='' TORCH_COMPILE_DISABLE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/navsim1.1:$PWD:/path/to/alpasim_grpc"
export MAIN_REFERENCE_ROOT=/path/to/main_reference
python -m pytest -q tests
```

测试涵盖 main 默认 DDIM/waypoint 的参数、IL loss、推理轨迹；GRPO 公式及梯度、epoch scheduler；token/混合采样、VQA answer mask/路由/导出；checkpoint 失败和恢复以及双 rank Gloo。main 模型测试必须设置参考目录，不能把 skip 当作兼容性通过。CPU eager 回归不验证 CUDA 编译、真实 VLM 训练或实际闭环仿真分数。

NAVSIM 1.1 PDMS 与 2.0 EPDMS 分开报告。当前训练 reward 保留 main 自定义权重，2.0 reward 仍是单场景训练信号；它不能等同官方两阶段 EPDMS 聚合或 human penalty filtering。正式评测继续使用各树的官方入口。

### 可复跑的收尾验收

在已有 `navsim` 环境和数据环境变量下，从仓库根目录执行：

```bash
python scripts/verify/run_cpu_checks.py \
  --main-reference-root "$PWD/_main_reference" \
  --nuscenes-root /data/yongkangli/datasets/nuscenes \
  --pydeps /data/yongkangli/rcdtest/pydeps \
  --output-dir "$PWD/exp/cpu_acceptance_$(date +%Y%m%d_%H%M%S)"
```

该入口顺序运行 NAVSIM 1.1/2.0 全量测试、nuScenes mini 的模板策略训练/验证、checkpoint 导出与推理，核对全部验证 token 和有限的 8×3 轨迹。任何跳过的测试会使验收失败；每次使用新的输出目录，保存日志、JUnit 和 `acceptance.json`。这是框架集成验收，模板策略成绩不代表 ReCogDrive 模型质量。

空闲卡上的 planner 对照命令：

```bash
CUDA_VISIBLE_DEVICES=0 MAIN_REFERENCE_ROOT="$PWD/_main_reference" \
python scripts/verify/benchmark_planner.py \
  --device cuda --mode both --precision bf16 --compile-mode none \
  --batch-size 4 --tokens 8 --warmup 10 --iterations 50 \
  --output "$PWD/exp/planner_bf16.json"
```

脚本核对 pinned main 源码 SHA，拒绝在仍有计算任务的卡上运行；main/current 依次加载相同参数，比较 loss、梯度和轨迹，并记录延迟中位数/p95、吞吐、峰值显存。`--compile-mode default` 可做编译对照。该测量只有合成 VLM features 下的 planner，不覆盖真实 VLM、数据加载、多卡通信或驾驶分数；端到端验收应同时保存训练的 `metrics.csv` 和正式 benchmark 输出。

## 性能参考与后续服务器验收

[LoongForge](https://github.com/baidu-baige/LoongForge) 和 [LingBot-VLA](https://github.com/Robbyant/lingbot-vla) 的 FSDP2、分块编译、选择性 activation checkpoint、prefetch 等模式已在当前框架中使用。VQA 变长批次和同任务跨 rank 采样针对本次混合训练补齐；不会把参考仓库的新 action loss 或 Muon 自动换成 main 默认配方。

LoongForge 的 [per-microbatch CUDA Graph](https://baidu-baige.github.io/LoongForge/blog/2026-06-loongforge-groot-n16-acceleration.html) 是可选后续实验：先验证固定形状 planner 子图的输出/loss/梯度，再按相同数据、精度、global batch 和更新预算测 throughput、step latency 与峰值显存。参考仓库的硬件和任务不同，不能拿其吞吐数字声明本项目已最优。

后续仍需用真实资产执行完整 InternVL/Qwen IL、RL、VQA/planning 训练；复测 NAVSIM 指标、nuScenes/Waymo/PhysicalAI 开环指标，运行 NeuroNCAP/HUGSIM/AlpaSim 真闭环。此前未回收服务线程的 NeuroNCAP checker 曾在完成断言后退出 SIGSEGV。新的正式 CPU 检查显式 shutdown/server_close/join，在新快照中完整 HTTP 往返及进程退出均通过；这不代表真实 NeuRAD 仿真出分。旧服务器目录是局部上传副本，不能作为整个分支通过的证据；每轮使用新目录和文件 hash 清单。


2026-10-03 实际验证：2027 CPU pytest 已通过25项（在NAVSIM 1.1和2.0各执行一次）（含双rank DDP/Gloo、pinned main数值对照）；真实本地 Qwen3-VL2B+LoRA4 的VQA与planning forward/backward均通过，VQA loss=0.230431且196个LoRA B梯度非零，planning loss=1.341359且60个planner梯度非零。首次CPU smoke在可选合并阶段达到300秒限时；后续独立的真实CPU export-only验证已成功完成LoRA合并和HF导出。safetensors不含LoRA/wrapper键，config/processor可重载，抽查权重符合base+delta的BF16舍入。当前环境的HF保存会经Accelerate误触DeepSpeed/MPS探测并崩溃；导出器对已确认未包装的独立CPU模型局部跳过该无关检测并恢复函数，不更改依赖。lmdeploy真实评测仍待评测GPU与该可选依赖。

## 本轮收尾结果（2026-10-03）

工程验收通过：2027 完整独立快照在 NAVSIM 1.1/2.0 各执行 67 项，零失败、零跳过。覆盖实际 pinned-main GRPO 网络/梯度、真实 GRPO Agent checkpoint 导出→普通 actor 重载、双 rank DDP/DCP、VQA、混合验证、三个模拟协议和 PhysicalAI 合成转换。两树模板策略的 nuScenes mini 训练/验证/导出/推理完整跑通，各 59 个验证 token、8×3 有限轨迹；两种 wheel 构建及 DriveBench 资源检查通过。GPU 轮次修复 FSDP 参数 dtype 后，NAVSIM 1.1/2.0 完整测试又各通过 68 项。

服务器测试目录：`/data/yongkangli/rcdtest/refactor_closeout_20261003_1df674d2`，使用 `snapshot_final.tar.gz` 与 `SNAPSHOT_MANIFEST.json`；早期 `snapshot.tar.gz` 仅用于保留排错过程。验收日志在 `exp/cpu_acceptance_verified`。本地原始证据在 `.codex-tasks/20261002-recogdrive-refactor/raw`，main 对照见 [Main_Compatibility](Main_Compatibility.md)。用户已有 ROADMAP 未修改；本轮没有 commit/push。

GPU smoke 已完成：planner eager/compile 的 FP32/BF16 对照和真实 Qwen3-VL-2B 的 DDP、FSDP、activation checkpointing、compile、保存/重载均有记录，详见 [GPU 验证记录](GPU_Validation.md)。本轮资源检查时 2027 的 GPU 0/1 空闲，GPU 2/3 仍由其他任务占用；没有中止他人进程，2031 未用于本轮测试。完整 NAVSIM 训练、lmdeploy 生成、全量数据指标和真实模拟器分数仍待正式验收，不能据此声明性能全局最优。

## 工程收尾复验（2026-10-04）

本轮补齐 Waymo 显式 stage-3 reward、VQA/planning GRPO 交替入口、闭环 evaluator 错误传播和资源清理等缺口。最终 2027 独立副本在 NAVSIM 1.1/2.0 各跑 118 项，0 失败、0 跳过；每棵树另以 nuScenes mini 模板策略完成训练、checkpoint 导出与重载推理，分别处理 59 个验证 token。新增定向回归 50 项通过。逐项结果、边界、运行命令和证据位置见 [Repo Completion](Repo_Completion.md)。

2026-10-04 planner compile GPU 对照的 FP32/BF16 数值检查均通过。此次仍是合成特征下的 planner-only 测量，不是完整 VLM 吞吐或数据集成绩。真实 Waymo TFRecord、VQA+GRPO 训练、lmdeploy 基准评分和 NeuroNCAP/HUGSIM/AlpaSim 官方闭环成绩尚未执行；因此这份验收说明仓库框架路径可交付，不能据此声称全部基准性能最优。
