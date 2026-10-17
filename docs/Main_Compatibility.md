# main 与 update-cache 的对照

基线是 `main / d54404796de7a44ca418b96057e3f8c3de3e8c0d`；本轮从 `update-cache / 3743d7384bc6e066f509a734acecc4f4454878e0` 开始。下表描述收尾后的代码。功能接入与实际数据集成绩分开验收，完整运行方式见 [Refactor_Iteration](Refactor_Iteration.md)。

| 部分 | main | 收尾后的 update-cache |
| --- | --- | --- |
| 模型来源 | `navsim/agents/recogdrive` | `recogdrive/models/recogdrive` 单一来源；两个 NAVSIM 树通过旧路径重导出 |
| 默认数据路径 | NAVSIM 缓存 VLM 特征，4 s / 8 个 waypoint | 默认同样使用 hidden-state cache；在线编码通过 `CACHE_HIDDEN_STATE=False` 显式选择 |
| 默认规划目标 | DDIM / waypoint 的 IL 与直接 GRPO log-probability 目标 | 保留默认数学目标；PPO、Flow、delta 是显式选项，不能混作 main 默认对照 |
| GRPO 更新 | 原直接 log-prob × advantage 目标及 BC | `grpo_objective=main`、1 次更新、KL=0；可训练 VLM 的梯度保留，PPO 内更新重算 conditioning |
| 学习率 | IL：3 epoch warmup / 200 epoch cosine；RL：10 epoch，每 epoch 调度 | `lr_schedule=main` 保留原数值与运算顺序；`step` 调度显式启用 |
| 训练框架 | Lightning | 独立 Trainer、DDP/FSDP2、梯度累积、EMA、异步恢复状态；更换框架不等于改变 loss |
| 数据与任务 | NAVSIM planning；独立 VQA 预训练/评测代码 | NAVSIM、nuScenes、WOD、PhysicalAI 与 JSONL VQA 注册；共享 VLM，按同任务批次交替训练 |
| 评估 | NAVSIM 1.1 PDMS 与独立 VQA evaluator | 1.1/2.0 官方评分入口分开；开环与闭环协议单独接入；checkpoint 的评测副本不受 top-k 清理影响 |
| 安装与导出 | NAVSIM 包与仓库内模型代码 | 两树 wheel 含 canonical 模型、Hydra 和 DriveBench 资源；普通 backbone 权重保留，HF export 含 processor/tokenizer |

## 对照证据与范围

`scripts/verify/export_main_reference.py` 导出未修改的基线源码，`benchmark_planner.py` 核对其 commit 和七个文件的 SHA。CPU 回归检查默认 small / 8-waypoint DDIM 的初始化参数、IL loss 和推理轨迹；带非零 modulation 的固定 GRPO rollout 使用真实 planner 网络与 log-probability 路径，对照 policy/BC/total loss、VLM 输入梯度与全部 actor 参数梯度。只替换评分输入和采样链，main 源码保持不变。学习率按 epoch 严格比较。

训练边界另有回归：默认保留尾 batch；seed 进入 Python/NumPy/Torch 和采样器；VQA 只监督答案；跨 rank 任务一致；验证按有效样本或答案 token 聚合；异步保存失败没有完成标记；恢复游标、累积尾部和评测 snapshot 均受检查。

2026-10-04 在 RTX 3090 复跑 compile planner 对照，使用合成输入、batch 4、256 feature tokens、10 次预热和 50 次迭代。FP32 推理中位数从 main 的 89.07 ms 降到当前 27.45 ms，训练从 81.19 ms 降到 47.23 ms；同一既有阈值 `atol=2e-5, rtol=2e-4` 下 loss 相同、梯度最大绝对差 `1.49e-8`、轨迹最大绝对差 `1.91e-5`，对照通过。BF16 继续保持 `atol=0.01, rtol=0.01`，loss、梯度和轨迹与 main 完全相同；当前推理 110.43 ms（main 108.32 ms），训练 88.79 ms（main 82.07 ms）。这些是 planner-only 中位数，不包含 VLM、数据读取、多卡通信或驾驶评分，不构成全局性能最优结论。此前真实 Qwen3-VL-2B 与 nuScenes-mini 图像的 DDP/FSDP/activation checkpointing/compile/导出重载 smoke 记录见 [GPU 验证记录](GPU_Validation.md)；其规划标签和 VQA JSONL 是 smoke 输入。随机 persistent worker 增强的内部 RNG 当前不在恢复状态中，跨重启不承诺逐样本 bitwise 一致。最新工程验收和原始数字见 [Repo Completion](Repo_Completion.md)。

NAVSIM 2.0 的 EPDMS 不能直接与 main 的 1.1 PDMS 比较；其训练 PDM 信号也不是官方两阶段聚合成绩。默认 GRPO 保留 NAVSIM PDM；新增 Waymo ADE / 条件 RFS 是显式奖励选择，不改变 NAVSIM 默认目标。VQA 批次使用答案监督，planning 批次使用原 GRPO/PPO 目标；PPO 的重复更新只用于 planning。Waymo 公开训练缺少 rater labels 时使用显式 logged-target ADE 配方；逐帧 RFS 平均不等于官方 scenario-cluster average_score，后者需官方元数据或评分服务。
