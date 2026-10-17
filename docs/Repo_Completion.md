# ReCogDrive 工程收尾验收（2026-10-04）

本页的118/tree结果绑定下述前轮源码。之后新增的多场景→官方聚合编排、实际CLI/OS信号检查与完整需求对照，以 [原始需求专项终版](Requirements_Acceptance_Final.md) 及 `20261004-recogdrive-final` 新台账为准；本页旧结果不覆盖后续改动。

## 结论

本轮代码框架收尾通过：NAVSIM 1.1/2.0 回归、mini 数据集端到端模板策略、Waymo 与混合任务入口、闭环评估生命周期及 planner GPU 对照均有可复核记录。NAVSIM 默认配方以 `main / d54404796de7a44ca418b96057e3f8c3de3e8c0d` 为基线；新能力由配置或独立 launcher 显式启用。

该结论验收代码路径和小规模验证，不代表完整 ReCogDrive 训练成绩、所有数据集正式分数或全局性能最优。下一阶段仍需使用真实训练数据、VQA 评测集和官方模拟器资产跑正式 benchmark。

## 已完成的工程路径

| 领域 | 收尾结果 | 证据边界 |
| --- | --- | --- |
| 共享模型与训练框架 | 模型、数据源、任务 pipeline、训练和 evaluator 解耦；NAVSIM 1.1/2.0 旧导入兼容到同一模型/adapter 实现；混合数据按任务和来源配置 | 两树完整 CPU 回归及 mini 模板策略集成运行通过 |
| main 行为兼容 | NAVSIM 默认 waypoint/DDIM、直接 GRPO 目标及 BC、默认学习率计划与缓存路径保持；PPO、Flow 和其他新目标需显式启用 | pinned main 数值与行为回归；不是对所有新数据集的算法等价声明 |
| VQA + planning | 一个可训练 VLM 按任务权重交替处理同任务 global batch；VQA 仅监督答案 token；planning 可走 IL 或 GRPO；launcher 关闭 hidden-state-only 路径以支持共享 VLM 更新 | 新路由、rank 同步和输入/奖励来源 guard 纳入新增定向回归；此前真实 Qwen GPU smoke 验证的是 IL 混合训练，未验证真实数据上的 VQA+GRPO |
| Waymo E2E stage 3 | 默认负 logged-target ADE 奖励；可选逐帧 RFS 需要独立提供训练 rater 标签；stage 2/3 校验 5 秒 horizon 和训练数据来源 | 奖励及 launcher 用合成/假数据覆盖；未运行真实 WOD TFRecords。逐帧 RFS 平均不等同官方场景聚合分 |
| NeuroNCAP / HUGSIM / AlpaSim | 训练后和独立 checkpoint 评估；读取训练 agent 配置；启动服务和模拟器，等待 readiness，收集有限官方指标；失败返回失败并清理本次子进程 | fake 外部进程与结果文件测试；没有真实 NeuRAD、3DGS、NuRec 运行或官方成绩 |
| nuScenes / PhysicalAI | 注册 IL、转换和开环评估适配器 | nuScenes mini 模板策略完成 fit/export/reload/infer；PhysicalAI 转换使用合成输入，未跑授权全量数据 |

## 自动化验收结果

- 2026-10-04 的新源码快照上，NAVSIM 1.1 和 2.0 各 **118 passed、0 skipped**。
- 两棵 NAVSIM 树分别完成 nuScenes mini 模板策略训练、checkpoint 导出和重载推理；每棵树验证预测 **59 个 token**，输出轨迹有限且形状正确。模板策略验证分数不代表 ReCogDrive 模型质量。
- 新增代码的定向回归 **50 项通过**，覆盖 Waymo reward / launcher 校验、VQA 与 GRPO 任务路由、模拟器 readiness/超时/失败传播/清理和结果读取。真实数据与外部模拟器仍须按上表单独验收。
- 2026-10-04 RTX 3090 compile planner 对照：batch 4、256 feature tokens、10 次预热、50 次迭代；main 和当前实现依次运行。

| 精度 | 阶段 | main 中位数 | 当前中位数 | 数值检查 |
| --- | --- | ---: | ---: | --- |
| FP32 | 推理 | 89.07 ms | 27.45 ms | 通过；loss 差 0，梯度最大差 `1.49e-8`，轨迹最大差 `1.91e-5` |
| FP32 | 训练 | 81.19 ms | 47.23 ms | 同一组 loss / 梯度 / 轨迹阈值通过 |
| BF16 | 推理 | 108.32 ms | 110.43 ms | 通过；loss、梯度、轨迹差均为 0 |
| BF16 | 训练 | 82.07 ms | 88.79 ms | 通过；loss、梯度、轨迹差均为 0 |

FP32 使用原有 `atol=2e-5, rtol=2e-4`；BF16 使用原有 `atol=0.01, rtol=0.01`，阈值未放宽。planner 输入是合成 VLM features，计时不含 VLM、数据读取、多卡通信或仿真。BF16 的兼容修复保留了 main 的数值边界，因此该配置没有加速。

2026-10-03 真实 Qwen3-VL-2B + nuScenes-mini 图像的 DDP/FSDP/activation-checkpointing/compile 与 checkpoint 导出重载 smoke 是另一轮历史证据，详情见 [GPU 验证记录](GPU_Validation.md)；它没有训练真实 VQA 题集或产生驾驶 benchmark 分数。

## 复跑入口

VQA 与 NAVSIM PDM planning 的混合 GRPO 使用同一个 VLM。准备与模型一致的 VLM 权重、在线图像 cache、NAVSIM planning 训练 metric cache、VQA train/val JSONL 和 stage 2 checkpoint：

```bash
CHECKPOINT=/path/to/stage2.ckpt \
METRIC_CACHE_PATH=/path/to/navtrain_metric_cache \
CACHE_PATH=/path/to/navsim_image_path_cache \
VLM_PATH=/path/to/matching_vlm \
VQA_TRAIN_JSONL=/path/to/vqa/train.jsonl \
VQA_VAL_JSONL=/path/to/vqa/val.jsonl \
GPUS=4 bash scripts/train/run_recogdrive_vqa_planning_rl.sh
```

`CACHE_PATH` 提供 NAVSIM 图像/样本输入，`METRIC_CACHE_PATH` 提供同一规划训练 split 的 PDM cache。`MODEL_FAMILY`、VLM、采样方法、horizon 和 LoRA 配置须与 checkpoint 匹配；调整任务比例用 `PLANNING_WEIGHT` 与 `VQA_WEIGHT`。Waymo 5 秒 stage-3 配方及显式 ADE/RFS 选择见 [Waymo RL](Waymo_RL.md)；独立 checkpoint 闭环评估见 [Closed-loop Evaluation](Closedloop_Evaluation.md)。

## 证据位置与未覆盖项

本地原始日志和 JSON 位于 `.codex-tasks/20261004-recogdrive-complete/raw/`，包括 `CPU_ACCEPTANCE.json`、`targeted_20261004_144622.log`、`GPU_PLANNER.json`、`PLANNER_fp32.json` 和 `PLANNER_bf16.json`。服务器测试副本位于 `/data/yongkangli/rcdtest/repo_complete_20261004_f5977f3c`；完整源文件数、最终归档哈希和逐文件校验以本轮最终 `SNAPSHOT_MANIFEST.json` 及 `source_*.tar.gz` 为准。

尚未覆盖：完整 NAVSIM/nuScenes/WOD/PhysicalAI 模型训练与正式分数、真实 VQA+GRPO 训练和 lmdeploy 评分、Waymo 官方场景聚合、NeuroNCAP/HUGSIM/AlpaSim 真仿真成绩，以及不同数据规模下的端到端性能最优性。验收范围到此可用于工程交付；上述结果需要各自真实资产和正式 benchmark 才能确认。
