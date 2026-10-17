# GPU 验证记录（2026-10-03）

本页记录 2027 服务器上的 planner 数值/耗时对照，以及真实 Qwen3-VL 训练框架 smoke。环境为 Python 3.9.25、PyTorch 2.6.0+cu124、CUDA 12.4、RTX 3090。对照基线固定为 `main / d54404796de7a44ca418b96057e3f8c3de3e8c0d`。结果用于确认本轮改造的数值和运行路径，不代表全量训练或官方 benchmark 成绩。

## Planner 对照

两种实现使用相同初始化、合成 VLM features、batch 4、feature tokens 256、8-waypoint DDIM 小配置、AdamW（lr 1e-4、weight decay 0），10 次预热后计时 50 次。表中耗时为中位数；编译模式为 PyTorch 默认 compile。每次先跑 main，再跑当前实现。

| 模式 / 精度 | 推理中位数（main → 当前） | 训练中位数（main → 当前） | 数值对照 |
| --- | ---: | ---: | --- |
| eager / FP32 | 109.66 → 106.56 ms（1.03×） | 89.50 → 88.81 ms（1.01×） | 通过；loss/轨迹相同，梯度最大绝对差 2.38e-7 |
| eager / BF16 | 135.97 → 129.16 ms（1.05×） | 135.19 → 135.21 ms（约 1.00×） | 通过；loss、梯度、轨迹完全相同 |
| compile / FP32 | 87.09 → 26.29 ms（3.31×） | 73.93 → 52.07 ms（1.42×） | 通过；loss 相同，梯度最大绝对差 1.12e-8，轨迹最大绝对差 1.91e-5 |
| compile / BF16（修复后） | 103.59 → 106.94 ms（当前慢约 3%） | 84.26 → 74.64 ms（1.13×） | 通过；loss、梯度、轨迹完全相同 |

FP32 编译对照采用原有严格阈值 `atol=2e-5, rtol=2e-4`；BF16 沿用 `atol=0.01, rtol=0.01`，没有放宽阈值。编译 FP32 当前实现训练峰值 allocated/reserved 为 694.6/740 MiB，main 为 696.8/740 MiB；推理为 163.1/178 MiB 与 154.4/178 MiB。

初版 BF16 编译虽然更快，但 main 与当前实现的迭代轨迹张量最大绝对差为 0.302，超出既有检查阈值；模拟精度转换和关闭 TF32 也没有解决。定位后，`recogdrive/models/recogdrive/blocks/attention.py` 在 BF16 autocast 编译时保留 main 的 RoPE 数值边界，FP32 仍走已知序列长度的编译快路径。修复后 BF16 的 loss、全部梯度和轨迹完全一致，代价是推理编译加速不再成立。FSDP 的统一 master dtype 修正在 `recogdrive/distributed/parallelize.py`；新增 FSDP 精度回归通过，完整 CPU 测试在 NAVSIM 1.1 和 2.0 各 68 项通过。

## Qwen3-VL 混合训练 smoke

使用本地 Qwen3-VL-2B 权重、LoRA rank 4 和真实 nuScenes-mini CAM_FRONT 图像（缩放为 112×112）。规划特征/目标和单行 VQA JSONL 是合成输入；任务顺序为 planning → VQA → planning，规划 543 tokens、VQA 112 tokens。各配置运行 3 次 optimizer update，随后写出带完成标记的 DCP 恢复 checkpoint 并导出 `last.ckpt`，再通过新的 `infer.load_agent` 加载 checkpoint 推理，验证输出为有限的 1×8×3 轨迹。该 smoke 检查完成 checkpoint 可见性和推理重载，没有验证从 checkpoint 恢复 optimizer 状态后继续训练。**这验证了训练与导出路径，不是数据集训练结果或官方评分。**

| 策略 / 设置 | 结果 | 每 rank 峰值 allocated | 备注 |
| --- | --- | ---: | --- |
| DDP，2 rank，FP32 | 通过 | 6.79 GiB | frozen backbone 复制配置为 `replicate_frozen=["backbone"]` |
| FSDP，1 rank，BF16 | 通过 | 16.14 GiB | `replicate_frozen=[]` |
| FSDP，2 rank，BF16 | 通过 | 12.11 GiB | `replicate_frozen=[]` |
| FSDP，2 rank，BF16 + 28 层 activation checkpointing | 通过 | 12.11 GiB | 完成 3 次更新、导出和重载 |
| FSDP，1 rank，BF16 + 28 层 activation checkpointing + compile | 通过 | 16.14 GiB | 84.09 秒包含冷编译 |
| FSDP，2 rank，BF16 + 28 层 activation checkpointing + compile | 通过 | 12.11 GiB | 78.41 秒包含冷编译；日志有 graph break / fallback，不能当作稳定端到端吞吐 |

本轮最终测试源码快照位于 `/data/yongkangli/rcdtest/refactor_closeout_20261003_1df674d2/snapshot_gpu_final.tar.gz`；清单与 SHA-256 在同目录的 `GPU_FINAL_MANIFEST.json`。更早的 `snapshot_final.tar.gz` 和 `SNAPSHOT_MANIFEST.json` 是 GPU 测试前的工程收尾历史快照，不包含本轮 GPU 后续改动。本地原始 JSON 和完整日志保存在 `.codex-tasks/20261003-recogdrive-gpu/raw/`，服务器输出位于 `/data/yongkangli/rcdtest/refactor_closeout_20261003_1df674d2/exp/gpu_20261003/`。测试 harness 为 `scripts/verify/check_gpu_training.py`。该 smoke 不等于仓库默认配方已经切换为 BF16/FSDP；默认混合训练 preset 仍为 DDP FP32。

## 复跑方式

在服务器选定已有 `navsim` 环境、确认所选 GPU 空闲后，从代码副本根目录运行。示例路径对应本轮 2027 测试副本；改为其他目录时同步修改数据、地图、模型和输出路径。权重必须是本地可读文件，输出目录每次使用新路径。

```bash
conda activate navsim
cd /data/yongkangli/rcdtest/refactor_closeout_20261003_1df674d2
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export OPENSCENE_DATA_ROOT=/data/yongkangli/NAVSIM/dataset
export NUPLAN_MAPS_ROOT=/data/yongkangli/NAVSIM/dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="$PWD/navsim1.1:$PWD:/data/yongkangli/rcdtest/pydeps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
unset TORCH_COMPILE_DISABLE

# 先检查 GPU 0 和 1 空闲；此命令会使用这两张卡。
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/verify/check_gpu_training.py \
  --strategy ddp \
  --model-path /data/yongkangli/ckpts/Qwen3-VL-2B-Instruct \
  --nuscenes-root /data/yongkangli/datasets/nuscenes \
  --max-tokens 1024 \
  --output-dir /data/yongkangli/rcdtest/exp/recogdrive_gpu_ddp_YYYYMMDD_HHMMSS

# FSDP + activation checkpointing + compile 的复跑示例；同样先检查所选 GPU 空闲。
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/verify/check_gpu_training.py \
  --strategy fsdp --activation-checkpointing \
  --activation-checkpointing-class Qwen3VLTextDecoderLayer --compile \
  --model-path /data/yongkangli/ckpts/Qwen3-VL-2B-Instruct \
  --nuscenes-root /data/yongkangli/datasets/nuscenes \
  --max-tokens 1024 \
  --output-dir /data/yongkangli/rcdtest/exp/recogdrive_gpu_fsdp_compile_YYYYMMDD_HHMMSS
```

`--max-tokens 1024` 必须显式传入：该 harness 默认上限 512，短于本次 543-token planning 输入。planner 的单卡编译对照可用下面命令；先确认 GPU 0 空闲，或把两个 GPU 参数都改为另一张空闲卡：

```bash
CUDA_VISIBLE_DEVICES=0 MAIN_REFERENCE_ROOT="$PWD/_main_reference" \
python scripts/verify/benchmark_planner.py \
  --device cuda --mode both --precision fp32 --compile-mode default \
  --batch-size 4 --tokens 256 --warmup 10 --iterations 50 \
  --output /data/yongkangli/rcdtest/exp/planner_fp32_YYYYMMDD_HHMMSS.json
```

## 范围与未完成项

本轮最终资源检查时，2027 的 GPU 0/1 已释放，GPU 2/3 仍有其他任务；没有中止其他人的进程。2031 没有用于本轮运行。已通过的 GPU smoke 不覆盖完整 NAVSIM 训练、真实 VQA 题集、lmdeploy 生成、NAVSIM 官方 PDMS/EPDMS、nuScenes/Waymo/PhysicalAI 完整数据指标，也不覆盖 NeuroNCAP、HUGSIM、AlpaSim 的真实模拟器出分。现有结果不能据此声称性能全局最优；FP32 planner 编译性能提升只适用于上述合成输入与尺寸，部署结论需以目标数据、模型和评分入口复测。
