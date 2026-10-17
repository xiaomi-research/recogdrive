# ReCogDrive 原始需求与专项终版验收

本文件逐项对应用户最初七项计划。实现、受控测试、真实资产实验分别记录；“代码已实现”不表示所有数据集正式成绩已经验收。唯一执行项目为当前 `recogdrive-1` / `update-cache`，未发布、commit或push，未修改用户 `ROADMAP.md` 或 UniDriveVLA。

## 源码版本与证据规则

- main 数学基线：`d54404796de7a44ca418b96057e3f8c3de3e8c0d`。
- Git HEAD：`3743d7384bc6e066f509a734acecc4f4454878e0`；大量工作区改动未提交，因此 HEAD 不能单独标识验收版本。
- 前轮冻结版：641-file archive SHA `c2d8896d9ca90272f174aaabdb3eff25ae34b56b76a80e64fbc293950687879d`；其两树各118项、50定向及GPU结果属于该版。
- 本轮完整源版本、逐文件 SHA、命令、运行状态和范围，以 `.codex-tasks/20261004-recogdrive-final/raw/{SOURCE_MANIFEST.json,EVIDENCE_LEDGER.json,ACCEPTANCE_FINAL.json}` 为准。最终文档更新后检查执行/config文件是否仍与本轮运行快照一致。
- 旧 67/68/118 项结果均不得代替新闭环计划改动的回归。语法检查不代替子进程失败、超时、SIGTERM、奖励梯度或配置继承检查。

## 最初七项需求逐项对应

| 原始要求 | 现有实现 | 主要文件 / 入口 | 验收结论与范围 |
| --- | --- | --- | --- |
| 任务1：模型、训练、数据、pipeline、评估解耦；多数据集混训评估 | canonical模型/adapter、loader和evaluator注册、MixtureDataset、按同任务global batch采样、统一Trainer | `recogdrive/models/recogdrive`、`adapters`、`data/registry.py`、`data/mixture.py`、`train/trainer.py`、`eval/registry.py` | 已实现并有CPU接口/混合采样/导出回归；各真实数据集混训收敛和正式指标未运行 |
| 任务2：参考LoongForge、LingBot-VLA，性能最优 | FSDP2/DDP、分块compile、activation checkpoint、prefetch、异步DCP、梯度累积、显式Muon/step LR | `distributed/parallelize.py`、`checkpoint.py`、`data/prefetch.py`、`train/args.py` | 优化能力已实现；planner有限GPU对照通过。端到端最优性未验证，BF16保main边界的配置没有加速，不能标为“全局最优” |
| 任务3：NAVSIM/nuScenes/WaymoE2E/PhysicalAI训练评估；NeuroNCAP/HUGSIM/AlpaSim | 数据转换/loader、IL、开环指标、NAVSIM PDM/官方评估入口、Waymo显式GRPO奖励、闭环策略服务与仿真/聚合命令计划 | `data/{navsim,nuscenes,waymoe2e,physicalai}.py`、`eval/{navsim,openloop,closedloop}.py`、`closedloop/*` | 支持的代码路径已接入；公开Waymo训练默认ADE，RFS有标签条件；官方资产/全量训练/闭环出分与Waymo类别聚合见未运行清单 |
| 任务4：repo优化规整 | 单一模型来源、两个NAVSIM树旧导入重导出、package/资源/配置/CLI整理、可追踪tests | `setup.py`、两树`setup.py`和兼容模块、`scripts/*`、`tests/test_package_layout.py` | 工程与wheel资源回归覆盖；当前改动仍待用户代码审阅/另行授权提交 |
| 任务5：其他特性，确保后续无需相关优化 | rank RNG/游标恢复、尾部累积flush、EMA、teacher过滤、HF/LoRA导出、top-k评估快照、失败传播与自有进程清理、显式扩展配置 | `distributed/checkpoint.py`、`train/trainer.py`、`infer.py`、`eval/registry.py` | 明确功能已实现并回归；“未来永久不需要任何优化”不是可证明的验收标准。persistent worker随机增强跨重启bitwise状态未保存 |
| 任务6：VLA支持VQA/planning混合迭代 | 共享VLM；VQA仅答案监督且每批一次LM更新；planning使用IL或GRPO/PPO，重复rollout只作用planning；DDP unused与来源guard | `data/vqa.py`、`models/recogdrive/utils/vqa_preprocess.py`、`adapters/navsim/agent.py`、`scripts/train/run_recogdrive_vqa_planning_{il,rl}.sh` | 新CPU实际adapter/小模型梯度和路由回归；旧真实Qwen GPU证据只覆盖当时IL版本，当前真实VQA+GRPO GPU训练未运行 |
| 任务7：训练评估逻辑与main一致 | 默认NAVSIM 4s/8-waypoint DDIM、IL/直接GRPO+BC、main epoch LR及hidden-state cache；扩展显式启用；pinned源码哈希对照 | `tests/test_main_model_compat.py`、`test_main_grpo_compat.py`、`test_recipe_compat.py`、`benchmark_planner.py` | 已验证默认配置范围的loss/梯度/轨迹/LR；不是所有新增算法和benchmark与main等价。1.1 PDMS不能与2.0 EPDMS直接比较 |

## 专项实现与功能证据

| 专项 | 实现和检查 | 支持边界 |
| --- | --- | --- |
| Waymo强化训练/奖励 | `WaymoADEReward`只读训练`targets.gz`，5s负平均XY距离；`WaymoRFSReward`只读独立训练`eval.gz`，复用逐帧RFS；路径/finite/horizon/同cache-split/非空guard；真实reward→GRPO优势→backward受控检查 | ADE是研究代理奖励，不是官方RFS。公开train只有future logs，公开val反馈不用于训练；RFS需要额外训练标签。官方cluster average缺元数据/评分规则或服务，未伪实现 |
| 完整闭环仿真与评分编排 | `simulator_command`保留单任务；`simulator_commands`串行多个场景；每任务要求独立新鲜`episode_result_files`；独立`aggregate_command`调用官方评分程序；明确超时、阶段日志和`closedloop_run_status.json` | 程序只编排、读取官方产物，不自创全benchmark平均公式。真实NeuRAD/3DGS/NuRec/官方服务部署未运行；受控进程测试不是官方分数 |
| 配置继承 | 保留完整agent target、LoRA、horizon、归一化、VLM选项；只强制必要infer状态；HUGSIM传递原Python/devkit/config，CUDA可见索引映射保持选择 | 外部模型checkpoint训练配置需要用户提供并匹配；旧checkpoint未内嵌完整训练配置，不宣称自动推断所有模型参数 |
| 评估失败传播 | 场景退出非零/缺产物/旧产物/重复产物/聚合退出或超时/非有限指标均失败；standalone CLI非零；训练在PG释放、全部最终评测与monitor清理后抛出请求评估失败 | 评测未配置时不会自动产生正式分数；官方聚合器自身必须执行其有效性/完整性检查 |
| 进程清理 | 为自有runner建立进程组；TERM→grace→kill/reap；训练/standalone中断取消；受控实际OS SIGTERM验证真实评估子进程消失 | 只处理自己创建的OS进程组。官方Docker/远程服务的生命周期仍由官方部署/命令负责，不擅自停止外部容器或他人任务 |
| VQA/GRPO交替 | 每个global batch同任务；VQA不调用奖励或复用rollout；planning按mu更新；规划源符合PDM/Waymo奖励数据要求；step预算和恢复边界回归 | 新CPU小模型检查可证明控制/梯度路径，不能代替真实VLM多卡收敛与benchmark指标 |

## 变更与使用

训练入口和阶段配方见 [Train/Eval](Train_Eval.md)、[Waymo RL](Waymo_RL.md)。新闭环批次计划、独立checkpoint评估和官方聚合命令的配置见 [Closedloop Evaluation](Closedloop_Evaluation.md)。本轮只补闭环完整计划和专项验证，不改变默认NAVSIM模型/loss。

通过、历史失败已修复、当前失败、未运行清单分别见证据台账；每项含源码版本和检查范围。历史SIGSEGV/测试夹具/缺Hydra group及BF16偏差记录保留，不作为新代码失败或新代码通过的替代证明。

## 本轮实际检查结果与状态清单

本轮运行版归档 SHA 为 `0a1d276f4538cdb2ed007a2ef3e0aaac35916a5651d2df1593b8dd6b35106303`（643-file，未提交工作区）。最终文档补充后的交付归档以终版manifest/JSON为准；只有在全部执行/config文件哈希一致时，下面结果才覆盖交付代码。

| 证据ID | 状态 | 源码和检查范围 | 实际结果 |
| --- | --- | --- | --- |
| E_TARGETED_CURRENT | 通过 | 本轮运行版；Waymo合成奖励/梯度、VQA-GRPO小模型路由、实际子进程/CLI/OS信号/场景与聚合错误检查 | 61 passed，0 failed，0 skipped |
| E_CPU_CURRENT | 通过 | 同一运行版；两树全部pytest、真实nuScenes-mini模板策略fit/export/reload/infer | 1.1和2.0各129 passed，0 failed，0 skipped；各59个完整验证token |
| E_PLANNER_REUSED | 仅按依赖哈希复用组件证据 | c2d前轮GPU记录；7个planner源码文件逐一与本轮匹配 | FP32/BF16原阈值数值通过；不证明新闭环/GPU VQA-GRPO；BF16没有速度收益 |
| E_QWEN_IL_HISTORY | 仅历史 | 2026-10-03各run自己的source_hashes和参数 | 真Qwen IL GPU smoke已跑；当前trainer/GRPO/新增编排不能拿它代替运行 |
| F_NEW_STAGE_DETAILS | 历史失败，已修复 | 9fb运行版；CLI只传回笼统child exit 1 | 当时59通过1失败；修具体stage cause后61定向及129/tree通过，原日志保留 |
| F_OLD_SIGSEGV / F_OLD_HYDRA / F_OLD_BF16_NUMERIC | 历史失败，修复过程保留 | 各原始archive/component-file哈希 | 分别包含崩溃夹具/独立val缺失、缺必填Hydra split、BF16旧轨迹偏差；后续通过不等于已证明崩溃唯一原因 |

本轮当前未解决的失败检查为0；这不表示未运行的实验已通过。预期的子进程失败、超时、SIGTERM用例得到非零/清理结果后，测试本身才算通过。语法检查只是辅助检查，未用于替代功能验收。

未运行清单：真实Waymo TFRecord模型训练与val、独立训练RFS反馈、Waymo官方cluster average_score、当前真VLM VQA/GRPO多卡训练与收敛、官方NeuroNCAP/HUGSIM/AlpaSim资产部署/完整仿真/评分、完整NAVSIM/nuScenes/PhysicalAI模型指标、真实VQA/lmdeploy评分、长期端到端最佳性。原因和后续步骤按台账N_*条目逐项记录；GPU是否空闲不能替代缺少数据、资产、部署与官方规则的条件。

已知限制包括：Waymo ADE为显式代理奖励；逐帧RFS不等于官方类别分；HUGSIM全benchmark聚合须由所选benchmark release的正确程序/规则提供；旧checkpoint完整训练配置由调用方提供并匹配；随机persistent worker增强跨重启不保证bitwise；官方Docker/远程资源不由通用OS进程组清理擅自处理。没有发布、推送或删除他人文件/停止他人任务。

## 后续GPU与正式资产验收步骤

1. 先核对终版manifest及原始main哈希，确认所选卡无其他任务；使用既有`navsim`环境，保持数据划分、初始化、horizon、归一化、batch、精度、更新预算一致。
2. 运行 `scripts/verify/check_gpu_training.py --max-tokens 1024` 的当前源码Qwen IL DDP/FSDP/AC/compile保存重载检查。该脚本验证IL smoke，不提供GRPO参数。
3. 真实VQA/GRPO用 `MAX_STEPS=3 GPUS=2` 配合 `run_recogdrive_vqa_planning_rl.sh`，显式 `agent.grpo_sample_time=2`；提供匹配checkpoint、NAVSIM在线cache/PDM train cache、VQA train/val JSONL。检查两个任务真实LoRA梯度、更新次数、checkpoint恢复、有限轨迹，再扩大预算。
4. Waymo使用真实training TFRecords转换cache，先5s stage2，再显式ADE stage3；RFS只在独立训练反馈齐全时测试。记录每类reward、样本数、loss和真实val逐帧指标；官方类别聚合由官方元数据/服务验收。
5. 分别在官方NeuroNCAP/HUGSIM/AlpaSim环境准备合法资产、版本、全部场景清单和官方聚合argv；先单场景，再完整计划。检查所有期望产物、新鲜度、官方聚合退出码、mapped字段、完整性、失败后无自有残留进程。
6. 正式跑1.1 PDMS、2.0两阶段EPDMS、nuScenes/Waymo/PhysicalAI开环及VQA题集；单独保存官方分数。测稳定端到端吞吐/p95/显存，多次重复同预算对照后再判断优化收益。
