# ReCogDrive 最终路线图（七点）

这台只验通 step、正确；大训你去集群，同一脚本只改路径和卡数。

**没有要再问的了。先不开工。**

```
4 GRPO 公式通  →  2/3 接同一套 RL
1 DeepSpeed 2 卡通  →  6 才能上 Muon+DS
5 用现成 navtrain / navtest
7 Qwen 与 1 的 InternVL 烟测并列，不挡 2–6
```

---

## 已锁死

| 项 | 定案 |
|---|---|
| 输入 | 路径 cache + 在线 VLM；不打 hidden cache |
| 配方 | 不减 tile、不改 2800 / 分辨率 / prompt；VLM 不改精度 |
| 注意力 | 强制 SDPA（3090 上 FA 更慢） |
| DeepSpeed | 这台 **2 卡**，ZeRO-1，钉 **0.15.4**；关 optimizer CPU offload |
| GRPO | `π_old`=采样 logprob 缓存；`π_ref`=冻 IL 做 BC；**μ=4 + clip**；planner 只回 loss |
| PDM / Ray | 同 token 一次仿真 + LRU；Ray 先量，4 token 默认仍 batch；不上 HTTP reward |
| 2.0 数据 | **navtrain 训、navtest 评**。two-stage / navhard 今晚不做 |
| Waymo | **今晚不验** |
| 目录 | 本仓两个树：`navsim1.1/`（PDMS）和 `navsim2.0/`（官方 EPDMS + 同一套 agents）。`PYTHONPATH` 只指其中一个 |
| Muon | 必须和 DS 一起通；装不上只记阻塞 |
| 分数 | 1.1 PDMS 与 2.0 EPDMS 不合表 |
| Qwen | `vlm_type=qwen`；骨干 **Qwen3-VL-2B**；权重用 **UniDriveVLA-Base Stage1**（VLM 预训练），不是 Stage3 整网 |

路径（这台）：

```
code       = /data/yongkangli/recogdrive
  navsim1.1/   # 1.1 + ReCogDrive
  navsim2.0/   # 官方 2.0 + 同一套 agents
OPENSCENE  = /data/yongkangli/NAVSIM/dataset
  navsim_logs/{trainval,test}  → shengzhenli navtrain / navtest
  sensor_blobs/{trainval,test} → 同上
  maps → /data/shengzhenli/navtrain/maps
InternVL   = /data/yongkangli/ckpts/InternVL3-2B
Qwen VLM   = UniDriveVLA-Base Stage1
             (owl10/UniDriveVLA_Nusc_Base_Stage1；按 config 认 Qwen3，不按目录名)
path_cache = /data/yongkangli/NAVSIM/exp/recogdrive_agent_cache_internvl3_no_hidden_state
metric_1.1 = /data/shengzhenli/DrivoR-main/exp/metric_cache
python_1.1 = /data/yongkangli/miniconda3/envs/navsim/bin/python
```

warmup 已在 `dataset/warmup_two_stage`，今晚不用。navhard 盘不够（`/data` ~47G / 100%），不下。

---

## 1. 训练速度：IL / RL · DeepSpeed · PDM · Ray

已做：IL SDPA **715ms**；RL 同 token PDM+LRU **411ms**，分数差 0。

今晚验：IL DS 2 卡 4 step；脚本去掉 `BLOCKING=1` / offload / micro-batch 错位；Ray 有空再量。

集群：改 `GPUS`，不再挖 3090 SDPA。

---

## 2. DDPM × waypoint / delta

代码已有，和 sampler 正交。

今晚验：各 IL 2 step；核 DDPM-GRPO 时间步，不对齐先修。

集群：各一条完整 IL，再短 RL。

---

## 3. Flow + Flow-GRPO

IL / SDE / logprob 已有。Flow-GRPO 走第 4 点同一套。默认 `sde`。

今晚验：Flow IL 2 step；Flow-GRPO 2 step、μ=4。

集群：Flow IL 一条，再 Flow-GRPO。

---

## 4. GRPO：π_old / π_ref

| 角色 | 实现 |
|---|---|
| `π` | 可训 head |
| `π_old` | 采样时 logprob 缓存 |
| `π_ref` | 冻 IL，BC（`bc=0.1`，`kl=0`） |
| clip / μ | `eps=0.2`；trainer 连 step；yaml 默认 1，脚本盖成 4 |

今晚验：1.1 RL DS 2 卡 2 step。第 2 inner `ratio` 离开 1，且不再跑 VLM/PDM。

集群：先 DDIM waypoint RL（要有 IL ckpt），再接到 2/3。

---

## 5. NAVSIM 1.1 / 2.0 / Waymo

```
NAVSIM_TREE=1.1   # PYTHONPATH=$REPO/navsim1.1
NAVSIM_TREE=2.0   # PYTHONPATH=$REPO/navsim2.0
```

脚本：`run_recogdrive_internvl3_navsim20_{il,rl}.sh`（`NAVSIM_TREE=2.0`）。数据仍是 navtrain / navtest。2.0 RL/评需要 **2.0 metric cache**。分数不合表。Waymo 不验。

集群：1.1 大训；2.0 正式 navtrain/navtest；Waymo 以后挂 1.1。

---

## 6. 统一 Muon + DeepSpeed

还没做。1 没通则 6 不开。

今晚验：Muon+DS IL 2 step。不通只记原因，不用 AdamW 顶替。

---

## 7. QwenVL，预训练用 UniDriveVLA

**你要的：** 同一套 planner 能挂 Qwen；VLM 初始化用 UniDriveVLA 那份，不是裸的官方 Qwen。

**已有：** `vlm_type=qwen`、`run_recogdrive_qwenvl3_*.sh`、hidden size 可从 config 推断。

**还做：**

1. 按 `config.json` 的 `model_type` / `architectures` 认 **Qwen3-VL**。目录名没有 `qwen3`（例如 `UniDriveVLA_Nusc_Base_Stage1`）也必须走 Qwen3，不能掉进 Qwen2.5。
2. 权重：**UniDriveVLA-Base Stage1**（VLM 预训练）。Stage3 是整网 VLA，不当 backbone。
3. `vlm_hidden_size` 从该 ckpt 读（Qwen3-VL-2B 一般是 2048，不是 InternVL 的 1536）。
4. 缺 transformers Qwen3 / `qwen_vl_utils` 时，用 UniDriveVLA 的 `qwenvl3/transformers_replace`，不另造一份。
5. 路径 cache 与 InternVL **分开**，不能混 hidden 维。

今晚验：Qwen + UniDriveVLA Stage1，IL 2 step，loss 非 NaN。RL 等 InternVL 那条 T3 通了再各 2 step（可选，不挡 2–6）。

集群：Qwen 正式 IL/RL 各一条，配方与 InternVL 对齐，只换 VLM。

---

## 今晚顺序（失败停）

| # | 点 | 动作 | 通过 |
|---|---|---|---|
| T1 | 1 | DS import；2 张空卡 | 0.15.4；不抢卡 |
| T2 | 1 | 1.1 InternVL IL DS 2 卡 4 step | 两卡在训，非 NaN |
| T3 | 4 | 1.1 InternVL RL DS 2 卡 2 step μ=4 | ratio/clip；inner2 r≠1 |
| T4 | 2 | DDPM waypoint / delta 各 2 step | 非 NaN，维对 |
| T5 | 3 | Flow IL + Flow-GRPO 各 2 step | 非 NaN |
| T6 | 6 | Muon+DS IL 2 step | 非 NaN 或记阻塞 |
| T7 | 5 | `navsim2.0` **navtrain** IL 2 step | 非 NaN |
| T8 | 5 | 2.0 EPDMS **navtest 1 条** | 出分 |
| T9 | 5 | 2.0 RL 2 step，EPDMS reward | 非 NaN |
| T10 | 7 | Qwen + UniDriveVLA Stage1 IL 2 step | 认成 Qwen3；非 NaN |
| T11 | 1 | Ray（有空） | 分数差 0；不快就扔 |

---

## 后续计划（已定，先不实现）

2026-09 调研结论，完整对比见画布 `dataset-benchmark-survey`。都复用现有样本约定和适配层接口，模型不改。

| 项 | 做什么 | 依赖与约束 | 验收 |
|---|---|---|---|
| PhysicalAI-AV 训练 | 转换器：7 路 mp4 抽 2 Hz 关键帧（decord）存 JPEG，egomotion（约 100 Hz）重采样出 4 帧历史和 0.5 s 目标，7 路相机映射到规范视角；注册成数据源，可进 mixture | 需签 NVIDIA AV 数据许可、禁止再分发衍生数据；全量 TB 级，服务器盘只够子集 | 子集训练 loss 正常；与 nuScenes / WOD 混训跑通 |
| PhysicalAI-AV 评测 | 开环：抽一部分片段做 L2 / ADE；闭环走下面 AlpaSim 的 PAI-AV 赛道 | 同上 | 进程内评测出分 |
| AlpaSim 闭环 | 第三种适配层（闭环在线推理）：gRPC `EgodriverService`，按 egomotion 维护 0.5 s 历史，路线推导航指令，图像走内存，打成容器；nuPlan 赛道（navtest 的 MTGS 重建）优先 | 每次 Drive 调用 0.1 s 内、显存 16 GiB、镜像 40 GiB、断网；2026 挑战赛榜单 10/31 关闭，先实测 2B VLM + DiT 的单步延迟 | 本地用 NuRec 样例场景跑通一段闭环；容器过官方校验 |
| HUGSIM 闭环 | 复用闭环适配层，接 HUGSIM 的推理服务协议（KITTI-360 / Waymo / nuScenes / PandaSet 的 70+ 序列、400+ 场景） | 3DGS 资产体积与渲染卡 | 跑通若干场景并出 HUGSIM 驾驶分 |
| NeuroNCAP 闭环 | 复用闭环适配层，接 NeuroNCAP 的推理服务（nuScenes 安全场景：静止、正面、侧面） | NeuRAD 渲染环境；模型需 nuScenes 多视角输入 | 出 NeuroNCAP 分数和碰撞率 |

顺序：AlpaSim（有时间窗）→ 闭环适配层稳定后接 HUGSIM、NeuroNCAP → PhysicalAI-AV 子集训练与评测。

2026-10-02 状态：五项代码已接入 update-cache（`recogdrive/closedloop/{alpasim,hugsim,neuroncap}.py`、`recogdrive/data/physicalai.py`、`evaluator.name=physicalai`、`docker/alpasim/Dockerfile`）。已验证：三个协议按各自官方客户端格式用 nuScenes mini 回放打通（模板策略、CPU），PhysicalAI-AV 用官方布局的合成 clip 跑通转换、数据源和指标。未做：真模型闭环（缺空卡）、真仿真器出分（NuRec / HUGSIM 3DGS / NeuRAD + nuScenes trainval 资产）、PhysicalAI-AV 真数据（等许可 token）、AlpaSim 0.1 s 单步延迟实测、Dockerfile 构建（2027 无 docker，2031 根分区满）。

---

## 你集群

1. 多卡 DS 正式 IL/RL（InternVL）
2. DDPM waypoint / delta 完整 IL，再短 RL
3. Flow IL，再 Flow-GRPO
4. DDIM waypoint RL μ=4（先有 IL ckpt）
5. 1.1 大训；2.0 正式 navtrain/navtest；Waymo 以后
6. Muon+DS 短对照（T6 通了才上）
7. Qwen + UniDriveVLA Stage1 正式 IL/RL
