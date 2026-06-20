# -*- coding: utf-8 -*-
import os
import math
import json
import logging
from pathlib import Path
from datetime import timedelta
from typing import Tuple, Dict, List, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn.functional as F

import torch
import torch.nn.utils.rnn as rnn_utils
from torch import Tensor
from torch.utils.data import DataLoader

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
    DeepSpeedPlugin,
)
import shutil
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, WaymoE2ECacheOnlyDataset

from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

logger = get_logger("recogdrive_trainer")
LOG_LEVEL = "INFO"
logger.setLevel(LOG_LEVEL)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"





def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Any]]:
    """
    支持两种 feature 形态：
      1) cache_hidden_state=True（online）:
           features[i] 包含:
             - history_trajectory: (Th, 3)
             - high_command_one_hot: (3,)
             - status_feature: (D,)
             - last_hidden_state: (Ti, H)
      2) cache_hidden_state=False（offline）:
           features[i] 包含:
             - history_trajectory: (Th, 3)
             - high_command_one_hot: (3,)
             - status_feature: (D,)
             - image_path_tensor: (L_i,)  # 路径字符编码

    targets[i]:
      - trajectory: (L, 3)

    tokens_list[i]:
      - GRPO 用的 token（可以是任意对象）
    """
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack(
        [f["history_trajectory"] for f in features_list], dim=0
    ).cpu()
    high_command_one_hot = torch.stack(
        [f["high_command_one_hot"] for f in features_list], dim=0
    ).cpu()
    status_feature = torch.stack(
        [f["status_feature"] for f in features_list], dim=0
    ).cpu()

    f0 = features_list[0]

    features: Dict[str, torch.Tensor] = {
        "history_trajectory": history_trajectory,
        "high_command_one_hot": high_command_one_hot,
        "status_feature": status_feature,
    }

    if "last_hidden_state" in f0:
        # ---------- 模式 1：online（已有 hidden state，直接 pad） ----------
        last_hidden_state = rnn_utils.pad_sequence(
            [f["last_hidden_state"] for f in features_list],
            batch_first=True,
            padding_value=0.0,
        ).clone().detach()   # 防止误反传到 backbone
        features["last_hidden_state"] = last_hidden_state.cpu()

    elif "image_path_tensor" in f0:
        # ---------- 模式 2：offline（只缓存路径，后面再算 hidden state） ----------
        # 每个 sample: image_path_tensor: (Li,)
        # pad 成 (B, L_max)，pad 值必须是 0，方便 _decode_paths_from_tensor 以 0 截断
        image_path_tensor = rnn_utils.pad_sequence(
            [f["image_path_tensor"] for f in features_list],
            batch_first=True,
            padding_value=0,   # long 型 0
        )
        features["image_path_tensor"] = image_path_tensor.cpu()
    else:
        raise KeyError(
            "features must contain either 'last_hidden_state' or 'image_path_tensor'. "
            f"Got keys: {list(f0.keys())}"
        )

    # 目标：一如既往
    trajectory = torch.stack(
        [t["trajectory"] for t in targets_list], dim=0
    ).cpu()
    targets = {"trajectory": trajectory}

    return features, targets, list(tokens_list)

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """完全照你原来的 build_datasets 实现。"""
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [
            log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs
        ]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data




class State:
    seed: int = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None

    train_epochs: int = None
    train_steps: int = None
    num_updates_per_epochs: int = None

    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None

    output_dir: str = None




class ReCogDriveTrainer:
    """
    结构对齐你给的 UnifiedTrainer：
      - _init_distributed / _init_logging / _init_directories
      - prepare_dataset / prepare_val_dataset / prepare_models
      - prepare_trainable_parameters / prepare_for_training / prepare_trackers
      - train()
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.state = State()

        self.agent: AbstractAgent = instantiate(cfg.agent)

        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: LRScheduler | None = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        if self.state.accelerator.is_main_process:
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self.save_folder = self.cfg.output_dir


    def _init_distributed(self):
        """
        初始化 Accelerate (+ 可选 DeepSpeed)。
        DeepSpeed 的 config 完全使用 cfg.deepspeed，不再在代码里加额外字段。
        """
        logging_dir = Path(self.cfg.output_dir, "./logging_dir")
        project_config = ProjectConfiguration(
            project_dir=self.cfg.output_dir,
            logging_dir=logging_dir,
        )

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
            static_graph=True,
        )

        nccl_timeout = int(getattr(self.cfg, "nccl_timeout", 1800))
        init_pg_kwargs = InitProcessGroupKwargs(
            backend="nccl",
            timeout=timedelta(seconds=nccl_timeout),
        )

        if torch.cuda.is_available()  and self.cfg.deepspeed.bf16.enabled == True:
            mixed_precision = "bf16"
        else:
            mixed_precision = "fp16" 
        print("mixed_precision",mixed_precision)
        grad_accum = int(getattr(self.cfg, "gradient_accumulation_steps", 1))

        ds_plugin = None
        ds_cfg = None
        if getattr(self.cfg, "use_deepspeed", False):
            raw_cfg = getattr(self.cfg, "deepspeed", None)
            if raw_cfg is None:
                raise ValueError("use_deepspeed=True 但 cfg.deepspeed 为空，请在 config 里补上 deepspeed 配置。")

            ds_cfg = OmegaConf.to_container(raw_cfg, resolve=True) if hasattr(OmegaConf, "to_container") else dict(raw_cfg)

            ds_cfg.setdefault(
                "train_micro_batch_size_per_gpu",
                int(self.cfg.dataloader.params.batch_size),
            )
            ds_cfg.setdefault("gradient_accumulation_steps", grad_accum)

            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=ds_cfg,
                gradient_accumulation_steps=grad_accum,
            )

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=grad_accum,
            mixed_precision=mixed_precision,
            log_with=None,
            kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        self.state.accelerator = accelerator

        if ds_cfg is not None and accelerator.is_main_process:
            logger.info("Using DeepSpeed with config: %s", json.dumps(ds_cfg, indent=2))

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        # 9) seed
        if getattr(self.cfg, "seed", None) is not None:
            from accelerate.utils import set_seed

            self.state.seed = self.cfg.seed
            set_seed(self.cfg.seed)

        logger.info("Initialized Accelerator with mixed_precision=%s", mixed_precision)
        logger.info(accelerator.state, main_process_only=False)

    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        logger.info("Initialized ReCogDriveTrainer")
        logger.info(self.state.accelerator.state, main_process_only=False)

    def _init_directories(self):
        if self.state.accelerator.is_main_process:
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.cfg.output_dir


    def prepare_dataset(self):
        logger.info("Building ReCogDrive training dataset")
        agent = self.agent

        if getattr(self.cfg, "use_cache_without_dataset", False):
            logger.info("Using CacheOnlyDataset for training")
            assert not self.cfg.force_cache_computation, \
                "force_cache_computation must be False when using cached data"
            assert self.cfg.cache_path is not None, \
                "cache_path must be provided when using cached data"
            if getattr(self.cfg, "waymoe2e", False):
                self.train_dataset = WaymoE2ECacheOnlyDataset(
                    cache_path=self.cfg.cache_path,
                    split=getattr(self.cfg, "waymoe2e_train_split", "training"),
                    target_builders=agent.get_target_builders(),
                )
            else:
                self.train_dataset = CacheOnlyDataset(
                    cache_path=self.cfg.cache_path,
                    feature_builders=agent.get_feature_builders(),
                    target_builders=agent.get_target_builders(),
                    log_names=self.cfg.train_logs,
                )
        else:
            logger.info("Building SceneLoader for training")
            train_data, _ = build_datasets(self.cfg, agent)
            self.train_dataset = train_data

        self.train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=custom_collate_fn,
            shuffle=True,
            **self.cfg.dataloader.params,
        )
        logger.info(f"Train samples: {len(self.train_dataset)}")

    def prepare_val_dataset(self):
        logger.info("Building ReCogDrive validation dataset")
        agent = self.agent

        if getattr(self.cfg, "use_cache_without_dataset", False):
            logger.info("Using CacheOnlyDataset for validation")
            assert not self.cfg.force_cache_computation, \
                "force_cache_computation must be False when using cached data"
            assert self.cfg.cache_path is not None, \
                "cache_path must be provided when using cached data"
            if getattr(self.cfg, "waymoe2e", False):
                self.val_dataset = WaymoE2ECacheOnlyDataset(
                    cache_path=self.cfg.cache_path,
                    split=getattr(self.cfg, "waymoe2e_val_split", "val"),
                    target_builders=agent.get_target_builders(),
                )
            else:
                self.val_dataset = CacheOnlyDataset(
                    cache_path=self.cfg.cache_path,
                    feature_builders=agent.get_feature_builders(),
                    target_builders=agent.get_target_builders(),
                    log_names=self.cfg.val_logs,
                )
        else:
            logger.info("Building SceneLoader for validation")
            _, val_data = build_datasets(self.cfg, agent)
            self.val_dataset = val_data

        self.val_dataloader = DataLoader(
            self.val_dataset,
            collate_fn=custom_collate_fn,
            shuffle=False,
            **self.cfg.dataloader.params,
        )
        logger.info(f"Val samples: {len(self.val_dataset)}")

    # ---------------- 模型 / 优化器 ----------------

    def prepare_models(self):
        logger.info("Initializing agent models / loading checkpoints")
        if hasattr(self.agent, "initialize"):
            self.agent.initialize()

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters / optimizer / scheduler")
        accelerator = self.state.accelerator

        trainable_params = []
        for name, p in self.agent.named_parameters():
            if p.requires_grad:
                trainable_params.append((name, p))

        self.state.num_trainable_parameters = sum(p.numel() for _, p in trainable_params)
        logger.info(f"Trainable params: {self.state.num_trainable_parameters}")

        if accelerator.is_main_process:
            def _format_param(name: str, p: torch.nn.Parameter) -> str:
                try:
                    shape_str = str(tuple(p.shape))
                except Exception:
                    shape_str = "<no-shape>"
                try:
                    n = p.numel()
                except Exception:
                    n = 0
                return f"{name:<80} | shape={shape_str:<22} | numel={n:>12d} | requires_grad={p.requires_grad}"

            os.makedirs(self.save_folder, exist_ok=True)
            out_txt = os.path.join(self.save_folder, "trainable_params.txt")
            with open(out_txt, "w") as f:
                for n, p in trainable_params:
                    f.write(_format_param(n, p) + "\n")
            logger.info(f"Wrote detailed parameter list to {out_txt}")

        optim_cfg = self.agent.get_optimizers()
        if isinstance(optim_cfg, dict):
            self.optimizer = optim_cfg["optimizer"]
            self.lr_scheduler = optim_cfg.get("lr_scheduler", None)
        else:
            self.optimizer = optim_cfg
            self.lr_scheduler = None

        if len(self.optimizer.param_groups) > 0:
            self.state.learning_rate = self.optimizer.param_groups[0].get("lr", None)

        dataset_size = len(self.train_dataset)
        per_device_bs = self.cfg.dataloader.params.batch_size
        world_size = self.state.accelerator.num_processes
        grad_accum = int(getattr(self.cfg, "gradient_accumulation_steps", 1))
        num_upd_per_epoch = math.ceil(dataset_size / (per_device_bs * world_size * grad_accum))
        self.state.num_updates_per_epochs = num_upd_per_epoch

        trainer_params = getattr(self.cfg, "trainer", {}).get("params", {})
        if isinstance(trainer_params, DictConfig):
            trainer_params = OmegaConf.to_container(trainer_params, resolve=True)
        self.state.train_epochs = int(trainer_params.get("max_epochs", 10))

        logger.info(f"Num updates / epoch       = {self.state.num_updates_per_epochs}")
        logger.info(f"Train epochs              = {self.state.train_epochs}")

    def prepare_for_training(self):
        """
        跟你原来的 UnifiedTrainer 一样，用 accelerator.prepare 包起来，
        支持多卡多机。
        """
        logger.info("Wrapping models / optimizers / dataloaders with Accelerator")
        accel = self.state.accelerator

        if self.lr_scheduler is not None:
            (self.agent,
             self.optimizer,
             self.train_dataloader,
             self.val_dataloader,
             self.lr_scheduler) = accel.prepare(
                self.agent,
                self.optimizer,
                self.train_dataloader,
                self.val_dataloader,
                self.lr_scheduler,
            )
        else:
            (self.agent,
             self.optimizer,
             self.train_dataloader,
             self.val_dataloader) = accel.prepare(
                self.agent,
                self.optimizer,
                self.train_dataloader,
                self.val_dataloader,
            )

    def prepare_trackers(self):
        logger.info("Initializing trackers")
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        self.state.accelerator.init_trackers("recogdrive_train", config=cfg_dict)


    def train(self):
        logger.info("Starting training")

        accel = self.state.accelerator
        use_grpo = bool(getattr(self.agent, "grpo", False))

        per_device_bs = self.cfg.dataloader.params.batch_size
        world_size = accel.num_processes
        grad_accum = int(getattr(self.cfg, "gradient_accumulation_steps", 1))
        global_bs = per_device_bs * world_size * grad_accum

        logger.info(
            f"Effective Global Batch Size = {global_bs} "
            f"(= {per_device_bs} per_device x {world_size} world x {grad_accum} grad_accum)"
        )

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(self.state.train_epochs):
            self.agent.train()
            running_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(self.train_dataloader):
                features, targets, tokens_list = batch

                with accel.accumulate(self.agent):
                    if use_grpo:

                        preds = self.agent.forward(features, targets, tokens_list)
                        loss_bundle = preds
                        loss = loss_bundle.loss
                        reward = loss_bundle.reward
                        policy_loss = loss_bundle.policy_loss
                        bc_loss = loss_bundle.bc_loss
                    else:
                        preds = self.agent.forward(features, targets, tokens_list)
                        loss = preds.loss
                        reward = policy_loss = bc_loss = None

                    accel.backward(loss)
                    if accel.sync_gradients and accel.distributed_type != DistributedType.DEEPSPEED:
                        pass

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                loss_det = accel.reduce(loss.detach(), reduction="mean")
                running_loss += loss_det.item()
                num_steps += 1

                if accel.sync_gradients:
                    global_step += 1

                logs = {"loss": loss_det.item()}
                if self.state.learning_rate is not None:
                    logs["lr"] = self.optimizer.param_groups[0].get("lr", None)

                if use_grpo:
                    logs["reward"] = float(accel.reduce(reward.detach(), reduction="mean")) if reward is not None else 0.0
                    logs["policy_loss"] = float(accel.reduce(policy_loss.detach(), reduction="mean")) if policy_loss is not None else 0.0
                    logs["bc_loss"] = float(accel.reduce(bc_loss.detach(), reduction="mean")) if bc_loss is not None else 0.0

                if (global_step % 10 == 0) or (step == 0):
                    accel.log(logs, step=global_step)
                    if accel.is_main_process:
                        msg = f"[epoch {epoch+1} step {global_step}] loss={logs['loss']:.4f}"
                        if "lr" in logs and logs["lr"] is not None:
                            msg += f", lr={logs['lr']:.6g}"
                        if use_grpo:
                            msg += (
                                f", reward={logs['reward']:.4f}, "
                                f"policy={logs['policy_loss']:.4f}, "
                                f"bc={logs['bc_loss']:.4f}"
                            )
                        logger.info(msg)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.validate(use_grpo=use_grpo, global_step=global_step)

            if accel.is_main_process:
                avg_train_loss = running_loss / max(num_steps, 1)
                logger.info(
                    f"Epoch {epoch+1}/{self.state.train_epochs} finished. "
                    f"train/loss_epoch={avg_train_loss:.4f}, val/loss_epoch={val_loss:.4f}"
                )

                last_ckpt = os.path.join(self.save_folder, "last.ckpt")
                self._save_checkpoint(last_ckpt)

                last_vlm_dir = self._vlm_dir_for_ckpt_path(last_ckpt)
                self._save_vlm_safetensors(last_vlm_dir)

                self._update_best_checkpoints(val_loss=val_loss, epoch=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(
                        f"New best val loss {val_loss:.6f}, updating best small ckpt "
                        f"and best VLM snapshot."
                    )

                    best_path = os.path.join(self.save_folder, "best.ckpt")
                    self._save_checkpoint(best_path)

                    best_vlm_dir = self._vlm_dir_for_ckpt_path(best_path)
                    self._save_vlm_safetensors(best_vlm_dir)

        accel.end_training()

    @staticmethod
    def _vlm_dir_for_ckpt_path(ckpt_path: str) -> str:
        base, _ = os.path.splitext(ckpt_path)
        return base + "_vlm"

    def _save_checkpoint(self, path: str):
        """
        保存当前 agent 的『小 ckpt』：
        - 只包含 action_head 等导航部分
        - 显式排除 VLM/backbone（避免把几 GB 的大模型存进去）
        """
        accel = self.state.accelerator
        unwrapped_agent = accel.unwrap_model(self.agent)

        full_sd = unwrapped_agent.state_dict()

        filtered_sd = {
            k: v
            for k, v in full_sd.items()
            if not k.startswith("backbone.")
            and not k.startswith("model.")
        }

        ckpt = {"state_dict": filtered_sd}
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved to {path} (params={len(filtered_sd)})")

    def _update_best_checkpoints(self, val_loss: float, epoch: int, k: int = 5):
        """
        维护 val_loss 最小的 K 个小 ckpt：
          - 文件名: epoch_{epoch}_val_{val_loss}.ckpt
          - 信息记录在 best_checkpoints.json
          - 多余的旧 ckpt 会被删除（连同对应的 VLM 目录）
        """
        ckpt_name = f"epoch_{epoch+1:04d}_val_{val_loss:.6f}.ckpt"
        ckpt_path = os.path.join(self.save_folder, ckpt_name)

        self._save_checkpoint(ckpt_path)

        vlm_dir = self._vlm_dir_for_ckpt_path(ckpt_path)
        self._save_vlm_safetensors(vlm_dir)

        meta_path = os.path.join(self.save_folder, "best_checkpoints.json")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    best_list = json.load(f)
            except Exception:
                best_list = []
        else:
            best_list = []

        best_list.append({"path": ckpt_name, "val_loss": float(val_loss)})

        best_list.sort(key=lambda x: x["val_loss"])
        best_list = best_list[:k]

        keep_names = {item["path"] for item in best_list}

        for fname in os.listdir(self.save_folder):
            if not fname.startswith("epoch_") or not fname.endswith(".ckpt"):
                continue
            if fname not in keep_names:
                try:
                    os.remove(os.path.join(self.save_folder, fname))
                    logger.info(f"Removed old checkpoint: {fname}")
                except OSError:
                    pass

                obsolete_vlm_dir = self._vlm_dir_for_ckpt_path(
                    os.path.join(self.save_folder, fname)
                )
                if os.path.isdir(obsolete_vlm_dir):
                    shutil.rmtree(obsolete_vlm_dir, ignore_errors=True)
                    logger.info(f"Removed old VLM snapshot: {obsolete_vlm_dir}")

        with open(meta_path, "w") as f:
            json.dump(best_list, f, indent=2)

        logger.info(f"Top-{k} checkpoints updated: {best_list}")
    
    def _save_vlm_safetensors(self, out_dir: str):
        """
        通用版：把当前 agent.backbone.model 存到指定 out_dir（HF + safetensors）。
        - 会处理 shared tensor（全部 clone 到 CPU）
        - 只在 train_backbone=True 且 backbone 存在时生效
        """
        accel = self.state.accelerator
        unwrapped_agent = accel.unwrap_model(self.agent)

        if not getattr(unwrapped_agent, "train_backbone", False):
            return

        backbone = getattr(unwrapped_agent, "backbone", None)
        if backbone is None or not hasattr(backbone, "model"):
            logger.warning("train_backbone=True 但 agent.backbone 或 backbone.model 为空，跳过保存 VLM。")
            return

        model = backbone.model

        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Saving finetuned VLM (HF format, safetensors) to: {out_dir}")

        with torch.no_grad():
            raw_state_dict = model.state_dict()
            cloned_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in raw_state_dict.items()
            }

        model.save_pretrained(
            out_dir,
            state_dict=cloned_state_dict,
            safe_serialization=True,
            max_shard_size="10GB",
        )

        if getattr(backbone, "tokenizer", None) is not None:
            backbone.tokenizer.save_pretrained(out_dir)

        logger.info(f"Finetuned VLM snapshot saved to {out_dir}.")

    def _save_vlm_safetensors_if_needed(self):
        """
        老接口保留：保存一个「默认」的 vlm_finetuned 目录。
        （如果你后面不再用这个名字，也可以不用它）
        """
        vlm_out_dir = os.path.join(self.save_folder, "vlm_finetuned")
        self._save_vlm_safetensors(vlm_out_dir)


    @torch.inference_mode()
    def validate(self, use_grpo: bool, global_step: int) -> float:
        accel = self.state.accelerator
        self.agent.eval()

        total_loss = 0.0
        total_steps = 0

        for batch in self.val_dataloader:
            features, targets, tokens_list = batch

            if use_grpo:
                preds = self.agent.forward(features, targets)
            else:
                preds = self.agent.forward(features, targets, tokens_list)

            loss = self.agent.compute_loss(features, targets, preds)

            loss_det = accel.reduce(loss.detach(), reduction="mean")
            total_loss += loss_det.item()
            total_steps += 1

        mean_val_loss = total_loss / max(total_steps, 1)
        if accel.is_main_process:
            logger.info(f"[step {global_step}] val/loss_epoch={mean_val_loss:.4f}")
        accel.log({"val/loss_epoch": mean_val_loss}, step=global_step)
        return mean_val_loss




@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = ReCogDriveTrainer(cfg)
    trainer.prepare_dataset()
    trainer.prepare_val_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    trainer.prepare_for_training()
    trainer.prepare_trackers()
    trainer.train()


if __name__ == "__main__":
    main()
