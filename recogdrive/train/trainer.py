"""Training loop for VLA agents. Parallelism, data pipelines and evaluators are plugged in."""

import contextlib
import gc
import inspect
import json
import logging
import math
import random
import signal
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from recogdrive.adapters import check_policy
from recogdrive.data import DevicePrefetcher, build_split, import_plugins, loader_name, make_loader
from recogdrive.distributed import DistributedContext, ResumeCheckpointer, export_model_state, gradient_sync, parallelize
from recogdrive.eval import build_evaluators
from recogdrive.train.args import TrainArgs
from recogdrive.train.ema import EMA
from recogdrive.train.monitor import Monitor

logger = logging.getLogger("recogdrive")

EXTRA_METRICS = ("reward", "policy_loss", "bc_loss", "clip_frac", "ratio_mean")
VLM_PREFIX = "backbone.model."
ADAPTER_MARK = ".lora_"  # peft adapter parameter names
EMA_SUFFIX = "-EMA"


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)


def _dataset_tasks(dataset):
    tasks = getattr(dataset, "sample_tasks", None)
    if tasks is not None:
        return set(tasks)
    children = getattr(dataset, "datasets", None)
    if children is not None:
        return set().union(*(_dataset_tasks(child) for child in children)) if children else set()
    child = getattr(dataset, "dataset", None)
    if child is not None and child is not dataset:
        return _dataset_tasks(child)
    return {str(getattr(dataset, "task", "planning"))}


def _dataset_leaves(dataset):
    from recogdrive.data.navsim import NavsimSamples

    if isinstance(dataset, NavsimSamples):
        return [dataset]
    children = getattr(dataset, "datasets", None)
    if children is not None:
        return [leaf for child in children for leaf in _dataset_leaves(child)]
    child = getattr(dataset, "dataset", None)
    if child is not None and child is not dataset:
        return _dataset_leaves(child)
    return [dataset]


def _batch_task(features) -> str:
    task = features.get("task", "planning")
    if isinstance(task, (list, tuple)):
        if not task:
            raise ValueError("task batch is empty")
        tasks = set(task)
        if len(tasks) != 1:
            raise ValueError(f"mixed task batch is unsupported: {sorted(tasks)}")
        task = task[0]
    if task not in ("planning", "vqa"):
        raise ValueError(f"unsupported batch task {task!r}; expected 'planning' or 'vqa'")
    return task


def _raise_sigterm(signum, frame) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def _batch_size(features, targets, tokens) -> int:
    if tokens is not None:
        return len(tokens)
    for values in (targets, features):
        for key, value in values.items():
            if key not in ("pixel_values", "image_grid_thw") and torch.is_tensor(value) and value.ndim:
                return int(value.shape[0])
        for value in values.values():
            if isinstance(value, (list, tuple)):
                return len(value)
    if "num_patches" in features:
        return len(features["num_patches"])
    return 1


def _uses_pdm_reward(reward) -> bool:
    if reward is None or not any(
        cls.__name__ == "PDMReward" and cls.__module__ == "recogdrive.adapters.navsim.reward"
        for cls in type(reward).__mro__
    ):
        return False
    from recogdrive.adapters.navsim.reward import PDMReward

    return isinstance(reward, PDMReward)


def _is_builtin_recogdrive_agent(agent) -> bool:
    return any(cls.__name__ == "ReCogDriveAgent" and cls.__module__ == "recogdrive.adapters.navsim.agent"
               for cls in type(agent).__mro__)


class Trainer:
    def __init__(self, cfg: DictConfig):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.cfg = cfg
        self.args = TrainArgs.from_cfg(cfg)
        self.ctx = DistributedContext(self.args.nccl_timeout)
        self.args.validate(self.ctx.world_size)
        _seed_everything(self.args.seed)
        import_plugins(cfg)
        self.agent = instantiate(cfg.agent)
        check_policy(self.agent)
        self.output_dir = Path(cfg.output_dir)
        if self.ctx.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = Monitor(cfg, self.output_dir, self.ctx.is_main)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_interval = "epoch"
        self.train_loader = None
        self.val_loader = None
        self.clip_params = []
        self.ema = None
        self.global_batch = 0
        self.checkpointer = ResumeCheckpointer(self.output_dir / "checkpoints", self.ctx, self.args.async_save)
        self.evaluators = build_evaluators(cfg, self.output_dir)
        self.evaluation_errors = []
        self.start_epoch = 0
        self.epoch = 0
        self.global_step = 0
        self.best_val = math.inf
        self.micro = 0
        self.start_batch = 0
        self.resume_rng = None
        self.resume_epoch_rng = None

    def fit(self) -> None:
        owns_signal_handler = threading.current_thread() is threading.main_thread()
        previous_sigterm = signal.getsignal(signal.SIGTERM) if owns_signal_handler else None
        if owns_signal_handler:
            signal.signal(signal.SIGTERM, _raise_sigterm)
        try:
            self.build_data()
            if hasattr(self.agent, "initialize"):
                self.agent.initialize()
            self.model = parallelize(self.agent, self.args, self.ctx)
            self.build_optimizer()
            if self.args.ema_decay > 0:
                # EMA scaling rule: the same averaging horizon in samples at any global batch size.
                decay = self.args.ema_decay ** (self.global_batch / self.args.ema_reference_batch)
                self.ema = EMA(self.agent, decay)
                if self.ctx.is_main:
                    logger.info("ema decay=%.6f (%s at global batch %d, running %d) over %d trainable tensors", decay,
                                self.args.ema_decay, self.args.ema_reference_batch, self.global_batch, len(self.ema.params))
            if self.args.resume:
                self.resume()
            self.train_loop()
            self.checkpointer.wait()
            live = [evaluator for evaluator in self.evaluators
                    if getattr(evaluator, "in_process", False) and evaluator.on_end]
            if live:
                with self.averaged():
                    for evaluator in live:
                        evaluator.during_training(self, None, "final")
            self.finish()
        except BaseException:
            for evaluator in self.evaluators:
                cancel = getattr(evaluator, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except BaseException:
                        logger.exception("could not cancel evaluator %s after training failed",
                                         getattr(evaluator, "benchmark", type(evaluator).__name__))
            try:
                self.checkpointer.wait()
            except BaseException:
                logger.exception("could not finish pending checkpoint after training failed")
            try:
                self.ctx.close()
            except BaseException:
                logger.exception("could not close distributed context after training failed")
            try:
                self.monitor.close()
            except BaseException:
                logger.exception("could not close training monitor after training failed")
            raise
        finally:
            if owns_signal_handler:
                signal.signal(signal.SIGTERM, previous_sigterm)

    def build_data(self) -> None:
        params = self.cfg.dataloader.params
        rank, world = self.ctx.rank, self.ctx.world_size
        train_ds = build_split(self.cfg, self.agent, "train")
        self.validate_train_data(train_ds)
        seed = int(self.args.seed or 0)
        self.train_loader = make_loader(train_ds, params, True, rank, world, seed=seed)
        if not len(self.train_loader):
            raise ValueError("training loader has no batches; check the split/cache or reduce the global batch size")
        if not self.args.skip_validation:
            val_ds = build_split(self.cfg, self.agent, "val")
            self.val_loader = make_loader(val_ds, params, False, rank, world, seed=seed)
            if not len(self.val_loader):
                raise ValueError("validation loader has no batches; check its split/cache or explicitly skip validation")
        self.global_batch = int(params.batch_size) * world * self.args.grad_accum
        if self.ctx.is_main:
            logger.info(
                "data=%s train=%d val=%s global_batch=%d steps/epoch=%d",
                loader_name(self.cfg),
                len(train_ds),
                len(self.val_loader.dataset) if self.val_loader is not None else "skipped",
                self.global_batch,
                math.ceil(len(self.train_loader) * self.inner_steps() / self.args.grad_accum),
            )

    def validate_train_data(self, dataset) -> None:
        tasks = _dataset_tasks(dataset)
        if len(tasks) > 1 and not bool(self.cfg.get("mixture_batch_by_task", False)):
            raise ValueError("mixed-task training requires mixture_batch_by_task=true so each batch has one task")
        needs_unused = len(tasks) > 1 or (
            _is_builtin_recogdrive_agent(self.agent)
            and ("vqa" in tasks or bool(getattr(self.agent, "train_backbone", False)))
        )
        if needs_unused and self.args.strategy == "ddp" and not self.args.find_unused_parameters:
            raise ValueError("this DDP training path has task-specific unused parameters; set train.find_unused_parameters=true")
        if getattr(self.agent, "grpo", False):
            reward = getattr(getattr(self.agent, "action_head", None), "reward_fn", None)
            unsupported = tasks - {"planning", "vqa"}
            if unsupported:
                raise ValueError(f"GRPO does not support training tasks {sorted(unsupported)}")
            if "planning" not in tasks:
                raise ValueError("GRPO training requires at least one planning source")
            leaves = _dataset_leaves(dataset)
            planning = [leaf for leaf in leaves if "planning" in _dataset_tasks(leaf)]
            vqa = [leaf for leaf in leaves if _dataset_tasks(leaf) == {"vqa"}]
            if not planning:
                raise ValueError("GRPO training requires at least one planning source")
            if vqa:
                from recogdrive.data.vqa import VQADataset

                if not all(isinstance(source, VQADataset) for source in vqa):
                    raise ValueError("GRPO mixtures support only the registered VQA dataset for VQA batches")
            if _uses_pdm_reward(reward):
                from recogdrive.data.navsim import NavsimSamples

                if not all(isinstance(source, NavsimSamples) for source in planning):
                    raise ValueError("the built-in NAVSIM PDM GRPO reward requires NAVSIM-only planning sources")
            elif getattr(self.agent, "grpo_reward", None) in ("waymo_rfs", "waymo_ade"):
                from recogdrive.adapters.waymoe2e import reward as waymo_reward_module
                from recogdrive.data.waymoe2e import WaymoE2ECacheOnlyDataset

                kind = self.agent.grpo_reward
                expected_reward = getattr(waymo_reward_module,
                                          "WaymoRFSReward" if kind == "waymo_rfs" else "WaymoADEReward", None)
                if expected_reward is None or not isinstance(reward, expected_reward) or not all(
                    isinstance(source, WaymoE2ECacheOnlyDataset) and len(source)
                    and (kind != "waymo_rfs" or source.labelled_only) for source in planning
                ):
                    requirement = "labelled-only" if kind == "waymo_rfs" else "non-empty"
                    raise ValueError(f"Waymo {kind} GRPO requires {requirement} Waymo planning data")
                reward_cache = Path(reward.cache_path).expanduser().resolve()
                if any(Path(source.cache_path).expanduser().resolve() != reward_cache
                       or source.split != reward.split for source in planning):
                    raise ValueError(f"Waymo {kind} training data must use the reward's same cache root and split")

    def inner_steps(self) -> int:
        """Optimizer passes per batch: GRPO reuses one rollout for grpo_num_iterations updates."""
        return int(getattr(self.agent, "grpo_num_iterations", 1) or 1) if getattr(self.agent, "grpo", False) else 1

    def _mixed_grpo_batch_counts(self, epoch: int) -> list[int]:
        """Return the actual forward count of each homogeneous task batch for schedule sizing."""
        sample_tasks = getattr(self.train_loader.dataset, "sample_tasks", None)
        if sample_tasks is None:
            raise ValueError("mixed-task GRPO needs per-sample task labels to size its optimizer schedule")
        sampler = self.train_loader.sampler
        previous_epoch = getattr(sampler, "epoch", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        try:
            counts = []
            for indices in self.train_loader.batch_sampler:
                tasks = {sample_tasks[index] for index in indices}
                if len(tasks) != 1:
                    raise ValueError(f"GRPO sampler produced a mixed or empty task batch: {sorted(tasks)}")
                task = next(iter(tasks))
                if task not in ("planning", "vqa"):
                    raise ValueError(f"unsupported GRPO training task {task!r}")
                counts.append(self.inner_steps() if task == "planning" else 1)
            return counts
        finally:
            if previous_epoch is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(previous_epoch)

    @staticmethod
    def _scheduled_steps(batch_counts: list[int], grad_accum: int) -> int:
        micro = steps = 0
        for batch_index, count in enumerate(batch_counts):
            for inner in range(count):
                next_micro = micro + 1
                sync = next_micro == grad_accum or (batch_index == len(batch_counts) - 1 and inner == count - 1)
                if sync:
                    steps += 1
                    micro = 0
                else:
                    micro = next_micro
        return steps

    def build_optimizer(self) -> None:
        """The model's optimizer and schedule; a get_optimizers(total_steps, steps_per_epoch) gets the training
        length in optimizer steps. A scheduler given as {"scheduler", "interval": "step"} (Lightning's form)
        advances every optimizer step, otherwise every epoch."""
        # With one GRPO pass every task contributes one forward, so the historical constant-rate sizing is exact.
        mixed_grpo = (bool(getattr(self.agent, "grpo", False)) and self.inner_steps() > 1
                      and len(_dataset_tasks(self.train_loader.dataset)) > 1)
        if mixed_grpo:
            # Exact step-LR sizing is O(epochs × batches) here, but enumerates sampler indices only; the main and
            # one-pass paths avoid this startup scan entirely.
            epoch_steps = []
            completed = micro = 0
            reached_limit = False
            for epoch in range(self.args.max_epochs):
                counts = self._mixed_grpo_batch_counts(epoch)
                epoch_steps.append(self._scheduled_steps(counts, self.args.grad_accum))
                for batch_index, count in enumerate(counts):
                    for inner in range(count):
                        next_micro = micro + 1
                        last_forward = batch_index == len(counts) - 1 and inner == count - 1
                        sync = next_micro == self.args.grad_accum or last_forward
                        micro = 0 if sync else next_micro
                        completed += int(sync)
                    if self.args.max_steps and completed >= self.args.max_steps and micro == 0:
                        reached_limit = True
                        break
                if reached_limit:
                    break
            steps_per_epoch = max(1, round(sum(epoch_steps) / max(len(epoch_steps), 1)))
            full_steps = completed if reached_limit else sum(epoch_steps)
            total_steps = full_steps
        else:
            schedule = None
            steps_per_epoch = max(1, math.ceil(len(self.train_loader) * self.inner_steps() / self.args.grad_accum))
            full_steps = steps_per_epoch * self.args.max_epochs
            total_steps = full_steps
        if self.args.max_steps and not mixed_grpo:
            total_steps = min(total_steps, self.args.max_steps)
            if getattr(self.agent, "grpo", False) and self.args.max_steps < full_steps:
                completed = 0
                for _ in range(self.args.max_epochs):
                    if completed + steps_per_epoch < self.args.max_steps:
                        completed += steps_per_epoch
                        continue
                    needed = self.args.max_steps - completed
                    for batch_end in range(1, len(self.train_loader) + 1):
                        forwards = batch_end * self.inner_steps()
                        safe = forwards % self.args.grad_accum == 0 or batch_end == len(self.train_loader)
                        if safe and math.ceil(forwards / self.args.grad_accum) >= needed:
                            total_steps = completed + math.ceil(forwards / self.args.grad_accum)
                            break
                    break
        get_optimizers = self.agent.get_optimizers
        if "total_steps" in inspect.signature(get_optimizers).parameters:
            optim_cfg = get_optimizers(total_steps=total_steps, steps_per_epoch=steps_per_epoch)
        else:
            optim_cfg = get_optimizers()
        scheduler = None
        if isinstance(optim_cfg, dict):
            self.optimizer = optim_cfg["optimizer"]
            scheduler = optim_cfg.get("lr_scheduler")
        else:
            self.optimizer = optim_cfg
        self.lr_interval = "epoch"
        if isinstance(scheduler, dict):
            self.lr_interval = scheduler.get("interval", "epoch")
            scheduler = scheduler["scheduler"]
            if self.lr_interval not in ("step", "epoch"):
                raise ValueError(f"lr scheduler interval must be step or epoch, got {self.lr_interval!r}")
        self.lr_scheduler = scheduler
        self.clip_params = [p for p in self.agent.parameters() if p.requires_grad]
        low = [p for p in self.clip_params if p.dtype != torch.float32]
        if low and not getattr(self.optimizer, "fp32_master", False):
            raise ValueError(
                f"{len(low)} trainable parameters are {low[0].dtype} and {type(self.optimizer).__name__} would update "
                "them in that precision; use an optimizer with fp32 master weights (optimizer_type=muon), "
                "train.strategy=fsdp, or train.precision=fp32")

    def resume(self) -> None:
        path = self.checkpointer.latest()
        if path is None:
            if self.ctx.is_main:
                logger.info("resume: no checkpoint under %s, starting fresh", self.checkpointer.root)
            return
        extra = self.checkpointer.load(path, self.model, self.optimizer, self.ema)
        self.start_epoch = int(extra["epoch"])
        self.start_batch = int(extra.get("batch", 0))
        self.global_step = int(extra["global_step"])
        self.best_val = float(extra["best_val"])
        rank_state = extra.get("rank_state", {})
        self.resume_rng = rank_state.get("rng")
        self.resume_epoch_rng = rank_state.get("epoch_rng")
        if self.lr_scheduler is not None and extra.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
        if self.ctx.is_main:
            logger.info("resumed from %s at epoch %d batch %d step %d", path, self.start_epoch,
                        self.start_batch, self.global_step)

    def capture_rng_state(self) -> dict:
        state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        if self.ctx.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.ctx.device)
        return state

    def restore_rng_state(self, state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and self.ctx.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"], self.ctx.device)

    def train_loop(self) -> None:
        args, model, agent, device = self.args, self.model, self.agent, self.ctx.device
        use_grpo = bool(getattr(agent, "grpo", False))
        inner_steps = self.inner_steps()
        batches = DevicePrefetcher(self.train_loader, device)
        sampler = getattr(self.train_loader, "sampler", None)
        stop_requested = False
        save_requested = False
        for epoch in range(self.start_epoch, args.max_epochs):
            self.epoch = epoch
            model.train()
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            batch_index = self.start_batch if epoch == self.start_epoch else 0
            if batch_index:
                if self.resume_epoch_rng is not None:
                    self.restore_rng_state(self.resume_epoch_rng)
            elif epoch == self.start_epoch and self.resume_rng is not None:
                self.restore_rng_state(self.resume_rng)
            epoch_rng = self.capture_rng_state()
            batch_iter = iter(batches)
            for _ in range(batch_index):
                try:
                    next(batch_iter)
                except StopIteration as exc:
                    raise RuntimeError(f"resume batch {batch_index} exceeds the {len(self.train_loader)} batches in epoch {epoch}") from exc
            if batch_index and self.resume_rng is not None:
                self.restore_rng_state(self.resume_rng)
            self.resume_rng = self.resume_epoch_rng = None
            self.start_batch = 0
            loss_sum = torch.zeros((), device=device)
            loss_count = 0
            window = {"time": time.perf_counter(), "wait": batches.wait_s, "step": self.global_step,
                      "input_samples": 0}
            while batch_index < len(self.train_loader):
                try:
                    features, targets, tokens = next(batch_iter)
                except StopIteration as exc:
                    raise RuntimeError(f"training loader ended before batch {batch_index}") from exc
                window["input_samples"] += _batch_size(features, targets, tokens)
                batch_task = _batch_task(features)
                batch_inner_steps = inner_steps if use_grpo and batch_task == "planning" else 1
                log_due = False
                log_grad_norm = None
                for inner in range(batch_inner_steps):
                    next_micro = self.micro + 1
                    last_forward = batch_index == len(self.train_loader) - 1 and inner == batch_inner_steps - 1
                    sync = next_micro == args.grad_accum or last_forward
                    with gradient_sync(model, sync):
                        if use_grpo and batch_task == "planning":
                            out = model(features, targets, tokens, reuse_rollout=inner > 0)
                        else:
                            out = model(features, targets, tokens)
                        loss = out["loss"] if isinstance(out, dict) else out.loss
                        (loss / args.grad_accum).backward()
                    loss_sum += loss.detach().float()
                    loss_count += 1
                    self.micro = 0 if sync else next_micro
                    if not sync:
                        continue
                    if next_micro < args.grad_accum:
                        scale = args.grad_accum / next_micro
                        for parameter in self.clip_params:
                            if parameter.grad is not None:
                                parameter.grad.mul_(scale)
                    grad_norm = None
                    if args.grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.clip_params, args.grad_clip, foreach=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.lr_scheduler is not None and self.lr_interval == "step":
                        self.lr_scheduler.step()
                    if self.ema is not None:
                        self.ema.update()
                    self.global_step += 1
                    if self.global_step % args.log_every == 0:
                        log_due = True
                        log_grad_norm = grad_norm
                    if args.save_every_n_steps and self.global_step % args.save_every_n_steps == 0:
                        save_requested = True
                    if args.max_steps and self.global_step >= args.max_steps:
                        stop_requested = True
                if use_grpo and batch_task == "planning":
                    agent.clear_grpo_rollout()
                batch_index += 1
                if log_due:
                    self.log_train(epoch, out, loss, log_grad_norm, window, batches)
                safe_boundary = self.micro == 0
                last_batch = batch_index == len(self.train_loader)
                # ponytail: GRPO max_steps waits for a full rollout and clean accumulation boundary; exact
                # mid-rollout resume would require storing its potentially large cached rollout.
                if safe_boundary and not last_batch and (save_requested or stop_requested):
                    self.save_resume(epoch, batch_index, epoch_rng)
                    save_requested = False
                    if stop_requested:
                        return
            if self.lr_scheduler is not None and self.lr_interval == "epoch":
                self.lr_scheduler.step()
            train_loss = self.ctx.all_reduce_mean(loss_sum / max(loss_count, 1)).item()
            self.end_epoch(epoch, train_loss, use_grpo)
            save_requested = False
            if stop_requested:
                break

    def log_train(self, epoch: int, out, loss, grad_norm, window: dict, batches: DevicePrefetcher) -> None:
        names, values = ["loss"], [loss.detach().float()]
        for key in EXTRA_METRICS:
            value = out.get(key) if isinstance(out, dict) else getattr(out, key, None)
            if torch.is_tensor(value):
                names.append(key)
                values.append(value.detach().float().mean())
        if grad_norm is not None:
            names.append("grad_norm")
            values.append((grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm).float())
        stats = self.ctx.all_reduce_mean(torch.stack(values)).tolist()
        now = time.perf_counter()
        steps = max(self.global_step - window["step"], 1)
        elapsed = now - window["time"]
        waited = batches.wait_s - window["wait"]
        input_samples = self.ctx.all_reduce_sum(
            torch.tensor(float(window.get("input_samples", 0)), device=self.ctx.device)
        ).item()
        window.update(time=now, wait=batches.wait_s, step=self.global_step, input_samples=0)
        if not self.ctx.is_main:
            return
        metrics = {f"train/{name}": value for name, value in zip(names, stats)}
        metrics.update({
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "perf/step_ms": 1000.0 * elapsed / steps,
            "perf/optimizer_steps_s": steps / max(elapsed, 1e-9),
            "perf/data_wait_ms": 1000.0 * waited / steps,
            "perf/samples_per_s": input_samples / max(elapsed, 1e-9),
        })
        if torch.cuda.is_available():
            metrics["perf/max_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        self.monitor.log(metrics, self.global_step, epoch + 1)
        logger.info("[epoch %d step %d] %s", epoch + 1, self.global_step,
                    " ".join(f"{key.split('/', 1)[1]}={value:.4g}" for key, value in metrics.items()))

    def averaged(self):
        """Context in which the model holds its EMA weights (a no-op without EMA)."""
        return self.ema.swapped() if self.ema is not None else contextlib.nullcontext()

    def ckpt_suffixes(self):
        return ["", EMA_SUFFIX] if self.ema is not None else [""]

    @torch.no_grad()
    def validate(self, use_grpo: bool) -> float:
        self.model.eval()
        device = self.ctx.device
        dataset = self.val_loader.dataset
        task_weights = getattr(dataset, "task_loss_weights", None)
        if not task_weights:
            tasks = sorted(_dataset_tasks(dataset))
            task_weights = {task: 1.0 / len(tasks) for task in tasks}
        tasks = list(task_weights)
        task_index = {task: index for index, task in enumerate(tasks)}
        loss_sums = torch.zeros(len(tasks), device=device)
        loss_counts = torch.zeros(len(tasks), device=device)
        batch_sampler = getattr(self.val_loader, "batch_sampler", None)
        valid_batch_sizes = getattr(batch_sampler, "valid_batch_sizes", None)
        with self.averaged():
            for batch_index, (features, targets, tokens) in enumerate(DevicePrefetcher(self.val_loader, device)):
                preds = self.model(features, targets) if use_grpo else self.model(features, targets, tokens)
                valid_count = valid_batch_sizes[batch_index] if valid_batch_sizes is not None else \
                    _batch_size(features, targets, tokens)
                if valid_count == 0:
                    continue
                batch_tasks = features.get("task", "planning")
                if isinstance(batch_tasks, (list, tuple)):
                    task_set = set(batch_tasks[:valid_count])
                    if len(task_set) != 1:
                        raise ValueError(f"validation batch mixes tasks: {sorted(task_set)}")
                    task = next(iter(task_set))
                else:
                    task = batch_tasks
                if task not in task_index:
                    raise ValueError(f"validation task {task!r} has no configured loss weight")
                loss = self.agent.compute_loss(features, targets, preds).detach().float()
                if task == "vqa":
                    count = (features["labels"][:valid_count] != -100).sum().to(torch.float32)
                else:
                    count = loss.new_tensor(float(valid_count))
                index = task_index[task]
                loss_sums[index] += loss * count
                loss_counts[index] += count
        stats = self.ctx.all_reduce_sum(torch.stack((loss_sums, loss_counts)))
        means = {}
        for index, task in enumerate(tasks):
            if stats[1, index] > 0:
                means[task] = stats[0, index] / stats[1, index]
        missing = set(tasks) - set(means)
        if missing:
            raise ValueError(f"validation has no effective samples/tokens for tasks: {sorted(missing)}")
        self.monitor.log({f"val/task/{task}/loss": value.item() for task, value in means.items()},
                         self.global_step, self.epoch + 1)
        weight_sum = sum(float(task_weights[task]) for task in means)
        return sum(float(task_weights[task]) * means[task].item() for task in means) / weight_sum

    def end_epoch(self, epoch: int, train_loss: float, use_grpo: bool) -> None:
        val_loss = None
        if self.val_loader is not None and (epoch + 1) % self.args.val_every_n_epochs == 0:
            val_loss = self.validate(use_grpo)
        improved = val_loss is not None and val_loss < self.best_val
        if improved:
            self.best_val = val_loss
        epoch_metrics = {"epoch/train_loss": train_loss}
        if val_loss is not None:
            epoch_metrics["val/loss"] = val_loss
        self.monitor.log(epoch_metrics, self.global_step, epoch + 1)
        self.monitor.plot()
        tagged = self.export(epoch, train_loss, val_loss, improved)
        self.evaluate(epoch, tagged)
        if val_loss is not None:
            if self.ctx.is_main:
                self.prune_topk(val_loss, f"epoch_{epoch + 1:04d}_val_{val_loss:.6f}.ckpt")
            self.ctx.barrier()
        self.save_resume(epoch + 1, 0)

    def evaluate(self, epoch: int, ckpt: Path) -> None:
        """Periodic evaluation of the epoch's checkpoint; every rank takes part, with the model holding its weights."""
        due = [e for e in self.evaluators if e.every_n_epochs and (epoch + 1) % e.every_n_epochs == 0]
        if due:
            with self.averaged():
                for evaluator in due:
                    evaluator.during_training(self, ckpt, f"epoch_{epoch + 1:04d}")
        for evaluator in self.evaluators:
            self.report(evaluator, evaluator.poll() if self.ctx.is_main else [])

    def report(self, evaluator, results) -> None:
        for tag, result in results:
            logger.info("eval %s %s: %s", evaluator.benchmark, tag, result)
            if "error" in result:
                self.evaluation_errors.append({"benchmark": evaluator.benchmark, "tag": tag,
                                               "error": str(result["error"])})
            self.monitor.log({f"eval/{evaluator.benchmark}/{key}": value for key, value in result.items()},
                             self.global_step, self.epoch + 1)

    def export(self, epoch: int, train_loss: float, val_loss: Optional[float], improved: bool) -> Path:
        """Writes last.ckpt (and last-EMA.ckpt), then the top-k and best copies. Returns, on every rank, the
        checkpoint evaluators score for this epoch: the tagged top-k copy, or last without validation (EMA twin
        with EMA)."""
        suffixes = self.ckpt_suffixes()
        for suffix in suffixes:
            with self.averaged() if suffix else contextlib.nullcontext():
                state = export_model_state(self.model)
            if self.ctx.is_main:
                self.write_ckpt(state, self.output_dir / f"last{suffix}.ckpt")
        tag = None if val_loss is None else f"epoch_{epoch + 1:04d}_val_{val_loss:.6f}"
        tagged = self.output_dir / f"{tag or 'last'}{suffixes[-1]}.ckpt"
        if not self.ctx.is_main:
            return tagged
        logger.info(
            "epoch %d/%d train/loss=%.4f val/loss=%s -> %s",
            epoch + 1, self.args.max_epochs, train_loss,
            "skipped" if val_loss is None else f"{val_loss:.4f}", self.output_dir / "last.ckpt",
        )
        if tag is None:
            return tagged
        for suffix in suffixes:
            last = self.output_dir / f"last{suffix}.ckpt"
            self.copy_ckpt(last, self.output_dir / f"{tag}{suffix}.ckpt")
            if improved:
                self.copy_ckpt(last, self.output_dir / f"best{suffix}.ckpt")
        if improved:
            logger.info("new best val/loss=%.6f", val_loss)
        return tagged

    def write_ckpt(self, state: dict, path: Path) -> None:
        """HF-compatible VLM weights go to a side export; adapters and non-HF backbone weights stay in the checkpoint."""
        if _is_builtin_recogdrive_agent(self.agent):
            teachers = ("action_head.old_policy.", "action_head.ref_policy.")
            state = {key: value for key, value in state.items()
                     if not key.startswith(teachers)}
        backbone_model = getattr(getattr(self.agent, "backbone", None), "model", None)
        hf_export = callable(getattr(backbone_model, "save_pretrained", None))
        trainable_vlm = hf_export and any(
            name.startswith(VLM_PREFIX) and ADAPTER_MARK not in name and parameter.requires_grad
            for name, parameter in self.agent.named_parameters()
        )
        vlm = {key[len(VLM_PREFIX):]: value for key, value in state.items()
               if trainable_vlm and key.startswith(VLM_PREFIX) and ADAPTER_MARK not in key}
        base_keys = {key for key in state if hf_export and key.startswith(VLM_PREFIX) and ADAPTER_MARK not in key}
        weights = {key: value for key, value in state.items() if key not in base_keys}
        torch.save({"state_dict": weights, "vlm_export": bool(vlm)}, path)
        self.save_vlm(vlm, path)

    def save_vlm(self, vlm_state: dict, ckpt: Path) -> None:
        if not vlm_state:
            return
        backbone = getattr(self.agent, "backbone", None)
        out_dir = vlm_dir(ckpt)
        out_dir.mkdir(parents=True, exist_ok=True)
        backbone.model.save_pretrained(out_dir, state_dict=vlm_state, safe_serialization=True, max_shard_size="10GB")
        saved = []
        for preprocessor in (getattr(backbone, "processor", None), getattr(backbone, "tokenizer", None)):
            if preprocessor is not None and all(preprocessor is not item for item in saved):
                preprocessor.save_pretrained(out_dir)
                saved.append(preprocessor)

    def copy_ckpt(self, src: Path, dst: Path) -> None:
        shutil.copyfile(src, dst)
        if vlm_dir(src).is_dir():
            shutil.rmtree(vlm_dir(dst), ignore_errors=True)
            shutil.copytree(vlm_dir(src), vlm_dir(dst))

    def prune_topk(self, val_loss: float, name: str) -> None:
        meta_path = self.output_dir / "best_checkpoints.json"
        best = []
        if meta_path.is_file():
            try:
                best = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                best = []
        best.append({"path": name, "val_loss": float(val_loss)})
        best = sorted(best, key=lambda item: item["val_loss"])[: self.args.top_k]
        keep = {item["path"] for item in best}
        keep |= {Path(name).stem + EMA_SUFFIX + ".ckpt" for name in keep}
        for path in self.output_dir.glob("epoch_*.ckpt"):
            if path.name not in keep:
                path.unlink(missing_ok=True)
                shutil.rmtree(vlm_dir(path), ignore_errors=True)
        meta_path.write_text(json.dumps(best, indent=2))

    def save_resume(self, epoch: int, batch: int = 0, epoch_rng: Optional[dict] = None) -> None:
        extra = {
            "epoch": epoch,
            "batch": batch,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "rank_state": {"rng": self.capture_rng_state(), "epoch_rng": epoch_rng},
        }
        self.checkpointer.save(self.global_step, self.model, self.optimizer, extra, self.ema)

    def finish(self) -> None:
        suffix = self.ckpt_suffixes()[-1]
        best = self.output_dir / f"best{suffix}.ckpt"
        final = best if best.is_file() else self.output_dir / f"last{suffix}.ckpt"
        # Release the GPUs before the final evaluation runs in its own processes. torch internals behind the
        # compiled blocks keep references to the weights, so their storage is dropped explicitly; FSDP2 shards
        # (DTensor) go with their modules.
        for tensor in [*self.agent.parameters(), *self.agent.buffers()]:
            if not hasattr(tensor, "to_local"):
                tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        self.model = self.optimizer = self.lr_scheduler = self.ema = None
        self.agent = None
        self.clip_params = []
        self.train_loader = self.val_loader = None
        for evaluator in self.evaluators:
            evaluator.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.ctx.is_main:
                logger.info("released GPU memory: %.2f GiB still allocated, %.2f GiB reserved",
                            torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30)
        self.ctx.close()
        if self.ctx.is_main:
            for evaluator in self.evaluators:
                try:
                    results = evaluator.finish(final)
                except Exception as exc:
                    logger.exception("eval %s final evaluation failed", evaluator.benchmark)
                    results = [("final", {"error": f"{type(exc).__name__}: {exc}"})]
                self.report(evaluator, results)
        self.monitor.close()
        if self.ctx.is_main and self.evaluation_errors:
            failure_log = self.output_dir / "evaluation_errors.json"
            failure_log.write_text(json.dumps(self.evaluation_errors, indent=2), encoding="utf-8")
            raise RuntimeError(f"{len(self.evaluation_errors)} requested evaluations failed; see {failure_log}")


def vlm_dir(ckpt: Path) -> Path:
    return ckpt.with_name(ckpt.stem + "_vlm")
