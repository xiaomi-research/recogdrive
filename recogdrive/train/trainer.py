"""Training loop for VLA agents. Parallelism, data pipelines and evaluators are plugged in."""

import contextlib
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from recogdrive.data import DevicePrefetcher, build_split, loader_name, make_loader
from recogdrive.distributed import DistributedContext, ResumeCheckpointer, export_model_state, gradient_sync, parallelize
from recogdrive.eval import build_evaluator
from recogdrive.train.args import TrainArgs
from recogdrive.train.ema import EMA

logger = logging.getLogger("recogdrive")

EXTRA_METRICS = ("reward", "policy_loss", "bc_loss", "clip_frac", "ratio_mean")
BACKBONE_PREFIX = "backbone."
VLM_PREFIX = "backbone.model."
EMA_SUFFIX = "-EMA"


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
        if self.args.seed is not None:
            torch.manual_seed(int(self.args.seed))
        self.agent = instantiate(cfg.agent)
        self.output_dir = Path(cfg.output_dir)
        if self.ctx.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.clip_params = []
        self.ema = None
        self.global_batch = 0
        self.checkpointer = ResumeCheckpointer(self.output_dir / "checkpoints", self.ctx, self.args.async_save)
        self.evaluator = build_evaluator(cfg, self.output_dir) if self.ctx.is_main else None
        self.start_epoch = 0
        self.global_step = 0
        self.best_val = math.inf
        self.micro = 0

    def fit(self) -> None:
        self.build_data()
        if hasattr(self.agent, "initialize"):
            self.agent.initialize()
        self.model = parallelize(self.agent, self.args, self.ctx)
        self.build_optimizer()
        if self.args.ema_decay > 0:
            self.ema = EMA(self.agent, self.args.ema_decay)
            if self.ctx.is_main:
                logger.info("ema decay=%s over %d trainable tensors", self.args.ema_decay, len(self.ema.params))
        if self.args.resume:
            self.resume()
        try:
            self.train_loop()
        finally:
            self.checkpointer.wait()
        self.finish()

    def build_data(self) -> None:
        params = self.cfg.dataloader.params
        rank, world = self.ctx.rank, self.ctx.world_size
        train_ds = build_split(self.cfg, self.agent, "train")
        self.train_loader = make_loader(train_ds, params, True, rank, world)
        if not self.args.skip_validation:
            self.val_loader = make_loader(build_split(self.cfg, self.agent, "val"), params, False, rank, world)
        self.global_batch = int(params.batch_size) * world * self.args.grad_accum
        if self.ctx.is_main:
            logger.info(
                "data=%s train=%d val=%s global_batch=%d steps/epoch=%d",
                loader_name(self.cfg),
                len(train_ds),
                len(self.val_loader.dataset) if self.val_loader is not None else "skipped",
                self.global_batch,
                len(self.train_loader) // self.args.grad_accum,
            )

    def build_optimizer(self) -> None:
        optim_cfg = self.agent.get_optimizers()
        if isinstance(optim_cfg, dict):
            self.optimizer = optim_cfg["optimizer"]
            self.lr_scheduler = optim_cfg.get("lr_scheduler")
        else:
            self.optimizer = optim_cfg
        if type(self.optimizer).__name__ == "Muon" and self.args.strategy == "fsdp" and self.ctx.world_size > 1:
            raise ValueError(
                "Muon orthogonalizes whole matrices but FSDP gives each rank a shard; "
                "use train.strategy=ddp train.precision=fp32 with optimizer_type=muon"
            )
        self.clip_params = [p for p in self.agent.parameters() if p.requires_grad]

    def resume(self) -> None:
        path = self.checkpointer.latest()
        if path is None:
            if self.ctx.is_main:
                logger.info("resume: no checkpoint under %s, starting fresh", self.checkpointer.root)
            return
        extra = self.checkpointer.load(path, self.model, self.optimizer, self.ema)
        self.start_epoch = int(extra["epoch"])
        self.global_step = int(extra["global_step"])
        self.best_val = float(extra["best_val"])
        if self.lr_scheduler is not None and extra.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
        if self.ctx.is_main:
            logger.info("resumed from %s at epoch %d step %d", path, self.start_epoch, self.global_step)

    def train_loop(self) -> None:
        args, model, agent, device = self.args, self.model, self.agent, self.ctx.device
        use_grpo = bool(getattr(agent, "grpo", False))
        inner_steps = int(getattr(agent, "grpo_num_iterations", 1) or 1) if use_grpo else 1
        batches = DevicePrefetcher(self.train_loader, device)
        sampler = getattr(self.train_loader, "sampler", None)
        stop = False
        for epoch in range(self.start_epoch, args.max_epochs):
            model.train()
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            loss_sum = torch.zeros((), device=device)
            loss_count = 0
            window = {"time": time.perf_counter(), "wait": batches.wait_s, "step": self.global_step}
            for features, targets, tokens in batches:
                for inner in range(inner_steps):
                    self.micro += 1
                    sync = self.micro % args.grad_accum == 0
                    with gradient_sync(model, sync):
                        if use_grpo:
                            out = model(features, targets, tokens, reuse_rollout=inner > 0)
                        else:
                            out = model(features, targets, tokens)
                        loss = out.loss
                        (loss / args.grad_accum).backward()
                    loss_sum += loss.detach().float()
                    loss_count += 1
                    if not sync:
                        continue
                    grad_norm = None
                    if args.grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.clip_params, args.grad_clip, foreach=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update()
                    self.global_step += 1
                    if self.global_step % args.log_every == 0:
                        self.log_train(epoch, out, loss, grad_norm, window, batches)
                    if args.save_every_n_steps and self.global_step % args.save_every_n_steps == 0:
                        self.save_resume(epoch)
                    if args.max_steps and self.global_step >= args.max_steps:
                        stop = True
                        break
                if use_grpo:
                    agent.clear_grpo_rollout()
                if stop:
                    break
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            train_loss = self.ctx.all_reduce_mean(loss_sum / max(loss_count, 1)).item()
            self.end_epoch(epoch, train_loss, use_grpo)
            if stop:
                break

    def log_train(self, epoch: int, out, loss, grad_norm, window: dict, batches: DevicePrefetcher) -> None:
        names, values = ["loss"], [loss.detach().float()]
        for key in EXTRA_METRICS:
            value = getattr(out, key, None)
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
        window.update(time=now, wait=batches.wait_s, step=self.global_step)
        if not self.ctx.is_main:
            return
        metrics = " ".join(f"{name}={value:.4f}" for name, value in zip(names, stats))
        logger.info(
            "[epoch %d step %d] %s lr=%.3g step_ms=%.1f data_wait_ms=%.1f samples/s=%.1f",
            epoch + 1,
            self.global_step,
            metrics,
            self.optimizer.param_groups[0]["lr"],
            1000.0 * elapsed / steps,
            1000.0 * waited / steps,
            self.global_batch * steps / max(elapsed, 1e-9),
        )

    def averaged(self):
        """Context in which the model holds its EMA weights (a no-op without EMA)."""
        return self.ema.swapped() if self.ema is not None else contextlib.nullcontext()

    def ckpt_suffixes(self):
        return ["", EMA_SUFFIX] if self.ema is not None else [""]

    @torch.no_grad()
    def validate(self, use_grpo: bool) -> float:
        self.model.eval()
        device = self.ctx.device
        total = torch.zeros((), device=device)
        count = 0
        with self.averaged():
            for features, targets, tokens in DevicePrefetcher(self.val_loader, device):
                preds = self.model(features, targets) if use_grpo else self.model(features, targets, tokens)
                total += self.agent.compute_loss(features, targets, preds).detach().float()
                count += 1
        sums = self.ctx.all_reduce_sum(torch.stack([total, torch.tensor(float(count), device=device)]))
        return (sums[0] / sums[1].clamp(min=1.0)).item()

    def end_epoch(self, epoch: int, train_loss: float, use_grpo: bool) -> None:
        val_loss = None
        if self.val_loader is not None and (epoch + 1) % self.args.val_every_n_epochs == 0:
            val_loss = self.validate(use_grpo)
        improved = val_loss is not None and val_loss < self.best_val
        if improved:
            self.best_val = val_loss
        self.export(epoch, train_loss, val_loss, improved)
        self.save_resume(epoch + 1)

    def export(self, epoch: int, train_loss: float, val_loss: Optional[float], improved: bool) -> None:
        """Writes last.ckpt (and last-EMA.ckpt), then the top-k and best copies; evaluators get the EMA ones."""
        suffixes = self.ckpt_suffixes()
        for suffix in suffixes:
            with self.averaged() if suffix else contextlib.nullcontext():
                state = export_model_state(self.model)
            if self.ctx.is_main:
                self.write_ckpt(state, self.output_dir / f"last{suffix}.ckpt")
        if not self.ctx.is_main:
            return
        logger.info(
            "epoch %d/%d train/loss=%.4f val/loss=%s -> %s",
            epoch + 1, self.args.max_epochs, train_loss,
            "skipped" if val_loss is None else f"{val_loss:.4f}", self.output_dir / "last.ckpt",
        )
        if val_loss is None:
            return
        tag = f"epoch_{epoch + 1:04d}_val_{val_loss:.6f}"
        for suffix in suffixes:
            last = self.output_dir / f"last{suffix}.ckpt"
            self.copy_ckpt(last, self.output_dir / f"{tag}{suffix}.ckpt")
            if improved:
                self.copy_ckpt(last, self.output_dir / f"best{suffix}.ckpt")
        self.prune_topk(val_loss, f"{tag}.ckpt")
        if improved:
            logger.info("new best val/loss=%.6f", val_loss)
        if self.evaluator is not None:
            for name, result in self.evaluator.poll():
                logger.info("eval %s: %s", name, result)
            if self.evaluator.every_n_epochs and (epoch + 1) % self.evaluator.every_n_epochs == 0:
                self.evaluator.submit(self.output_dir / f"{tag}{suffixes[-1]}.ckpt", f"epoch_{epoch + 1:04d}")

    def write_ckpt(self, state: dict, path: Path) -> None:
        weights = {k: v for k, v in state.items() if not k.startswith(BACKBONE_PREFIX)}
        vlm = {k[len(VLM_PREFIX):]: v for k, v in state.items() if k.startswith(VLM_PREFIX)}
        torch.save({"state_dict": weights}, path)
        self.save_vlm(vlm, path)

    def save_vlm(self, vlm_state: dict, ckpt: Path) -> None:
        if not vlm_state:
            return
        backbone = getattr(self.agent, "backbone", None)
        out_dir = vlm_dir(ckpt)
        out_dir.mkdir(parents=True, exist_ok=True)
        backbone.model.save_pretrained(out_dir, state_dict=vlm_state, safe_serialization=True, max_shard_size="10GB")
        if getattr(backbone, "tokenizer", None) is not None:
            backbone.tokenizer.save_pretrained(out_dir)

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

    def save_resume(self, next_epoch: int) -> None:
        extra = {
            "epoch": next_epoch,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
        }
        self.checkpointer.save(self.global_step, self.model, self.optimizer, extra, self.ema)

    def finish(self) -> None:
        suffix = self.ckpt_suffixes()[-1]
        best = self.output_dir / f"best{suffix}.ckpt"
        final = best if best.is_file() else self.output_dir / f"last{suffix}.ckpt"
        # Release the GPUs before the final evaluation runs in its own processes.
        self.model = self.optimizer = self.lr_scheduler = self.ema = None
        self.agent = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.ctx.close()
        if self.evaluator is not None:
            for tag, result in self.evaluator.finish(final):
                logger.info("eval %s: %s", tag, result)


def vlm_dir(ckpt: Path) -> Path:
    return ckpt.with_name(ckpt.stem + "_vlm")
