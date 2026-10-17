#!/usr/bin/env python3
"""Run a tiny real Qwen3-VL ReCogDrive GPU train/export/reload acceptance pass.

Example (run from the repository root in the NAVSIM environment):
  torchrun --standalone --nproc_per_node=2 scripts/verify/check_gpu_training.py \
    --strategy ddp --nuscenes-root /data/yongkangli/datasets/nuscenes \
    --model-path /data/yongkangli/ckpts/Qwen3-VL-2B-Instruct --max-tokens 1024 \
    --output-dir /data/yongkangli/rcdtest/exp/recogdrive_gpu_smoke_ddp_YYYYMMDD_HHMMSS

This is a code-path smoke test, not a dataset score or throughput benchmark. Planning
features/targets and the one-row VQA question are synthetic; the image defaults to a
downscaled nuScenes-mini CAM_FRONT frame. Use --synthetic-image only when no local frame
is available. The launcher must check GPU availability before torchrun; worker ranks must
not treat their sibling ranks as competing jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

OUTPUT_DIR_CLAIMED = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="must be a new directory")
    parser.add_argument("--model-path", type=Path,
                        default=Path(os.environ.get("QWEN3_VL_PATH", "/data/yongkangli/ckpts/Qwen3-VL-2B-Instruct")))
    parser.add_argument("--nuscenes-root", type=Path,
                        default=Path(os.environ.get("NUSCENES_ROOT", "/data/yongkangli/datasets/nuscenes")))
    parser.add_argument("--image", type=Path, help="use this local image instead of nuScenes mini")
    parser.add_argument("--synthetic-image", action="store_true", help="allow a generated image when no frame is available")
    parser.add_argument("--strategy", choices=("ddp", "fsdp"), default="ddp",
                        help="ddp requires two or more ranks; fsdp uses PyTorch FSDP2")
    parser.add_argument("--compile", action="store_true", help="enable TrainArgs compile_blocks")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--activation-checkpointing-class", default="Qwen3VLTextDecoderLayer")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def choose_source_image(args: argparse.Namespace, output_dir: Path) -> tuple[Path, str]:
    if args.image is not None:
        source = args.image.expanduser().resolve(strict=True)
        return source, "user_image"
    root = args.nuscenes_root.expanduser().resolve()
    front = root / "samples" / "CAM_FRONT"
    if (root / "v1.0-mini" / "scene.json").is_file() and front.is_dir():
        source = next((path for path in front.iterdir() if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
        if source is not None:
            return source.resolve(strict=True), "nuscenes_mini_cam_front"
    if not args.synthetic_image:
        raise FileNotFoundError(
            "nuScenes mini CAM_FRONT was not found; pass --image PATH or explicitly allow --synthetic-image"
        )
    from PIL import Image, ImageDraw

    source = output_dir / "synthetic_source.png"
    image = Image.new("RGB", (112, 112), (120, 175, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 54, 111, 111), fill=(70, 75, 80))
    draw.polygon(((42, 111), (53, 54), (59, 54), (78, 111)), fill=(205, 195, 150))
    draw.rectangle((8, 67, 23, 79), fill=(180, 45, 35))
    image.save(source)
    return source, "synthetic_image"


def prepare_image(source: Path, output_dir: Path) -> dict:
    from PIL import Image

    destination = output_dir / "smoke_front.jpg"
    with Image.open(source) as image:
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image)
        image.thumbnail((112, 112), resampling.LANCZOS)
        canvas = Image.new("RGB", (112, 112), (96, 96, 96))
        canvas.paste(image, ((112 - image.width) // 2, (112 - image.height) // 2))
        canvas.save(destination, quality=90)
    with Image.open(destination) as image:
        size = list(image.size)
    return {"source": str(source), "path": str(destination), "width_height": size,
            "sha256": sha256(destination)}


def build_config(args, output_dir: Path, image_path: Path, jsonl_path: Path):
    from omegaconf import OmegaConf

    activation = [args.activation_checkpointing_class] if args.activation_checkpointing else []
    strategy = args.strategy
    # PyTorch 2.6 FSDP2 cannot detach frozen base weights from individual LoRA layers.
    # Shard the complete model on that path; DDP keeps the frozen VLM replicated.
    replicate_frozen = ["backbone"] if strategy == "ddp" else []
    return OmegaConf.create({
        "output_dir": str(output_dir),
        "experiment_name": "recogdrive_gpu_smoke",
        "data_loader": "gpu_training_smoke",
        "mixture_batch_by_task": True,
        "mixture": [],
        "vqa": {"train_jsonl": str(jsonl_path), "image_root": str(image_path.parent),
                "max_length": args.max_tokens},
        "plugins": [],
        "augment": None,
        "agent": {
            "_target_": "recogdrive.adapters.navsim.agent.ReCogDriveAgent",
            "_convert_": "all",
            "trajectory_sampling": {
                "_target_": "nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",
                "_convert_": "all", "time_horizon": 4, "interval_length": 0.5,
            },
            "vlm_path": str(args.model_path), "checkpoint_path": None, "cam_type": "single",
            "vlm_type": "qwen", "dit_type": "small", "sampling_method": "ddim",
            "cache_mode": False, "cache_hidden_state": False, "train_backbone": False,
            "lora_rank": 4, "lora_alpha": 16.0, "lora_dropout": 0.0,
            "training_target": "waypoint", "lr": 1e-4, "lr_schedule": "step",
            "optimizer_type": "adamw", "vlm_max_length": args.max_tokens,
        },
        "train": {
            "strategy": strategy, "precision": "fp32" if strategy == "ddp" else "bf16",
            "find_unused_parameters": True, "replicate_frozen": replicate_frozen,
            "compile": args.compile, "activation_checkpointing": activation,
            "activation_checkpointing_layers": None, "grad_accum": 1, "grad_clip": 0.0,
            "max_epochs": 1, "max_steps": 0, "log_every": 1, "async_save": True,
            "skip_validation": True, "seed": args.seed, "nccl_timeout": args.timeout,
        },
        "trainer": {"params": {"max_epochs": 1, "accumulate_grad_batches": 1,
                                 "gradient_clip_val": 0.0, "check_val_every_n_epoch": 1}},
        "dataloader": {"params": {"batch_size": 1, "drop_last": False, "num_workers": 0,
                                    "pin_memory": True}},
        "evaluator": {"name": None},
        "monitor": {"wandb": "disabled"},
        "infer": {"checkpoint": str(output_dir / "last.ckpt")},
    })


def install_smoke_loader(image_path: Path):
    import torch
    from torch.utils.data import Dataset

    from recogdrive.data import LOADERS
    from recogdrive.data.mixture import MixtureDataset
    from recogdrive.data.registry import with_worker_transform

    class TinyPlanningDataset(Dataset):
        task = "planning"

        def __init__(self, image):
            self.image = str(image)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            history = torch.zeros((4, 3), dtype=torch.float32)
            command = torch.tensor([0.0, 1.0, 0.0, 0.0])
            status = torch.zeros(8, dtype=torch.float32)
            future = torch.zeros((8, 3), dtype=torch.float32)
            future[:, 0] = torch.arange(1, 9, dtype=torch.float32) * 0.5
            features = {"task": "planning", "camera_paths": [("front", self.image)],
                        "history_trajectory": history, "high_command_one_hot": command,
                        "status_feature": status}
            return features, {"trajectory": future}, "gpu-smoke-planning"

    def loader(cfg, agent, split):
        if split != "train":
            raise ValueError("GPU smoke loader only supports the train split")
        planning = with_worker_transform(TinyPlanningDataset(image_path), agent)
        vqa_loader = LOADERS.get("vqa")
        if vqa_loader is None:
            raise RuntimeError("the standard VQA loader is not registered")
        vqa = vqa_loader(cfg, agent, split)
        # The task batch sampler orders one homogeneous planning batch then one homogeneous VQA batch.
        mixture = MixtureDataset([planning, vqa], batch_by_task=True)
        mixture.worker_transform_applied = True
        return mixture

    LOADERS["gpu_training_smoke"] = loader


def task_of(features) -> str:
    task = features.get("task", "planning")
    if isinstance(task, (tuple, list)):
        tasks = set(task)
        if len(tasks) != 1:
            raise ValueError("GPU smoke batches must contain exactly one task")
        task = next(iter(tasks))
    if task not in ("planning", "vqa"):
        raise ValueError("unexpected smoke task: {}".format(task))
    return task


def source_hashes(root: Path) -> dict:
    import importlib

    modules = {
        "agent.py": "recogdrive.adapters.navsim.agent",
        "backbone.py": "recogdrive.models.recogdrive.backbone",
        "trainer.py": "recogdrive.train.trainer",
        "parallelize.py": "recogdrive.distributed.parallelize",
        "checkpoint.py": "recogdrive.distributed.checkpoint",
        "registry.py": "recogdrive.data.registry",
        "mixture.py": "recogdrive.data.mixture",
        "vqa.py": "recogdrive.data.vqa",
    }

    files = {"check_gpu_training.py": Path(__file__).resolve()}
    files.update({name: Path(importlib.import_module(module).__file__).resolve()
                  for name, module in modules.items()})
    result = {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()}
    aggregate = hashlib.sha256("".join("{}:{}\n".format(name, item["sha256"])
                                      for name, item in sorted(result.items())).encode("utf-8")).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), check=True,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"git_head": commit, "files": result, "aggregate_sha256": aggregate}


def create_trainer_class():
    import torch
    import torch.distributed as dist

    from recogdrive.train.trainer import Trainer

    class GPUTrainingSmokeTrainer(Trainer):
        def __init__(self, cfg, args, image_path, hashes):
            super().__init__(cfg)
            self.smoke_args = args
            self.smoke_image = image_path
            self.hashes = hashes
            self.active_task = None
            self.last_losses = {}
            self.grad_steps = {"planning": [], "vqa": []}
            self.task_sequence = []
            self.tokens_seen = {"planning": [], "vqa": []}
            self.measuring_inference = False
            self.hook_handles = []
            self.training_started = time.perf_counter()
            self.training_peak_allocated = 0
            self.training_peak_reserved = 0
            self.activation_wrappers = 0
            self.inference_ok = False
            self.inference_error = None
            self.inference_tokens = None
            self.prediction_shape = None

        @staticmethod
        def _local(tensor):
            return tensor.to_local() if hasattr(tensor, "to_local") else tensor

        def _task_pre_hook(self, module, inputs):
            if inputs and isinstance(inputs[0], dict):
                self.active_task = task_of(inputs[0])

        def _task_forward_hook(self, module, inputs, output):
            task = self.active_task
            loss = output.get("loss") if isinstance(output, dict) else getattr(output, "loss", None)
            if task and torch.is_tensor(loss):
                self.last_losses[task] = loss.detach().float()

        def _qwen_pre_hook(self, module, inputs, kwargs):
            ids = kwargs.get("input_ids")
            task = self.active_task
            if task and torch.is_tensor(ids):
                length = int(ids.shape[-1])
                if self.measuring_inference:
                    self.inference_tokens_seen = length
                else:
                    self.tokens_seen[task].append(length)
                if length > self.smoke_args.max_tokens:
                    raise ValueError("{} Qwen input is {} tokens, limit is {}".format(
                        task, length, self.smoke_args.max_tokens))

        def build_optimizer(self):
            super().build_optimizer()
            self.hook_handles.append(self.agent.register_forward_pre_hook(self._task_pre_hook))
            self.hook_handles.append(self.agent.register_forward_hook(self._task_forward_hook))
            self.hook_handles.append(self.agent.backbone.model.register_forward_pre_hook(
                self._qwen_pre_hook, with_kwargs=True))
            self.activation_wrappers = sum(
                hasattr(module, "_checkpoint_wrapped_module") for module in self.agent.modules())
            if self.smoke_args.activation_checkpointing and self.activation_wrappers == 0:
                raise RuntimeError("activation checkpointing was requested but no matching Qwen layer was wrapped")

            original_step = self.optimizer.step

            def measured_step(*args, **kwargs):
                task = self.active_task
                if task not in self.grad_steps:
                    raise RuntimeError("optimizer step did not follow a homogeneous task forward")
                self.task_sequence.append(task)
                parameter_count = torch.zeros((), dtype=torch.float64, device=self.ctx.device)
                element_count = torch.zeros((), dtype=torch.float64, device=self.ctx.device)
                squared_norm = torch.zeros((), dtype=torch.float64, device=self.ctx.device)
                candidates = []
                for parameter in self.clip_params:
                    if parameter.grad is None:
                        continue
                    grad = self._local(parameter.grad.detach())
                    nonzero = torch.count_nonzero(grad)
                    parameter_count += nonzero.gt(0).double()
                    element_count += nonzero.double()
                    norm_sq = grad.float().square().sum().double()
                    squared_norm += norm_sq
                    candidates.append((parameter, norm_sq))
                selected_parameter, selected = None, None
                if candidates:
                    index = int(torch.stack([norm for _, norm in candidates]).argmax().item())
                    selected_parameter = candidates[index][0]
                    selected = self._local(selected_parameter.detach()).clone()
                original_step(*args, **kwargs)
                changed = torch.zeros((), dtype=torch.float64, device=self.ctx.device)
                if selected is not None:
                    delta = self._local(selected_parameter.detach()) - selected
                    changed = torch.count_nonzero(delta).double()
                stats = torch.stack((parameter_count, element_count, squared_norm, changed))
                if self.ctx.world_size > 1:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                if self.args.strategy == "ddp":
                    stats /= self.ctx.world_size
                loss = self.last_losses.get(task)
                self.grad_steps[task].append({
                    "loss": float(loss.item()) if loss is not None else None,
                    "nonzero_grad_parameter_tensors": int(stats[0].item()),
                    "nonzero_grad_elements": int(stats[1].item()),
                    "gradient_l2": float(stats[2].sqrt().item()),
                    "parameter_update_nonzero_elements": int(stats[3].item()),
                })

            self.optimizer.step = measured_step

        def build_data(self):
            super().build_data()
            sampler = self.train_loader.batch_sampler
            task_ids = self.train_loader.dataset.sample_tasks
            by_task = {}
            for batch, valid_size in zip(sampler.batches, sampler.valid_batch_sizes):
                if batch:
                    by_task.setdefault(task_ids[batch[0]], []).append((batch, valid_size))
            if not by_task.get("planning") or not by_task.get("vqa"):
                raise RuntimeError("the smoke mixture sampler did not produce both homogeneous tasks")
            planning = by_task["planning"][0]
            vqa = by_task["vqa"][0]
            # Reuse the sampler's real per-rank batches, adding a second planning batch to prove task switching back.
            sampler.batches = [planning[0], vqa[0], planning[0]]
            sampler.valid_batch_sizes = [planning[1], vqa[1], planning[1]]

        def _release_training_weights(self):
            for tensor in [*self.agent.parameters(), *self.agent.buffers()]:
                if not hasattr(tensor, "to_local"):
                    tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
            self.model = self.optimizer = self.lr_scheduler = self.ema = None
            self.clip_params = []
            self.train_loader = self.val_loader = None
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles.clear()
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        def finish(self):
            torch.cuda.synchronize(self.ctx.device)
            self.training_seconds = time.perf_counter() - self.training_started
            self.training_peak_allocated = torch.cuda.max_memory_allocated(self.ctx.device)
            self.training_peak_reserved = torch.cuda.max_memory_reserved(self.ctx.device)
            checkpoint = self.output_dir / "last.ckpt"
            try:
                if not checkpoint.is_file() or self.checkpointer.latest() is None:
                    raise FileNotFoundError("Trainer.fit did not write last.ckpt and a completed resume checkpoint")
                self.ctx.barrier()
                self._release_training_weights()
                from recogdrive.infer import load_agent
                from recogdrive.data.registry import collate, with_worker_transform

                started = time.perf_counter()
                loaded = load_agent(self.cfg, self.ctx.device)
                sample = {"task": "planning", "camera_paths": [("front", str(self.smoke_image))],
                          "history_trajectory": torch.zeros((4, 3), dtype=torch.float32),
                          "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0, 0.0]),
                          "status_feature": torch.zeros(8, dtype=torch.float32)}
                transform = loaded.worker_transform()
                if transform is not None:
                    sample = transform(sample)
                features, targets, tokens = collate([(sample, {"trajectory": torch.zeros((8, 3))},
                                                       "gpu-smoke-reload")])
                self.active_task = "planning"
                self.measuring_inference = True
                self.hook_handles.append(loaded.backbone.model.register_forward_pre_hook(
                    self._qwen_pre_hook, with_kwargs=True))
                with torch.no_grad():
                    prediction = loaded(features, targets, tokens)
                trajectory = prediction["pred_traj"]
                if tuple(trajectory.shape) != (1, 8, 3) or not torch.isfinite(trajectory).all().item():
                    raise RuntimeError("checkpoint reload did not produce a finite (1, 8, 3) trajectory")
                self.prediction_shape = list(trajectory.shape)
                self.inference_tokens = getattr(self, "inference_tokens_seen", None)
                self.inference_seconds = time.perf_counter() - started
                self.inference_ok = True
                self.agent = loaded
            except BaseException:
                self.inference_error = traceback.format_exc()
                self.inference_seconds = None
                self.inference_ok = False

            task_report = {}
            for task in ("planning", "vqa"):
                steps = self.grad_steps[task]
                task_report[task] = {
                    "optimizer_steps": len(steps), "steps": steps,
                    "max_input_tokens": max(self.tokens_seen[task]) if self.tokens_seen[task] else None,
                    "finite_losses": bool(steps) and all(item["loss"] is not None and math.isfinite(item["loss"])
                                                          for item in steps),
                    "has_nonzero_gradients": bool(steps) and all(
                        item["nonzero_grad_parameter_tensors"] > 0 and item["nonzero_grad_elements"] > 0
                        and math.isfinite(item["gradient_l2"]) and item["gradient_l2"] > 0
                        for item in steps),
                    "optimizer_changed_parameter": bool(steps) and all(
                        item["parameter_update_nonzero_elements"] > 0 for item in steps),
                }
            local_ok = (all(item["optimizer_steps"] > 0 and item["finite_losses"]
                            and item["has_nonzero_gradients"] and item["optimizer_changed_parameter"]
                            for item in task_report.values()) and self.inference_ok
                        and self.task_sequence[:3] == ["planning", "vqa", "planning"]
                        and self.inference_tokens is not None
                        and self.inference_tokens <= self.smoke_args.max_tokens
                        and all(item["max_input_tokens"] is not None
                                and item["max_input_tokens"] <= self.smoke_args.max_tokens
                                for item in task_report.values()))
            local_report = {
                "rank": self.ctx.rank, "local_rank": self.ctx.local_rank, "success": bool(local_ok),
                "tasks": task_report, "task_sequence": self.task_sequence, "checkpoint": str(checkpoint),
                "checkpoint_exported": checkpoint.is_file(),
                "resume_checkpoint": str(self.checkpointer.latest()) if self.checkpointer.latest() else None,
                "checkpoint_reloaded_for_planning": self.inference_ok,
                "checkpoint_reload_error": self.inference_error,
                "prediction_shape": self.prediction_shape, "reload_planning_tokens": self.inference_tokens,
                "strategy": self.args.strategy, "precision": self.args.precision,
                "compile": self.smoke_args.compile,
                "activation_checkpointing_wrappers": int(self.activation_wrappers),
                "training_seconds": self.training_seconds, "reload_seconds": self.inference_seconds,
                "training_peak_allocated_bytes": int(self.training_peak_allocated),
                "training_peak_reserved_bytes": int(self.training_peak_reserved),
                "gpu": torch.cuda.get_device_name(self.ctx.device),
                "compute_capability": list(torch.cuda.get_device_capability(self.ctx.device)),
            }
            import torch.distributed as dist
            reports = [None] * self.ctx.world_size
            if self.ctx.world_size > 1:
                dist.all_gather_object(reports, local_report)
            else:
                reports[0] = local_report
            overall_ok = all(row["success"] for row in reports)
            report_write_error = None
            if self.ctx.is_main:
                try:
                    import transformers
                    import peft

                    summary = {
                        "schema_version": 1, "success": bool(overall_ok), "strategy": self.args.strategy,
                        "world_size": self.ctx.world_size,
                        "scope": "real Qwen3-VL-2B, LoRA rank 4, alternating VQA/planning batches; one downscaled image; synthetic planning state/target and one-row VQA JSONL; no official score",
                        "data": {"image": self.image_info,
                                 "synthetic_planning_features_and_target": True,
                                 "synthetic_vqa_jsonl": str(self.output_dir / "vqa_smoke.jsonl"),
                                 "max_tokens": self.smoke_args.max_tokens},
                        "model_path": str(self.smoke_args.model_path),
                        "versions": {"python": sys.version, "torch": torch.__version__,
                                     "cuda_runtime": torch.version.cuda, "transformers": transformers.__version__,
                                     "peft": peft.__version__},
                        "source_hashes": self.hashes, "ranks": reports,
                    }
                    write_json(self.output_dir / "gpu_training_acceptance.json", summary)
                    print(json.dumps({"success": summary["success"], "ranks": reports}, indent=2), flush=True)
                except BaseException:
                    report_write_error = traceback.format_exc()
            flag = torch.tensor([1 if report_write_error is None else 0], dtype=torch.int32,
                                device=self.ctx.device)
            if self.ctx.world_size > 1:
                dist.broadcast(flag, src=0)
            try:
                super().finish()
            finally:
                if not overall_ok or not int(flag.item()):
                    raise RuntimeError("GPU training acceptance failed; inspect gpu_training_acceptance.json "
                                       "or rank output for details" + ("\n" + report_write_error
                                                                        if report_write_error else ""))

    return GPUTrainingSmokeTrainer


def run(args, root, ctx):
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        ctx.close()
        raise RuntimeError("this acceptance script requires a CUDA GPU")
    if args.strategy == "ddp" and ctx.world_size < 2:
        ctx.close()
        raise RuntimeError("the DDP check requires at least two torchrun ranks")
    global OUTPUT_DIR_CLAIMED
    destination = args.output_dir.expanduser().resolve()
    claim_error = None
    if ctx.is_main:
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except BaseException as exc:
            claim_error = "output directory must be new: {} ({})".format(destination, exc)
    claim = [claim_error]
    if ctx.world_size > 1:
        dist.broadcast_object_list(claim, src=0)
    if claim[0]:
        ctx.close()
        raise FileExistsError(claim[0])
    OUTPUT_DIR_CLAIMED = destination

    prepared = [None]
    if ctx.is_main:
        try:
            source, image_kind = choose_source_image(args, destination)
            image_info = prepare_image(source, destination)
            image_info["kind"] = image_kind
            row = {"id": "gpu-smoke-vqa", "image": Path(image_info["path"]).name,
                   "question": "What is visible in the road scene?", "answer": "A road with a vehicle."}
            jsonl_path = destination / "vqa_smoke.jsonl"
            jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            prepared[0] = {"image": image_info, "jsonl": str(jsonl_path), "error": None}
        except BaseException:
            prepared[0] = {"error": traceback.format_exc()}
    if ctx.world_size > 1:
        dist.broadcast_object_list(prepared, src=0)
    if prepared[0]["error"]:
        ctx.close()
        raise RuntimeError(prepared[0]["error"])
    ctx.barrier()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    install_smoke_loader(Path(prepared[0]["image"]["path"]))
    cfg = build_config(args, destination, Path(prepared[0]["image"]["path"]), Path(prepared[0]["jsonl"]))
    hashes = source_hashes(root)
    Trainer = create_trainer_class()
    trainer = Trainer(cfg, args, Path(prepared[0]["image"]["path"]), hashes)
    trainer.image_info = prepared[0]["image"]
    torch.cuda.reset_peak_memory_stats(ctx.device)
    trainer.fit()


def main():
    args = parse_args()
    if args.max_tokens < 64:
        raise ValueError("--max-tokens must be at least 64")
    root = Path(__file__).resolve().parents[2]
    if not args.model_path.is_dir():
        raise FileNotFoundError("Qwen model directory not found: {}".format(args.model_path))

    import torch.distributed as dist
    from recogdrive.distributed import DistributedContext

    group_preexisted = dist.is_initialized()
    try:
        ctx = DistributedContext(args.timeout)
        run(args, root, ctx)
    finally:
        if not group_preexisted and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        if OUTPUT_DIR_CLAIMED is not None and os.environ.get("RANK", "0") == "0":
            report = OUTPUT_DIR_CLAIMED / "gpu_training_acceptance.json"
            if not report.exists():
                try:
                    write_json(report, {"schema_version": 1, "success": False,
                                        "error": "{}: {}".format(type(exc).__name__, exc),
                                        "traceback": traceback.format_exc()})
                except OSError:
                    pass
        print("GPU training acceptance failed: {}".format(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
