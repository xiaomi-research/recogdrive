from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

from omegaconf import DictConfig, OmegaConf


@dataclass
class TrainArgs:
    strategy: str = "fsdp"                      # fsdp | ddp
    precision: str = "bf16"                     # compute dtype of FSDP-managed parameters: bf16 | fp32
    reshard_after_forward: Union[bool, int] = False
    hsdp_shard_size: Optional[int] = None       # shard within groups of this size, replicate across them
    fsdp_wrap_modules: List[str] = field(default_factory=list)
    replicate_frozen: List[str] = field(default_factory=list)
    prefetch_distance: int = 1
    compile: bool = True
    activation_checkpointing: List[str] = field(default_factory=list)
    ddp_bucket_cap_mb: int = 25
    ddp_find_unused_parameters: bool = False
    ddp_static_graph: bool = False
    log_every: int = 10
    save_every_n_steps: int = 0
    resume: bool = False
    async_save: bool = True
    top_k: int = 5
    grad_accum: int = 1
    grad_clip: float = 0.0
    ema_decay: float = 0.0                      # 0: off; e.g. 0.999 validates and exports the averaged weights too
    max_epochs: int = 10
    max_steps: int = 0
    val_every_n_epochs: int = 1
    skip_validation: bool = False
    seed: Optional[int] = None
    nccl_timeout: int = 1800

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "TrainArgs":
        section = OmegaConf.select(cfg, "train")
        values = OmegaConf.to_container(section, resolve=True) if section is not None else {}
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown train.* keys: {sorted(unknown)}")
        trainer = OmegaConf.select(cfg, "trainer.params")
        trainer = OmegaConf.to_container(trainer, resolve=True) if trainer is not None else {}
        values.setdefault("max_epochs", int(trainer.get("max_epochs", 10)))
        values.setdefault("grad_clip", float(trainer.get("gradient_clip_val") or 0.0))
        values.setdefault("grad_accum", int(trainer.get("accumulate_grad_batches", 1)))
        values.setdefault("val_every_n_epochs", int(trainer.get("check_val_every_n_epoch", 1)))
        for key in ("max_steps", "skip_validation", "seed", "nccl_timeout"):
            value = OmegaConf.select(cfg, key)
            if value is not None:
                values.setdefault(key, value)
        return cls(**values)

    def validate(self, world_size: int) -> None:
        if self.strategy not in ("fsdp", "ddp"):
            raise ValueError(f"train.strategy must be fsdp or ddp, got {self.strategy!r}")
        if self.strategy == "fsdp" and self.precision not in ("bf16", "fp32"):
            raise ValueError(f"fsdp supports precision bf16 or fp32, got {self.precision!r}")
        if self.strategy == "ddp" and self.precision != "fp32":
            # DDP has no per-module precision policy; autocast would also change the frozen VLM's numerics.
            raise ValueError("ddp runs parameters in their own dtype; set train.precision=fp32 or use fsdp")
        if not isinstance(self.reshard_after_forward, bool) and self.reshard_after_forward < 1:
            raise ValueError("train.reshard_after_forward must be a bool or a positive shard group size")
        if self.hsdp_shard_size and world_size % self.hsdp_shard_size:
            raise ValueError(f"world size {world_size} is not divisible by train.hsdp_shard_size {self.hsdp_shard_size}")
        if self.grad_accum < 1:
            raise ValueError("gradient accumulation must be >= 1")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"train.ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.log_every < 1:
            raise ValueError("train.log_every must be >= 1")
