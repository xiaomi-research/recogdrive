import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


class DistributedContext:
    """Process group, device and mesh for one training process.

    The process group is always initialized, also for a single GPU, so one code path
    (and one set of numerics) covers 1 to N GPUs.
    """

    def __init__(self, timeout_s: int = 1800):
        # Both must be set before the first CUDA allocation and the first NCCL call.
        os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.is_main = self.rank == 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        else:
            self.device = torch.device("cpu")
        if not dist.is_initialized():
            kwargs = {"timeout": timedelta(seconds=timeout_s)}
            if self.device.type == "cuda":
                kwargs["backend"] = "nccl"
                kwargs["device_id"] = self.device
            else:
                kwargs["backend"] = "gloo"
            dist.init_process_group(**kwargs)
        self.gloo = None

    def build_mesh(self, hsdp_shard_size: Optional[int]):
        from torch.distributed.device_mesh import init_device_mesh

        device_type = self.device.type
        if not hsdp_shard_size or hsdp_shard_size == self.world_size:
            return init_device_mesh(device_type, (self.world_size,), mesh_dim_names=("fsdp",))
        if self.world_size % hsdp_shard_size:
            raise ValueError(f"world size {self.world_size} is not divisible by hsdp_shard_size {hsdp_shard_size}")
        return init_device_mesh(
            device_type,
            (self.world_size // hsdp_shard_size, hsdp_shard_size),
            mesh_dim_names=("replicate", "shard"),
        )

    def gloo_group(self):
        """CPU group for background checkpoint writes, so they never queue behind NCCL work. Collective on first call."""
        if self.gloo is None:
            self.gloo = dist.new_group(backend="gloo")
        return self.gloo

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach().clone()
        if self.world_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            out /= self.world_size
        return out

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach().clone()
        if self.world_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def barrier(self) -> None:
        if dist.is_initialized() and self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if dist.is_initialized():
            self.barrier()
            dist.destroy_process_group()
