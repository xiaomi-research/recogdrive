"""Muon for 2D weights (higher-rank weights are flattened to rows x rest); AdamW for 1D. Per-parameter state.

Same-shape matrices are orthogonalized as one batch. FSDP2 parameters are DTensor shards: weight decay,
momentum and AdamW run on the local shards (optimizer state stays DTensor for sharded checkpoints). For the
whole-matrix Newton-Schulz step, the shards of all matrices are packed into one buffer and all-gathered
once; each rank orthogonalizes the matrices it owns, and one reduce-scatter returns every rank its rows of
every result. Newton-Schulz runs in bf16, so exchanging bf16 is exact: the result matches one process.

Parameters kept in a lower precision (bf16 working weights under DDP) get an fp32 master copy in the optimizer
state, as in DMuon: momentum, weight decay and updates are fp32 on the master, which is then written back to the
parameter. fp32 parameters are updated in place with no extra copy.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.adamw import adamw

try:
    from torch.distributed.tensor import DTensor, Replicate, Shard
except ImportError:  # torch without public DTensor: no FSDP2 parameters either
    DTensor = Replicate = Shard = None

EXCHANGE_ELEMENTS = 1 << 26  # full matrix elements per gather/scatter round (128 MiB in bf16)


def orthogonalize_via_newton_schulz(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz orthogonalization of a matrix, or of a batch of matrices in the last two dims."""
    # Keller Jordan Muon coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    matrix = grad.to(dtype=torch.bfloat16 if grad.dtype != torch.float64 else grad.dtype)
    transposed = matrix.size(-2) > matrix.size(-1)
    if transposed:
        matrix = matrix.mT
    matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = matrix @ matrix.mT
        matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix
    if transposed:
        matrix = matrix.mT
    return matrix.to(dtype=grad.dtype)


def orthogonalize_all(matrices: List[torch.Tensor], steps: int) -> List[torch.Tensor]:
    """Newton-Schulz with same-shape matrices stacked into one batch."""
    out: List[Optional[torch.Tensor]] = [None] * len(matrices)
    by_shape: Dict[Tuple[int, ...], List[int]] = {}
    for index, matrix in enumerate(matrices):
        by_shape.setdefault(tuple(matrix.shape), []).append(index)
    for indices in by_shape.values():
        results = orthogonalize_via_newton_schulz(torch.stack([matrices[i] for i in indices]), steps)
        for i, result in zip(indices, results):
            out[i] = result
    return out


def is_sharded(tensor: torch.Tensor) -> bool:
    return DTensor is not None and isinstance(tensor, DTensor)


def local(tensor: torch.Tensor) -> torch.Tensor:
    """The rank's own shard of a DTensor (a view), or the tensor itself."""
    return tensor.to_local() if is_sharded(tensor) else tensor


def update_scale(param: torch.Tensor) -> float:
    rows = param.shape[0]
    return max(1.0, rows / (param.numel() // rows)) ** 0.5


class Muon(Optimizer):
    fp32_master = True  # updates of bf16 parameters go through fp32 master copies

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            self.adamw_step([p for p in params if p.ndim < 2], group)
            matrices = [p for p in params if p.ndim >= 2]
            decay = 1.0 - group["lr"] * group["weight_decay"]
            for param in matrices:
                if decay != 1.0:
                    self.master(param).mul_(decay)
            updates = [self.momentum_update(param, group) for param in matrices]
            plain = [(p, u) for p, u in zip(matrices, updates) if not is_sharded(p)]
            sharded = [(p, u) for p, u in zip(matrices, updates) if is_sharded(p)]
            flat = [u.reshape(u.shape[0], -1) for _, u in plain]
            for (param, _), result in zip(plain, orthogonalize_all(flat, group["ns_steps"])):
                self.master(param).add_(result.reshape(param.shape), alpha=-group["lr"] * update_scale(param))
            self.sharded_muon_step(sharded, group)
            for param in params:
                self.publish(param)
        return loss

    def master(self, param: torch.Tensor) -> torch.Tensor:
        """The fp32 tensor updates land on: the parameter's local shard, or its fp32 master copy."""
        if param.dtype == torch.float32:
            return local(param)
        state = self.state[param]
        if "master" not in state:
            state["master"] = param.detach().float()  # a DTensor stays a DTensor, for sharded checkpoints
        return local(state["master"])

    def load_state_dict(self, state_dict: dict) -> None:
        """torch casts floating-point state to each parameter's dtype; the fp32 masters and moments of bf16
        parameters are put back as saved."""
        saved_ids = [i for group in state_dict["param_groups"] for i in group["params"]]
        wide = {i: {k: v for k, v in state.items() if k != "step" and torch.is_tensor(v) and v.dtype == torch.float32}
                for i, state in state_dict["state"].items()}
        super().load_state_dict(state_dict)
        for i, param in zip(saved_ids, [p for group in self.param_groups for p in group["params"]]):
            if param.dtype != torch.float32:
                for key, value in wide.get(i, {}).items():
                    self.state[param][key] = value.to(device=param.device)

    def publish(self, param: torch.Tensor) -> None:
        master = self.state[param].get("master")
        if master is not None:
            local(param).copy_(local(master))

    def momentum_update(self, param: torch.Tensor, group: dict) -> torch.Tensor:
        """Momentum on the local shard; returns the (Nesterov) update as a local tensor."""
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param.grad, dtype=torch.float32)
        grad, buffer = local(param.grad).float(), local(state["momentum_buffer"])
        buffer.lerp_(grad, 1.0 - group["momentum"])
        return grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer

    def sharded_muon_step(self, items: List[Tuple[torch.Tensor, torch.Tensor]], group: dict) -> None:
        if not items:
            return
        mesh = items[0][0].device_mesh
        for param, _ in items:
            placements = param.placements
            if placements[-1] != Shard(0) or not all(isinstance(p, Replicate) for p in placements[:-1]):
                raise NotImplementedError(f"Muon expects FSDP2/HSDP row sharding, got {placements}")
        shard_group = mesh.get_group(mesh.ndim - 1)
        rounds: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[]]
        size = 0
        for item in items:
            if rounds[-1] and size + item[0].numel() > EXCHANGE_ELEMENTS:
                rounds.append([])
                size = 0
            rounds[-1].append(item)
            size += item[0].numel()
        for batch in rounds:
            self.exchange_and_orthogonalize(batch, group, shard_group)

    def exchange_and_orthogonalize(self, items, group: dict, shard_group) -> None:
        rank, world = dist.get_rank(shard_group), dist.get_world_size(shard_group)
        device = local(items[0][0]).device
        rows = [param.shape[0] for param, _ in items]
        cols = [param.numel() // param.shape[0] for param, _ in items]
        chunks = [math.ceil(r / world) for r in rows]  # FSDP2 splits rows like torch.chunk
        widths = [chunk * col for chunk, col in zip(chunks, cols)]
        offsets = [0]
        for width in widths:
            offsets.append(offsets[-1] + width)
        total = offsets[-1]

        send = torch.zeros(total, dtype=torch.bfloat16, device=device)
        for i, (_, update) in enumerate(items):
            send[offsets[i]:offsets[i] + update.numel()] = update.reshape(-1)
        gathered = torch.empty(world * total, dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(gathered, send, group=shard_group)
        gathered = gathered.view(world, total)

        owned = [i for i in range(len(items)) if i % world == rank]
        fulls = [gathered[:, offsets[i]:offsets[i + 1]].reshape(world * chunks[i], cols[i])[:rows[i]] for i in owned]
        scatter = torch.zeros(world, total, dtype=torch.bfloat16, device=device)
        for i, result in zip(owned, orthogonalize_all(fulls, group["ns_steps"])):
            padded = torch.zeros(world * chunks[i], cols[i], dtype=torch.bfloat16, device=device)
            padded[:rows[i]] = result
            scatter[:, offsets[i]:offsets[i + 1]] = padded.view(world, widths[i])
        mine = torch.empty(total, dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(mine, scatter.view(-1), op=dist.ReduceOp.SUM, group=shard_group)

        for i, (param, _) in enumerate(items):
            target = self.master(param)
            result = mine[offsets[i]:offsets[i] + target.numel()].view_as(target)
            target.add_(result, alpha=-group["lr"] * update_scale(param))

    def adamw_step(self, params: List[torch.Tensor], group: dict) -> None:
        if not params:
            return
        grads, exp_avgs, exp_avg_sqs, steps = [], [], [], []
        for param in params:
            state = self.state[param]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param.grad, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param.grad, dtype=torch.float32)
            if not torch.is_tensor(state.get("step")):  # also converts int steps of older checkpoints
                state["step"] = torch.tensor(float(state.get("step", 0)), device=local(param).device)
            grads.append(local(param.grad).float())
            exp_avgs.append(local(state["exp_avg"]))
            exp_avg_sqs.append(local(state["exp_avg_sq"]))
            steps.append(state["step"])
        beta1, beta2 = group["adamw_betas"]
        adamw(
            [self.master(p) for p in params], grads, exp_avgs, exp_avg_sqs, [], steps,
            fused=grads[0].is_cuda, amsgrad=False, beta1=beta1, beta2=beta2, lr=group["lr"],
            weight_decay=group["weight_decay"], eps=group["adamw_eps"], maximize=False,
        )
