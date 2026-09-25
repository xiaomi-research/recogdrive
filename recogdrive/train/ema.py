"""Exponential moving average of the trainable parameters, kept in their training layout.

Under FSDP2 the average is stored as DTensors with the parameters' sharding: averaging a shard gives the
shard of the average, so an update is one fused lerp per rank with no communication, and resume state is
saved sharded by DCP like the optimizer's.
"""

import contextlib
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim.swa_utils import get_ema_multi_avg_fn

try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:  # torch without FSDP2: no module holds gathered parameters
    FSDPModule = ()


def local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.model = model
        self.decay = decay
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.names: List[str] = [name for name, _ in named]
        self.params: List[torch.Tensor] = [p for _, p in named]
        with torch.no_grad():
            self.shadow: List[torch.Tensor] = [p.detach().clone() for p in self.params]
        self.average = get_ema_multi_avg_fn(decay)

    def update(self) -> None:
        self.average([local(s) for s in self.shadow], [local(p) for p in self.params], None)

    @torch.no_grad()
    def reset(self) -> None:
        for p, s in zip(self.params, self.shadow):
            local(s).copy_(local(p))

    @torch.no_grad()
    def swap(self) -> None:
        # FSDP2 keeps gathered parameters after a forward without backward (reshard_after_forward=False);
        # drop them so the next forward gathers the swapped shards.
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        for p, s in zip(self.params, self.shadow):
            p_local, s_local = local(p), local(s)
            held = p_local.clone()
            p_local.copy_(s_local)
            s_local.copy_(held)

    @contextlib.contextmanager
    def swapped(self):
        """The model holds the averaged weights inside the block."""
        self.swap()
        try:
            yield
        finally:
            self.swap()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return dict(zip(self.names, self.shadow))
