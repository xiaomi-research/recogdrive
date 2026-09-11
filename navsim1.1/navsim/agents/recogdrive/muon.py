"""Muon for 2D weights; AdamW moments for 1D. Per-parameter state, ZeRO-1 safe."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


def orthogonalize_via_newton_schulz(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    # Keller Jordan Muon coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    matrix = grad.to(dtype=torch.bfloat16 if grad.dtype != torch.float64 else grad.dtype)
    transposed = matrix.size(-2) > matrix.size(-1)
    if transposed:
        matrix = matrix.mT
    matrix = matrix / (matrix.norm() + 1e-7)
    for _ in range(steps):
        gram = matrix @ matrix.mT
        matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix
    if transposed:
        matrix = matrix.mT
    return matrix.to(dtype=grad.dtype)


class Muon(Optimizer):
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
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                if param.ndim >= 2:
                    self._muon_update(param, grad, group)
                else:
                    self._adamw_update(param, grad, group)
        return loss

    def _muon_update(self, param: torch.Tensor, grad: torch.Tensor, group: dict) -> None:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        buffer = state["momentum_buffer"]
        buffer.lerp_(grad, 1.0 - group["momentum"])
        update = grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
        update = orthogonalize_via_newton_schulz(update, steps=group["ns_steps"])
        scale = max(1.0, param.size(-2) / param.size(-1)) ** 0.5
        param.add_(update, alpha=-group["lr"] * scale)

    def _adamw_update(self, param: torch.Tensor, grad: torch.Tensor, group: dict) -> None:
        state = self.state[param]
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
            state["step"] = 0
        state["step"] += 1
        beta1, beta2 = group["adamw_betas"]
        state["exp_avg"].lerp_(grad, 1.0 - beta1)
        state["exp_avg_sq"].lerp_(grad.square(), 1.0 - beta2)
        bias1 = 1.0 - beta1 ** state["step"]
        bias2 = 1.0 - beta2 ** state["step"]
        denom = state["exp_avg_sq"].sqrt().div_(bias2 ** 0.5).add_(group["adamw_eps"])
        param.addcdiv_(state["exp_avg"] / bias1, denom, value=-group["lr"])


def reject_deepspeed_owned_optimizer(ds_cfg: dict) -> None:
    """Muon+DS uses the client optimizer. DS must not spawn AdamW."""
    if ds_cfg and "optimizer" in ds_cfg:
        raise ValueError(
            "DeepSpeed config sets optimizer=...; Muon+DS forbids a DS-owned AdamW. "
            "Remove deepspeed.optimizer so Accelerate keeps navsim.agents.recogdrive.muon.Muon."
        )


def client_optimizer_name(optimizer: object) -> str:
    inner = optimizer
    for _ in range(4):
        nested = getattr(inner, "optimizer", None)
        if nested is None or nested is inner:
            break
        inner = nested
    return type(inner).__name__
