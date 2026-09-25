"""Model export for evaluators and sharded training state for resume.

Both skip frozen parameters: they are re-created from the pretrained weights when the model
is built, and gathering a frozen VLM every save costs several GB of copies for nothing.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

STEP_DIR = re.compile(r"^step_(\d+)$")
DONE_FILE = "trainer_state.pt"


def canonical(name: str) -> str:
    """Parameter name without DDP and activation-checkpoint wrapper prefixes, as in DCP state dicts."""
    name = name.replace("_checkpoint_wrapped_module.", "")
    return name[len("module."):] if name.startswith("module.") else name


def export_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Full, unsharded weights on rank 0 (CPU); empty dict elsewhere. Collective."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True),
    )


def training_state(model: nn.Module, optimizer, ema=None):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_sd, optim_sd = get_state_dict(model, optimizer, options=StateDictOptions(ignore_frozen_params=True))
    state = {"model": model_sd, "optim": optim_sd}
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


class ResumeCheckpointer:
    """Sharded model + optimizer state via torch.distributed.checkpoint, written in the background."""

    def __init__(self, root: Path, ctx, async_save: bool = True):
        self.root = Path(root)
        self.ctx = ctx
        self.async_save = async_save
        self.pending = None

    def complete(self):
        if not self.root.is_dir():
            return []
        return sorted(
            (int(m.group(1)), p)
            for p in self.root.iterdir()
            if (m := STEP_DIR.match(p.name)) and (p / DONE_FILE).is_file()
        )

    def latest(self) -> Optional[Path]:
        done = self.complete()
        return done[-1][1] if done else None

    def save(self, step: int, model: nn.Module, optimizer, extra: dict, ema=None) -> None:
        import shutil

        import torch.distributed.checkpoint as dcp

        self.wait()
        path = self.root / f"step_{step:08d}"
        if self.ctx.is_main:
            # Keep the newest finished state until the one being written now is finished too.
            for _, old in self.complete()[:-1]:
                shutil.rmtree(old, ignore_errors=True)
        extra = {**extra, "ema": ema is not None}
        state = training_state(model, optimizer, ema)

        def finish(_=None):
            if self.ctx.is_main:
                torch.save(extra, path / DONE_FILE)
                logger.info("Resume state written to %s", path)

        # DCP only exchanges plan metadata between ranks; gloo does that on the CPU (object gathers over NCCL
        # have crashed with SIGSEGV here).
        if self.async_save:
            self.pending = dcp.async_save(state, checkpoint_id=str(path), process_group=self.ctx.gloo_group())
            self.pending.add_done_callback(finish)
        else:
            dcp.save(state, checkpoint_id=str(path), process_group=self.ctx.gloo_group())
            finish()

    def load(self, path: Path, model: nn.Module, optimizer, ema=None) -> dict:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

        extra = torch.load(path / DONE_FILE, map_location="cpu", weights_only=False)
        saved_ema = ema if extra.get("ema") else None
        state = training_state(model, optimizer, saved_ema)
        dcp.load(state, checkpoint_id=str(path), process_group=self.ctx.gloo_group())
        # Frozen parameters are not in the resume state; anything else missing is a real mismatch.
        result = set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optim"],
            options=StateDictOptions(ignore_frozen_params=True, strict=False),
        )
        frozen = {canonical(name) for name, p in model.named_parameters() if not p.requires_grad}
        missing = [key for key in result.missing_keys if canonical(key) not in frozen]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"resume state {path} does not match the model: missing {missing[:8]}, "
                               f"unexpected {list(result.unexpected_keys)[:8]}")
        if ema is not None and saved_ema is None:
            logger.info("resume state %s has no EMA; the average restarts from the loaded weights", path)
            ema.reset()
        return extra

    def wait(self) -> None:
        if self.pending is not None:
            self.pending.result()
            self.pending = None
