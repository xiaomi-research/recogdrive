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


def export_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Full, unsharded weights on rank 0 (CPU); empty dict elsewhere. Collective."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True),
    )


def training_state(model: nn.Module, optimizer):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_sd, optim_sd = get_state_dict(model, optimizer, options=StateDictOptions(ignore_frozen_params=True))
    return {"model": model_sd, "optim": optim_sd}


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

    def save(self, step: int, model: nn.Module, optimizer, extra: dict) -> None:
        import shutil

        import torch.distributed.checkpoint as dcp

        self.wait()
        path = self.root / f"step_{step:08d}"
        if self.ctx.is_main:
            # Keep the newest finished state until the one being written now is finished too.
            for _, old in self.complete()[:-1]:
                shutil.rmtree(old, ignore_errors=True)
        state = training_state(model, optimizer)

        def finish(_=None):
            if self.ctx.is_main:
                torch.save(extra, path / DONE_FILE)
                logger.info("Resume state written to %s", path)

        if self.async_save:
            self.pending = dcp.async_save(state, checkpoint_id=str(path), process_group=self.ctx.gloo_group())
            self.pending.add_done_callback(finish)
        else:
            dcp.save(state, checkpoint_id=str(path))
            finish()

    def load(self, path: Path, model: nn.Module, optimizer) -> dict:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

        state = training_state(model, optimizer)
        dcp.load(state, checkpoint_id=str(path))
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optim"],
            options=StateDictOptions(ignore_frozen_params=True),
        )
        return torch.load(path / DONE_FILE, map_location="cpu", weights_only=False)

    def wait(self) -> None:
        if self.pending is not None:
            self.pending.result()
            self.pending = None
