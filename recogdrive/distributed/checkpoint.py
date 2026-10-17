"""Model export for evaluators and sharded training state for resume.

Both skip frozen parameters: they are re-created from the pretrained weights when the model
is built, and gathering a frozen VLM every save costs several GB of copies for nothing.
"""

import logging
import os
import re
import tempfile
import threading
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
    """Sharded training state; async completion is confirmed and published by foreground wait()."""

    def __init__(self, root: Path, ctx, async_save: bool = True):
        self.root = Path(root)
        self.ctx = ctx
        self.async_save = async_save
        self.pending = None
        self.callback_error = None
        self.process_group = None
        if self.world_size > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise RuntimeError("a dedicated checkpoint group requires an initialized process group")
            self.process_group = dist.new_group(backend="gloo")
        else:
            self.process_group = ctx.gloo_group()

    @property
    def rank(self) -> int:
        return int(getattr(self.ctx, "rank", 0))

    @property
    def world_size(self) -> int:
        return int(getattr(self.ctx, "world_size", 1))

    def barrier(self) -> None:
        if self.world_size > 1:
            import torch.distributed as dist

            dist.barrier(group=self.process_group)

    @staticmethod
    def _atomic_save(value, path: Path) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _finish_async(self, future, event: threading.Event) -> None:
        try:
            future.result()
        except BaseException as exc:
            self.callback_error = exc
        finally:
            event.set()

    def _publish_done(self, path: Path, extra: dict) -> None:
        error = None
        if getattr(self.ctx, "is_main", True):
            try:
                self._atomic_save(extra, path / DONE_FILE)
            except BaseException as exc:
                error = exc
        if self.world_size > 1:
            import torch.distributed as dist

            succeeded = torch.tensor([error is None], dtype=torch.int32)
            dist.all_reduce(succeeded, op=dist.ReduceOp.MIN, group=self.process_group)
            if not succeeded.item():
                raise error or RuntimeError("checkpoint completion metadata failed on another rank")
        elif error is not None:
            raise error
        if getattr(self.ctx, "is_main", True):
            logger.info("Resume state written to %s", path)

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
        self.root.mkdir(parents=True, exist_ok=True)
        if path.exists() and (path / DONE_FILE).is_file():
            raise FileExistsError(f"refusing to replace completed resume state {path}")
        if getattr(self.ctx, "is_main", True):
            # Keep the newest finished state until the one being written now is finished too.
            for _, old in self.complete()[:-1]:
                shutil.rmtree(old, ignore_errors=True)
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
        self.barrier()
        extra = {**extra, "ema": ema is not None}
        rank_state = extra.pop("rank_state", None)
        if rank_state is not None:
            self._atomic_save(rank_state, path / f"rank_state_{self.rank:05d}.pt")
            self.barrier()
        state = training_state(model, optimizer, ema)

        # DCP only exchanges plan metadata between ranks; gloo does that on the CPU (object gathers over NCCL
        # have crashed with SIGSEGV here).
        if self.async_save:
            future = dcp.async_save(state, checkpoint_id=str(path), process_group=self.process_group)
            event = threading.Event()
            self.pending = (future, event, path, extra)
            self.callback_error = None
            future.add_done_callback(lambda done: self._finish_async(done, event))
        else:
            dcp.save(state, checkpoint_id=str(path), process_group=self.process_group)
            self.barrier()
            self._publish_done(path, extra)

    def load(self, path: Path, model: nn.Module, optimizer, ema=None) -> dict:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_state_dict

        extra = torch.load(path / DONE_FILE, map_location="cpu", weights_only=False)
        rank_state_path = path / f"rank_state_{self.rank:05d}.pt"
        if rank_state_path.is_file():
            extra["rank_state"] = torch.load(rank_state_path, map_location="cpu", weights_only=False)
        saved_ema = ema if extra.get("ema") else None
        state = training_state(model, optimizer, saved_ema)
        dcp.load(state, checkpoint_id=str(path), process_group=self.process_group)
        # Frozen parameters are not in the resume state; they load their own current values, since set_state_dict
        # (torch 2.6) looks every parameter up when the model is wrapped in DDP.
        for key, value in get_model_state_dict(model).items():
            state["model"].setdefault(key, value)
        # anything else missing is a real mismatch
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
            future, event, path, extra = self.pending
            local_error = None
            try:
                future.result()
            except BaseException as exc:
                local_error = exc
            event.wait()
            local_error = local_error or self.callback_error
            self.pending = None
            self.callback_error = None
            if self.world_size > 1:
                import torch.distributed as dist

                succeeded = torch.tensor([local_error is None], dtype=torch.int32)
                dist.all_reduce(succeeded, op=dist.ReduceOp.MIN, group=self.process_group)
                if not succeeded.item():
                    raise local_error or RuntimeError("asynchronous checkpoint save failed on another rank")
            elif local_error is not None:
                raise local_error

            self._publish_done(path, extra)
