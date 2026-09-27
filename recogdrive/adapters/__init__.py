"""Adapters plug a model into the framework through one contract, so data sources and benchmarks never touch it.

A policy adapter is a torch.nn.Module with

    trajectory_sampling    num_poses and interval_length (0.5 s) of the predicted trajectory
    forward(features, targets=None, tokens=None)
                           training: an object with `.loss` (extra metrics as attributes are logged);
                           eval: {"pred_traj": (batch, num_poses, 3)} ego-frame poses (x, y, heading)
    compute_loss(features, targets, predictions)   validation loss from the eval forward
    get_optimizers()       an optimizer, or {"optimizer": ..., "lr_scheduler": ...}; the scheduler advances once per epoch,
                           or every optimizer step when given as {"scheduler": ..., "interval": "step"}. A
                           get_optimizers(total_steps, steps_per_epoch) receives the training length in optimizer steps
    get_target_builders()  target transforms data sources apply (may be empty)

and optionally

    worker_transform(image_augment=None)            per-sample preprocessing in the dataloader workers
    initialize()                                    loads weights before parallelization
    get_sensor_config() / get_feature_builders()    NAVSIM scene features (navsim loader, closed-loop evaluation)

Batches reach forward in their own dtype (fp32 targets and poses keep their precision); under FSDP mixed precision
the model casts its inputs to its parameters' compute dtype itself. Data sources emit the sample contract of
recogdrive.data.registry and benchmarks read only `pred_traj`, so adding either leaves the model alone. adapters/navsim is ReCogDrive as a NAVSIM agent; adapters/template.py is the minimal
adapter to copy for a new model.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

REQUIRED = ("trajectory_sampling", "forward", "compute_loss", "get_optimizers", "get_target_builders")


def check_policy(model) -> None:
    missing = [name for name in REQUIRED if not hasattr(model, name)]
    if missing:
        raise TypeError(f"{type(model).__name__} is not a policy adapter, it lacks {missing} (see recogdrive.adapters)")


@torch.no_grad()
def predict(model, loader, ctx) -> Optional[Dict[str, np.ndarray]]:
    """`pred_traj` of every sample of `loader` (each rank runs its shard), gathered into {token: poses} on rank 0;
    None on the other ranks."""
    from recogdrive.data import DevicePrefetcher

    was_training = model.training
    model.eval()
    mine: Dict[str, np.ndarray] = {}
    for features, _, tokens in DevicePrefetcher(loader, ctx.device):
        mine.update(zip(tokens, model(features)["pred_traj"].float().cpu().numpy()))
    model.train(was_training)
    gathered = [None] * ctx.world_size if ctx.is_main else None
    dist.gather_object(mine, gathered, dst=0, group=ctx.gloo_group())
    return {token: poses for part in gathered for token, poses in part.items()} if ctx.is_main else None
