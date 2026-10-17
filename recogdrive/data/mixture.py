"""Weighted multi-source training, `data_loader: mixture`:

    mixture:
      - {loader: navsim, weight: 1.0}
      - {loader: waymoe2e, weight: 0.5, overrides: {cache_path: /data/wod_cache}}

Each source is a registered loader built from the config with its overrides merged in. A training epoch draws about
as many samples as the sources hold together, rounded up to equal per-rank counts, with each source's probability
proportional to its weight (with replacement); validation concatenates the sources' validation splits.
Sources follow the same sample contract and horizon.
"""

import logging
from collections import Counter

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from recogdrive.data.registry import LOADERS, register, with_worker_transform

logger = logging.getLogger(__name__)


class MixtureDataset(ConcatDataset):
    def __init__(self, datasets, weights=None, batch_by_task=False, source_weights=None):
        super().__init__(datasets)
        # per-sample weight w / len: each source's share of the draws is proportional to w
        if weights is not None and any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("mixture training sources must each contain at least one sample")
        if weights is not None and len(weights) != len(datasets):
            raise ValueError("mixture training needs one weight per source")
        self.sample_weights = (torch.cat([torch.full((len(d),), w / len(d), dtype=torch.double)
                                          for d, w in zip(datasets, weights)]) if weights is not None else None)
        self.batch_by_task = batch_by_task
        source_weights = source_weights or weights or [1.0] * len(datasets)
        if len(source_weights) != len(datasets):
            raise ValueError("mixture needs one loss weight per source")
        self.sample_tasks = []
        task_weights = Counter()
        for dataset, weight in zip(datasets, source_weights):
            if not len(dataset):
                continue
            tasks = getattr(dataset, "sample_tasks", None)
            if tasks is None:
                tasks = [getattr(dataset, "task", "planning")] * len(dataset)
            if len(tasks) != len(dataset):
                raise ValueError("a mixture source's sample_tasks must have one task per sample")
            self.sample_tasks.extend(tasks)
            counts = Counter(tasks)
            for task, count in counts.items():
                task_weights[task] += float(weight) * count / len(dataset)
        total_weight = sum(task_weights.values())
        self.task_loss_weights = {task: weight / total_weight for task, weight in task_weights.items()}
        self.worker_transform_applied = True


@register("mixture")
def mixture_loader(cfg, agent, split: str):
    sources = OmegaConf.select(cfg, "mixture") or []
    if not sources:
        raise ValueError("data_loader=mixture needs a `mixture` list of {loader, weight, overrides}")
    datasets, weights = [], []
    augment = OmegaConf.select(cfg, "augment") if split == "train" else None
    for source in sources:
        name = str(source["loader"]).lower()
        if name == "mixture" or name not in LOADERS:
            raise KeyError(f"mixture source {name!r} is not a registered loader: {sorted(set(LOADERS) - {'mixture'})}")
        weight = float(source.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"mixture source {name!r} needs a positive weight, got {weight}")
        source_cfg = OmegaConf.merge(cfg, source.get("overrides") or {})
        datasets.append(with_worker_transform(LOADERS[name](source_cfg, agent, split), agent, augment))
        weights.append(weight)
        logger.info("mixture %s: %s %d samples, weight %g", split, name, len(datasets[-1]), weight)
    batch_by_task = bool(OmegaConf.select(cfg, "mixture_batch_by_task"))
    return MixtureDataset(datasets, weights if split == "train" else None, batch_by_task,
                          source_weights=weights)
