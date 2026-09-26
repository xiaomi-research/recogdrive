"""Weighted multi-source training, `data_loader: mixture`:

    mixture:
      - {loader: navsim, weight: 1.0}
      - {loader: waymoe2e, weight: 0.5, overrides: {cache_path: /data/wod_cache}}

Each source is a registered loader built from the config with its overrides merged in. A training epoch draws as many
samples as the sources hold together, each source with probability proportional to its weight (with replacement);
validation concatenates the sources' validation splits. Sources follow the same sample contract and horizon.
"""

import logging

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from recogdrive.data.registry import LOADERS, register

logger = logging.getLogger(__name__)


class MixtureDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        # per-sample weight w / len: each source's share of the draws is proportional to w
        self.sample_weights = torch.cat([torch.full((len(d),), w / len(d), dtype=torch.double)
                                         for d, w in zip(datasets, weights)])


@register("mixture")
def mixture_loader(cfg, agent, split: str):
    sources = OmegaConf.select(cfg, "mixture") or []
    if not sources:
        raise ValueError("data_loader=mixture needs a `mixture` list of {loader, weight, overrides}")
    datasets, weights = [], []
    for source in sources:
        name = str(source["loader"]).lower()
        if name == "mixture" or name not in LOADERS:
            raise KeyError(f"mixture source {name!r} is not a registered loader: {sorted(set(LOADERS) - {'mixture'})}")
        weight = float(source.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"mixture source {name!r} needs a positive weight, got {weight}")
        source_cfg = OmegaConf.merge(cfg, source.get("overrides") or {})
        datasets.append(LOADERS[name](source_cfg, agent, split))
        weights.append(weight)
        logger.info("mixture %s: %s %d samples, weight %g", split, name, len(datasets[-1]), weight)
    return MixtureDataset(datasets, weights) if split == "train" else ConcatDataset(datasets)
