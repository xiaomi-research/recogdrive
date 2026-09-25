"""Dataset registry and the batch contract every data pipeline follows.

A loader returns a torch Dataset whose items are (features: dict, targets: dict[, token]).
The token is needed by trainers that look samples up elsewhere, e.g. RL rewards from a metric cache.
Batching is generic: equal shapes stack, variable-length sequences pad with zeros, VLM image
tiles (`pixel_values`) concatenate along dim 0, and anything that is not a tensor becomes a list.

A model may define `worker_transform()` returning a per-sample transform of the features dict
(or None); it runs in the dataloader workers, so model-specific preprocessing stays off the
training process and out of the datasets.

Host-side metadata (prompt text, file paths, tile counts) must stay Python objects: every batch
tensor goes to the GPU before forward, and reading a GPU tensor from Python stalls the device.
"""

from typing import Any, Callable, Dict

import torch
import torch.nn.utils.rnn as rnn_utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler

LoaderFn = Callable[[Any, Any, str], Any]
LOADERS: Dict[str, LoaderFn] = {}


def register(name: str) -> Callable[[LoaderFn], LoaderFn]:
    def deco(fn: LoaderFn) -> LoaderFn:
        LOADERS[name] = fn
        return fn
    return deco


def loader_name(cfg: Any) -> str:
    name = OmegaConf.select(cfg, "data_loader") if isinstance(cfg, DictConfig) else getattr(cfg, "data_loader", None)
    if name:
        return str(name).lower()
    if getattr(cfg, "use_cache_without_dataset", False) and getattr(cfg, "waymoe2e", False):
        return "waymoe2e"
    return "navsim"


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform: Callable[[dict], dict]):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        features, *rest = self.dataset[index]
        return (self.transform(features), *rest)


def build_split(cfg: Any, agent: Any, split: str):
    name = loader_name(cfg)
    fn = LOADERS.get(name)
    if fn is None:
        raise KeyError(
            f"Unknown data loader {name!r}. Registered: {sorted(LOADERS)}. "
            "Register a loader for a new dataset; the trainer does not change."
        )
    dataset = fn(cfg, agent, split)
    transform = agent.worker_transform() if hasattr(agent, "worker_transform") else None
    return dataset if transform is None else TransformedDataset(dataset, transform)


def collate_field(key: str, values):
    first = values[0]
    if key == "num_patches":
        return [int(n) for v in values for n in (v.reshape(-1).tolist() if torch.is_tensor(v) else [v])]
    if not torch.is_tensor(first):
        return list(values)
    if key == "pixel_values":
        return torch.cat(values, dim=0)
    if all(v.shape == first.shape for v in values):
        return torch.stack(values, dim=0)
    if first.dim() >= 1 and all(v.dim() == first.dim() and v.shape[1:] == first.shape[1:] for v in values):
        return rnn_utils.pad_sequence(list(values), batch_first=True, padding_value=0)
    raise ValueError(f"cannot batch {key!r} with shapes {[tuple(v.shape) for v in values]}")


def collate_dicts(dicts):
    return {key: collate_field(key, [d[key] for d in dicts]) for key in dicts[0]}


def collate(batch):
    columns = list(zip(*batch))
    features_list, targets_list = columns[0], columns[1]
    tokens = columns[2] if len(columns) > 2 else [None] * len(batch)
    features = collate_dicts(features_list)
    if "pixel_values" in features and "num_patches" not in features:
        features["num_patches"] = [f["pixel_values"].shape[0] for f in features_list]
    return features, collate_dicts(targets_list), list(tokens)


def dataloader_kwargs(params) -> dict:
    kwargs = OmegaConf.to_container(params, resolve=True) if isinstance(params, DictConfig) else dict(params)
    kwargs.pop("shuffle", None)
    if int(kwargs.get("num_workers") or 0) > 0:
        kwargs.setdefault("persistent_workers", True)
    else:
        kwargs.pop("prefetch_factor", None)
        kwargs.pop("persistent_workers", None)
    return kwargs


def make_loader(dataset, params, train: bool, rank: int, world: int) -> DataLoader:
    kwargs = dataloader_kwargs(params)
    # Fixed batch shapes keep compiled graphs and cuDNN plans valid for the whole epoch.
    drop_last = bool(kwargs.pop("drop_last", train))
    if world > 1:
        kwargs["sampler"] = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=train, drop_last=drop_last)
    else:
        kwargs["shuffle"] = train
    return DataLoader(dataset, collate_fn=collate, drop_last=drop_last, **kwargs)
