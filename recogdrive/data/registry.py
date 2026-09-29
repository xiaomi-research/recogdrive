"""Dataset registry and the sample contract every data source follows.

A loader returns a torch Dataset whose items are (features: dict, targets: dict, token: str):

    features["history_trajectory"]    (4, 3) ego poses (x, y, heading) at t = -1.5, -1.0, -0.5, 0 s
    features["high_command_one_hot"]  (4,)   left, straight, right, unknown
    features["status_feature"]        (8,)   command one-hot, velocity (vx, vy), acceleration (ax, ay)
    features["camera_paths"]          [(view, image path)], every camera the source has, front first; views are
                                      front, front_left, front_right, left, right, back_left, back_right, back
                                      (NAVSIM caches carry the front view only, as `image_path_tensor`)
    targets["trajectory"]             (horizon(model), 3) future ego poses at 0.5 s spacing

All poses are in the current ego frame (x forward, y left, rear axle). The token names the sample; RL rewards and
benchmarks look samples up by it. A dataset may define `eval_info(token)` with benchmark-only data (object boxes,
rater trajectories) that never enters a batch. New sources register with `@register(name)`, from their own module
listed in the `plugins` config if they live outside this package; the trainer does not change.

Batching is generic: equal shapes stack, variable-length sequences pad with zeros, VLM image tiles (`pixel_values`)
concatenate along dim 0, anything that is not a tensor becomes a list, and keys only some samples of a batch have
(source-specific extras in a mixture) are dropped.

A model may define `worker_transform(image_augment=None)` returning a per-sample transform of the features dict (or
None); it runs in the dataloader workers, so model-specific preprocessing stays off the training process and out of
the datasets. The training split gets the `augment` config: ego-state dropout before that transform, photometric
image augmentation inside it.

Host-side metadata (prompt text, file paths, tile counts) must stay Python objects: every batch
tensor goes to the GPU before forward, and reading a GPU tensor from Python stalls the device.
"""

import importlib
import logging
from typing import Any, Callable, Dict

import torch
import torch.nn.utils.rnn as rnn_utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

logger = logging.getLogger(__name__)

POSE_INTERVAL_S = 0.5
LoaderFn = Callable[[Any, Any, str], Any]
LOADERS: Dict[str, LoaderFn] = {}


def register(name: str) -> Callable[[LoaderFn], LoaderFn]:
    def deco(fn: LoaderFn) -> LoaderFn:
        LOADERS[name] = fn
        return fn
    return deco


def import_plugins(cfg: Any) -> None:
    """Imports the modules listed in `plugins`, which register more loaders or evaluators."""
    for module in OmegaConf.select(cfg, "plugins") or []:
        importlib.import_module(str(module))


def horizon(model: Any) -> int:
    """Number of 0.5 s target poses the model predicts (its trajectory_sampling)."""
    sampling = getattr(model, "trajectory_sampling", None)
    if sampling is None:
        raise AttributeError(f"{type(model).__name__} must expose trajectory_sampling (num_poses, interval_length)")
    if abs(float(sampling.interval_length) - POSE_INTERVAL_S) > 1e-6:
        raise ValueError(f"data sources provide poses every {POSE_INTERVAL_S} s, the model samples every "
                         f"{sampling.interval_length} s")
    return int(sampling.num_poses)


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

    @property
    def sample_weights(self):
        return getattr(self.dataset, "sample_weights", None)

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
    augment = OmegaConf.select(cfg, "augment") if split == "train" else None
    return with_worker_transform(fn(cfg, agent, split), agent, augment)


def with_worker_transform(dataset, agent, augment=None):
    from torchvision.transforms import Compose

    from recogdrive.data.augment import ego_dropout, image_augment

    images = image_augment(augment)
    make = getattr(agent, "worker_transform", None)
    transform = None if make is None else make(image_augment=images) if images is not None else make()
    if images is not None and transform is None:
        logger.warning("augment: the model reads no images in the dataloader, image augmentation is skipped")
    steps = [step for step in (ego_dropout(augment), transform) if step is not None]
    if not steps:
        return dataset
    return TransformedDataset(dataset, steps[0] if len(steps) == 1 else Compose(steps))


def dataset_of(dataset):
    """The source dataset under the worker transform."""
    return dataset.dataset if isinstance(dataset, TransformedDataset) else dataset


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
    keys = [key for key in dicts[0] if all(key in d for d in dicts[1:])]
    return {key: collate_field(key, [d[key] for d in dicts]) for key in keys}


def collate(batch):
    columns = list(zip(*batch))
    features_list, targets_list = columns[0], columns[1]
    tokens = columns[2] if len(columns) > 2 else [None] * len(batch)
    features = collate_dicts(features_list)
    if "pixel_values" in features and "num_patches" not in features:
        features["num_patches"] = [f["pixel_values"].shape[0] for f in features_list]
    return features, collate_dicts(targets_list), list(tokens)


def deal(indices, costs, world: int, per_rank: int):
    """Splits one global batch among the ranks, per_rank samples each: costliest first, each to the least loaded
    rank with room (ties keep the draw order)."""
    loads, bins = [0.0] * world, [[] for _ in range(world)]
    for index in sorted(indices, key=lambda i: -float(costs[i])):
        rank = min((r for r in range(world) if len(bins[r]) < per_rank), key=lambda r: loads[r])
        bins[rank].append(index)
        loads[rank] += float(costs[index])
    return bins


class WeightedDistributedSampler(Sampler):
    """Draws len(weights) indices with replacement, proportional to the weights, identically on every rank from
    (seed, epoch), and yields this rank's share. Given per-sample costs, each global batch (batch_size per rank) is
    dealt so the ranks of a step carry about the same load; a step waits for its slowest rank."""

    def __init__(self, weights, rank: int, world: int, seed: int = 0, costs=None, batch_size: int = 1):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.rank, self.world, self.seed = rank, world, seed
        self.per_rank = len(self.weights) // world
        self.costs, self.batch_size = costs, batch_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        drawn = torch.multinomial(self.weights, self.per_rank * self.world, replacement=True, generator=generator)
        drawn = drawn.tolist()
        if self.costs is None or self.world == 1:
            return iter(drawn[self.rank::self.world])
        step = self.batch_size * self.world
        whole = len(drawn) // step * step
        mine = [index for start in range(0, whole, step)
                for index in deal(drawn[start:start + step], self.costs, self.world, self.batch_size)[self.rank]]
        return iter(mine + drawn[whole + self.rank::self.world])

    def __len__(self) -> int:
        return self.per_rank


def tile_costs(dataset):
    """Per-sample cost of a mixture for balancing ranks: the image tiles the model's worker transform makes of each
    source's first sample (the vision encoder is what varies between samples; a source's images share a size).
    None for one source, samples without tiles, or sources with the same tile count."""
    sizes = [len(d) for d in getattr(dataset_of(dataset), "datasets", [])]
    if len(sizes) < 2:
        return None
    tiles, start = [], 0
    for size in sizes:
        pixels = dataset[start][0].get("pixel_values")
        if pixels is None:
            return None
        tiles.append(pixels.shape[0])
        start += size
    if len(set(tiles)) == 1:
        return None
    logger.info("mixture: dealing each step's samples to balance the ranks' image tiles (per source: %s)", tiles)
    return torch.repeat_interleave(torch.tensor(tiles, dtype=torch.double), torch.tensor(sizes)).tolist()


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
    weights = getattr(dataset, "sample_weights", None)
    if train and weights is not None:
        costs = tile_costs(dataset) if world > 1 else None
        kwargs["sampler"] = WeightedDistributedSampler(weights, rank, world, costs=costs,
                                                       batch_size=int(kwargs.get("batch_size") or 1))
    elif world > 1:
        kwargs["sampler"] = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=train, drop_last=drop_last)
    else:
        kwargs["shuffle"] = train
    return DataLoader(dataset, collate_fn=collate, drop_last=drop_last, **kwargs)
