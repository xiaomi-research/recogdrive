import inspect
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from recogdrive.data.registry import horizon, register


def scene_loader(cfg, agent, scene_filter):
    from navsim.common.dataloader import SceneLoader

    kwargs = {}
    # NAVSIM 2.0 two-stage splits (navhard / navsafe) add synthetic second-stage scenes
    if getattr(scene_filter, "include_synthetic_scenes", False) and "synthetic_scenes_path" in inspect.signature(SceneLoader).parameters:
        kwargs = {"synthetic_scenes_path": Path(OmegaConf.select(cfg, "synthetic_scenes_path")),
                  "synthetic_sensor_path": Path(OmegaConf.select(cfg, "synthetic_sensor_path"))}
    return SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=getattr(agent, "load_image_path", False),
        **kwargs,
    )


def navsim_scene(cfg, agent, split: str):
    from hydra.utils import instantiate

    from navsim.planning.training.dataset import Dataset as SceneDataset

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    logs = cfg.train_logs if split == "train" else cfg.val_logs
    if scene_filter.log_names is not None:
        scene_filter.log_names = [name for name in scene_filter.log_names if name in logs]
    else:
        scene_filter.log_names = logs
    return SceneDataset(
        scene_loader=scene_loader(cfg, agent, scene_filter),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )


class AgentInputs(Dataset):
    """Features of each scene's AgentInput, with its token and no targets: closed-loop evaluation input."""

    def __init__(self, loader, feature_builders):
        self.loader = loader
        self.builders = feature_builders
        self.tokens = list(loader.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        agent_input = self.loader.get_agent_input_from_token(token)
        features = {}
        for builder in self.builders:
            features.update(builder.compute_features(agent_input))
        return features, {}, token


def agent_inputs(cfg, agent) -> AgentInputs:
    """Every scene of cfg.train_test_split, as the agent sees it at test time."""
    from hydra.utils import instantiate

    return AgentInputs(scene_loader(cfg, agent, instantiate(cfg.train_test_split.scene_filter)),
                       agent.get_feature_builders())


class NavsimSamples(Dataset):
    """Preserve NAVSIM's legacy pair while adding its scene token at the loader boundary."""

    task = "planning"

    def __init__(self, dataset, num_poses: int):
        self.dataset = dataset
        self.num_poses = num_poses
        self.tokens = getattr(dataset, "tokens", None)
        if self.tokens is None:
            self.tokens = getattr(getattr(dataset, "_scene_loader", None), "tokens", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) not in (2, 3):
            raise ValueError(f"NAVSIM samples must contain (features, targets[, token]); got {type(sample).__name__}")
        features, targets = sample[:2]
        token = sample[2] if len(sample) == 3 else None
        if token is None and self.tokens is not None:
            token = self.tokens[index]
        if token is None:
            raise ValueError(f"NAVSIM sample {index} has no traceable scene token")
        if not isinstance(targets, dict) or "trajectory" not in targets:
            raise ValueError(f"NAVSIM sample {token!r} has no targets['trajectory']; check its cache/builders")
        trajectory = targets["trajectory"]
        if not torch.is_tensor(trajectory) or trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise TypeError(f"NAVSIM sample {token!r} target 'trajectory' must have shape (poses, 3)")
        if trajectory.shape[0] < self.num_poses:
            raise ValueError(f"NAVSIM sample {token!r} cache has {trajectory.shape[0]} trajectory poses; "
                             f"the model requires {self.num_poses}")
        return features, targets, token


@register("navsim")
def navsim_loader(cfg, agent, split: str):
    """train / val: scenes of train_logs / val_logs; test: every scene of train_test_split without targets."""
    if split == "test":
        return agent_inputs(cfg, agent)
    if not getattr(cfg, "use_cache_without_dataset", False):
        return NavsimSamples(navsim_scene(cfg, agent, split), horizon(agent))
    from navsim.planning.training.dataset import CacheOnlyDataset

    if cfg.force_cache_computation:
        raise AssertionError("force_cache_computation must be False when using cached data")
    if cfg.cache_path is None:
        raise AssertionError("cache_path must be provided when using cached data")
    return NavsimSamples(CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.train_logs if split == "train" else cfg.val_logs,
    ), horizon(agent))
