from pathlib import Path

from torch.utils.data import Dataset

from recogdrive.data.registry import register


def scene_loader(cfg, agent, scene_filter):
    from navsim.common.dataloader import SceneLoader

    return SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=getattr(agent, "load_image_path", False),
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


@register("navsim")
def navsim_loader(cfg, agent, split: str):
    if not getattr(cfg, "use_cache_without_dataset", False):
        return navsim_scene(cfg, agent, split)
    from navsim.planning.training.dataset import CacheOnlyDataset

    if cfg.force_cache_computation:
        raise AssertionError("force_cache_computation must be False when using cached data")
    if cfg.cache_path is None:
        raise AssertionError("cache_path must be provided when using cached data")
    return CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.train_logs if split == "train" else cfg.val_logs,
    )
