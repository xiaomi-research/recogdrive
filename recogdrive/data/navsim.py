from pathlib import Path

from recogdrive.data.registry import register


def navsim_scene(cfg, agent, split: str):
    from hydra.utils import instantiate

    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset import Dataset

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    logs = cfg.train_logs if split == "train" else cfg.val_logs
    if scene_filter.log_names is not None:
        scene_filter.log_names = [name for name in scene_filter.log_names if name in logs]
    else:
        scene_filter.log_names = logs
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    return Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )


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
