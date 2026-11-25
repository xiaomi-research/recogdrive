"""Read-only checks for the ReCogDrive NAVSIM 1.1 scoring command."""

import pickle
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from yaml import YAMLError


def scoring_config_errors(cfg):
    target = OmegaConf.select(cfg, "agent._target_", default="")
    if not target.endswith(".ReCogDriveAgent"):
        return []
    errors = []
    if cfg.agent.cam_type != "single":
        errors.append("The released ReCogDrive planner uses the front camera. Set agent.cam_type=single.")
    if cfg.agent.cache_hidden_state or cfg.agent.cache_mode or cfg.use_cache_without_dataset:
        errors.append("This scorer regenerates features from raw camera images. Set agent.cache_hidden_state=false, "
                      "agent.cache_mode=false and use_cache_without_dataset=false.")
    return errors


def _compare_recipe(cfg, label, path, required_fields, report, explicit=None):
    path = Path(explicit) if explicit else path
    if not path.is_file():
        report["errors" if explicit else "warnings"].append(f"{label} config unavailable at {path}; model/view origin unknown.")
        return
    try:
        saved = OmegaConf.load(path)
        if not OmegaConf.is_dict(saved.get("agent")):
            raise ValueError("expected an agent section")
        for field in required_fields:
            active = OmegaConf.select(cfg.agent, field)
            original = OmegaConf.select(saved.agent, field, throw_on_resolution_failure=False)
            if active is not None and original is None:
                report["warnings"].append(f"{label} config does not record agent.{field}.")
            elif active is not None and original != active:
                report["errors"].append(f"{label} agent.{field}: saved {original!r}, requested {active!r}.")
    except (OSError, ValueError, OmegaConfBaseException, YAMLError) as exc:
        report["errors"].append(f"Cannot read {label} config {path}: {exc}.")


def _check_feature_cache(cfg, report, loader):
    from navsim.planning.training.dataset import is_nonempty_cache_file, load_feature_target_from_pickle

    root = Path(cfg.preflight.feature_cache_path)
    paths = {}
    for token, log in loader.token_to_log_file.items():
        path = root / log / token
        if all(is_nonempty_cache_file(path / builder) for builder in ("internvl_feature.gz", "trajectory_target.gz")):
            paths[token] = path
    report["missing_feature_cache_tokens"] = sorted(set(loader.tokens) - paths.keys())
    if report["missing_feature_cache_tokens"]:
        report["errors"].append("Some selected scenes do not have both feature-cache files.")
    if not paths:
        return
    # shortcut: Inspect one payload for mode and width, scan all payloads when auditing mixed cache builds.
    token = min(paths)
    report["inspected_feature_cache_token"] = token
    try:
        features = load_feature_target_from_pickle(paths[token] / "internvl_feature.gz")
        if not isinstance(features, dict):
            raise ValueError("expected a dictionary")
        if "last_hidden_state" in features:
            shape = features["last_hidden_state"].shape
            width = 3584 if cfg.agent.vlm_size == "large" else 1536
            if len(shape) != 2 or shape[-1] != width:
                report["errors"].append(f"Feature cache token {token}: hidden shape {tuple(shape)}, expected width {width}.")
            _compare_recipe(cfg, "feature cache", root / "metadata/hydra/config.yaml",
                            ("vlm_type", "vlm_size", "cam_type"), report, cfg.preflight.feature_cache_config)
        elif "image_path_tensor" not in features:
            report["errors"].append(f"Feature cache token {token} has no hidden states or camera path.")
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError, RuntimeError) as exc:
        report["errors"].append(f"Cannot read feature cache token {token}: {exc}.")


def check_inputs(cfg):
    from hydra.utils import instantiate

    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from navsim.planning.training.dataset import is_nonempty_cache_file

    report = {"errors": scoring_config_errors(cfg), "warnings": []}
    if not OmegaConf.select(cfg, "agent._target_", default="").endswith(".ReCogDriveAgent"):
        report["errors"].append("Select agent=recogdrive_agent for this check.")
        return report
    checkpoint = Path(cfg.agent.checkpoint_path or "")
    if not is_nonempty_cache_file(checkpoint):
        report["errors"].append(f"Missing or empty planner checkpoint: {checkpoint}.")
    else:
        _compare_recipe(cfg, "checkpoint", checkpoint.parent / "code/hydra/config.yaml",
                        ("vlm_type", "vlm_size", "dit_type", "cam_type"), report, cfg.preflight.checkpoint_config)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    log_root = Path(cfg.navsim_log_path)
    present_logs = {path.stem for path in log_root.glob("*.pkl") if path.is_file()}
    selected_logs = set(scene_filter.log_names) if scene_filter.log_names is not None else present_logs
    report.update(logs_in_split=len(present_logs & selected_logs), logs_outside_split=len(present_logs - selected_logs),
                  missing_logs=sorted(selected_logs - present_logs))
    if report["missing_logs"]:
        report["warnings"].append("Some logs named by the selected split are missing.")
    if not log_root.is_dir():
        report["errors"].append(f"Missing NAVSIM log directory: {log_root}.")
        return report
    try:
        loader = SceneLoader(log_root, None, scene_filter, SensorConfig.build_no_sensors())
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        report["errors"].append(f"Cannot read NAVSIM logs: {exc}.")
        return report
    tokens = set(loader.tokens)
    report["scenes"] = len(tokens)
    if not tokens:
        report["errors"].append("No scenes matched the selected logs and scene filter.")
    if scene_filter.tokens is not None and scene_filter.max_scenes is None:
        report["missing_scene_tokens"] = sorted(set(scene_filter.tokens) - tokens)
        if report["missing_scene_tokens"]:
            report["errors"].append("Some requested scene tokens are absent.")
    image_root = Path(cfg.sensor_blobs_path)
    report["missing_front_camera_tokens"] = []
    for token, frames in loader.scene_frames_dicts.items():
        cameras = frames[scene_filter.num_history_frames - 1]["cams"]
        front = next((camera.get("data_path") for key, camera in cameras.items() if key.lower() == "cam_f0"), None)
        if not front or not is_nonempty_cache_file(image_root / front):
            report["missing_front_camera_tokens"].append(token)
    report["missing_front_camera_tokens"].sort()
    if report["missing_front_camera_tokens"]:
        report["errors"].append("Some selected scenes are missing their front-camera file.")

    try:
        metrics = MetricCacheLoader(Path(cfg.metric_cache_path)).metric_cache_paths
        report["missing_metric_cache_tokens"] = sorted(
            token for token in tokens if token not in metrics or not is_nonempty_cache_file(Path(metrics[token]))
        )
        if report["missing_metric_cache_tokens"]:
            report["errors"].append("Some selected scenes are missing their metric-cache file.")
    except (OSError, IndexError, ValueError) as exc:
        report["errors"].append(f"Cannot read metric cache at {cfg.metric_cache_path}: {exc}.")

    if cfg.preflight.feature_cache_path:
        _check_feature_cache(cfg, report, loader)
    return report
