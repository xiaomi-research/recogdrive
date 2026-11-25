import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from navsim.planning.script.recogdrive_preflight import check_inputs
from navsim.planning.training.dataset import dump_feature_target_to_pickle
from navsim.visualization.trajectory_plot import save_trajectory_plot


@pytest.fixture
def inputs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 16)).save(images / "front.jpg")
    frame = {"token": "first", "log_name": "selected", "cams": {"CAM_F0": {"data_path": "front.jpg"}}}
    (logs / "selected.pkl").write_bytes(pickle.dumps([frame] * 4))
    (logs / "outside.pkl").write_bytes(pickle.dumps([]))
    (logs / "README.txt").write_text("this is not a log")
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"preflight checks path and recipe")
    metric = tmp_path / "metric"
    entry = metric / "selected/first/metric_cache.pkl"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"preflight checks metric presence")
    (metric / "metadata").mkdir()
    (metric / "metadata/index.csv").write_text(f"file_name\n{entry.as_posix()}\n")
    cfg = OmegaConf.create({
        "agent": {"_target_": "navsim.agents.recogdrive.recogdrive_agent.ReCogDriveAgent",
                  "checkpoint_path": str(checkpoint), "cam_type": "single", "vlm_type": "internvl",
                  "vlm_size": "small", "dit_type": "small", "cache_mode": False, "cache_hidden_state": False},
        "use_cache_without_dataset": False, "navsim_log_path": str(logs),
        "sensor_blobs_path": str(images), "metric_cache_path": str(metric),
        "train_test_split": {"scene_filter": {"_target_": "navsim.common.dataclasses.SceneFilter",
                                              "num_history_frames": 4, "num_future_frames": 0, "has_route": False,
                                              "log_names": ["selected", "missing_log"], "tokens": ["first"]}},
        "preflight": {"enabled": True, "checkpoint_config": None, "feature_cache_path": None,
                      "feature_cache_config": None},
    })
    return cfg, entry


def test_preflight_exits_before_starting_distributed_scoring(inputs, capsys, monkeypatch):
    from navsim.planning.script.run_pdm_score_recogdrive import main

    cfg, metric = inputs
    monkeypatch.setenv("RANK", "0")

    def fail_if_started(*args, **kwargs):
        pytest.fail("The CPU check started distributed scoring")

    monkeypatch.setattr(torch.distributed, "init_process_group", fail_if_started)
    main.__wrapped__(cfg)
    report = json.loads(capsys.readouterr().out)
    assert report["scenes"] == 1
    assert report["logs_outside_split"] == 1
    assert report["missing_logs"] == ["missing_log"]
    assert report["missing_metric_cache_tokens"] == []

    metric.unlink()
    cfg.train_test_split.scene_filter.tokens.append("absent")
    with pytest.raises(ValueError, match="input check failed"):
        main.__wrapped__(cfg)
    report = json.loads(capsys.readouterr().out)
    assert report["missing_scene_tokens"] == ["absent"]
    assert report["missing_metric_cache_tokens"] == ["first"]


def test_reports_camera_and_checkpoint_and_cache_shape(inputs, tmp_path):
    cfg, _ = inputs
    (Path(cfg.sensor_blobs_path) / "front.jpg").unlink()
    cfg.agent.cache_hidden_state = True
    recipe = tmp_path / "recipe.yaml"
    OmegaConf.save({"agent": {"vlm_type": "internvl", "vlm_size": "large",
                               "dit_type": "small", "cam_type": "multi"}}, recipe)
    cfg.preflight.checkpoint_config = str(recipe)
    root = tmp_path / "cache"
    path = root / "selected/first"
    path.mkdir(parents=True)
    dump_feature_target_to_pickle(path / "internvl_feature.gz", {"last_hidden_state": torch.zeros(2, 3584)})
    dump_feature_target_to_pickle(path / "trajectory_target.gz", {"trajectory": torch.zeros(8, 3)})
    cfg.preflight.feature_cache_path = str(root)

    report = check_inputs(cfg)
    assert report["missing_front_camera_tokens"] == ["first"]
    assert report["inspected_feature_cache_token"] == "first"
    assert any("checkpoint agent.cam_type" in error for error in report["errors"])
    assert any("checkpoint agent.vlm_size" in error for error in report["errors"])
    assert any("expected width 1536" in error for error in report["errors"])
    assert any("raw camera images" in error for error in report["errors"])


def test_plot_saves_the_selected_scene(inputs, tmp_path):
    cfg, _ = inputs
    agent_input = SimpleNamespace(
        cameras=[SimpleNamespace(cam_f0=SimpleNamespace(image=Path(cfg.sensor_blobs_path) / "front.jpg"))],
        ego_statuses=[SimpleNamespace(ego_pose=[-1, 0, 0]), SimpleNamespace(ego_pose=[0, 0, 0])],
    )
    save_trajectory_plot(agent_input, SimpleNamespace(poses=[[1, 0, 0], [2, 1, 0]]), "first", tmp_path)
    with Image.open(tmp_path / "first.png") as result:
        assert result.width > result.height


def test_documented_preflight_does_not_save_over_run_config():
    root = Path(__file__).resolve().parents[1]
    config = root / "navsim/planning/script/config/pdm_scoring"
    with initialize_config_dir(config_dir=str(config), version_base=None):
        cfg = compose(config_name="default_run_pdm_score", return_hydra_config=True,
                      overrides=["agent=recogdrive_agent", "+preflight=check", "agent.cache_hidden_state=false",
                                 "train_test_split.scene_filter.tokens=[first]"])
    assert list(cfg.train_test_split.scene_filter.tokens) == ["first"]
    assert cfg.preflight.enabled
    assert cfg.hydra.output_subdir is None
    assert cfg.hydra.runtime.choices["hydra/job_logging"] == "stdout"
