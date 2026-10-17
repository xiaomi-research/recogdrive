"""Waymo reward selection and the explicit five-second RL launcher are integrated."""
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import pytest
import numpy as np
import torch
from navsim.planning.training.dataset import dump_feature_target_to_pickle

from recogdrive.adapters.navsim.agent import ReCogDriveAgent
from recogdrive.adapters.waymoe2e.reward import WaymoRFSReward
from recogdrive.data.waymoe2e import WaymoE2ECacheOnlyDataset
from recogdrive.train.args import TrainArgs
from recogdrive.train.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def labelled_cache(root):
    frame = root / "training" / "sample"
    frame.mkdir(parents=True, exist_ok=True)
    dump_feature_target_to_pickle(frame / "eval.gz", {
        "rater_trajectories": [np.zeros((20, 2))], "rater_scores": np.array([8.0]), "init_speed": 0.0,
    })
    dump_feature_target_to_pickle(frame / "features.gz", {"history_trajectory": torch.zeros(4, 3)})
    dump_feature_target_to_pickle(frame / "targets.gz", {"trajectory": torch.zeros(10, 3)})


def test_waymo_reward_configuration_fails_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="grpo_reward_cache_path"):
        ReCogDriveAgent(TrajectorySampling(time_horizon=5, interval_length=0.5),
                       grpo=True, grpo_reward="waymo_rfs")
    with pytest.raises(ValueError, match="5 s horizon"):
        ReCogDriveAgent(TrajectorySampling(time_horizon=4, interval_length=0.5),
                       grpo=True, grpo_reward="waymo_rfs", grpo_reward_cache_path=str(tmp_path))


def test_waymo_training_guard_rejects_unlabelled_or_other_cache(tmp_path):
    labelled_cache(tmp_path)
    reward = WaymoRFSReward(str(tmp_path), "training")
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = OmegaConf.create({})
    trainer.args = TrainArgs(strategy="fsdp")
    trainer.agent = SimpleNamespace(grpo=True, grpo_reward="waymo_rfs", action_head=SimpleNamespace(reward_fn=reward))
    dataset = WaymoE2ECacheOnlyDataset.__new__(WaymoE2ECacheOnlyDataset)
    dataset.tokens = [tmp_path / "training" / "sample"]
    dataset.cache_path = tmp_path.resolve()
    dataset.split = "training"
    dataset.labelled_only = False
    with pytest.raises(ValueError, match="labelled-only"):
        trainer.validate_train_data(dataset)
    dataset.labelled_only = True
    trainer.validate_train_data(dataset)
    dataset.cache_path = tmp_path / "other"
    with pytest.raises(ValueError, match="same cache root"):
        trainer.validate_train_data(dataset)


@pytest.mark.parametrize("tree", ["1.1", "2.0"])
@pytest.mark.parametrize("reward_kind", ["waymo_ade", "waymo_rfs"])
def test_waymo_rl_launcher_resolves_training_reward_config(tmp_path, tree, reward_kind, monkeypatch):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("shell launcher check requires bash")
    captured = tmp_path / "argv"
    fake = tmp_path / "torchrun"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$RECOGDRIVE_CAPTURE"\n')
    fake.chmod(0o755)
    checkpoint = tmp_path / "stage2.ckpt"
    checkpoint.write_bytes(b"launcher fixture")
    cache = tmp_path / "cache"
    labelled_cache(cache)
    env = os.environ.copy()
    env.update(PATH=str(tmp_path) + os.pathsep + env["PATH"], RECOGDRIVE_CAPTURE=str(captured),
               NAVSIM_TREE=tree, GPUS="1", GPUS_PER_NODE="1", CHECKPOINT=str(checkpoint),
               CACHE_PATH=str(cache), LOG_FILE=str(tmp_path / "launch.log"), VLM_PATH="/weights/fixed",
               NAVSIM_EXP_ROOT=str(tmp_path / "exp"), MODEL_FAMILY="internvl3", WAYMO_GRPO_REWARD=reward_kind)
    for key in ("NAVSIM_DEVKIT_ROOT", "WAYMOE2E_TRAIN_SPLIT", "WAYMOE2E_VAL_SPLIT"):
        env.pop(key, None)
    subprocess.run([bash, str(ROOT / "scripts/train/run_recogdrive_waymoe2e_stage3_rl.sh")],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=30)
    argv = captured.read_bytes().decode().split("\0")[:-1]
    entry = next(i for i, arg in enumerate(argv) if arg.endswith(".py"))
    monkeypatch.setenv("NAVSIM_EXP_ROOT", env["NAVSIM_EXP_ROOT"])
    with initialize_config_dir(config_dir=str(ROOT / ("navsim" + tree) / "navsim/planning/script/config/training"),
                               version_base=None):
        cfg = compose(config_name="default_training", overrides=argv[entry + 1:])
    assert cfg.agent.grpo and cfg.agent.grpo_reward == reward_kind
    assert cfg.agent.sampling_method == "ddim"
    assert cfg.agent.trajectory_sampling.time_horizon == 5
    assert not cfg.agent.cache_hidden_state
    assert cfg.waymoe2e_labelled_only == (reward_kind == "waymo_rfs")
    assert cfg.agent.grpo_reward_cache_path == str(cache)
    assert cfg.agent.grpo_reward_split == cfg.waymoe2e_train_split == "training"
