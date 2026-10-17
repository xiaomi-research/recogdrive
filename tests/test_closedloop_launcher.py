"""Simulator-owned launchers preserve the selected runtime and serialized training model."""
import os
from pathlib import Path
import shutil
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


@pytest.mark.parametrize("tree", ["1.1", "2.0"])
def test_serialized_agent_and_python_survive_hugsim_launcher(tmp_path, tree, monkeypatch):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("shell launcher check requires bash")
    root = Path(__file__).resolve().parents[1]
    capture = tmp_path / "argv"
    python = tmp_path / "selected-python"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$RECOGDRIVE_CAPTURE"\n'
                      'printf "%s" "$CUDA_VISIBLE_DEVICES" > "$RECOGDRIVE_CAPTURE.gpu"\n')
    python.chmod(0o755)
    config_dir = tmp_path / "config"
    (config_dir / "agent").mkdir(parents=True)
    original = OmegaConf.load(root / ("navsim" + tree) / "navsim/planning/script/config/common/agent/recogdrive_agent.yaml")
    original.vlm_type = "qwen"
    original.lora_rank = 4
    original.trajectory_sampling.time_horizon = 5
    original.cache_hidden_state = False
    OmegaConf.save(original, config_dir / "agent/trained_agent.yaml")
    episode = tmp_path / "episode"
    env = os.environ.copy()
    env.update(NAVSIM_DEVKIT_ROOT=str(root / ("navsim" + tree)), NAVSIM_EXP_ROOT=str(tmp_path / "exp"),
               RECOGDRIVE_PYTHON=str(python), RECOGDRIVE_CAPTURE=str(capture),
               AGENT_CONFIG_DIR=str(config_dir), CHECKPOINT=str(tmp_path / "model.ckpt"),
               CUDA_VISIBLE_DEVICES="2,3")
    subprocess.run([bash, str(root / "scripts/infer/run_recogdrive_hugsim.sh"), "0", str(episode)],
                   cwd=root, env=env, check=True, capture_output=True, text=True, timeout=30)
    argv = capture.read_bytes().decode().split("\0")[:-1]
    assert capture.with_suffix(".gpu").read_text() == "2"
    monkeypatch.setenv("NAVSIM_EXP_ROOT", env["NAVSIM_EXP_ROOT"])
    with initialize_config_dir(config_dir=str(root / ("navsim" + tree) / "navsim/planning/script/config/training"),
                               version_base=None):
        cfg = compose(config_name="default_training", overrides=argv[1:])
    assert cfg.agent.vlm_type == "qwen" and cfg.agent.lora_rank == 4
    assert cfg.agent.trajectory_sampling.time_horizon == 5
    assert not cfg.agent.cache_hidden_state and cfg.infer.serve == "hugsim"
    assert cfg.output_dir == str(episode)
