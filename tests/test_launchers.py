"""Resolve real shell launchers with a captured torchrun, without allocating GPUs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tree", ["1.1", "2.0"])
@pytest.mark.parametrize("script,hidden", [
    ("train/run_recogdrive_train.sh", True),
    ("train/run_recogdrive_vqa_planning_il.sh", False),
    ("cache_dataset/run_caching_recogdrive_hidden_state.sh", True),
    ("cache_dataset/run_caching_recogdrive_no_hidden_state.sh", False),
])
def test_launcher_hydra_overrides_select_the_intended_cache_mode(tmp_path, tree, script, hidden):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("shell launcher validation needs bash")
    captured = tmp_path / "argv"
    torchrun = tmp_path / "torchrun"
    torchrun.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$RECOGDRIVE_CAPTURE"\n')
    torchrun.chmod(0o755)
    python = tmp_path / "python"
    python.write_bytes(torchrun.read_bytes())
    python.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=str(tmp_path) + os.pathsep + env["PATH"], RECOGDRIVE_CAPTURE=str(captured),
               NAVSIM_TREE=tree, GPUS="1", GPUS_PER_NODE="1", MODEL_FAMILY="internvl3",
               TRAIN_STAGE="il", TRAIN_TEST_SPLIT="mini", VLM_PATH="/weights/fixed", NAVSIM_EXP_ROOT=str(tmp_path / "exp"),
               LOG_FILE=str(tmp_path / "launch.log"), VQA_TRAIN_JSONL="/vqa/train.jsonl",
               VQA_VAL_JSONL="/vqa/val.jsonl")
    for key in ("CACHE_HIDDEN_STATE", "CACHE_PATH", "NAVSIM_DEVKIT_ROOT", "CACHE_ROOT", "WAYMOE2E"):
        env.pop(key, None)
    subprocess.run([bash, str(ROOT / "scripts" / script)], cwd=ROOT, env=env,
                   check=True, capture_output=True, text=True, timeout=30)
    argv = captured.read_bytes().decode().split("\0")[:-1]
    entry_index = next(i for i, arg in enumerate(argv) if arg.endswith(".py"))
    config_dir = ROOT / ("navsim" + tree) / "navsim/planning/script/config/training"
    # Hydra resolves oc.env in-process; use the same temporary experiment root.
    previous = os.environ.get("NAVSIM_EXP_ROOT")
    os.environ["NAVSIM_EXP_ROOT"] = env["NAVSIM_EXP_ROOT"]
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="default_training", overrides=argv[entry_index + 1:])
        assert cfg.agent.cache_hidden_state is hidden
        assert ("_no_hidden_state" in cfg.cache_path) == (not hidden)
    finally:
        if previous is None:
            os.environ.pop("NAVSIM_EXP_ROOT", None)
        else:
            os.environ["NAVSIM_EXP_ROOT"] = previous
