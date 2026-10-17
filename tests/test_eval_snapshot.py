"""An asynchronous evaluator must outlive top-k pruning of the training checkpoint."""
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from recogdrive.eval.registry import SubprocessEvaluator
from recogdrive.eval.navsim import NavsimEvaluator
from recogdrive.infer import vlm_export_path


@pytest.mark.parametrize("gpus", [None, "0"])
def test_periodic_evaluation_preserves_checkpoint_and_vlm_after_pruning(tmp_path, gpus):
    checkpoint = tmp_path / "epoch_0001.ckpt"
    checkpoint.write_bytes(b"fixed weights")
    vlm = tmp_path / "epoch_0001_vlm"
    vlm.mkdir()
    (vlm / "config.json").write_text('{"model_type":"fixed"}')
    evaluator = SubprocessEvaluator(OmegaConf.create({}), OmegaConf.create({"gpus": gpus}), tmp_path)
    evaluator.submit = Mock()
    evaluator.during_training(SimpleNamespace(ctx=SimpleNamespace(is_main=True)), checkpoint, "epoch_0001")
    saved = evaluator.deferred[0][0] if gpus is None else evaluator.submit.call_args.args[0]
    checkpoint.unlink()
    shutil.rmtree(vlm)
    assert saved.read_bytes() == b"fixed weights"
    assert (saved.with_name(saved.stem + "_vlm") / "config.json").is_file()


def test_cached_navsim_policy_defers_fresh_scene_evaluation_to_online_model(tmp_path):
    cfg = OmegaConf.create({"agent": {"cache_hidden_state": True, "cache_mode": False,
                                      "grpo": True, "train_backbone": False}})
    evaluator = NavsimEvaluator(cfg, OmegaConf.create({"gpus": None}), tmp_path)
    evaluator.generate = Mock(side_effect=AssertionError("cached policy has no online VLM"))
    checkpoint = tmp_path / "weights.ckpt"
    torch.save({"state_dict": {}, "vlm_export": False}, checkpoint)
    evaluator.during_training(SimpleNamespace(agent=SimpleNamespace(cache_hidden_state=True),
                                             ctx=SimpleNamespace(is_main=True)), checkpoint, "epoch_0001")
    assert not evaluator.in_process and len(evaluator.deferred) == 1
    config_dir = tmp_path / "config"
    evaluator.agent_config(checkpoint, config_dir)
    evaluation = OmegaConf.load(config_dir / "agent/trained_agent.yaml")
    assert evaluation.cache_hidden_state is False and evaluation.grpo is False


def test_online_navsim_final_replay_does_not_reload_a_second_policy(tmp_path):
    evaluator = NavsimEvaluator(OmegaConf.create({"agent": {"cache_hidden_state": False}}),
                               OmegaConf.create({"on_end": True}), tmp_path)
    evaluator.submit = Mock()
    final = tmp_path / "weights.ckpt"
    final.write_bytes(b"fixed weights")
    assert evaluator.in_process and evaluator.finish(final) == []
    evaluator.submit.assert_not_called()


def test_checkpoint_metadata_ignores_stale_vlm_and_requires_full_export(tmp_path):
    checkpoint = tmp_path / "weights.ckpt"
    directory = tmp_path / "weights_vlm"
    directory.mkdir()
    torch.save({"state_dict": {}, "vlm_export": False}, checkpoint)
    assert vlm_export_path(checkpoint) is None
    torch.save({"state_dict": {}, "vlm_export": True}, checkpoint)
    assert vlm_export_path(checkpoint) == directory
    directory.rmdir()
    with pytest.raises(FileNotFoundError, match="requires its full VLM export"):
        vlm_export_path(checkpoint)


def test_legacy_checkpoint_keeps_vlm_directory_convention(tmp_path):
    checkpoint = tmp_path / "weights.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    assert vlm_export_path(checkpoint) is None
    directory = tmp_path / "weights_vlm"
    directory.mkdir()
    assert vlm_export_path(checkpoint) == directory
