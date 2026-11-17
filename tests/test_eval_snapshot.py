"""An asynchronous evaluator must outlive top-k pruning of the training checkpoint."""
import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from recogdrive.eval.registry import SubprocessEvaluator
from recogdrive.eval.navsim import NavsimEvaluator
from recogdrive.infer import vlm_export_path
from scripts.verify.navsim_eval import main as navsim_eval


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


@pytest.mark.parametrize("scenario", ["score", "failure", "missing_score"])
def test_standalone_navhard_inherits_training_agent_and_reports_scorer_failure(tmp_path, monkeypatch, scenario):
    devkit = tmp_path / "navsim2.0"
    scorer = devkit / "navsim/planning/script/run_pdm_score.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text(
        "import pathlib, sys\n"
        "args = dict(arg.split('=', 1) for arg in sys.argv[1:])\n"
        "assert args['train_test_split'] == 'navhard_two_stage' and args['agent'] == 'trained_agent'\n"
        "assert pathlib.Path(args['metric_cache_path']).is_dir()\n"
        "out = pathlib.Path(args['output_dir'])\n"
        "if args['scenario'] == 'failure': sys.exit(7)\n"
        "if args['scenario'] == 'score':\n"
        "    (out / 'scores.csv').write_text('token,extended_pdm_score\\nextended_pdm_score_combined,0.75\\n')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("recogdrive.eval.navsim.devkit_root", lambda: devkit)
    agent = {"vlm_type": "qwen", "vlm_path": "/trained/vlm", "cam_type": "multi",
             "sampling_method": "flow", "lora_rank": 128, "cameras": ["front", "front_left"],
             "front_short_side": 960, "side_short_side": 480, "vlm_max_length": 8704,
             "action_norm_min": [-3.0, -2.0, -1.0], "grpo": True, "cache_mode": True,
             "cache_hidden_state": True, "train_backbone": True}
    config = tmp_path / "train.yaml"
    OmegaConf.save(OmegaConf.create({"agent": agent}), config)
    checkpoint = tmp_path / "last.ckpt"
    torch.save({"state_dict": {}, "vlm_export": False}, checkpoint)
    cache = tmp_path / "metric_cache"
    cache.mkdir()
    output = tmp_path / "eval_run"
    assert navsim_eval(["--training-config", str(config), "--checkpoint", str(checkpoint), "--metric-cache", str(cache),
                        "--output-dir", str(output), "--", f"scenario={scenario}"]) == (scenario != "score")
    actual = OmegaConf.to_container(OmegaConf.load(output / "eval/standalone/config/agent/trained_agent.yaml"))
    assert actual == {**agent, "checkpoint_path": str(checkpoint.resolve()), "grpo": False, "cache_mode": False,
                      "cache_hidden_state": False, "train_backbone": False}
    result = json.loads((output / "standalone_result.json").read_text())
    assert result["split"] == "navhard_two_stage"
    if scenario == "score":
        assert result["metrics"]["extended_pdm_score"] == 0.75
    else:
        assert ("exit 7" if scenario == "failure" else "no score csv") in result["metrics"]["error"]


def test_standalone_navhard_rejects_an_override_of_the_trained_agent(capsys):
    with pytest.raises(SystemExit) as exc:
        navsim_eval(["--training-config", "train.yaml", "--checkpoint", "last.ckpt",
                     "--metric-cache", "cache", "--output-dir", "new_eval", "--", "agent.vlm_type=internvl"])
    assert exc.value.code == 2 and "agent.vlm_type is fixed" in capsys.readouterr().err
