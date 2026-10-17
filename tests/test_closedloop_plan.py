"""Scene commands and an external score aggregator are executed as one checked plan.

These subprocesses are fixtures, not official simulator runs or benchmark scores.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

from omegaconf import OmegaConf
import pytest
import torch

from recogdrive.eval.closedloop import run


def _write(path, value):
    return [sys.executable, "-c",
            "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); "
            "p.write_text(json.dumps(json.loads(sys.argv[2])))", str(path), json.dumps(value)]


def _plan(root):
    episodes = [root / "episode0.json", root / "episode1.json"]
    return {
        "protocol": "hugsim", "policy_owned_by_simulator": True,
        "checkpoint": "fixture.ckpt", "vlm_path": "unused", "agent_config_dir": str(root),
        "policy_python": sys.executable, "navsim_devkit_root": str(root),
        "simulator_commands": [_write(episodes[0], {"hdscore": 0.2}),
                               _write(episodes[1], {"hdscore": 0.8})],
        "episode_result_files": [str(path) for path in episodes],
        "aggregate_command": [sys.executable, "-c",
            "import json,pathlib,sys; r=pathlib.Path(sys.argv[1]); "
            "a=json.loads((r/'episode0.json').read_text()); b=json.loads((r/'episode1.json').read_text()); "
            "(r/'aggregate.json').write_text(json.dumps({'score':(2*a['hdscore']+b['hdscore'])/3}))", str(root)],
        "simulator_timeout_s": 3, "aggregate_timeout_s": 3,
        "result_file": "aggregate.json", "metrics": {"score": "score"},
        "simulator_env": [], "policy_env": [],
    }


def test_two_scenes_then_external_aggregator_preserves_its_score(tmp_path):
    assert run(_plan(tmp_path), tmp_path) == {"score": pytest.approx(0.4)}
    status = json.loads((tmp_path / "closedloop_run_status.json").read_text())
    assert status["success"]
    assert [stage["kind"] for stage in status["stages"]] == ["simulator", "simulator", "aggregate"]
    assert [stage["returncode"] for stage in status["stages"]] == [0, 0, 0]


@pytest.mark.parametrize("failure", ["scene_exit", "scene_missing", "aggregate_exit", "aggregate_missing"])
def test_scene_or_aggregate_failure_never_publishes_success(tmp_path, failure):
    config = _plan(tmp_path)
    if failure == "scene_exit":
        config["simulator_commands"][1] = [sys.executable, "-c", "raise SystemExit(7)"]
    elif failure == "scene_missing":
        config["simulator_commands"][0] = [sys.executable, "-c", "pass"]
    elif failure == "aggregate_exit":
        config["aggregate_command"] = [sys.executable, "-c", "raise SystemExit(8)"]
    else:
        config["aggregate_command"] = [sys.executable, "-c", "pass"]
    with pytest.raises((RuntimeError, ValueError)):
        run(config, tmp_path)
    status = json.loads((tmp_path / "closedloop_run_status.json").read_text())
    assert not status["success"] and "error" in status
    assert not (tmp_path / "recogdrive_result.json").exists()
    if failure.startswith("scene"):
        assert all(stage["kind"] != "aggregate" for stage in status["stages"])


def test_aggregate_timeout_reaps_its_owned_process(tmp_path):
    config = _plan(tmp_path)
    pid_path = tmp_path / "aggregate.pid"
    config["aggregate_command"] = [sys.executable, "-c",
        "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)", str(pid_path)]
    config["aggregate_timeout_s"] = 0.3
    with pytest.raises(TimeoutError):
        run(config, tmp_path)
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_path.read_text()), 0)
    assert not json.loads((tmp_path / "closedloop_run_status.json").read_text())["success"]


def test_multi_scene_plan_requires_complete_artifacts_and_aggregator(tmp_path):
    config = _plan(tmp_path)
    config["aggregate_command"] = None
    with pytest.raises(ValueError, match="multi-scenario"):
        run(config, tmp_path)
    config = _plan(tmp_path)
    config["episode_result_files"] = config["episode_result_files"][:1]
    with pytest.raises(ValueError, match="one artifact"):
        run(config, tmp_path)


def test_multi_scene_cannot_count_the_same_result_twice(tmp_path):
    config = _plan(tmp_path)
    config["episode_result_files"][1] = config["episode_result_files"][0]
    config["simulator_commands"][1] = _write(tmp_path / "episode0.json", {"hdscore": 0.8})
    with pytest.raises(ValueError, match="distinct"):
        run(config, tmp_path)


def test_plan_serialization_preserves_agent_and_expands_scene_output_paths(tmp_path, monkeypatch):
    from recogdrive.eval.closedloop import HUGSIMEvaluator

    monkeypatch.setattr("recogdrive.eval.closedloop.vlm_export_path", lambda _: None)
    monkeypatch.setattr(HUGSIMEvaluator, "_devkit_root", staticmethod(lambda: tmp_path))
    agent = {"_target_": "custom.Agent", "vlm_path": "/weights/base", "lora_rank": 4,
             "grpo": True, "cache_hidden_state": True, "trajectory_sampling": {"time_horizon": 5}}
    evaluator = HUGSIMEvaluator(OmegaConf.create({"agent": agent}), OmegaConf.create({
        "simulator_commands": [["python", "official_scene.py", "{episode_output_dir}"] for _ in range(2)],
        "episode_result_files": ["{episode_output_dir}/eval.json" for _ in range(2)],
        "aggregate_command": ["python", "official_aggregate.py", "{output_dir}"],
        "metrics": {"score": "score"}, "result_file": "aggregate.json",
    }), tmp_path)
    output = tmp_path / "run"
    evaluator.command(tmp_path / "model.ckpt", output, "plan")
    serialized = json.loads((output / "closedloop_runner.json").read_text())
    assert serialized["simulator_command"] is None
    assert serialized["simulator_commands"][0][-1].endswith("episodes/0000")
    assert serialized["simulator_commands"][1][-1].endswith("episodes/0001")
    assert len(set(serialized["episode_result_files"])) == 2
    restored = OmegaConf.load(output / "config/agent/trained_agent.yaml")
    assert restored._target_ == agent["_target_"] and restored.lora_rank == 4
    assert restored.trajectory_sampling.time_horizon == 5
    assert not restored.grpo and not restored.cache_hidden_state


def test_actual_standalone_cli_returns_nonzero_when_aggregator_fails(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}, "vlm_export": False}, checkpoint)
    cfg = OmegaConf.create({"agent": {"vlm_path": "unused"}, "evaluator": {
        "name": None, "gpus": None, "hugsim": {
            "simulator_command": _write(tmp_path / "scene.json", {"hdscore": 0.4}),
            "episode_result_files": [str(tmp_path / "scene.json")],
            "aggregate_command": [sys.executable, "-c", "raise SystemExit(9)"],
            "result_file": "score.json", "metrics": {"score": "score"},
        }}})
    config_path = tmp_path / "training.yaml"
    OmegaConf.save(cfg, config_path)
    result = subprocess.run([sys.executable, "-m", "recogdrive.eval.closedloop",
        "--training-config", str(config_path), "--checkpoint", str(checkpoint), "--protocol", "hugsim",
        "--output-dir", str(tmp_path / "output")], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    report = json.loads((tmp_path / "output" / "standalone_result.json").read_text())
    assert "status 9" in report["metrics"]["error"]


def test_os_sigterm_training_cleanup_stops_real_evaluation_child(tmp_path):
    script = textwrap.dedent('''
        import pathlib,sys,time
        import torch
        from types import SimpleNamespace
        from omegaconf import OmegaConf
        import recogdrive.train.trainer as trainer_module
        from recogdrive.eval.registry import SubprocessEvaluator
        root=pathlib.Path(sys.argv[1])
        evaluator=SubprocessEvaluator(OmegaConf.create({}),OmegaConf.create({}),root)
        evaluator.launch([sys.executable,'-c',
            'import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)',
            str(root/'evaluation.pid')],root,'pending',dict(__import__('os').environ))
        trainer=trainer_module.Trainer.__new__(trainer_module.Trainer)
        trainer.agent=torch.nn.Linear(1,1)
        trainer.args=SimpleNamespace(ema_decay=0,resume=False)
        trainer.evaluators=[evaluator]
        trainer.checkpointer=SimpleNamespace(wait=lambda: None)
        trainer.ctx=SimpleNamespace(close=lambda: (root/'context.closed').write_text('yes'))
        trainer.monitor=SimpleNamespace(close=lambda: (root/'monitor.closed').write_text('yes'))
        trainer.build_data=trainer.build_optimizer=lambda: None
        trainer_module.parallelize=lambda *_: trainer.agent
        def loop():
            (root/'ready').write_text('yes')
            time.sleep(60)
        trainer.train_loop=loop
        try: trainer.fit()
        except KeyboardInterrupt: raise SystemExit(130)
    ''')
    with (tmp_path / "training.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            while not ((tmp_path / "ready").is_file() and (tmp_path / "evaluation.pid").is_file()):
                if proc.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail((tmp_path / "training.log").read_text())
                time.sleep(0.05)
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=15) == 130
            assert (tmp_path / "context.closed").is_file() and (tmp_path / "monitor.closed").is_file()
            if os.name == "posix":
                with pytest.raises(ProcessLookupError):
                    os.kill(int((tmp_path / "evaluation.pid").read_text()), 0)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)
