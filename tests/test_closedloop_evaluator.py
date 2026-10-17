"""Closed-loop process lifecycle and official-metric parsing checks."""

import json
import os
from pathlib import Path
import socket
import sys
import textwrap
import time

import pytest
from omegaconf import OmegaConf

from recogdrive.eval import EVALUATORS
from recogdrive.eval.closedloop import HUGSIMEvaluator, _probe_ready, run


def _free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def _config(out_dir, command, **overrides):
    config = {
        "protocol": "hugsim",
        "policy_owned_by_simulator": True,
        "simulator_command": command,
        "simulator_timeout_s": 3,
        "policy_startup_timeout_s": 2,
        "policy_env": [],
        "simulator_env": [],
        "result_file": "eval.json",
        "metrics": {"hdscore": "hdscore", "route_completion": "rc"},
        "checkpoint": "unused.ckpt",
        "vlm_path": "unused-vlm",
        "agent_config_dir": str(out_dir / "config"),
        "policy_python": sys.executable,
        "navsim_devkit_root": str(out_dir),
    }
    config["simulator_command"] = [
        arg.replace("{output_dir}", str(out_dir)) for arg in command
    ]
    config.update(overrides)
    return config


def _write_result(value):
    payload = json.dumps(value)
    return [sys.executable, "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1],'eval.json').write_text(" + repr(payload) + ")",
            "{output_dir}"]


@pytest.mark.parametrize("name", ["neuroncap", "hugsim", "alpasim"])
def test_closedloop_evaluators_are_registered_as_subprocess_evaluators(name):
    assert name in EVALUATORS
    assert issubclass(EVALUATORS[name], __import__("recogdrive.eval.registry", fromlist=["SubprocessEvaluator"])
                      .SubprocessEvaluator)


def test_empty_child_environment_mappings_are_valid(tmp_path):
    evaluator = HUGSIMEvaluator(OmegaConf.create({"agent": {"vlm_path": "unused"}}),
                                OmegaConf.create({"simulator_env": {}, "policy_env": {}}), tmp_path)
    env = evaluator.child_env()
    assert not any(key.startswith("RECOGDRIVE_CLOSEDLOOP_") for key in env)


def test_policy_readiness_uses_the_protocol_health_probe(tmp_path):
    port = _free_port()
    script = textwrap.dedent("""
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == '/alive' else 404)
                self.end_headers()
                self.wfile.write(b'true')
            def log_message(self, *args):
                pass
        HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
    """).replace("PORT", str(port))
    from recogdrive.eval.closedloop import _start, _stop_owned, _wait_ready

    process = _start([sys.executable, "-c", script], os.environ.copy(), tmp_path / "policy.log")
    try:
        _wait_ready(process, "127.0.0.1", port, "http", 3)
        assert _probe_ready("127.0.0.1", port, "http")
    finally:
        _stop_owned(process)
    assert not _probe_ready("127.0.0.1", port, "tcp")


def test_policy_exit_before_readiness_is_reported_and_owned_process_is_stopped(tmp_path):
    from recogdrive.eval.closedloop import _start, _stop_owned, _wait_ready

    process = _start([sys.executable, "-c", "raise SystemExit(9)"], os.environ.copy(), tmp_path / "policy.log")
    time.sleep(0.1)
    try:
        with pytest.raises(RuntimeError, match="exited before readiness"):
            _wait_ready(process, "127.0.0.1", 1, "tcp", 1)
    finally:
        _stop_owned(process)
    assert process.poll() == 9


def test_simulator_nonzero_exit_is_an_evaluation_error(tmp_path):
    config = _config(tmp_path, [sys.executable, "-c", "raise SystemExit(7)"])
    with pytest.raises(RuntimeError, match="status 7"):
        run(config, tmp_path)


@pytest.mark.parametrize("key,timeout", [
    ("simulator_timeout_s", 0), ("simulator_timeout_s", float("nan")),
    ("policy_startup_timeout_s", float("inf")), ("policy_startup_timeout_s", True),
])
def test_nonpositive_or_nonfinite_timeouts_fail_before_starting_processes(tmp_path, key, timeout):
    config = _config(tmp_path, [sys.executable, "-c", "pass"], **{key: timeout})
    with pytest.raises(ValueError, match="finite and greater than zero"):
        run(config, tmp_path)


def test_occupied_foreign_policy_port_is_rejected_without_touching_listener(tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    config = _config(tmp_path, [sys.executable, "-c", "pass"], protocol="neuroncap",
                     policy_owned_by_simulator=False, policy_command=[sys.executable, "-c", "pass"],
                     policy_host="127.0.0.1", policy_port=port, readiness="tcp",
                     result_file="metrics.json", metrics={"score": "score"})
    try:
        with pytest.raises(RuntimeError, match="refusing to use an unowned listener"):
            run(config, tmp_path)
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            assert listener.fileno() >= 0
    finally:
        listener.close()


def test_simulator_timeout_terminates_only_its_owned_process_group(tmp_path):
    pid_file = tmp_path / "simulator.pid"
    script = "import os,pathlib,time; pathlib.Path(r'{}').write_text(str(os.getpid())); time.sleep(30)".format(pid_file)
    config = _config(tmp_path, [sys.executable, "-c", script], simulator_timeout_s=0.2)
    with pytest.raises(TimeoutError, match="simulator exceeded"):
        run(config, tmp_path)
    pid = int(pid_file.read_text())
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("result,expected", [(None, "matched 0 files"), ({"hdscore": float("nan"), "rc": 1}, "finite numeric")])
def test_missing_or_nonfinite_official_metrics_fail(tmp_path, result, expected):
    command = [sys.executable, "-c", "pass"] if result is None else _write_result(result)
    config = _config(tmp_path, command)
    with pytest.raises((ValueError, TypeError), match=expected):
        run(config, tmp_path)


def test_hugsim_output_is_collected_through_subprocess_evaluator_lifecycle(tmp_path, monkeypatch):
    import recogdrive.eval.closedloop as closedloop

    monkeypatch.setattr(closedloop, "vlm_export_path", lambda _: None)
    monkeypatch.setattr(HUGSIMEvaluator, "_devkit_root", staticmethod(lambda: tmp_path))
    evaluator = HUGSIMEvaluator(
        OmegaConf.create({"agent": {"vlm_path": "unused"}}),
        OmegaConf.create({
            "on_end": True,
            "simulator_command": _write_result({"hdscore": 0.73, "rc": 0.91}),
            "result_file": "eval.json",
            "metrics": {"hdscore": "hdscore", "route_completion": "rc"},
            "simulator_timeout_s": 3,
        }), tmp_path,
    )
    checkpoint = tmp_path / "unused.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    evaluator.submit(checkpoint, "scenario")
    assert evaluator.collect(block=True) == [
        ("scenario", {"hdscore": 0.73, "route_completion": 0.91})
    ]


def test_hugsim_base_config_is_copied_and_run_output_isolated(tmp_path, monkeypatch):
    import recogdrive.eval.closedloop as closedloop

    monkeypatch.setattr(closedloop, "vlm_export_path", lambda _: None)
    monkeypatch.setattr(HUGSIMEvaluator, "_devkit_root", staticmethod(lambda: tmp_path))
    source = tmp_path / "nuscenes_base.yaml"
    source.write_text("output_dir: /old/results\nltf_path: /old/agent.sh\nmodel_base: ./models\n")
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "eval" / "hugsim" / "scenario"
    evaluator = HUGSIMEvaluator(
        OmegaConf.create({"agent": {"_target_": "custom.Agent", "vlm_path": "unused", "grpo": True,
                                  "cache_hidden_state": True, "lora_rank": 4,
                                  "trajectory_sampling": {"time_horizon": 4}}}),
        OmegaConf.create({
            "simulator_base_config": str(source),
            "simulator_command": [sys.executable, "-c", "pass", "{hugsim_base_config}"],
            "policy_overrides": ["agent.trajectory_sampling.time_horizon=6", "agent.lora_rank=8"],
        }), tmp_path,
    )
    evaluator.command(checkpoint, output, "scenario")
    copied = OmegaConf.load(output / "config" / "hugsim_base.yaml")
    assert copied.output_dir == str((output / "hugsim_").resolve())
    assert copied.ltf_path.endswith("scripts/infer/run_recogdrive_hugsim.sh")
    assert copied.model_base == "./models"
    assert OmegaConf.load(source).output_dir == "/old/results"
    agent = OmegaConf.load(output / "config" / "agent" / "trained_agent.yaml")
    assert agent._target_ == "custom.Agent"
    assert agent.trajectory_sampling.time_horizon == 6 and agent.lora_rank == 8
    assert agent.grpo is False and agent.cache_hidden_state is False
    runner = json.loads((output / "closedloop_runner.json").read_text())
    assert runner["result_file"] == "hugsim_ltf/*/eval.json"
    assert runner["agent_config_dir"] == str((output / "config").resolve())


def test_standalone_mode_routes_saved_checkpoint_through_configured_hugsim_command(tmp_path, monkeypatch):
    import recogdrive.eval.closedloop as closedloop

    monkeypatch.setattr(closedloop, "vlm_export_path", lambda _: None)
    monkeypatch.setattr(HUGSIMEvaluator, "_devkit_root", staticmethod(lambda: tmp_path))
    training_config = tmp_path / "training.yaml"
    config = OmegaConf.create({
        "agent": {"vlm_path": "unused"},
        "evaluator": {
            "name": None, "gpus": None, "on_end": False,
            "hugsim": {
                "simulator_command": _write_result({"hdscore": 0.81, "rc": 0.95}),
                "result_file": "eval.json",
                "metrics": {"hdscore": "hdscore", "route_completion": "rc"},
                "simulator_timeout_s": 3,
            },
        },
    })
    OmegaConf.save(config, training_config)
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"saved checkpoint")
    output = tmp_path / "standalone"
    monkeypatch.setattr(sys, "argv", [
        "closedloop", "--training-config", str(training_config), "--checkpoint", str(checkpoint),
        "--protocol", "hugsim", "--output-dir", str(output),
    ])
    assert closedloop._main() == 0
    result = json.loads((output / "standalone_result.json").read_text())
    assert result["protocol"] == "hugsim"
    assert result["metrics"] == {"hdscore": 0.81, "route_completion": 0.95}
    assert (output / "checkpoint" / checkpoint.name).read_bytes() == checkpoint.read_bytes()


def test_alpasim_and_neuroncap_official_text_artifacts_keep_metric_names(tmp_path):
    from recogdrive.eval.closedloop import _official_values

    text = tmp_path / "metrics_results.txt"
    text.write_text("collision_at_fault: mean=0.25 std=0.1\noffroad: mean=0.0 std=0.0\n")
    assert _official_values(text, {"collision_at_fault": "collision_at_fault", "offroad": "offroad"}) == {
        "collision_at_fault": 0.25, "offroad": 0.0
    }
    log = tmp_path / "neuroncap.log"
    log.write_text("| Overall | ncap_score | CR@0.0s |\n| model | 2.111 | 0.604 |\n")
    assert _official_values(log, {"ncap_score": "ncap_score", "collision_rate": "CR@0.0s"}) == {
        "ncap_score": 2.111, "collision_rate": 0.604
    }


def test_neuroncap_default_maps_episode_score_without_misreporting_collision_flag(tmp_path):
    from recogdrive.eval.closedloop import DEFAULT_METRICS, _official_values

    result = tmp_path / "metrics.json"
    result.write_text(json.dumps({"ncap_score": 2.111, "any_collide@0.0s": True}))
    assert _official_values(result, DEFAULT_METRICS["neuroncap"]) == {"ncap_score": 2.111}
