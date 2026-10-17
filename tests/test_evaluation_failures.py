"""Requested benchmark failures remain failures after the distributed training phase."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import os
import sys
import time

import pytest
from torch import nn
from omegaconf import OmegaConf

from recogdrive.eval.registry import SubprocessEvaluator
from recogdrive.train.trainer import Trainer
from recogdrive.models.recogdrive.diffusion_planner import GRPOConfig


def test_single_candidate_cannot_produce_group_relative_advantages():
    with pytest.raises(ValueError, match="at least 2"):
        GRPOConfig(sample_time=1)


def test_cancel_reaps_its_owned_parent_and_child_group(tmp_path):
    script = (
        "import json,os,pathlib,signal,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        "signal.signal(signal.SIGTERM,lambda *_: (child.wait(),sys.exit(0)))\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid]))\n"
        "time.sleep(60)\n"
    )
    pids_path = tmp_path / "pids.json"
    evaluator = SubprocessEvaluator(OmegaConf.create({}), OmegaConf.create({}), tmp_path)
    try:
        evaluator.launch([sys.executable, "-c", script, str(pids_path)], tmp_path, "pending", os.environ.copy())
        deadline = time.monotonic() + 5
        while not pids_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pids_path.is_file()
        evaluator.cancel()
        assert not evaluator.jobs and not evaluator.deferred
        if os.name == "posix":
            for pid in json.loads(pids_path.read_text()):
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
    finally:
        evaluator.cancel()


@pytest.mark.parametrize("result", [{}, {"score": float("nan")}, {"score": float("inf")}, {"csv": "output.csv"}, {"status": True}])
def test_completed_evaluator_rejects_empty_or_nonfinite_metrics(tmp_path, result):
    evaluator = SubprocessEvaluator(OmegaConf.create({}), OmegaConf.create({}), tmp_path)
    evaluator.read_result = Mock(return_value=result)
    evaluator.jobs = [("final", SimpleNamespace(returncode=0, poll=lambda: 0), tmp_path)]
    collected = evaluator.poll()
    assert len(collected) == 1 and "error" in collected[0][1]


@pytest.mark.parametrize("raise_from_finish", [False, True])
def test_eval_failure_is_raised_after_cleanup_and_all_final_evaluators(tmp_path, raise_from_finish):
    trainer = Trainer.__new__(Trainer)
    trainer.output_dir = tmp_path
    trainer.agent = nn.Linear(1, 1)
    trainer.ema = None
    trainer.ckpt_suffixes = lambda: [""]
    trainer.ctx = SimpleNamespace(is_main=True, close=Mock())
    trainer.monitor = SimpleNamespace(log=Mock(), close=Mock())
    trainer.evaluation_errors = []
    trainer.global_step = trainer.epoch = 0
    bad = SimpleNamespace(benchmark="bad", release=Mock(), finish=Mock(return_value=[("final", {"error": "exit 7"})]))
    if raise_from_finish:
        bad.finish.side_effect = ValueError("bad simulator configuration")
    good = SimpleNamespace(benchmark="good", release=Mock(), finish=Mock(return_value=[("final", {"score": 1.0})]))
    trainer.evaluators = [bad, good]
    # A periodic failure is remembered while the other distributed ranks can keep training.
    trainer.report(bad, [("epoch_0001", {"error": "missing score"})])
    with pytest.raises(RuntimeError, match="2 requested evaluations failed"):
        trainer.finish()
    trainer.ctx.close.assert_called_once()
    trainer.monitor.close.assert_called_once()
    good.finish.assert_called_once()
    failures = json.loads((tmp_path / "evaluation_errors.json").read_text())
    assert [failure["tag"] for failure in failures] == ["epoch_0001", "final"]
