import json
from pathlib import Path

import pytest
import requests


@pytest.fixture
def evaluator(monkeypatch):
    drivebench = Path(__file__).resolve().parents[1] / "vqa_evaluation" / "DriveBench"
    monkeypatch.syspath_prepend(str(drivebench))
    from evaluate import eval as evaluator

    return evaluator


def sample():
    return {"question_type": "behavior", "tag": [0], "scene_token": "scene", "frame_token": "frame",
            "question": "What should the car do?", "answer": "B", "pred": "B"}


def test_offline_mcq_does_not_need_judge_files(evaluator, tmp_path):
    result = evaluator.EvaluationSuit(None, str(tmp_path / "judge.jsonl"),
                                    tmp_path / "missing_desc.json", eval_gpt=False)
    result.forward(sample())
    assert result.evaluation()["behavior"]["MCQ"]["accuracy"] == 1.0
    assert not (tmp_path / "judge.jsonl").exists()


def test_gpt_requires_credentials_before_loading_metadata(evaluator):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        evaluator.EvaluationSuit(None, None, None, eval_gpt=True)


def test_gpt_calls_judge_and_resumes_only_for_same_prediction(evaluator, monkeypatch, tmp_path):
    from evaluate import request

    calls = []

    def post(url, *, headers, json, timeout):
        calls.append(json)
        assert timeout == 30 and headers["Authorization"] == "Bearer fixture-key"
        response = requests.Response()
        response.status_code = 200
        response._content = (b'{"choices":[{"message":{"content":"Total Score: 87"}}],'
                             b'"usage":{"total_tokens":12}}')
        return response

    monkeypatch.setattr(request.requests, "post", post)
    desc = tmp_path / "visual_description.json"
    desc.write_text(json.dumps({"scene": {"key_frames": {"frame": {"key_object_infos": {}}}}}))
    log = tmp_path / "judge.jsonl"
    suite = evaluator.EvaluationSuit("fixture-key", str(log), desc, temperature=0.2, max_tokens=512)
    row = sample()
    assert suite.gpt_score_eval.forward(({**row, "desc": None}, "{QUESTION} {GT} {PRED} {DESC}")) == 87
    assert len(calls) == 1 and calls[0]["temperature"] == 0.2 and calls[0]["max_tokens"] == 512

    resumed = evaluator.EvaluationSuit("fixture-key", str(log), desc, temperature=0.2, max_tokens=512)
    assert resumed.gpt_score_eval.forward(({**row, "desc": None}, "{QUESTION}")) == 87
    assert len(calls) == 1
    assert resumed.gpt_score_eval.forward(({**row, "desc": None, "pred": "C"}, "{QUESTION}")) == 87
    assert len(calls) == 2

    changed = evaluator.EvaluationSuit("fixture-key", str(log), desc, temperature=0.3, max_tokens=512)
    assert changed.gpt_score_eval.forward(({**row, "desc": None}, "{QUESTION}")) == 87
    assert len(calls) == 3 and calls[-1]["temperature"] == 0.3


@pytest.mark.parametrize("status, attempts", [(401, 1), (429, 3)])
def test_gpt_failures_are_bounded(evaluator, monkeypatch, tmp_path, status, attempts):
    from evaluate import request

    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        response = requests.Response()
        response.status_code = status
        return response

    monkeypatch.setattr(request.requests, "post", post)
    monkeypatch.setattr(request.time, "sleep", lambda _: None)
    judge = request.GPTEvaluation("fixture-key", log_file=str(tmp_path / "judge.jsonl"))
    with pytest.raises(requests.HTTPError):
        judge.call_chatgpt("example")
    assert len(calls) == attempts
