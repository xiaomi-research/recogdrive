import json
from pathlib import Path

from vqa_evaluation.DriveBench.inference.utils import save_predictions


def test_predictions_preserve_an_existing_output_directory(tmp_path):
    class Predictions:
        def write_json(self, path):
            (Path(path) / "part.json").write_text('{"question_id": 1, "answer": "A"}\n')

    output = tmp_path / "a prediction run"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep")
    save_predictions(Predictions(), str(output))
    assert marker.read_text() == "keep"
    assert json.loads(output.with_suffix(".json").read_text()) == [{"question_id": 1, "answer": "A"}]
    assert not list(tmp_path.glob(".drivebench-*"))
