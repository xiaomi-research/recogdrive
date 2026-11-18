import asyncio
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from PIL import Image

from recogdrive.data import cognitive
from recogdrive.data.vqa import VQADataset


def configure_cache(monkeypatch, tmp_path):
    image = tmp_path / "front.jpg"
    Image.new("RGB", (32, 18), "red").save(image)
    monkeypatch.setattr(cognitive, "_cache_dir", tmp_path / ".cache")
    monkeypatch.setattr(cognitive, "_model_id", "test-vlm")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    calls = []

    async def infer(question, path):
        calls.append((question, path))
        return f"answer {len(calls)}"

    monkeypatch.setattr(cognitive, "infer", infer)
    return image, calls


def test_actual_generated_conversations_load_as_self_contained_turns(monkeypatch, tmp_path):
    image, _ = configure_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(cognitive, "select_images", lambda *args: ([str(image)], "<image>\n", "front view"))
    status = SimpleNamespace(ego_pose=[0, 0, 0], driving_command=[0, 1, 0])
    row = asyncio.run(cognitive.process_data_and_create_qa_pair(
        17, [status], [None], None, [], [], [], "scene", None, None,
    ))
    path = tmp_path / "generated.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = VQADataset(path, tmp_path)
    assert len(dataset) == len(row["conversations"]) // 2 >= 5
    for index in range(len(dataset)):
        features, targets, token = dataset[index]
        assert token == f"17:{index}"
        assert features["image_paths"] == [str(image)]
        assert "<image>" not in features["question"]
        assert features["answer"] == targets["answer"]


@pytest.mark.parametrize("turns,match", [
    ([], "complete"),
    ([{"from": "human", "value": "question"}], "complete"),
    ([{"from": "gpt", "value": "q"}, {"from": "human", "value": "a"}], "alternating"),
    ([{"from": "human", "value": "<image>"}, {"from": "gpt", "value": "a"}], "non-empty"),
    ([{"from": "human", "value": "q"}, {"from": "gpt", "value": ""}], "non-empty"),
])
def test_malformed_conversations_are_rejected(tmp_path, turns, match):
    path = tmp_path / "vqa.jsonl"
    path.write_text(json.dumps({"image": "front.jpg", "conversations": turns}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        VQADataset(path, tmp_path)


def test_flat_schema_and_image_root_validation_are_preserved(tmp_path):
    path = tmp_path / "vqa.jsonl"
    row = {"id": "original", "image": "front.jpg", "question": "Q?", "answer": "A."}
    path.write_text(json.dumps(row), encoding="utf-8")
    assert VQADataset(path, tmp_path)[0][2] == "original"
    row["image"] = "../escape.jpg"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        VQADataset(path, tmp_path)


def test_cache_identity_includes_image_question_model_and_endpoint(monkeypatch, tmp_path):
    image, calls = configure_cache(monkeypatch, tmp_path)

    def query(question="Q?"):
        return asyncio.run(cognitive.vlm_qa("description", "same-scene", str(image), question))[1]

    assert query() == query() == "answer 1"
    assert query("Different question?") == "answer 2"
    Image.new("RGB", (32, 18), "blue").save(image)
    assert query() == "answer 3"
    monkeypatch.setattr(cognitive, "_model_id", "other-vlm")
    assert query() == "answer 4"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.invalid/v1")
    assert query() == "answer 5"
    assert len(calls) == 5


def test_generation_namespaces_geometry_cache_and_publishes_only_complete_output(monkeypatch, tmp_path):
    monkeypatch.setattr(cognitive, "open_loader", lambda *args: SimpleNamespace(tokens=["scene"]))
    monkeypatch.setattr(cognitive, "load_sample", lambda *args: ())
    namespaces = []

    async def process(*args, **kwargs):
        namespaces.append(cognitive._cache_dir)
        return {"image": ["front.jpg"], "conversations": [
            {"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A."},
        ]}

    monkeypatch.setattr(cognitive, "process_data_and_create_qa_pair", process)
    cfg = OmegaConf.create({"seed": 0})
    path = tmp_path / "vqa.jsonl"
    for view in ["single", "multi_3"]:
        assert asyncio.run(cognitive.generate(cfg, path, tmp_path / ".cache", "train", view, "base", None, 1)) == 1
    assert namespaces[0] != namespaces[1]
    cfg.output_dir = "/different/run/timestamp"
    assert asyncio.run(cognitive.generate(cfg, path, tmp_path / ".cache", "train", "single", "base", None, 1)) == 1
    assert namespaces[0] == namespaces[2]
    complete = path.read_bytes()

    async def fail(*args, **kwargs):
        raise RuntimeError("VLM unavailable")

    monkeypatch.setattr(cognitive, "process_data_and_create_qa_pair", fail)
    with pytest.raises(RuntimeError, match="VLM unavailable"):
        asyncio.run(cognitive.generate(cfg, path, tmp_path / ".cache", "train", "single", "base", None, 1))
    assert path.read_bytes() == complete

    async def empty(*args, **kwargs):
        return None

    monkeypatch.setattr(cognitive, "process_data_and_create_qa_pair", empty)
    with pytest.raises(RuntimeError, match="no valid"):
        asyncio.run(cognitive.generate(cfg, path, tmp_path / ".cache", "train", "single", "base", None, 1))
    assert path.read_bytes() == complete
