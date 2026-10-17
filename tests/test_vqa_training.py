import sys
import json
from unittest.mock import patch
from types import ModuleType, SimpleNamespace

import torch
from torch import nn
from omegaconf import OmegaConf

from recogdrive.adapters.navsim.agent import ReCogDriveAgent
from recogdrive.data.registry import collate
from recogdrive.data.vqa import vqa_loader
from recogdrive.models.recogdrive.backbone import RecogDriveBackbone
from recogdrive.models.recogdrive.utils.vqa_preprocess import answer_only_tensors


def test_vqa_source_only_requires_the_adapter_transform(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"image": "front.jpg", "question": "What color?", "answer": "Gray."}) + "\n")

    class Policy:
        def vqa_worker_transform(self, max_length=None):
            return lambda features: {**features, "prepared": True}

    dataset = vqa_loader(OmegaConf.create({"vqa": {"train_jsonl": str(source)}}), Policy(), "train")
    features, targets, token = dataset[0]
    assert features["prepared"] and features["task"] == "vqa" and token == "1"
    assert targets["answer"] == "Gray."


def test_planning_inference_rejects_vqa_before_allocating_a_model():
    from recogdrive.infer import run

    with patch("recogdrive.distributed.DistributedContext", side_effect=AssertionError("must not allocate")):
        try:
            run(OmegaConf.create({"data_loader": "vqa", "infer": {}}))
        except ValueError as exc:
            assert "VQA answer generation" in str(exc)
        else:
            raise AssertionError("planning inference accepted VQA samples")


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16)

    def forward(self, input_ids, logits_to_keep, **kwargs):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden[:, -logits_to_keep:]))


class FakePlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.called = False

    def forward(self, hidden, inputs):
        self.called = True
        return SimpleNamespace(loss=self.weight + hidden.sum() * 0)


class FakeTunerLayer(nn.Module):
    active_adapters = ["default"]

    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.delta = nn.Parameter(torch.ones_like(self.base_layer.weight))
        self.merged_adapters = []

    def merge(self, safe_merge=False):
        with torch.no_grad():
            self.base_layer.weight.add_(self.delta)
        self.merged_adapters.append("default")

    def get_base_layer(self):
        return self.base_layer

    def forward(self, x):
        return self.base_layer(x) + nn.functional.linear(x, self.delta)


def fake_agent():
    backbone = RecogDriveBackbone.__new__(RecogDriveBackbone)
    nn.Module.__init__(backbone)
    backbone.model_type = "qwen"
    backbone.model = FakeQwen()

    agent = ReCogDriveAgent.__new__(ReCogDriveAgent)
    nn.Module.__init__(agent)
    agent.backbone = backbone
    agent.action_head = FakePlanner()
    agent.device = "cpu"
    agent.vlm_trainable = True
    agent.cache_hidden_state = True
    agent.grpo = False
    agent._lr = 1e-3
    agent.optimizer_type = "adamw"
    agent.lr_schedule = "step"
    return agent


def test_vqa_labels_gradient_routing_and_planning_path():
    short = answer_only_tensors([1, 2], [1, 2, 3, 4], max_length=6)
    long = answer_only_tensors([5, 6], [5, 6, 7, 8, 9], max_length=6)
    assert short["input_ids"].tolist() == [1, 2, 3, 4]
    assert short["labels"].tolist() == [-100, -100, 3, 4]
    features, targets, _ = collate([
        ({**short, "task": "vqa", "answer_lengths": short["answer_length"]}, {"answer": "ok"}, "short"),
        ({**long, "task": "vqa", "answer_lengths": long["answer_length"]}, {"answer": "yes"}, "long"),
    ])
    assert features["input_ids"].tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert features["labels"].tolist() == [[-100, -100, -100, 3, 4], [-100, -100, 7, 8, 9]]

    agent = fake_agent()
    loss = agent(features, targets).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert agent.backbone.model.embedding.weight.grad is not None

    try:
        agent({"task": ["vqa", "planning"]}, {})
    except ValueError as exc:
        assert "mixed task" in str(exc)
    else:
        raise AssertionError("mixed task batch was accepted")

    agent.vlm_trainable = False
    try:
        agent({"task": ["vqa"]}, {})
    except RuntimeError as exc:
        assert "trainable VLM" in str(exc)
    else:
        raise AssertionError("VQA training accepted a frozen VLM")
    agent.vlm_trainable = True

    planner = agent.action_head
    out = agent(
        {"task": ["planning"], "history_trajectory": torch.zeros(1, 4, 3),
         "high_command_one_hot": torch.tensor([[0., 1., 0., 0.]]), "status_feature": torch.zeros(1, 8),
         "last_hidden_state": torch.ones(1, 2, 8)},
        {"trajectory": torch.zeros(1, 2, 3)},
    )
    assert planner.called and out.loss.item() == 1.0


def test_vqa_optimizer_includes_trainable_vlm_parameters():
    agent = fake_agent()
    cfg = agent.get_optimizers(total_steps=10, steps_per_epoch=2)
    params = {id(parameter) for group in cfg["optimizer"].param_groups for parameter in group["params"]}
    assert id(agent.backbone.model.embedding.weight) in params


def test_trainable_vlm_modes_reject_cached_feature_paths():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    for mode, cache_hidden_state, cache_mode in (
        ("train_backbone", True, False),
        ("train_backbone", False, True),
        ("lora_rank", True, False),
        ("lora_rank", False, True),
    ):
        options = {mode: True, "cache_hidden_state": cache_hidden_state, "cache_mode": cache_mode}
        try:
            ReCogDriveAgent(trajectory_sampling, **options)
        except ValueError as exc:
            assert "needs the online VLM" in str(exc)
            assert "cache_hidden_state=False and cache_mode=False" in str(exc)
        else:
            raise AssertionError(f"trainable VLM configuration accepted cached inputs: {options}")


def test_lora_export_merges_and_restores_base_layout(monkeypatch):
    peft = ModuleType("peft")
    peft.__path__ = []
    tuners = ModuleType("peft.tuners")
    tuners.__path__ = []
    tuner_utils = ModuleType("peft.tuners.tuners_utils")
    tuner_utils.BaseTunerLayer = FakeTunerLayer
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.tuners", tuners)
    monkeypatch.setitem(sys.modules, "peft.tuners.tuners_utils", tuner_utils)

    backbone = RecogDriveBackbone.__new__(RecogDriveBackbone)
    nn.Module.__init__(backbone)
    backbone.model = nn.Sequential(FakeTunerLayer())
    layer = backbone.model[0]
    expected = layer.base_layer.weight.detach() + layer.delta.detach()
    merged = backbone.merge_lora_for_export()
    assert isinstance(merged[0], nn.Linear)
    assert torch.allclose(merged[0].weight, expected)


def test_vqa_internvl_uses_front_and_side_image_budgets(monkeypatch):
    from recogdrive.models.recogdrive.utils import vqa_preprocess

    calls = []

    def fake_load_image(path, short_side=None):
        calls.append((path, short_side))
        return torch.zeros({960: 4, 480: 1, None: 2}[short_side], 3, 2, 2)

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

    monkeypatch.setattr(vqa_preprocess, "load_image", fake_load_image)
    agent = ReCogDriveAgent.__new__(ReCogDriveAgent)
    nn.Module.__init__(agent)
    agent.backbone = SimpleNamespace(tokenizer=FakeTokenizer(), max_length=256, num_image_token=2)
    agent.vlm_type = "internvl"
    agent.short_sides = (960, 480)
    sample = {"task": "vqa", "image_paths": [f"view{i}" for i in range(6)],
              "question": "What is visible?", "answer": "A vehicle."}

    features = agent.vqa_worker_transform()(sample)
    assert calls == [("view0", 960)] + [(f"view{i}", 480) for i in range(1, 6)]
    assert features["num_patches"].tolist() == [4, 1, 1, 1, 1, 1]
    assert features["pixel_values"].shape[0] == 9

    calls.clear()
    agent.short_sides = (None, None)
    no_resize = agent.vqa_worker_transform()(sample)
    assert calls == [(f"view{i}", None) for i in range(6)]
    assert no_resize["num_patches"].tolist() == [2] * 6
