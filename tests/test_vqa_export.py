import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
import transformers.modeling_utils as modeling_utils

from recogdrive.eval.vqa import DriveLMEvaluator
from recogdrive.eval.vqa import export


def test_lora_checkpoint_is_merged_from_cpu_copy_and_cached(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def parameters(self):
            return iter((self.weight,))

        def state_dict(self):
            return {"proj.weight": self.weight}

        def save_pretrained(self, path, **kwargs):
            assert modeling_utils.unwrap_model(self) is self
            Path(path, "config.json").write_text("{}", encoding="utf-8")
            Path(path, "model.safetensors").write_bytes(b"fake model weights")

    class FakeTokenizer:
        def save_pretrained(self, path):
            Path(path, "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class FakeBackbone:
        def __init__(self):
            self.model = FakeModel()
            self.tokenizer = FakeTokenizer()
            self.adapter = torch.nn.Parameter(torch.tensor([-9.0]))

        def merge_lora_for_export(self):
            with torch.no_grad():
                self.model.weight.add_(self.adapter)
            return self.model

    class FakeAgent:
        def __init__(self):
            self.backbone = FakeBackbone()

        def state_dict(self):
            return {"backbone.model.proj.lora_A.default.weight": self.backbone.adapter}

        def load_state_dict(self, state, strict=False):
            with torch.no_grad():
                self.backbone.adapter.copy_(state["backbone.model.proj.lora_A.default.weight"])

    checkpoint = tmp_path / "epoch.ckpt"
    agent_config = tmp_path / "agent.yaml"
    agent_config.write_text("lora_rank: 2\n", encoding="utf-8")
    torch.save({"state_dict": {
        "backbone.model.proj.lora_A.default.weight": torch.tensor([5.0]),
    }}, checkpoint)

    training_agent = FakeAgent()
    export_agents = []
    original_unwrap = modeling_utils.unwrap_model

    def make_agent(_):
        agent = FakeAgent()
        export_agents.append(agent)
        return agent

    monkeypatch.setattr(export, "_instantiate_agent", make_agent)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    key, first = export.acquire_export(checkpoint, agent_config, tmp_path / "cache")
    key2, second = export.acquire_export(checkpoint, agent_config, tmp_path / "cache")
    assert key == key2 and first == second
    assert len(export_agents) == 1
    export_agent = export_agents[0]
    assert export_agent.backbone.adapter.item() == 5.0  # checkpoint overwrote injected defaults
    assert export_agent.backbone.model.weight.item() == 6.0
    assert modeling_utils.unwrap_model is original_unwrap
    assert training_agent.backbone.adapter.item() == -9.0
    assert training_agent.backbone.model.weight.item() == 1.0
    assert json.loads((first / "recogdrive-export.json").read_text())["fingerprint"] == key
    export.release_export(key, first, tmp_path / "cache")
    export.release_export(key, second, tmp_path / "cache")
    assert not first.exists()

    torch.save({"state_dict": {
        "backbone.model.proj.lora_A.default.weight": torch.tensor([8.0]),
    }}, checkpoint)
    changed_key, changed = export.acquire_export(checkpoint, agent_config, tmp_path / "cache")
    assert changed_key != key and len(export_agents) == 2
    assert export_agents[1].backbone.model.weight.item() == 9.0
    export.release_export(changed_key, changed, tmp_path / "cache")


def test_vqa_evaluator_routes_lora_through_export_and_keeps_regular_path(tmp_path):
    torch.save({"state_dict": {}, "vlm_export": False}, tmp_path / "planner.ckpt")
    cfg = OmegaConf.create({
        "agent": {"lora_rank": 4, "vlm_path": "/models/base", "checkpoint_path": "/old.ckpt", "grpo": True},
    })
    section = OmegaConf.create({"gpus": "2", "vqa": {"data": "questions.json", "backend": "pytorch"}})
    evaluator = DriveLMEvaluator(cfg, section, tmp_path)
    command = evaluator.command(tmp_path / "planner.ckpt", tmp_path / "eval", "epoch_1")
    boundary = command.index("--")
    assert command[1:3] == ["-m", "recogdrive.eval.vqa.export"]
    assert command[command.index("--cache-root") + 1] == str(tmp_path / ".vqa_exports")
    assert command[boundary + 1:boundary + 4] == ["drivelm", "--model", export.MODEL_TOKEN]
    assert command[boundary + 4:] == ["--out", str(tmp_path / "eval"), "--data", "questions.json",
                                       "--backend", "pytorch"]
    saved_agent = OmegaConf.load(tmp_path / "eval" / "vqa_agent.yaml")
    assert saved_agent.checkpoint_path is None and saved_agent.grpo is False and saved_agent.lora_rank == 4
    child_env = evaluator.child_env()
    assert child_env["CUDA_VISIBLE_DEVICES"] == ""
    assert child_env[export.CUDA_ENV] == "2"

    cfg.agent.lora_rank = 0
    regular = DriveLMEvaluator(cfg, section, tmp_path)
    regular_cmd = regular.command(tmp_path / "planner.ckpt", tmp_path / "eval", "epoch_1")
    assert regular_cmd[1:3] == ["-m", "recogdrive.eval.vqa"]
    assert "recogdrive.eval.vqa.export" not in regular_cmd
    assert export.CUDA_ENV not in regular.child_env()


def test_unwrap_override_is_restored_when_hf_save_fails(tmp_path):
    original_unwrap = modeling_utils.unwrap_model

    class FailingModel:
        def save_pretrained(self, path, **kwargs):
            assert modeling_utils.unwrap_model(self) is self
            raise RuntimeError("synthetic save failure")

    try:
        export._save_pretrained_unwrapped(FailingModel(), tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "synthetic save failure"
    else:
        raise AssertionError("expected synthetic save failure")
    assert modeling_utils.unwrap_model is original_unwrap

    class WrappedModel:
        module = FailingModel()

    try:
        export._save_pretrained_unwrapped(WrappedModel(), tmp_path)
    except TypeError:
        pass
    else:
        raise AssertionError("expected wrapped model to be rejected")
    assert modeling_utils.unwrap_model is original_unwrap
