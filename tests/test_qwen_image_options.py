from types import SimpleNamespace

import pytest
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from recogdrive.adapters.navsim.agent import ReCogDriveAgent, prepare_vlm_inputs
from recogdrive.data.registry import collate
from recogdrive.models.recogdrive import backbone
from recogdrive.models.recogdrive.utils import vqa_preprocess


def test_qwen_resize_keeps_the_right_view_and_survives_mixed_batches(tmp_path):
    image = Image.new("RGB", (1600, 900))
    path = tmp_path / "front.jpg"
    image.save(path)
    images = [("left", image), ("front", str(path))]
    features = {"camera_paths": images, "history_trajectory": torch.zeros(4, 3),
                "high_command_one_hot": torch.tensor([0., 1., 0., 0.])}
    native = prepare_vlm_inputs(dict(features), False)
    assert "image_short_sides" not in native
    single = prepare_vlm_inputs(dict(features), False, short_sides=(960, 480))
    assert single["image_short_sides"] == [960]
    multi = prepare_vlm_inputs(dict(features), False, cameras=["left", "front"], short_sides=(960, 480))
    sizes = backbone.qwen_image_content(multi["image_paths"], multi["image_short_sides"])
    assert [(item["resized_width"], item["resized_height"]) for item in sizes] == [(853, 480), (1707, 960)]
    left_only = {**features, "camera_paths": images[:1]}
    matched = [prepare_vlm_inputs(sample, False, cameras=["left", "front"], short_sides=(960, None))
               for sample in (dict(features), left_only)]
    batch, _, _ = collate([(sample, {}, str(i)) for i, sample in enumerate(matched)])
    assert batch["image_short_sides"] == [[None, 960], [None]]


@pytest.mark.parametrize("opts,error", [
    ({"qwen_image_short_sides": [960]}, "two positive integers"),
    ({"qwen_image_short_sides": [960, -1]}, "two positive integers"),
    ({"qwen_max_input_tokens": 4, "vlm_type": "internvl"}, "require a Qwen VLM"),
    ({"qwen_max_input_tokens": 4, "cache_hidden_state": True}, "cache_hidden_state=False"),
])
def test_incompatible_options_fail_before_model_loading(opts, error):
    config = dict(vlm_type="qwen", cache_hidden_state=False)
    config.update(opts)
    with pytest.raises(ValueError, match=error):
        ReCogDriveAgent(TrajectorySampling(num_poses=8, interval_length=0.5), **config)


def test_qwen_default_accepts_long_prompts_and_explicit_budget_rejects_them(monkeypatch):
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return BatchFeature({"input_ids": torch.ones(1, 2801, dtype=torch.long)})

    calls = []
    monkeypatch.setattr(backbone, "process_vision_info", lambda messages: ([], []))
    policy = SimpleNamespace(device="cpu", max_length=2800, qwen_max_input_tokens=None,
                             system_prompt=backbone.system_message, tokenizer=Processor(),
                             model=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(hidden_states=["ok"]))
    forward = backbone.RecogDriveBackbone.forward_qwen
    assert forward(policy, ["image.jpg"], ["Q?"]) == "ok"
    policy.qwen_max_input_tokens = 2800
    with pytest.raises(ValueError, match="qwen_max_input_tokens=2800"):
        forward(policy, ["image.jpg"], ["Q?"])
    assert len(calls) == 1


def test_qwen_vqa_uses_the_configured_image_sizes(monkeypatch, tmp_path):
    path = tmp_path / "view.png"
    Image.new("RGB", (1600, 900)).save(path)
    captured = []

    class Processor:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return "prompt" if add_generation_prompt else "full"

        def __call__(self, *, text, **kwargs):
            return BatchFeature({"input_ids": torch.tensor([[1, 2]] if text == ["prompt"] else [[1, 2, 3]])})

    def vision(messages):
        captured.append(messages[1]["content"][:-1])
        return [], None

    monkeypatch.setattr(vqa_preprocess, "process_vision_info", vision)
    example = {"image_paths": [str(path)] * 2, "question": "Q?", "answer": "A."}
    vqa_preprocess.prepare_vqa_features(example, Processor(), "qwen", 4)
    features = vqa_preprocess.prepare_vqa_features(example, Processor(), "qwen", 4, short_sides=(960, 480))
    assert all("resized_height" not in part for part in captured[0])
    assert [part["resized_height"] for part in captured[1]] == [960, 480]
    assert features["labels"].tolist() == [-100, -100, 3]
