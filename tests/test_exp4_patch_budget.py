from PIL import Image
import pytest
import torch

from recogdrive.adapters.navsim.agent import prepare_vlm_inputs
from recogdrive.models.recogdrive.utils.internvl_preprocess import load_image
from recogdrive.models.recogdrive.utils.vqa_preprocess import prepare_vqa_features


@pytest.mark.parametrize("size,default_tiles", [((1600, 900), 9), ((900, 1600), 9), ((640, 640), 5)])
def test_fixed_tiles_and_default_tiling(size, default_tiles):
    image = Image.new("RGB", size, "orange")
    assert load_image(image, short_side=960, num_patches=3).shape == (3, 3, 448, 448)
    assert load_image(image, short_side=480, num_patches=1).shape == (1, 3, 448, 448)
    assert load_image(image).shape == (default_tiles, 3, 448, 448)


def test_exp4_uses_fixed_tiles_in_planning_and_vqa_and_rejects_missing_views(tmp_path):
    cameras = ["front", "front_left", "front_right", "left", "right", "back_left", "back_right", "back"]
    image = tmp_path / "view.jpg"
    Image.new("RGB", (1600, 900)).save(image)
    views = [(camera, str(image)) for camera in cameras]
    features = {"history_trajectory": torch.zeros(4, 3), "high_command_one_hot": torch.tensor([0., 1., 0., 0.]),
                "camera_paths": views}
    kwargs = {"short_sides": (960, 480), "patch_counts": (3, 1)}
    planned = prepare_vlm_inputs(features, True, cameras=cameras, **kwargs)
    assert planned["num_patches"].tolist() == [3] + [1] * 7
    assert planned["pixel_values"].shape == (10, 3, 448, 448)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

    vqa = prepare_vqa_features({"image_paths": [str(image)] * 8, "question": "Q?", "answer": "A."},
                               Tokenizer(), "internvl", 400, num_image_token=2, **kwargs)
    assert vqa["num_patches"].tolist() == planned["num_patches"].tolist()
    for missing in ([], views[:1], views[:-1]):
        with pytest.raises(ValueError, match="rebuild the multi-view cache"):
            prepare_vlm_inputs({**features, "camera_paths": missing}, True, cameras=cameras, **kwargs)
