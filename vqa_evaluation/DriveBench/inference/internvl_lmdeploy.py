"""
This example shows how to use Ray Data for running offline batch inference
with InternVL models using the transformers library, distributively on a multi-node cluster.
"""

import os
import warnings
import argparse
from typing import Any, Dict, List

import torch
import ray
import numpy as np 
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from PIL import Image
from packaging.version import Version

from lmdeploy import (
    ChatTemplateConfig,
    GenerationConfig,
    PytorchEngineConfig,
    TurbomindEngineConfig,
    pipeline,
)
from lmdeploy.vl import load_image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from utils import replace_system_prompt, save_predictions, replace_system_prompt_new


assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"
warnings.filterwarnings("ignore", message="The `use_flash_attention_2` flag is deprecated")


system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""

# --- InternVL Image Preprocessing Functions ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Builds the image transformation pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the best aspect ratio for tiling the image."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Dynamically splits the image into tiles."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def process_pil_image(image: Image.Image, image_size: int, max_num: int):
    """Processes a PIL image into pixel value tensors for InternVL."""
    transform = build_transform(input_size=image_size)
    images = dynamic_preprocess(image, image_size=image_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def parse_arguments():
    parser = argparse.ArgumentParser(description='InternVL Multi-GPU Inference with Transformers and Ray')
    parser.add_argument('--model', type=str, required=True, help='Path to InternVL model')
    parser.add_argument('--data', type=str, required=True, help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--system_prompt', type=str, required=True, help='System prompt file')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of GPUs to use')
    parser.add_argument('--image_size', type=int, default=448, help='Image size for preprocessing')
    parser.add_argument('--max_num_tiles', type=int, default=12, help='Maximum number of tiles per image')
    parser.add_argument('--corruption', type=str, default='', help='Corruption type')
    parser.add_argument('--temperature', type=float, default=0.01, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.001, help='Top-p for sampling')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate')
    return parser.parse_args()


class LLMPredictor:
    def __init__(self, model_path: str, system_prompt: str, generation_config: Dict,
                 image_size: int, max_num_tiles: int, corruption: str):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pipe = pipeline(
            model_path,
            backend_config=TurbomindEngineConfig(session_len=30000,dtype='bfloat16'),
            chat_template_config=ChatTemplateConfig(model_name='internvl2_5', meta_instruction=system_prompt)
        )
        
        self.system_prompt = system_prompt
        self.generation_config = generation_config
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.corruption = corruption

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        questions = batch['question']
        image_paths_batch = batch['image_path']
        batch_size = len(questions)

        prompts = []
        valid_indices = []

        # generation_config = GenerationConfig(
        #         max_new_tokens=self.generation_config["max_new_tokens"],
        #         min_new_tokens=50,
        #         do_sample=self.generation_config["do_sample"],
        #         temperature=self.generation_config["temperature"],
        #         top_p=self.generation_config["top_p"]
        # )
        generation_config = GenerationConfig(
                max_new_tokens=4096,
                min_new_tokens=50,
                do_sample=True,
                temperature=0.01,
                top_p=0.001,
                top_k=1
        )
        for i in range(batch_size):
            sample_filenames = image_paths_batch[i]
            sample_question = questions[i]

            # 拼接路径
            relative_paths = list(sample_filenames.values())
            image_paths = [
                os.path.join("/mnt/evad_fs/worldmodel/yongkangli/Dataset_vqa/toolkit-main", p)
                for p in relative_paths if isinstance(p, str)
            ]

            sample_images = []
            for img_path in image_paths:
                if self.corruption and len(self.corruption) > 1 and self.corruption != 'NoImage':
                    img_path = img_path.replace(
                        'nuscenes/samples', f'corruption/{self.corruption}'
                    )

                if self.corruption == 'NoImage':
                    # 生成空白图
                    blank_np_array = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                    img = Image.fromarray(blank_np_array)
                    sample_images.append(img)
                else:
                    try:
                        img = load_image(img_path)  # ✅ 新的 load_image
                        sample_images.append(img)
                    except Exception as e:
                        print(f"Warning: Could not load image {img_path}, skipping. Error: {e}")

            if not sample_images:
                continue

            # 替换系统 prompt
            updated_system_prompt, image_placeholders = replace_system_prompt_new(
                self.system_prompt, image_paths
            )
            prompt_text = f"{image_placeholders}{updated_system_prompt}{sample_question}"

            # ✅ 每个 sample 的 prompt 是 (text, [img1, img2, ...])
            prompts.append((prompt_text, sample_images))
            valid_indices.append(i)

        if not prompts:
            batch['pred'] = [None] * batch_size
            return batch

        # === 调用 lmdeploy ===
        responses = self.pipe(prompts, gen_config=generation_config)

        # === 结果对齐 ===
        full_predictions = [None] * batch_size
        for idx, resp in zip(valid_indices, responses):
            full_predictions[idx] = resp.text if hasattr(resp, "text") else str(resp)

        batch['pred'] = full_predictions
        return batch


def main():
    args = parse_arguments()

    with open(args.system_prompt, 'r') as f:
        system_prompt = f.read()

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True if args.temperature > 0 else False,
    }

    num_instances = args.num_processes

    ds = ray.data.read_json(args.data)

    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_args=(
            args.model,
            system_prompt,
            generation_config,
            args.image_size,
            args.max_num_tiles,
            args.corruption
        ),
        num_gpus=1,
        concurrency=num_instances,
        batch_size=4,
    )

    save_predictions(ds, args.output)
    

if __name__ == '__main__':
    main()
