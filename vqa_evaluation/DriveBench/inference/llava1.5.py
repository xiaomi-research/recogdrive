"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-node cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""


import os
import warnings
import argparse
import numpy as np
from PIL import Image
from typing import Any, Dict
from packaging.version import Version

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

from inference.utils import replace_system_prompt, save_predictions


assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"


def parse_arguments():
    parser = argparse.ArgumentParser(description='VLM Multi-GPU Inference')
    parser.add_argument('--model', type=str, required=True, help='VLMs')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--system_prompt', type=str, required=True,
                        help='System prompt file')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--max_model_len', type=int, default=8192,
                        help='Maximum model length')
    parser.add_argument('--num_images_per_prompt', type=int, default=6,
                        help='Maximum number of images per prompt')
    parser.add_argument('--corruption', type=str, default='',
                        help='Corruption type')
    # Hyperparameters
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.2,
                        help='Top-p for sampling')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    return parser.parse_args()


class LLMPredictor:
    def __init__(self, model_name, system_prompt, sampling_params,
                 num_images_per_prompt, max_model_len, tensor_parallel_size, corruption):
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": num_images_per_prompt},
            tensor_parallel_size=tensor_parallel_size
        )
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params
        self.corruption = corruption

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        
        questions = batch['question']
        filenames = batch['image_path']
        batch_size = len(questions)

        # Load images and build image placeholders and multi_modal_data
        image_placeholders = [''] * batch_size
        multi_modal_datas = [dict(image=[]) for _ in range(batch_size)]
        system_prompts = [self.system_prompt] * batch_size

        for idx, sample_filenames in enumerate(filenames):
            image_paths = list(sample_filenames.values())
            image_index = 1
            system_prompts[idx] = replace_system_prompt(system_prompts[idx], image_paths)
            for filename in image_paths:
                img_path = filename
                if self.corruption and len(self.corruption) > 1 and self.corruption != 'NoImage':
                    img_path = img_path.replace('nuscenes/samples', f'corruption/{self.corruption}')
                if self.corruption == 'NoImage':
                    # Generate a blank image
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((224, 224))
                    except Exception as e:
                        print(f"Error loading image: {img_path}, error: {e}")
                        exit(1)
                placeholder = f"<image>"
                image_placeholders[idx] += placeholder + "\n"
                # Add to multi_modal_data
                multi_modal_datas[idx]["image"].append(img)
                image_index += 1

        # Build the prompt
        prompts = ["USER: "] * batch_size
        prompts = [prompt + image_placeholder for prompt, image_placeholder in zip(prompts, image_placeholders)]

        # Add system prompt
        prompts = [prompt + system_prompt for prompt, system_prompt in zip(prompts, system_prompts)]

        # Add question
        prompts = [prompt + question + "\nASSISTANT:" for prompt, question in zip(prompts, questions)]

        # batch input list
        batch_inputs = [{"prompt": prompt, "multi_modal_data": multi_modal_data} for prompt, multi_modal_data in zip(prompts, multi_modal_datas)]

        outputs = self.llm.generate(
            batch_inputs,
            self.sampling_params,
            use_tqdm=False
        )

        batch['prompts'] = prompts
        if outputs is None:
            warnings.warn("[Warning]: outputs is None")
            batch['pred'] = None
            return batch

        generated_text = []
        for output in outputs:
            generated_text.append(output.outputs[0].text)
        batch['pred'] = generated_text
        return batch


def main():
    args = parse_arguments()

    with open(args.system_prompt, 'r') as f:
        system_prompt = f.read()

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # Used GPU = tensor_parallel_size * num_processes
    tensor_parallel_size = 1
    num_instances = args.num_processes

    # Read input data
    ds = ray.data.read_json(args.data)

    # For tensor_parallel_size > 1, create placement groups
    def scheduling_strategy_fn():
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        resources_kwarg["num_gpus"] = 1
    else:
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    # Apply batch inference for all input data
    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_args=(
            args.model,
            system_prompt,
            sampling_params,
            args.num_images_per_prompt,
            args.max_model_len,
            tensor_parallel_size,
            args.corruption
        ),
        # Set the concurrency to the number of LLM instances
        concurrency=num_instances,
        # Specify the batch size for inference
        batch_size=1,
        **resources_kwarg,
    )

    save_predictions(ds, args.output)
    

if __name__ == '__main__':
    main()