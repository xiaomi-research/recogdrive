import os
import re
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple


def replace_system_prompt_new(prompt: str, image_paths: List[str]) -> Tuple[str, str]:
    """
    根据提供的图像路径，替换系统提示中的特定句子，并生成图像占位符字符串。

    Args:
        prompt (str): 包含待替换句子的原始系统提示。
        image_paths (List[str]): 对应不同摄像头的图像文件路径列表。

    Returns:
        Tuple[str, str]: 一个元组，包含:
            - updated_prompt (str): 更新后的系统提示字符串。
            - image_placeholders (str): 根据实际图像顺序格式化好的图像占位符字符串。
    """
    # 摄像头顺序及其对应的视图名称映射
    camera_order = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]
    view_mapping = {
        "CAM_FRONT": "<FRONT VIEW>:",
        "CAM_FRONT_LEFT": "<FRONT LEFT VIEW>:",
        "CAM_FRONT_RIGHT": "<FRONT RIGHT VIEW>:",
        "CAM_BACK": "<BACK VIEW>:",
        "CAM_BACK_LEFT": "<BACK LEFT VIEW>:",
        "CAM_BACK_RIGHT": "<BACK RIGHT VIEW>:",
    }

    camera_pattern = r'samples/([^/]+)/'

    extracted_cameras = []
    for path in image_paths:
        match = re.search(camera_pattern, path)
        if match:
            camera_name = match.group(1)
            if camera_name in camera_order:
                extracted_cameras.append(camera_name)
            else:
                raise ValueError(f"在路径 '{path}' 中发现无法识别的摄像头名称 '{camera_name}'。")
        else:
            raise ValueError(f"无法从路径 '{path}' 中提取摄像头名称。")

    unique_cameras = []
    seen = set()
    for cam in extracted_cameras:
        if cam not in seen:
            unique_cameras.append(cam)
            seen.add(cam)

    ordered_cameras = [cam for cam in camera_order if cam in unique_cameras]

    if not ordered_cameras:
        raise ValueError("未提供有效的摄像头图像。")
        
    cameras_str = ", ".join(ordered_cameras)
    if len(ordered_cameras) == 1:
        new_sentence = f"You are provided with a single camera image: [{cameras_str}]."
    else:
        new_sentence = f"You are provided with {len(ordered_cameras)} camera images in the sequence [{cameras_str}]."

    placeholder_parts = []
    for cam in ordered_cameras:
        view_name = view_mapping.get(cam, f"<{cam}>:") # 如果映射中没有，则使用默认格式
        placeholder_parts.append(f"{view_name}\n<image>\n")
    image_placeholders = "".join(placeholder_parts)

    original_sentence_pattern = r"You are provided with up to six camera images in the sequence \[CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT\]\."

    updated_prompt, num_subs = re.subn(original_sentence_pattern, new_sentence, prompt)

    if num_subs == 0:
        print("警告: 在 prompt 中未找到要替换的原始句子，未进行替换。")

    # 返回更新后的 prompt 和新生成的占位符
    return updated_prompt, image_placeholders

def replace_system_prompt(prompt: str, image_paths: List[str]) -> str:
    """
    Replaces the specific sentence in the system prompt to reflect only the provided camera images.

    Args:
        prompt (str): The original system prompt containing the sentence to be replaced.
        image_paths (List[str]): A list of image file paths corresponding to different cameras.

    Returns:
        str: The updated system prompt with the specified sentence adjusted to include only the available cameras.
    """
    # Order of cameras
    camera_order = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]

    # Assumes camera name is the directory name after 'samples/'
    camera_pattern = r'samples/([^/]+)/'

    # Extract camera names from image paths
    extracted_cameras = []
    for path in image_paths:
        match = re.search(camera_pattern, path)
        if match:
            camera_name = match.group(1)
            if camera_name in camera_order:
                extracted_cameras.append(camera_name)
            else:
                raise ValueError(f"Unrecognized camera name '{camera_name}' in path '{path}'.")
        else:
            raise ValueError(f"Unable to extract camera name from path '{path}'.")

    # Remove duplicates while preserving order
    unique_cameras = []
    seen = set()
    for cam in extracted_cameras:
        if cam not in seen:
            unique_cameras.append(cam)
            seen.add(cam)

    ordered_cameras = [cam for cam in camera_order if cam in unique_cameras]

    if not ordered_cameras:
        raise ValueError("No valid camera images provided.")
    else:
        cameras_str = ", ".join(ordered_cameras)
        if len(ordered_cameras) == 1:
            new_sentence = f"You are provided with a single camera image: [{cameras_str}]."
        else:
            new_sentence = f"You are provided with {len(ordered_cameras)} camera images in the sequence [{cameras_str}]."

    # Define the original sentence to be replaced
    original_sentence_pattern = r"You are provided with up to six camera images in the sequence \[CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT\]\."

    updated_prompt, num_subs = re.subn(original_sentence_pattern, new_sentence, prompt)

    if num_subs == 0:
        print("Warning: Original sentence not found in the prompt. No replacement made.")

    return updated_prompt


def load_json_files(input_folder):
    """
    Load JSON data from all files in a directory
    """
    data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            with open(os.path.join(input_folder, file_name), 'r') as f:
                for line in f:
                    try:
                        line_data = json.loads(line.strip())
                        data.append(line_data)
                    except json.JSONDecodeError as e:
                        print(f"Skipping line in {file_name} due to error: {e}")
    return data


def save_json(data, output_file):
    """
    Save JSON data to a file.
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def save_predictions(dataset, output: str) -> None:
    destination = Path(output + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".drivebench-") as parts:
        dataset.write_json(parts)
        save_json(load_json_files(parts), str(destination))
