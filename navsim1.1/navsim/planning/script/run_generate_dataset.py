from typing import Tuple
from pathlib import Path
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from tqdm.auto import tqdm
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset_For_Traj
from navsim.planning.training.agent_lightning_module import AgentLightningModule
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import base64
import os
import io
import multiprocessing
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig, PytorchEngineConfig
from PIL import Image
from openai import OpenAI
from lmdeploy.vl import load_image
import numpy as np
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color
import cv2


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Dataset_For_Traj:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names  if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )

    train_data = Dataset_For_Traj(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data


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





def format_number(n, decimal_places=2):
    if isinstance(n, torch.Tensor):
        n = n.item()  
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)


def process_data_and_create_qa_pair(idx, ego_statuses, cameras,future_trajectory, token, prompt_type='base', cam_type='single', is_3d_det=False, is_cot_generate=False,r1_zero=False):
    # Original trajectory QA (no changes)
    history_trajectory = []
    for i in range(4):
        ego_status = ego_statuses[i]
        history_trajectory.append({
            "x": format_number(ego_status.ego_pose[0]),
            "y": format_number(ego_status.ego_pose[1]),
            "heading": format_number(ego_status.ego_pose[2])
        })

    high_command_one_hot = ego_statuses[-1].driving_command
    navigation_commands = ['turn left', 'go straight', 'turn right']
    command_str = [navigation_commands[i] for i in range(len(high_command_one_hot)) if high_command_one_hot[i] == 1]
    command_str = command_str[0] if command_str else "unknown"

    image_paths = []
    image_prompt_lines = []  
    image_prompt = "" 
    if cam_type == 'single':
        image_paths.append(str(cameras[-1].cam_f0.image))
        image_prompt_lines.append("<image>\n")
        image_prompt = "1. Visual perception from front camera view\n"
    else:  
        image_paths.append(str(cameras[-1].cam_f0.image))
        image_prompt = "1. Visual perception from front camera view\n"

    future_trajectory_points = []
    for traj_point in future_trajectory:
        future_trajectory_points.append(f"({format_number(traj_point[0])}, {format_number(traj_point[1])}, {format_number(traj_point[2])})")

    future_trajectory_str = f"Here is the planning trajectory [PT, {', '.join(future_trajectory_points)}]."

    image_prompt_lines_str = "".join(image_prompt_lines) 

    common_prompt = f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt}2. Historical motion context (last 4 timesteps):{" ".join([f'-t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n3. Active navigation command: [{command_str.upper()}]"""  # Common prompt up to the velocity/acceleration

    output_requirements = """\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places""" # Output requirements as a separate string

    if prompt_type == 'base':
        question = f"{image_prompt_lines_str}\n{common_prompt}{output_requirements}"
    elif prompt_type == 'vel_and_acc':
        current_ego_status = ego_statuses[-1]
        current_velocity = current_ego_status.ego_velocity
        current_acceleration = current_ego_status.ego_acceleration

        velocity_acceleration_info = f"\n4. Current velocity: ({format_number(current_velocity[0])}, {format_number(current_velocity[1])})\n5. Current acceleration: ({format_number(current_acceleration[0])}, {format_number(current_acceleration[1])})"

        question = f"{image_prompt_lines_str}\n{common_prompt}{velocity_acceleration_info}{output_requirements}"
    else:  
        question = f"{image_prompt_lines_str}\n{common_prompt}{output_requirements}"

    answer = future_trajectory_str



    qa_pair = {
            "id": idx,
            "image": image_paths,
            "token": token,
            "conversations": [
                {"from": "system", "value": system_message},
                {"from": "human", "value": question}, 
                {"from": "gpt", "value": answer},
            ]
    }


    return qa_pair


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building SceneLoader")
    train_data = build_datasets(cfg, agent)
    logger.info("Building Datasets")
    logger.info("Num training samples: %d", len(train_data))

    
    qa_pairs = []
    with open('navsim_traj_front_view.jsonl', 'w', encoding='utf-8') as f:
        for idx, (ego_statuses, cameras, future_trajectory, token) in enumerate(tqdm(train_data)):
            qa_pair = process_data_and_create_qa_pair(idx, ego_statuses, cameras, future_trajectory,token)
            json.dump(qa_pair, f, ensure_ascii=False)
            f.write('\n')
            f.flush() 
        f.close() 

if __name__ == "__main__":
    main()
