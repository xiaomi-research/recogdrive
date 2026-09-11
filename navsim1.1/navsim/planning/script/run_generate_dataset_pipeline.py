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
from navsim.planning.training.dataset import Dataset_For_Pipeline
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
from openai import OpenAI, AsyncOpenAI
from lmdeploy.vl import load_image
import numpy as np
import asyncio
import math
import cv2  # Import OpenCV for image reading
import random

pedal_status = {
    'const': 'KEEP',
    'accelerate': 'ACCELERATE',
    'decelerate': 'DECELERATE',
    'stop': 'STOP'
}

path_status = {
    'right turn': 'RIGHT_TURN',
    'right lane change': 'RIGHT_CHANGE',
    'left turn': 'LEFT_TURN',
    'left lane change': 'LEFT_CHANGE',
    'straight': 'STRAIGHT'
}

aclient = AsyncOpenAI(
    api_key="",
    base_url="",
)
client = OpenAI(
    api_key="",
    base_url="",
)
models = client.models.list()
model_id = models.data[0].id
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

async def async_qwen_vl_72b_infer(args, max_retries=6, retry_delay=5):
    """
    异步调用 Qwen-VL-72B 模型进行推理，支持重试和超时。

    Args:
        args: 包含查询和图像文件路径的参数对象。
        max_retries: 最大重试次数。
        retry_delay: 重试延迟（秒）。

    Returns:
        模型的文本输出。
    """
    for attempt in range(max_retries):
        try:
            with open(args.img_file, "rb") as image_file:
                img = Image.open(image_file)
                img_resized = img.resize((960, 540))

                buffer = io.BytesIO()
                img_resized.save(buffer, format="JPEG")
                buffer.seek(0)

                encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

            completion: ChatCompletion = await aclient.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": args.query
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ],
                }],
                timeout=600
            )

            text_outputs = completion.choices[0].message.content
            return text_outputs

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"All {max_retries} attempts failed. Error: {e}")
                raise 


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Dataset_For_Internvl:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
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

    train_data = Dataset_For_Pipeline(
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

async def get_img_description_qa(img_path, img_type, token):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/img_desc"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        question = "Suppose you are driving, and I'm providing you with the image " \
                    f"captured by the car's {img_type}, generate a description of the driving scene " \
                    "which includes the key factors for driving planning, including the positions " \
                    "and movements of vehicles and pedestrians; prevailing weather conditions; " \
                    "time of day, distinguishing between daylight and nighttime; road conditions, " \
                    "indicating smooth surfaces or the presence of obstacles; and the status of traffic lights " \
                    "which influence your decision making, specifying whether they are red or green. " \
                    "The description should be concise, providing an accurate understanding " \
                    "of the driving environment to facilitate informed decision-making."
        return question, answer
    else:
        question = "Suppose you are driving, and I'm providing you with the image " \
                    f"captured by the car's {img_type}, generate a description of the driving scene " \
                    "which includes the key factors for driving planning, including the positions " \
                    "and movements of vehicles and pedestrians; prevailing weather conditions; " \
                    "time of day, distinguishing between daylight and nighttime; road conditions, " \
                    "indicating smooth surfaces or the presence of obstacles; and the status of traffic lights " \
                    "which influence your decision making, specifying whether they are red or green. " \
                    "The description should be concise, providing an accurate understanding " \
                    "of the driving environment to facilitate informed decision-making."

        args = type('Args', (), {
            "query": question,
            "img_file": img_path,
        })()

        answer = await async_qwen_vl_72b_infer(args)
        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)
        return question, answer

async def get_traffic_congestion_qa(cf_img_path, token):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/traf_cong"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        question = "Based on the provided forward-facing image from a car's perspective, " \
                    "please analyze the current traffic congestion level on the road. " \
                    "Determine whether the road is heavily congested, moderately congested, or clear. " \
                    "Then, based on the congestion level, advise whether the driving behavior should be cautious or normal. "
        return question, answer
    else:
        question = "Based on the provided forward-facing image from a car's perspective, " \
                    "please analyze the current traffic congestion level on the road. " \
                    "Determine whether the road is heavily congested, moderately congested, or clear. " \
                    "Then, based on the congestion level, advise whether the driving behavior should be cautious or normal. "

        args = type('Args', (), {
            "query": question,
            "img_file": cf_img_path,
        })()

        answer = await async_qwen_vl_72b_infer(args)
        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)
        return question, answer

async def get_traffic_light_qa(cf_img_path, token):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/traf_light"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        question = "Given the provided forward-facing image from a car's perspective, " \
                    "identify if there is a traffic light that affects the car's behavior. " \
                    "Please respond with a complete sentence describing the traffic light status, " \
                    "such as 'The traffic light is red', 'The traffic light is green', 'The traffic light is yellow', or 'There is no traffic light visible'."
        if "There is no traffic light" in answer:
            return None,None
        return question, answer
    else:
        question = "Given the provided forward-facing image from a car's perspective, " \
                    "identify if there is a traffic light that affects the car's behavior. " \
                    "Please respond with a complete sentence describing the traffic light status, " \
                    "such as 'The traffic light is red', 'The traffic light is green', 'The traffic light is yellow', or 'There is no traffic light visible'."

        args = type('Args', (), {
            "query": question,
            "img_file": cf_img_path,
        })()

        answer = await async_qwen_vl_72b_infer(args)
        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)
        return question, answer
        
async def get_road_sign_qa(cf_img_path, token):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/road_sign"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        question = "Based on the provided forward-facing image from a car's perspective, " \
                    "identify and describe the road markings and signs, including traffic lines, " \
                    "road signs, pedestrian crossings, and speed bumps. " \
                    "Explain their meaning and advise on the appropriate driving action in a concise, " \
                    "single paragraph without using bullet points or numbered lists."
        return question, answer
    else:
        question = "Based on the provided forward-facing image from a car's perspective, " \
                    "identify and describe the road markings and signs, including traffic lines, " \
                    "road signs, pedestrian crossings, and speed bumps. " \
                    "Explain their meaning and advise on the appropriate driving action in a concise, " \
                    "single paragraph without using bullet points or numbered lists."

        args = type('Args', (), {
            "query": question,
            "img_file": cf_img_path,
        })()

        answer = await async_qwen_vl_72b_infer(args)
        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)
        return question, answer

async def get_driving_influence_qa(cf_img_path, token):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/driving_influence"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        question = "Based on the provided forward-facing image from a car's perspective, " \
                    "identify the most influential object that affects the current driving situation " \
                    "(e.g., a vehicle ahead, a pedestrian, a traffic light). " \
                    "Describe why this object is influential, including its current state or actions " \
                    "(e.g., braking, turning, moving). " \
                    "Finally, advise the driver on the appropriate driving action in a concise paragraph."
        return question, answer
    else:
        question = "Based on the provided forward-facing image from a car's perspective, " \
                    "identify the most influential object that affects the current driving situation " \
                    "(e.g., a vehicle ahead, a pedestrian, a traffic light). " \
                    "Describe why this object is influential, including its current state or actions " \
                    "(e.g., braking, turning, moving). " \
                    "Finally, advise the driver on the appropriate driving action in a concise paragraph."

        args = type('Args', (), {
            "query": question,
            "img_file": cf_img_path,
        })()

        answer = await async_qwen_vl_72b_infer(args)
        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)
        return question, answer

async def get_vru_qa(token, agent_boxes, agent_names, vru_dis_thresh=40.0):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/vru"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        question = f"Do you see any vulnerable road users within {int(vru_dis_thresh)} meters ahead of you, " \
                    "such as cyclists, motorcyclists, or pedestrians?"
        if "No, I don't see any vulnerable road users ahead of me, such as bicycles, motorcycles, or pedestrians." in answer:
            return None,None
        return question, answer
    else:
        question = f"Do you see any vulnerable road users within {int(vru_dis_thresh)} meters ahead of you, " \
                    "such as cyclists, motorcyclists, or pedestrians?"
        vru_list = []
        num_objects = len(agent_boxes)
        vru_classes = ['motorcycle', 'pedestrian']

        for i in range(num_objects):
            box = agent_boxes[i]
            obj_loc = box[:2]

            obj_cls = agent_names[i]
            x_dis, y_dis = box[0], box[1]  # agent_boxes直接提取x和y

            if obj_cls in vru_classes and np.linalg.norm(obj_loc) < vru_dis_thresh:
                if y_dis <= -2.0:
                    lat_pos = f" and {float(abs(y_dis)):.2f} meters to the right"
                elif y_dis >= 2.0:
                    lat_pos = f" and {float(abs(y_dis)):.2f} meters to the left"
                else:
                    lat_pos = ""
                vru_description = f"a {obj_cls} located {float(abs(x_dis)):.2f} meters ahead of me{lat_pos}"
                vru_list.append(vru_description)

        if vru_list:
            answer = "Yes, I see " + ", and ".join(vru_list) + "."
        else:
            answer = "No, I don't see any vulnerable road users ahead of me, " \
                     "such as bicycles, motorcycles, or pedestrians."

        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)

        return question, answer

async def get_mot_pred_qa(token, agent_boxes, agent_names, agent_vel, dis_thresh=40.0):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/mot_pred"
    file_path = os.path.join(vqa_dir, f"{token}.txt")
    img_type = 'front'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        question = "You are driving, I will now provide you with the location " \
                   f"and velocity information of dynamic objects in the {img_type} view image. " \
                   "Please predict their future driving behaviors, " \
                   "which can be divided into SPEED decisions and PATH decisions. " \
                   "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, " \
                   "while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN." \
                   "I will now provide you with the position and velocity information of the dynamic objects: \n"

        num_objects = len(agent_boxes)
        obj_cnt = 0
        for i in range(num_objects):
            box = agent_boxes[i]
            obj_loc = box[:2]
            if np.linalg.norm(obj_loc, axis=-1) >= dis_thresh:
                continue
            x_dis, y_dis = box[0], box[1]
            obj_vel = agent_vel[i]
            obj_cls = agent_names[i]
            obj_speed = np.linalg.norm(obj_vel[0, :2], axis=-1)
            obj_cnt = obj_cnt + 1
            if x_dis >= 0:
                log_describe = f"{int(x_dis)} meters ahead"
            else:
                log_describe = f"{abs(int(x_dis))} meters behind"
            if y_dis >= 0:
                lat_describe = f"{int(y_dis)} meters to the left"
            else:
                lat_describe = f"{abs(int(y_dis))} meters to the right"

            obj_info = f'Object {obj_cnt}: {obj_cls}, {log_describe}, {lat_describe}, speed of {int(obj_speed)} m/s.'
            question = question + obj_info + '\n'

        question_end = "Please predict the future driving behaviors of these objects " \
                       f"based on the {img_type} view image. " \
                       "For example, a well-formatted answer should be like:\n" \
                       "Object 1: KEEP, STRAIGHT\n" \
                       "Object 2: DECELERATE, RIGHT_TURN\n" \
                       "Object 3: ACCELERATE, LEFT_CHANGE\n"

        question = question + question_end

        return question, answer
    else:
        question = "You are driving, I will now provide you with the location " \
                   f"and velocity information of dynamic objects in the {img_type} view image. " \
                   "Please predict their future driving behaviors, " \
                   "which can be divided into SPEED decisions and PATH decisions. " \
                   "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, " \
                   "while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN." \
                   "I will now provide you with the position and velocity information of the dynamic objects: \n"

        num_objects = len(agent_boxes)
        obj_cnt = 0
        answer = ""
        for i in range(num_objects):
            box = agent_boxes[i]
            obj_loc = box[:2]
            if np.linalg.norm(obj_loc, axis=-1) >= dis_thresh:
                continue
            x_dis, y_dis = box[0], box[1]
            obj_vel = agent_vel[i]
            obj_pedal_status = get_obj_acc_or_dec_from_vel(obj_vel)
            obj_wheel_status = get_obj_turn_or_lane_change_from_vel(obj_vel)
            obj_speed_plan = pedal_status[obj_pedal_status]
            obj_path_plan = path_status[obj_wheel_status]
            obj_cls = agent_names[i]
            obj_speed = np.linalg.norm(obj_vel[0, :2], axis=-1)
            obj_cnt = obj_cnt + 1
            if x_dis >= 0:
                log_describe = f"{int(x_dis)} meters ahead"
            else:
                log_describe = f"{abs(int(x_dis))} meters behind"
            if y_dis >= 0:
                lat_describe = f"{int(y_dis)} meters to the left"
            else:
                lat_describe = f"{abs(int(y_dis))} meters to the right"

            obj_info = f'Object {obj_cnt}: {obj_cls}, {log_describe}, {lat_describe}, speed of {int(obj_speed)} m/s.'

            question = question + obj_info + '\n'
            answer = answer + f"Object {obj_cnt}: {obj_speed_plan}, {obj_path_plan}\n"

        question_end = "Please predict the future driving behaviors of these objects " \
                       f"based on the {img_type} view image. " \
                       "For example, a well-formatted answer should be like:\n" \
                       "Object 1: KEEP, STRAIGHT\n" \
                       "Object 2: DECELERATE, RIGHT_TURN\n" \
                       "Object 3: ACCELERATE, LEFT_CHANGE\n"

        question = question + question_end

        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)

        return question, answer


def get_obj_acc_or_dec(trajectory, vel_diff_thresh=3.0):
    velocity = np.linalg.norm(trajectory[1:,:2] - trajectory[:-1,:2], axis=-1) / 0.5

    if np.max(velocity) < 2.0:
        return "stop"

    vel_diff = velocity[-1] - velocity[0]

    if vel_diff >= vel_diff_thresh:
        return "accelerate"
    elif vel_diff <= -vel_diff_thresh:
        return "decelerate"
    else:
        return "const"

def get_obj_turn_or_lane_change(trajectory, lat_thresh=4.0, angle_thresh=5.0):
    # 提取横向位置和纵向位置
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    # 计算车辆角度变化
    endpoint_angle = math.degrees(math.atan2(x[-1], y[-1]))
    angle_diff = endpoint_angle - 90.0

    # 判断是否进行变道或转弯
    if y[-1] > lat_thresh and angle_diff <= -angle_thresh:
        return "left turn"
    elif y[-1] > lat_thresh and abs(angle_diff) < angle_thresh:
        return "left lane change"
    elif y[-1] <= -lat_thresh and angle_diff >= angle_thresh:
        return "right turn"
    elif y[-1] <= -lat_thresh and abs(angle_diff) < angle_thresh:
        return "right lane change"
    else:
        return "straight"



def get_obj_acc_or_dec_from_vel(velocity, vel_diff_thresh=3.0):
    """
    根据速度信息判断加速/减速/保持/停止状态。

    Args:
        velocity: 速度数组，形状为 (8, 3)。
        vel_diff_thresh: 速度变化阈值。

    Returns:
        'accelerate', 'decelerate', 'const', 或 'stop'。
    """
    speed = np.linalg.norm(velocity[:, :2], axis=-1)  # 计算速度大小
    vel_diff = speed[-1] - speed[0]

    if np.max(speed) < 2.0:
        return "stop"
    elif vel_diff >= vel_diff_thresh:
        return "accelerate"
    elif vel_diff <= -vel_diff_thresh:
        return "decelerate"
    else:
        return "const"

def get_obj_turn_or_lane_change_from_vel(velocity, lat_thresh=4.0, angle_thresh=5.0):
    """
    根据速度信息判断转向/变道/直行状态。

    Args:
        velocity: 速度数组，形状为 (8, 3)。
        lat_thresh: 横向偏移阈值。
        angle_thresh: 角度变化阈值。

    Returns:
        'right turn', 'right lane change', 'left turn', 'left lane change', 或 'straight'。
    """
    x_diff = velocity[-1, 0] - velocity[-2, 0]  # 计算最后两点的 x 坐标差
    y_diff = velocity[-1, 1] - velocity[-2, 1]  # 计算最后两点的 y 坐标差
    endpoint_angle = math.degrees(math.atan2(x_diff, y_diff)) # 计算最后两点的角度
    angle_diff = endpoint_angle - math.degrees(velocity[-2, 2]) # 计算最后两点角度和heading的差值

    if y_diff > lat_thresh and angle_diff <= -angle_thresh:
        return "left turn"
    elif y_diff > lat_thresh and abs(angle_diff) < angle_thresh:
        return "left lane change"
    elif y_diff <= -lat_thresh and angle_diff >= angle_thresh:
        return "right turn"
    elif y_diff <= -lat_thresh and abs(angle_diff) < angle_thresh:
        return "right lane change"
    else:
        return "straight"

def get_decision(ego_speed_plan, ego_path_plan):
    pedal_decision = {
        'KEEP': 'maintain the current speed',
        'ACCELERATE': 'accelerate',
        'DECELERATE': 'decelerate',
        'STOP': 'stop the car'
    }

    path_decision = {
        'RIGHT_TURN': 'turn right',
        'RIGHT_CHANGE': 'change to the right lane',
        'LEFT_TURN': 'turn left',
        'LEFT_CHANGE': 'change to the left lane',
        'STRAIGHT': 'go straight'
    }

    if ego_speed_plan == 'STOP':
        return pedal_decision[ego_speed_plan]
    else:
        return pedal_decision[ego_speed_plan] + ' and ' + path_decision[ego_path_plan]


async def get_plan_qa(token, future_trajectory_points, history_trajectory, command_str):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/plan"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        trajectory_str = chr(2).join([f'  - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])
        question = f"Your historical trajectories are {trajectory_str}," \
                   f"the navigation command is '{command_str}', " \
                   "based on the understanding of the driving scene and the navigation information, " \
                   "what is your plan for the next three seconds? " \
                   "Please answer your SPEED plan and your PATH plan. " \
                   "SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, " \
                   "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. " \
                   "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
        return question, answer
    else:
        trajectory_str = chr(2).join([f'  - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])
        question = f"Your historical trajectories are {trajectory_str}," \
                   f"the navigation command is '{command_str}', " \
                   "based on the understanding of the driving scene and the navigation information, " \
                   "what is your plan for the next three seconds? " \
                   "Please answer your SPEED plan and your PATH plan. " \
                   "SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, " \
                   "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. " \
                   "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."


        # 将 future_trajectory_points 转换为 NumPy 数组
        future_trajectory_np = np.array([[float(p.split(',')[0][1:]), float(p.split(',')[1]), float(p.split(',')[2][:-1])] for p in future_trajectory_points])

        ego_pedal_status = get_obj_acc_or_dec(future_trajectory_np)
        ego_speed_plan = pedal_status[ego_pedal_status]

        ego_path_plan = get_obj_turn_or_lane_change(future_trajectory_np)
        ego_path_plan = path_status[ego_path_plan]

        answer = ego_speed_plan + ', ' + ego_path_plan + '\n'

        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)
        return question, answer

async def get_plan_explaination_qa(cf_img_path, token, future_trajectory_points, history_trajectory, command_str):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/plan_explain"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        future_trajectory_np = np.array([[float(p.split(',')[0][1:]), float(p.split(',')[1]), float(p.split(',')[2][:-1])] for p in future_trajectory_points])
        ego_pedal_status = get_obj_acc_or_dec(future_trajectory_np)
        ego_speed_plan = pedal_status[ego_pedal_status]
        ego_path_plan = get_obj_turn_or_lane_change(future_trajectory_np)
        ego_path_plan = path_status[ego_path_plan]
        decision = get_decision(ego_speed_plan, ego_path_plan)

        trajectory_str = chr(2).join([f'  - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])

        question = "You are driving, " \
                   f"Your historical trajectories are {trajectory_str}," \
                   f"and the navigation command is '{command_str}', " \
                   "your driving decision for the next four seconds is to " \
                   f"{decision}. " \
                   "Based on the provided image of the driving environment, " \
                   "explain the most likely reason for this decision in one or two concise sentence."
        return question, answer
    else:
        future_trajectory_np = np.array([[float(p.split(',')[0][1:]), float(p.split(',')[1]), float(p.split(',')[2][:-1])] for p in future_trajectory_points])
        ego_pedal_status = get_obj_acc_or_dec(future_trajectory_np)
        ego_speed_plan = pedal_status[ego_pedal_status]
        ego_path_plan = get_obj_turn_or_lane_change(future_trajectory_np)
        ego_path_plan = path_status[ego_path_plan]
        decision = get_decision(ego_speed_plan, ego_path_plan)

        trajectory_str = chr(2).join([f'  - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])

        question = "You are driving, " \
                   f"Your historical trajectories are {trajectory_str}," \
                   f"and the navigation command is '{command_str}', " \
                   "your driving decision for the next four seconds is to " \
                   f"{decision}. " \
                   "Based on the provided image of the driving environment, " \
                   "explain the most likely reason for this decision in one or two concise sentence."

        args = type('Args', (), {
            "query": question,
            "img_file": cf_img_path,
        })()

        answer = await async_qwen_vl_72b_infer(args)
        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)
        return question, answer

async def get_driving_behavior_qa(cf_img_path, token, history_trajectory):
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/driving_behavior"
    file_path = os.path.join(vqa_dir, f"{token}.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            answer = f.read()
        trajectory_str = chr(2).join([f'  - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])
        question = f"Based on the provided image and the historical trajectory: {trajectory_str}, describe the current driving behavior of the vehicle in one or two concise sentences."
        return question, answer
    else:
        history_trajectory_np = np.array([[float(t['x']), float(t['y']), float(t['heading'])] for t in history_trajectory])

        ego_pedal_status = get_obj_acc_or_dec(history_trajectory_np)
        ego_path_plan = get_obj_turn_or_lane_change(history_trajectory_np)
        ego_speed_plan = pedal_status[ego_pedal_status]
        ego_path_plan = path_status[ego_path_plan]
        decision = get_decision(ego_speed_plan, ego_path_plan)

        trajectory_str = chr(2).join([f'  - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])
        question = f"Based on the provided image and the historical trajectory: {trajectory_str}, describe the current driving behavior of the vehicle in one or two concise sentences."

        args = type('Args', (), {
            "query": question,
            "img_file": cf_img_path,
        })()

        answer = await async_qwen_vl_72b_infer(args)
        os.makedirs(vqa_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(answer)
        return question, answer

async def get_traj_results_qa(token):
    """
    根据传入的 token，从 traj_results_new 目录中查找对应的 JSON 文件。
    如果存在，读取并返回其中的 question 和 answer；否则返回 None。
    """
    base_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/traj_results_new"
    file_path = os.path.join(base_dir, f"{token}.json")
    
    if not os.path.exists(file_path):
        return None,None

    # 读取 JSON 文件
    with open(file_path, "r") as f:
        data = json.load(f)
    
    question = data.get("question", "")
    answer = data.get("answer", "")
    
    return question, answer

def format_number(n, decimal_places=2):
    if isinstance(n, torch.Tensor):
        n = n.item()  
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)

async def get_dis_cal_qa(cf_img_path, token, agent_boxes, agent_names, box_2d, image_width=None, image_height=None):
    # 文件存储路径
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/distance_calculation_new"
    file_path = os.path.join(vqa_dir, f"{token}.json")  # 保存为 JSON 文件

    # 如果文件已经存在，则直接加载数据
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            qa_data = json.load(f)
        question = qa_data["question"]
        answer = qa_data["answer"]
        return question, answer

    if len(box_2d) == 0:  # 检查 box_2d 是否为空
        return None, None  # 如果为空，则跳过

    # 如果未提供图像尺寸，则读取图像
    if image_width is None or image_height is None:
        img = cv2.imread(cf_img_path)
        if img is None:
            return None, None
        image_height, image_width, _ = img.shape

    # 将 box_2d 转换为列表
    box_2d_list = box_2d

    # 筛选出坐标合法且在图像尺寸范围内的框
    filtered_boxes = []
    filtered_agent_boxes = []  # 存储对应的3D框
    filtered_boxes_names = []
    for i, box in enumerate(box_2d_list):
        x1, y1, x2, y2 = box
        if x1 >= 0 and y1 >= 0 and x2 <= image_width and y2 <= image_height:
            filtered_boxes.append(box)
            filtered_agent_boxes.append(agent_boxes[i])
            filtered_boxes_names.append(agent_names[i])
    if len(filtered_boxes) < 2:
        return None, None  # 如果不足两个框，则返回

    # 辅助函数：判断两个框是否有重合（即交叉区域面积 > 0）
    def boxes_overlap(box1, box2):
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2
        x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
        y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
        return x_overlap > 0 and y_overlap > 0

    # 构造所有不重合的框对
    valid_pairs = []
    n_boxes = len(filtered_boxes)
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            if not boxes_overlap(filtered_boxes[i], filtered_boxes[j]):
                valid_pairs.append((i, j))

    if not valid_pairs:
        # 如果没有满足条件的框对，则删除文件（如果存在）并返回 None
        if os.path.exists(file_path):
            os.remove(file_path)
        return None, None

    # 从满足条件的框对中随机选择一对
    box1_idx, box2_idx = random.choice(valid_pairs)

    # 对框进行归一化处理
    normalized_boxes = [normalize_coordinates(box, image_width, image_height) for box in filtered_boxes]

    box1, box2 = normalized_boxes[box1_idx], normalized_boxes[box2_idx]
    obj_name1, obj_name2 = filtered_boxes_names[box1_idx], filtered_boxes_names[box2_idx]

    # 构造包含框信息的对象描述
    obj_with_box1 = f"<{obj_name1}><FRONT VIEW><box>{box1}</box>"
    obj_with_box2 = f"<{obj_name2}><FRONT VIEW><box>{box2}</box>"

    # 问句模板
    question_templates = [
        f"How far apart are the {obj_with_box1} and the {obj_with_box2}?",
        f"What is the distance between the {obj_with_box1} and the {obj_with_box2}?",
        f"Calculate the separation between the {obj_with_box1} and the {obj_with_box2}.",
        f"Can you tell me the distance between the {obj_with_box1} and the {obj_with_box2}?",
        f"What's the gap between the {obj_with_box1} and the {obj_with_box2}?",
    ]
    question = random.choice(question_templates)

    # 计算3D距离（这里只使用 x, y 坐标）
    dist_3d = np.linalg.norm(filtered_agent_boxes[box1_idx][:2] - filtered_agent_boxes[box2_idx][:2])

    answer = f"The {obj_with_box1} and the {obj_with_box2} are approximately {dist_3d:.2f} meters apart."
    qa_data = {"question": question, "answer": answer}

    os.makedirs(vqa_dir, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(qa_data, f)  # 保存为 JSON

    return question, answer

async def get_3d_info_qa(cf_img_path, token, agent_boxes, agent_names, box_2d, image_width=None, image_height=None):
    import os
    import json
    import random
    import cv2
    import numpy as np

    # 定义保存问答数据的目录
    vqa_dir = "/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/vqa/3d_info_qa"
    file_path = os.path.join(vqa_dir, f"{token}.json")
    
    # 如果文件已存在，则直接加载问答数据
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            qa_data = json.load(f)
        return qa_data.get("qa_pairs", [])
    
    # 如果没有2D框数据，则返回空列表
    if len(box_2d) == 0:
        return []
    
    # 如果未提供图像尺寸，则读取图像以获取尺寸
    if image_width is None or image_height is None:
        img = cv2.imread(cf_img_path)
        if img is None:
            return []
        image_height, image_width, _ = img.shape

    # 假设 box_2d 已经是列表格式
    box_2d_list = box_2d

    # 筛选出在图像范围内的2D框，并同时筛选对应的3D框和物体类别
    filtered_boxes = []
    filtered_agent_boxes = []
    filtered_boxes_names = []
    for i, box in enumerate(box_2d_list):
        x1, y1, x2, y2 = box
        if x1 >= 0 and y1 >= 0 and x2 <= image_width and y2 <= image_height:
            filtered_boxes.append(box)
            filtered_agent_boxes.append(agent_boxes[i])
            filtered_boxes_names.append(agent_names[i])
            
    if len(filtered_boxes) == 0:
        return []
    
    # 根据有效框的数量动态决定生成多少个问答对（最多5个）
    indices = list(range(len(filtered_boxes)))
    if len(indices) > 5:
        indices = random.sample(indices, 5)
    
    qa_pairs = []
    for i in indices:
        # 使用归一化后的2D框作为问句提示（请确保已定义 normalize_coordinates 函数）
        normalized_box = normalize_coordinates(filtered_boxes[i], image_width, image_height)
        question_templates = [
            f"What is the 3D information of the object with the 2D box {normalized_box}?",
            f"Please provide the 3D details for the object whose 2D bounding box is {normalized_box}.",
            f"Can you tell me the 3D information for the object located at 2D box {normalized_box}?"
        ]
        question = random.choice(question_templates)
        
        # 获取对应的3D信息，假设 agent_box 格式为 [x, y, length, width, height, heading]
        agent_box = filtered_agent_boxes[i]
        x, y, l, w, h, heading = agent_box[:6]
        class_label = filtered_boxes_names[i]
        
        answer = (f"The object is a {class_label} location: ({x:.2f}, {y:.2f}), "
                  f"length: {l:.2f}, width: {w:.2f}, height: {h:.2f}, heading: {heading:.2f}.")
        qa_pairs.append({"question": question, "answer": answer})
    
    qa_data = {"qa_pairs": qa_pairs}
    os.makedirs(vqa_dir, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(qa_data, f)
    
    return qa_pairs

async def get_3d_det_qa(agent_boxes, agent_names):

    if len(agent_boxes) == 0:
        return None,None

    question_detection = f"Detect every bicycle, pedestrian, and vehicle in 3D sorted from nearest to farthest and respond in the format: Object N: (x, y, z, l, w, h, heading, name). " \
        f"Where x, y, z are the center coordinates of the object in ego-coordinate system, " \
        f"l, w, h are length, width, height of the bounding box, " \
        f"heading is the object heading, and name is the object class name, sorted by distance from nearest to farthest."
    
    answer_detection_lines = [] # 使用列表来存储每一行的回答，最后再拼接成字符串

    for i in range(len(agent_boxes)):
        box = agent_boxes[i]
        name = agent_names[i] # Now agent_names is correctly ordered
        x, y, z, l, w, h, heading = box[0], box[1], box[2], box[3], box[4], box[5], box[6]
        answer_line = f"Object {i+1}: ({x:.2f}, {y:.2f}, {z:.2f}, {l:.2f}, {w:.2f}, {h:.2f}, {heading:.2f}, {name})" # 保留两位小数, 移除 text, 添加 Object N: 前缀
        answer_detection_lines.append(answer_line)

    answer_detection = " ".join(answer_detection_lines) # 将所有行用空格拼接成最终答案

    return question_detection, answer_detection


def normalize_coordinates(box, image_width, image_height):
    """
    Normalizes 2D bounding box coordinates to a range of 0-1000.
    """
    x1, y1, x2, y2 = box.tolist()  # Convert tensors to lists
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box

async def process_data_and_create_qa_pair(idx, ego_statuses, cameras, future_trajectory, agent_states, agent_labels, agent_names,token, velocity_3d,box_2d, prompt_type='base', cam_type='single', is_3d_det=False, is_cot_generate=False,r1_zero=False):
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
    image_prompt_lines = []  # 存储每个视角的图像描述
    image_prompt = "" 
    if cam_type == 'single':
        image_paths.append(str(cameras[-1].cam_f0.image))
        image_prompt_lines.append("<FRONT VIEW>:\n<image>\n")
        image_prompt = "1. Visual perception from front camera view\n"
    elif cam_type == 'multi_3':
        original_image_paths = [str(cameras[-1].cam_f0.image), str(cameras[-1].cam_l0.image), str(cameras[-1].cam_r0.image)]
        updated_image_data = process_multi_3_images(original_image_paths) # 调用图片处理函数
        updated_image_paths, updated_image_prompt = updated_image_data # 获取处理后的路径和 prompt
        if updated_image_paths: # 检查是否处理成功
            image_paths = updated_image_paths # 更新 image_paths 为处理后的路径 (只包含拼接图路径)
            image_prompt_lines = [] # 清空 image_prompt_lines，使用新的
            image_prompt_lines.append("<FRONT VIEW>:\n<image>\n")
            image_prompt = updated_image_prompt # 使用 process_multi_3_images 返回的 prompt
        else:
            print("error")
    elif cam_type == 'multi_5':
        image_paths.extend([str(cameras[-1].cam_f0.image), str(cameras[-1].cam_l0.image), str(cameras[-1].cam_l1.image), str(cameras[-1].cam_r1.image), str(cameras[-1].cam_r2.image)])
        image_prompt = "1. Visual perception from front, left1, left2, right1 and right2 camera views\n"
    elif cam_type == 'cont':
        for i in range(4):
            image_paths.append(str(cameras[i].cam_f0.image))
            image_prompt_lines.append(f"<FRONT VIEW>Frame-{i+1}: <image>\n") # Frame-1, Frame-2, Frame-3, Frame-4 (从过去到现在)
        image_prompt = "1. Visual perception from continuous front camera views of the last 4 timesteps\n"
    else:  
        image_paths.append(str(cameras[-1].cam_f0.image))
        image_prompt = "1. Visual perception from front camera view\n"

    future_trajectory_points = []
    for traj_point in future_trajectory:
        future_trajectory_points.append(f"({format_number(traj_point[0])}, {format_number(traj_point[1])}, {format_number(traj_point[2])})")

    future_trajectory_str = f"Here is the planning trajectory [PT, {', '.join(future_trajectory_points)}]."

    image_prompt_lines_str = "".join(image_prompt_lines) 

    common_prompt = f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt}2. Historical motion context (last 4 timesteps):{chr(2).join([f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n3. Active navigation command: [{command_str.upper()}]"""  # Common prompt up to the velocity/acceleration

    output_requirements = """\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places""" # Output requirements as a separate string

    if prompt_type == 'base':
        question = f"{image_prompt_lines_str}\n{common_prompt}{output_requirements}"
    elif prompt_type == 'vel_and_acc':
        current_ego_status = ego_statuses[-1]
        current_velocity = current_ego_status.ego_velocity
        current_acceleration = current_ego_status.ego_acceleration

        velocity_acceleration_info = f"\n4. Current velocity: ({format_number(current_velocity[0])}, {format_number(current_velocity[1])})\n5. Current acceleration: ({format_number(current_acceleration[0])}, {format_number(current_acceleration[1])})"

        question = f"{image_prompt_lines_str}\n{common_prompt}{velocity_acceleration_info}{output_requirements}"
    else:  # Default to 'base' if prompt_type is not recognized
        question = f"{image_prompt_lines_str}\n{common_prompt}{output_requirements}"

    answer = future_trajectory_str  # Trajectory QA answer

    qa_pair = {
            "id": idx,
            "image": image_paths,
            "conversations": [
            ]
        }
    assert len(agent_states) == len(box_2d) == len(agent_names)
    agent_boxes = []
    agent_boxes_2d = []
    agent_names_all = []
    for i, label in enumerate(agent_labels):
        if label:  # Only consider valid agents
            agent_boxes.append(agent_states[i])
            agent_boxes_2d.append(box_2d[i])
            agent_names_all.append(agent_names[i])
    
    desc_question, desc_answer = await get_img_description_qa(image_paths[0], "front", token)

    qa_pair["conversations"].append({"from": "human", "value": desc_question})
    qa_pair["conversations"].append({"from": "gpt", "value": desc_answer})

    trafc_question, trafc_answer = await get_traffic_congestion_qa(image_paths[0], token)

    qa_pair["conversations"].append({"from": "human", "value": trafc_question})
    qa_pair["conversations"].append({"from": "gpt", "value": trafc_answer})

    trafl_question, trafl_answer = await get_traffic_light_qa(image_paths[0], token)

    if trafl_answer != None: 
        qa_pair["conversations"].append({"from": "human", "value": trafl_question})
        qa_pair["conversations"].append({"from": "gpt", "value": trafl_answer})

    road_question, road_answer = await get_road_sign_qa(image_paths[0],token)

    qa_pair["conversations"].append({"from": "human", "value": road_question})
    qa_pair["conversations"].append({"from": "gpt", "value": road_answer})

    drive_inf_question, drive_inf_answer = await get_driving_influence_qa(image_paths[0],token)

    qa_pair["conversations"].append({"from": "human", "value": drive_inf_question})
    qa_pair["conversations"].append({"from": "gpt", "value": drive_inf_answer})

    vru_question, vru_answer = await get_vru_qa(token, agent_boxes, agent_names)
    if vru_answer != None:
        qa_pair["conversations"].append({"from": "human", "value": vru_question})
        qa_pair["conversations"].append({"from": "gpt", "value": vru_answer})
    
    plan_question, plan_answer = await get_plan_qa(token, future_trajectory_points, history_trajectory, command_str)

    qa_pair["conversations"].append({"from": "human", "value": plan_question})
    qa_pair["conversations"].append({"from": "gpt", "value": plan_answer})

    plan_exp_question, plan_exp_answer = await get_plan_explaination_qa(image_paths[0], token, future_trajectory_points, history_trajectory, command_str)

    qa_pair["conversations"].append({"from": "human", "value": plan_exp_question})
    qa_pair["conversations"].append({"from": "gpt", "value": plan_exp_answer})

    driving_beh_question, driving_beh_answer = await get_driving_behavior_qa(image_paths[0],token, history_trajectory)

    qa_pair["conversations"].append({"from": "human", "value": driving_beh_question})
    qa_pair["conversations"].append({"from": "gpt", "value": driving_beh_answer})

    traj_question, traj_answer = await get_traj_results_qa(token)

    if traj_answer != None:
        qa_pair["conversations"].append({"from": "human", "value": traj_question})
        qa_pair["conversations"].append({"from": "gpt", "value": traj_answer})
    

    image_context_token = "<image>\n"
    qa_pair["conversations"][0]['value'] = image_context_token + qa_pair["conversations"][0]['value']


    
    dis_cal_question, dis_cal_answer = await get_dis_cal_qa(image_paths[0],token, agent_boxes, agent_names_all,agent_boxes_2d)
    if dis_cal_question != None:
        qa_pair["conversations"].append({"from": "human", "value": dis_cal_question})
        qa_pair["conversations"].append({"from": "gpt", "value": dis_cal_answer})

    qa_pairs_3d = await get_3d_info_qa(image_paths[0], token, agent_boxes, agent_names_all, agent_boxes_2d)
    if qa_pairs_3d:
        for qa in qa_pairs_3d:
            qa_pair["conversations"].append({"from": "human", "value": qa["question"]})
            qa_pair["conversations"].append({"from": "gpt", "value": qa["answer"]})


    return qa_pair


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    async def main(cfg: DictConfig) -> None:
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Global Seed set to {cfg.seed}")
        logger.info(f"Path where all results are stored: {cfg.output_dir}")

        logger.info("Building Agent")
        agent: AbstractAgent = instantiate(cfg.agent)

        logger.info("Building SceneLoader")
        train_data = build_datasets(cfg, agent)
        logger.info("Building Datasets")
        logger.info("Num training samples: %d", len(train_data))

        batch_size = 256
        qa_pairs = []

        with open('navsim_traj_base_pipeline_v5.jsonl', 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(train_data), batch_size)):
                batch = [train_data[j] for j in range(i, min(i + batch_size, len(train_data)))] #手动创建批次
                tasks = [
                    process_data_and_create_qa_pair(
                        idx, ego_statuses, cameras, future_trajectory, agent_states, agent_labels, agent_names, token, velocity_3d,box_2d
                    )
                    for idx, (ego_statuses, cameras, future_trajectory, agent_states, agent_labels, agent_names, token, velocity_3d,box_2d) in enumerate(batch, start=i)
                ]
                results = await asyncio.gather(*tasks)

                for qa_pair in results:
                    json.dump(qa_pair, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
            f.close()

    asyncio.run(main(cfg))

if __name__ == "__main__":
    hydra_main()