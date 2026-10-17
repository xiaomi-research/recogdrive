"""Hierarchical cognitive VQA on NAVSIM scenes.

Generation asks an OpenAI-compatible VLM (image description, congestion, lights, signs,
influence, plan explanation, driving behavior). Refinement adds geometric answers (VRU,
SPEED/PATH plan, distance, 3D info, 3D detection) plus an external trajectory-result file.
Quality control drops negative traffic-light and VRU answers and skips any field the loader
does not have. Intermediate VLM text is cached under ``<out>/.cache`` (or beside the jsonl).

    python -m recogdrive.data.cognitive --config-path <training config dir> --config-name default_training \
        train_test_split=navtrain out=/path/to/vqa.jsonl split=train cam_type=single prompt_type=base

Reads ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, and optional ``COGNITIVE_VLM_MODEL``.
``traj_results=/path/to/traj_results_new`` (or ``TRAJ_RESULTS``) supplies per-token JSON.
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("recogdrive.data.cognitive")

PEDAL_STATUS = {"const": "KEEP", "accelerate": "ACCELERATE", "decelerate": "DECELERATE", "stop": "STOP"}
PATH_STATUS = {
    "right turn": "RIGHT_TURN",
    "right lane change": "RIGHT_CHANGE",
    "left turn": "LEFT_TURN",
    "left lane change": "LEFT_CHANGE",
    "straight": "STRAIGHT",
}
CAM_TYPES = ("single", "multi_3", "multi_5", "cont")
PROMPT_TYPES = ("base", "vel_and_acc")
OURS = ("out", "split", "cam_type", "prompt_type", "traj_results", "batch_size")
VRU_NEGATIVE = (
    "No, I don't see any vulnerable road users ahead of me, such as bicycles, motorcycles, or pedestrians."
)

_cache_dir = None
_traj_dir = None
_client = None
_model_id = None
_box2d_warned = False

IMG_DESC_Q = (
    "Suppose you are driving, and I'm providing you with the image captured by the car's {img_type}, "
    "generate a description of the driving scene which includes the key factors for driving planning, "
    "including the positions and movements of vehicles and pedestrians; prevailing weather conditions; "
    "time of day, distinguishing between daylight and nighttime; road conditions, indicating smooth "
    "surfaces or the presence of obstacles; and the status of traffic lights which influence your "
    "decision making, specifying whether they are red or green. The description should be concise, "
    "providing an accurate understanding of the driving environment to facilitate informed decision-making."
)
TRAFFIC_CONG_Q = (
    "Based on the provided forward-facing image from a car's perspective, please analyze the current "
    "traffic congestion level on the road. Determine whether the road is heavily congested, moderately "
    "congested, or clear. Then, based on the congestion level, advise whether the driving behavior "
    "should be cautious or normal. "
)
TRAFFIC_LIGHT_Q = (
    "Given the provided forward-facing image from a car's perspective, identify if there is a traffic "
    "light that affects the car's behavior. Please respond with a complete sentence describing the "
    "traffic light status, such as 'The traffic light is red', 'The traffic light is green', "
    "'The traffic light is yellow', or 'There is no traffic light visible'."
)
ROAD_SIGN_Q = (
    "Based on the provided forward-facing image from a car's perspective, identify and describe the "
    "road markings and signs, including traffic lines, road signs, pedestrian crossings, and speed bumps. "
    "Explain their meaning and advise on the appropriate driving action in a concise, single paragraph "
    "without using bullet points or numbered lists."
)
DRIVING_INFLUENCE_Q = (
    "Based on the provided forward-facing image from a car's perspective, identify the most influential "
    "object that affects the current driving situation (e.g., a vehicle ahead, a pedestrian, a traffic "
    "light). Describe why this object is influential, including its current state or actions (e.g., "
    "braking, turning, moving). Finally, advise the driver on the appropriate driving action in a "
    "concise paragraph."
)


def cache_file(kind, token, suffix):
    token = str(token)
    if not token or token in (".", "..") or "/" in token or "\\" in token:
        raise ValueError(f"unsafe scene token {token!r}")
    return Path(_cache_dir) / kind / f"{token}{suffix}"


def read_text(kind, token):
    if _cache_dir is None:
        return None
    path = cache_file(kind, token, ".txt")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


def write_text(kind, token, text):
    if _cache_dir is None:
        return
    path = cache_file(kind, token, ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_json(kind, token):
    if _cache_dir is None:
        return None
    path = cache_file(kind, token, ".json")
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(kind, token, payload):
    if _cache_dir is None:
        return
    path = cache_file(kind, token, ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def async_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        key = os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL")
        if not key or not base:
            raise RuntimeError("Set OPENAI_API_KEY and OPENAI_BASE_URL for cognitive VLM calls")
        _client = AsyncOpenAI(api_key=key, base_url=base)
    return _client


async def model_id():
    global _model_id
    if _model_id is None:
        _model_id = os.environ.get("COGNITIVE_VLM_MODEL")
        if not _model_id:
            models = await async_client().models.list()
            _model_id = models.data[0].id
    return _model_id


async def async_qwen_vl_72b_infer(args, max_retries=6, retry_delay=5):
    """Async OpenAI-compatible VLM call. Retries transient failures."""
    from PIL import Image

    client = async_client()
    model = await model_id()
    for attempt in range(max_retries):
        try:
            with open(args.img_file, "rb") as image_file:
                img = Image.open(image_file)
                img_resized = img.resize((960, 540))
                buffer = io.BytesIO()
                img_resized.save(buffer, format="JPEG")
                buffer.seek(0)
                encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
            completion = await client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ],
                }],
                timeout=600,
            )
            return completion.choices[0].message.content
        except Exception as exc:
            if attempt < max_retries - 1:
                logger.warning("VLM attempt %d failed (%s); retrying in %ds", attempt + 1, exc, retry_delay)
                await asyncio.sleep(retry_delay)
            else:
                logger.error("VLM failed after %d attempts: %s", max_retries, exc)
                raise


def _query(query, img_file):
    return type("Args", (), {"query": query, "img_file": img_file})()


async def infer(query, img_file):
    if not img_file or not os.path.isfile(img_file):
        raise FileNotFoundError(img_file or "")
    return await async_qwen_vl_72b_infer(_query(query, img_file))


async def vlm_qa(kind, token, image, question, drop_if=None):
    answer = read_text(kind, token)
    if answer is None:
        answer = await infer(question, image)
        if not isinstance(answer, str) or not answer.strip():
            return None, None
        write_text(kind, token, answer)
    if drop_if is not None and drop_if(answer):
        return None, None
    return question, answer


async def get_img_description_qa(img_path, img_type, token):
    return await vlm_qa("img_desc", token, img_path, IMG_DESC_Q.format(img_type=img_type))


async def get_traffic_congestion_qa(cf_img_path, token):
    return await vlm_qa("traf_cong", token, cf_img_path, TRAFFIC_CONG_Q)


async def get_traffic_light_qa(cf_img_path, token):
    return await vlm_qa(
        "traf_light", token, cf_img_path, TRAFFIC_LIGHT_Q,
        drop_if=lambda answer: "There is no traffic light" in answer,
    )


async def get_road_sign_qa(cf_img_path, token):
    return await vlm_qa("road_sign", token, cf_img_path, ROAD_SIGN_Q)


async def get_driving_influence_qa(cf_img_path, token):
    return await vlm_qa("driving_influence", token, cf_img_path, DRIVING_INFLUENCE_Q)


async def get_vru_qa(token, agent_boxes, agent_names, vru_dis_thresh=40.0):
    question = (
        f"Do you see any vulnerable road users within {int(vru_dis_thresh)} meters ahead of you, "
        "such as cyclists, motorcyclists, or pedestrians?"
    )
    answer = read_text("vru", token)
    if answer is None:
        vru_list = []
        names = list(agent_names)
        for i, box in enumerate(agent_boxes):
            box = np.asarray(box, dtype=np.float64).reshape(-1)
            if box.size < 2 or i >= len(names):
                continue
            obj_loc = box[:2]
            if str(names[i]) not in ("motorcycle", "pedestrian") or np.linalg.norm(obj_loc) >= vru_dis_thresh:
                continue
            x_dis, y_dis = float(box[0]), float(box[1])
            if y_dis <= -2.0:
                lat_pos = f" and {abs(y_dis):.2f} meters to the right"
            elif y_dis >= 2.0:
                lat_pos = f" and {abs(y_dis):.2f} meters to the left"
            else:
                lat_pos = ""
            vru_list.append(f"a {names[i]} located {abs(x_dis):.2f} meters ahead of me{lat_pos}")
        if vru_list:
            answer = "Yes, I see " + ", and ".join(vru_list) + "."
        else:
            answer = VRU_NEGATIVE
        write_text("vru", token, answer)
    if VRU_NEGATIVE in answer:
        return None, None
    return question, answer


def get_obj_acc_or_dec(trajectory, vel_diff_thresh=3.0):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 2:
        return "const"
    velocity = np.linalg.norm(trajectory[1:, :2] - trajectory[:-1, :2], axis=-1) / 0.5
    if np.max(velocity) < 2.0:
        return "stop"
    vel_diff = velocity[-1] - velocity[0]
    if vel_diff >= vel_diff_thresh:
        return "accelerate"
    if vel_diff <= -vel_diff_thresh:
        return "decelerate"
    return "const"


def get_obj_turn_or_lane_change(trajectory, lat_thresh=4.0, angle_thresh=5.0):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 1 or trajectory.shape[1] < 2:
        return "straight"
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    endpoint_angle = math.degrees(math.atan2(x[-1], y[-1]))
    angle_diff = endpoint_angle - 90.0
    if y[-1] > lat_thresh and angle_diff <= -angle_thresh:
        return "left turn"
    if y[-1] > lat_thresh and abs(angle_diff) < angle_thresh:
        return "left lane change"
    if y[-1] <= -lat_thresh and angle_diff >= angle_thresh:
        return "right turn"
    if y[-1] <= -lat_thresh and abs(angle_diff) < angle_thresh:
        return "right lane change"
    return "straight"


def get_obj_acc_or_dec_from_vel(velocity, vel_diff_thresh=3.0):
    """SPEED from a (T, 3) velocity history: accelerate, decelerate, const, or stop."""
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[0] < 1 or velocity.shape[1] < 2:
        return "const"
    speed = np.linalg.norm(velocity[:, :2], axis=-1)
    vel_diff = speed[-1] - speed[0]
    if np.max(speed) < 2.0:
        return "stop"
    if vel_diff >= vel_diff_thresh:
        return "accelerate"
    if vel_diff <= -vel_diff_thresh:
        return "decelerate"
    return "const"


def get_obj_turn_or_lane_change_from_vel(velocity, lat_thresh=4.0, angle_thresh=5.0):
    """PATH from the last two velocity rows. Column 2 is heading in radians."""
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[0] < 2 or velocity.shape[1] < 2:
        return "straight"
    x_diff = velocity[-1, 0] - velocity[-2, 0]
    y_diff = velocity[-1, 1] - velocity[-2, 1]
    endpoint_angle = math.degrees(math.atan2(x_diff, y_diff))
    heading = math.degrees(velocity[-2, 2]) if velocity.shape[1] > 2 else 0.0
    angle_diff = endpoint_angle - heading
    if y_diff > lat_thresh and angle_diff <= -angle_thresh:
        return "left turn"
    if y_diff > lat_thresh and abs(angle_diff) < angle_thresh:
        return "left lane change"
    if y_diff <= -lat_thresh and angle_diff >= angle_thresh:
        return "right turn"
    if y_diff <= -lat_thresh and abs(angle_diff) < angle_thresh:
        return "right lane change"
    return "straight"


def get_decision(ego_speed_plan, ego_path_plan):
    pedal_decision = {
        "KEEP": "maintain the current speed",
        "ACCELERATE": "accelerate",
        "DECELERATE": "decelerate",
        "STOP": "stop the car",
    }
    path_decision = {
        "RIGHT_TURN": "turn right",
        "RIGHT_CHANGE": "change to the right lane",
        "LEFT_TURN": "turn left",
        "LEFT_CHANGE": "change to the left lane",
        "STRAIGHT": "go straight",
    }
    if ego_speed_plan == "STOP":
        return pedal_decision[ego_speed_plan]
    return pedal_decision[ego_speed_plan] + " and " + path_decision[ego_path_plan]


def format_number(n, decimal_places=2):
    if hasattr(n, "item"):
        n = n.item()
    n = float(n)
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    return f"{n:+.{decimal_places}f}"


def parse_future_points(future_trajectory_points):
    rows = []
    for point in future_trajectory_points:
        parts = point.split(",")
        rows.append([float(parts[0][1:]), float(parts[1]), float(parts[2][:-1])])
    return np.asarray(rows, dtype=np.float64)


def history_text(history_trajectory):
    return chr(2).join(
        f'  - t-{3 - i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)
    )


def plan_from_future(future_trajectory_points):
    future = parse_future_points(future_trajectory_points)
    speed = PEDAL_STATUS[get_obj_acc_or_dec(future)]
    path = PATH_STATUS[get_obj_turn_or_lane_change(future)]
    return speed, path


async def get_plan_qa(token, future_trajectory_points, history_trajectory, command_str):
    question = (
        f"Your historical trajectories are {history_text(history_trajectory)},"
        f"the navigation command is '{command_str}', "
        "based on the understanding of the driving scene and the navigation information, "
        "what is your plan for the next three seconds? "
        "Please answer your SPEED plan and your PATH plan. "
        "SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, "
        "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. "
        "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
    )
    answer = read_text("plan", token)
    if answer is None:
        if not future_trajectory_points:
            return None, None
        speed, path = plan_from_future(future_trajectory_points)
        answer = speed + ", " + path + "\n"
        write_text("plan", token, answer)
    return question, answer


async def get_plan_explaination_qa(cf_img_path, token, future_trajectory_points, history_trajectory, command_str):
    if not future_trajectory_points:
        return None, None
    speed, path = plan_from_future(future_trajectory_points)
    decision = get_decision(speed, path)
    question = (
        "You are driving, "
        f"Your historical trajectories are {history_text(history_trajectory)},"
        f"and the navigation command is '{command_str}', "
        "your driving decision for the next four seconds is to "
        f"{decision}. "
        "Based on the provided image of the driving environment, "
        "explain the most likely reason for this decision in one or two concise sentence."
    )
    return await vlm_qa("plan_explain", token, cf_img_path, question)


async def get_driving_behavior_qa(cf_img_path, token, history_trajectory):
    question = (
        f"Based on the provided image and the historical trajectory: {history_text(history_trajectory)}, "
        "describe the current driving behavior of the vehicle in one or two concise sentences."
    )
    return await vlm_qa("driving_behavior", token, cf_img_path, question)


async def get_traj_results_qa(token):
    """Question and answer from ``<traj_results_new>/<token>.json``, if that file exists."""
    if not _traj_dir:
        return None, None
    path = Path(_traj_dir) / f"{token}.json"
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    question, answer = data.get("question") or "", data.get("answer") or ""
    if not question or not answer:
        return None, None
    return question, answer


def image_size(path):
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return None, None


def normalize_coordinates(box, image_width, image_height):
    if hasattr(box, "tolist"):
        box = box.tolist()
    x1, y1, x2, y2 = (float(v) for v in box)
    return [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000),
    ]


def _boxes_in_frame(agent_boxes, agent_names, box_2d, image_width, image_height):
    kept_boxes, kept_agents, kept_names = [], [], []
    names = list(agent_names)
    for i, box in enumerate(box_2d):
        if box is None or i >= len(agent_boxes) or i >= len(names):
            continue
        values = np.asarray(box, dtype=np.float64).reshape(-1)
        if values.size < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in values[:4])
        if x1 >= 0 and y1 >= 0 and x2 <= image_width and y2 <= image_height:
            kept_boxes.append(values[:4])
            kept_agents.append(np.asarray(agent_boxes[i], dtype=np.float64).reshape(-1))
            kept_names.append(names[i])
    return kept_boxes, kept_agents, kept_names


async def get_dis_cal_qa(cf_img_path, token, agent_boxes, agent_names, box_2d, image_width=None, image_height=None):
    cached = read_json("distance_calculation_new", token)
    if isinstance(cached, dict) and cached.get("question") and cached.get("answer"):
        return cached["question"], cached["answer"]
    if box_2d is None or len(box_2d) == 0 or any(box is None for box in box_2d):
        return None, None
    if image_width is None or image_height is None:
        image_width, image_height = image_size(cf_img_path)
    if not image_width or not image_height:
        return None, None
    filtered_boxes, filtered_agents, filtered_names = _boxes_in_frame(
        agent_boxes, agent_names, box_2d, image_width, image_height,
    )
    if len(filtered_boxes) < 2:
        return None, None

    def boxes_overlap(box1, box2):
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2
        x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
        y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
        return x_overlap > 0 and y_overlap > 0

    valid_pairs = [
        (i, j) for i in range(len(filtered_boxes)) for j in range(i + 1, len(filtered_boxes))
        if not boxes_overlap(filtered_boxes[i], filtered_boxes[j])
    ]
    if not valid_pairs:
        return None, None
    box1_idx, box2_idx = random.choice(valid_pairs)
    normalized = [normalize_coordinates(box, image_width, image_height) for box in filtered_boxes]
    box1, box2 = normalized[box1_idx], normalized[box2_idx]
    obj_name1, obj_name2 = filtered_names[box1_idx], filtered_names[box2_idx]
    obj_with_box1 = f"<{obj_name1}><FRONT VIEW><box>{box1}</box>"
    obj_with_box2 = f"<{obj_name2}><FRONT VIEW><box>{box2}</box>"
    question = random.choice([
        f"How far apart are the {obj_with_box1} and the {obj_with_box2}?",
        f"What is the distance between the {obj_with_box1} and the {obj_with_box2}?",
        f"Calculate the separation between the {obj_with_box1} and the {obj_with_box2}.",
        f"Can you tell me the distance between the {obj_with_box1} and the {obj_with_box2}?",
        f"What's the gap between the {obj_with_box1} and the {obj_with_box2}?",
    ])
    dist_3d = np.linalg.norm(filtered_agents[box1_idx][:2] - filtered_agents[box2_idx][:2])
    answer = f"The {obj_with_box1} and the {obj_with_box2} are approximately {dist_3d:.2f} meters apart."
    write_json("distance_calculation_new", token, {"question": question, "answer": answer})
    return question, answer


async def get_3d_info_qa(cf_img_path, token, agent_boxes, agent_names, box_2d, image_width=None, image_height=None):
    cached = read_json("3d_info_qa", token)
    if isinstance(cached, dict) and isinstance(cached.get("qa_pairs"), list):
        return cached["qa_pairs"]
    if box_2d is None or len(box_2d) == 0 or any(box is None for box in box_2d):
        return []
    if image_width is None or image_height is None:
        image_width, image_height = image_size(cf_img_path)
    if not image_width or not image_height:
        return []
    filtered_boxes, filtered_agents, filtered_names = _boxes_in_frame(
        agent_boxes, agent_names, box_2d, image_width, image_height,
    )
    if not filtered_boxes:
        return []
    indices = list(range(len(filtered_boxes)))
    if len(indices) > 5:
        indices = random.sample(indices, 5)
    qa_pairs = []
    for i in indices:
        agent_box = filtered_agents[i]
        if agent_box.size < 7:
            continue
        normalized_box = normalize_coordinates(filtered_boxes[i], image_width, image_height)
        question = random.choice([
            f"What is the 3D information of the object with the 2D box {normalized_box}?",
            f"Please provide the 3D details for the object whose 2D bounding box is {normalized_box}.",
            f"Can you tell me the 3D information for the object located at 2D box {normalized_box}?",
        ])
        x, y, _z, length, width, height, heading = (float(v) for v in agent_box[:7])
        answer = (
            f"The object is a {filtered_names[i]} location: ({x:.2f}, {y:.2f}), "
            f"length: {length:.2f}, width: {width:.2f}, height: {height:.2f}, heading: {heading:.2f}."
        )
        qa_pairs.append({"question": question, "answer": answer})
    if qa_pairs:
        write_json("3d_info_qa", token, {"qa_pairs": qa_pairs})
    return qa_pairs


async def get_3d_det_qa(agent_boxes, agent_names):
    names = list(agent_names)
    rows = []
    for i, box in enumerate(agent_boxes):
        if i >= len(names):
            break
        values = np.asarray(box, dtype=np.float64).reshape(-1)
        if values.size < 7:
            continue
        rows.append((values, names[i]))
    if not rows:
        return None, None
    question = (
        "Detect every bicycle, pedestrian, and vehicle in 3D sorted from nearest to farthest and respond "
        "in the format: Object N: (x, y, z, l, w, h, heading, name). "
        "Where x, y, z are the center coordinates of the object in ego-coordinate system, "
        "l, w, h are length, width, height of the bounding box, "
        "heading is the object heading, and name is the object class name, sorted by distance from "
        "nearest to farthest."
    )
    lines = []
    for i, (box, name) in enumerate(rows):
        x, y, z, length, width, height, heading = (float(v) for v in box[:7])
        lines.append(
            f"Object {i + 1}: ({x:.2f}, {y:.2f}, {z:.2f}, {length:.2f}, {width:.2f}, {height:.2f}, "
            f"{heading:.2f}, {name})"
        )
    return question, " ".join(lines)


def check_heuristics():
    """Fails if the SPEED/PATH rules drift. Run before a generation job."""
    assert get_obj_acc_or_dec(np.zeros((4, 3))) == "stop"
    straight = np.stack([np.array([i, 0.0, 0.0]) for i in range(4)])
    assert get_obj_acc_or_dec(straight) == "const"
    assert get_obj_turn_or_lane_change(straight) == "straight"
    accel = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [3.0, 0, 0]], dtype=float)
    assert get_obj_acc_or_dec(accel) == "accelerate"
    left = np.array([[0, 0, 0], [0, 2, 0], [0.2, 6, 0], [1, 10, 0]], dtype=float)
    assert get_obj_turn_or_lane_change(left) == "left turn"
    vel = np.tile(np.array([[3.0, 0.0, 0.0]]), (8, 1))
    assert get_obj_acc_or_dec_from_vel(vel) == "const"
    assert get_obj_turn_or_lane_change_from_vel(vel) == "straight"
    turning = np.zeros((8, 3))
    turning[-2] = [0.0, 0.0, 0.2]
    turning[-1] = [0.0, 5.0, 0.0]
    assert get_obj_turn_or_lane_change_from_vel(turning) == "left turn"
    assert get_decision("STOP", "LEFT_TURN") == "stop the car"
    assert get_decision("KEEP", "STRAIGHT") == "maintain the current speed and go straight"
    assert format_number(0.001) == 0.0
    assert format_number(1.2) == "+1.20"
    parsed = parse_future_points([f"({format_number(1.234)}, {format_number(-0.2)}, {format_number(0)})"])
    assert parsed.shape == (1, 3) and abs(parsed[0, 0] - 1.23) < 1e-9 and abs(parsed[0, 1] + 0.2) < 1e-9


def camera_file(cameras, index, name):
    if index >= len(cameras) or index < -len(cameras):
        return None
    cam = getattr(cameras[index], name, None)
    image = getattr(cam, "image", None)
    if isinstance(image, Path):
        image = str(image)
    if isinstance(image, str) and os.path.isfile(image):
        return image
    return None


def process_multi_3_images(image_paths, token):
    """Stitch front/left/right into one JPEG. Returns (paths, prompt) or None."""
    if _cache_dir is None or len(image_paths) != 3 or any(not path for path in image_paths):
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    opened = []
    try:
        for image_path in image_paths:
            with Image.open(image_path) as src:
                opened.append(src.convert("RGB"))
        height = min(img.height for img in opened)
        resized = [img.resize((max(1, int(img.width * height / img.height)), height)) for img in opened]
        canvas = Image.new("RGB", (sum(img.width for img in resized), height))
        x = 0
        for img in resized:
            canvas.paste(img, (x, 0))
            x += img.width
        path = cache_file("multi_3", token, ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, format="JPEG")
    except Exception:
        logger.warning("multi_3 stitch failed for %s", token, exc_info=True)
        return None
    finally:
        for img in opened:
            img.close()
    return [str(path)], "1. Visual perception from stitched front, left and right camera views\n"


def select_images(cameras, cam_type, token):
    front = camera_file(cameras, -1, "cam_f0")
    if cam_type == "multi_3":
        paths = [camera_file(cameras, -1, name) for name in ("cam_f0", "cam_l0", "cam_r0")]
        stitched = process_multi_3_images(paths, token)
        if stitched:
            return stitched[0], "<FRONT VIEW>:\n<image>\n", stitched[1]
        if front:
            logger.warning("multi_3 stitch unavailable for %s; using the front camera", token)
    if cam_type == "multi_5":
        paths = [
            camera_file(cameras, -1, name)
            for name in ("cam_f0", "cam_l0", "cam_l1", "cam_r1", "cam_r2")
        ]
        paths = [path for path in paths if path]
        if paths:
            return paths, "", "1. Visual perception from front, left1, left2, right1 and right2 camera views\n"
    if cam_type == "cont":
        paths, lines = [], []
        for i in range(min(4, len(cameras))):
            path = camera_file(cameras, i, "cam_f0")
            if not path:
                continue
            paths.append(path)
            lines.append(f"<FRONT VIEW>Frame-{len(paths)}: <image>\n")
        if paths:
            return paths, "".join(lines), (
                "1. Visual perception from continuous front camera views of the last 4 timesteps\n"
            )
    if not front:
        return None, "", ""
    return [front], "<FRONT VIEW>:\n<image>\n", "1. Visual perception from front camera view\n"


def command_of(ego_status):
    command = np.asarray(getattr(ego_status, "driving_command", [])).reshape(-1)
    names = ("turn left", "go straight", "turn right")
    hits = [names[i] for i in range(min(len(names), command.size)) if command[i] == 1]
    return hits[0] if hits else "unknown"


def vector2(status, name):
    value = getattr(status, name, None)
    if value is None:
        return None
    value = np.asarray(value).reshape(-1)
    if value.size < 2:
        return None
    return value


def project_box2d(boxes, camera):
    """Front-camera 2D boxes. None when intrinsics or the navsim projection helpers are missing."""
    global _box2d_warned
    if camera is None or getattr(camera, "intrinsics", None) is None:
        return None
    if getattr(camera, "sensor2lidar_rotation", None) is None:
        return None
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    try:
        from navsim.common.enums import BoundingBoxIndex
        from navsim.visualization.camera import (
            _rotation_3d_in_axis,
            _transform_annotations_to_camera,
            _transform_points_to_image,
        )

        boxes_camera = _transform_annotations_to_camera(
            boxes, np.asarray(camera.sensor2lidar_rotation), np.asarray(camera.sensor2lidar_translation),
        )
        positions = boxes_camera[:, BoundingBoxIndex.POSITION]
        dimensions = boxes_camera[:, BoundingBoxIndex.DIMENSION]
        heading = boxes_camera[:, BoundingBoxIndex.HEADING]
        corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]] - np.array([0.5, 0.5, 0.5])
        corners = dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
        corners = _rotation_3d_in_axis(corners, heading, axis=1)
        corners += positions.reshape(-1, 1, 3)
        box_corners, _ = _transform_points_to_image(corners.reshape(-1, 3), np.asarray(camera.intrinsics))
        box_corners = box_corners.reshape(-1, 8, 2)
        return np.asarray(
            [[corner[:, 0].min(), corner[:, 1].min(), corner[:, 0].max(), corner[:, 1].max()] for corner in box_corners],
            dtype=np.float32,
        )
    except Exception:
        if not _box2d_warned:
            logger.warning("2D boxes unavailable; distance and 3D-info QA will be skipped", exc_info=True)
            _box2d_warned = True
        return None


def empty_agents():
    return (
        np.zeros((0, 7), np.float32),
        np.zeros((0,), dtype=bool),
        [],
        np.zeros((0, 9, 3), np.float32),
        None,
    )


def future_velocities(scene, kept, future_frames=8):
    if not kept:
        return np.zeros((0, future_frames + 1, 3), np.float32)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    rows = []
    for _distance, _box, _name, track, vel, _box2d in kept:
        series = [np.asarray(vel, dtype=np.float32).reshape(-1)[:3]]
        for step in range(frame_idx + 1, frame_idx + future_frames + 1):
            nxt = np.zeros(3, np.float32)
            if track is not None and step < len(scene.frames):
                ann = scene.frames[step].annotations
                tokens = list(getattr(ann, "track_tokens", []))
                if track in tokens:
                    nxt = np.asarray(ann.velocity_3d[tokens.index(track)], dtype=np.float32).reshape(-1)[:3]
            series.append(nxt)
        rows.append(np.stack(series))
    return np.stack(rows)


def front_agents(annotations, scene, camera):
    """Bicycle, pedestrian, and vehicle boxes inside the front 60 degree FOV and a 32 m lidar window."""
    boxes = np.asarray(getattr(annotations, "boxes", []), dtype=np.float32)
    names = [str(name) for name in getattr(annotations, "names", [])]
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1) if boxes.size else boxes.reshape(0, 7)
    n = len(names)
    if n == 0 or boxes.shape[0] != n or boxes.shape[1] < 7:
        return empty_agents()
    velocities = np.asarray(getattr(annotations, "velocity_3d", np.zeros((n, 3))), dtype=np.float32)
    if velocities.shape[0] != n:
        velocities = np.zeros((n, 3), np.float32)
    tracks = list(getattr(annotations, "track_tokens", []))
    projected = project_box2d(boxes, camera)
    kept = []
    for i in range(n):
        if names[i] not in ("bicycle", "pedestrian", "vehicle"):
            continue
        x, y = float(boxes[i, 0]), float(boxes[i, 1])
        if not (-32 <= x <= 32 and -32 <= y <= 32):
            continue
        angle = math.degrees(math.atan2(y, x))
        if angle < -30 or angle > 30:
            continue
        box2d = None if projected is None else projected[i]
        track = tracks[i] if i < len(tracks) else None
        kept.append((math.hypot(x, y), boxes[i].astype(np.float32), names[i], track, velocities[i], box2d))
    kept.sort(key=lambda item: item[0])
    states = np.stack([item[1] for item in kept]) if kept else np.zeros((0, 7), np.float32)
    labels = np.ones(len(kept), dtype=bool)
    agent_names = [item[2] for item in kept]
    box2d = None
    if projected is not None:
        box2d = np.stack([item[5] for item in kept]) if kept else np.zeros((0, 4), np.float32)
    return states, labels, agent_names, future_velocities(scene, kept), box2d


class _CameraAgent:
    """Sensor config for SceneLoader: camera paths only, no lidar and no VLM."""

    load_image_path = True

    def __init__(self, cam_type, num_history):
        self.cam_type = cam_type
        self.num_history = num_history

    def get_sensor_config(self):
        import dataclasses

        from navsim.common.dataclasses import SensorConfig

        if self.cam_type == "cont":
            config = SensorConfig.build_no_sensors()
            config.cam_f0 = True
            return config
        current = max(self.num_history - 1, 0)
        return dataclasses.replace(SensorConfig.build_all_sensors(include=[current]), lidar_pc=False)


def open_loader(cfg, split, cam_type):
    from hydra.utils import instantiate

    from recogdrive.data.navsim import scene_loader

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    if split == "train":
        logs = list(cfg.train_logs)
    elif split == "val":
        logs = list(cfg.val_logs)
    elif split == "all":
        logs = None
    else:
        raise ValueError(f"split must be train, val, or all, got {split!r}")
    if logs is not None:
        if scene_filter.log_names is not None:
            scene_filter.log_names = [name for name in scene_filter.log_names if name in set(logs)]
        else:
            scene_filter.log_names = logs
    agent = _CameraAgent(cam_type, int(scene_filter.num_history_frames))
    return scene_loader(cfg, agent, scene_filter)


def load_sample(loader, token):
    scene = loader.get_scene_from_token(token)
    agent_input = scene.get_agent_input()
    ego_statuses = list(agent_input.ego_statuses)
    cameras = list(agent_input.cameras)
    if not ego_statuses or not cameras:
        logger.warning("skip %s: missing ego status or cameras", token)
        return None
    try:
        future = np.asarray(scene.get_future_trajectory(num_trajectory_frames=8).poses)
    except Exception:
        logger.warning("skip future trajectory for %s", token)
        future = None
    frame_idx = scene.scene_metadata.num_history_frames - 1
    try:
        frame = scene.frames[frame_idx]
        agents = front_agents(frame.annotations, scene, frame.cameras.cam_f0)
    except Exception:
        logger.warning("skip agents for %s", token, exc_info=True)
        agents = empty_agents()
    states, labels, names, velocity_3d, box_2d = agents
    return ego_statuses, cameras, future, states, labels, names, token, velocity_3d, box_2d


def add_turn(qa_pair, question, answer):
    if question and answer:
        qa_pair["conversations"].append({"from": "human", "value": question})
        qa_pair["conversations"].append({"from": "gpt", "value": answer})


def planning_question(image_prompt_lines, image_prompt, history_trajectory, command_str, prompt_type, ego_status):
    # The nbsp matches the original planning prompt byte-for-byte.
    context = chr(2).join(
        f' \xa0 - t-{3 - i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)
    )
    common = (
        f"As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt}"
        f"2. Historical motion context (last 4 timesteps):{context}\n"
        f"3. Active navigation command: [{command_str.upper()}]"
    )
    requirements = (
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )
    if prompt_type == "vel_and_acc":
        velocity, acceleration = vector2(ego_status, "ego_velocity"), vector2(ego_status, "ego_acceleration")
        if velocity is not None and acceleration is not None:
            common += (
                f"\n4. Current velocity: ({format_number(velocity[0])}, {format_number(velocity[1])})"
                f"\n5. Current acceleration: ({format_number(acceleration[0])}, {format_number(acceleration[1])})"
            )
    return f"{image_prompt_lines}\n{common}{requirements}"


async def process_data_and_create_qa_pair(
    idx, ego_statuses, cameras, future_trajectory, agent_states, agent_labels, agent_names, token,
    velocity_3d, box_2d, prompt_type="base", cam_type="single",
):
    del velocity_3d  # gathered for the old signature; SPEED/PATH uses the future trajectory
    history_src = list(ego_statuses)[:4]
    history_trajectory = [
        {
            "x": format_number(status.ego_pose[0]),
            "y": format_number(status.ego_pose[1]),
            "heading": format_number(status.ego_pose[2]),
        }
        for status in history_src
        if getattr(status, "ego_pose", None) is not None and len(status.ego_pose) >= 3
    ]
    if not history_trajectory:
        return None
    command_str = command_of(ego_statuses[-1])
    image_paths, image_prompt_lines, image_prompt = select_images(cameras, cam_type, token)
    if not image_paths:
        logger.warning("skip %s: no camera image", token)
        return None
    future_trajectory_points = []
    if future_trajectory is not None:
        for traj_point in np.asarray(future_trajectory):
            if len(traj_point) < 3:
                continue
            future_trajectory_points.append(
                f"({format_number(traj_point[0])}, {format_number(traj_point[1])}, {format_number(traj_point[2])})"
            )
    future_trajectory_str = f"Here is the planning trajectory [PT, {', '.join(future_trajectory_points)}]."
    qa_pair = {"id": idx, "image": image_paths, "conversations": []}

    agent_boxes, agent_names_all, agent_boxes_2d = [], [], []
    labels = list(agent_labels) if agent_labels is not None else []
    names = list(agent_names) if agent_names is not None else []
    states = agent_states if agent_states is not None else []
    has_2d = box_2d is not None and len(box_2d) == len(states) == len(names)
    for i, label in enumerate(labels):
        if not label or i >= len(states) or i >= len(names):
            continue
        agent_boxes.append(states[i])
        agent_names_all.append(names[i])
        agent_boxes_2d.append(box_2d[i] if has_2d else None)

    front = image_paths[0]
    add_turn(qa_pair, *(await get_img_description_qa(front, "front", token)))
    add_turn(qa_pair, *(await get_traffic_congestion_qa(front, token)))
    add_turn(qa_pair, *(await get_traffic_light_qa(front, token)))
    add_turn(qa_pair, *(await get_road_sign_qa(front, token)))
    add_turn(qa_pair, *(await get_driving_influence_qa(front, token)))
    add_turn(qa_pair, *(await get_vru_qa(token, agent_boxes, agent_names_all)))
    if future_trajectory_points:
        add_turn(qa_pair, *(await get_plan_qa(token, future_trajectory_points, history_trajectory, command_str)))
        add_turn(
            qa_pair,
            *(await get_plan_explaination_qa(front, token, future_trajectory_points, history_trajectory, command_str)),
        )
    add_turn(qa_pair, *(await get_driving_behavior_qa(front, token, history_trajectory)))
    add_turn(qa_pair, *(await get_traj_results_qa(token)))
    if future_trajectory_points:
        question = planning_question(
            image_prompt_lines, image_prompt, history_trajectory, command_str, prompt_type, ego_statuses[-1],
        )
        # Placeholders are counted once on the first turn, so the planning turn keeps the view labels only.
        add_turn(qa_pair, question.replace("<image>", ""), future_trajectory_str)
    if has_2d:
        add_turn(qa_pair, *(await get_dis_cal_qa(front, token, agent_boxes, agent_names_all, agent_boxes_2d)))
        for pair in await get_3d_info_qa(front, token, agent_boxes, agent_names_all, agent_boxes_2d):
            add_turn(qa_pair, pair.get("question"), pair.get("answer"))
    add_turn(qa_pair, *(await get_3d_det_qa(agent_boxes, agent_names_all)))
    if not qa_pair["conversations"]:
        return None
    # One <image> placeholder per camera file, on the first human turn.
    present = sum(turn["value"].count("<image>") for turn in qa_pair["conversations"])
    missing = len(image_paths) - present
    if missing > 0:
        qa_pair["conversations"][0]["value"] = ("<image>\n" * missing) + qa_pair["conversations"][0]["value"]
    return qa_pair


def resolve_output(out):
    path = Path(out).expanduser()
    if path.suffix == ".jsonl":
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = path.parent / ".cache"
    else:
        path.mkdir(parents=True, exist_ok=True)
        cache = path / ".cache"
        path = path / "vqa.jsonl"
    cache.mkdir(parents=True, exist_ok=True)
    return path, cache


def apply_log_env(cfg):
    if os.environ.get("TRAIN_LOGS"):
        cfg.train_logs = [name for name in os.environ["TRAIN_LOGS"].split(",") if name]
    if os.environ.get("VAL_LOGS"):
        cfg.val_logs = [name for name in os.environ["VAL_LOGS"].split(",") if name]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


async def generate(cfg, jsonl_path, cache, split, cam_type, prompt_type, traj_dir, batch_size):
    global _cache_dir, _traj_dir
    _cache_dir = cache
    _traj_dir = str(traj_dir) if traj_dir else None
    check_heuristics()
    seed_all(int(getattr(cfg, "seed", 0) or 0))
    loader = open_loader(cfg, split, cam_type)
    tokens = list(loader.tokens)
    if not tokens:
        raise SystemExit(f"no scenes for split {split!r}")
    logger.info("%d scenes, writing %s", len(tokens), jsonl_path)
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(items, **_kwargs):
            return items
    written = 0
    loaded = False
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for start in tqdm(range(0, len(tokens), batch_size), desc="cognitive-vqa"):
            samples = []
            for offset, token in enumerate(tokens[start:start + batch_size]):
                try:
                    sample = load_sample(loader, token)
                except Exception:
                    if not loaded and start == 0 and offset == 0:
                        raise
                    logger.warning("skip %s", token, exc_info=True)
                    continue
                loaded = True
                if sample is not None:
                    samples.append((start + offset, token, sample))
            results = await asyncio.gather(*[
                process_data_and_create_qa_pair(idx, *sample, prompt_type=prompt_type, cam_type=cam_type)
                for idx, _token, sample in samples
            ], return_exceptions=True)
            failed = None
            for (_idx, token, _sample), qa_pair in zip(samples, results):
                if isinstance(qa_pair, (FileNotFoundError, ValueError)):
                    logger.warning("skip %s: %s", token, qa_pair)
                    continue
                if isinstance(qa_pair, Exception):
                    failed = qa_pair
                    logger.error("scene %s failed: %s", token, qa_pair)
                    continue
                if not qa_pair:
                    continue
                json.dump(qa_pair, handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                written += 1
            if failed is not None:
                raise failed
    logger.info("wrote %d records to %s (cache %s)", written, jsonl_path, cache)
    return written


def parse_cli(argv):
    config_path = os.environ.get("CONFIG_PATH")
    config_name = os.environ.get("CONFIG_NAME", "default_training")
    ours, overrides = {}, []
    flags = {
        "--config-path": "config_path", "--config-name": "config_name", "--out": "out", "--split": "split",
        "--cam-type": "cam_type", "--prompt-type": "prompt_type", "--traj-results": "traj_results",
        "--batch-size": "batch_size",
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        name, value, consumed = arg, None, 1
        if arg in flags:
            if i + 1 >= len(argv):
                raise SystemExit(f"missing value for {arg}")
            name, value, consumed = arg, argv[i + 1], 2
        elif arg.startswith("--") and "=" in arg:
            name, value = arg.split("=", 1)
        if name in flags and value is not None:
            key = flags[name]
            if key == "config_path":
                config_path = value
            elif key == "config_name":
                config_name = value
            else:
                ours[key] = value
            i += consumed
            continue
        if "=" in arg and not arg.startswith("-"):
            key, raw = arg.split("=", 1)
            bare = key.lstrip("+")
            if bare in OURS:
                ours[bare] = raw
            else:
                overrides.append(arg)
        else:
            overrides.append(arg)
        i += 1
    ours.setdefault("out", os.environ.get("OUT"))
    ours.setdefault("split", os.environ.get("SPLIT", "train"))
    ours.setdefault("cam_type", os.environ.get("CAM_TYPE", "single"))
    ours.setdefault("prompt_type", os.environ.get("PROMPT_TYPE", "base"))
    if "traj_results" not in ours and os.environ.get("TRAJ_RESULTS"):
        ours["traj_results"] = os.environ["TRAJ_RESULTS"]
    ours.setdefault("batch_size", os.environ.get("BATCH_SIZE", "256"))
    return config_path, config_name, ours, overrides


def load_cfg(config_path, config_name, overrides):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path(config_path).resolve()), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path, config_name, ours, overrides = parse_cli(argv)
    if not config_path or not ours.get("out"):
        raise SystemExit(
            "usage: python -m recogdrive.data.cognitive --config-path <training config dir> "
            "--config-name default_training train_test_split=<split> out=<vqa.jsonl or dir> "
            "[split=train|val|all] [cam_type=single|multi_3|multi_5|cont] "
            "[prompt_type=base|vel_and_acc] [traj_results=<dir>] [batch_size=N]"
        )
    cam_type, prompt_type = ours["cam_type"], ours["prompt_type"]
    if cam_type not in CAM_TYPES:
        raise SystemExit(f"cam_type must be one of {CAM_TYPES}, got {cam_type!r}")
    if prompt_type not in PROMPT_TYPES:
        raise SystemExit(f"prompt_type must be one of {PROMPT_TYPES}, got {prompt_type!r}")
    if int(ours["batch_size"]) < 1:
        raise SystemExit("batch_size must be >= 1")
    jsonl_path, cache = resolve_output(ours["out"])
    if not any(arg.split("=", 1)[0].lstrip("+") == "experiment_name" for arg in overrides if "=" in arg):
        overrides = ["experiment_name=cognitive_vqa", *overrides]
    cfg = load_cfg(config_path, config_name, overrides)
    apply_log_env(cfg)
    asyncio.run(generate(
        cfg, jsonl_path, cache, ours["split"], cam_type, prompt_type, ours.get("traj_results"),
        int(ours["batch_size"]),
    ))


if __name__ == "__main__":
    main()
