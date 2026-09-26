"""nuScenes keyframes as ReCogDrive samples.

Keyframes are 2 Hz, the same spacing as NAVSIM, so one sample is the six camera images, 3 past keyframes plus the
current one as ego history, and the next horizon(model) keyframes as the target, all in the current ego frame
(x forward, y left) of the LIDAR_TOP keyframe, whose timestamp the annotations share. Velocity and acceleration are
finite differences of keyframe poses; the mini split ships without CAN bus data.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from recogdrive.data.registry import horizon, register

HISTORY = 3
POSE_SENSOR = "LIDAR_TOP"
CAMERAS = {"CAM_FRONT": "front", "CAM_FRONT_LEFT": "front_left", "CAM_FRONT_RIGHT": "front_right",
           "CAM_BACK_LEFT": "back_left", "CAM_BACK_RIGHT": "back_right", "CAM_BACK": "back"}
BOX_CATEGORIES = ("vehicle.", "human.pedestrian.")  # objects the ST-P3 / UniAD collision rate counts
TURN_THRESHOLD_M = 2.0  # lateral offset of the last future pose, as in UniAD / VAD planning commands
# nuscenes.utils.splits.mini_train / mini_val
MINI_SPLITS = {
    "train": ["scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-1077", "scene-1094", "scene-1100"],
    "val": ["scene-0103", "scene-0916"],
}


def quaternion_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_local(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Global (x, y, yaw) poses expressed in the frame of `origin`."""
    c, s = math.cos(origin[2]), math.sin(origin[2])
    dx, dy = poses[:, 0] - origin[0], poses[:, 1] - origin[1]
    heading = (poses[:, 2] - origin[2] + math.pi) % (2 * math.pi) - math.pi
    return np.stack([c * dx + s * dy, -s * dx + c * dy, heading], axis=1)


def rotate_to(vec: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1]])


def command_one_hot(future_local: np.ndarray) -> np.ndarray:
    """NAVSIM order: left, straight, right, unknown."""
    lateral = future_local[-1, 1]
    index = 0 if lateral > TURN_THRESHOLD_M else 2 if lateral < -TURN_THRESHOLD_M else 1
    out = np.zeros(4, dtype=np.float32)
    out[index] = 1.0
    return out


def scene_splits(version: str):
    if version == "v1.0-mini":
        return MINI_SPLITS
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:
        raise ImportError(f"nuScenes {version} splits need nuscenes-devkit (pip install nuscenes-devkit)") from exc
    return create_splits_scenes()


class NuScenesDataset(Dataset):
    def __init__(self, root: str, version: str, split: str, future: int, target_builders=None):
        self.root = Path(root)
        self.tables = self.root / version
        self.future = future
        self.target_builders = target_builders
        scenes = {s["name"]: s for s in self.load("scene")}
        samples = {s["token"]: s for s in self.load("sample")}
        poses = {p["token"]: p for p in self.load("ego_pose")}
        keyframes = {}  # (sample token, channel) -> sample_data
        for sd in self.load("sample_data"):
            channel = sd["filename"].split("/")[1] if sd["is_key_frame"] else None
            if channel == POSE_SENSOR or channel in CAMERAS:
                keyframes[sd["sample_token"], channel] = sd
        self.tracks = []   # per scene: tokens, camera paths, global poses (N, 3), timestamps (N,) in seconds
        self.items = []    # (track, index of the current keyframe)
        for name in scene_splits(version)[split]:
            if name not in scenes:
                continue
            tokens, token = [], scenes[name]["first_sample_token"]
            while token:
                tokens.append(token)
                token = samples[token]["next"]
            lidar = [keyframes[t, POSE_SENSOR] for t in tokens]
            pose = np.array(
                [[*poses[d["ego_pose_token"]]["translation"][:2], quaternion_yaw(poses[d["ego_pose_token"]]["rotation"])]
                 for d in lidar],
                dtype=np.float64,
            )
            stamps = np.array([d["timestamp"] for d in lidar], dtype=np.float64) * 1e-6
            cameras = [[(view, str(self.root / keyframes[t, channel]["filename"]))
                        for channel, view in CAMERAS.items() if (t, channel) in keyframes] for t in tokens]
            track = len(self.tracks)
            self.tracks.append((tokens, cameras, pose, stamps))
            self.items += [(track, i) for i in range(HISTORY, len(tokens) - future)]
        if not self.items:
            raise ValueError(f"no nuScenes {version}/{split} samples with {future} future keyframes under {self.root}")
        self.index = {self.tracks[track][0][i]: n for n, (track, i) in enumerate(self.items)}
        self.boxes = None

    def load(self, name: str):
        return json.loads((self.tables / f"{name}.json").read_text())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from navsim.planning.training.dataset import transform_targets_after_load

        track, i = self.items[idx]
        tokens, cameras, pose, stamps = self.tracks[track]
        origin = pose[i]
        history = to_local(pose[i - HISTORY : i + 1], origin)
        future = to_local(pose[i + 1 : i + 1 + self.future], origin)
        v_now = (pose[i, :2] - pose[i - 1, :2]) / (stamps[i] - stamps[i - 1])
        v_prev = (pose[i - 1, :2] - pose[i - 2, :2]) / (stamps[i - 1] - stamps[i - 2])
        accel = (v_now - v_prev) / (0.5 * (stamps[i] - stamps[i - 2]))
        command = command_one_hot(future)
        status = np.concatenate([command, rotate_to(v_now, origin[2]), rotate_to(accel, origin[2])])
        features = {
            "history_trajectory": torch.tensor(history, dtype=torch.float32),
            "high_command_one_hot": torch.tensor(command),
            "status_feature": torch.tensor(status, dtype=torch.float32),
            "camera_paths": cameras[i],
        }
        targets = transform_targets_after_load(
            {"trajectory": torch.tensor(future, dtype=torch.float32)}, self.target_builders
        )
        return features, targets, tokens[i]

    def eval_info(self, token: str) -> dict:
        """Ground-truth future poses and, per future keyframe, the vehicle / pedestrian boxes (x, y, length, width,
        yaw) in the current ego frame. The annotation tables load on the first call."""
        if self.boxes is None:
            categories = {c["token"]: c["name"] for c in self.load("category")}
            kept = {i["token"] for i in self.load("instance") if categories[i["category_token"]].startswith(BOX_CATEGORIES)}
            self.boxes = {}
            for a in self.load("sample_annotation"):
                if a["instance_token"] in kept:
                    width, length = a["size"][:2]
                    self.boxes.setdefault(a["sample_token"], []).append(
                        (*a["translation"][:2], quaternion_yaw(a["rotation"]), length, width))
        track, i = self.items[self.index[token]]
        tokens, _, pose, _ = self.tracks[track]
        steps = []
        for t in tokens[i + 1 : i + 1 + self.future]:
            boxes = np.array(self.boxes.get(t, []), dtype=np.float64).reshape(-1, 5)
            local = to_local(boxes[:, :3], pose[i])
            steps.append(np.column_stack([local[:, :2], boxes[:, 3:5], local[:, 2]]))
        return {"trajectory": to_local(pose[i + 1 : i + 1 + self.future], pose[i]), "boxes": steps}


@register("nuscenes")
def nuscenes_loader(cfg, agent, split: str):
    if getattr(agent, "cache_hidden_state", False):
        raise ValueError("nuscenes loader feeds images; set agent.cache_hidden_state=False")
    root = OmegaConf.select(cfg, "nuscenes.root")
    if not root:
        raise ValueError("set nuscenes.root to the directory that holds samples/ and v1.0-*/")
    return NuScenesDataset(
        root,
        OmegaConf.select(cfg, "nuscenes.version") or "v1.0-mini",
        "train" if split == "train" else "val",
        horizon(agent),
        agent.get_target_builders(),
    )
