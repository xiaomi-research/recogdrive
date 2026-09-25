"""nuScenes keyframes as ReCogDrive samples.

Keyframes are 2 Hz, the same spacing as NAVSIM, so one sample is the front camera image,
3 past keyframes plus the current one as ego history, and the next 8 keyframes (4 s) as the
target, all in the current ego frame (x forward, y left). Velocity and acceleration are finite
differences of keyframe poses; the mini split ships without CAN bus data.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from recogdrive.data.registry import register

HISTORY = 3
FUTURE = 8
CAMERA = "CAM_FRONT"
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
    def __init__(self, root: str, version: str, split: str, target_builders=None):
        self.root = Path(root)
        self.target_builders = target_builders
        tables = self.root / version
        load = lambda name: json.loads((tables / f"{name}.json").read_text())
        scenes = {s["name"]: s for s in load("scene")}
        samples = {s["token"]: s for s in load("sample")}
        poses = {p["token"]: p for p in load("ego_pose")}
        prefix = f"samples/{CAMERA}/"
        front = {
            sd["sample_token"]: sd
            for sd in load("sample_data")
            if sd["is_key_frame"] and sd["filename"].startswith(prefix)
        }
        wanted = scene_splits(version)[split]
        self.tracks = []   # per scene: tokens, image paths, global poses (N, 3), timestamps (N,) in seconds
        self.items = []    # (track, index of the current keyframe)
        for name in wanted:
            if name not in scenes:
                continue
            tokens, token = [], scenes[name]["first_sample_token"]
            while token:
                tokens.append(token)
                token = samples[token]["next"]
            data = [front[t] for t in tokens]
            pose = np.array(
                [[*poses[d["ego_pose_token"]]["translation"][:2], quaternion_yaw(poses[d["ego_pose_token"]]["rotation"])]
                 for d in data],
                dtype=np.float64,
            )
            stamps = np.array([d["timestamp"] for d in data], dtype=np.float64) * 1e-6
            track = len(self.tracks)
            self.tracks.append((tokens, [str(self.root / d["filename"]) for d in data], pose, stamps))
            self.items += [(track, i) for i in range(HISTORY, len(tokens) - FUTURE)]
        if not self.items:
            raise ValueError(f"no nuScenes {version}/{split} samples under {self.root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from navsim.planning.training.dataset import transform_targets_after_load

        track, i = self.items[idx]
        tokens, images, pose, stamps = self.tracks[track]
        origin = pose[i]
        history = to_local(pose[i - HISTORY : i + 1], origin)
        future = to_local(pose[i + 1 : i + 1 + FUTURE], origin)
        v_now = (pose[i, :2] - pose[i - 1, :2]) / (stamps[i] - stamps[i - 1])
        v_prev = (pose[i - 1, :2] - pose[i - 2, :2]) / (stamps[i - 1] - stamps[i - 2])
        accel = (v_now - v_prev) / (0.5 * (stamps[i] - stamps[i - 2]))
        command = command_one_hot(future)
        status = np.concatenate([command, rotate_to(v_now, origin[2]), rotate_to(accel, origin[2])])
        features = {
            "history_trajectory": torch.tensor(history, dtype=torch.float32),
            "high_command_one_hot": torch.tensor(command),
            "status_feature": torch.tensor(status, dtype=torch.float32),
            "image_path_tensor": torch.tensor([ord(c) for c in images[i]], dtype=torch.long),
        }
        targets = transform_targets_after_load(
            {"trajectory": torch.tensor(future, dtype=torch.float32)}, self.target_builders
        )
        return features, targets, tokens[i]


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
        agent.get_target_builders(),
    )
