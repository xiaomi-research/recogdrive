"""Closed-loop driving: a trained policy behind a simulator's online inference protocol.

A simulator streams ego poses and camera frames and asks for a plan. `Driver` turns one episode's stream into the
sample contract of recogdrive.data.registry (ego history at -1.5 .. 0 s, ego-frame velocity and acceleration, the
navigation command, camera images held in memory) and returns the policy's `pred_traj`, so any policy adapter drives
every simulator. Protocol modules translate the simulators' messages:

    alpasim    AlpaSim (NVIDIA) egodriver gRPC service
    hugsim     HUGSIM agent process (pickles over FIFOs)
    neuroncap  NeuroNCAP HTTP inference server (/alive, /reset, /infer)

`run_recogdrive_infer.py infer.serve=<protocol> infer.port=<port> infer.checkpoint=...` with the training config's
agent starts one (scripts/infer/run_recogdrive_closedloop.sh).
"""

import importlib
import logging
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from recogdrive.data.nuscenes import to_local
from recogdrive.data.prefetch import to_device
from recogdrive.data.registry import POSE_INTERVAL_S, collate

logger = logging.getLogger(__name__)

HISTORY = 3
COMMANDS = ("left", "straight", "right", "unknown")  # high_command_one_hot order
UNIAD_COMMANDS = {0: 2, 1: 0, 2: 1}  # UniAD's 0 right / 1 left / 2 straight (NeuroNCAP, HUGSIM) -> COMMANDS


def to_world(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Ego-frame (x, y, heading) poses of the frame `origin` (global x, y, yaw) in global coordinates."""
    c, s = math.cos(origin[2]), math.sin(origin[2])
    x, y = poses[:, 0], poses[:, 1]
    return np.stack([origin[0] + c * x - s * y, origin[1] + s * x + c * y, poses[:, 2] + origin[2]], axis=1)


class Driver:
    """One policy, one episode at a time: `observe` ego poses as they come (any rate, global frame, seconds), then
    `plan` at the current time. Before the first observed pose the history continues it at constant velocity."""

    def __init__(self, agent, device):
        self.agent, self.device = agent, torch.device(device)
        make = getattr(agent, "worker_transform", None)
        self.transform = make() if make is not None else None
        self.reset()

    def reset(self) -> None:
        self.track = []  # (t, x, y, yaw)

    def observe(self, t: float, x: float, y: float, yaw: float) -> None:
        while self.track and self.track[-1][0] >= t:  # a resent or rewound pose replaces the later ones
            self.track.pop()
        self.track.append((float(t), float(x), float(y), float(yaw)))

    def history(self, now: float, speed: Optional[float] = None) -> np.ndarray:
        """Global poses at now - 1.5, -1.0, -0.5, 0 s. Without two observed poses the ego moves at `speed` (or stands)
        along its heading."""
        if not self.track:
            raise RuntimeError("plan() before any observed ego pose")
        t, x, y, yaw = np.array(self.track, dtype=np.float64).T
        yaw = np.unwrap(yaw)
        query = now - POSE_INTERVAL_S * np.arange(HISTORY, -1, -1)
        if len(t) > 1:
            vx, vy = (x[1] - x[0]) / (t[1] - t[0]), (y[1] - y[0]) / (t[1] - t[0])
        else:
            vx, vy = (speed or 0.0) * math.cos(yaw[0]), (speed or 0.0) * math.sin(yaw[0])
        before = np.minimum(query - t[0], 0.0)
        return np.stack([np.interp(query, t, x) + vx * before, np.interp(query, t, y) + vy * before,
                         np.interp(query, t, yaw)], axis=1)

    def features(self, now: float, images: Sequence[Tuple[str, object]], command: int,
                 velocity: Optional[Iterable[float]] = None, acceleration: Optional[Iterable[float]] = None) -> dict:
        """Sample-contract features. images: (view, PIL image or path), front first; command: index into COMMANDS;
        velocity / acceleration: ego frame (forward, left), finite differences of the history when None."""
        velocity = None if velocity is None else np.asarray(velocity, dtype=np.float64)[:2]
        acceleration = None if acceleration is None else np.asarray(acceleration, dtype=np.float64)[:2]
        world = self.history(now, None if velocity is None else float(np.hypot(*velocity)))
        local = to_local(world, world[-1])
        v_now = (local[3, :2] - local[2, :2]) / POSE_INTERVAL_S
        v_prev = (local[2, :2] - local[1, :2]) / POSE_INTERVAL_S
        velocity = v_now if velocity is None else velocity
        acceleration = (v_now - v_prev) / POSE_INTERVAL_S if acceleration is None else acceleration
        one_hot = np.zeros(len(COMMANDS), dtype=np.float32)
        one_hot[command] = 1.0
        return {
            "history_trajectory": torch.tensor(local, dtype=torch.float32),
            "high_command_one_hot": torch.tensor(one_hot),
            "status_feature": torch.tensor(np.concatenate([one_hot, velocity, acceleration]), dtype=torch.float32),
            "camera_paths": list(images),
        }

    @torch.no_grad()
    def plan(self, now: float, images, command: int, velocity=None, acceleration=None) -> np.ndarray:
        """The policy's (num_poses, 3) trajectory in the current ego frame, poses every 0.5 s from now + 0.5 s."""
        features = self.features(now, images, command, velocity, acceleration)
        if self.transform is not None:
            features = self.transform(features)
        batch, _, _ = collate([(features, {}, "closedloop")])
        return self.agent(to_device(batch, self.device))["pred_traj"][0].float().cpu().numpy()


def serve(cfg, protocol: str) -> None:
    from recogdrive.infer import load_agent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    driver = Driver(load_agent(cfg, device), device)
    port = int(OmegaConf.select(cfg, "infer.port") or 0)
    logger.info("closed-loop %s server on port %s", protocol, port or "default")
    importlib.import_module(f"recogdrive.closedloop.{protocol}").serve(driver, port, cfg)
