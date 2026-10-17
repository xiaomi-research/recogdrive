"""Per-frame WOD-E2E Rater Feedback Score rewards for GRPO."""

import re
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import torch

from recogdrive.eval.openloop import RFS_SECONDS, pad_raters, rater_feedback_score, to_4hz


class WaymoRFSReward:
    """Score each candidate against its training frame's cached WOD-E2E rater labels.

    ``tokens`` has one entry per candidate trajectory, as in the planner reward contract. GRPO repeats each
    training token once per sampled candidate. Labels are read only from the configured training split.
    """

    _FORBIDDEN_SPLITS = {"val", "valid", "validation", "test", "testing", "eval", "evaluation"}

    def __init__(self, cache_path: str, split: str = "training"):
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.split = str(split)
        if not self.split or self.split in (".", "..") or Path(self.split).name != self.split:
            raise ValueError(f"Waymo RFS reward split must be one cache directory name, got {self.split!r}")
        if self.split.lower() in self._FORBIDDEN_SPLITS:
            raise ValueError(f"Waymo RFS rewards cannot read validation/test split {self.split!r}; use a training split")
        self.split_path = (self.cache_path / self.split).resolve()
        try:
            self.split_path.relative_to(self.cache_path)
        except ValueError as exc:
            raise ValueError(f"Waymo RFS reward split escapes cache root: {self.split!r}") from exc
        if not self.split_path.is_dir():
            raise FileNotFoundError(f"Waymo RFS training cache split does not exist: {self.split_path}")
        if not any(path.is_dir() and self._nonzero_file(path / "eval.gz") for path in self.split_path.iterdir()):
            raise ValueError(f"no rater-labelled WOD-E2E training frames in {self.split_path}")

    @staticmethod
    def _nonzero_file(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _labels(self, token: str):
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", token) or token in (".", ".."):
            raise ValueError(f"invalid WOD-E2E training frame token: {token!r}")
        return self._load_labels(token)

    @lru_cache(maxsize=128)
    def _load_labels(self, token: str):
        frame_dir = (self.split_path / token).resolve()
        try:
            frame_dir.relative_to(self.split_path)
        except ValueError as exc:
            raise ValueError(f"WOD-E2E token escapes training cache: {token!r}") from exc
        path = frame_dir / "eval.gz"
        if not self._nonzero_file(path):
            raise FileNotFoundError(f"missing rater labels for WOD-E2E training frame {token!r}: {path}")

        # This is the repository's local cache format, produced from the configured WOD-E2E training files.
        from navsim.planning.training.dataset import load_feature_target_from_pickle

        info = load_feature_target_from_pickle(path)
        try:
            trajectories_raw = info["rater_trajectories"]
            trajectories = [np.asarray(t, dtype=np.float64) for t in trajectories_raw]
            scores = np.asarray(info["rater_scores"], dtype=np.float64)
            speed = float(info["init_speed"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"malformed WOD-E2E rater labels for training frame {token!r}") from exc

        if not trajectories or scores.shape != (len(trajectories),):
            raise ValueError(f"WOD-E2E rater label count/score shape mismatch for training frame {token!r}")
        for index, trajectory in enumerate(trajectories):
            if trajectory.ndim != 2 or trajectory.shape[1] != 2 or trajectory.shape[0] == 0:
                raise ValueError(f"WOD-E2E rater trajectory {index} has invalid shape for {token!r}: "
                                 f"{trajectory.shape}; expected (T, 2), T > 0")
            if not np.isfinite(trajectory).all():
                raise ValueError(f"WOD-E2E rater trajectory {index} contains non-finite values for {token!r}")
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 10):
            raise ValueError(f"WOD-E2E rater scores must be finite values in [0, 10] for {token!r}")
        if not np.isfinite(speed) or speed < 0:
            raise ValueError(f"WOD-E2E initial speed must be finite and non-negative for {token!r}")
        return trajectories, scores, speed

    def prefetch(self, tokens: List[str], per_scene: int) -> None:
        """Read and validate labels before the expensive policy rollout; the cache is process-local."""
        for token in tokens:
            self._labels(token)

    def __call__(self, trajectories: torch.Tensor, tokens: List[str]) -> torch.Tensor:
        if not torch.is_tensor(trajectories) or trajectories.ndim != 3 or trajectories.shape[-1] != 3:
            shape = tuple(trajectories.shape) if torch.is_tensor(trajectories) else type(trajectories).__name__
            raise ValueError(f"Waymo RFS candidates must have shape (N, T, 3), got {shape}")
        if not torch.is_floating_point(trajectories):
            raise ValueError("Waymo RFS candidates must be floating-point trajectories")
        required_poses = max(RFS_SECONDS) * 2
        if trajectories.shape[1] < required_poses:
            raise ValueError(f"Waymo RFS requires a {max(RFS_SECONDS)} s horizon at 0.5 s spacing "
                             f"({required_poses} poses), got {trajectories.shape[1]}")
        if len(tokens) != trajectories.shape[0]:
            raise ValueError(f"Waymo RFS token count {len(tokens)} does not match candidate count "
                             f"{trajectories.shape[0]}")

        poses = trajectories.detach().float().cpu().numpy()
        if not np.isfinite(poses).all():
            raise ValueError("Waymo RFS candidates must be finite floating-point trajectories")
        rewards = []
        horizon_s = max(RFS_SECONDS)
        points = horizon_s * 4
        for candidate, token in zip(poses, tokens):
            rater_trajectories, scores, init_speed = self._labels(token)
            pred = to_4hz(candidate, horizon_s)
            raters, scores = pad_raters(rater_trajectories, scores, points)
            rewards.append(rater_feedback_score(pred, raters, scores, init_speed))
        rewards = np.asarray(rewards, dtype=np.float32)
        if not np.isfinite(rewards).all():
            raise ValueError("Waymo RFS produced non-finite candidate rewards")
        return torch.as_tensor(rewards, device=trajectories.device, dtype=trajectories.dtype)


class WaymoADEReward:
    """Negative mean XY waypoint error against logged training futures, for public WOD-E2E training frames."""

    _FORBIDDEN_SPLITS = WaymoRFSReward._FORBIDDEN_SPLITS
    _POSES = 10  # 5 s at the cache's 0.5 s waypoint interval.

    def __init__(self, cache_path: str, split: str = "training"):
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.split = str(split)
        if not self.split or self.split in (".", "..") or Path(self.split).name != self.split:
            raise ValueError(f"Waymo ADE reward split must be one cache directory name, got {self.split!r}")
        if self.split.lower() in self._FORBIDDEN_SPLITS:
            raise ValueError(f"Waymo ADE rewards cannot read validation/test split {self.split!r}; use a training split")
        self.split_path = (self.cache_path / self.split).resolve()
        try:
            self.split_path.relative_to(self.cache_path)
        except ValueError as exc:
            raise ValueError(f"Waymo ADE reward split escapes cache root: {self.split!r}") from exc
        if not self.split_path.is_dir():
            raise FileNotFoundError(f"Waymo ADE training cache split does not exist: {self.split_path}")
        if not any(path.is_dir() and WaymoRFSReward._nonzero_file(path / "features.gz")
                   and WaymoRFSReward._nonzero_file(path / "targets.gz") for path in self.split_path.iterdir()):
            raise ValueError(f"no WOD-E2E training samples with features and logged futures in {self.split_path}")

    def _target(self, token: str):
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", token) or token in (".", ".."):
            raise ValueError(f"invalid WOD-E2E training frame token: {token!r}")
        return self._load_target(token)

    @lru_cache(maxsize=128)
    def _load_target(self, token: str):
        frame_dir = (self.split_path / token).resolve()
        try:
            frame_dir.relative_to(self.split_path)
        except ValueError as exc:
            raise ValueError(f"WOD-E2E token escapes training cache: {token!r}") from exc
        path = frame_dir / "targets.gz"
        if not WaymoRFSReward._nonzero_file(path):
            raise FileNotFoundError(f"missing logged waypoints for WOD-E2E training frame {token!r}: {path}")

        # Read the converter's local training cache only; it contains the original waypoint targets.
        from navsim.planning.training.dataset import load_feature_target_from_pickle

        info = load_feature_target_from_pickle(path)
        try:
            target = info["trajectory"]
            if torch.is_tensor(target):
                target = target.detach().cpu().numpy()
            target = np.asarray(target, dtype=np.float32)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"malformed logged waypoints for WOD-E2E training frame {token!r}") from exc
        if target.ndim != 2 or target.shape[0] < self._POSES or target.shape[1] < 2:
            raise ValueError(f"WOD-E2E training target has invalid shape for {token!r}: {target.shape}; "
                             f"expected at least ({self._POSES}, 2)")
        if not np.isfinite(target).all():
            raise ValueError(f"WOD-E2E training target contains non-finite values for {token!r}")
        return target

    def prefetch(self, tokens: List[str], per_scene: int) -> None:
        """Read and validate logged waypoints before the policy rollout; the cache is process-local."""
        for token in tokens:
            self._target(token)

    def __call__(self, trajectories: torch.Tensor, tokens: List[str]) -> torch.Tensor:
        if not torch.is_tensor(trajectories) or trajectories.ndim != 3 or trajectories.shape[-1] != 3:
            shape = tuple(trajectories.shape) if torch.is_tensor(trajectories) else type(trajectories).__name__
            raise ValueError(f"Waymo ADE candidates must have shape (N, T, 3), got {shape}")
        if not torch.is_floating_point(trajectories):
            raise ValueError("Waymo ADE candidates must be floating-point trajectories")
        if trajectories.shape[1] < self._POSES:
            raise ValueError(f"Waymo ADE requires a 5 s horizon at 0.5 s spacing ({self._POSES} poses), "
                             f"got {trajectories.shape[1]}")
        if len(tokens) != trajectories.shape[0]:
            raise ValueError(f"Waymo ADE token count {len(tokens)} does not match candidate count "
                             f"{trajectories.shape[0]}")

        poses = trajectories.detach().float().cpu().numpy()
        if not np.isfinite(poses).all():
            raise ValueError("Waymo ADE candidates must be finite floating-point trajectories")
        rewards = []
        for candidate, token in zip(poses, tokens):
            target = self._target(token)
            errors = np.linalg.norm(candidate[:self._POSES, :2] - target[:self._POSES, :2], axis=-1)
            rewards.append(-float(errors.mean()))
        rewards = np.asarray(rewards, dtype=np.float32)
        if not np.isfinite(rewards).all():
            raise ValueError("Waymo ADE produced non-finite candidate rewards")
        return torch.as_tensor(rewards, device=trajectories.device, dtype=trajectories.dtype)
