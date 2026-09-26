"""Open-loop benchmarks scored in the training process: the trainer's model predicts every sample of a registered data
source's validation split on the training GPUs (like validation), and the main rank computes the metrics on CPU from
`pred_traj` and the dataset's `eval_info`, so no second model or GPU job is involved.

nuscenes  L2 and collision rate at 1 / 2 / 3 s under both conventions of the planning literature (GPT-Driver reports
          both): ST-P3 / VAD average over all steps up to t, UniAD takes the step at t. Collisions follow their
          occupancy check: a 200 x 200 grid at 0.5 m around the ego holds the future vehicle / pedestrian boxes, the
          axis-aligned ego box (4.084 x 1.85 m, centre 0.5 m ahead of the reference point) is placed at each
          waypoint, and steps where the ground truth itself collides are not counted. Trajectories are in the ego
          (rear axle) frame rather than the UniAD LiDAR frame.
waymoe2e  WOD-E2E Rater Feedback Score (a port of waymo_open_dataset/metrics/python/rater_feedback_utils.py, Apache
          2.0, averaged over frames rather than scenario clusters) and ADE at 3 / 5 s against the best-rated rater
          trajectory. The model's 0.5 s poses are interpolated to the 4 Hz grid; RFS and ADE@5s need a 5 s horizon.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

from recogdrive.eval.registry import register

logger = logging.getLogger(__name__)

POSE_INTERVAL_S = 0.5
BEV_RANGE_M, BEV_CELL_M = 50.0, 0.5
EGO_LENGTH_M, EGO_WIDTH_M, EGO_OFFSET_M = 4.084, 1.85, 0.5
NUSC_SECONDS = (1, 2, 3)

RFS_SECONDS = (3, 5)
RFS_FREQUENCY_HZ = 4
RFS_BASE_THRESHOLDS = np.array([1.0, 1.8])  # lateral thresholds at 3 / 5 s; longitudinal ones are 4x
RFS_LNG_MULTIPLIER = 4.0
RFS_DECAY = 0.1
RFS_FLOOR = 4.0
RFS_RATERS = 3


# ----------------------------------------------------------------------------- nuScenes

def cell_centers() -> np.ndarray:
    size = int(2 * BEV_RANGE_M / BEV_CELL_M)
    return (np.arange(size) + 0.5) * BEV_CELL_M - BEV_RANGE_M


def cell_span(low: float, high: float) -> slice:
    """Cells whose centres lie in [low, high]."""
    size = int(2 * BEV_RANGE_M / BEV_CELL_M)
    start = max(math.ceil((low + BEV_RANGE_M) / BEV_CELL_M - 0.5), 0)
    stop = min(math.floor((high + BEV_RANGE_M) / BEV_CELL_M - 0.5) + 1, size)
    return slice(start, max(start, stop))


def occupancy(boxes: np.ndarray) -> np.ndarray:
    """(x, y, length, width, yaw) boxes rasterized onto the ego-centred grid (cell centres inside a box)."""
    centers = cell_centers()
    grid = np.zeros((len(centers), len(centers)), dtype=bool)
    for x, y, length, width, yaw in boxes:
        reach = 0.5 * math.hypot(length, width)
        rows, cols = cell_span(x - reach, x + reach), cell_span(y - reach, y + reach)
        if rows.start == rows.stop or cols.start == cols.stop:
            continue
        dx, dy = np.meshgrid(centers[rows] - x, centers[cols] - y, indexing="ij")
        c, s = math.cos(yaw), math.sin(yaw)
        grid[rows, cols] |= (np.abs(c * dx + s * dy) <= 0.5 * length) & (np.abs(-s * dx + c * dy) <= 0.5 * width)
    return grid


def ego_collides(grid: np.ndarray, x: float, y: float) -> bool:
    rows = cell_span(x - 0.5 * EGO_LENGTH_M + EGO_OFFSET_M, x + 0.5 * EGO_LENGTH_M + EGO_OFFSET_M)
    cols = cell_span(y - 0.5 * EGO_WIDTH_M, y + 0.5 * EGO_WIDTH_M)
    return bool(grid[rows, cols].any())


def nuscenes_metrics(samples: List[tuple]) -> Dict[str, float]:
    """samples: (pred (T, 2+), gt (T, 2+), boxes per future step). L2 in m, collision rate in %."""
    steps = 2 * max(NUSC_SECONDS)
    l2, collided = [], []
    for pred, gt, boxes in samples:
        l2.append(np.linalg.norm(pred[:steps, :2] - gt[:steps, :2], axis=1))
        grids = [occupancy(b) for b in boxes[:steps]]
        gt_hit = np.array([ego_collides(g, *gt[t, :2]) for t, g in enumerate(grids)])
        hit = np.array([ego_collides(g, *pred[t, :2]) for t, g in enumerate(grids)])
        collided.append(hit & ~gt_hit)
    l2, collided = np.stack(l2), np.stack(collided).astype(np.float64)
    out = {}
    for second in NUSC_SECONDS:
        k = 2 * second
        out[f"stp3/L2_{second}s"] = float(l2[:, :k].mean(axis=1).mean())
        out[f"stp3/collision_{second}s"] = 100.0 * float(collided[:, :k].mean(axis=1).mean())
        out[f"uniad/L2_{second}s"] = float(l2[:, k - 1].mean())
        out[f"uniad/collision_{second}s"] = 100.0 * float(collided[:, k - 1].mean())
    for protocol in ("stp3", "uniad"):
        for metric in ("L2", "collision"):
            out[f"{protocol}/{metric}_avg"] = float(np.mean([out[f"{protocol}/{metric}_{s}s"] for s in NUSC_SECONDS]))
    out["samples"] = len(samples)
    return out


# ----------------------------------------------------------------------------- WOD-E2E

def to_4hz(poses: np.ndarray, seconds: float) -> np.ndarray:
    """0.5 s poses (from t = 0.5 s) linearly interpolated, with the origin at t = 0, to 4 Hz up to `seconds`."""
    times = POSE_INTERVAL_S * np.arange(len(poses) + 1)
    xy = np.vstack([np.zeros((1, 2)), poses[:, :2]])
    query = np.arange(1, int(seconds * RFS_FREQUENCY_HZ) + 1) / RFS_FREQUENCY_HZ
    return np.stack([np.interp(query, times, xy[:, 0]), np.interp(query, times, xy[:, 1])], axis=1)


def pad_raters(trajectories: List[np.ndarray], scores: np.ndarray, points: int):
    """Truncates or pads (repeating the last one) to RFS_RATERS trajectories of `points` waypoints."""
    trajectories, scores = list(trajectories[:RFS_RATERS]), list(scores[:RFS_RATERS])
    trajectories += [trajectories[-1]] * (RFS_RATERS - len(trajectories))
    scores += [scores[-1]] * (RFS_RATERS - len(scores))
    fixed = [np.vstack([t[:points]] + [t[-1:]] * max(points - len(t), 0)) for t in trajectories]
    return np.stack(fixed), np.asarray(scores, dtype=np.float64)


def rater_feedback_score(pred: np.ndarray, raters: np.ndarray, scores: np.ndarray, init_speed: float) -> float:
    """pred (T, 2) and raters (P, T, 2) at 4 Hz from t = 0.25 s, scores (P,)."""
    lng = np.diff(np.concatenate([np.zeros((len(raters), 1, 2)), raters], axis=1), axis=1)  # (P, T, 2)
    still = np.linalg.norm(lng, axis=-1) == 0
    lng[:, 0] = np.where(still[:, :1], [1.0, 0.0], lng[:, 0])
    for t in range(1, lng.shape[1]):
        lng[:, t] = np.where(still[:, t:t + 1], lng[:, t - 1], lng[:, t])
    lat = np.stack([-lng[..., 1], lng[..., 0]], axis=-1)
    lng = lng / np.linalg.norm(lng, axis=-1, keepdims=True)
    lat = lat / np.linalg.norm(lat, axis=-1, keepdims=True)
    offset = pred[None] - raters
    at = np.array(RFS_SECONDS) * RFS_FREQUENCY_HZ - 1
    lng_dist = np.abs((lng * offset).sum(-1))[:, at]  # (P, 2)
    lat_dist = np.abs((lat * offset).sum(-1))[:, at]
    scale = np.clip(0.5 + 0.5 * (init_speed - 1.4) / (11 - 1.4), 0.5, 1.0)
    normalized = np.maximum(lng_dist / (scale * RFS_BASE_THRESHOLDS * RFS_LNG_MULTIPLIER),
                            lat_dist / (scale * RFS_BASE_THRESHOLDS))
    within = bool(np.any(np.all(normalized <= 1.0, axis=1)))
    score = float(np.max(scores[:, None] * RFS_DECAY ** np.maximum(normalized - 1.0, 0.0), axis=0).mean())
    return score if within else max(RFS_FLOOR, score)


def waymo_metrics(samples: List[tuple], horizon_s: float) -> Dict[str, float]:
    """samples: (pred 0.5 s poses, eval_info with rater_trajectories / rater_scores / init_speed)."""
    long_enough = horizon_s >= max(RFS_SECONDS)
    rfs, ade3, ade5 = [], [], []
    for poses, info in samples:
        points = int(min(horizon_s, max(RFS_SECONDS)) * RFS_FREQUENCY_HZ)
        pred = to_4hz(poses, points / RFS_FREQUENCY_HZ)
        raters, scores = pad_raters(info["rater_trajectories"], info["rater_scores"], max(RFS_SECONDS) * RFS_FREQUENCY_HZ)
        best = raters[int(np.argmax(scores))]
        ade3.append(np.linalg.norm(pred[:12] - best[:12], axis=1).mean())
        if long_enough:
            ade5.append(np.linalg.norm(pred - best, axis=1).mean())
            rfs.append(rater_feedback_score(pred, raters, scores, float(info["init_speed"])))
    out = {"ade_3s": float(np.mean(ade3)), "samples": len(samples)}
    if long_enough:
        out.update(rfs=float(np.mean(rfs)), ade_5s=float(np.mean(ade5)))
    return out


# ----------------------------------------------------------------------------- evaluators

class OpenLoopEvaluator:
    """`evaluator.<name>.data_overrides` are config overrides for the data source (e.g. {nuscenes: {root: ...}}). A
    dataset with `eval_indices()` is predicted on those samples only (WOD-E2E: the rater-labelled frames)."""

    in_process = True
    source = ""

    def __init__(self, cfg, section, output_dir: Path):
        self.cfg = cfg
        self.section = section
        self.root = output_dir / "eval"
        self.every_n_epochs = int(OmegaConf.select(section, "every_n_epochs") or 0)
        self.on_end = bool(OmegaConf.select(section, "on_end", default=True))
        self.dataset = self.loader = None
        self.results = []

    def eval_loader(self, trainer):
        if self.loader is None:
            from torch.utils.data import Subset

            from recogdrive.data import LOADERS, make_loader
            from recogdrive.data.registry import with_worker_transform

            source_cfg = OmegaConf.merge(self.cfg, OmegaConf.select(self.section, "data_overrides") or {})
            self.dataset = LOADERS[self.source](source_cfg, trainer.agent, "val")
            scored = Subset(self.dataset, self.dataset.eval_indices()) if hasattr(self.dataset, "eval_indices") else self.dataset
            self.loader = make_loader(with_worker_transform(scored, trainer.agent), trainer.cfg.dataloader.params, False,
                                      trainer.ctx.rank, trainer.ctx.world_size)
        return self.loader

    def during_training(self, trainer, ckpt: Path, tag: str) -> None:
        """Called on every rank with the model holding the weights to score."""
        from recogdrive.adapters import predict

        predictions = predict(trainer.model, self.eval_loader(trainer), trainer.ctx)
        if not trainer.ctx.is_main:
            return
        sampling = trainer.agent.trajectory_sampling
        metrics = self.score(self.dataset, predictions, float(sampling.num_poses * sampling.interval_length))
        out_dir = self.root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{self.source}_metrics.json").write_text(json.dumps(metrics, indent=2))
        self.results.append((tag, metrics))

    def score(self, dataset, predictions: Dict[str, np.ndarray], horizon_s: float) -> Dict[str, float]:
        raise NotImplementedError

    def poll(self):
        results, self.results = self.results, []
        return results

    def release(self) -> None:
        self.dataset = self.loader = None

    def finish(self, final_ckpt: Optional[Path]):
        return self.poll()


@register("nuscenes")
class NuScenesEvaluator(OpenLoopEvaluator):
    source = "nuscenes"

    def score(self, dataset, predictions, horizon_s):
        if horizon_s < max(NUSC_SECONDS):
            raise ValueError(f"nuScenes planning metrics need a {max(NUSC_SECONDS)} s horizon, the model predicts {horizon_s} s")
        samples = []
        for token, poses in predictions.items():
            info = dataset.eval_info(token)
            samples.append((poses, info["trajectory"], info["boxes"]))
        return nuscenes_metrics(samples)


@register("waymoe2e")
class WaymoE2EEvaluator(OpenLoopEvaluator):
    source = "waymoe2e"

    def score(self, dataset, predictions, horizon_s):
        if horizon_s < max(RFS_SECONDS):
            logger.warning("WOD-E2E: the model predicts %.1f s, RFS and ADE@5s need %d s (agent.trajectory_sampling)",
                           horizon_s, max(RFS_SECONDS))
        samples = [(poses, info) for token, poses in predictions.items() if (info := dataset.eval_info(token))]
        if not samples:
            raise ValueError("no rater-labelled frames in the WOD-E2E split; convert the val TFRecords with this version")
        return waymo_metrics(samples, horizon_s)
