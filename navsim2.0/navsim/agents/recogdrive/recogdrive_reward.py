"""GRPO reward on NAVSIM: PDM score of each sampled trajectory against its scene's metric cache.

Loading a metric cache (unpickling its map geometry, ~0.1-0.2 s, 16 MiB resident) and scoring (~30 ms per
trajectory) are Python-bound. With workers > 0 they run in spawned processes: each scene of a batch gets its share of
the workers, prefetch() has them load it while the policy runs on the GPU, and its trajectories are split among them.
"""

import dataclasses
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_pred_scores
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def scorer_config() -> PDMScorerConfig:
    # NAVSIM 2.0 renamed comfortable_weight to history_comfort_weight.
    names = {f.name for f in dataclasses.fields(PDMScorerConfig)}
    comfort = "comfortable_weight" if "comfortable_weight" in names else "history_comfort_weight"
    return PDMScorerConfig(progress_weight=10.0, ttc_weight=5.0, **{comfort: 2.0})


class PDMScenes:
    """Metric caches (the loader keeps the recent ones) and a PDM simulator / scorer, for one process."""

    def __init__(self, metric_cache_path: str):
        self.metric_cache = MetricCacheLoader(Path(metric_cache_path))
        self.sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(self.sampling)
        self.scorer = PDMScorer(self.sampling, scorer_config())

    def score(self, token: str, poses: np.ndarray) -> List[float]:
        """Each trajectory is scored against the PDM reference alone, so any split of them scores the same."""
        return pdm_pred_scores(self.metric_cache.get_from_token(token), [Trajectory(p) for p in poses],
                               self.sampling, self.simulator, self.scorer)


worker_scenes: Optional[PDMScenes] = None


def start_worker(metric_cache_path: str) -> None:
    global worker_scenes
    worker_scenes = PDMScenes(metric_cache_path)
    worker_scenes.metric_cache._lru_max = 16  # a scene recurs once an epoch: keep about a batch, not 256 x 16 MiB


def load_in_worker(token: str) -> None:
    worker_scenes.metric_cache.get_from_token(token)


def score_in_worker(token: str, poses: np.ndarray) -> List[float]:
    return worker_scenes.score(token, poses)


class PDMReward:
    def __init__(self, metric_cache_path: str, workers: int = 0):
        self.workers = workers
        self.scenes = None if workers else PDMScenes(metric_cache_path)
        # spawn: the training process holds CUDA and NCCL threads a fork would copy mid-flight. Started now, the
        # workers' imports overlap model loading instead of delaying the first rollout.
        self.pools = [ProcessPoolExecutor(1, mp_context=get_context("spawn"), initializer=start_worker,
                                          initargs=(metric_cache_path,)) for _ in range(workers)]
        for pool in self.pools:
            pool.submit(int)

    def shares(self, tokens: List[str], per_scene: int) -> Dict[str, List[ProcessPoolExecutor]]:
        """The workers of each scene, at most one per trajectory; the same for a batch's prefetch() and scoring."""
        scenes = list(dict.fromkeys(tokens))
        share = max(1, min(per_scene, self.workers // max(1, len(scenes))))
        return {token: [self.pools[(i * share + j) % self.workers] for j in range(share)]
                for i, token in enumerate(scenes)}

    def prefetch(self, tokens: List[str], per_scene: int) -> None:
        """Starts loading these scenes' metric caches in the workers that will score their per_scene trajectories
        (a failed load resurfaces when scored)."""
        if self.workers:
            for token, pools in self.shares(tokens, per_scene).items():
                for pool in pools:
                    pool.submit(load_in_worker, token)

    def __call__(self, trajectories: torch.Tensor, tokens: List[str]) -> torch.Tensor:
        poses = trajectories.detach().float().cpu().numpy()  # numpy has no bf16
        groups: Dict[str, List[int]] = {}
        for i, token in enumerate(tokens):
            groups.setdefault(token, []).append(i)
        rewards = [0.0] * len(tokens)
        if self.workers:
            shares = self.shares(tokens, max(map(len, groups.values())))
            jobs = [(part, pool.submit(score_in_worker, token, poses[part]))
                    for token, idxs in groups.items()
                    for part, pool in zip(np.array_split(np.array(idxs), len(shares[token])), shares[token]) if len(part)]
            scored = [(part, job.result()) for part, job in jobs]
        else:
            scored = [(idxs, self.scenes.score(token, poses[idxs])) for token, idxs in groups.items()]
        for idxs, scores in scored:
            for i, score in zip(idxs, scores):
                rewards[i] = score
        return torch.tensor(rewards, device=trajectories.device, dtype=trajectories.dtype)
