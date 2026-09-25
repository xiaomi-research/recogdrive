"""GRPO reward on NAVSIM: PDM score of each sampled trajectory against its scene's metric cache."""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

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


class PDMReward:
    def __init__(self, metric_cache_path: str):
        self.metric_cache = MetricCacheLoader(Path(metric_cache_path))
        self.sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.config = scorer_config()
        self.simulator = PDMSimulator(self.sampling)
        self.scorer = PDMScorer(self.sampling, self.config)

    def __call__(self, trajectories: torch.Tensor, tokens: List[str]) -> torch.Tensor:
        poses = trajectories.detach().cpu().numpy()
        groups: Dict[str, List[int]] = {}
        for i, token in enumerate(tokens):
            groups.setdefault(token, []).append(i)
        caches = {token: self.metric_cache.get_from_token(token) for token in groups}
        jobs = [(token, idxs, [Trajectory(poses[i]) for i in idxs]) for token, idxs in groups.items()]

        scores = None
        # ponytail: 4 tokens 上线程比串行 batch 慢；token 多再开
        if len(jobs) >= 8:
            try:
                scores = self.score_threaded(jobs, caches)
            except Exception:
                scores = None
        if scores is None:
            scores = {
                token: pdm_pred_scores(
                    metric_cache=caches[token],
                    model_trajectories=trajs,
                    future_sampling=self.sampling,
                    simulator=self.simulator,
                    scorer=self.scorer,
                )
                for token, _, trajs in jobs
            }

        rewards = [0.0] * len(tokens)
        for token, idxs, _ in jobs:
            for i, score in zip(idxs, scores[token]):
                rewards[i] = score
        return torch.tensor(rewards, device=trajectories.device, dtype=trajectories.dtype)

    def score_threaded(self, jobs, caches):
        def run(job):
            token, _, trajs = job
            # ponytail: sim/scorer mutate; one instance per thread
            simulator, scorer = PDMSimulator(self.sampling), PDMScorer(self.sampling, self.config)
            return token, pdm_pred_scores(caches[token], trajs, self.sampling, simulator, scorer)

        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
            return dict(pool.map(run, jobs))
