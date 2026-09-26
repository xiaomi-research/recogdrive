"""NAVSIM PDMS (1.1) / EPDMS (2.0) through each tree's own scoring entry.

During training the trainer's model generates the trajectories on the training GPUs (like validation), and
the official scoring runs on CPU in the background with a replay agent, so evaluation never loads a second
model onto the GPUs training uses. The final evaluation runs the trained agent once training has ended.
The agent's eval forward returns `pred_traj`, (batch, poses, 3) in the ego frame at 0.5 s spacing.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from recogdrive.eval.registry import SubprocessEvaluator, average_row, register

logger = logging.getLogger(__name__)

ENTRIES = {
    "navsim1.1": ("run_pdm_score_recogdrive.py", True),   # torch.distributed across GPUs
    "navsim2.0": ("run_pdm_score_one_stage.py", False),   # worker pool inside one process
}
SCORE_ENTRIES = {"navsim1.1": "run_pdm_score.py", "navsim2.0": "run_pdm_score_one_stage.py"}
POSE_INTERVAL_S = 0.5


def devkit_root() -> Path:
    import navsim

    return Path(navsim.__file__).resolve().parents[1]


@register("navsim")
class NavsimEvaluator(SubprocessEvaluator):
    def __init__(self, cfg, section, output_dir: Path):
        super().__init__(cfg, section, output_dir)
        self.loader = None

    def split_overrides(self) -> List[str]:
        split = OmegaConf.select(self.section, "split") or "navtest"
        return [f"train_test_split={split}"] + [str(o) for o in (OmegaConf.select(self.section, "split_overrides") or [])]

    def overrides(self, config_dir: Path, out_dir: Path, tag: str, agent: str) -> List[str]:
        out = [
            f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{config_dir.resolve().as_posix()}]",
            f"agent={agent}",
            *self.split_overrides(),
            f"experiment_name={OmegaConf.select(self.cfg, 'experiment_name') or 'recogdrive'}_{tag}",
            f"output_dir={out_dir.resolve().as_posix()}",
        ]
        metric_cache = OmegaConf.select(self.section, "metric_cache_path")
        if metric_cache:
            out.append(f"metric_cache_path={metric_cache}")
        return out + [str(o) for o in (OmegaConf.select(self.section, "overrides") or [])]

    def agent_config(self, ckpt: Path, config_dir: Path) -> None:
        agent = OmegaConf.to_container(self.cfg.agent, resolve=True)
        agent["checkpoint_path"] = str(ckpt)
        agent["grpo"] = False
        agent["cache_mode"] = False
        vlm = ckpt.with_name(ckpt.stem + "_vlm")
        if agent.get("train_backbone") and vlm.is_dir():
            agent["vlm_path"] = str(vlm)
        agent["train_backbone"] = False
        (config_dir / "agent").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(agent), config_dir / "agent" / "trained_agent.yaml")

    def command(self, ckpt: Path, out_dir: Path, tag: str) -> List[str]:
        root = devkit_root()
        if root.name not in ENTRIES:
            raise ValueError(f"cannot pick a NAVSIM scoring entry for devkit {root}")
        script, distributed = ENTRIES[root.name]
        config_dir = out_dir / "config"
        self.agent_config(ckpt, config_dir)
        overrides = self.overrides(config_dir, out_dir, tag, "trained_agent")
        entry = str(root / "navsim" / "planning" / "script" / script)
        if not distributed:
            return [sys.executable, entry, *overrides]
        nproc = OmegaConf.select(self.section, "nproc") or self.visible_gpus()
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", entry, *overrides]

    def replay_command(self, trajectories: Path, out_dir: Path, tag: str) -> List[str]:
        root = devkit_root()
        config_dir = out_dir / "config"
        (config_dir / "agent").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create({"_target_": "recogdrive.eval.navsim_replay.ReplayAgent",
                              "trajectories_path": str(trajectories.resolve())}),
            config_dir / "agent" / "replay_agent.yaml",
        )
        entry = str(root / "navsim" / "planning" / "script" / SCORE_ENTRIES[root.name])
        return [sys.executable, entry, *self.overrides(config_dir, out_dir, tag, "replay_agent")]

    def release(self) -> None:
        self.loader = None  # its worker transform holds the agent, VLM included

    def during_training(self, trainer, ckpt: Path, tag: str) -> None:
        out_dir = self.root / tag
        trajectories = self.generate(trainer, out_dir)
        if trainer.ctx.is_main:
            env = {**self.child_env(), "CUDA_VISIBLE_DEVICES": ""}
            self.launch(self.replay_command(trajectories, out_dir, tag), out_dir, tag, env)

    def eval_loader(self, trainer):
        if self.loader is None:
            from hydra import compose
            from hydra.core.hydra_config import HydraConfig

            from recogdrive.data import make_loader
            from recogdrive.data.navsim import agent_inputs
            from recogdrive.data.registry import with_worker_transform

            split_cfg = compose(config_name=HydraConfig.get().job.config_name, overrides=self.split_overrides())
            dataset = with_worker_transform(agent_inputs(split_cfg, trainer.agent), trainer.agent)
            self.loader = make_loader(dataset, trainer.cfg.dataloader.params, False, trainer.ctx.rank,
                                      trainer.ctx.world_size)
        return self.loader

    @torch.no_grad()
    def generate(self, trainer, out_dir: Path) -> Path:
        """Trajectories of every split scene from the trainer's model, gathered into one pickle on rank 0."""
        from recogdrive.data import DevicePrefetcher

        model, ctx = trainer.model, trainer.ctx
        was_training = model.training
        model.eval()
        mine: Dict[str, object] = {}
        for features, _, tokens in DevicePrefetcher(self.eval_loader(trainer), ctx.device):
            poses = model(features)["pred_traj"].float().cpu().numpy()
            mine.update(zip(tokens, poses))
        model.train(was_training)
        gathered = [None] * ctx.world_size if ctx.is_main else None
        dist.gather_object(mine, gathered, dst=0, group=ctx.gloo_group())
        path = out_dir / "trajectories.pkl"
        if ctx.is_main:
            from navsim.common.dataclasses import Trajectory
            from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

            merged = {token: poses for part in gathered for token, poses in part.items()}
            trajectories = {
                token: Trajectory(poses, TrajectorySampling(num_poses=len(poses), interval_length=POSE_INTERVAL_S))
                for token, poses in merged.items()
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(trajectories, f)
            logger.info("eval %s: %d trajectories from the training model -> %s", out_dir.name, len(trajectories), path)
        return path

    def visible_gpus(self) -> int:
        gpus = OmegaConf.select(self.section, "gpus")
        if gpus is not None:
            return len([g for g in str(gpus).split(",") if g.strip()])
        return max(torch.cuda.device_count(), 1)

    def read_result(self, out_dir: Path) -> Dict[str, float]:
        csvs = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            raise ValueError(f"no score csv in {out_dir}")
        row = average_row(csvs[-1])
        return {"score": row["score"], "csv": str(csvs[-1])}
