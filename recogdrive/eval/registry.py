"""Evaluator registry. `evaluator.name` is one registered benchmark or a list of them; `evaluator.<name>` holds
settings of one benchmark over the shared ones. Subprocess evaluators score a saved checkpoint in child processes,
so the training loop never blocks on them and every benchmark keeps its own official scoring code; in-process ones
(`in_process = True`) score the live model, also at the end of training before it is released.

An evaluator provides every_n_epochs, on_end, during_training(trainer, ckpt, tag) (every rank), poll() and
finish(final_ckpt) (main rank, (tag, metrics) pairs), and release()."""

import csv
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

EVALUATORS: Dict[str, type] = {}

# Launcher variables of the training job; a child that inherits them would join the training rendezvous.
LAUNCH_ENV = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
    "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
)


def register(name: str):
    def deco(cls):
        EVALUATORS[name] = cls
        return cls
    return deco


def build_evaluators(cfg: DictConfig, output_dir: Path) -> list:
    section = OmegaConf.select(cfg, "evaluator")
    names = OmegaConf.select(cfg, "evaluator.name") if section is not None else None
    if not names:
        return []
    evaluators = []
    for name in [names] if isinstance(names, str) else list(names):
        cls = EVALUATORS.get(str(name))
        if cls is None:
            raise KeyError(f"Unknown evaluator {name!r}. Registered: {sorted(EVALUATORS)}")
        own = OmegaConf.select(section, str(name))
        # a fresh root: Hydra's struct flag on `section` would reject the benchmark's own keys
        merged = OmegaConf.merge(OmegaConf.create(), section, own) if isinstance(own, DictConfig) else section
        evaluator = cls(cfg, merged, Path(output_dir))
        evaluator.benchmark = str(name)
        evaluator.root = Path(output_dir) / "eval" / str(name)
        evaluators.append(evaluator)
    return evaluators


class SubprocessEvaluator:
    """Built on every rank; only the main rank launches jobs.

    During training, a job that needs GPUs runs at once only on GPUs training does not use
    (`evaluator.gpus`); otherwise it waits until training has released its GPUs.
    """

    def __init__(self, cfg: DictConfig, section: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.section = section
        self.root = output_dir / "eval"
        self.every_n_epochs = int(OmegaConf.select(section, "every_n_epochs") or 0)
        self.on_end = bool(OmegaConf.select(section, "on_end", default=True))
        self.jobs: List[Tuple[str, subprocess.Popen, Path]] = []
        self.deferred: List[Tuple[Path, str]] = []

    def command(self, ckpt: Path, out_dir: Path, tag: str) -> List[str]:
        raise NotImplementedError

    def read_result(self, out_dir: Path) -> Dict[str, float]:
        raise NotImplementedError

    def child_env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in LAUNCH_ENV and not k.startswith("TORCHELASTIC")}
        gpus = OmegaConf.select(self.section, "gpus")
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus)
        return env

    def launch(self, cmd: List[str], out_dir: Path, tag: str, env: Dict[str, str]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "eval.log", "w")
        logger.info("eval %s started: %s", tag, " ".join(cmd))
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self.jobs.append((tag, proc, out_dir))

    def submit(self, ckpt: Path, tag: str) -> None:
        out_dir = self.root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        self.launch(self.command(Path(ckpt), out_dir, tag), out_dir, tag, self.child_env())

    def release(self) -> None:
        """Drops references into the training process (model, data) before training frees its GPUs."""

    def during_training(self, trainer, ckpt: Path, tag: str) -> None:
        """Called on every rank at an evaluation epoch, with the model holding the weights `ckpt` holds."""
        if not trainer.ctx.is_main:
            return
        if OmegaConf.select(self.section, "gpus") is not None:
            self.submit(ckpt, tag)
            return
        logger.info("eval %s waits for the end of training; set evaluator.gpus to GPUs training does not use "
                    "to run it now", tag)
        self.deferred.append((snapshot(Path(ckpt), self.root / tag), tag))

    def collect(self, block: bool) -> List[Tuple[str, Dict]]:
        done, running = [], []
        for tag, proc, out_dir in self.jobs:
            if block:
                proc.wait()
            if proc.poll() is None:
                running.append((tag, proc, out_dir))
                continue
            if proc.returncode != 0:
                done.append((tag, {"error": f"exit {proc.returncode}, see {out_dir / 'eval.log'}"}))
                continue
            try:
                done.append((tag, self.read_result(out_dir)))
            except (OSError, ValueError, KeyError) as exc:
                done.append((tag, {"error": f"{exc}, see {out_dir}"}))
        self.jobs = running
        return done

    def poll(self) -> List[Tuple[str, Dict]]:
        return self.collect(block=False)

    def finish(self, final_ckpt: Optional[Path]) -> List[Tuple[str, Dict]]:
        """Runs the deferred and the final evaluation one at a time, once training has freed its GPUs."""
        results = self.collect(block=True)
        queue = list(self.deferred)
        if self.on_end and final_ckpt is not None and Path(final_ckpt).is_file():
            queue.append((Path(final_ckpt), "final"))
        for ckpt, tag in queue:
            self.submit(ckpt, tag)
            results += self.collect(block=True)
        self.deferred = []
        return results


def snapshot(ckpt: Path, out_dir: Path) -> Path:
    """Copy of a checkpoint (and its `_vlm` export) that later epochs cannot overwrite."""
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    copy = out_dir / ckpt.name
    shutil.copyfile(ckpt, copy)
    vlm = ckpt.with_name(ckpt.stem + "_vlm")
    if vlm.is_dir():
        shutil.copytree(vlm, copy.with_name(copy.stem + "_vlm"), dirs_exist_ok=True)
    return copy


def average_row(csv_path: Path) -> Dict[str, float]:
    """Summary row of a NAVSIM score csv: `average*` (one-stage) or `extended_pdm_score_combined` (two-stage)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in reversed(rows):
        if str(row.get("token", "")).startswith(("average", "extended_pdm_score_combined")):
            out = {}
            for key, value in row.items():
                if not key:  # the csv's unnamed index column
                    continue
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    continue
            return out
    raise ValueError(f"no average row in {csv_path}")
