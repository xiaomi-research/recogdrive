"""Evaluator registry. An evaluator scores a saved checkpoint in child processes, so the
training loop never blocks on it and every benchmark keeps its own official scoring code."""

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


def build_evaluator(cfg: DictConfig, output_dir: Path):
    section = OmegaConf.select(cfg, "evaluator")
    name = OmegaConf.select(cfg, "evaluator.name") if section is not None else None
    if not name:
        return None
    cls = EVALUATORS.get(str(name))
    if cls is None:
        raise KeyError(f"Unknown evaluator {name!r}. Registered: {sorted(EVALUATORS)}")
    return cls(cfg, section, Path(output_dir))


class SubprocessEvaluator:
    def __init__(self, cfg: DictConfig, section: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.section = section
        self.root = output_dir / "eval"
        self.every_n_epochs = int(OmegaConf.select(section, "every_n_epochs") or 0)
        self.on_end = bool(OmegaConf.select(section, "on_end", default=True))
        self.jobs: List[Tuple[str, subprocess.Popen, Path]] = []

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

    def submit(self, ckpt: Path, tag: str) -> None:
        out_dir = self.root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.command(Path(ckpt), out_dir, tag)
        log = open(out_dir / "eval.log", "w")
        logger.info("eval %s started: %s", tag, " ".join(cmd))
        proc = subprocess.Popen(cmd, env=self.child_env(), stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self.jobs.append((tag, proc, out_dir))

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
        if self.on_end and final_ckpt is not None and Path(final_ckpt).is_file():
            self.submit(final_ckpt, "final")
        return self.collect(block=True)


def average_row(csv_path: Path) -> Dict[str, float]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in reversed(rows):
        if str(row.get("token", "")).startswith("average"):
            out = {}
            for key, value in row.items():
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    continue
            return out
    raise ValueError(f"no average row in {csv_path}")
