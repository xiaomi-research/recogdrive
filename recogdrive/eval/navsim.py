"""NAVSIM PDMS (1.1) / EPDMS (2.0) on a training checkpoint, through the tree's own scoring entry."""

import sys
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

from recogdrive.eval.registry import SubprocessEvaluator, average_row, register

ENTRIES = {
    "navsim1.1": ("run_pdm_score_recogdrive.py", True),   # torch.distributed across GPUs
    "navsim2.0": ("run_pdm_score_one_stage.py", False),   # worker pool inside one process
}


def devkit_root() -> Path:
    import navsim

    return Path(navsim.__file__).resolve().parents[1]


@register("navsim")
class NavsimEvaluator(SubprocessEvaluator):
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
        section = self.section
        overrides = [
            f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{config_dir.resolve().as_posix()}]",
            "agent=trained_agent",
            f"train_test_split={OmegaConf.select(section, 'split') or 'navtest'}",
            f"experiment_name={OmegaConf.select(self.cfg, 'experiment_name') or 'recogdrive'}_{tag}",
            f"output_dir={out_dir.resolve().as_posix()}",
        ]
        metric_cache = OmegaConf.select(section, "metric_cache_path")
        if metric_cache:
            overrides.append(f"metric_cache_path={metric_cache}")
        overrides += [str(o) for o in (OmegaConf.select(section, "overrides") or [])]
        entry = str(root / "navsim" / "planning" / "script" / script)
        if not distributed:
            return [sys.executable, entry, *overrides]
        nproc = OmegaConf.select(section, "nproc") or self.visible_gpus()
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", entry, *overrides]

    def visible_gpus(self) -> int:
        gpus = OmegaConf.select(self.section, "gpus")
        if gpus is not None:
            return len([g for g in str(gpus).split(",") if g.strip()])
        import torch

        return max(torch.cuda.device_count(), 1)

    def read_result(self, out_dir: Path) -> Dict[str, float]:
        csvs = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            raise ValueError(f"no score csv in {out_dir}")
        row = average_row(csvs[-1])
        return {"score": row["score"], "csv": str(csvs[-1])}
