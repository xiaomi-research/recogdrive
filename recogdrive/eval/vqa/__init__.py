"""Driving VQA benchmarks on the VLM (DriveLM, LingoQA, DriveBench).

Run standalone with `python -m recogdrive.eval.vqa`, or as a training evaluator (`evaluator.name=drivelm`,
options under `evaluator.vqa`): it scores the `_vlm` export next to a checkpoint when the backbone was
trained, otherwise `agent.vlm_path`.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

from recogdrive.eval.registry import SubprocessEvaluator, register


class VqaEvaluator(SubprocessEvaluator):
    bench = ""

    def model_path(self, ckpt: Path) -> str:
        vlm = ckpt.with_name(ckpt.stem + "_vlm")
        return str(vlm) if vlm.is_dir() else str(OmegaConf.select(self.cfg, "agent.vlm_path"))

    def command(self, ckpt: Path, out_dir: Path, tag: str) -> List[str]:
        cmd = [sys.executable, "-m", "recogdrive.eval.vqa", self.bench,
               "--model", self.model_path(ckpt), "--out", str(out_dir)]
        options = OmegaConf.to_container(OmegaConf.select(self.section, "vqa") or {}, resolve=True)
        for key, value in options.items():
            if value is None or value is False or value == "":
                continue
            flag = "--" + key.replace("_", "-")
            cmd += [flag] if value is True else [flag, str(value)]
        return cmd

    def read_result(self, out_dir: Path) -> Dict:
        return json.loads((out_dir / "score.json").read_text(encoding="utf-8"))


@register("drivelm")
class DriveLMEvaluator(VqaEvaluator):
    bench = "drivelm"


@register("lingoqa")
class LingoQAEvaluator(VqaEvaluator):
    bench = "lingoqa"


@register("drivebench")
class DriveBenchEvaluator(VqaEvaluator):
    bench = "drivebench"
