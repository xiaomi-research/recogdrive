"""Driving VQA benchmarks on the VLM (DriveLM, LingoQA, DriveBench).

Run standalone with `python -m recogdrive.eval.vqa`, or as a training evaluator (`evaluator.name=drivelm`,
options under `evaluator.vqa`): it scores a checkpoint's `_vlm` export when present, otherwise `agent.vlm_path`;
LoRA checkpoints are merged into a temporary CPU export before evaluation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

from recogdrive.eval.registry import SubprocessEvaluator, register
from recogdrive.infer import vlm_export_path


class VqaEvaluator(SubprocessEvaluator):
    bench = ""

    def __init__(self, cfg, section, output_dir: Path):
        super().__init__(cfg, section, output_dir)
        self.output_dir = Path(output_dir)
        self.lora = bool(OmegaConf.select(cfg, "agent.lora_rank"))

    def model_path(self, ckpt: Path) -> str:
        vlm = vlm_export_path(ckpt)
        return str(vlm) if vlm is not None else str(OmegaConf.select(self.cfg, "agent.vlm_path"))

    def command(self, ckpt: Path, out_dir: Path, tag: str) -> List[str]:
        options = OmegaConf.to_container(OmegaConf.select(self.section, "vqa") or {}, resolve=True)
        if self.lora:
            out_dir.mkdir(parents=True, exist_ok=True)
            agent = OmegaConf.to_container(self.cfg.agent, resolve=True)
            agent["checkpoint_path"] = None
            agent["grpo"] = False
            agent_config = out_dir / "vqa_agent.yaml"
            OmegaConf.save(OmegaConf.create(agent), agent_config)
            model_arg = "__RECOGDRIVE_VQA_MODEL__"
            cmd = [sys.executable, "-m", "recogdrive.eval.vqa.export",
                   "--checkpoint", str(ckpt), "--agent-config", str(agent_config),
                   "--cache-root", str(self.output_dir / ".vqa_exports")]
            vqa_cmd = [self.bench, "--model", model_arg, "--out", str(out_dir)]
        else:
            cmd = [sys.executable, "-m", "recogdrive.eval.vqa", self.bench,
                   "--model", self.model_path(ckpt), "--out", str(out_dir)]
            vqa_cmd = cmd
        for key, value in options.items():
            if value is None or value is False or value == "":
                continue
            flag = "--" + key.replace("_", "-")
            vqa_cmd += [flag] if value is True else [flag, str(value)]
        if self.lora:
            cmd += ["--", *vqa_cmd]
        return cmd

    def child_env(self) -> Dict[str, str]:
        env = super().child_env()
        if self.lora:
            env["RECOGDRIVE_VQA_CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "__unset__")
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

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
