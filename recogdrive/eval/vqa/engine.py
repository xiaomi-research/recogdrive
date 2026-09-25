"""Batched VLM answering with lmdeploy (TurboMind): one worker process per GPU, each on its own shard.

A sample is a dict with `id`, `prompt` (text with one `<image>` per image) and `images` (paths or PIL images).
Shards append one JSON line per answer, so an interrupted run resumes where it stopped.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

SAMPLING = dict(do_sample=True, temperature=0.01, top_p=0.001, top_k=1)


def answer(model: str, samples: List[dict], system_prompt: str, max_new_tokens: int, min_new_tokens: int,
           batch_size: int, session_len: int, backend: str = "turbomind") -> Iterator[Tuple[str, Optional[str]]]:
    from lmdeploy import ChatTemplateConfig, GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
    from lmdeploy.vl import load_image

    engine = TurbomindEngineConfig if backend == "turbomind" else PytorchEngineConfig
    pipe = pipeline(
        model,
        backend_config=engine(session_len=session_len),
        chat_template_config=ChatTemplateConfig(model_name="internvl2_5", meta_instruction=system_prompt),
    )
    generation = GenerationConfig(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, **SAMPLING)
    for start in range(0, len(samples), batch_size):
        ids, prompts = [], []
        for sample in samples[start:start + batch_size]:
            images = []
            for image in sample["images"]:
                try:
                    images.append(load_image(image) if isinstance(image, str) else image)
                except OSError as exc:
                    print(f"skipping unreadable image {image}: {exc}", flush=True)
            if images:
                ids.append(sample["id"])
                prompts.append((sample["prompt"], images))
            else:
                yield sample["id"], None
        if prompts:
            for sample_id, response in zip(ids, pipe(prompts, gen_config=generation)):
                yield sample_id, response.text


def shard_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"answers.shard{index}.jsonl"


def answered(out_dir: Path) -> Dict[str, Optional[str]]:
    """Answers from every shard file, including those of an earlier run with another GPU count."""
    out: Dict[str, Optional[str]] = {}
    for path in sorted(out_dir.glob("answers.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line another shard is still writing
                continue
            out[row["id"]] = row["answer"]
    return out


def run_shard(out_dir: Path, index: int, samples: List[dict], **engine) -> None:
    done = answered(out_dir)
    todo = [s for s in samples if s["id"] not in done]
    if not todo:
        return
    with shard_path(out_dir, index).open("a", encoding="utf-8") as f:
        for sample_id, text in answer(samples=todo, **engine):
            f.write(json.dumps({"id": sample_id, "answer": text}, ensure_ascii=False) + "\n")
            f.flush()


def visible_gpus() -> List[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
    return [str(i) for i in range(len(listed))] or ["0"]


def run_all(argv: List[str], gpus: List[str], out_dir: Path) -> Tuple[Dict[str, Optional[str]], List[int]]:
    """Runs `python -m recogdrive.eval.vqa <argv> --shard i/n` on every GPU; returns the merged answers and
    the shards' exit codes. Completeness is judged from the answers: engines can crash while tearing down."""
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "recogdrive.eval.vqa", *argv, "--shard", f"{index}/{len(gpus)}"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
        )
        for index, gpu in enumerate(gpus)
    ]
    codes = [proc.wait() for proc in procs]
    return answered(out_dir), codes
