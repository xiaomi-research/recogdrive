"""Local JSONL VQA records; VLM tokenization and image preprocessing live with the model."""

import json
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from recogdrive.data.registry import TransformedDataset, register


class VQADataset(Dataset):
    task = "vqa"

    def __init__(self, jsonl, image_root):
        self.path = Path(jsonl).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve()
        self.samples = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid VQA JSONL at {self.path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"VQA JSONL row {line_number} must be an object")
                images = row.get("images", row.get("image_paths", row.get("image", row.get("image_path"))))
                if isinstance(images, dict):
                    images = list(images.values())
                if isinstance(images, str):
                    images = [images]
                if not isinstance(images, list) or not images or not all(isinstance(p, str) and p for p in images):
                    raise ValueError(f"VQA JSONL row {line_number} needs image, image_path, images, or image_paths")
                question, answer = row.get("question"), row.get("answer")
                if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"VQA JSONL row {line_number} needs non-empty string question and answer")
                paths = []
                for image in images:
                    image_path = Path(image).expanduser()
                    image_path = (image_path if image_path.is_absolute() else self.image_root / image_path).resolve()
                    try:
                        image_path.relative_to(self.image_root)
                    except ValueError as exc:
                        raise ValueError(f"VQA image path escapes vqa.image_root: {image_path}") from exc
                    paths.append(str(image_path))
                self.samples.append((str(row.get("id", line_number)), paths, question, answer))
        if not self.samples:
            raise ValueError(f"No VQA examples found in {self.path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        token, image_paths, question, answer = self.samples[index]
        return {"task": "vqa", "image_paths": image_paths, "question": question, "answer": answer}, {"answer": answer}, token


@register("vqa")
def vqa_loader(cfg, agent, split: str):
    if split not in ("train", "val"):
        raise ValueError(f"VQA loader supports train/val splits, got {split!r}")
    transform = getattr(agent, "vqa_worker_transform", None)
    if not callable(transform):
        raise TypeError("the policy adapter must provide vqa_worker_transform() for VQA data")
    jsonl = OmegaConf.select(cfg, f"vqa.{split}_jsonl") or OmegaConf.select(cfg, "vqa.jsonl")
    if not jsonl:
        raise ValueError(f"Set vqa.{split}_jsonl or vqa.jsonl to a local JSONL file")
    image_root = OmegaConf.select(cfg, "vqa.image_root") or Path(jsonl).expanduser().resolve().parent
    max_length = OmegaConf.select(cfg, "vqa.max_length")
    return TransformedDataset(VQADataset(jsonl, image_root), transform(max_length))
