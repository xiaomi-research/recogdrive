"""Strip UniDriveVLA mmengine keys down to a HF Qwen3-VL directory.

Expects a Stage1 checkpoint whose tensors are prefixed with
``planning_head.qwen3_vl_with_expert.qwen3_vl.`` and a Qwen3-VL config dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

PREFIXES = (
    "planning_head.qwen3_vl_with_expert.qwen3_vl.",
    "qwen3_vl_with_expert.qwen3_vl.",
    "qwen3_vl.",
)


def strip_unidrive_prefix(key: str) -> Optional[str]:
    for prefix in PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def export_qwen_state(src_state: dict) -> dict:
    exported = {}
    for key, value in src_state.items():
        stripped = strip_unidrive_prefix(key)
        if stripped is not None:
            exported[stripped] = value
    if not exported:
        raise ValueError("No UniDriveVLA Qwen3-VL tensors found. Check Stage1 ckpt keys.")
    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path, help="HF Qwen3-VL-2B dir with config.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import torch

    blob = torch.load(args.ckpt, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    exported = export_qwen_state(blob)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"):
        src = args.config_dir / name
        if src.exists():
            (args.out / name).write_bytes(src.read_bytes())
    torch.save(exported, args.out / "pytorch_model.bin")
    print("WROTE", args.out, "TENSORS", len(exported))


if __name__ == "__main__":
    main()
