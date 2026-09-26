"""Inference with a trained checkpoint on any registered data source: `run_recogdrive_infer.py` (Hydra, the training
config) with

    infer.checkpoint=/path/to/epoch_xxx.ckpt   weights exported by the trainer (its `_vlm` export is used if present)
    infer.split=val                            train | val | test (NAVSIM test: every scene of train_test_split)
    infer.visualize=16                         the first N samples drawn to <output_dir>/vis/<token>.png
    data_loader=nuscenes ...                   the data source, configured as for training (or infer.data_loader)

It writes <output_dir>/predictions.json, {token: [[x, y, heading], ...]} in the ego frame at 0.5 s spacing. Launched
with torchrun, the samples are sharded over the GPUs.
"""

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger("recogdrive")

EVAL_AGENT = {"checkpoint_path": None, "grpo": False, "cache_mode": False, "train_backbone": False}


def load_weights(agent, ckpt: Path) -> None:
    state = torch.load(ckpt, map_location="cpu")["state_dict"]
    state = {k[len("agent."):] if k.startswith("agent.") else k: v for k, v in state.items()}
    missing, unexpected = agent.load_state_dict(state, strict=False)
    # checkpoints hold the trained parameters; frozen ones (the VLM from vlm_path, fixed planner parts) keep their values
    trainable = {name for name, p in agent.named_parameters() if p.requires_grad}
    missing = [k for k in missing if k in trainable]
    if missing or unexpected:
        raise ValueError(f"{ckpt} does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")


def run(cfg) -> None:
    from recogdrive.adapters import check_policy, predict
    from recogdrive.data import LOADERS, dataset_of, import_plugins, loader_name, make_loader
    from recogdrive.data.registry import with_worker_transform
    from recogdrive.distributed import DistributedContext

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    ctx = DistributedContext()
    import_plugins(cfg)
    ckpt = Path(str(OmegaConf.select(cfg, "infer.checkpoint") or ""))
    if not ckpt.is_file():
        raise FileNotFoundError(f"infer.checkpoint={ckpt} is not a file")
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    agent_cfg.update({key: value for key, value in EVAL_AGENT.items() if key in agent_cfg})
    vlm = ckpt.with_name(ckpt.stem + "_vlm")
    if vlm.is_dir():
        agent_cfg["vlm_path"] = str(vlm)
    agent = instantiate(agent_cfg)
    check_policy(agent)
    load_weights(agent, ckpt)
    agent.to(ctx.device).eval()

    name = str(OmegaConf.select(cfg, "infer.data_loader") or loader_name(cfg))
    split = str(OmegaConf.select(cfg, "infer.split") or "val")
    dataset = with_worker_transform(LOADERS[name](cfg, agent, split), agent)
    predictions = predict(agent, make_loader(dataset, cfg.dataloader.params, False, ctx.rank, ctx.world_size), ctx)
    if ctx.is_main:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions.json").write_text(
            json.dumps({token: np.round(poses, 4).tolist() for token, poses in predictions.items()}))
        logger.info("%d %s/%s predictions from %s -> %s", len(predictions), name, split, ckpt, out_dir / "predictions.json")
        count = int(OmegaConf.select(cfg, "infer.visualize") or 0)
        if count:
            visualize(dataset_of(dataset), predictions, out_dir / "vis", count, getattr(agent, "target_to_waypoint", None))
    ctx.close()


def front_image(features: dict):
    for view, path in features.get("camera_paths", []):
        if view == "front":
            return path
    tensor = features.get("image_path_tensor")
    return "".join(map(chr, itertools.takewhile(bool, tensor.tolist()))) if tensor is not None else None


def visualize(dataset, predictions, out_dir: Path, count: int, to_waypoint=None) -> None:
    """Front camera next to a bird's-eye view (x forward up, y left) of history, ground truth and prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(count, len(dataset))):
        features, targets, token = dataset[index]
        if token not in predictions:
            continue
        fig, bev = plt.subplots(figsize=(5, 5))
        tracks = [("history", features["history_trajectory"], "gray")]
        if "trajectory" in targets:
            truth = targets["trajectory"]
            tracks.append(("ground truth", to_waypoint(truth) if to_waypoint else truth, "tab:green"))
        tracks.append(("prediction", predictions[token], "tab:red"))
        for label, poses, color in tracks:
            poses = np.asarray(poses, dtype=np.float64)
            bev.plot(-poses[:, 1], poses[:, 0], "o-", ms=3, color=color, label=label)
        bev.set_aspect("equal", adjustable="datalim")  # a straight drive keeps a square panel
        bev.set_title(token, fontsize=8)
        bev.set_xlabel("right (m)")
        bev.set_ylabel("forward (m)")
        bev.grid(alpha=0.3)
        bev.legend()
        fig.tight_layout()
        fig.canvas.draw()
        plot = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)
        path = front_image(features)
        panels = [plot]
        if path:  # pasted with PIL: matplotlib's imshow is not needed for a photo
            camera = Image.open(path).convert("RGB")
            panels.insert(0, camera.resize((round(camera.width * plot.height / camera.height), plot.height)))
        canvas = Image.new("RGB", (sum(p.width for p in panels), plot.height), "white")
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        canvas.save(out_dir / f"{token}.png")
    logger.info("visualized %d samples -> %s", min(count, len(dataset)), out_dir)
