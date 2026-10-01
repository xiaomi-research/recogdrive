"""NVIDIA PhysicalAI Autonomous Vehicles (Hugging Face nvidia/PhysicalAI-Autonomous-Vehicles). The data is gated
by the NVIDIA AV dataset license, which forbids distributing derived data: converted caches stay on your machines.

`python -m recogdrive.data.physicalai --src <download> --out <cache_path>` converts the clips of a download in the
Hugging Face layout (clip_index.parquet, labels/egomotion/egomotion.chunk_*.zip, camera/<camera>/<camera>.chunk_*.zip):
2 Hz keyframes from the 30 fps videos saved as JPEG (front only, or every mapped camera with --views all), the ~100 Hz
egomotion interpolated at 0.5 s steps (history t = -1.5..0 s, target t = 0.5..5 s) in the current ego frame (the rig:
rear axle on the ground, x forward, y left), velocity and acceleration rotated from the clip's anchor frame into it.
The data has no route, so the command is the turn of the 4 s future path (the nuScenes / UniAD rule). Clips go to the
split clip_index.parquet gives them. The cache has the waymoe2e layout and loads with data_loader=physicalai
(physicalai.root=<cache_path>); every sample is scored by evaluator.name=physicalai (L2 / ADE / FDE).
"""

import argparse
import io
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from recogdrive.data.nuscenes import command_one_hot, quaternion_yaw, rotate_to, to_local
from recogdrive.data.registry import POSE_INTERVAL_S, horizon, register
from recogdrive.data.waymoe2e import WaymoE2ECacheOnlyDataset

CAMERAS = {  # camera_front_tele_30fov has no canonical view
    "camera_front_wide_120fov": "front", "camera_cross_left_120fov": "left", "camera_cross_right_120fov": "right",
    "camera_rear_left_70fov": "back_left", "camera_rear_right_70fov": "back_right", "camera_rear_tele_30fov": "back",
}
HISTORY, FUTURE, COMMAND_POSES = 3, 10, 8


def read_egomotion(frame) -> dict:
    """Arrays over time (s): x, y, yaw (unwrapped) and the anchor-frame velocity / acceleration."""
    frame = frame.sort_values("timestamp")
    yaw = [quaternion_yaw(q) for q in frame[["qw", "qx", "qy", "qz"]].to_numpy()]
    return {"t": frame["timestamp"].to_numpy() * 1e-6, "x": frame["x"].to_numpy(), "y": frame["y"].to_numpy(),
            "yaw": np.unwrap(yaw), **{k: frame[k].to_numpy() for k in ("vx", "vy", "ax", "ay")}}


def keyframe_samples(ego: dict, frame_times: np.ndarray):
    """(time, history (4, 3), target (FUTURE, 3), command, status) at every 0.5 s keyframe the egomotion covers from
    1.5 s before to 5 s after and a video frame lies within half a frame interval of."""
    half_frame = 0.5 * float(np.median(np.diff(frame_times)))
    first = max(ego["t"][0] + HISTORY * POSE_INTERVAL_S, frame_times[0] - half_frame)
    last = min(ego["t"][-1] - FUTURE * POSE_INTERVAL_S, frame_times[-1] + half_frame)
    for now in np.arange(np.ceil(first / POSE_INTERVAL_S), np.floor(last / POSE_INTERVAL_S) + 1) * POSE_INTERVAL_S:
        times = now + POSE_INTERVAL_S * np.arange(-HISTORY, FUTURE + 1)
        poses = np.stack([np.interp(times, ego["t"], ego[k]) for k in ("x", "y", "yaw")], axis=1)
        local = to_local(poses, poses[HISTORY])
        heading = poses[HISTORY, 2]
        velocity = rotate_to([np.interp(now, ego["t"], ego[k]) for k in ("vx", "vy")], heading)
        acceleration = rotate_to([np.interp(now, ego["t"], ego[k]) for k in ("ax", "ay")], heading)
        command = command_one_hot(local[HISTORY + 1 : HISTORY + 1 + COMMAND_POSES])
        status = np.concatenate([command, velocity, acceleration]).astype(np.float32)
        yield now, local[: HISTORY + 1], local[HISTORY + 1 :], command, status


def video_frames(data: bytes, timestamps_us: np.ndarray, times: np.ndarray):
    """RGB frames of an mp4 nearest to `times` (s)."""
    import decord

    nearest = np.abs(timestamps_us[None, :] * 1e-6 - times[:, None]).argmin(axis=1)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(data)
        f.flush()
        return decord.VideoReader(f.name).get_batch(nearest.tolist()).asnumpy()


def convert(src: Path, out: Path, all_views: bool = False, stride: int = 1, max_clips: int = 0) -> int:
    import pandas as pd
    from navsim.planning.training.dataset import dump_feature_target_to_pickle as dump

    index = pd.read_parquet(src / "clip_index.parquet")
    written = clips = 0
    for ego_zip in sorted((src / "labels" / "egomotion").glob("egomotion.chunk_*.zip")):
        chunk = ego_zip.name[len("egomotion."):-len(".zip")]
        cameras = {}
        for name, view in CAMERAS.items():
            path = src / "camera" / name / f"{name}.{chunk}.zip"
            if (view == "front" or all_views) and path.is_file():
                cameras[name] = path
        if "camera_front_wide_120fov" not in cameras:
            continue
        with zipfile.ZipFile(ego_zip) as zf:
            members = sorted(n for n in zf.namelist() if n.endswith(".egomotion.parquet"))
            for member in members:
                if max_clips and clips >= max_clips:
                    return written
                clip = member[: -len(".egomotion.parquet")]
                if clip not in index.index:
                    continue
                split = str(index.at[clip, "split"])
                ego = read_egomotion(pd.read_parquet(io.BytesIO(zf.read(member))))
                videos = {}
                for name, path in cameras.items():
                    with zipfile.ZipFile(path) as cz:
                        stamps = pd.read_parquet(io.BytesIO(cz.read(f"{clip}.{name}.timestamps.parquet")))
                        videos[name] = (cz.read(f"{clip}.{name}.mp4"), stamps.sort_values("frame_index")["timestamp"].to_numpy())
                samples = list(keyframe_samples(ego, videos["camera_front_wide_120fov"][1] * 1e-6))[::stride]
                if not samples:
                    continue
                times = np.array([s[0] for s in samples])
                frames = {name: video_frames(data, stamps, times) for name, (data, stamps) in videos.items()}
                image_dir = (out / "images" / split).resolve()
                image_dir.mkdir(parents=True, exist_ok=True)
                for k, (now, history, target, command, status) in enumerate(samples):
                    token = f"{re.sub(r'[^A-Za-z0-9_-]', '_', clip)}-{round(now * 1e3):06d}"
                    camera_paths = []
                    for name in cameras:
                        view = CAMERAS[name]
                        path = image_dir / (f"{token}.jpg" if view == "front" else f"{token}_{view}.jpg")
                        Image.fromarray(frames[name][k]).save(path, quality=95)
                        camera_paths.append((view, str(path)))
                    sample_dir = out / split / token
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    dump(sample_dir / "features.gz", {
                        "history_trajectory": torch.tensor(history, dtype=torch.float32),
                        "high_command_one_hot": torch.tensor(command),
                        "status_feature": torch.tensor(status),
                        "camera_paths": camera_paths,
                    })
                    trajectory = torch.tensor(target, dtype=torch.float32)
                    dump(sample_dir / "targets.gz", {"trajectory": trajectory})
                    dump(sample_dir / "eval.gz", {"trajectory": trajectory})
                    written += 1
                clips += 1
    return written


@register("physicalai")
def physicalai_loader(cfg, agent, split: str):
    root = OmegaConf.select(cfg, "physicalai.root")
    if not root:
        raise ValueError("set physicalai.root to a cache built by python -m recogdrive.data.physicalai")
    return WaymoE2ECacheOnlyDataset(root, "train" if split == "train" else "val", horizon(agent),
                                    agent.get_target_builders())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhysicalAI-AV download (Hugging Face layout) -> recogdrive cache")
    parser.add_argument("--src", required=True, type=Path, help="holds clip_index.parquet, labels/ and camera/")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--views", choices=["front", "all"], default="front", help="all: every mapped camera")
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th 2 Hz keyframe")
    parser.add_argument("--max-clips", type=int, default=0, help="stop after this many clips (0: all)")
    args = parser.parse_args()
    written = convert(args.src, args.out, args.views == "all", args.stride, args.max_clips)
    print(f"wrote {written} samples to {args.out}")
