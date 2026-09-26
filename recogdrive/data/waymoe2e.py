"""Waymo Open Dataset End-to-End Driving (WOD-E2E).

Training reads the cache layout `<cache_path>/<split>/<token>/{features,targets}.gz` (plus `eval.gz` with the rater
trajectories of labelled frames). `python -m recogdrive.data.waymoe2e --out <cache_path> --split training
<tfrecords...>` builds it from the raw TFRecords: camera JPEGs to disk (front only, or all 8 with --views all), 4 Hz
ego states resampled to NAVSIM's 0.5 s spacing (history t = -1.5..0 s, target t = 0.5..5 s) in the current ego frame
(x forward, y left). The loader cuts the target to the model's horizon: 8 poses (4 s) like NAVSIM, or 10 (5 s) for
the WOD-E2E metrics.
"""

import argparse
import io
import re
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from recogdrive.data.registry import horizon, register

CAMERAS = {1: "front", 2: "front_left", 3: "front_right", 4: "left", 5: "right", 6: "back_left", 7: "back", 8: "back_right"}
CAMERA_ASPECT = 16 / 9  # NAVSIM camera shape: tiles into 9 patches, which the VLM prompt budget (2800 tokens) is sized for
HISTORY_IDX = [-7, -5, -3, -1]           # past states: 16 at 4 Hz, t = -3.75 .. 0 s
FUTURE_IDX = list(range(1, 20, 2))        # future states: 20 at 4 Hz, t = 0.25 .. 5 s -> 0.5 .. 5 s
INTENT_TO_COMMAND = {2: 0, 1: 1, 3: 2, 0: 3}  # WOD LEFT/STRAIGHT/RIGHT/UNKNOWN -> NAVSIM left, straight, right, unknown
STILL_M = 0.05
UNLABELLED = -1.0  # preference_score of frames without rater trajectories


def e2ed_frame_class():
    """Only the E2EDFrame fields training needs; protobuf skips every other field of the WOD Frame."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    F = descriptor_pb2.FieldDescriptorProto
    fdp = descriptor_pb2.FileDescriptorProto(name="rcd_wod_e2ed.proto", package="rcd_wod", syntax="proto2")

    def message(name, fields):
        m = fdp.message_type.add(name=name)
        for field, number, kind, repeated, type_name in fields:
            f = m.field.add(name=field, number=number, type=kind,
                            label=F.LABEL_REPEATED if repeated else F.LABEL_OPTIONAL)
            if type_name:
                f.type_name = f".rcd_wod.{type_name}"

    message("CameraImage", [("name", 1, F.TYPE_INT32, False, None), ("image", 2, F.TYPE_BYTES, False, None)])
    message("Context", [("name", 1, F.TYPE_STRING, False, None)])
    message("Frame", [("context", 1, F.TYPE_MESSAGE, False, "Context"),
                      ("timestamp_micros", 2, F.TYPE_INT64, False, None),
                      ("images", 4, F.TYPE_MESSAGE, True, "CameraImage")])
    message("EgoTrajectoryStates", [(n, i, F.TYPE_FLOAT, True, None) for i, n in enumerate(
        ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "accel_x", "accel_y"], start=1)]
        + [("preference_score", 8, F.TYPE_FLOAT, False, None)])
    message("E2EDFrame", [("frame", 1, F.TYPE_MESSAGE, False, "Frame"),
                          ("future_states", 5, F.TYPE_MESSAGE, False, "EgoTrajectoryStates"),
                          ("past_states", 6, F.TYPE_MESSAGE, False, "EgoTrajectoryStates"),
                          ("intent", 7, F.TYPE_INT32, False, None),
                          ("preference_trajectories", 8, F.TYPE_MESSAGE, True, "EgoTrajectoryStates")])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    desc = pool.FindMessageTypeByName("rcd_wod.E2EDFrame")
    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(desc)
    return message_factory.MessageFactory(pool).GetPrototype(desc)


def read_tfrecord(path):
    with open(path, "rb") as f:
        while True:
            header = f.read(12)  # uint64 length + uint32 length crc
            if len(header) < 12:
                return
            data = f.read(struct.unpack("<Q", header[:8])[0])
            f.read(4)  # data crc
            yield data


def path_headings(xy: np.ndarray, now: int) -> np.ndarray:
    """Central-difference tangent of a uniformly sampled path; heading at `now` is 0 (current ego frame).

    While the ego stands still, the heading nearest to `now` in time is kept.
    """
    step = np.empty_like(xy)
    step[1:-1] = xy[2:] - xy[:-2]
    # second-order one-sided differences at the ends (the 5 s target is the last state), as accurate as central ones
    step[0] = -3 * xy[0] + 4 * xy[1] - xy[2]
    step[-1] = 3 * xy[-1] - 4 * xy[-2] + xy[-3]
    moving = np.hypot(step[:, 0], step[:, 1]) > STILL_M
    raw = np.arctan2(step[:, 1], step[:, 0])
    headings = np.zeros(len(xy))
    for i in range(now + 1, len(xy)):
        headings[i] = raw[i] if moving[i] else headings[i - 1]
    for i in range(now - 1, -1, -1):
        headings[i] = raw[i] if moving[i] else headings[i + 1]
    return headings


def frame_to_sample(frame):
    past, future = frame.past_states, frame.future_states
    if len(past.pos_x) < 16 or len(future.pos_x) < 20:
        return None
    past_xy = np.stack([past.pos_x, past.pos_y], axis=1).astype(np.float64)
    future_xy = np.stack([future.pos_x, future.pos_y], axis=1).astype(np.float64)
    headings = path_headings(np.vstack([past_xy, future_xy]), now=len(past_xy) - 1)
    past_head, future_head = headings[: len(past_xy)], headings[len(past_xy):]
    history = np.column_stack([past_xy[HISTORY_IDX], past_head[HISTORY_IDX]])
    target = np.column_stack([future_xy[FUTURE_IDX], future_head[FUTURE_IDX]])
    command = np.zeros(4, dtype=np.float32)
    command[INTENT_TO_COMMAND.get(int(frame.intent), 3)] = 1.0
    velocity = [past.vel_x[-1], past.vel_y[-1]] if len(past.vel_x) else list((past_xy[-1] - past_xy[-2]) * 4.0)
    accel = [past.accel_x[-1], past.accel_y[-1]] if len(past.accel_x) else [0.0, 0.0]
    status = np.concatenate([command, velocity, accel]).astype(np.float32)
    return history, target, command, status


def rater_labels(frame):
    """Rater trajectories (4 Hz from t = 0.25 s) and scores for the WOD-E2E Rater Feedback Score; None if unlabelled."""
    rated = [t for t in frame.preference_trajectories if t.preference_score != UNLABELLED and len(t.pos_x)]
    if not rated:
        return None
    past = frame.past_states
    return {
        "rater_trajectories": [np.stack([t.pos_x, t.pos_y], axis=1).astype(np.float64) for t in rated],
        "rater_scores": np.array([t.preference_score for t in rated], dtype=np.float64),
        "init_speed": float(np.hypot(past.vel_x[-1], past.vel_y[-1])) if len(past.vel_x) else 0.0,
    }


def center_crop(image: Image.Image, aspect: float = CAMERA_ASPECT) -> Image.Image:
    w, h = image.size
    cw, ch = min(w, round(h * aspect)), min(h, round(w / aspect))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


def convert(tfrecords, out: Path, split: str, stride: int = 1, all_views: bool = False) -> int:
    from navsim.planning.training.dataset import dump_feature_target_to_pickle as dump

    frame_class = e2ed_frame_class()
    image_dir = (out / "images" / split).resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record_path in tfrecords:
        for index, raw in enumerate(read_tfrecord(record_path)):
            if index % stride:
                continue
            frame = frame_class()
            frame.ParseFromString(raw)
            sample = frame_to_sample(frame)
            images = {CAMERAS[img.name]: img.image for img in frame.frame.images if img.name in CAMERAS}
            if sample is None or "front" not in images:
                continue
            history, target, command, status = sample
            token = re.sub(r"[^A-Za-z0-9_.-]", "_", frame.frame.context.name)  # "<segment>-<frame index>"
            camera_paths = []
            for view in CAMERAS.values():
                if view not in images or (view != "front" and not all_views):
                    continue
                path = image_dir / (f"{token}.jpg" if view == "front" else f"{token}_{view}.jpg")
                image = Image.open(io.BytesIO(images[view]))
                (center_crop(image) if view == "front" else image).save(path, quality=95)
                camera_paths.append((view, str(path)))
            sample_dir = out / split / token
            sample_dir.mkdir(parents=True, exist_ok=True)
            dump(sample_dir / "features.gz", {
                "history_trajectory": torch.tensor(history, dtype=torch.float32),
                "high_command_one_hot": torch.tensor(command),
                "status_feature": torch.tensor(status),
                "camera_paths": camera_paths,
            })
            dump(sample_dir / "targets.gz", {"trajectory": torch.tensor(target, dtype=torch.float32)})
            labels = rater_labels(frame)
            if labels is not None:
                dump(sample_dir / "eval.gz", labels)
            written += 1
    return written


def nonzero_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


class WaymoE2ECacheOnlyDataset(torch.utils.data.Dataset):
    """Samples from `<cache_path>/<split>/<token>/{features,targets}.gz` (this converter or RAP-style caches)."""

    def __init__(self, cache_path: str, split: str, future: int = 8, target_builders=None):
        split_path = Path(cache_path) / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Waymo E2E cache split {split_path} does not exist")
        self.future = future
        self.target_builders = target_builders
        self.tokens = sorted(
            path for path in split_path.iterdir()
            if path.is_dir() and nonzero_file(path / "features.gz") and nonzero_file(path / "targets.gz")
        )
        self.index = {path.name: path for path in self.tokens}

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int):
        from navsim.planning.training.dataset import load_feature_target_from_pickle, transform_targets_after_load

        path = self.tokens[index]
        features = load_feature_target_from_pickle(path / "features.gz")
        targets = load_feature_target_from_pickle(path / "targets.gz")
        trajectory = targets["trajectory"]
        if len(trajectory) < self.future:
            raise ValueError(f"{path} holds {len(trajectory)} target poses, the model predicts {self.future}; "
                             "rebuild the cache with this converter (5 s targets)")
        targets["trajectory"] = trajectory[: self.future]
        return features, transform_targets_after_load(targets, self.target_builders), path.name

    def eval_indices(self):
        """Frames with rater labels (one per WOD-E2E segment), the only ones the benchmark scores."""
        return [i for i, path in enumerate(self.tokens) if nonzero_file(path / "eval.gz")]

    def eval_info(self, token: str) -> dict:
        """Rater trajectories, scores and initial speed of a labelled frame (empty for unlabelled ones)."""
        from navsim.planning.training.dataset import load_feature_target_from_pickle

        path = self.index[token] / "eval.gz"
        return load_feature_target_from_pickle(path) if nonzero_file(path) else {}


@register("waymoe2e")
def waymoe2e_loader(cfg, agent, split: str):
    if cfg.cache_path is None:
        raise AssertionError("cache_path must be provided when using cached data")
    default = "training" if split == "train" else "val"
    key = "waymoe2e_train_split" if split == "train" else "waymoe2e_val_split"
    return WaymoE2ECacheOnlyDataset(
        cache_path=cfg.cache_path,
        split=getattr(cfg, key, default),
        future=horizon(agent),
        target_builders=agent.get_target_builders(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WOD-E2E TFRecords -> recogdrive waymoe2e cache")
    parser.add_argument("tfrecords", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", required=True, help="cache split folder, e.g. training or val")
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    parser.add_argument("--views", choices=["front", "all"], default="front", help="all: the 8 cameras, for multi-view")
    args = parser.parse_args()
    written = convert(args.tfrecords, args.out, args.split, args.stride, args.views == "all")
    print(f"wrote {written} samples to {args.out / args.split}")
