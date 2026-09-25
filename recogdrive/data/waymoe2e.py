"""Waymo Open Dataset End-to-End Driving (WOD-E2E).

Training reads the cache layout `<cache_path>/<split>/<token>/{features,targets}.gz`.
`python -m recogdrive.data.waymoe2e --out <cache_path> --split training <tfrecords...>` builds it from
the raw TFRecords: front camera JPEG to disk, 4 Hz ego states resampled to NAVSIM's 0.5 s spacing
(history t = -1.5..0 s, target t = 0.5..4 s) in the current ego frame (x forward, y left).
"""

import argparse
import io
import re
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from recogdrive.data.registry import register

FRONT_CAMERA = 1
CAMERA_ASPECT = 16 / 9  # NAVSIM camera shape: tiles into 9 patches, which the VLM prompt budget (2800 tokens) is sized for
HISTORY_IDX = [-7, -5, -3, -1]           # past states: 16 at 4 Hz, t = -3.75 .. 0 s
FUTURE_IDX = list(range(1, 16, 2))        # future states: 20 at 4 Hz, t = 0.25 .. 5 s -> 0.5 .. 4 s
INTENT_TO_COMMAND = {2: 0, 1: 1, 3: 2, 0: 3}  # WOD LEFT/STRAIGHT/RIGHT/UNKNOWN -> NAVSIM left, straight, right, unknown
STILL_M = 0.05


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
        ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "accel_x", "accel_y"], start=1)])
    message("E2EDFrame", [("frame", 1, F.TYPE_MESSAGE, False, "Frame"),
                          ("future_states", 5, F.TYPE_MESSAGE, False, "EgoTrajectoryStates"),
                          ("past_states", 6, F.TYPE_MESSAGE, False, "EgoTrajectoryStates"),
                          ("intent", 7, F.TYPE_INT32, False, None)])
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
    step[0], step[-1] = xy[1] - xy[0], xy[-1] - xy[-2]
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
    if len(past.pos_x) < 16 or len(future.pos_x) < 16:
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
    image = next((img.image for img in frame.frame.images if img.name == FRONT_CAMERA), None)
    return history, target, command, status, image


def center_crop(image: Image.Image, aspect: float = CAMERA_ASPECT) -> Image.Image:
    w, h = image.size
    cw, ch = min(w, round(h * aspect)), min(h, round(w / aspect))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


def convert(tfrecords, out: Path, split: str, stride: int = 1) -> int:
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
            if sample is None or sample[-1] is None:
                continue
            history, target, command, status, image = sample
            token = re.sub(r"[^A-Za-z0-9_.-]", "_", frame.frame.context.name)  # "<segment>-<frame index>"
            image_path = image_dir / f"{token}.jpg"
            center_crop(Image.open(io.BytesIO(image))).save(image_path, quality=95)
            sample_dir = out / split / token
            sample_dir.mkdir(parents=True, exist_ok=True)
            dump(sample_dir / "features.gz", {
                "history_trajectory": torch.tensor(history, dtype=torch.float32),
                "high_command_one_hot": torch.tensor(command),
                "status_feature": torch.tensor(status),
                "image_path_tensor": torch.tensor([ord(c) for c in str(image_path)], dtype=torch.long),
            })
            dump(sample_dir / "targets.gz", {"trajectory": torch.tensor(target, dtype=torch.float32)})
            written += 1
    return written


def nonzero_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


class WaymoE2ECacheOnlyDataset(torch.utils.data.Dataset):
    """Samples from `<cache_path>/<split>/<token>/{features,targets}.gz` (this converter or RAP-style caches)."""

    def __init__(self, cache_path: str, split: str, target_builders=None):
        split_path = Path(cache_path) / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Waymo E2E cache split {split_path} does not exist")
        self.target_builders = target_builders
        self.tokens = sorted(
            path for path in split_path.iterdir()
            if path.is_dir() and nonzero_file(path / "features.gz") and nonzero_file(path / "targets.gz")
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int):
        from navsim.planning.training.dataset import load_feature_target_from_pickle, transform_targets_after_load

        path = self.tokens[index]
        features = load_feature_target_from_pickle(path / "features.gz")
        targets = transform_targets_after_load(load_feature_target_from_pickle(path / "targets.gz"), self.target_builders)
        return features, targets, path.name


@register("waymoe2e")
def waymoe2e_loader(cfg, agent, split: str):
    if cfg.cache_path is None:
        raise AssertionError("cache_path must be provided when using cached data")
    default = "training" if split == "train" else "val"
    key = "waymoe2e_train_split" if split == "train" else "waymoe2e_val_split"
    return WaymoE2ECacheOnlyDataset(
        cache_path=cfg.cache_path,
        split=getattr(cfg, key, default),
        target_builders=agent.get_target_builders(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WOD-E2E TFRecords -> recogdrive waymoe2e cache")
    parser.add_argument("tfrecords", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", required=True, help="cache split folder, e.g. training or val")
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    args = parser.parse_args()
    print(f"wrote {convert(args.tfrecords, args.out, args.split, args.stride)} samples to {args.out / args.split}")
