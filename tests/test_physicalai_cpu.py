"""PhysicalAI-AV converter regression on synthetic Hugging Face layout data.

This checks conversion, loading, and open-loop metrics with generated fixtures; it is not a benchmark score.
"""
import importlib.util
import io
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pd = pytest.importorskip("pandas")
pytest.importorskip("decord")
pytest.importorskip("PIL")
if not any(importlib.util.find_spec(engine) for engine in ("pyarrow", "fastparquet")):
    pytest.skip("PhysicalAI synthetic parquet test needs pyarrow or fastparquet", allow_module_level=True)

from omegaconf import OmegaConf

from recogdrive.data.physicalai import convert, physicalai_loader
from recogdrive.data.nuscenes import to_local
from recogdrive.eval.openloop import displacement_metrics

SPEED, YAW_RATE, FPS, SECONDS = 8.0, 0.1, 30, 20
CLIPS = {"11111111-aaaa-4bbb-8ccc-000000000001": "train",
         "11111111-aaaa-4bbb-8ccc-000000000002": "val"}
CAMS = ("camera_front_wide_120fov", "camera_cross_left_120fov", "camera_front_tele_30fov")
FRAME_OFFSET_US = 1_000


def arc(t):
    return np.stack([SPEED / YAW_RATE * np.sin(YAW_RATE * t),
                     SPEED / YAW_RATE * (1 - np.cos(YAW_RATE * t)), YAW_RATE * t], axis=1)


def parquet_bytes(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


def build(src: Path):
    index = pd.DataFrame({"chunk": [0] * len(CLIPS), "split": list(CLIPS.values())},
                         index=pd.Index(list(CLIPS), name="clip_id"))
    index.to_parquet(src / "clip_index.parquet")
    (src / "labels" / "egomotion").mkdir(parents=True)
    t = np.arange(-2.0, 26.0, 0.01)
    p = arc(t)
    ego = pd.DataFrame({"timestamp": np.round(t * 1e6).astype(np.int64), "qx": 0.0, "qy": 0.0,
                        "qz": np.sin(p[:, 2] / 2), "qw": np.cos(p[:, 2] / 2), "x": p[:, 0], "y": p[:, 1],
                        "z": 0.0, "vx": SPEED * np.cos(p[:, 2]), "vy": SPEED * np.sin(p[:, 2]), "vz": 0.0,
                        "ax": -SPEED * YAW_RATE * np.sin(p[:, 2]),
                        "ay": SPEED * YAW_RATE * np.cos(p[:, 2]), "az": 0.0,
                        "curvature": YAW_RATE / SPEED})
    with zipfile.ZipFile(src / "labels" / "egomotion" / "egomotion.chunk_0000.zip", "w") as zf:
        for clip in CLIPS:
            zf.writestr(f"{clip}.egomotion.parquet", parquet_bytes(ego))
    frames = FPS * SECONDS
    with tempfile.TemporaryDirectory(prefix="recogdrive-physicalai-video-") as tmp:
        video = Path(tmp) / "v.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 96))
        for i in range(frames):
            writer.write(np.full((96, 160, 3), i % 250, dtype=np.uint8))
        writer.release()
        data = video.read_bytes()
    stamps = pd.DataFrame({"frame_index": np.arange(frames),
                           "timestamp": np.round(np.arange(frames) * 1e6 / FPS).astype(np.int64)
                           + FRAME_OFFSET_US})
    for cam in CAMS:
        (src / "camera" / cam).mkdir(parents=True)
        with zipfile.ZipFile(src / "camera" / cam / f"{cam}.chunk_0000.zip", "w") as zf:
            for clip in CLIPS:
                zf.writestr(f"{clip}.{cam}.mp4", data)
                zf.writestr(f"{clip}.{cam}.timestamps.parquet", parquet_bytes(stamps))


def test_physicalai_synthetic_converter_and_metrics_cpu():
    with tempfile.TemporaryDirectory(prefix="recogdrive-physicalai-") as temp:
        root = Path(temp)
        src, out = root / "src", root / "cache"
        src.mkdir()
        build(src)
        written = convert(src, out, all_views=True)
        keyframes = int(np.floor((FPS * SECONDS - 1) / FPS / 0.5)) + 1
        assert written == keyframes * len(CLIPS), (written, keyframes)
        agent = SimpleNamespace(trajectory_sampling=SimpleNamespace(num_poses=8, interval_length=0.5),
                                get_target_builders=lambda: [])
        cfg = OmegaConf.create({"physicalai": {"root": str(out)}})
        train, val = physicalai_loader(cfg, agent, "train"), physicalai_loader(cfg, agent, "val")
        assert len(train) == len(val) == keyframes
        worst = 0.0
        for index in range(len(val)):
            features, targets, token = val[index]
            now = int(token.rsplit("-", 1)[1]) * 1e-3
            truth = to_local(arc(now + 0.5 * np.arange(-3, 11)), arc(np.array([now]))[0])
            worst = max(worst, np.abs(features["history_trajectory"].numpy() - truth[:4]).max(),
                        np.abs(targets["trajectory"].numpy() - truth[4:12]).max())
            status = features["status_feature"].numpy()
            assert status[:4].tolist() == [1, 0, 0, 0], status  # left arc: 6.3 m lateral at 4 s
            assert np.allclose(status[4:], [SPEED, 0.0, 0.0, SPEED * YAW_RATE], atol=1e-3), status
            views = dict(features["camera_paths"])
            assert set(views) == {"front", "left"}, views
            frame = int(np.abs(np.arange(FPS * SECONDS) * 1e6 / FPS + FRAME_OFFSET_US
                               - now * 1e6).argmin())
            value = float(np.asarray(cv2.imread(views["front"])).mean())
            assert abs(value - frame % 250) < 4, (token, value, frame)
        assert worst < 1e-3, worst
        assert len(val.eval_indices()) == len(val) and val.eval_info(token)["trajectory"].shape == (10, 3)
        gt = val.eval_info(token)["trajectory"].numpy()
        exact = displacement_metrics([(gt[:8], gt)])
        shifted = displacement_metrics([(gt[:8] + [0.0, 1.0, 0.0], gt)])
        assert exact["ade"] == 0 and abs(shifted["L2_3s"] - 1) < 1e-6 \
            and abs(shifted["fde"] - 1) < 1e-6, shifted
