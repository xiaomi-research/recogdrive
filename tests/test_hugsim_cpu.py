"""HUGSIM FIFO protocol regression with a template policy and nuScenes mini scene.

The mock client exercises HUGSIM's message and coordinate transforms; it is not a simulator score.
"""
import errno
import os
import pickle
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIFO_TIMEOUT_S = 30.0


def _check_agent(process, deadline):
    if process.poll() is not None:
        raise RuntimeError("HUGSIM FIFO agent exited before completing the protocol")
    if time.monotonic() >= deadline:
        raise TimeoutError("HUGSIM FIFO protocol timed out")


def _fifo_write(path, value, process):
    deadline = time.monotonic() + _FIFO_TIMEOUT_S
    while True:
        _check_agent(process, deadline)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            break
        except OSError as exc:
            if exc.errno not in (errno.ENXIO, errno.ENOENT):
                raise
            time.sleep(0.01)
    try:
        data = memoryview(pickle.dumps(value))
        while data:
            _check_agent(process, deadline)
            try:
                written = os.write(fd, data)
                data = data[written:]
            except BlockingIOError:
                select.select([], [fd], [], min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        os.close(fd)


def _fifo_read(path, process):
    deadline = time.monotonic() + _FIFO_TIMEOUT_S
    while True:
        _check_agent(process, deadline)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            break
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise
            time.sleep(0.01)
    try:
        data = bytearray()
        while True:
            _check_agent(process, deadline)
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                chunk = None
            if chunk:
                data.extend(chunk)
            elif data:
                return pickle.loads(data)
            else:
                time.sleep(0.01)  # a FIFO with no writer reports EOF until the agent opens it
    finally:
        os.close(fd)


def test_hugsim_template_fifo_protocol_cpu():
    if not hasattr(os, "mkfifo"):
        pytest.skip("HUGSIM's FIFO protocol requires POSIX named pipes")
    pytest.importorskip("nuplan.planning.simulation.trajectory.trajectory_sampling")
    pytest.importorskip("PIL")

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from PIL import Image

    from recogdrive.adapters.template import EgoMLPPolicy
    from recogdrive.closedloop import UNIAD_COMMANDS, Driver, hugsim
    from recogdrive.data.nuscenes import CAMERAS, NuScenesDataset

    root = Path(os.environ.get("NUSCENES_ROOT", "/data/yongkangli/datasets/nuscenes"))
    if not (root / "v1.0-mini").is_dir():
        pytest.skip("set NUSCENES_ROOT to a local nuScenes mini dataset")

    dataset = NuScenesDataset(root, "v1.0-mini", "val", 8, [])
    tokens, cameras, pose, stamps = dataset.tracks[0]
    times = np.arange(0.0, stamps[-1] - stamps[0], 0.25)
    t0 = stamps[0]
    xs, ys = np.interp(times + t0, stamps, pose[:, 0]), np.interp(times + t0, stamps, pose[:, 1])
    yaws = np.interp(times + t0, stamps, np.unwrap(pose[:, 2]))
    with Image.open(dict(cameras[0])["front"]) as image:
        front = np.array(image.resize((800, 450)))

    torch.manual_seed(2026)
    policy = EgoMLPPolicy(TrajectorySampling(num_poses=8, interval_length=0.5)).eval()
    server_code = """
import sys
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf
from recogdrive.adapters.template import EgoMLPPolicy
from recogdrive.closedloop import Driver
from recogdrive.closedloop.hugsim import serve
torch.manual_seed(2026)
policy = EgoMLPPolicy(TrajectorySampling(num_poses=8, interval_length=0.5)).eval()
serve(Driver(policy, "cpu"), 0, OmegaConf.create({"output_dir": sys.argv[1]}))
"""

    with tempfile.TemporaryDirectory(prefix="recogdrive-hugsim-") as temp:
        directory = Path(temp)
        output_dir = directory / "agent-output"
        output_dir.mkdir()
        log_path = directory / "agent.log"
        process = None
        with log_path.open("wb") as log_file:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-c", server_code, str(output_dir)],
                    cwd=str(_REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                obs_pipe, plan_pipe = output_dir / "obs_pipe", output_dir / "plan_pipe"
                reference = Driver(policy, "cpu")
                worst = 0.0
                black = np.zeros_like(front)
                for k, now in enumerate(times):
                    rgb = {cam: (front if view == "front" else black) for cam, view in CAMERAS.items()}
                    info = {"ego_box": [xs[k], ys[k], 0.0, 1.6, 3.0, 1.5, yaws[k]],
                            "ego_velo": 4.0 + 0.1 * k, "accelerate": 0.1, "timestamp": float(now),
                            "command": k % 3, "ego_steer": 0.0}
                    _fifo_write(obs_pipe, ({"rgb": rgb}, info), process)
                    plan = _fifo_read(plan_pipe, process)
                    imu = plan[:, [1, 0]].copy()  # HUGSIM lidar (x right, y forward) -> IMU (x forward, y left)
                    imu[:, 1] *= -1
                    reference.observe(now, xs[k], ys[k], yaws[k])
                    expected = reference.plan(now, [("front", None)], UNIAD_COMMANDS[k % 3],
                                              (4.0 + 0.1 * k, 0.0), (0.1, 0.0))
                    assert plan.shape == (8, 2)
                    worst = max(worst, float(np.abs(imu - expected[:, :2]).max()))

                _fifo_write(obs_pipe, "Done", process)
                process.wait(timeout=_FIFO_TIMEOUT_S)
                assert process.returncode == 0
                assert worst < 1e-9, worst

                images = []
                driver = Driver(policy, "cpu")
                driver.plan = lambda now, views, *args: images.extend(views) or np.zeros((8, 3))
                hugsim.plan(driver, {"rgb": rgb}, info)
                assert [view for view, _ in images] == ["front"], images
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
