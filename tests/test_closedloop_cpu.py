"""Closed-loop layer checks on nuScenes mini (CPU): Driver features equal the nuscenes data source's, NeuroNCAP server
round trip with NeuRAD-encoded images, in-memory images give the same InternVL tiles as their files."""
import base64
import io
import json
import os
import errno
import pickle
import socket
import threading
import sys
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf
from PIL import Image

from recogdrive.adapters.template import EgoMLPPolicy
from recogdrive.closedloop import COMMANDS, Driver
from recogdrive.closedloop import neuroncap
from recogdrive.data.nuscenes import CAMERAS, NuScenesDataset
from recogdrive.eval.registry import SubprocessEvaluator

ROOT = os.environ.get("NUSCENES_ROOT", "/data/yongkangli/datasets/nuscenes")


def check_features(dataset, driver):
    worst = 0.0
    for index in range(0, len(dataset), 7):
        features, _, token = dataset[index]
        track, i = dataset.items[index]
        _, cameras, pose, _ = dataset.tracks[track]
        driver.reset()
        for k in range(i + 1):
            driver.observe(0.5 * k, *pose[k])
        mine = driver.features(0.5 * i, cameras[i], int(features["high_command_one_hot"].argmax()))
        for key in ("history_trajectory", "high_command_one_hot"):
            worst = max(worst, float((mine[key] - features[key]).abs().max()))
        assert mine["camera_paths"] == features["camera_paths"]
    # status: the loader differentiates over real keyframe stamps, the driver over 0.5 s; compare with equal stamps
    print(f"history / command equal the nuscenes loader's (max diff {worst:.2e})", flush=True)
    assert worst < 1e-4
    driver.reset()
    driver.observe(10.0, 1.0, 2.0, np.pi / 2)
    history = driver.history(10.0, speed=4.0)
    assert np.allclose(history[:, 0], 1.0) and np.allclose(history[:, 1], [-4.0, -2.0, 0.0, 2.0]), history
    print("one observed pose: history continues it at the given speed", flush=True)


def encode(array):  # neurad-studio nerfstudio/scripts/closed_loop/main.py _torch_to_bytestr, as NeuroNCAP forwards it
    buffer = io.BytesIO()
    torch.save(torch.from_numpy(array), buffer)
    return base64.b64encode(buffer.getvalue()).decode()


def post(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                     method="GET" if data is None else "POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def check_neuroncap_requests(dataset, driver, url):
    for _ in range(50):
        try:
            assert post(f"{url}/alive") is True
            break
        except OSError:
            time.sleep(0.1)
    track, _ = dataset.items[0]
    tokens, cameras, pose, stamps = dataset.tracks[track]
    assert post(f"{url}/reset", {}) is True
    reference = Driver(driver.agent, "cpu")
    for k in range(6):
        arrays = {cam: np.array(Image.open(dict(cameras[k])[view])) for cam, view in CAMERAS.items()}
        ego2world = np.eye(4)
        c, s = np.cos(pose[k, 2]), np.sin(pose[k, 2])
        ego2world[:2, :2] = [[c, -s], [s, c]]
        ego2world[:2, 3] = pose[k, :2]
        canbus = np.zeros(16)
        canbus[13], canbus[7] = 5.0, 0.3
        request = {"images": {cam: encode(a) for cam, a in arrays.items()}, "ego2world": ego2world.tolist(),
                   "canbus": canbus.tolist(), "timestamp": int(stamps[k] * 1e6), "command": 1,
                   "calibration": {"camera2image": {}, "camera2ego": {}, "lidar2ego": np.eye(4).tolist()}}
        response = post(f"{url}/infer", request)
        decoded = neuroncap.decode_image(request["images"]["CAM_FRONT"])
        assert np.array_equal(np.array(decoded), arrays["CAM_FRONT"])
        reference.observe(int(stamps[k] * 1e6) * 1e-6, *pose[k])
        images = [(view, Image.fromarray(arrays[cam])) for cam, view in CAMERAS.items()]
        expected = reference.plan(int(stamps[k] * 1e6) * 1e-6, images, COMMANDS.index("left"), canbus[13:15], canbus[7:9])
        assert len(response["trajectory"]) == 6 and response["aux_outputs"] == {}
        assert np.allclose(response["trajectory"], expected[:6, :2], atol=1e-6), (response["trajectory"], expected)
    print(f"NeuroNCAP server: 6 steps of {tokens[0][:8]}.. round trip, 6 (x, y) per step, same as the driver", flush=True)


def check_neuroncap(dataset, driver):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    servers = []
    make_server = neuroncap.HTTPServer

    def track_server(*args, **kwargs):
        server = make_server(*args, **kwargs)
        servers.append(server)
        return server

    thread = threading.Thread(target=neuroncap.serve, args=(driver, port, OmegaConf.create({})), daemon=True)
    with patch.object(neuroncap, "HTTPServer", track_server):
        thread.start()
        try:
            check_neuroncap_requests(dataset, driver, f"http://127.0.0.1:{port}")
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            thread.join(timeout=10)
            assert not thread.is_alive(), "NeuroNCAP service thread did not stop"


def check_tiles(dataset):
    from recogdrive.adapters.navsim.agent import prepare_vlm_inputs

    features, _, _ = dataset[0]
    in_memory = dict(features, camera_paths=[(v, Image.open(p)) for v, p in features["camera_paths"]])
    a = prepare_vlm_inputs(dict(features), load_tiles=True)
    b = prepare_vlm_inputs(in_memory, load_tiles=True)
    assert torch.equal(a["pixel_values"], b["pixel_values"]) and a["vlm_question"] == b["vlm_question"]
    a = prepare_vlm_inputs(dict(features), load_tiles=True, cameras=list(CAMERAS.values()), short_sides=(960, 480))
    b = prepare_vlm_inputs(dict(in_memory), load_tiles=True, cameras=list(CAMERAS.values()), short_sides=(960, 480))
    assert torch.equal(a["pixel_values"], b["pixel_values"]) and torch.equal(a["num_patches"], b["num_patches"])
    print(f"in-memory images: same InternVL tiles as their files (front {tuple(a['pixel_values'].shape)})", flush=True)


def main():
    policy = EgoMLPPolicy(TrajectorySampling(num_poses=8, interval_length=0.5)).eval()
    dataset = NuScenesDataset(ROOT, "v1.0-mini", "val", 8, [])
    driver = Driver(policy, "cpu")
    check_features(dataset, driver)
    check_neuroncap(dataset, driver)
    check_tiles(dataset)


def test_qwen_accepts_one_rendered_pil_image(monkeypatch):
    from types import SimpleNamespace

    from transformers.feature_extraction_utils import BatchFeature

    from recogdrive.models.recogdrive import backbone

    rendered = Image.new("RGB", (32, 18), "white")

    def process(messages):
        assert messages[1]["content"][0]["image"] is rendered
        return [rendered], None

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "question"

        def __call__(self, *, images, **kwargs):
            assert images == [rendered]
            return BatchFeature({"input_ids": torch.ones(1, 4, dtype=torch.long)})

    hidden = torch.zeros(1, 4, 8)
    policy = SimpleNamespace(device="cpu", tokenizer=Processor(), system_prompt=backbone.system_message,
                             qwen_max_input_tokens=None,
                             model=lambda **kwargs: SimpleNamespace(hidden_states=[hidden]))
    monkeypatch.setattr(backbone, "process_vision_info", process)
    assert backbone.RecogDriveBackbone.forward_qwen(policy, [rendered], ["Q?"]) is hidden


@pytest.mark.skipif(os.name != "posix", reason="HUGSIM uses POSIX FIFOs")
@pytest.mark.parametrize("simulator_reader", [True, False])
def test_hugsim_policy_failure_notifies_reader_and_never_blocks(tmp_path, simulator_reader):
    obs_pipe, plan_pipe = tmp_path / "obs_pipe", tmp_path / "plan_pipe"
    os.mkfifo(obs_pipe)
    os.mkfifo(plan_pipe)
    reader = os.open(plan_pipe, os.O_RDONLY | os.O_NONBLOCK) if simulator_reader else None
    code = (
        "import sys\nfrom types import SimpleNamespace\nfrom recogdrive.closedloop import hugsim\n"
        "def fail(*args): raise RuntimeError('expected policy failure')\n"
        "hugsim.plan=fail\nhugsim.serve(object(),0,SimpleNamespace(output_dir=sys.argv[1]))\n"
    )
    evaluator = SubprocessEvaluator(OmegaConf.create({}), OmegaConf.create({}), tmp_path)
    try:
        evaluator.launch([sys.executable, "-c", code, str(tmp_path)], tmp_path, "client", os.environ.copy())
        process = evaluator.jobs[0][1]
        deadline = time.monotonic() + 20
        while True:
            try:
                writer = os.open(obs_pipe, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as exc:
                assert exc.errno == errno.ENXIO
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.02)
        with os.fdopen(writer, "wb") as stream:
            pickle.dump(({}, {}), stream)
        assert process.wait(timeout=20) != 0
        assert "expected policy failure" in (tmp_path / "eval.log").read_text()
        if reader is not None:
            assert pickle.loads(os.read(reader, 4096)) is None
    finally:
        if reader is not None:
            os.close(reader)
        evaluator.cancel()


@pytest.mark.skipif(os.name != "posix", reason="HUGSIM uses POSIX FIFOs")
def test_hugsim_refuses_existing_regular_file_without_overwriting(tmp_path):
    from recogdrive.closedloop import hugsim

    existing = tmp_path / "obs_pipe"
    existing.write_bytes(b"preserve this file")
    with pytest.raises(ValueError, match="not a FIFO"):
        hugsim.serve(object(), 0, type("Config", (), {"output_dir": str(tmp_path)})())
    assert existing.read_bytes() == b"preserve this file"


def test_closedloop_cpu_on_nuscenes_mini():
    if not (Path(ROOT) / "v1.0-mini").is_dir():
        pytest.skip("set NUSCENES_ROOT to a local nuScenes mini dataset")
    main()


if __name__ == "__main__":
    main()
