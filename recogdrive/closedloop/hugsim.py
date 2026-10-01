"""HUGSIM (Zhou et al., github.com/hyzhou404/HUGSIM) agent. HUGSIM's closed_loop.py starts the agent once per episode
(scripts/infer/run_recogdrive_hugsim.sh in place of its LTF / UniAD / VAD launcher) and exchanges pickles with it over
two FIFOs in the episode's output directory (`output_dir`): obs_pipe carries (obs, info) every 0.25 s and 'Done' at
the end, plan_pipe takes the plan. Like HUGSIM's own clients, the agent unpickles what its local simulator writes.

obs["rgb"][CAM_*] are (450, 800, 3) RGB renders of the six nuScenes cameras (black back views on Waymo and KITTI-360,
which are dropped). info["ego_box"] is the ego (x, y, z, w, l, h, yaw) in the global frame HUGSIM scores in (x forward,
y left at yaw 0), info["ego_velo"] its speed, info["accelerate"] the last longitudinal acceleration and info["command"]
0 right / 1 left / 2 straight. The plan is the predicted (x, y) every 0.5 s in HUGSIM's lidar frame (x right,
y forward), as its LTF client sends NAVSIM trajectories; its iLQR controller tracks it and HD-Score rates it.
"""

import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from recogdrive.closedloop import UNIAD_COMMANDS
from recogdrive.data.nuscenes import CAMERAS


def plan(driver, obs: dict, info: dict) -> np.ndarray:
    x, y, _, _, _, _, yaw = info["ego_box"]
    now = float(info["timestamp"])
    driver.observe(now, x, y, yaw)
    rgb = obs["rgb"]
    images = [(view, Image.fromarray(rgb[cam])) for cam, view in CAMERAS.items() if cam in rgb and rgb[cam].any()]
    poses = driver.plan(now, images, UNIAD_COMMANDS[int(info["command"])],
                        (float(info["ego_velo"]), 0.0), (float(info["accelerate"]), 0.0))
    return np.stack([-poses[:, 1], poses[:, 0]], axis=1)


def serve(driver, port: int, cfg) -> None:
    directory = Path(str(cfg.output_dir))
    directory.mkdir(parents=True, exist_ok=True)
    obs_pipe, plan_pipe = directory / "obs_pipe", directory / "plan_pipe"
    for pipe in (obs_pipe, plan_pipe):
        try:
            os.mkfifo(pipe)
        except FileExistsError:  # the simulator made it first
            pass
    while True:
        with open(obs_pipe, "rb") as f:
            message = pickle.loads(f.read())
        if isinstance(message, str):  # 'Done'
            return
        with open(plan_pipe, "wb") as f:
            f.write(pickle.dumps(plan(driver, *message)))
