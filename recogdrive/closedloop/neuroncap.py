"""NeuroNCAP (Ljungbergh et al., ECCV 2024) model server: the HTTP API of its reference UniAD server
(github.com/wljungbergh/UniAD, inference/server.py), which neuro-ncap's ModelAPI calls every 0.5 s.

    GET  /alive
    POST /reset   a new episode
    POST /infer   {images: {CAM_*: base64 of a torch.save'd uint8 (h, w, 3) RGB tensor}, ego2world: 4x4,
                   canbus: 16 (nuScenes CAN bus, ego frame: acceleration 7:10, velocity 13:16), timestamp: us,
                   command: 0 right / 1 left / 2 straight, calibration}
               -> {trajectory: 6 (x, y) at t + 0.5 .. 3.0 s in the rear-axle frame (x forward, y left), aux_outputs: {}}

The engine tracks the points with its LQR controller and takes headings from consecutive points. The NCAP score and
collision rate do not read aux outputs (detections, used only for recall diagnostics), so none are sent.
"""

import base64
import io
import json
import logging
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from recogdrive.closedloop import UNIAD_COMMANDS
from recogdrive.data.nuscenes import CAMERAS

logger = logging.getLogger(__name__)

DEFAULT_PORT = 9000
POINTS = 6


def decode_image(text: str) -> Image.Image:
    return Image.fromarray(torch.load(io.BytesIO(base64.b64decode(text)), weights_only=True).numpy())


def infer(driver, request: dict) -> dict:
    pose = np.asarray(request["ego2world"], dtype=np.float64)
    now = int(request["timestamp"]) * 1e-6
    driver.observe(now, pose[0, 3], pose[1, 3], math.atan2(pose[1, 0], pose[0, 0]))
    canbus = request["canbus"]
    images = [(view, decode_image(request["images"][cam])) for cam, view in CAMERAS.items() if cam in request["images"]]
    poses = driver.plan(now, images, UNIAD_COMMANDS[int(request["command"])], canbus[13:15], canbus[7:9])
    return {"trajectory": poses[:POINTS, :2].tolist(), "aux_outputs": {}}


def serve(driver, port: int, cfg) -> None:
    if driver.agent.trajectory_sampling.num_poses < POINTS:
        raise ValueError(f"NeuroNCAP tracks {POINTS} poses (3 s), the model predicts "
                         f"{driver.agent.trajectory_sampling.num_poses}")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/alive":
                self.reply(True)
            else:
                self.send_error(404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/reset":
                driver.reset()
                self.reply(True)
            elif self.path == "/infer":
                self.reply(infer(driver, json.loads(body)))
            else:
                self.send_error(404)

    host = str(OmegaConf.select(cfg, "infer.host") or "127.0.0.1")
    logger.info("NeuroNCAP model server on %s:%d", host, port or DEFAULT_PORT)
    HTTPServer((host, port or DEFAULT_PORT), Handler).serve_forever()
