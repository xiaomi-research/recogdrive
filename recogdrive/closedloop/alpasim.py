"""AlpaSim (NVIDIA, github.com/NVlabs/alpasim) egodriver: the gRPC service egodriver.EgodriverService through which the
AlpaSim runtime queries a policy. The stubs are the repo's src/grpc package (alpasim_grpc).

Per session the runtime submits camera JPEGs (PhysicalAI-AV logical ids such as camera_front_wide_120fov), the
estimated ego poses (active local -> rig transforms at absolute microsecond timestamps; the rig is the rear axle on
the ground, x forward, y left) with rig-frame velocities and accelerations, and the route (rig-frame waypoints), then
calls drive every control step (0.1 s). drive returns the current pose followed by the policy's poses, 0.5 s apart,
as local -> rig transforms at their absolute timestamps, which the runtime's MPC tracks. The route has no turn
signal, so the command is its turn at the distance the ego covers over the horizon (at least 10 m), by the nuScenes
rule. Concurrent rollouts (the challenge runs 2 per container) keep separate histories and share the model.

ALPASIM_DRIVER_HOST / ALPASIM_DRIVER_PORT, the challenge container's variables, override infer.host / infer.port.
"""

import io
import logging
import math
import os
import threading
from concurrent import futures

import grpc
import numpy as np
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
from omegaconf import OmegaConf
from PIL import Image

from recogdrive.closedloop import COMMANDS, Driver, to_world
from recogdrive.data.nuscenes import command_one_hot, quaternion_yaw
from recogdrive.data.physicalai import CAMERAS

logger = logging.getLogger(__name__)

DEFAULT_PORT = 6789
POSE_INTERVAL_US = 500_000
MIN_LOOKAHEAD_M = 10.0


def route_command(waypoints: np.ndarray, distance: float) -> int:
    """Index into COMMANDS: how the route turns `distance` metres (arc length) ahead of the ego."""
    ahead = waypoints[waypoints[:, 0] > 0.0]
    if not len(ahead):
        return COMMANDS.index("straight")
    arc = np.cumsum(np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), ahead]), axis=0), axis=1))
    return int(np.argmax(command_one_hot(ahead[min(int(np.searchsorted(arc, distance)), len(ahead) - 1)][None])))


def pose_at(x: float, y: float, z: float, yaw: float, timestamp_us: int):
    return common_pb2.PoseAtTime(
        pose=common_pb2.Pose(vec=common_pb2.Vec3(x=x, y=y, z=z),
                             quat=common_pb2.Quat(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))),
        timestamp_us=timestamp_us)


class Session:
    def __init__(self, driver: Driver):
        self.driver = driver
        self.images = {}  # view -> latest JPEG bytes
        self.route = np.zeros((0, 2))
        self.pose = None  # latest common_pb2.PoseAtTime
        self.state = None  # its common_pb2.DynamicState


class EgoDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self, driver: Driver):
        self.driver = driver
        self.sessions = {}
        self.unknown = set()
        # ponytail: one forward at a time across rollouts; batching the concurrent drive calls is the upgrade
        self.lock = threading.Lock()
        sampling = driver.agent.trajectory_sampling
        self.horizon_s = float(sampling.num_poses * sampling.interval_length)

    def start_session(self, request, context):
        self.sessions[request.session_uuid] = Session(Driver(self.driver.agent, self.driver.device))
        return common_pb2.SessionRequestStatus()

    def close_session(self, request, context):
        self.sessions.pop(request.session_uuid, None)
        return common_pb2.Empty()

    def submit_image_observation(self, request, context):
        image = request.camera_image
        view = CAMERAS.get(image.logical_id)
        if view is not None:
            self.sessions[request.session_uuid].images[view] = image.image_bytes
        elif image.logical_id not in self.unknown:
            self.unknown.add(image.logical_id)
            logger.warning("camera %s has no view (recogdrive.data.physicalai.CAMERAS); its images are ignored",
                           image.logical_id)
        return common_pb2.Empty()

    def submit_egomotion_observation(self, request, context):
        session = self.sessions[request.session_uuid]
        for at in request.trajectory.poses:
            q = at.pose.quat
            session.driver.observe(at.timestamp_us * 1e-6, at.pose.vec.x, at.pose.vec.y,
                                   quaternion_yaw((q.w, q.x, q.y, q.z)))
            session.pose = at
        if request.dynamic_states:
            session.state = request.dynamic_states[-1]
        return common_pb2.Empty()

    def submit_route(self, request, context):
        self.sessions[request.session_uuid].route = np.array(
            [[w.x, w.y] for w in request.route.waypoints], dtype=np.float64).reshape(-1, 2)
        return common_pb2.Empty()

    def submit_recording_ground_truth(self, request, context):
        return common_pb2.Empty()

    def drive(self, request, context):
        session = self.sessions[request.session_uuid]
        if session.pose is None or "front" not in session.images:
            return egodriver_pb2.DriveResponse()
        velocity = acceleration = None
        if session.state is not None:
            velocity = (session.state.linear_velocity.x, session.state.linear_velocity.y)
            acceleration = (session.state.linear_acceleration.x, session.state.linear_acceleration.y)
        speed = 0.0 if velocity is None else math.hypot(*velocity)
        command = route_command(session.route, max(speed * self.horizon_s, MIN_LOOKAHEAD_M))
        views = [view for view in CAMERAS.values() if view in session.images]  # front first
        images = [(view, Image.open(io.BytesIO(session.images[view]))) for view in views]
        with self.lock:
            poses = session.driver.plan(request.time_now_us * 1e-6, images, command, velocity, acceleration)
        current = session.pose.pose
        q = current.quat
        origin = np.array([current.vec.x, current.vec.y, quaternion_yaw((q.w, q.x, q.y, q.z))])
        trajectory = [session.pose] + [
            pose_at(float(x), float(y), current.vec.z, float(yaw), request.time_now_us + POSE_INTERVAL_US * (k + 1))
            for k, (x, y, yaw) in enumerate(to_world(poses, origin))]
        return egodriver_pb2.DriveResponse(trajectory=common_pb2.Trajectory(poses=trajectory))

    def get_version(self, request, context):
        return common_pb2.VersionId(version_id="recogdrive", git_hash="", grpc_api_version=API_VERSION_MESSAGE)


def serve(driver: Driver, port: int, cfg) -> None:
    host = os.environ.get("ALPASIM_DRIVER_HOST") or str(OmegaConf.select(cfg, "infer.host") or "127.0.0.1")
    port = int(os.environ.get("ALPASIM_DRIVER_PORT") or port or DEFAULT_PORT)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(EgoDriver(driver), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("AlpaSim egodriver on %s:%d", host, port)
    server.wait_for_termination()
