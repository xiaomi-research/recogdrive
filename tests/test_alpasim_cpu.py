"""AlpaSim protocol regression with a template policy and nuScenes mini scene.

This exercises the RPC contract against a local mock runtime; it is not an AlpaSim simulator score.
"""
import math
import os
from concurrent import futures
from pathlib import Path

import numpy as np
import pytest


def test_alpasim_template_grpc_protocol_cpu():
    grpc = pytest.importorskip("grpc")
    pytest.importorskip("alpasim_grpc.v0.egodriver_pb2_grpc")

    from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from recogdrive.adapters.template import EgoMLPPolicy
    from recogdrive.closedloop import COMMANDS, Driver, to_world
    from recogdrive.closedloop.alpasim import EgoDriver, pose_at, route_command
    from recogdrive.data.nuscenes import NuScenesDataset, quaternion_yaw, to_local

    root = Path(os.environ.get("NUSCENES_ROOT", "/data/yongkangli/datasets/nuscenes"))
    if not (root / "v1.0-mini").is_dir():
        pytest.skip("set NUSCENES_ROOT to a local nuScenes mini dataset")

    s = np.linspace(0, 60, 61)
    straight = np.stack([s, 0 * s], 1)
    left = np.stack([30 * np.sin(s / 30), 30 * (1 - np.cos(s / 30))], 1)
    assert route_command(straight, 20.0) == COMMANDS.index("straight")
    assert route_command(left, 20.0) == COMMANDS.index("left")
    assert route_command(left * [1, -1], 20.0) == COMMANDS.index("right")
    assert route_command(np.zeros((0, 2)), 20.0) == COMMANDS.index("straight")
    assert route_command(left - [100, 0], 20.0) == COMMANDS.index("straight")  # route entirely behind

    def decoded(at):
        q = at.pose.quat
        return at.timestamp_us * 1e-6, at.pose.vec.x, at.pose.vec.y, quaternion_yaw((q.w, q.x, q.y, q.z))

    dataset = NuScenesDataset(root, "v1.0-mini", "val", 8, [])
    tokens, cameras, pose, stamps = dataset.tracks[0]
    policy = EgoMLPPolicy(TrajectorySampling(num_poses=8, interval_length=0.5)).eval()
    executor = futures.ThreadPoolExecutor(4)
    server = grpc.server(executor)
    started = False
    channel = None
    stub = None
    sessions = []
    try:
        egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(EgoDriver(Driver(policy, "cpu")), server)
        port = server.add_insecure_port("127.0.0.1:0")
        assert port
        server.start()
        started = True
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(channel).result(timeout=30)
        stub = egodriver_pb2_grpc.EgodriverServiceStub(channel)
        assert stub.get_version(common_pb2.Empty(), timeout=30).version_id == "recogdrive"
        for uuid in ("a", "b"):
            stub.start_session(egodriver_pb2.DriveSessionRequest(session_uuid=uuid), timeout=30)
            sessions.append(uuid)
        assert not stub.drive(egodriver_pb2.DriveRequest(
            session_uuid="a", time_now_us=int(stamps[0] * 1e6)), timeout=30).trajectory.poses

        reference = Driver(policy, "cpu")
        worst = 0.0
        for k in range(len(tokens) - 8):
            t_us = int(stamps[k] * 1e6)
            jpeg = Path(dict(cameras[k])["front"]).read_bytes()
            velocity = to_local(pose[k:k + 2], pose[k])[1, :2] / max(stamps[k + 1] - stamps[k], 1e-3)
            route = to_local(pose[k + 1:], pose[k])[:, :2]
            results = {}
            for uuid, shift in (("a", 0.0), ("b", 100.0)):
                for logical_id in ("camera_front_wide_120fov", "camera_front_tele_30fov"):
                    stub.submit_image_observation(egodriver_pb2.RolloutCameraImage(
                        session_uuid=uuid, camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
                            frame_start_us=t_us - 30_000, frame_end_us=t_us, image_bytes=jpeg,
                            logical_id=logical_id)), timeout=30)
                at = pose_at(pose[k, 0] + shift, pose[k, 1], 0.5, pose[k, 2], t_us)
                state = common_pb2.DynamicState(
                    linear_velocity=common_pb2.Vec3(x=velocity[0], y=velocity[1]),
                    linear_acceleration=common_pb2.Vec3(x=0.2, y=-0.1))
                stub.submit_egomotion_observation(egodriver_pb2.RolloutEgoTrajectory(
                    session_uuid=uuid, trajectory=common_pb2.Trajectory(poses=[at]),
                    dynamic_states=[state]), timeout=30)
                stub.submit_route(egodriver_pb2.RouteRequest(session_uuid=uuid, route=egodriver_pb2.Route(
                    timestamp_us=t_us,
                    waypoints=[common_pb2.Vec3(x=x, y=y) for x, y in route])), timeout=30)
                results[uuid] = stub.drive(egodriver_pb2.DriveRequest(
                    session_uuid=uuid, time_now_us=t_us, time_query_us=t_us + 100_000), timeout=30
                ).trajectory.poses
                if uuid == "a":
                    t, x, y, yaw = decoded(at)
                    reference.observe(t, x, y, yaw)
                    v = (state.linear_velocity.x, state.linear_velocity.y)
                    a = (state.linear_acceleration.x, state.linear_acceleration.y)
                    command = route_command(np.array([[w.x, w.y] for w in egodriver_pb2.Route(
                        waypoints=[common_pb2.Vec3(x=x_, y=y_) for x_, y_ in route]).waypoints]),
                        max(math.hypot(*v) * 4.0, 10.0))
                    plan = reference.plan(t, [("front", None)], command, v, a)
                    expected = to_world(plan, np.array([x, y, yaw]))
            for uuid, shift in (("a", 0.0), ("b", 100.0)):
                poses = results[uuid]
                assert len(poses) == 9 and poses[0].timestamp_us == t_us
                assert [p.timestamp_us for p in poses[1:]] == [t_us + 500_000 * (i + 1) for i in range(8)]
                got = np.array([decoded(p)[1:] for p in poses[1:]]) - [shift, 0.0, 0.0]
                diff = np.abs(got - expected)
                diff[:, 2] = np.abs((diff[:, 2] + np.pi) % (2 * np.pi) - np.pi)
                worst = max(worst, float(diff.max()))
        assert worst < 2e-3, worst
    finally:
        if stub is not None:
            for uuid in sessions:
                try:
                    stub.close_session(egodriver_pb2.DriveSessionCloseRequest(session_uuid=uuid), timeout=5)
                except grpc.RpcError:
                    pass
        if channel is not None:
            channel.close()
        if started:
            server.stop(0).wait(timeout=5)
        executor.shutdown(wait=True)
