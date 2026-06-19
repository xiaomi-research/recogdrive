from typing import Literal

import torch


TrajectoryTargetType = Literal["waypoint", "delta"]


def validate_training_target(training_target: str) -> TrajectoryTargetType:
    if training_target not in ("waypoint", "delta"):
        raise ValueError(
            f"Unsupported training_target: {training_target!r}. "
            "Choose 'waypoint' or 'delta'."
        )
    return training_target  # type: ignore[return-value]


def waypoint_to_delta(trajectory: torch.Tensor, interval_length: float) -> torch.Tensor:
    """Converts waypoint poses to per-step linear/angular velocity deltas."""
    if interval_length <= 0:
        raise ValueError("interval_length must be positive when converting waypoints to delta.")

    origin = torch.zeros_like(trajectory[..., :1, :])
    previous = torch.cat([origin, trajectory[..., :-1, :]], dim=-2)
    diff = trajectory - previous
    heading_diff = torch.atan2(torch.sin(diff[..., 2:3]), torch.cos(diff[..., 2:3]))
    diff = torch.cat([diff[..., :2], heading_diff], dim=-1)
    return diff / interval_length


def delta_to_waypoint(delta: torch.Tensor, interval_length: float) -> torch.Tensor:
    """Converts per-step linear/angular velocity deltas back to waypoint poses."""
    if interval_length <= 0:
        raise ValueError("interval_length must be positive when converting delta to waypoints.")

    trajectory = torch.cumsum(delta * interval_length, dim=-2)
    heading = torch.atan2(torch.sin(trajectory[..., 2:3]), torch.cos(trajectory[..., 2:3]))
    return torch.cat([trajectory[..., :2], heading], dim=-1)
