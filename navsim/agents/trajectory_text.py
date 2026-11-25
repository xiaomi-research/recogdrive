"""Read a Stage 1 VLM's labeled trajectory without evaluating its text."""

import re
from typing import Optional

import numpy as np


_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_POINT = re.compile(rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)")
_TRAJECTORY = re.compile(
    rf"\bPT\b\s*(?:=\s*\[|,?)\s*({_POINT.pattern}(?:\s*,\s*{_POINT.pattern})*)\s*,?\s*\]"
)


def parse_trajectory_text(text: str, num_poses: int) -> Optional[np.ndarray]:
    """Return a complete finite trajectory, or None for an invalid answer."""
    match = _TRAJECTORY.search(text or "")
    if match is None:
        return None
    points = _POINT.findall(match.group(1))
    if len(points) != num_poses:
        return None
    with np.errstate(over="ignore"):
        poses = np.asarray(points, dtype=np.float32)
    return poses if np.isfinite(poses).all() else None
