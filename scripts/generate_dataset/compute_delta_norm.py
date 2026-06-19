import argparse
import json
import re
from pathlib import Path

import numpy as np


TRAJECTORY_PATTERN = re.compile(
    r"\[PT(?:,\s*)?((?:\([-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+\)(?:,\s*)?)+)\]"
)
POINT_PATTERN = re.compile(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)")


def iter_answer_text(item):
    if "answer" in item:
        yield item["answer"]
    for conversation in item.get("conversations", []):
        if conversation.get("from") in {"gpt", "assistant"}:
            yield conversation.get("value", "")


def parse_waypoints(text):
    match = TRAJECTORY_PATTERN.search(text)
    if match is None:
        return None
    points = [tuple(map(float, point)) for point in POINT_PATTERN.findall(match.group(1))]
    if not points:
        return None
    return np.asarray(points, dtype=np.float64)


def waypoint_to_delta(waypoints, interval_length):
    previous = np.concatenate([np.zeros_like(waypoints[:1]), waypoints[:-1]], axis=0)
    diff = waypoints - previous
    diff[:, 2] = np.arctan2(np.sin(diff[:, 2]), np.cos(diff[:, 2]))
    return diff / interval_length


def main():
    parser = argparse.ArgumentParser(description="Compute delta target norm bounds from NAVSIM-Traj JSONL.")
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to NAVSIM-Traj jsonl.")
    parser.add_argument("--interval-length", type=float, default=0.5, help="Trajectory interval length in seconds.")
    parser.add_argument("--percentile", type=float, default=None, help="Optional symmetric clipping percentile, e.g. 0.1 uses [0.1, 99.9].")
    args = parser.parse_args()

    deltas = []
    skipped = 0
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            waypoints = None
            for answer in iter_answer_text(item):
                waypoints = parse_waypoints(answer)
                if waypoints is not None:
                    break
            if waypoints is None:
                skipped += 1
                continue
            deltas.append(waypoint_to_delta(waypoints, args.interval_length))

    if not deltas:
        raise RuntimeError(f"No [PT, ...] waypoint trajectories found in {args.jsonl}.")

    delta_array = np.concatenate(deltas, axis=0)
    if args.percentile is None:
        delta_min = delta_array.min(axis=0)
        delta_max = delta_array.max(axis=0)
    else:
        lower = args.percentile
        upper = 100.0 - args.percentile
        delta_min = np.percentile(delta_array, lower, axis=0)
        delta_max = np.percentile(delta_array, upper, axis=0)

    print(f"Loaded trajectories: {len(deltas)}")
    print(f"Skipped lines: {skipped}")
    print("Delta norm constants:")
    print(f"DELTA_NORM_MIN = [{delta_min[0]:.6f}, {delta_min[1]:.6f}, {delta_min[2]:.6f}]")
    print(f"DELTA_NORM_MAX = [{delta_max[0]:.6f}, {delta_max[1]:.6f}, {delta_max[2]:.6f}]")
    print("Use this override for delta training:")
    print(f"agent.training_target=delta")


if __name__ == "__main__":
    main()
