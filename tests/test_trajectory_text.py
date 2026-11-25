import numpy as np
import pytest

from navsim.agents.trajectory_text import parse_trajectory_text


@pytest.mark.parametrize("wrapper", [
    "[PT, {points}]",
    "**PT, {points}].",
    "```python\nPT = [{points}].  # noqa\n```",
])
def test_stage1_trajectory_accepts_wrapped_points(wrapper):
    expected = np.array([(x, 0.125, -0.25) for x in range(1, 9)], dtype=np.float32)
    points = ", ".join(f"(+{x:.2f}, {y:.3f}, {heading:.2f})" for x, y, heading in expected)
    result = parse_trajectory_text(wrapper.format(points=points), len(expected))
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32


@pytest.mark.parametrize("answer,num_poses", [
    ("The image shows an intersection.", 8),
    ("[PT, (1.0, 0.0, 0.0)]", 8),
    ("[PT, (1e999, 0.0, 0.0)]", 1),
])
def test_stage1_trajectory_rejects_invalid_answers(answer, num_poses):
    assert parse_trajectory_text(answer, num_poses) is None
