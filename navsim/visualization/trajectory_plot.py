"""Plot the exact prediction used for the score beside its front-camera image."""

from pathlib import Path

import numpy as np


def save_trajectory_plot(agent_input, trajectory, token: str, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if token in ("", ".", "..") or Path(token).name != token:
        raise ValueError(f"Invalid scene token: {token!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (camera, bev) = plt.subplots(1, 2, figsize=(10, 4))
    try:
        with Image.open(agent_input.cameras[-1].cam_f0.image) as image:
            camera.imshow(image)
        camera.axis("off")
        history = [ego.ego_pose for ego in agent_input.ego_statuses]
        for label, poses in (("History", history), ("Prediction", trajectory.poses)):
            points = np.asarray(poses)
            bev.plot(-points[:, 1], points[:, 0], "o-", label=label)
        bev.set(xlabel="Right (m)", ylabel="Forward (m)", title=token, aspect="equal")
        bev.grid(alpha=0.3)
        bev.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{token}.png", dpi=150)
    finally:
        plt.close(fig)
