"""Template policy adapter: an MLP planner on ego history and status. Copy this file to plug a new model in; it trains
on every registered data source and every open-loop benchmark scores it, neither knowing the model. Its config is
`agent=recogdrive_template` (config/common/agent/recogdrive_template.yaml in either NAVSIM tree).
"""

from types import SimpleNamespace

import torch
from torch import nn

STATUS_DIM, HISTORY_POSES = 8, 4  # sample contract: status_feature (8,), history_trajectory (4, 3)


class EgoMLPPolicy(nn.Module):
    load_image_path = True  # NAVSIM scenes: camera paths, never pixels

    def __init__(self, trajectory_sampling, hidden_dim: int = 256, lr: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.trajectory_sampling = trajectory_sampling
        self.num_poses = int(trajectory_sampling.num_poses)
        self.lr, self.weight_decay = lr, weight_decay
        self.mlp = nn.Sequential(
            nn.Linear(STATUS_DIM + HISTORY_POSES * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.num_poses * 3),
        )

    def forward(self, features, targets=None, tokens=None):
        inputs = torch.cat([features["status_feature"], features["history_trajectory"].flatten(1)], dim=1)
        compute = self.mlp[0].weight.dtype  # bf16 under FSDP mixed precision
        poses = self.mlp(inputs.to(compute)).float().view(-1, self.num_poses, 3)
        if self.training:
            return SimpleNamespace(loss=nn.functional.l1_loss(poses, targets["trajectory"]))
        return {"pred_traj": poses}

    def compute_loss(self, features, targets, predictions):
        return nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_target_builders(self):
        return []

    def get_sensor_config(self):
        import dataclasses

        from navsim.common.dataclasses import SensorConfig

        return dataclasses.replace(SensorConfig.build_all_sensors(include=[3]), lidar_pc=False)

    def get_feature_builders(self):
        from recogdrive.adapters.navsim.features import ReCogDriveFeatureBuilder

        return [ReCogDriveFeatureBuilder(cache_hidden_state=False)]
