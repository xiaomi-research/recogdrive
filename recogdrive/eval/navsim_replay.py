"""A NAVSIM agent that returns trajectories produced elsewhere, looked up by scene token.

NAVSIM's scoring scripts give agents with requires_scene=True the scene, whose metadata carries the token,
so they can score the trainer's trajectories without loading a model or touching a GPU.
"""

import inspect
import pickle

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig


class ReplayAgent(AbstractAgent):
    def __init__(self, trajectories_path: str):
        with open(trajectories_path, "rb") as f:
            self.trajectories = pickle.load(f)
        kwargs = {"requires_scene": True}
        if "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters:  # NAVSIM 2.0
            kwargs["trajectory_sampling"] = next(iter(self.trajectories.values())).trajectory_sampling
        super().__init__(**kwargs)

    def name(self) -> str:
        return "replay"

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input, scene=None):
        return self.trajectories[scene.scene_metadata.initial_token]
