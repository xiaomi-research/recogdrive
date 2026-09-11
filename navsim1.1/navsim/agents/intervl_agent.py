
import os
import re
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch
from lmdeploy import (
    ChatTemplateConfig,
    GenerationConfig,
    PytorchEngineConfig,
    TurbomindEngineConfig,
    pipeline,
)
from lmdeploy.vl import load_image
from PIL import Image
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import (
    AgentInput,
    Annotations,
    Scene,
    SensorConfig,
    Trajectory,
)
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""



def format_number(n, decimal_places=2):
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)

class InternVLFeatureBuilder(AbstractFeatureBuilder):

    def __init__(self):
        """Initializes the feature builder."""
        pass

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "internvl_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras
        return {"ego_statuses": ego_statuses,"cameras": cameras}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)

        return {
            "trajectory": torch.tensor(future_trajectory.poses),
        }

class InternVLAgent(AbstractAgent):

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: Optional[str] = None,
        prompt_type: Optional[str] = 'base',
        cam_type: Optional[str] = 'single',
    ):
        """Initializes the InternVLAgent.

        Args:
            trajectory_sampling (TrajectorySampling): The specification for sampling future trajectories.
            checkpoint_path (Optional[str]): Path to the model checkpoint to be loaded. Defaults to None.
            prompt_type (Optional[str]): Specifies the content of the text prompt.
                - 'base': Includes history, command, and visual perception.
                - 'vel_and_acc': Adds current velocity and acceleration to the base prompt.
                Defaults to 'base'.
            cam_type (Optional[str]): Specifies the camera view configuration.
                - 'single': Uses only the front camera view from the current timestep.
                - 'multi_view': Uses all six surrounding camera views from the current timestep.
                - 'cont': Uses continuous front camera views from the last 4 timesteps.
                Defaults to 'single'.
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.checkpoint_path = checkpoint_path
        self.prompt_type = prompt_type
        self.cam_type = cam_type
        self.pipe = pipeline(self.checkpoint_path, backend_config=PytorchEngineConfig(session_len=8192,dtype='bfloat16'), chat_template_config=ChatTemplateConfig(model_name='internvl2_5', meta_instruction=system_message))
        
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [InternVLFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor],targets=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        ego_statuses = features["ego_statuses"]
        cameras = features["cameras"]

        history_trajectory = []
        for i in range(4):
            ego_status = ego_statuses[i]
            history_trajectory.append({
                "x": format_number(ego_status.ego_pose[0]),
                "y": format_number(ego_status.ego_pose[1]),
                "heading": format_number(ego_status.ego_pose[2])
            })

        high_command_one_hot = ego_statuses[-1].driving_command
        navigation_commands = ['turn left', 'go straight', 'turn right']
        command_str = [navigation_commands[i] for i in range(len(high_command_one_hot)) if high_command_one_hot[i] == 1]
        command_str = command_str[0] if command_str else "unknown"

        image_paths = []
        image_prompt_lines = []
        image_prompt_desc = ""

        if self.cam_type == 'single':
            image_paths.append(str(cameras[-1].cam_f0.image))
            image_prompt_lines.append("<FRONT VIEW>:\n<image>\n")
            image_prompt_desc = "1. Visual perception from front camera view\n"
        elif self.cam_type == 'multi_view':
            image_paths.extend([str(cameras[-1].cam_f0.image), str(cameras[-1].cam_l0.image), str(cameras[-1].cam_r0.image), str(cameras[-1].cam_l2.image), str(cameras[-1].cam_r2.image), str(cameras[-1].cam_b0.image)])
            image_prompt_lines.append("<FRONT VIEW>:\n<image>\n<FRONT LEFT VIEW>:\n<image>\n<FRONT RIGHT VIEW>:\n<image>\n<BACK LEFT VIEW>:\n<image>\n<BACK RIGHT VIEW>:\n<image>\n<BACK VIEW>:\n<image>\n")
            #image_prompt_lines.append("<FRONT VIEW>:\n<image>\n<FRONT LEFT VIEW>':\n<image>\n<FRONT RIGHT VIEW>':\n<image>\n<BACK LEFT VIEW>':\n<image>\n<BACK RIGHT VIEW>':\n<image>\n<BACK VIEW>:\n<image>\n")
            image_prompt_desc = "1. Visual perception from the six surrounding camera views\n"
        elif self.cam_type == 'cont':
            for i in range(4):
                image_paths.append(str(cameras[i].cam_f0.image))
                image_prompt_lines.append(f"<FRONT VIEW>Frame-{i+1}: <image>\n")
            image_prompt_desc = "1. Visual perception from continuous front camera views of the last 4 timesteps\n"

        pixel_values = [load_image(image_path) for image_path in image_paths]

        generation_config = GenerationConfig(
                max_new_tokens=512,
                min_new_tokens=50,
                do_sample=True,
                temperature=0.0
        )

        image_prompt_str = "".join(image_prompt_lines)

        common_prompt = f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt_desc}2. Historical motion context (last 4 timesteps):{" ".join([f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n3. Active navigation command: [{command_str.upper()}]"""

        output_requirements = ("\nOutput requirements:\n- Predict 8 future trajectory points\n"
                               "- Each point format: (x:float, y:float, heading:float)\n"
                               "- Use [PT, ...] to encapsulate the trajectory\n"
                               "- Maintain numerical precision to 2 decimal places")

        if self.prompt_type == 'vel_and_acc':
            current_ego_status = ego_statuses[-1]
            vel_acc_info = (f"\n4. Current velocity: ({format_number(current_ego_status.ego_velocity[0])}, {format_number(current_ego_status.ego_velocity[1])})"
                            f"\n5. Current acceleration: ({format_number(current_ego_status.ego_acceleration[0])}, {format_number(current_ego_status.ego_acceleration[1])})")
            question = f"{image_prompt_str}\n{common_prompt}{vel_acc_info}{output_requirements}"
        else: 
            question = f"{''.join([f'<image>' for i in range(len(image_paths))])}\n{common_prompt}{output_requirements}"

        prompts = [(question, pixel_values)]
        
        responses = self.pipe(prompts, gen_config=generation_config)

        answers = [response.text for response in responses]

        full_match = re.search(r'\[PT(?:, )?((?:\([-+]?\d*\.\d+, [-+]?\d*\.\d+, [-+]?\d*\.\d+\)(?:, )?){8})\]', answers[0])
        if full_match:
            coords_matches = re.findall(r'\(([-+]?\d*\.\d+), ([-+]?\d*\.\d+), ([-+]?\d*\.\d+)\)', full_match.group(1))
            if len(coords_matches) == 8:
                coordinates = [tuple(map(float, coord)) for coord in coords_matches]
                coordinates_array = np.array(coordinates, dtype=np.float32)
                return {"trajectory": coordinates_array.reshape(1, self._trajectory_sampling.num_poses, 3)}

        print("Error parsing trajectory, returning zeros:", answer)
        return {"trajectory": np.zeros((1, self._trajectory_sampling.num_poses, 3), dtype=np.float32)}


    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}

        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0)

        return Trajectory(poses)

    def compute_loss(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)

