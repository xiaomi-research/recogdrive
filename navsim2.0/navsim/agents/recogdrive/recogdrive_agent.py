from typing import Any, List, Dict, Optional, Union
import contextlib
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoConfig
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .recogdrive_features import ReCogDriveFeatureBuilder ,TrajectoryTargetBuilder
from .recogdrive_backbone import RecogDriveBackbone, hidden_size_from_vlm_config
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from .trajectory_utils import delta_to_waypoint, validate_training_target
from .muon import Muon


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single', 
        vlm_type: Optional[str] = 'internvl', 
        dit_type: Optional[str] = 'small', 
        sampling_method: Optional[str] = 'ddim', 
        cache_mode: bool = False, 
        cache_hidden_state: bool = True, 
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
        vlm_size: Optional[str] = 'small', 
        train_backbone: bool = False,
        training_target: str = "waypoint",
        vlm_hidden_size: Optional[int] = None,
        flow_noise_level: float = 0.7,
        flow_sde_type: str = "sde",
        grpo_num_iterations: int = 1,
        grpo_clip_eps: float = 0.2,
        grpo_kl_coef: float = 0.0,
        grpo_bc_coeff: float = 0.1,
        grpo_sample_time: int = 8,
        optimizer_type: str = "adamw",
    ):
        super().__init__(trajectory_sampling)
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        self.grpo = grpo
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone
        self.training_target = validate_training_target(training_target)
        self.vlm_hidden_size = vlm_hidden_size
        self.flow_noise_level = flow_noise_level
        self.flow_sde_type = flow_sde_type
        self.grpo_num_iterations = max(int(grpo_num_iterations), 1)
        self.grpo_clip_eps = float(grpo_clip_eps)
        self.grpo_kl_coef = float(grpo_kl_coef)
        self.grpo_bc_coeff = float(grpo_bc_coeff)
        self.grpo_sample_time = int(grpo_sample_time)
        self.optimizer_type = str(optimizer_type or "adamw").lower()
        if self.optimizer_type not in ("adamw", "muon"):
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type!r}")
        self._grpo_rollout = None
        if self.vlm_hidden_size is None and self.vlm_type.lower() == "qwen" and self.vlm_path:
            self.vlm_hidden_size = self._infer_vlm_hidden_size(self.vlm_path)

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=device
            )

            if not self.train_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.backbone.eval()
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True

        if self.dit_type == "large":
            cfg = make_recogdrive_config(
                self.dit_type,
                action_dim=3,
                action_horizon=8,
                grpo=self.grpo,
                input_embedding_dim=1536,
                sampling_method=sampling_method,
                training_target=self.training_target,
                delta_interval_length=self._trajectory_sampling.interval_length,
                vlm_hidden_size=self.vlm_hidden_size,
            )
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(
                self.dit_type,
                action_dim=3,
                action_horizon=8,
                grpo=self.grpo,
                input_embedding_dim=384,
                sampling_method=sampling_method,
                training_target=self.training_target,
                delta_interval_length=self._trajectory_sampling.interval_length,
                vlm_hidden_size=self.vlm_hidden_size,
            )

        cfg.vlm_size = self.vlm_size
        cfg.grpo_cfg.flow_noise_level = self.flow_noise_level
        cfg.grpo_cfg.flow_sde_type = self.flow_sde_type

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            cfg.grpo_cfg.num_iterations = self.grpo_num_iterations
            cfg.grpo_cfg.clip_eps = self.grpo_clip_eps
            cfg.grpo_cfg.kl_coef = self.grpo_kl_coef
            cfg.grpo_cfg.bc_coeff = self.grpo_bc_coeff
            cfg.grpo_cfg.sample_time = self.grpo_sample_time
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in ckpt.items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            self.load_state_dict(filtered_ckpt, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(
            trajectory_sampling=self._trajectory_sampling,
            training_target=self.training_target,
        )]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
            cache_mode=self.cache_mode,
        )]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None and not self.train_backbone:
            self.backbone.eval()
        return self

    def _vlm_questions(self, history_trajectory: torch.Tensor, high_command_one_hot: torch.Tensor) -> List[str]:
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)
        history_trajectory = history_trajectory.detach().cpu()
        high_command_one_hot = high_command_one_hot.detach().cpu()
        navigation_commands = ['turn left', 'go straight', 'turn right']
        command_indices = torch.argmax(high_command_one_hot, dim=-1)
        questions = []
        for i in range(high_command_one_hot.shape[0]):
            sample = history_trajectory[i]
            command_str = navigation_commands[int(command_indices[i])]
            history_str = ' '.join([
                f'   - t-{3-j}: ({format_number(float(sample[j, 0]))}, '
                f'{format_number(float(sample[j, 1]))}, '
                f'{format_number(float(sample[j, 2]))})'
                for j in range(sample.shape[0])
            ])
            questions.append(
                "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                "1. Visual perception from front camera view\n"
                f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                f"3. Active navigation command: [{command_str.upper()}]\n"
                "Output requirements:\n- Predict 8 future trajectory points\n"
                "- Each point format: (x:float, y:float, heading:float)\n"
                "- Use [PT, ...] to encapsulate the trajectory\n"
                "- Maintain numerical precision to 2 decimal places"
            )
        return questions

    def clear_grpo_rollout(self) -> None:
        self._grpo_rollout = None

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets=None,
        tokens_list=None,
        reuse_rollout: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if reuse_rollout and self.grpo and self._grpo_rollout is not None:
            return self.action_head.grpo_loss_from_rollout(self._grpo_rollout)

        questions = None
        num_patches_list = None
        image_paths = None
        if not self.cache_hidden_state:
            questions = self._vlm_questions(features["history_trajectory"], features["high_command_one_hot"])
            if "num_patches" in features:
                num_patches_list = [int(n) for n in features["num_patches"].tolist()]
            if "pixel_values" not in features and "image_path_tensor" in features:
                path_tensor = features["image_path_tensor"]
                if path_tensor.ndim == 1:
                    path_tensor = path_tensor.unsqueeze(0)
                image_paths = self._decode_paths_from_tensor(path_tensor)

        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda(non_blocking=True)
        if targets:
            for key, tensor in targets.items():
                if isinstance(tensor, torch.Tensor):
                    targets[key] = tensor.cuda(non_blocking=True)

        model_dtype = next(self.action_head.parameters()).dtype

        history_trajectory = features["history_trajectory"]
        high_command_one_hot = features["high_command_one_hot"]
        
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"]
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            if self.vlm_type.lower() == "qwen":
                pixel_values_cat = image_paths
                num_patches_list = None
            elif "pixel_values" in features:
                pixel_values_cat = features["pixel_values"]
            else:
                pixel_values_list = [load_image(path) for path in image_paths]
                num_patches_list = [p.shape[0] for p in pixel_values_list]
                pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda(non_blocking=True)

            vlm_ctx = torch.inference_mode() if not self.train_backbone else contextlib.nullcontext()
            with vlm_ctx:
                outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

        status_feature = features["status_feature"]
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)

        if self.training and not self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            return self.action_head(last_hidden_state, action_inputs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
            self._grpo_rollout = self.action_head.collect_grpo_rollout(
                last_hidden_state, action_inputs, tokens_list
            )
            return self.action_head.grpo_loss_from_rollout(self._grpo_rollout)
        else: 
            action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype)})
            return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            target_trajectory = self._target_to_waypoint(targets["trajectory"]).to(predictions["pred_traj"].device)
            return torch.nn.functional.l1_loss(predictions["pred_traj"], target_trajectory)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        params = list(self.action_head.parameters())
        if self.backbone is not None and self.train_backbone:
            params += list(self.backbone.parameters())

        if self.optimizer_type == "muon":
            optimizer = Muon(params, lr=self._lr, weight_decay=1e-4, adamw_betas=(0.9, 0.95))
        else:
            optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
            optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

    def _target_to_waypoint(self, target: torch.Tensor) -> torch.Tensor:
        if self.training_target == "delta":
            return delta_to_waypoint(target, self._trajectory_sampling.interval_length)
        return target

    @staticmethod
    def _infer_vlm_hidden_size(vlm_path: str) -> Optional[int]:
        try:
            return hidden_size_from_vlm_config(AutoConfig.from_pretrained(vlm_path, trust_remote_code=True))
        except Exception as exc:
            print(f"Warning: failed to infer VLM hidden size from {vlm_path}: {exc}")
            return None

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
    training_target: str = "waypoint",
    delta_interval_length: float = 0.5,
    vlm_hidden_size: Optional[int] = None,
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.
        training_target (str): Train the planner on 'waypoint' poses or 'delta' velocities.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
        training_target=validate_training_target(training_target),
        delta_interval_length=delta_interval_length,
        vlm_hidden_size=vlm_hidden_size,
    )
    
    return config
