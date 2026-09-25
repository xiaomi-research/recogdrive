"""ReCogDrive as a NAVSIM agent: sensors, feature/target builders, trajectory inference, and the
training glue (checkpoint loading, loss, optimizers, GRPO reward). The network is the backbone and
diffusion planner imported below."""

from typing import Any, List, Dict, Optional, Union
import contextlib
import functools
import inspect
import itertools
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

from recogdrive.adapters.navsim.features import ReCogDriveFeatureBuilder, TrajectoryTargetBuilder
from recogdrive.models.recogdrive.backbone import RecogDriveBackbone, hidden_size_from_vlm_config
from recogdrive.models.recogdrive.diffusion_planner import ReCogDriveDiffusionPlanner, make_recogdrive_config
from recogdrive.models.recogdrive.muon import Muon
from recogdrive.models.recogdrive.trajectory_utils import delta_to_waypoint, validate_training_target
from recogdrive.models.recogdrive.utils.internvl_preprocess import load_image
from recogdrive.models.recogdrive.utils.lr_scheduler import WarmupCosLR
from recogdrive.models.recogdrive.utils.utils import format_number, build_from_configs

# NAVSIM 2.0's AbstractAgent takes the trajectory sampling, 1.1's takes nothing.
AGENT_TAKES_SAMPLING = "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters


def prepare_vlm_inputs(features: Dict[str, torch.Tensor], load_tiles: bool) -> Dict[str, torch.Tensor]:
    """Runs in dataloader workers: prompt and image path as Python strings, so the model never reads
    them back from GPU tensors, plus the InternVL tiles when load_tiles."""
    path_tensor = features.get("image_path_tensor")
    if not isinstance(path_tensor, torch.Tensor):
        return features
    if path_tensor.ndim == 1:
        path_tensor = path_tensor.unsqueeze(0)
    image_paths = ReCogDriveAgent.decode_paths_from_tensor(path_tensor)
    features["image_paths"] = image_paths[0]
    features["vlm_question"] = ReCogDriveAgent.vlm_questions(
        features["history_trajectory"], features["high_command_one_hot"]
    )[0]
    if load_tiles:
        pixel_values_list = [load_image(path) for path in image_paths]
        features["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        features["num_patches"] = torch.tensor([pv.shape[0] for pv in pixel_values_list])
    return features


class ReCogDriveAgent(AbstractAgent):
    load_image_path = True

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
        super().__init__(trajectory_sampling) if AGENT_TAKES_SAMPLING else super().__init__()
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
            self.vlm_hidden_size = self.infer_vlm_hidden_size(self.vlm_path)

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
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            cfg.grpo_cfg.num_iterations = self.grpo_num_iterations
            cfg.grpo_cfg.clip_eps = self.grpo_clip_eps
            cfg.grpo_cfg.kl_coef = self.grpo_kl_coef
            cfg.grpo_cfg.bc_coeff = self.grpo_bc_coeff
            cfg.grpo_cfg.sample_time = self.grpo_sample_time
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()
        if self.grpo:
            from recogdrive.adapters.navsim.reward import PDMReward

            self.action_head.reward_fn = PDMReward(self.metric_cache_path)
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def worker_transform(self):
        if self.cache_hidden_state:
            return None
        # Qwen consumes image paths itself; InternVL tiles are built in the dataloader workers.
        return functools.partial(prepare_vlm_inputs, load_tiles=self.vlm_type.lower() != "qwen")

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

    @staticmethod
    def vlm_questions(history_trajectory: torch.Tensor, high_command_one_hot: torch.Tensor) -> List[str]:
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
            index = int(command_indices[i])
            command_str = navigation_commands[index] if index < len(navigation_commands) else "unknown"
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
            if "vlm_question" in features:
                questions = list(features["vlm_question"])
            else:
                questions = self.vlm_questions(features["history_trajectory"], features["high_command_one_hot"])
            if "num_patches" in features:
                counts = features["num_patches"]
                num_patches_list = [int(n) for n in (counts.tolist() if torch.is_tensor(counts) else counts)]
            if "image_paths" in features:
                image_paths = list(features["image_paths"])
            elif "image_path_tensor" in features:
                path_tensor = features["image_path_tensor"]
                if path_tensor.ndim == 1:
                    path_tensor = path_tensor.unsqueeze(0)
                image_paths = self.decode_paths_from_tensor(path_tensor)

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

            vlm_ctx = torch.no_grad() if not self.train_backbone else contextlib.nullcontext()
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
            target_trajectory = self.target_to_waypoint(targets["trajectory"]).to(predictions["pred_traj"].device)
            return torch.nn.functional.l1_loss(predictions["pred_traj"], target_trajectory)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        params = list(self.action_head.parameters())
        if self.backbone is not None and self.train_backbone:
            params += list(self.backbone.parameters())

        if self.optimizer_type == "muon":
            optimizer = Muon(params, lr=self._lr, weight_decay=1e-4, adamw_betas=(0.9, 0.95))
        else:
            optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95), fused=torch.cuda.is_available()))
            optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        # One host copy for the whole batch; per-element .item() on a CUDA tensor syncs once per character.
        return ["".join(map(chr, itertools.takewhile(bool, row))) for row in path_tensor.tolist()]

    def target_to_waypoint(self, target: torch.Tensor) -> torch.Tensor:
        if self.training_target == "delta":
            return delta_to_waypoint(target, self._trajectory_sampling.interval_length)
        return target

    @staticmethod
    def infer_vlm_hidden_size(vlm_path: str) -> Optional[int]:
        try:
            return hidden_size_from_vlm_config(AutoConfig.from_pretrained(vlm_path, trust_remote_code=True))
        except Exception as exc:
            print(f"Warning: failed to infer VLM hidden size from {vlm_path}: {exc}")
            return None
