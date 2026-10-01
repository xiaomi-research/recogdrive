"""ReCogDrive as a NAVSIM agent: sensors, feature/target builders, trajectory inference, and the
training glue (checkpoint loading, loss, optimizers, GRPO reward). The network is the backbone and
diffusion planner imported below."""

from typing import Any, List, Dict, Optional, Tuple, Union
import contextlib
import dataclasses
import functools
import inspect
import itertools
import os
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoConfig

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
from recogdrive.models.recogdrive.utils.utils import format_number

# NAVSIM 2.0's AbstractAgent takes the trajectory sampling, 1.1's takes nothing.
AGENT_TAKES_SAMPLING = "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters


SURROUND_VIEWS = ("front", "front_left", "front_right", "left", "right", "back_left", "back_right", "back")


def camera_views(features: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(view, path) pairs of a sample: `camera_paths` (every view the source has, front first) or
    the front-only `image_path_tensor` of NAVSIM caches."""
    if "camera_paths" in features:
        return [tuple(view) for view in features["camera_paths"]]
    path_tensor = features.get("image_path_tensor")
    if not isinstance(path_tensor, torch.Tensor):
        return []
    return [("front", ReCogDriveAgent.decode_paths_from_tensor(path_tensor.reshape(1, -1))[0])]


def prepare_vlm_inputs(
    features: Dict[str, Any],
    load_tiles: bool,
    num_poses: int = 8,
    cameras: Optional[List[str]] = None,
    short_sides: Tuple[Optional[int], Optional[int]] = (None, None),
    image_augment=None,
) -> Dict[str, Any]:
    """Runs in dataloader workers: prompt and image paths as Python strings, so the model never reads
    them back from GPU tensors, plus the InternVL tiles when load_tiles. `cameras` selects the
    views of a multi-view prompt (front / other views rescaled to `short_sides`); None keeps the
    single front view at its native resolution."""
    views = camera_views(features)
    if not views:
        return features
    if cameras:
        available = dict(views)
        views = [(view, available[view]) for view in cameras if view in available]
        sizes = [short_sides[0] if view == "front" else short_sides[1] for view, _ in views]
        features["image_paths"] = [path for _, path in views]
    else:
        views = [next((view for view in views if view[0] == "front"), views[0])]
        sizes = [None]
        features["image_paths"] = views[0][1]
    features["vlm_question"] = ReCogDriveAgent.vlm_questions(
        features["history_trajectory"], features["high_command_one_hot"],
        num_poses=num_poses, views=[view for view, _ in views] if cameras else None,
    )[0]
    if load_tiles:
        pixel_values_list = [load_image(path, short_side=size, augment=image_augment)
                             for (_, path), size in zip(views, sizes)]
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
        grpo_reward_workers: Optional[int] = None,
        optimizer_type: str = "adamw",
        cameras: Optional[List[str]] = None,
        front_short_side: int = 960,
        side_short_side: int = 480,
        vlm_max_length: Optional[int] = None,
        action_norm_min: Optional[List[float]] = None,
        action_norm_max: Optional[List[float]] = None,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,
    ):
        """cam_type 'multi' feeds `cameras` (default: every surround view the data source has) with the
        front view rescaled to front_short_side and the others to side_short_side; 'single' feeds the
        native-resolution front view. vlm_max_length (the padded prompt length) defaults to 2800 for single view and
        8704 for multi-view (WOD-E2E's 8 views take ~8240 tokens, nuScenes' 6 ~6450). The prediction horizon is
        trajectory_sampling.num_poses. lora_rank > 0 trains LoRA adapters on the frozen VLM (lora_targets, default
        the language model's attention and MLP projections); train_backbone fine-tunes the whole VLM instead.
        grpo_reward_workers: processes per rank scoring the GRPO reward (None: half the CPU cores per rank, at most
        16; 0: in the training process)."""
        super().__init__(trajectory_sampling) if AGENT_TAKES_SAMPLING else super().__init__()
        self._trajectory_sampling = self.trajectory_sampling = trajectory_sampling
        if cam_type not in ("single", "multi"):
            raise ValueError(f"cam_type must be 'single' or 'multi', got {cam_type!r}")
        self.cameras = list(cameras or SURROUND_VIEWS) if cam_type == "multi" else None
        self.short_sides = (front_short_side, side_short_side)
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
        if lora_rank and train_backbone:
            raise ValueError("train_backbone fine-tunes the whole VLM and lora_rank adds adapters to a frozen one; set one")
        self.vlm_trainable = bool(train_backbone or lora_rank)
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
                device=device,
                max_length=vlm_max_length or (8704 if self.cameras else 2800),
            )
            if lora_rank:
                self.backbone.add_lora(lora_rank, lora_alpha, lora_dropout, lora_targets)
            for name, p in self.backbone.named_parameters():
                p.requires_grad = train_backbone or (bool(lora_rank) and ".lora_" in name)
            if not self.vlm_trainable:
                self.backbone.eval()
        elif lora_rank:
            raise ValueError("lora_rank needs the online VLM: set cache_hidden_state=False and cache_mode=False")

        embedding_dims = {"large": 1536, "small": 384}
        if self.dit_type not in embedding_dims:
            raise ValueError(f"dit_type must be one of {sorted(embedding_dims)}, got {self.dit_type!r}")
        cfg = make_recogdrive_config(
            self.dit_type,
            action_dim=3,
            action_horizon=self._trajectory_sampling.num_poses,
            grpo=self.grpo,
            input_embedding_dim=embedding_dims[self.dit_type],
            sampling_method=sampling_method,
            training_target=self.training_target,
            delta_interval_length=self._trajectory_sampling.interval_length,
            vlm_hidden_size=self.vlm_hidden_size,
            action_norm_min=action_norm_min,
            action_norm_max=action_norm_max,
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

            if grpo_reward_workers is None:
                ranks = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
                grpo_reward_workers = min(16, max(1, (os.cpu_count() or 2) // (2 * ranks)))
            self.action_head.reward_fn = PDMReward(self.metric_cache_path, grpo_reward_workers)

    def name(self) -> str:
        return self.__class__.__name__

    def worker_transform(self, image_augment=None):
        """Per-sample preprocessing run in dataloader workers; image_augment (PIL -> PIL) is passed for
        training splits only."""
        if self.cache_hidden_state:
            return None
        # Qwen consumes image paths itself; InternVL tiles are built in the dataloader workers.
        return functools.partial(
            prepare_vlm_inputs,
            load_tiles=self.vlm_type.lower() != "qwen",
            num_poses=self._trajectory_sampling.num_poses,
            cameras=self.cameras,
            short_sides=self.short_sides,
            image_augment=image_augment,
        )

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
            print(f"Loaded {len(filtered_ckpt)} of {len(ckpt)} checkpoint tensors from {self.checkpoint_path}")

    def get_sensor_config(self) -> SensorConfig:
        # the features read the current frame's camera paths only, never the lidar
        return dataclasses.replace(SensorConfig.build_all_sensors(include=[3]), lidar_pc=False)

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
            multi_view=self.cameras is not None,
        )]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            # frozen parts of the VLM (all of it, or all but the LoRA layers) keep inference behaviour
            for module in self.backbone.modules():
                params = list(module.parameters())
                if params and not any(p.requires_grad for p in params):
                    module.eval()
        return self

    @staticmethod
    def vlm_questions(
        history_trajectory: torch.Tensor,
        high_command_one_hot: torch.Tensor,
        num_poses: int = 8,
        views: Optional[List[str]] = None,
    ) -> List[str]:
        if views:
            images = "".join(f"<{view.replace('_', ' ').upper()} VIEW>:\n<image>\n" for view in views)
            perception = f"1. Visual perception from {len(views)} surround camera views\n"
        else:
            images, perception = "<image>\n", "1. Visual perception from front camera view\n"
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
                f"{images}As an autonomous driving system, predict the vehicle's trajectory based on:\n"
                f"{perception}"
                f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                f"3. Active navigation command: [{command_str.upper()}]\n"
                f"Output requirements:\n- Predict {num_poses} future trajectory points\n"
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
        if self.training and self.grpo and tokens_list is not None:
            # metric caches load during the VLM forward
            self.action_head.reward_fn.prefetch(list(tokens_list), self.grpo_sample_time)

        questions = None
        num_patches_list = None
        image_paths = None
        if not self.cache_hidden_state:
            if "vlm_question" in features:
                questions = list(features["vlm_question"])
            else:
                questions = self.vlm_questions(features["history_trajectory"], features["high_command_one_hot"],
                                               num_poses=self._trajectory_sampling.num_poses)
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

            vlm_ctx = torch.no_grad() if not self.vlm_trainable else contextlib.nullcontext()
            with vlm_ctx:
                last_hidden_state = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)

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

        features: Dict[str, Any] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        transform = self.worker_transform()
        if transform is not None:
            features = transform(features)
        # batch of one: per-image tiles and patch counts are already flat
        batch = {}
        for key, value in features.items():
            if key == "num_patches":
                batch[key] = value.tolist()
            elif key == "pixel_values":
                batch[key] = value
            else:
                batch[key] = value.unsqueeze(0) if torch.is_tensor(value) else [value]

        with torch.no_grad():
            predictions = self.forward(batch)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses, self._trajectory_sampling)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        return self.compute_trajectory(agent_input)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            target_trajectory = self.target_to_waypoint(targets["trajectory"]).to(predictions["pred_traj"].device)
            return torch.nn.functional.l1_loss(predictions["pred_traj"], target_trajectory)

    def get_optimizers(
        self, total_steps: Optional[int] = None, steps_per_epoch: Optional[int] = None
    ) -> Union[Optimizer, Dict[str, LRScheduler]]:
        """Warmup (3 epochs for imitation learning, none for GRPO) and cosine decay over the whole training. Given the
        training length in optimizer steps (the recogdrive trainer) the schedule advances every step; without it
        (Lightning) it advances every epoch over the recipe's 200 (IL) / 10 (GRPO) epochs."""
        params = list(self.action_head.parameters())
        if self.backbone is not None and self.vlm_trainable:
            params += [p for p in self.backbone.parameters() if p.requires_grad]

        if self.optimizer_type == "muon":
            optimizer = Muon(params, lr=self._lr, weight_decay=1e-4, adamw_betas=(0.9, 0.95))
        else:
            optimizer = torch.optim.AdamW(params, lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95),
                                          fused=torch.cuda.is_available())
        
        warmup_epochs, min_lr, recipe_epochs = (0, 0.0, 10) if self.grpo else (3, 1e-6, 200)
        if total_steps:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=min_lr, total_steps=total_steps,
                                    warmup_steps=warmup_epochs * (steps_per_epoch or 0))
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=min_lr, total_steps=recipe_epochs,
                                warmup_steps=warmup_epochs)
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
