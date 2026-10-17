"""Mixed VQA and planning-GRPO trainer routing, data guards and cleanup."""

import signal
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from omegaconf import OmegaConf

from recogdrive.adapters.navsim.agent import ReCogDriveAgent
from recogdrive.adapters.waymoe2e.reward import WaymoADEReward, WaymoRFSReward
from recogdrive.data.mixture import MixtureDataset
from recogdrive.data.navsim import NavsimSamples
from recogdrive.data.vqa import VQADataset
from recogdrive.data.waymoe2e import WaymoE2ECacheOnlyDataset
from recogdrive.models.recogdrive.diffusion_planner import ReCogDriveDiffusionPlanner
from recogdrive.train.args import TrainArgs
from recogdrive.train.trainer import Trainer


class TinySharedVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = nn.Linear(2, 2, bias=False)
        self.planning_calls = 0
        self.vqa_calls = 0

    def forward(self, pixels, questions, num_patches_list=None):
        self.planning_calls += 1
        return self.lora(pixels).unsqueeze(1)

    def forward_vqa(self, features):
        self.vqa_calls += 1
        return self.lora(features["vqa_x"]).square().mean()


class TinyMixedDataset(Dataset):
    sample_tasks = ["planning", "vqa", "planning"]

    def __len__(self):
        return len(self.sample_tasks)

    def __getitem__(self, index):
        task = self.sample_tasks[index]
        if task == "vqa":
            return {"task": "vqa", "vqa_x": torch.tensor([0.5, 1.5])}, {}, "vqa-1"
        return {
            "task": "planning",
            "vlm_question": "predict",
            "history_trajectory": torch.zeros(4, 3),
            "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "status_feature": torch.zeros(8),
            "pixel_values": torch.tensor([[1.0, 2.0]]),
        }, {"trajectory": torch.zeros(2, 3)}, f"planning-{index}"


class OrderedLoader:
    def __init__(self):
        self.dataset = TinyMixedDataset()
        self.batch_sampler = [[0], [1], [2]]
        self.sampler = SimpleNamespace()

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for indices in self.batch_sampler:
            features, targets, tokens = self.dataset[indices[0]]
            features = {key: ([value] if key in ("task", "vlm_question")
                              else value if key == "pixel_values"
                              else value.unsqueeze(0) if torch.is_tensor(value) else value)
                        for key, value in features.items()}
            targets = {key: value.unsqueeze(0) for key, value in targets.items()}
            yield features, targets, [tokens]


def _mixed_agent():
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    nn.Module.__init__(planner)
    planner.config = SimpleNamespace(sampling_method="ddim")
    planner.grpo_objective = "ppo"
    planner.clip_eps = 0.2
    planner.bc_coeff = 0.0
    planner.kl_coef = 0.0
    planner.sample_time = 2
    planner.gamma_denoising = 0.6
    planner.clip_advantage_lower_quantile = 0.0
    planner.clip_advantage_upper_quantile = 1.0
    planner.policy_scale = nn.Parameter(torch.tensor(0.25))
    calls = {"sample": 0, "reward": [], "prefetch": 0}

    def sample_chain(self, vl, his, status, **kwargs):
        calls["sample"] += 1
        chains = torch.ones(his.shape[0], 3, 2, 3)
        chains[:, 0, 0, 0] = torch.arange(1, his.shape[0] + 1, device=his.device)
        return chains, chains[:, -1]

    def get_logprobs(self, vl, his, status, chains, deterministic=False):
        groups = his.shape[0] // vl.shape[0]
        values = vl.mean((1, 2)).repeat_interleave(groups)
        per_candidate = self.policy_scale * values * chains[:, 0, 0, 0]
        return per_candidate.repeat_interleave(chains.shape[1] - 1).view(-1, 1, 1)

    def reward(trajectories, tokens):
        calls["reward"].append(list(tokens))
        return torch.arange(trajectories.shape[0], dtype=trajectories.dtype, device=trajectories.device)

    def prefetch(tokens, per_scene):
        calls["prefetch"] += 1

    reward.prefetch = prefetch
    planner.sample_chain = MethodType(sample_chain, planner)
    planner.get_logprobs = MethodType(get_logprobs, planner)
    planner.reward_fn = reward

    agent = ReCogDriveAgent.__new__(ReCogDriveAgent)
    nn.Module.__init__(agent)
    agent._trajectory_sampling = SimpleNamespace(num_poses=2, interval_length=0.5)
    agent.cache_hidden_state = False
    agent.grpo = True
    agent.grpo_objective = "ppo"
    agent.grpo_num_iterations = 2
    agent.grpo_sample_time = 2
    agent.vlm_trainable = True
    agent.vlm_type = "internvl"
    agent.backbone = TinySharedVLM()
    agent.action_head = planner
    agent.device = "cpu"
    agent._grpo_rollout = None
    agent.grpo_reward = "custom"
    return agent, calls


class LocalContext:
    device = torch.device("cpu")
    rank = 0
    world_size = 1
    is_main = True

    @staticmethod
    def all_reduce_mean(value):
        return value

    def close(self):
        pass


def _loop_trainer():
    agent, calls = _mixed_agent()
    trainer = Trainer.__new__(Trainer)
    trainer.args = TrainArgs(max_epochs=1, grad_accum=1, log_every=100)
    trainer.ctx = LocalContext()
    trainer.agent = agent
    trainer.model = agent
    trainer.optimizer = torch.optim.SGD(agent.parameters(), lr=0.01)
    trainer.lr_scheduler = None
    trainer.lr_interval = "step"
    trainer.ema = None
    trainer.train_loader = OrderedLoader()
    trainer.start_epoch = trainer.start_batch = trainer.global_step = trainer.micro = 0
    trainer.resume_rng = trainer.resume_epoch_rng = None
    trainer.end_epoch = lambda *args: None
    return trainer, calls


def test_vqa_planning_grpo_routes_one_vqa_update_between_planning_rollouts():
    trainer, calls = _loop_trainer()
    optimizer_steps = []
    original_step = trainer.optimizer.step
    trainer.optimizer.step = lambda *args, **kwargs: (optimizer_steps.append(trainer.global_step + 1),
                                                       original_step(*args, **kwargs))[1]
    vlm_gradients = []
    trainer.agent.backbone.lora.weight.register_hook(
        lambda gradient: vlm_gradients.append(gradient.detach().clone()) or gradient
    )

    trainer.train_loop()

    assert len(optimizer_steps) == 5  # 2 PPO planning updates + 1 VQA update + 2 PPO planning updates
    assert calls["sample"] == calls["prefetch"] == 2
    assert calls["reward"] == [["planning-0", "planning-0"], ["planning-2", "planning-2"]]
    assert trainer.agent.backbone.planning_calls == 4
    assert trainer.agent.backbone.vqa_calls == 1
    assert len(vlm_gradients) == 5
    assert all(torch.isfinite(gradient).all() for gradient in vlm_gradients)
    assert all(torch.count_nonzero(gradient).item() for gradient in vlm_gradients)


def _trainer_for_data_guard(agent):
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = OmegaConf.create({"mixture_batch_by_task": True})
    trainer.args = TrainArgs(strategy="ddp", find_unused_parameters=True)
    trainer.agent = agent
    return trainer


class _NavsimPayload(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {}, {}


def _vqa_source():
    source = VQADataset.__new__(VQADataset)
    source.samples = [("q", [], "q", "a")]
    return source


def _waymo_source(cache_path, labelled_only=False, split="training"):
    source = WaymoE2ECacheOnlyDataset.__new__(WaymoE2ECacheOnlyDataset)
    source.cache_path = Path(cache_path).resolve()
    source.split = split
    source.labelled_only = labelled_only
    source.tokens = ["frame-1"]
    return source


def _waymo_reward(reward_type, cache_path, split="training"):
    reward = reward_type.__new__(reward_type)
    reward.cache_path = Path(cache_path).resolve()
    reward.split = split
    return reward


def test_grpo_source_guards_allow_only_task_matched_planning_and_vqa_leaves(tmp_path):
    pdm_module_name = "recogdrive.adapters.navsim.reward"
    pdm_type = type("PDMReward", (), {"__module__": pdm_module_name})
    pdm_module = ModuleType(pdm_module_name)
    pdm_module.PDMReward = pdm_type
    navsim = NavsimSamples(_NavsimPayload(), 2)
    vqa = _vqa_source()
    mixture = MixtureDataset([navsim, vqa], [1.0, 1.0], batch_by_task=True)
    pdm_trainer = _trainer_for_data_guard(
        SimpleNamespace(grpo=True, grpo_reward="navsim_pdm", action_head=SimpleNamespace(reward_fn=pdm_type()))
    )
    with patch.dict("sys.modules", {pdm_module_name: pdm_module}):
        pdm_trainer.validate_train_data(mixture)
        with pytest.raises(ValueError, match="find_unused_parameters=true"):
            pdm_trainer.args.find_unused_parameters = False
            pdm_trainer.validate_train_data(mixture)
        pdm_trainer.args.find_unused_parameters = True
        waymo = _waymo_source(tmp_path)
        with pytest.raises(ValueError, match="NAVSIM-only"):
            pdm_trainer.validate_train_data(MixtureDataset([NavsimSamples(_NavsimPayload(), 2), waymo],
                                                            [1.0, 1.0], batch_by_task=True))

    waymo_cache = tmp_path / "waymo"
    waymo_cache.mkdir()
    waymo = _waymo_source(waymo_cache)
    waymo_vqa = MixtureDataset([waymo, vqa], [1.0, 1.0], batch_by_task=True)
    ade_trainer = _trainer_for_data_guard(SimpleNamespace(
        grpo=True, grpo_reward="waymo_ade", action_head=SimpleNamespace(
            reward_fn=_waymo_reward(WaymoADEReward, waymo_cache)
        )
    ))
    ade_trainer.validate_train_data(waymo_vqa)

    rfs_trainer = _trainer_for_data_guard(SimpleNamespace(
        grpo=True, grpo_reward="waymo_rfs", action_head=SimpleNamespace(
            reward_fn=_waymo_reward(WaymoRFSReward, waymo_cache)
        )
    ))
    with pytest.raises(ValueError, match="labelled-only"):
        rfs_trainer.validate_train_data(waymo_vqa)
    labelled = _waymo_source(waymo_cache, labelled_only=True)
    rfs_trainer.validate_train_data(MixtureDataset([labelled, vqa], [1.0, 1.0], batch_by_task=True))

    bad_binding = _waymo_source(tmp_path / "other")
    with pytest.raises(ValueError, match="same cache root and split"):
        ade_trainer.validate_train_data(bad_binding)
    unknown_task = SimpleNamespace(sample_tasks=["planning", "vqa", "custom"])
    custom = _trainer_for_data_guard(SimpleNamespace(
        grpo=True, action_head=SimpleNamespace(reward_fn=object())
    ))
    with pytest.raises(ValueError, match="does not support training tasks"):
        custom.validate_train_data(unknown_task)


def test_mixed_grpo_schedule_counts_vqa_once_and_keeps_full_ppo_budget():
    class ScheduleAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.grpo = True
            self.grpo_num_iterations = 2
            self.schedule = None

        def get_optimizers(self, total_steps=None, steps_per_epoch=None):
            self.schedule = (total_steps, steps_per_epoch)
            return torch.optim.SGD(self.parameters(), lr=0.1)

    dataset = MixtureDataset([_NavsimPayload(), _vqa_source()], [1.0, 1.0], batch_by_task=True)
    trainer = Trainer.__new__(Trainer)
    trainer.args = TrainArgs(max_epochs=100, max_steps=2, grad_accum=2)
    trainer.agent = ScheduleAgent()
    trainer.train_loader = SimpleNamespace(dataset=dataset)
    trainer.micro = 0
    sampled_epochs = []

    def batch_counts(epoch):
        sampled_epochs.append(epoch)
        return [2, 1, 2]

    trainer._mixed_grpo_batch_counts = batch_counts

    trainer.build_optimizer()

    assert Trainer._scheduled_steps([2, 1, 2], grad_accum=2) == 3
    assert trainer.agent.schedule == (3, 3)  # max_steps waits until the final full PPO rollout is complete
    assert sampled_epochs == [0]  # stop sizing once the max_steps boundary is known
    assert trainer.micro == 0


def test_single_iteration_mixed_grpo_uses_constant_schedule_without_sampler_scan():
    class ScheduleAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.grpo = True
            self.grpo_num_iterations = 1
            self.schedule = None

        def get_optimizers(self, total_steps=None, steps_per_epoch=None):
            self.schedule = (total_steps, steps_per_epoch)
            return torch.optim.SGD(self.parameters(), lr=0.1)

    dataset = MixtureDataset([_NavsimPayload(), _vqa_source()], [1.0, 1.0], batch_by_task=True)

    class Loader:
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return 4

    trainer = Trainer.__new__(Trainer)
    trainer.args = TrainArgs(max_epochs=2, grad_accum=2)
    trainer.agent = ScheduleAgent()
    trainer.train_loader = Loader(dataset)
    trainer._mixed_grpo_batch_counts = lambda epoch: (_ for _ in ()).throw(AssertionError("sampler scan"))

    trainer.build_optimizer()

    assert trainer.agent.schedule == (4, 2)


def test_fit_sigterm_cancels_eval_waits_checkpoint_and_closes_resources(monkeypatch):
    events = []

    class Evaluator:
        benchmark = "owned"

        def cancel(self):
            events.append("cancel")

        def finish(self, final):  # final evaluation must not start after an interrupted fit
            events.append("final")

    trainer = Trainer.__new__(Trainer)
    trainer.args = TrainArgs()
    trainer.agent = SimpleNamespace()
    trainer.evaluators = [Evaluator()]
    trainer.checkpointer = SimpleNamespace(wait=lambda: events.append("wait"))
    trainer.ctx = SimpleNamespace(close=lambda: events.append("close-pg"))
    trainer.monitor = SimpleNamespace(close=lambda: events.append("close-monitor"))
    trainer.build_data = lambda: None
    trainer.build_optimizer = lambda: None
    trainer.train_loop = lambda: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr("recogdrive.train.trainer.parallelize", lambda *args: object())

    with pytest.raises(KeyboardInterrupt, match="received signal"):
        trainer.fit()

    assert signal.getsignal(signal.SIGTERM) is previous
    assert events == ["cancel", "wait", "close-pg", "close-monitor"]
