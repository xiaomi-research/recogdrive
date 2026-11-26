from pathlib import Path
from types import SimpleNamespace

import navsim
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from recogdrive.adapters.navsim import agent as agent_module
from recogdrive.adapters.navsim import reward as reward_module


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_grpo_reverse_gate(monkeypatch, enabled, workers):
    def score(token, poses):
        return poses[:, 0, 1].tolist()

    class InProcessPool:
        def __init__(self, *args, initializer, initargs, **kwargs):
            initializer(*initargs)

        def submit(self, fn, *args):
            return SimpleNamespace(result=lambda: fn(*args))

    monkeypatch.setattr(reward_module, "worker_scenes", None)
    monkeypatch.setattr(reward_module, "ProcessPoolExecutor", InProcessPool)
    monkeypatch.setattr(reward_module, "PDMScenes", lambda _: SimpleNamespace(score=score, metric_cache=SimpleNamespace()))
    monkeypatch.setattr(agent_module, "ReCogDriveDiffusionPlanner", lambda cfg: torch.nn.Identity())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config_path = Path(navsim.__file__).parent / "planning/script/config/common/agent/recogdrive_agent.yaml"
    cfg = OmegaConf.load(config_path)
    assert cfg.grpo_reverse_gate is False
    cfg.grpo = True
    cfg.grpo_reward_workers = workers
    if enabled:
        cfg.grpo_reverse_gate = True
    reward = instantiate(cfg).action_head.reward_fn

    # First-step reverse, late reverse, sub-centimetre jitter, and forward motion.
    xs = torch.tensor([[1, 0.9], [-0.02, 1], [1, 0.995], [0, 1]], dtype=torch.float32)
    scores = torch.tensor([0.8, 0.4, 0.7, 0.2])
    trajs = torch.stack([xs, scores[:, None].expand_as(xs), torch.zeros_like(xs)], dim=-1)
    expected = torch.tensor([0, 0, 0.7, 0.2]) if enabled else scores
    torch.testing.assert_close(reward(trajs, ["a", "b", "a", "b"]), expected, rtol=0, atol=0)


def test_grpo_reverse_gate_rejects_waymo_reward():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    with pytest.raises(ValueError, match="grpo_reverse_gate requires grpo_reward=navsim_pdm"):
        agent_module.ReCogDriveAgent(TrajectorySampling(time_horizon=5, interval_length=0.5),
                                     grpo=True, grpo_reward="waymo_ade", grpo_reverse_gate=True)
