"""Real planner GRPO-to-inference checkpoint roundtrip through DCP on CPU."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import recogdrive.adapters.navsim.reward as reward_module
from recogdrive.adapters.navsim.agent import ReCogDriveAgent
from recogdrive.distributed.checkpoint import export_model_state
from recogdrive.infer import load_weights, vlm_export_path
from recogdrive.train.trainer import Trainer


def test_initialize_rejects_wrong_planner_size_before_changing_weights(tmp_path):
    agent = torch.nn.Module()
    agent.action_head = torch.nn.Linear(3, 2)
    agent.backbone = torch.nn.Linear(2, 2).requires_grad_(False)
    agent.checkpoint_path = tmp_path / "planner.ckpt"
    before = agent.action_head.bias.detach().clone()
    state = {"agent.action_head.weight": torch.ones(2, 1),
             "agent.action_head.bias": torch.full((2,), 7.0)}
    torch.save({"state_dict": state}, agent.checkpoint_path)
    with pytest.raises(ValueError, match="missing or incompatible weights.*action_head.weight"):
        ReCogDriveAgent.initialize(agent)
    torch.testing.assert_close(agent.action_head.bias, before, rtol=0, atol=0)

    state["agent.action_head.weight"] = torch.ones(2, 3)
    torch.save({"state_dict": state}, agent.checkpoint_path)
    ReCogDriveAgent.initialize(agent)
    torch.testing.assert_close(agent.action_head.weight, state["agent.action_head.weight"], rtol=0, atol=0)
    torch.testing.assert_close(agent.action_head.bias, state["agent.action_head.bias"], rtol=0, atol=0)


def test_real_grpo_planner_checkpoint_loads_for_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class FakePDMReward:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(reward_module, "PDMReward", FakePDMReward)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    owned_group = not dist.is_initialized()
    try:
        torch.manual_seed(314159)
        grpo_agent = ReCogDriveAgent(
            TrajectorySampling(time_horizon=4, interval_length=0.5),
            grpo=True,
            cache_hidden_state=True,
            cache_mode=False,
            grpo_reward_workers=0,
            reference_policy_checkpoint=str(tmp_path / "missing_reference.ckpt"),
            vlm_hidden_size=1536,
        ).cpu()
        assert grpo_agent.backbone is None
        assert hasattr(grpo_agent.action_head, "old_policy")

        torch.manual_seed(314159)
        inference_agent = ReCogDriveAgent(
            TrajectorySampling(time_horizon=4, interval_length=0.5),
            grpo=False,
            cache_hidden_state=True,
            cache_mode=False,
            vlm_hidden_size=1536,
        ).cpu()
        assert inference_agent.backbone is None
        assert not hasattr(inference_agent.action_head, "old_policy")

        if owned_group:
            rendezvous = (tmp_path / "gloo-rendezvous").as_uri()
            dist.init_process_group(
                "gloo", init_method=rendezvous, rank=0, world_size=1,
                timeout=timedelta(seconds=60),
            )
        assert dist.is_initialized()

        state = export_model_state(grpo_agent)
        teacher_prefixes = ("action_head.old_policy.", "action_head.ref_policy.")
        dcp_teacher_keys = {key for key in state if key.startswith(teacher_prefixes)}
        assert dcp_teacher_keys, "DCP must exercise the frozen GRPO teacher alias/buffer filtering"

        trainer = Trainer.__new__(Trainer)
        trainer.agent = grpo_agent
        checkpoint = tmp_path / "grpo.ckpt"
        trainer.write_ckpt(state, checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["vlm_export"] is False
        assert not dcp_teacher_keys.intersection(payload["state_dict"])
        assert vlm_export_path(checkpoint, payload) is None

        load_weights(inference_agent, checkpoint)
        exported_actor = {
            key: value for key, value in payload["state_dict"].items()
            if key.startswith("action_head.")
        }
        loaded_actor = inference_agent.action_head.state_dict()
        assert exported_actor
        for key, value in exported_actor.items():
            actor_key = key[len("action_head."):]
            assert actor_key in loaded_actor, actor_key
            torch.testing.assert_close(loaded_actor[actor_key], value, rtol=0, atol=0, msg=actor_key)

        generator = torch.Generator(device="cpu").manual_seed(2718)
        features = {
            "history_trajectory": torch.randn((1, 4, 3), generator=generator),
            "high_command_one_hot": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "status_feature": torch.randn((1, 8), generator=generator),
            "last_hidden_state": torch.randn((1, 2, 1536), generator=generator),
        }
        grpo_agent.eval()
        inference_agent.eval()
        with torch.no_grad():
            torch.manual_seed(1618)
            expected = grpo_agent(features)["pred_traj"]
            torch.manual_seed(1618)
            actual = inference_agent(features)["pred_traj"]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        if owned_group and dist.is_initialized():
            dist.destroy_process_group()
        torch.set_num_threads(old_threads)
