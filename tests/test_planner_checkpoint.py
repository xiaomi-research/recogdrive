import pytest
import torch

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent


def test_planner_checkpoint_requires_matching_weights(tmp_path):
    agent = ReCogDriveAgent.__new__(ReCogDriveAgent)
    torch.nn.Module.__init__(agent)
    agent.action_head = torch.nn.Linear(3, 2)
    agent.backbone = torch.nn.Linear(3, 3)
    agent.dit_type = "large"
    agent.vlm_size = "small"
    agent.checkpoint_path = tmp_path / "planner.ckpt"
    state = {"agent.action_head.weight": torch.ones(2, 3), "agent.action_head.bias": torch.zeros(2)}
    torch.save({"state_dict": state}, agent.checkpoint_path)
    agent.initialize()
    torch.testing.assert_close(agent.action_head.weight, state["agent.action_head.weight"])

    state["agent.action_head.weight"] = torch.zeros(2, 4)
    torch.save({"state_dict": state}, agent.checkpoint_path)
    with pytest.raises(ValueError, match="dit_type=large.*action_head.weight"):
        agent.initialize()
    torch.testing.assert_close(agent.action_head.weight, torch.ones(2, 3))


def test_grpo_passes_the_configured_imitation_weight(monkeypatch):
    from navsim.agents.recogdrive import recogdrive_agent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    class Planner(torch.nn.Linear):
        def __init__(self, _):
            super().__init__(1, 1)

        def cuda(self):
            return self

        def forward_grpo(self, features, inputs, tokens, bc_coeff):
            return bc_coeff

    monkeypatch.setattr(recogdrive_agent, "ReCogDriveDiffusionPlanner", Planner)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    agent = ReCogDriveAgent(TrajectorySampling(time_horizon=4, interval_length=0.5), grpo=True, grpo_bc_coeff=1.0)
    features = {"history_trajectory": torch.zeros(1, 4, 3), "high_command_one_hot": torch.zeros(1, 3),
                "status_feature": torch.zeros(1, 7), "last_hidden_state": torch.zeros(1, 2, 1536)}
    assert agent(features, {"trajectory": torch.zeros(1, 8, 3)}, ["scene"]) == 1.0
