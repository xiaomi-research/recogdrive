"""Small CPU check for the historical ReCogDrive GRPO objective."""

from types import MethodType, SimpleNamespace

import torch
from torch import nn

from recogdrive.adapters.navsim.agent import ReCogDriveAgent
from recogdrive.models.recogdrive.diffusion_planner import (
    GRPOConfig,
    GRPORollout,
    ReCogDriveDiffusionPlanner,
    clipped_pg_loss,
)


def make_planner(objective, log_probs):
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    nn.Module.__init__(planner)
    planner.config = SimpleNamespace(sampling_method="ddim")
    planner.grpo_objective = objective
    planner.clip_eps = 0.2
    planner.bc_coeff = 0.0
    planner.kl_coef = 0.0
    planner.test_log_probs = nn.Parameter(log_probs.clone())
    planner.get_logprobs = MethodType(
        lambda self, *args, **kwargs: self.test_log_probs.view(-1, 1, 1), planner
    )
    return planner


def make_rollout(log_probs, advantages, old_log_probs):
    return GRPORollout(
        his_rep=torch.zeros(4, 1),
        status_rep=torch.zeros(4, 1),
        vl=torch.zeros(4, 1, 1),
        his=torch.zeros(4, 1),
        status=torch.zeros(4, 1),
        chains=torch.zeros(4, 3, 1, 1),
        old_logprobs=old_log_probs,
        adv_weighted=advantages,
        teacher_chains=None,
        ref_logprobs=None,
        reward_mean=torch.tensor(0.0),
    )


def check_config():
    assert GRPOConfig().objective == "main"
    assert GRPOConfig(objective="ppo", num_iterations=2).num_iterations == 2

    for kwargs in (
        {"objective": "typo"},
        {"objective": "main", "num_iterations": 2},
        {"objective": "main", "kl_coef": 0.01},
    ):
        try:
            GRPOConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid GRPO config accepted: {kwargs}")

    mutated = GRPOConfig()
    mutated.objective = "typo"
    planner = ReCogDriveDiffusionPlanner.__new__(ReCogDriveDiffusionPlanner)
    nn.Module.__init__(planner)
    try:
        planner.init_grpo(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError("planner initialization accepted a mutated invalid objective")


def check_main_loss_and_gradient():
    raw_log_probs = torch.tensor([-9.0, -1.2, -0.7, -0.3, 0.0, 0.2, 1.4, 3.0])
    advantages = torch.tensor([-1.3, -0.7, -0.2, 0.0, 0.25, 0.75, 1.0, 1.4])
    # A deliberately different old policy proves main mode does not apply a PPO ratio.
    old_log_probs = torch.full_like(raw_log_probs, -4.0)
    planner = make_planner("main", raw_log_probs)
    result = planner.grpo_loss_from_rollout(
        make_rollout(raw_log_probs, advantages, old_log_probs)
    )

    expected = -(raw_log_probs.clamp(-5, 2) * advantages).mean()
    torch.testing.assert_close(result["policy_loss"], expected)
    torch.testing.assert_close(result["clip_frac"], torch.tensor(0.0))
    result["loss"].backward()
    expected_grad = torch.where(
        raw_log_probs.ge(-5) & raw_log_probs.le(2),
        -advantages / advantages.numel(),
        torch.zeros_like(advantages),
    )
    torch.testing.assert_close(planner.test_log_probs.grad, expected_grad)


def check_explicit_ppo_mode():
    log_probs = torch.tensor([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
    advantages = torch.tensor([-1.0, -0.5, 0.25, 1.0, -0.75, 0.4, 0.8, -0.2])
    old_log_probs = log_probs - torch.log(torch.tensor(2.0))
    planner = make_planner("ppo", log_probs)
    result = planner.grpo_loss_from_rollout(
        make_rollout(log_probs, advantages, old_log_probs)
    )
    expected, _ = clipped_pg_loss(log_probs, old_log_probs, advantages, planner.clip_eps)
    torch.testing.assert_close(result["policy_loss"], expected)
    torch.testing.assert_close(result["clip_frac"], torch.tensor(1.0))


def check_main_bc_term():
    log_probs = torch.tensor([-9.0, -1.2, -0.7, -0.3, 0.0, 0.2, 1.4, 3.0])
    advantages = torch.tensor([-1.3, -0.7, -0.2, 0.0, 0.25, 0.75, 1.0, 1.4])
    planner = make_planner("main", log_probs)
    planner.bc_coeff = GRPOConfig().bc_coeff
    rollout = make_rollout(log_probs, advantages, torch.zeros_like(log_probs))
    rollout.teacher_chains = torch.zeros(4, 3, 1, 1)
    result = planner.grpo_loss_from_rollout(rollout)

    expected_policy = -(log_probs.clamp(-5, 2) * advantages).mean()
    expected_bc = -log_probs.clamp(-5, 2).mean()
    torch.testing.assert_close(result["policy_loss"], expected_policy)
    torch.testing.assert_close(result["bc_loss"], expected_bc)
    torch.testing.assert_close(result["loss"], expected_policy + planner.bc_coeff * expected_bc)


def check_main_advantage_recipe():
    planner = make_planner("main", torch.zeros(8))
    planner.sample_time = 2
    planner.gamma_denoising = 0.5
    planner.clip_advantage_lower_quantile = 0.0
    planner.clip_advantage_upper_quantile = 1.0
    chains = torch.zeros(4, 3, 1, 1)
    trajectories = torch.zeros(4, 1, 3)
    planner.sample_chain = MethodType(
        lambda self, *args, **kwargs: (chains, trajectories), planner
    )
    planner.reward_fn = lambda *_: torch.tensor([1.0, 3.0, 2.0, 6.0])
    batch = SimpleNamespace(his_traj=torch.zeros(2, 1), status_feature=torch.zeros(2, 1))

    rollout = planner.collect_grpo_rollout(
        torch.zeros(2, 1, 1), batch, ["scene-a", "scene-b"]
    )
    rewards = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
    advantages = (
        (rewards - rewards.mean(dim=1, keepdim=True))
        / (rewards.std(dim=1, keepdim=True) + 1e-8)
    ).reshape(-1)
    discount = torch.tensor([0.5, 1.0])
    expected = (advantages.view(4, 1) * discount.view(1, 2)).reshape(-1)
    torch.testing.assert_close(rollout.adv_weighted, expected)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 4, bias=False)
        self.calls = 0

    def forward(self, pixels, questions, num_patches_list=None):
        self.calls += 1
        return self.projection(pixels).unsqueeze(1)


def make_trainable_agent_for_grpo():
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
    calls = {"sample": 0, "reward": 0}

    def sample_chain(self, vl, his, status, **kwargs):
        calls["sample"] += 1
        # Different trajectories give the normalized group advantages different score-function gradients.
        chains = torch.zeros(2, 3, 2, 3)
        chains[0].fill_(1.0)
        chains[1].fill_(2.0)
        return chains, chains[:, -1]

    def get_logprobs(self, vl, his, status, chains, deterministic=False):
        groups = his.shape[0] // vl.shape[0]
        values = vl.mean((1, 2)).repeat_interleave(groups)
        coefficient = chains[:, 0, 0, 0]
        per_trajectory = self.policy_scale * values * coefficient
        return per_trajectory.repeat_interleave(chains.shape[1] - 1).view(-1, 1, 1)

    def reward(trajectories, tokens):
        calls["reward"] += 1
        return torch.tensor([0.0, 1.0], device=trajectories.device)

    reward.prefetch = lambda tokens, per_scene: None
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
    agent.backbone = TinyVLM()
    agent.action_head = planner
    agent.device = "cpu"
    agent._grpo_rollout = None
    return agent, calls


def check_trainable_vlm_receives_grpo_gradients_across_inner_updates():
    agent, calls = make_trainable_agent_for_grpo()
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.01)
    features = {
        "task": ["planning"],
        "history_trajectory": torch.zeros(1, 4, 3),
        "high_command_one_hot": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        "status_feature": torch.zeros(1, 8),
        "pixel_values": torch.tensor([[1.0, 2.0]]),
    }
    targets = {"trajectory": torch.zeros(1, 2, 3)}

    for inner in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = agent(features, targets, ["scene"], reuse_rollout=inner > 0)
        result["loss"].backward()
        gradient = agent.backbone.projection.weight.grad
        assert gradient is not None and torch.count_nonzero(gradient).item() > 0
        optimizer.step()

    assert agent.backbone.calls == 2
    assert calls == {"sample": 1, "reward": 1}
    agent.clear_grpo_rollout()


def test_trainable_vlm_receives_grpo_gradients_across_inner_updates():
    check_trainable_vlm_receives_grpo_gradients_across_inner_updates()


if __name__ == "__main__":
    check_config()
    check_main_loss_and_gradient()
    check_explicit_ppo_mode()
    check_main_bc_term()
    check_main_advantage_recipe()
    check_trainable_vlm_receives_grpo_gradients_across_inner_updates()
    print("main GRPO compatibility checks passed")
