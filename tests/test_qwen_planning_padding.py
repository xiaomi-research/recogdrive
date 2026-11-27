import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.feature_extraction_utils import BatchFeature

from recogdrive.adapters.navsim.agent import ReCogDriveAgent
from recogdrive.data.registry import collate
from recogdrive.models.recogdrive import backbone
from recogdrive.models.recogdrive.blocks.attention import Attention
from recogdrive.models.recogdrive.diffusion_planner import ReCogDriveDiffusionPlanner, make_recogdrive_config


def make_planner(method, vlm_width=8):
    torch.manual_seed(51)
    cfg = make_recogdrive_config("small", action_dim=3, action_horizon=4, input_embedding_dim=16,
                                 sampling_method=method, num_inference_steps=3, vlm_hidden_size=vlm_width)
    cfg.hidden_size = 32
    cfg.diffusion_model_cfg.update(num_heads=2, head_dim=8, num_layers=2, output_dim=16)
    planner = ReCogDriveDiffusionPlanner(cfg)
    # A freshly initialized DiT has zero attention gates. Exercise its cross-attention as well as its mean.
    with torch.no_grad():
        for block in planner.model.transformer_blocks:
            block.adaLN_modulation[-1].bias.view(6, -1)[2].fill_(1)
    return planner


def mixed_context():
    torch.manual_seed(13)
    mask = torch.tensor([[False, False, True, True, True], [True, True, True, True, False]])
    states = torch.randn(2, 5, 8)
    actions = BatchFeature({"his_traj": torch.randn(2, 12), "status_feature": torch.randn(2, 8),
                            "action": torch.randn(2, 4, 3), "vl_attention_mask": mask})
    return states, states.masked_fill(~mask.unsqueeze(-1), 20), actions


def test_cached_mask_is_preserved_when_mixing_with_an_unpadded_record():
    batch, _, _ = collate([
        ({"last_hidden_state": torch.zeros(3, 8), "vl_attention_mask": torch.tensor([0, 1, 1])}, {}),
        ({"last_hidden_state": torch.zeros(2, 8)}, {}),
    ])
    assert batch["vl_attention_mask"].tolist() == [[False, True, True], [True, True, False]]


def test_generic_collate_leaves_non_tensor_hidden_metadata_as_a_list():
    batch, _, _ = collate([({"last_hidden_state": None}, {}), ({"last_hidden_state": "external"}, {})])
    assert batch == {"last_hidden_state": [None, "external"]}


def test_manual_attention_honors_the_same_grouped_key_mask_as_sdpa(monkeypatch):
    torch.manual_seed(5)
    attention = Attention(query_dim=16, heads=2, dim_head=8).eval()
    queries, keys = torch.randn(4, 3, 16), torch.randn(2, 5, 16)
    mask = torch.tensor([[0, 1, 0, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)[:, None, None, :]
    expected = attention(queries, keys, attention_mask=mask)
    monkeypatch.delattr(torch.nn.functional, "scaled_dot_product_attention")
    torch.testing.assert_close(attention(queries, keys, attention_mask=mask), expected)


@pytest.mark.parametrize("method", ["flow", "ddim", "ddpm"])
def test_il_and_inference_ignore_padded_features(method):
    planner = make_planner(method)
    states, changed, actions = mixed_context()
    results = []
    initial = torch.randn(2, 4, 3)
    for context in (states, changed):
        torch.manual_seed(35)
        context.requires_grad_()
        loss = planner(context, actions)["loss"]
        grad, = torch.autograd.grad(loss, context)
        assert grad[~actions.vl_attention_mask].count_nonzero() == 0
        assert grad[actions.vl_attention_mask].count_nonzero() > 0
        prediction = planner.get_action(context, actions, init_actions=initial, deterministic=True)["pred_traj"]
        results.append((loss, grad, prediction))
    torch.testing.assert_close(results[0], results[1])


@pytest.mark.parametrize("method", ["flow", "ddim", "ddpm"])
def test_grouped_rollout_and_ppo_reuse_ignore_padding(method, tmp_path):
    planner = make_planner(method)
    checkpoint = tmp_path / "reference.pt"
    torch.save({"state_dict": planner.state_dict()}, checkpoint)
    grpo = planner.config.grpo_cfg
    grpo.reference_policy_checkpoint = str(checkpoint)
    grpo.objective, grpo.num_iterations, grpo.kl_coef = "ppo", 2, 0.01
    planner.init_grpo(grpo)
    planner.reward_fn = lambda poses, tokens: poses[:, -1, 0]
    states, changed, actions = mixed_context()
    rollouts = []
    for context in (states, changed):
        torch.manual_seed(43)
        rollouts.append(planner.collect_grpo_rollout(context, actions, ["a", "b"], sample_time=3))
    for field in ("chains", "old_logprobs", "ref_logprobs", "teacher_chains", "adv_weighted"):
        torch.testing.assert_close(getattr(rollouts[0], field), getattr(rollouts[1], field))
    cached = planner.grpo_loss_from_rollout(rollouts[0])
    live = planner.grpo_loss_from_rollout(rollouts[0], vl_features=changed.requires_grad_())
    for key in ("loss", "policy_loss", "bc_loss", "kl_loss", "ratio_mean"):
        torch.testing.assert_close(cached[key], live[key])
    live["loss"].backward()
    assert changed.grad[~actions.vl_attention_mask].count_nonzero() == 0


@pytest.mark.parametrize("cached", [False, True], ids=["online-left-padding", "legacy-cache-right-padding"])
def test_agent_keeps_short_plan_when_batched_with_long_prompt(cached, monkeypatch):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    # Fixed patches and prompt IDs let a real, tiny Qwen3-VL run offline in both batch shapes.
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"][-1]["text"]

        def __call__(self, text, **kwargs):
            sequences = [torch.tensor([1, 120, 121, 121, 121, 121, 122] + [ord(c) - 93 for c in question])
                         for question in text]
            ids = pad_sequence([seq.flip(0) for seq in sequences], batch_first=True).flip(1)
            return BatchFeature({"input_ids": ids, "attention_mask": ids.ne(0).long(),
                                 "pixel_values": torch.linspace(0, 1, 4 * 48).reshape(4, 48).repeat(len(text), 1),
                                 "image_grid_thw": torch.tensor([[1, 2, 2]]).repeat(len(text), 1)})

    agent = ReCogDriveAgent.__new__(ReCogDriveAgent)
    nn.Module.__init__(agent)
    agent.action_head = make_planner("flow", vlm_width=32)
    vlm = backbone.RecogDriveBackbone.__new__(backbone.RecogDriveBackbone)
    nn.Module.__init__(vlm)
    vlm.device, vlm.model_type, vlm.system_prompt = "cpu", "qwen", ""
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}, pad_token_id=0),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=4,
                           temporal_patch_size=1, spatial_merge_size=1, out_hidden_size=32,
                           num_position_embeddings=16, deepstack_visual_indexes=[]),
        image_token_id=121, video_token_id=123, vision_start_token_id=120, vision_end_token_id=122,
    )
    vlm.tokenizer, vlm.model, vlm.qwen_max_input_tokens = Processor(), Qwen3VLForConditionalGeneration(cfg), None
    monkeypatch.setattr(backbone, "process_vision_info", lambda messages: ([], []))
    agent.device, agent.vlm_type, agent.backbone = "cpu", "qwen", vlm
    agent.cache_hidden_state, agent.grpo, agent.vlm_trainable = cached, False, False
    agent.eval()
    sample = agent.action_head.get_action
    monkeypatch.setattr(agent.action_head, "get_action", lambda ctx, action: sample(
        ctx, action, init_actions=ctx.new_zeros(len(ctx), 4, 3), deterministic=True))

    @torch.no_grad()
    def plan(*questions):
        samples = []
        for question in questions:
            features = {"history_trajectory": torch.zeros(4, 3), "high_command_one_hot": torch.zeros(4),
                        "status_feature": torch.zeros(8)}
            if cached:
                features["last_hidden_state"] = vlm(["frame.jpg"], [question])[0]
            else:
                features.update(vlm_question=question, image_paths=["frame.jpg"])
            samples.append((features, {}, question))
        return agent(collate(samples)[0])["pred_traj"]

    alone = plan("abcd")[0]
    together = plan("abcd", "abcdefghijklmnopqrst")
    torch.testing.assert_close(together[0], alone, rtol=1e-5, atol=1e-4)
    assert not torch.allclose(together[1], alone)
