"""CPU eager regression against the unmodified main-branch planner source.

This locks the default small, 8-waypoint DDIM/IL and inference behavior,
including a test-only nonzero DiT modulation state. GRPO's legacy loss and
gradient are covered separately at formula scope in test_main_grpo_compat.py.
Flow and numerically stabilized DDPM paths are extensions and are deliberately
not claimed as bitwise main matches here.
For Python 3.9, the reference loader defers type annotations only; source text
and algorithm bodies remain unchanged.
"""

import __future__
import copy
import importlib.util
import lzma
import os
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from transformers.feature_extraction_utils import BatchFeature

try:
    import torch._dynamo
except ImportError:
    torch._dynamo = None

from recogdrive.models.recogdrive.diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    make_recogdrive_config,
)


def load_main_planner_module():
    root_value = os.environ.get("MAIN_REFERENCE_ROOT")
    if not root_value:
        raise RuntimeError("Set MAIN_REFERENCE_ROOT to the exported main recogdrive package.")
    root = Path(root_value).resolve()
    package_dir = next(
        (
            path
            for path in (root / "recogdrive", root / "navsim" / "agents" / "recogdrive", root)
            if (path / "recogdrive_diffusion_planner.py").is_file()
        ),
        None,
    )
    if package_dir is None:
        raise FileNotFoundError(
            f"No main planner under {root}; export recogdrive_diffusion_planner.py and its relative modules."
        )

    package_name = "_recogdrive_main_reference"
    module_name = f"{package_name}.recogdrive_diffusion_planner"
    if module_name in sys.modules:
        return sys.modules[module_name]

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    blocks_name = f"{package_name}.blocks"
    blocks = types.ModuleType(blocks_name)
    blocks.__path__ = [str(package_dir / "blocks")]
    blocks.__package__ = blocks_name
    sys.modules[blocks_name] = blocks

    def load_reference_source(relative_module):
        source_path = package_dir.joinpath(*relative_module.split(".")).with_suffix(".py")
        reference_name = f"{package_name}.{relative_module}"
        spec = importlib.util.spec_from_file_location(reference_name, source_path)
        if spec is None:
            raise ImportError(f"Cannot load main reference module from {source_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[reference_name] = module
        source = source_path.read_text(encoding="utf-8")
        code = compile(
            source,
            str(source_path),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        return module

    for relative_module in (
        "blocks.rmsnorm",
        "blocks.rope",
        "blocks.encoder",
        "blocks.attention",
        "recogdrive_dit",
        "recogdrive_diffusion_planner",
    ):
        module = load_reference_source(relative_module)
    return module


def make_main_config(main_module):
    return main_module.ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8,
            "head_dim": 48,
            "num_layers": 16,
            "output_dim": 512,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=384,
        hidden_size=1024,
        action_dim=3,
        action_horizon=8,
        add_pos_embed=True,
        max_seq_len=8,
        ego_status_encoder_type="mlp",
        sampling_method="ddim",
        num_inference_steps=5,
        grpo=False,
        # This regression exercises network/IL/inference only. The unused old reward config names
        # NAVSIM 1.1's comfortable_weight, which does not exist in NAVSIM 2.0.
        grpo_cfg=main_module.GRPOConfig(scorer_config=None),
        vlm_size="small",
    )


def make_current_planner():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        sampling_method="ddim",
        num_inference_steps=5,
        grpo=False,
        training_target="waypoint",
        vlm_hidden_size=1536,
    )
    config.vlm_size = "small"
    return ReCogDriveDiffusionPlanner(config).cpu()


def make_fixed_batch():
    generator = torch.Generator(device="cpu").manual_seed(20261002)
    vl_features = torch.randn(1, 2, 1536, generator=generator)
    action_input = BatchFeature(
        data={
            "his_traj": torch.randn(1, 12, generator=generator),
            "status_feature": torch.randn(1, 8, generator=generator),
            "action": torch.randn(1, 8, 3, generator=generator),
        }
    )
    init_actions = torch.linspace(-0.75, 0.75, 24).view(1, 8, 3)
    return vl_features, action_input, init_actions


def compare_fixed_main_grpo(main_model, current_model, vl_features, action_input):
    """Compare real DDIM log-prob NN paths with a fixed, scorer-free rollout."""
    rollout_generator = torch.Generator(device="cpu").manual_seed(8675309)
    chains = torch.rand((2, 6, 8, 3), generator=rollout_generator) * 0.8 - 0.4  # B=1, G=2, K=5
    teacher_chains = torch.flip(chains[:1].clone(), dims=(1,))
    rewards = torch.tensor([0.2, 1.4])

    def fixed_sampler(self, vl, his, status, **kwargs):
        return chains, chains[:, -1]

    def fixed_teacher(self, vl, his, status, **kwargs):
        return teacher_chains, teacher_chains[:, -1]

    def bind_samples(model, teacher_name):
        teacher = copy.deepcopy(model)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.sample_chain = types.MethodType(fixed_teacher, teacher)
        setattr(model, teacher_name, teacher)
        model.sample_chain = types.MethodType(fixed_sampler, model)

    bind_samples(main_model, "old_policy")
    bind_samples(current_model, "ref_policy")

    main_model.clip_advantage_lower_quantile = 0.0
    main_model.clip_advantage_upper_quantile = 1.0
    main_model.gamma_denoising = 0.6
    main_model.min_logprob_denoising_std = 0.1
    main_model.metric_cache_loader = types.SimpleNamespace(metric_cache_paths={})
    current_model.grpo_objective = "main"
    current_model.clip_eps = 0.2
    current_model.kl_coef = 0.0
    current_model.bc_coeff = 0.1
    current_model.sample_time = 2
    current_model.gamma_denoising = 0.6
    current_model.min_logprob_denoising_std = 0.1
    current_model.clip_advantage_lower_quantile = 0.0
    current_model.clip_advantage_upper_quantile = 1.0
    current_model.reward_fn = lambda trajectories, tokens: rewards.to(trajectories)

    main_vl = vl_features.detach().clone().requires_grad_()
    current_vl = vl_features.detach().clone().requires_grad_()
    with tempfile.TemporaryDirectory() as directory:
        cache_file = Path(directory) / "scene.xz"
        with lzma.open(cache_file, "wb") as stream:
            pickle.dump({"fixed": True}, stream)
        main_model.metric_cache_loader.metric_cache_paths["scene"] = cache_file
        main_model.reward_fn = lambda trajectories, tokens, cache: rewards.to(trajectories)

        main_result = main_model.forward_grpo(
            main_vl, action_input, ["scene"], sample_time=2, bc_coeff=0.1, use_bc_loss=True
        )
        current_rollout = current_model.collect_grpo_rollout(
            current_vl, action_input, ["scene"], sample_time=2
        )
        current_result = current_model.grpo_loss_from_rollout(
            current_rollout, vl_features=current_vl
        )
        main_result["loss"].backward()
        current_result["loss"].backward()

    # Main repeats each sample's VLM tokens per trajectory; current caches those tokens and
    # expands the projected conditioning inside the planner. FP32 reductions may differ slightly.
    for key in ("loss", "policy_loss", "bc_loss"):
        torch.testing.assert_close(current_result[key], main_result[key], rtol=1e-4, atol=1e-5, msg=key)
    torch.testing.assert_close(current_vl.grad, main_vl.grad, rtol=2e-4, atol=2e-5)
    assert torch.count_nonzero(main_vl.grad).item() > 0

    main_parameters = dict(main_model.named_parameters())
    current_parameters = dict(current_model.named_parameters())
    main_actor = {name for name in main_parameters if not name.startswith("old_policy.")}
    current_actor = {name for name in current_parameters if not name.startswith("ref_policy.")}
    assert main_actor == current_actor
    encoder_name = next(name for name in main_actor if name.startswith("feature_encoder."))
    assert main_parameters[encoder_name].grad is not None
    assert torch.count_nonzero(main_parameters[encoder_name].grad).item() > 0
    for name in sorted(main_actor):
        main_gradient, current_gradient = main_parameters[name].grad, current_parameters[name].grad
        assert (main_gradient is None) == (current_gradient is None), name
        if main_gradient is not None:
            torch.testing.assert_close(current_gradient, main_gradient, rtol=2e-4, atol=2e-5, msg=name)


@unittest.skipUnless(
    os.environ.get("MAIN_REFERENCE_ROOT"),
    "set MAIN_REFERENCE_ROOT to the unmodified main recogdrive package",
)
class MainModelCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.previous_compile_disable = None
        if torch._dynamo is not None:
            cls.previous_compile_disable = torch._dynamo.config.disable
            torch._dynamo.config.disable = True

    @classmethod
    def tearDownClass(cls):
        if torch._dynamo is not None:
            torch._dynamo.config.disable = cls.previous_compile_disable
        torch.set_num_threads(cls.previous_num_threads)

    def test_small_waypoint_ddim_state_il_and_prediction(self):
        main_module = load_main_planner_module()

        torch.manual_seed(1729)
        main_model = main_module.ReCogDriveDiffusionPlanner(make_main_config(main_module)).cpu()
        torch.manual_seed(1729)
        current_model = make_current_planner()

        main_state = main_model.state_dict()
        current_state = current_model.state_dict()
        self.assertEqual(set(main_state), set(current_state))
        for name in main_state:
            self.assertEqual(main_state[name].shape, current_state[name].shape, name)
            torch.testing.assert_close(current_state[name], main_state[name], rtol=0, atol=0, msg=name)

        vl_features, action_input, init_actions = make_fixed_batch()
        main_model.train()
        current_model.train()
        with torch.no_grad():
            torch.manual_seed(31415)
            main_loss = main_model(vl_features, action_input)["loss"]
            torch.manual_seed(31415)
            current_loss = current_model(vl_features, action_input)["loss"]
        torch.testing.assert_close(current_loss, main_loss, rtol=2e-5, atol=2e-6)

        main_model.eval()
        current_model.eval()
        with torch.no_grad():
            torch.manual_seed(2718)
            main_prediction = main_model.get_action(
                vl_features, action_input, init_actions=init_actions.clone(), deterministic=True
            ).pred_traj
            torch.manual_seed(2718)
            current_prediction = current_model.get_action(
                vl_features, action_input, init_actions=init_actions.clone(), deterministic=True
            ).pred_traj
        torch.testing.assert_close(current_prediction, main_prediction, rtol=2e-4, atol=2e-5)

        torch.manual_seed(8675309)
        modulated = []
        with torch.no_grad():
            for name, parameter in main_model.named_parameters():
                if ".adaLN_modulation." in name or ".final_layer.modulation_proj." in name:
                    parameter.copy_(torch.randn_like(parameter) * 1e-3)
                    modulated.append(name)
        self.assertTrue(modulated, "main DiT modulation parameters were not found")
        main_parameters = dict(main_model.named_parameters())
        self.assertTrue(all(torch.count_nonzero(main_parameters[name]).item() > 0 for name in modulated))

        incompatible = current_model.load_state_dict(main_model.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        current_parameters = dict(current_model.named_parameters())
        for name in modulated:
            torch.testing.assert_close(current_parameters[name], main_parameters[name], rtol=0, atol=0)

        main_model.train()
        current_model.train()
        with torch.no_grad():
            torch.manual_seed(31415)
            main_loss = main_model(vl_features, action_input)["loss"]
            torch.manual_seed(31415)
            current_loss = current_model(vl_features, action_input)["loss"]
        torch.testing.assert_close(current_loss, main_loss, rtol=2e-5, atol=2e-6)

        main_model.eval()
        current_model.eval()
        with torch.no_grad():
            torch.manual_seed(2718)
            main_prediction = main_model.get_action(
                vl_features, action_input, init_actions=init_actions.clone(), deterministic=True
            ).pred_traj
            torch.manual_seed(2718)
            current_prediction = current_model.get_action(
                vl_features, action_input, init_actions=init_actions.clone(), deterministic=True
            ).pred_traj
        torch.testing.assert_close(current_prediction, main_prediction, rtol=2e-4, atol=2e-5)

        compare_fixed_main_grpo(main_model, current_model, vl_features, action_input)


if __name__ == "__main__":
    unittest.main()
