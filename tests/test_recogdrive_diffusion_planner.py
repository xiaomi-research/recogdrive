"""Focused tests for the diffusion planner's schedule bookkeeping.

The full ReCogDrive planner imports the NAVSIM and NuPlan runtime, which is
not needed to exercise its tensor-schedule logic.  These tests provide small
stand-ins for those integration modules and execute the planner methods with
real PyTorch tensors.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    if "." not in name or name.rsplit(".", 1)[-1] in {
        "navsim",
        "recogdrive_isolated",
        "common",
        "evaluate",
        "planning",
        "simulation",
        "planner",
        "pdm_planner",
        "scoring",
        "nuplan",
        "timm",
        "models",
        "layers",
        "blocks",
    }:
        module.__path__ = []
    sys.modules[name] = module
    return module


def _load_planner_module():
    """Load the planner with only the integration imports replaced."""

    previous_modules = dict(sys.modules)

    class IdentityModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, *args, **kwargs):
            return x

    class BatchFeature(dict):
        def __init__(self, data=None, **kwargs):
            super().__init__(data or {}, **kwargs)

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    class PretrainedConfig:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class TrajectorySampling:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class PDMScorerConfig:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    _module("timm")
    _module("timm.models")
    _module("timm.models.layers", Mlp=IdentityModule)
    _module("transformers", PretrainedConfig=PretrainedConfig)
    _module(
        "transformers.feature_extraction_utils",
        BatchFeature=BatchFeature,
    )

    # NAVSIM/NuPlan symbols are only used by methods outside these focused
    # schedule tests, so lightweight placeholders keep import-time coupling
    # out of the test environment.
    _module("navsim")
    _module("navsim.common")
    _module("navsim.common.dataclasses", Trajectory=object)
    _module("navsim.common.dataloader", MetricCacheLoader=object)
    _module("navsim.evaluate")
    _module(
        "navsim.evaluate.pdm_score",
        pdm_score=lambda *args, **kwargs: None,
    )
    _module("navsim.planning")
    _module("navsim.planning.simulation")
    _module("navsim.planning.simulation.planner")
    _module("navsim.planning.simulation.planner.pdm_planner")
    _module(
        "navsim.planning.simulation.planner.pdm_planner.scoring",
    )
    _module(
        "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer",
        PDMScorer=object,
        PDMScorerConfig=PDMScorerConfig,
    )
    _module(
        "navsim.planning.simulation.planner.pdm_planner.simulation",
    )
    _module(
        "navsim.planning.simulation.planner.pdm_planner.simulation."
        "pdm_simulator",
        PDMSimulator=object,
    )
    _module("nuplan")
    _module("nuplan.planning")
    _module("nuplan.planning.simulation")
    _module("nuplan.planning.simulation.trajectory")
    _module(
        "nuplan.planning.simulation.trajectory.trajectory_sampling",
        TrajectorySampling=TrajectorySampling,
    )

    package_name = "recogdrive_isolated"
    _module(package_name)
    _module(f"{package_name}.blocks")
    _module(
        f"{package_name}.blocks.encoder",
        ActionEncoder=IdentityModule,
        SinusoidalPositionalEncoding=IdentityModule,
        StateAttentionEncoder=IdentityModule,
        SwiGLUFFN=IdentityModule,
    )
    _module(f"{package_name}.recogdrive_dit", LightningDiT=IdentityModule)

    source = (
        Path(__file__).parents[1]
        / "navsim"
        / "agents"
        / "recogdrive"
        / "recogdrive_diffusion_planner.py"
    )
    module_name = f"{package_name}.recogdrive_diffusion_planner"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Restore the import table so these stand-ins cannot affect unrelated
        # tests that import the real NAVSIM stack afterwards.
        for name in list(sys.modules):
            if name == module_name:
                continue
            if name not in previous_modules:
                del sys.modules[name]
            elif sys.modules[name] is not previous_modules[name]:
                sys.modules[name] = previous_modules[name]
    return module


@pytest.fixture(scope="module")
def planner_module():
    return _load_planner_module()


def _new_ddpm_planner(module, *, train_steps=12, inference_steps=3):
    planner = object.__new__(module.ReCogDriveDiffusionPlanner)
    nn.Module.__init__(planner)
    planner.config = SimpleNamespace(
        sampling_method="ddpm",
        num_inference_steps=inference_steps,
        action_horizon=1,
    )
    planner._init_ddpm_sampler(
        module.DDPMConfig(num_train_timesteps=train_steps)
    )
    return planner


def test_ddpm_initialization_exposes_schedule_length(planner_module):
    planner = _new_ddpm_planner(planner_module)

    torch.manual_seed(0)
    sampled = planner.sample_time(
        256, device=torch.device("cpu"), dtype=torch.float32
    )

    assert planner.ddpm_num_train_timesteps == 12
    assert sampled.shape == (256,)
    assert sampled.dtype == torch.long
    assert int(sampled.min()) >= 0
    assert int(sampled.max()) < planner.ddpm_num_train_timesteps
    assert planner._ddpm_timesteps() == [8, 4, 0]


def test_ddpm_logprobs_reuse_reverse_sampling_schedule(planner_module):
    planner = _new_ddpm_planner(planner_module)
    planner.min_logprob_denoising_std = 0.1
    planner.feature_encoder = lambda features: features
    planner.his_traj_encoder = lambda features: features
    planner.ego_status_encoder = lambda features: features

    seen_timesteps = []

    def fake_p_mean_variance(
        self,
        x,
        t,
        index,
        vl_features,
        his_traj_features,
        ego_status_features,
        deterministic=True,
    ):
        seen_timesteps.append(t.detach().clone())
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    planner.p_mean_variance = types.MethodType(fake_p_mean_variance, planner)

    batch_size = 2
    chains = torch.zeros(batch_size, 4, 1, 1)
    log_probs = planner.get_logprobs(
        torch.zeros(batch_size, 1, 1),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1),
        chains,
    )

    assert log_probs.shape == (batch_size * 3, 1, 1)
    assert len(seen_timesteps) == 1
    assert seen_timesteps[0].tolist() == [8, 4, 0, 8, 4, 0]
