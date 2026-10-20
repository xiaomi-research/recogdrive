"""Device selection and dispatch tests for the ReCogDrive inference path.

The detection and dispatch tests need no accelerator. The parity test runs only on a host with an XPU.
"""

import unittest
from unittest import mock

import torch

from navsim.agents.recogdrive.utils import device as dev

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()

try:
    from navsim.agents.recogdrive import recogdrive_agent, recogdrive_backbone
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner

    HAS_NAVSIM = True
except ImportError:
    HAS_NAVSIM = False


def fake_host(*available):
    return mock.patch.object(dev, "is_available", lambda name: name == "cpu" or name in available)


class DeviceModuleTest(unittest.TestCase):
    def test_default_device_type(self):
        for available, expected in [((), "cpu"), (("cuda",), "cuda"), (("xpu",), "xpu"), (("cuda", "xpu"), "xpu")]:
            with self.subTest(available=available), fake_host(*available):
                self.assertEqual(dev.default_device_type(), expected)
                self.assertEqual(dev.resolve_device().type, expected)

    def test_explicit_device_wins(self):
        with fake_host("xpu"):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("xpu:1"), torch.device("xpu", 1))

    def test_bare_device_and_rank_index(self):
        with fake_host("cuda"):
            self.assertEqual(dev.resolve_device(None, 3), torch.device("cuda", 3))
            self.assertEqual(dev.resolve_device("cuda", 2), torch.device("cuda", 2))
        with fake_host():
            self.assertEqual(dev.resolve_device(None, 3), torch.device("cpu"))

    def test_device_type(self):
        self.assertEqual(dev.device_type("xpu:0"), "xpu")
        self.assertEqual(dev.device_type(torch.zeros(1).device), "cpu")


@unittest.skipUnless(HAS_NAVSIM, "navsim and nuplan are required")
class DispatchTest(unittest.TestCase):
    def build_agent(self, device=None, rank="0"):
        sampling = mock.Mock()
        with mock.patch.dict("os.environ", {"LOCAL_RANK": rank}), mock.patch.object(
            recogdrive_agent, "ReCogDriveDiffusionPlanner"
        ) as planner, mock.patch.object(recogdrive_agent, "RecogDriveBackbone") as backbone:
            agent = recogdrive_agent.ReCogDriveAgent(
                sampling, vlm_path="vlm", cache_hidden_state=False, device=device
            )
        return agent, planner, backbone

    def test_agent_detects_xpu(self):
        with fake_host("xpu"):
            agent, planner, backbone = self.build_agent(rank="1")
        self.assertEqual(agent.device, "xpu:1")
        planner.return_value.to.assert_called_once_with("xpu:1")
        self.assertEqual(backbone.call_args.kwargs["device"], "xpu:1")

    def test_agent_keeps_cuda_default(self):
        with fake_host("cuda"):
            agent, planner, backbone = self.build_agent()
        self.assertEqual(agent.device, "cuda:0")
        planner.return_value.to.assert_called_once_with("cuda:0")

    def test_agent_explicit_device_wins(self):
        with fake_host("xpu"):
            agent, _, backbone = self.build_agent(device="cpu")
        self.assertEqual(agent.device, "cpu")
        self.assertEqual(backbone.call_args.kwargs["device"], "cpu")

    def load_backbone(self, device, *available):
        with fake_host(*available), mock.patch.object(
            recogdrive_backbone.AutoModel, "from_pretrained"
        ) as model, mock.patch.object(recogdrive_backbone.AutoTokenizer, "from_pretrained") as tokenizer:
            tokenizer.return_value.convert_tokens_to_ids.return_value = 7
            recogdrive_backbone.RecogDriveBackbone("internvl", "ckpt", device=device)
        return model.call_args.kwargs

    def test_flash_attention_only_on_cuda(self):
        self.assertFalse(self.load_backbone("xpu:0", "xpu")["use_flash_attn"])
        self.assertFalse(self.load_backbone("cpu")["use_flash_attn"])
        self.assertTrue(self.load_backbone("cuda:0", "cuda")["use_flash_attn"])
        kwargs = self.load_backbone(None, "xpu")
        self.assertEqual(kwargs["device_map"], "xpu:0")
        self.assertFalse(kwargs["use_flash_attn"])


@unittest.skipUnless(HAS_NAVSIM and HAS_XPU, "navsim, nuplan and an XPU are required")
class PlannerParityTest(unittest.TestCase):
    def test_sampler_cpu_xpu_parity(self):
        from transformers.feature_extraction_utils import BatchFeature

        cfg = recogdrive_agent.make_recogdrive_config(
            "small", action_dim=3, action_horizon=8, input_embedding_dim=384
        )
        cfg.vlm_size = "small"
        torch.manual_seed(0)
        planner = ReCogDriveDiffusionPlanner(cfg).eval()
        with torch.no_grad():
            for p in planner.parameters():  # no zero-initialised layer may hide a mismatch
                p.copy_(torch.randn_like(p) * 0.05)
        vl = torch.randn(1, 40, 1536)
        his = torch.randn(1, 12)
        status = torch.randn(1, 8)
        noise = torch.randn(1, 8, 3)

        def run(model, device):
            inputs = BatchFeature({"state": torch.cat([status, his], 1).to(device), "his_traj": his.to(device),
                                   "status_feature": status.to(device)})
            with torch.no_grad():
                return model.get_action(vl.to(device), inputs, init_actions=noise.to(device), deterministic=True)[
                    "pred_traj"
                ].cpu()

        ref = run(planner, "cpu")
        self.assertGreater(ref.std().item(), 0.1)
        out = run(planner.to("xpu"), "xpu")
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
