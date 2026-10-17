import importlib.util
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from setuptools import Distribution
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parents[1]


def setup_options(path, tree):
    previous = Path.cwd()
    try:
        with patch.dict(os.environ, {"NAVSIM_TREE": tree}), patch("setuptools.setup") as setup:
            runpy.run_path(str(path), run_name="__main__")
            return setup.call_args.kwargs
    finally:
        os.chdir(previous)


class PackageLayoutTests(unittest.TestCase):
    def test_both_install_entries_include_shared_models_and_hydra_configs(self):
        for tree in ("1.1", "2.0"):
            for entry in (ROOT / "setup.py", ROOT / ("navsim" + tree) / "setup.py"):
                with self.subTest(tree=tree, entry=entry):
                    cfg = setup_options(entry, tree)
                    self.assertIn("recogdrive.models.recogdrive", cfg["packages"])
                    self.assertIn("recogdrive.data", cfg["packages"])
                    self.assertIn("navsim.agents.recogdrive", cfg["packages"])
                    self.assertEqual(cfg["version"], tree + ".0")
                    command = build_py(Distribution(cfg))
                    command.ensure_finalized()
                    command.get_data_files()
                    self.assertTrue(any("planning/script/config/training/default_training.yaml" == Path(name).as_posix()
                                        for _, _, _, files in command.data_files for name in files))

    def test_package_payload_contains_drivebench_runtime_resources(self):
        cfg = setup_options(ROOT / "setup.py", "1.1")
        with tempfile.TemporaryDirectory() as output:
            distribution = Distribution(cfg)
            distribution.script_name = str(ROOT / "setup.py")
            command = build_py(distribution)
            command.build_lib = output
            command.ensure_finalized()
            command.run()
            package = Path(output) / "recogdrive/eval/vqa/drivebench"
            for name in ("system_prompt.txt", "visual_description.json"):
                with self.subTest(resource=name):
                    self.assertEqual((package / name).read_bytes(),
                                     (ROOT / "recogdrive/eval/vqa/drivebench" / name).read_bytes())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "tensor dependencies are unavailable")
    def test_legacy_imports_share_the_canonical_model_class(self):
        from recogdrive.models.recogdrive.diffusion_planner import ReCogDriveDiffusionPlanner

        for tree in ("navsim1.1", "navsim2.0"):
            path = ROOT / tree / "navsim/agents/recogdrive/recogdrive_diffusion_planner.py"
            spec = importlib.util.spec_from_file_location("legacy_" + tree.replace(".", "_"), path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertIs(module.ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlanner)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "tensor dependencies are unavailable")
    def test_active_navsim_adapter_runs_cached_policy_on_cpu(self):
        import torch
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from recogdrive.adapters.navsim.agent import ReCogDriveAgent

        policy = ReCogDriveAgent(TrajectorySampling(num_poses=8, interval_length=0.5),
                                 cache_hidden_state=True).cpu().eval()
        features = {"last_hidden_state": torch.zeros(1, 2, 1536),
                    "history_trajectory": torch.zeros(1, 4, 3),
                    "status_feature": torch.zeros(1, 8),
                    "high_command_one_hot": torch.tensor([[0., 1., 0., 0.]])}
        with torch.no_grad():
            predicted = policy(features)["pred_traj"]
        self.assertEqual(tuple(predicted.shape), (1, 8, 3))
        self.assertTrue(torch.isfinite(predicted).all())


if __name__ == "__main__":
    unittest.main()
