import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PlannerBenchmarkSmokeTest(unittest.TestCase):
    def test_cpu_smoke_compares_pinned_main_and_current(self):
        reference = os.environ.get("MAIN_REFERENCE_ROOT")
        if not reference:
            candidate = ROOT / ".codex-tasks/20261002-recogdrive-refactor/raw/main_reference"
            if not (candidate / "REFERENCE.json").is_file():
                self.skipTest("export the pinned main source and set MAIN_REFERENCE_ROOT")
            reference = str(candidate)
        tool_spec = importlib.util.spec_from_file_location(
            "_planner_benchmark_for_test", ROOT / "scripts/verify/benchmark_planner.py")
        tool = importlib.util.module_from_spec(tool_spec)
        tool_spec.loader.exec_module(tool)
        tool.reference_package(reference)
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            help_result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/verify/benchmark_planner.py"), "--help"],
                cwd=ROOT, check=True, capture_output=True, text=True, timeout=30,
            )
            self.assertIn("synthetic inputs", help_result.stdout)
            self.skipTest("CPU model smoke requires PyTorch and transformers")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify/benchmark_planner.py"),
             "--device", "cpu", "--main-reference-root", reference,
             "--batch-size", "1", "--tokens", "2", "--warmup", "1", "--iterations", "2"],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=180,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["reference_commit"], "d54404796de7a44ca418b96057e3f8c3de3e8c0d")
        self.assertEqual([model["model"] for model in report["models"]], ["main", "current"])
        self.assertTrue(report["numeric_comparison"]["passed"])
        self.assertIn("synthetic planner-only", report["scope"])
        for model in report["models"]:
            self.assertIn("inference", model["phases"])
            self.assertIn("train", model["phases"])
            self.assertGreater(model["phases"]["train"]["latency_ms_median"], 0)


if __name__ == "__main__":
    unittest.main()
