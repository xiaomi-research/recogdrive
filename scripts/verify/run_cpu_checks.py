#!/usr/bin/env python3
"""Run both NAVSIM trees' regressions and real CPU train/export/inference checks.

Requires the existing NAVSIM environment, nuScenes mini and AlpaSim stubs.
Any skipped regression fails this acceptance gate. Outputs stay in a new directory.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-reference-root", required=True, type=Path)
    parser.add_argument("--nuscenes-root", required=True, type=Path)
    parser.add_argument("--pydeps", type=Path, help="directory containing alpasim_grpc")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("planner_check", ROOT / "scripts/verify/benchmark_planner.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    reference = verifier.reference_package(args.main_reference_root)
    dataset_root = args.nuscenes_root.resolve()
    if not (dataset_root / "v1.0-mini/scene.json").is_file():
        raise FileNotFoundError(f"nuScenes mini is missing under {dataset_root}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
                "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="", TORCH_COMPILE_DISABLE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONFAULTHANDLER="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MAIN_REFERENCE_ROOT=str(reference), NUSCENES_ROOT=str(dataset_root))
    reports = []

    def run(command, directory, name):
        log = directory / (name + ".log")
        print(f"Running {name}; log: {log}", flush=True)
        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            env["MASTER_PORT"] = str(port_probe.getsockname()[1])
        env["MASTER_ADDR"] = "127.0.0.1"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError(f"{name} failed (exit {result.returncode}); see {log}")

    for tree in ("1.1", "2.0"):
        directory = output / tree
        directory.mkdir()
        paths = [str(ROOT / ("navsim" + tree)), str(ROOT)]
        if args.pydeps:
            paths.append(str(args.pydeps.resolve()))
        env["PYTHONPATH"] = os.pathsep.join(paths)
        junit = directory / "pytest.xml"
        run([sys.executable, "-m", "pytest", "-q", "-ra", "tests", f"--junitxml={junit}"],
            directory, "pytest")
        cases = ET.parse(junit).getroot().findall(".//testcase")
        skipped = [case.attrib.get("name") for case in cases if case.find("skipped") is not None]
        if skipped or not cases:
            raise RuntimeError(f"{tree} acceptance requires executed tests; skipped={skipped}, cases={len(cases)}")
        scripts = ROOT / ("navsim" + tree) / "navsim/planning/script"
        training = directory / "training"
        common = ["agent=recogdrive_template", "train_test_split=mini", "data_loader=nuscenes",
                  f"nuscenes.root={dataset_root}", "nuscenes.version=v1.0-mini",
                  "dataloader.params.batch_size=32", "dataloader.params.num_workers=0", "seed=123",
                  "evaluator.name=null"]
        run([sys.executable, str(scripts / "run_recogdrive_train.py"), *common,
             "train.strategy=ddp", "train.precision=fp32", "train.compile=false",
             "train.activation_checkpointing=[]", "trainer.params.max_epochs=1",
             "trainer.params.accumulate_grad_batches=2", "train.log_every=1",
             f"output_dir={training}", f"hydra.run.dir={training}"], directory, "train")
        checkpoint = training / "last.ckpt"
        if not checkpoint.is_file() or not list(training.glob("checkpoints/step_*/trainer_state.pt")):
            raise RuntimeError(f"{tree} training did not produce export and completed resume state")
        inference = directory / "inference"
        run([sys.executable, str(scripts / "run_recogdrive_infer.py"), *common,
             f"infer.checkpoint={checkpoint}", f"output_dir={inference}",
             f"hydra.run.dir={inference}"], directory, "infer")
        predictions = json.loads((inference / "predictions.json").read_text(encoding="utf-8"))
        sys.path[:0] = paths
        from recogdrive.data.nuscenes import NuScenesDataset
        dataset = NuScenesDataset(dataset_root, "v1.0-mini", "val", 8, [])
        expected = set(dataset.index)
        if set(predictions) != expected or not all(
            len(poses) == 8 and all(len(pose) == 3 and all(math.isfinite(v) for v in pose) for pose in poses)
            for poses in predictions.values()
        ):
            raise RuntimeError(f"{tree} inference has missing tokens or invalid trajectories")
        reports.append({"tree": tree, "tests_passed": len(cases), "tests_skipped": 0,
                        "val_predictions": len(predictions), "checkpoint": str(checkpoint)})
    (output / "acceptance.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
