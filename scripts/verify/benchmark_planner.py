#!/usr/bin/env python3
"""Compare the pinned main planner with the current planner on synthetic inputs.

This measures planner-only inference or IL forward/backward/AdamW steps. It does
not include a VLM, data loading, distributed training, or driving benchmarks.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MAIN_COMMIT = "d54404796de7a44ca418b96057e3f8c3de3e8c0d"
MAIN_FILES = {
    "recogdrive_diffusion_planner.py": "c349bf196611034ebcdfb3dd30fe02e1ac8ee4fc6a76d65934b70dbf583d994a",
    "recogdrive_dit.py": "1af5c45d5eaa2d7fa2920c6f885c62f94e5cac39af111697a7fd0f22c2e8571e",
    "blocks/attention.py": "c40daea407abe77618eae32020ac03e80c1597a757994fb767cd4ef5aff132ce",
    "blocks/encoder.py": "c0ab0b8c7fe9daa70d05d0fbeeb207d36d259a57541013fced7079ddf2478f6f",
    "blocks/rmsnorm.py": "2257f84f1124e7f26cbd8182c94270b889a7b9a780ce669cdeaa17cfffd42a7b",
    "blocks/rope.py": "5c64f67db28f038b304e3d3b7de96ece44a6ec10d7e081b22afb2ee99875754f",
    "utils/lr_scheduler.py": "8d5b400dbb37e7cd39f1e07d0a681e782e02d8ffe9db64bc214828d0ad2bf3dc",
}


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=(f"Pinned main: {MAIN_COMMIT}. Export its source with "
                "python scripts/verify/export_main_reference.py <directory> --ref " + MAIN_COMMIT),
    )
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--main-reference-root", default=os.environ.get("MAIN_REFERENCE_ROOT"))
    p.add_argument("--mode", choices=("inference", "train", "both"), default="both")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--tokens", type=int, default=8, help="synthetic VLM feature token count")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    p.add_argument("--compile-mode", choices=("none", "default", "reduce-overhead", "max-autotune"), default="none")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--output", type=Path, help="optional JSON output path")
    return p


def reference_package(root):
    root = Path(root).resolve()
    package_dir = next(
        (path for path in (root / "recogdrive", root / "navsim/agents/recogdrive", root)
         if (path / "recogdrive_diffusion_planner.py").is_file()),
        None,
    )
    if package_dir is None:
        raise FileNotFoundError(f"No main planner source under {root}")
    metadata_path = package_dir / "REFERENCE.json"
    if not metadata_path.is_file():
        raise ValueError(f"Pinned reference metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("commit") != MAIN_COMMIT or metadata.get("files") != MAIN_FILES:
        raise ValueError(f"Reference is not the pinned main commit {MAIN_COMMIT}")
    for relative, expected in MAIN_FILES.items():
        source = (package_dir / relative).resolve()
        source.relative_to(package_dir.resolve())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Pinned main source hash mismatch: {relative}")
    return root


def check_idle_cuda():
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value or not value.strip() or value.strip() == "-1":
        raise RuntimeError("CUDA mode requires an explicit non-empty CUDA_VISIBLE_DEVICES")
    selectors = [item.strip() for item in value.split(",")]
    if any(not item for item in selectors) or len(set(selectors)) != len(selectors):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must list distinct visible GPU indices/UUIDs")
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    rows = []
    for row in gpu_result.stdout.splitlines():
        fields = [part.strip() for part in row.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"Could not parse nvidia-smi GPU row: {row!r}")
        rows.append({"index": fields[0], "uuid": fields[1], "memory_mib": int(fields[2]), "util_pct": int(fields[3])})
    selected = []
    for selector in selectors:
        matches = [row for row in rows if selector == row["index"] or row["uuid"].startswith(selector)]
        if len(matches) != 1:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES entry {selector!r} did not identify exactly one GPU")
        selected.append(matches[0])
    app_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    busy = []
    for row in app_result.stdout.splitlines():
        fields = [part.strip() for part in row.split(",")]
        if len(fields) >= 2 and any(gpu["uuid"] == fields[0] for gpu in selected):
            busy.append(row.strip())
    # Fail closed when an allocation is visible but nvidia-smi exposes no process.
    occupied = [gpu for gpu in selected if gpu["memory_mib"] > 512 or gpu["util_pct"] > 5]
    if busy or occupied:
        details = busy or [f"GPU {gpu['index']}: {gpu['memory_mib']} MiB, {gpu['util_pct']}%" for gpu in occupied]
        raise RuntimeError("Refusing to benchmark: selected GPU has an active allocation/compute load: " + "; ".join(details))
    return selected


def load_compat_helpers(reference_root):
    os.environ["MAIN_REFERENCE_ROOT"] = str(reference_root)
    path = ROOT / "tests/test_main_model_compat.py"
    spec = importlib.util.spec_from_file_location("_planner_benchmark_main_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_batch(batch_size, tokens, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(batch_size, tokens, 1536, generator=generator)
    actions = BatchFeature(data={
        "his_traj": torch.randn(batch_size, 12, generator=generator),
        "status_feature": torch.randn(batch_size, 8, generator=generator),
        "action": torch.randn(batch_size, 8, 3, generator=generator),
    })
    init_actions = torch.linspace(-0.75, 0.75, 24).view(1, 8, 3).expand(batch_size, -1, -1).contiguous()
    return features.to(device), BatchFeature(data={key: value.to(device) for key, value in actions.items()}), init_actions.to(device)


def seed_step(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def autocast_context(device, precision):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def gradient_snapshot(model):
    snapshot = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            snapshot[name] = parameter.grad.detach().float().cpu().clone()
    return snapshot


def numeric_delta(reference, candidate):
    if isinstance(reference, dict):
        if reference.keys() != candidate.keys():
            return {"same_keys": False, "finite": False, "max_abs": None,
                    "relative_l2": None, "elements": 0, "tensors": 0}
        max_abs = relative_l2 = 0.0
        elements = 0
        for key in reference:
            a, b = reference[key], candidate[key]
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                return {"same_keys": True, "finite": False, "max_abs": None,
                        "relative_l2": None, "elements": sum(value.numel() for value in reference.values()),
                        "tensors": len(reference)}
            difference = (a - b).abs()
            max_abs = max(max_abs, float(difference.max().item()) if difference.numel() else 0.0)
            relative_l2 = max(relative_l2, float(torch.linalg.vector_norm(difference).item() /
                                                    max(torch.linalg.vector_norm(a).item(), 1e-30)))
            elements += a.numel()
        return {"same_keys": True, "finite": True, "max_abs": max_abs,
                "relative_l2": relative_l2, "elements": elements, "tensors": len(reference)}
    a, b = torch.as_tensor(reference).float().cpu(), torch.as_tensor(candidate).float().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"finite": False, "max_abs": None, "relative_l2": None, "elements": a.numel()}
    return {"max_abs": float((a - b).abs().max().item()),
            "finite": True, "relative_l2": float(torch.linalg.vector_norm(a - b).item() /
                                 max(torch.linalg.vector_norm(a).item(), 1e-30)),
            "elements": a.numel()}


def close_enough(reference, candidate, atol, rtol):
    if isinstance(reference, dict):
        if reference.keys() != candidate.keys():
            return False
        return all(torch.isfinite(reference[key]).all() and torch.isfinite(candidate[key]).all()
                   and torch.allclose(reference[key], candidate[key], atol=atol, rtol=rtol)
                   for key in reference)
    reference, candidate = torch.as_tensor(reference).float().cpu(), torch.as_tensor(candidate).float().cpu()
    return bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()
                and torch.allclose(reference, candidate, atol=atol, rtol=rtol))


def summarize(times_ms, batch_size):
    ordered = sorted(times_ms)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    median = statistics.median(ordered)
    return {"latency_ms_median": median, "latency_ms_p95": p95,
            "samples_per_second": batch_size * 1000.0 / median}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_run(call, device, seed):
    seed_step(seed, device)
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        return start, end
    start = time.perf_counter()
    call()
    return (time.perf_counter() - start) * 1000.0


def event_ms(timing, device):
    if device.type == "cuda":
        start, end = timing
        return start.elapsed_time(end)
    return timing


def peak_memory(device):
    if device.type != "cuda":
        return None
    return {"allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}


def benchmark_phase(model, callable_model, optimizer, features, actions, init_actions, args, device, phase):
    model.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(phase == "train" and device.type == "cuda" and args.precision == "fp16"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    call = (lambda: callable_model(features, actions)) if phase == "train" else (
        lambda: callable_model(features, actions, init_actions=init_actions, deterministic=True))

    for index in range(args.warmup):
        seed_step(args.seed + index + 2000, device)
        if phase == "train":
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                loss = call()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), autocast_context(device, args.precision):
                call()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings = []
    for index in range(args.iterations):
        if phase == "train":
            def step():
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.precision):
                    loss = call()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            timing = timed_run(step, device, args.seed + index + 1000)
        else:
            timing = timed_run(lambda: _run_inference(call, device, args.precision), device, args.seed + index + 1000)
        timings.append(timing)
    synchronize(device)
    return {**summarize([event_ms(item, device) for item in timings], args.batch_size),
            "peak_memory": peak_memory(device)}


def _run_inference(call, device, precision):
    with torch.no_grad(), autocast_context(device, precision):
        call()


def run_variant(label, planner_class, config, base_state, current, args, device, batch):
    features, actions, init_actions = batch
    if current:
        model = planner_class()
        model.load_state_dict(base_state, strict=True)
    else:
        torch.manual_seed(args.seed)
        model = planner_class(config)
        model.load_state_dict(base_state, strict=True)
    model = model.to(device)
    model.train()
    def train_call(features, action_input):
        return model(features, action_input)["loss"]

    def inference_call(features, action_input, init_actions, deterministic=True):
        return model.get_action(features, action_input, init_actions=init_actions,
                                deterministic=deterministic)["pred_traj"]

    calls = {"train": train_call, "inference": inference_call}
    if args.compile_mode != "none":
        if device.type != "cuda":
            raise ValueError("torch.compile benchmark modes require --device cuda")
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile")
        calls = {name: torch.compile(call, mode=args.compile_mode) for name, call in calls.items()}

    result = {"model": label, "phases": {}}
    if args.mode in ("inference", "both"):
        model.eval()
        with torch.no_grad(), autocast_context(device, args.precision):
            inference = calls["inference"](features, actions, init_actions=init_actions, deterministic=True)
        result["prediction"] = inference.float().detach().cpu()
        result["phases"]["inference"] = benchmark_phase(
            model, calls["inference"], None, features, actions, init_actions, args, device, "inference")
    if args.mode in ("train", "both"):
        model.train()
        seed_step(args.seed + 77, device)
        model.zero_grad(set_to_none=True)
        with autocast_context(device, args.precision):
            loss = calls["train"](features, actions)
        loss.backward()
        result["loss"] = float(loss.detach().float().cpu().item())
        result["gradients"] = gradient_snapshot(model)
        model.zero_grad(set_to_none=True)
        model.load_state_dict(base_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        result["phases"]["train"] = benchmark_phase(
            model, calls["train"], optimizer, features, actions, init_actions, args, device, "train")
        del optimizer
    del calls
    model = None  # Keep nested callables' closure cell bound while releasing the model.
    del train_call, inference_call
    gc.collect()
    if args.compile_mode != "none" and hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
        gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return result


def main():
    args = parser().parse_args()
    if min(args.batch_size, args.tokens, args.iterations) < 1 or args.warmup < 0:
        raise SystemExit("batch-size, tokens, iterations must be positive and warmup non-negative")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive")
    if not args.main_reference_root:
        raise SystemExit("Set --main-reference-root or MAIN_REFERENCE_ROOT to the exported pinned main source")
    reference_root = reference_package(args.main_reference_root)
    selected = check_idle_cuda() if args.device == "cuda" else []
    try:
        global torch, BatchFeature
        import torch
        from transformers.feature_extraction_utils import BatchFeature
    except ImportError as exc:
        raise SystemExit("Planner benchmarking requires the repository's PyTorch and transformers environment") from exc
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(selected):
            raise SystemExit("CUDA_VISIBLE_DEVICES does not match the GPUs available to PyTorch")
        torch.cuda.set_device(0)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise SystemExit("The selected GPU does not support bfloat16")
    elif args.precision != "fp32":
        raise SystemExit("CPU smoke mode supports --precision fp32 only")
    try:
        import torch._dynamo
        previous_dynamo_disable = torch._dynamo.config.disable
        # The planner contains internal @torch.compile decorators; "none" disables those too.
        torch._dynamo.config.disable = args.compile_mode == "none"
    except ImportError:
        previous_dynamo_disable = None
    helpers = load_compat_helpers(reference_root)
    main_module = helpers.load_main_planner_module()
    main_config = helpers.make_main_config(main_module)
    torch.manual_seed(args.seed)
    main_model = main_module.ReCogDriveDiffusionPlanner(main_config).cpu()
    base_state = {name: value.detach().cpu().clone() for name, value in main_model.state_dict().items()}
    del main_model
    current_probe = helpers.make_current_planner()
    current_probe.load_state_dict(base_state, strict=True)
    del current_probe
    gc.collect()
    batch = make_batch(args.batch_size, args.tokens, args.seed + 1, device)

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = args.precision != "fp32"
    try:
        main_result = run_variant("main", main_module.ReCogDriveDiffusionPlanner, main_config,
                                  base_state, False, args, device, batch)
        current_result = run_variant("current", helpers.make_current_planner, None,
                                     base_state, True, args, device, batch)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        if previous_dynamo_disable is not None:
            torch._dynamo.config.disable = previous_dynamo_disable

    atol, rtol = ((2e-5, 2e-4) if args.precision == "fp32" else (1e-2, 1e-2))
    comparison = {"atol": atol, "rtol": rtol, "passed": True}
    if "loss" in main_result:
        comparison["loss"] = numeric_delta(main_result["loss"], current_result["loss"])
        # Scalar loss tolerances are checked directly; detailed tensor comparisons are below.
        comparison["loss"]["within_tolerance"] = math.isclose(
            main_result["loss"], current_result["loss"], rel_tol=rtol, abs_tol=atol)
        comparison["gradients"] = numeric_delta(main_result["gradients"], current_result["gradients"])
        comparison["gradients"]["within_tolerance"] = close_enough(
            main_result["gradients"], current_result["gradients"], atol, rtol)
        comparison["passed"] &= comparison["loss"]["within_tolerance"] and comparison["gradients"]["within_tolerance"]
    if "prediction" in main_result:
        comparison["prediction"] = numeric_delta(main_result["prediction"], current_result["prediction"])
        comparison["prediction"]["within_tolerance"] = close_enough(
            main_result["prediction"], current_result["prediction"], atol, rtol)
        comparison["passed"] &= comparison["prediction"]["within_tolerance"]
    for item in (main_result, current_result):
        item.pop("loss", None)
        item.pop("gradients", None)
        item.pop("prediction", None)
    output = {
        "scope": "synthetic planner-only; excludes VLM/backbone, data pipeline, DDP, and simulator evaluation",
        "reference_commit": MAIN_COMMIT,
        "reference_root": str(reference_root),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "current_source_sha256": {
            name: hashlib.sha256((ROOT / "recogdrive/models/recogdrive" / name).read_bytes()).hexdigest()
            for name in ("diffusion_planner.py", "dit.py", "blocks/attention.py", "blocks/encoder.py",
                         "blocks/rmsnorm.py", "blocks/rope.py", "utils/lr_scheduler.py")
        },
        "device": str(device),
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") if device.type == "cuda" else None,
        "selected_gpu_idle_check": selected,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "tokens": args.tokens,
        "precision": args.precision,
        "compile_mode": args.compile_mode,
        "decorated_compile_enabled": args.compile_mode != "none",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "optimizer": {"name": "AdamW", "lr": args.learning_rate, "weight_decay": 0.0} if args.mode != "inference" else None,
        "models": [main_result, current_result],
        "numeric_comparison": comparison,
    }
    serialized = json.dumps(output, indent=2, allow_nan=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
