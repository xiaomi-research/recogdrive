"""CPU-only LoRA merge before handing a checkpoint to the existing VQA evaluator."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

MODEL_TOKEN = "__RECOGDRIVE_VQA_MODEL__"
CUDA_ENV = "RECOGDRIVE_VQA_CUDA_VISIBLE_DEVICES"
BENCHES = {"drivelm", "lingoqa", "drivebench"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(checkpoint: Path, agent_config: Path) -> str:
    return hashlib.sha256(
        f"vqa-lora-export-v1:{_sha256(checkpoint)}:{_sha256(agent_config)}".encode("ascii")
    ).hexdigest()


def _valid_export(path: Path, key: str) -> bool:
    try:
        metadata = json.loads((path / "recogdrive-export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    return metadata.get("fingerprint") == key and (path / "config.json").is_file() and has_weights


def load_checkpoint_adapters(agent, state: dict) -> None:
    """Load every LoRA tensor from the planner checkpoint, overriding injected defaults exactly."""
    from recogdrive.distributed.checkpoint import canonical

    expected = {name: value for name, value in agent.state_dict().items()
                if name.startswith("backbone.model.") and ".lora_" in name}
    adapters = {}
    for name, value in state.items():
        name = canonical(name)
        name = name[len("agent."):] if name.startswith("agent.") else name
        if name.startswith("backbone.model.") and ".lora_" in name:
            if name in adapters:
                raise ValueError(f"duplicate LoRA tensor in checkpoint: {name}")
            adapters[name] = value
    missing, unexpected = expected.keys() - adapters.keys(), adapters.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(f"checkpoint LoRA keys do not match agent config; missing={sorted(missing)[:5]}, "
                         f"unexpected={sorted(unexpected)[:5]}")
    mismatched = [name for name in expected if expected[name].shape != adapters[name].shape]
    if mismatched:
        raise ValueError(f"checkpoint LoRA tensor shapes do not match: {mismatched[:5]}")
    agent.load_state_dict(adapters, strict=False)


def _instantiate_agent(config_path: Path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    config.checkpoint_path = None
    config.grpo = False
    if not OmegaConf.select(config, "lora_rank"):
        raise ValueError("VQA LoRA export requires agent.lora_rank > 0")
    return instantiate(config)


def _save_pretrained_unwrapped(model, destination: Path) -> None:
    """Save this fresh, unwrapped HF model without probing distributed runtimes."""
    import transformers.modeling_utils as modeling_utils

    if (hasattr(model, "module") or hasattr(model, "_orig_mod") or
            type(model).__name__ in {"DistributedDataParallel", "FullyShardedDataParallel"}):
        raise TypeError("LoRA export expects the raw HF model, not a distributed or compiled wrapper")

    original = modeling_utils.unwrap_model

    def identity(wrapped, recursive=False):
        return wrapped

    modeling_utils.unwrap_model = identity
    try:
        model.save_pretrained(destination, safe_serialization=True, max_shard_size="10GB")
    finally:
        modeling_utils.unwrap_model = original


def _export_to(checkpoint: Path, agent_config: Path, destination: Path, key: str) -> None:
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("LoRA VQA export must run with CUDA hidden; refusing to merge on a GPU")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved.get("state_dict") if isinstance(saved, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint} has no state_dict")

    agent = _instantiate_agent(agent_config)
    backbone = getattr(agent, "backbone", None)
    if backbone is None or not callable(getattr(backbone, "merge_lora_for_export", None)):
        raise RuntimeError("agent backbone does not support safe LoRA merge for export")
    if any(parameter.device.type != "cpu" for parameter in backbone.model.parameters()):
        raise RuntimeError("LoRA VQA export loaded VLM tensors off CPU")
    load_checkpoint_adapters(agent, state)
    model = backbone.merge_lora_for_export()
    if any(".lora_" in name for name in model.state_dict()):
        raise RuntimeError("LoRA adapter tensors remain after merge; refusing to export an incomplete VLM")
    _save_pretrained_unwrapped(model, destination)
    if getattr(backbone, "tokenizer", None) is not None:
        backbone.tokenizer.save_pretrained(destination)
    if not (destination / "config.json").is_file():
        raise RuntimeError("HF export did not write config.json")
    if not (any(destination.glob("*.safetensors")) or any(destination.glob("pytorch_model*.bin"))):
        raise RuntimeError("HF export did not write model weights")
    (destination / "recogdrive-export.json").write_text(
        json.dumps({"fingerprint": key}, sort_keys=True), encoding="utf-8"
    )


@contextmanager
def _cache_lock(root: Path):
    """Serialize exports so three VQA benchmarks do not load another full VLM at once."""
    # ponytail: one cache-wide lock limits concurrent exports; use per-fingerprint locks if export throughput matters.
    lock_path = root / ".export.lock"
    with lock_path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def acquire_export(checkpoint: Path, agent_config: Path, cache_root: Path):
    cache_root.mkdir(parents=True, exist_ok=True)
    key = fingerprint(checkpoint, agent_config)
    target = cache_root.resolve() / key
    with _cache_lock(cache_root):
        if target.is_symlink():
            raise RuntimeError(f"refusing symlinked LoRA VQA export cache: {target}")
        if target.exists() and not _valid_export(target, key):
            raise RuntimeError(f"LoRA VQA export cache is incomplete or stale: {target}")
        if not target.exists():
            staging = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
            try:
                _export_to(checkpoint, agent_config, staging, key)
                staging.rename(target)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        if not _valid_export(target, key):
            raise RuntimeError(f"LoRA VQA export did not produce a valid cache: {target}")
        leases_path = cache_root / "active-exports.json"
        try:
            leases = json.loads(leases_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            leases = {}
        leases[key] = int(leases.get(key, 0)) + 1
        temporary = leases_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(leases, sort_keys=True), encoding="utf-8")
        os.replace(temporary, leases_path)
    return key, target


def release_export(key: str, target: Path, cache_root: Path) -> None:
    with _cache_lock(cache_root):
        leases_path = cache_root / "active-exports.json"
        leases = json.loads(leases_path.read_text(encoding="utf-8"))
        count = int(leases.get(key, 0)) - 1
        if count > 0:
            leases[key] = count
        else:
            leases.pop(key, None)
            root = cache_root.resolve()
            if target.is_symlink() or target.resolve().parent != root or not _valid_export(target, key):
                raise RuntimeError(f"refusing to remove invalid LoRA VQA export cache: {target}")
            shutil.rmtree(target)
        temporary = leases_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(leases, sort_keys=True), encoding="utf-8")
        os.replace(temporary, leases_path)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        separator = argv.index("--")
    except ValueError as exc:
        raise ValueError("expected `--` followed by the ordinary VQA evaluator arguments") from exc
    parser = argparse.ArgumentParser(prog="python -m recogdrive.eval.vqa.export")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--agent-config", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    args = parser.parse_args(argv[:separator])
    vqa_args = argv[separator + 1:]
    if not vqa_args or vqa_args[0] not in BENCHES:
        raise ValueError(f"expected one of the supported VQA benchmarks: {sorted(BENCHES)}")
    if "--model" not in vqa_args or vqa_args[vqa_args.index("--model") + 1] != MODEL_TOKEN:
        raise ValueError("VQA evaluator command is missing the model placeholder")

    # The wrapper runs with CUDA hidden; restore the selected evaluation devices only for lmdeploy.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    key, model_path = acquire_export(args.checkpoint, args.agent_config, args.cache_root)
    model_index = vqa_args.index("--model") + 1
    vqa_args[model_index] = str(model_path)
    eval_env = os.environ.copy()
    selected_gpus = eval_env.pop(CUDA_ENV, "__unset__")
    if selected_gpus == "__unset__":
        eval_env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        eval_env["CUDA_VISIBLE_DEVICES"] = selected_gpus
    try:
        return subprocess.run([sys.executable, "-m", "recogdrive.eval.vqa", *vqa_args], env=eval_env).returncode
    finally:
        release_export(key, model_path, args.cache_root)


if __name__ == "__main__":
    raise SystemExit(main())
