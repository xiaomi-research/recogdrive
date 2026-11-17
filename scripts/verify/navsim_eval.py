"""Score a saved checkpoint with its training agent configuration."""

import argparse
import json
import signal
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recogdrive.eval.navsim import NavsimEvaluator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-config", "checkpoint", "metric-cache", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", default="navhard_two_stage")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    locked = {"agent", "hydra.searchpath", "train_test_split", "metric_cache_path", "output_dir"}
    for override in args.overrides:
        key = override.lstrip("+~").partition("=")[0]
        if key in locked or key.startswith(("agent.", "train_test_split.")):
            parser.error(f"{key} is fixed by the saved config or this launcher's options")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    metric_cache = args.metric_cache.expanduser().resolve(strict=True)
    if not checkpoint.is_file() or not metric_cache.is_dir():
        parser.error("--checkpoint must be a file and --metric-cache a directory")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must be a fresh or empty directory")
    cfg = OmegaConf.load(args.training_config.expanduser())
    if not OmegaConf.is_dict(OmegaConf.select(cfg, "agent")):
        parser.error("--training-config must contain the saved Hydra agent configuration")

    section = OmegaConf.create({
        "split": args.split, "metric_cache_path": str(metric_cache), "overrides": args.overrides,
    })
    evaluator = NavsimEvaluator(cfg, section, output)
    try:
        evaluator.submit(checkpoint, "standalone")
        (_, metrics), = evaluator.collect(block=True)
    except BaseException:
        evaluator.cancel()
        raise
    payload = {"split": args.split, "checkpoint": str(checkpoint), "metrics": metrics}
    result = json.dumps(payload, indent=2, allow_nan=False)
    (output / "standalone_result.json").write_text(result + "\n", encoding="utf-8")
    print(result)
    return int("error" in metrics)


def _interrupt(signum, _frame):
    raise InterruptedError(f"NAVSIM evaluation interrupted by signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())
