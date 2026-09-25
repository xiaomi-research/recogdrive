"""python -m recogdrive.eval.vqa {drivelm,lingoqa,drivebench} --model <InternVL dir> --data <file> --image-root <dir> --out <dir>

Answers every question with one lmdeploy worker per visible GPU, then writes the benchmark's prediction
files and score.json into --out.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

from recogdrive.eval.vqa.engine import run_all, run_shard, visible_gpus

BENCHES = ("drivelm", "lingoqa", "drivebench")


def parse(argv):
    parser = argparse.ArgumentParser(prog="python -m recogdrive.eval.vqa")
    parser.add_argument("bench", choices=BENCHES)
    parser.add_argument("--model", required=True, help="InternVL checkpoint directory (HF format)")
    parser.add_argument("--data", required=True, help="DriveLM / DriveBench question json, LingoQA val.parquet")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpus", default=None, help="comma-separated GPU ids; default: all visible")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backend", choices=("turbomind", "pytorch"), default="turbomind",
                        help="lmdeploy engine; turbomind matches the reported numbers")
    parser.add_argument("--judge", default="wayveai/Lingo-Judge", help="LingoQA: Lingo-Judge classifier")
    parser.add_argument("--corruption", default="", help="DriveBench: image corruption, e.g. Fog or NoImage")
    parser.add_argument("--gpt", action="store_true", help="DriveBench: GPT score, reads OPENAI_API_KEY")
    parser.add_argument("--shard", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse(argv)
    bench = importlib.import_module(f"recogdrive.eval.vqa.{args.bench}")
    args.out.mkdir(parents=True, exist_ok=True)
    samples = bench.load(args)
    if args.shard:
        index, count = map(int, args.shard.split("/"))
        run_shard(
            args.out, index, samples[index::count],
            model=args.model, system_prompt=bench.SYSTEM_PROMPT, max_new_tokens=bench.MAX_NEW_TOKENS,
            min_new_tokens=bench.MIN_NEW_TOKENS, batch_size=args.batch_size or bench.BATCH_SIZE,
            session_len=bench.SESSION_LEN, backend=args.backend,
        )
        return
    gpus = args.gpus.split(",") if args.gpus else visible_gpus()
    answers, codes = run_all(argv, gpus, args.out)
    missing = [s["id"] for s in samples if s["id"] not in answers]
    if missing:
        raise RuntimeError(f"{len(missing)} of {len(samples)} questions unanswered (shard exit codes {codes}); "
                           "run the same command again to resume")
    if any(codes):
        print(f"warning: shard exit codes {codes} after every question was answered", flush=True)
    scores = bench.finish(args, samples, answers)
    (args.out / "score.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
