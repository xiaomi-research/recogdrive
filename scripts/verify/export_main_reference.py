"""Export pinned main source for CPU regression in a server copy without Git metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


FILES = (
    "recogdrive_diffusion_planner.py", "recogdrive_dit.py", "blocks/attention.py",
    "blocks/encoder.py", "blocks/rmsnorm.py", "blocks/rope.py", "utils/lr_scheduler.py",
)


def export_reference(destination, ref="main"):
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], cwd=repo, text=True
    ).strip()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        source = "navsim/agents/recogdrive/" + name
        data = subprocess.check_output(["git", "show", commit + ":" + source], cwd=repo)
        target = (destination / name).resolve()
        target.relative_to(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"refusing to replace a different reference file: {target}")
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (destination / "REFERENCE.json").write_text(
        json.dumps({"commit": commit, "files": hashes}, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(hashes)} original files from {commit} to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--ref", default="main")
    args = parser.parse_args()
    export_reference(args.destination, args.ref)
