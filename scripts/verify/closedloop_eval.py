"""Public CLI for closed-loop evaluator jobs and checkpoint-only runs."""

import sys
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recogdrive.eval.closedloop import _handle_term, _main


if __name__ == "__main__":
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_term)
    raise SystemExit(_main())
