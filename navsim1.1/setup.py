"""Install this NAVSIM version together with the shared policy package."""
import os
import runpy
from pathlib import Path

os.environ["NAVSIM_TREE"] = "1.1"
runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"), run_name="__main__")
