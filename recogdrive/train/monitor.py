"""Training metrics on the main rank: `metrics.csv` (one row per step, epoch, key, value), optional Weights & Biases
(`monitor.wandb: online | offline`; training runs without the package), and `loss_curve.png` after every epoch."""

import csv
import logging
import time
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

CURVES = ("train/loss", "val/loss")


class Monitor:
    def __init__(self, cfg, output_dir: Path, is_main: bool):
        self.csv_path = output_dir / "metrics.csv"
        self.enabled = is_main
        self.run = None
        mode = str(OmegaConf.select(cfg, "monitor.wandb") or "disabled")
        if mode not in ("disabled", "online", "offline"):
            raise ValueError(f"monitor.wandb must be disabled, online or offline, got {mode!r}")
        if not is_main or mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            logger.warning("monitor.wandb=%s but wandb is not installed; metrics go to %s only", mode, self.csv_path)
            return
        id_file = output_dir / "wandb_run_id"  # a resumed training continues its run
        run_id = id_file.read_text().strip() if id_file.is_file() else wandb.util.generate_id()
        id_file.write_text(run_id)
        self.run = wandb.init(
            project=str(OmegaConf.select(cfg, "monitor.project") or "recogdrive"),
            name=OmegaConf.select(cfg, "experiment_name"),
            id=run_id, resume="allow", mode=mode, dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def log(self, metrics: dict, step: int, epoch: int) -> None:
        if not self.enabled:
            return
        values = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        new = not self.csv_path.is_file()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["step", "epoch", "key", "value", "time"])
            now = f"{time.time():.3f}"
            writer.writerows([step, epoch, key, value, now] for key, value in values.items())
        if self.run is not None:
            self.run.log({**values, "epoch": epoch}, step=step)

    def plot(self) -> None:
        if not self.enabled or not self.csv_path.is_file():
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        series = {}
        with open(self.csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["key"] in CURVES:  # the last value per step wins: a resumed run logs its steps again
                    series.setdefault(row["key"], {})[int(row["step"])] = float(row["value"])
        if not series:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key in CURVES:
            if key in series:
                steps = sorted(series[key])
                ax.plot(steps, [series[key][s] for s in steps], label=key, marker="o" if key == "val/loss" else None)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.csv_path.with_name("loss_curve.png"), dpi=120)
        plt.close(fig)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
