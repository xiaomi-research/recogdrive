# -*- coding: utf-8 -*-
"""Hydra entry for the recogdrive trainer."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hydra
from omegaconf import DictConfig

from recogdrive.train import Trainer

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
