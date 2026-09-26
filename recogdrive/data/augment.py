"""Training-split augmentation, configured by `augment`; everything is off by default.

Image augmentations are photometric only: a geometric one would move the scene against the trajectory labels.
"""

import random
from typing import Optional

import torch
from omegaconf import OmegaConf

COMMAND_DIM = 4  # status_feature: command one-hot, then velocity and acceleration


class EgoDropout:
    """Zeroes the ego history with probability `history` and the ego velocity / acceleration with probability
    `status` (the command stays), so the planner cannot lean on ego-state shortcuts. Runs before the prompt is built."""

    def __init__(self, history: float, status: float):
        self.history, self.status = history, status

    def __call__(self, features: dict) -> dict:
        if random.random() < self.history:
            features["history_trajectory"] = torch.zeros_like(features["history_trajectory"])
        if random.random() < self.status:
            status = features["status_feature"].clone()
            status[COMMAND_DIM:] = 0
            features["status_feature"] = status
        return features


def option(section, key: str) -> float:
    return float(OmegaConf.select(section, key) or 0.0) if section is not None else 0.0


def ego_dropout(section) -> Optional[EgoDropout]:
    history, status = option(section, "history_dropout"), option(section, "ego_status_dropout")
    return EgoDropout(history, status) if history or status else None


def image_augment(section):
    """PIL -> PIL transform, or None when every image augmentation is off."""
    import torchvision.transforms as T

    steps = []
    if option(section, "color_jitter.p"):
        jitter = T.ColorJitter(*(option(section, f"color_jitter.{key}") for key in ("brightness", "contrast", "saturation", "hue")))
        steps.append(T.RandomApply([jitter], p=option(section, "color_jitter.p")))
    if option(section, "grayscale_p"):
        steps.append(T.RandomGrayscale(option(section, "grayscale_p")))
    if option(section, "blur_p"):
        steps.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=option(section, "blur_p")))
    return T.Compose(steps) if steps else None
