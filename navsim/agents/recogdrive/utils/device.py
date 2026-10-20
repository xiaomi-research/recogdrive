"""Device selection for the ReCogDrive inference path (XPU, then CUDA, then CPU).

An explicit device always wins and is never swapped for another backend.
"""

from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in ("xpu", "cuda"):
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None, index: int = 0) -> torch.device:
    """Resolve ``None`` to the detected backend (card ``index``), anything else verbatim.

    A bare ``xpu`` or ``cuda`` becomes card ``index``.
    """
    if device is None or device == "":
        device_type = default_device_type()
        return torch.device("cpu") if device_type == "cpu" else torch.device(device_type, index)
    device = torch.device(device)
    if device.type in ("xpu", "cuda") and device.index is None:
        return torch.device(device.type, index)
    return device


def device_type(device: DeviceLike) -> str:
    """Backend name of a device, a device string or a tensor's device."""
    return torch.device(device).type
