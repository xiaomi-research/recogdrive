import time

import torch


def to_device(obj, device):
    if torch.is_tensor(obj):
        # Every tensor, integer ones included: FSDP2/DDP move any host tensor left in the inputs with a
        # blocking copy, which drains the GPU once per step. Host-side metadata travels as Python lists.
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def record_stream(obj, stream):
    if torch.is_tensor(obj):
        if obj.is_cuda:
            obj.record_stream(stream)
    elif isinstance(obj, dict):
        for v in obj.values():
            record_stream(v, stream)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            record_stream(v, stream)


class DevicePrefetcher:
    """Copies batch k+1 to the GPU on a side stream while batch k computes.

    `wait_s` accumulates the time the training loop spent blocked on the DataLoader.
    """

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.wait_s = 0.0

    def __len__(self):
        return len(self.loader)

    def next_batch(self, iterator, stream):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return None
        self.wait_s += time.perf_counter() - start
        if stream is None:
            return batch
        with torch.cuda.stream(stream):
            return to_device(batch, self.device)

    def __iter__(self):
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        pending = self.next_batch(iterator, stream)
        while pending is not None:
            if stream is not None:
                current = torch.cuda.current_stream(self.device)
                current.wait_stream(stream)
                record_stream(pending, current)
            batch, pending = pending, None
            yield batch
            pending = self.next_batch(iterator, stream)
