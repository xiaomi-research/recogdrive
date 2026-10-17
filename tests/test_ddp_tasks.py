from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers.feature_extraction_utils import BatchFeature

from recogdrive.distributed.parallelize import expose_output


class AlternatingHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2)
        self.planning = nn.Linear(2, 1)
        self.language = nn.Linear(2, 1)

    def forward(self, task, inputs):
        hidden = self.shared(inputs)
        loss = (self.planning(hidden) if task == "planning" else self.language(hidden)).square().mean()
        return BatchFeature({"loss": loss}) if task == "planning" else SimpleNamespace(loss=loss)


def ddp_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(123)
        model = AlternatingHeads()
        model.register_forward_hook(expose_output)
        wrapped = DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for task in ("planning", "vqa", "planning", "vqa"):
            optimizer.zero_grad(set_to_none=True)
            output = wrapped(task, torch.ones(2, 2) * (rank + 1))
            assert isinstance(output, dict)
            output["loss"].backward()
            assert model.shared.weight.grad is not None
            inactive = model.language if task == "planning" else model.planning
            assert inactive.weight.grad is None
            optimizer.step()
        local = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.zeros_like(local) for _ in range(2)]
        dist.all_gather(gathered, local)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class DistributedTaskTests(unittest.TestCase):
    def test_two_rank_alternating_heads_have_reduced_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(str(Path(directory) / "ddp-init"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
