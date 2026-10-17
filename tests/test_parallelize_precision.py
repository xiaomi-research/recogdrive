import torch
from torch import nn

from recogdrive.distributed.parallelize import cast_managed_parameters, replicated_params


def test_fsdp_master_cast_includes_frozen_managed_parameters():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.register_parameter(
                "base", nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
            )
            self.backbone.register_parameter("lora", nn.Parameter(torch.ones(2, dtype=torch.bfloat16)))
            self.head = nn.Module()
            self.head.register_parameter("frozen", nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False))
            self.head.register_parameter("trainable", nn.Parameter(torch.ones(2, dtype=torch.float32)))

    model = Model()
    replicated = replicated_params(model, ["backbone"])

    cast_managed_parameters(model, torch.float32, replicated)

    assert model.backbone.base.dtype == torch.bfloat16
    assert model.backbone.lora.dtype == torch.float32
    assert model.head.frozen.dtype == torch.float32
    assert model.head.trainable.dtype == torch.float32
