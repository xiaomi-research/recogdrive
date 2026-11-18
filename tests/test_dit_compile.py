"""Importing the planner must not start compilation when train.compile is false."""
import os
import subprocess
import sys
from unittest.mock import patch

import torch

from recogdrive.distributed.parallelize import compile_blocks
from recogdrive.models.recogdrive.dit import LightningDiTBlock


def test_planner_import_does_not_compile_or_start_worker_pool():
    script = '''
import sys
from unittest.mock import patch
import torch
with patch("torch.compile", side_effect=AssertionError("implicit compilation")):
    from recogdrive.models.recogdrive.dit import FinalLayer, LightningDiTBlock
assert "torch._inductor.async_compile" not in sys.modules
'''
    env = os.environ.copy()
    # Detect an accidental worker pool even if the surrounding regression disables Dynamo.
    env.pop("TORCH_COMPILE_DISABLE", None)
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=60)


def test_explicit_block_compile_keeps_modulation_output_and_gradient():
    block = LightningDiTBlock(dim=8, num_heads=2, head_dim=4)
    x = torch.randn(2, 3, 8, requires_grad=True)
    conditioning = torch.randn(2, 8, requires_grad=True)
    expected = block(x, conditioning)
    gradients = torch.autograd.grad(expected.sum(), (x, conditioning))
    original_compile = torch.compile
    with patch("torch.compile", side_effect=lambda fn: original_compile(fn, backend="eager")) as compile_fn:
        compile_blocks([[block]])
    compile_fn.assert_called_once()
    actual = block(x, conditioning)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(torch.autograd.grad(actual.sum(), (x, conditioning)), gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
