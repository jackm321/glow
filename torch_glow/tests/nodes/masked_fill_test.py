from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_masked_fill():
    """Test of the PyTorch aten::masked_fill Node."""

    def test_f(x, mask):
        return (x + x).masked_fill(mask, -1.0)

    x = torch.randn(6)
    mask = torch.BoolTensor([True, False, False, True, False, False])

    jitVsGlow(test_f, x, mask, expected_fused_ops={"aten::masked_fill"})


def test_masked_fill_2d():
    """Test of the PyTorch aten::masked_fill Node."""

    def test_f(x, mask):
        return (x + x).masked_fill(mask, 42.0)

    x = torch.randn(3, 2)
    mask = torch.BoolTensor([[True, False], [False, True], [False, False]])

    jitVsGlow(test_f, x, mask, expected_fused_ops={"aten::masked_fill"})
