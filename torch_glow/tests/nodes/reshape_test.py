from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow


def test_reshape():
    """Test of the PyTorch reshape Node on Glow."""

    def test_f(a):
        b = a + a
        return b.reshape([2, -1])

    x = torch.rand(2, 3, 4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::reshape"})


def test_reshape_no_neg():
    """Test of the PyTorch reshape Node on Glow."""

    def test_f(a):
        b = a + a
        return b.reshape([24])

    x = torch.rand(2, 3, 4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::reshape"})


def test_reshape_only_neg():
    """Test of the PyTorch reshape Node on Glow."""

    def test_f(a):
        b = a + a
        return b.reshape([-1])

    x = torch.rand(2, 3, 4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::reshape"})


def test_reshape_one_neg():
    """Test of the PyTorch reshape Node on Glow."""

    def test_f(a):
        b = a + a
        return b.reshape([2, -1, 4])

    x = torch.rand(2, 3, 3, 4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::reshape"})
