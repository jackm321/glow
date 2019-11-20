from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_unsqueeze_begin():
    """Test of the PyTorch aten::unsqueeze Node."""

    def test_f(x):
        return (x + x).unsqueeze(0)

    x = torch.randn(4, 2, 3)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})


def test_unsqueeze_middle():
    """Test of the PyTorch aten::unsqueeze Node."""

    def test_f(x):
        return (x + x).unsqueeze(2)

    x = torch.randn(4, 2, 3)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})


def test_unsqueeze_end():
    """Test of the PyTorch aten::unsqueeze Node."""

    def test_f(x):
        return (x + x).unsqueeze(3)

    x = torch.randn(4, 2, 3)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})


def test_unsqueeze_negative_index():
    """Test of the PyTorch aten::unsqueeze Node."""

    def test_f(x):
        return (x + x).unsqueeze(-1)

    x = torch.randn(4, 2, 3)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::unsqueeze"})
