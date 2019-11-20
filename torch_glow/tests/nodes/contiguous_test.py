from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_contiguous():
    """Test of the PyTorch aten::contiguous Node."""

    def test_f(x):
        return (x + x).contiguous()

    x = torch.randn(4)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::contiguous"})