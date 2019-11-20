from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_view():
    """Test of the PyTorch aten::view Node."""

    def test_f(x):
        return (x + x).view([2, 3])

    x = torch.randn(6)

    jitVsGlow(test_f, x, expected_fused_ops={"aten::view"})
