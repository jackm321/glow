from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests.utils import jitVsGlow


def test_type_as():
    """Test of the PyTorch aten::type_as Node."""

    def test_f(x, y):
        return (x + x).type_as(y)

    x = torch.zeros([4], dtype=torch.int32)
    y = torch.randn(4)

    jitVsGlow(test_f, x, y, expected_fused_ops={"aten::type_as"})