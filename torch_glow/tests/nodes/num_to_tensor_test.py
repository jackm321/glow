from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleNumToTensorModule(torch.nn.Module):
    def __init__(self, make_float=False):
        super(SimpleNumToTensorModule, self).__init__()
        self.forward = self._float_forward if make_float else self._forward

    def _float_forward(self, dummy):
        at0 = torch.ops.prim.NumToTensor(dummy.size(0)).to(torch.float)
        # Const floating number is torch.float64 by-default
        # Therefore we need to convert it to float32 once NumToTensor is
        # used
        at1 = torch.ops.prim.NumToTensor(1.2).to(torch.float)
        return torch.cat((at0.reshape(1), at1.reshape(1)))

    def _forward(self, dummy):
        at0 = torch.ops.prim.NumToTensor(dummy.size(0))
        at1 = torch.ops.prim.NumToTensor(dummy.size(1))
        return torch.cat((at0.reshape(1), at1.reshape(1)))


class TestNumToTensor(utils.TorchGlowTestCase):
    def test_NumToTensor_basic(self):
        """Basic test of the PyTorch NumToTensor Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNumToTensorModule(),
            torch.randn(5, 6, 7),
            fusible_ops={"prim::NumToTensor"},
            scripted=True,
            skip_to_glow=True,
        )

    def test_NumToTensor_float(self):
        """Basic test of the PyTorch NumToTensor Node on Glow."""

        utils.compare_tracing_methods(
            SimpleNumToTensorModule(True),
            torch.randn(5, 6, 7),
            fusible_ops={"prim::NumToTensor"},
            scripted=True,
            skip_to_glow=True,
        )
