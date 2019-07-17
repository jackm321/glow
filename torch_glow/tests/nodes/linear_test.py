import torch
import torch_glow
import torch.nn.functional as F

from tests.utils import jitVsGlow

# Basic test of the PyTorch add Node on Glow.
def test_linear_basic():

  def f(inputs, weight):
        return F.linear(inputs, weight)

  inputs = torch.randn(3, 2, 4)
  filters = torch.randn(8, 4)
#   bias = torch.randn(8)

  jitVsGlow(f, inputs, filters)

# Basic test of the PyTorch add Node on Glow.
# def test_linear_with_bias():

#   def f(inputs, weight, bias):
#         return F.linear(inputs, weight, bias)

#   inputs = torch.randn(3, 2, 4)
#   filters = torch.randn(8, 4)
#   bias = torch.randn(8)

#   jitVsGlow(f, inputs, filters, bias)


# tests:
# without bias
# with bias
# input greater than 2d?


# without bias: aten::matmul
# with bias  aten::mm then aten::add
# with bias and input larger than 2d: transpose (aten::t) then matmul then add_