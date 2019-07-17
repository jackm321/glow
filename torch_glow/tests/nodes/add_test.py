import torch
import torch_glow

from tests.utils import jitVsGlow

# Basic test of the PyTorch add Node on Glow.
def test_add_basic():

  def add_basic(a, b):
        c = a.add(b)
        return c.add(c)

  x = torch.randn(4)
  y = torch.randn(4)

  jitVsGlow(add_basic, x, y)

# Test of the PyTorch add_ Node on Glow.
def test_add_inplace():

  def add_inplace():
        a = torch.Tensor([1,2,3,4])
        a.add_(a)
        a.add_(a)
        return a

#   x = torch.randn(4)

#   print("x", x)

  jitVsGlow(add_inplace)