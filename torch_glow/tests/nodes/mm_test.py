import torch

from tests.utils import jitVsGlow

 # Test of the PyTorch mm on Glow.
def test_mm():

  def test_f(a, b):
    return a.mm(b+b)

  x = torch.randn(7, 4)
  y = torch.randn(4, 1)

  jitVsGlow(test_f, x, y)