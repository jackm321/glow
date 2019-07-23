import torch

from tests.utils import jitVsGlow

 # Test of the PyTorch mv on Glow.
def test_mv():

  def test_f(a, b):
    return a.mv(b+b)

  x = torch.randn(7, 4)
  y = torch.randn(4)

  jitVsGlow(test_f, x, y)