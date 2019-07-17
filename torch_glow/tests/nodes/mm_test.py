import torch
import torch_glow

from tests.utils import jitVsGlow

# Test of the PyTorch mm on Glow.
def test_mm():

  def mm(a, b):
        return a.mm(b+b)

  x = torch.randn(7, 4)
  y = torch.randn(4, 9)

  jitVsGlow(mm, x, y)