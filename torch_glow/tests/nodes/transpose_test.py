import torch
import torch_glow

from tests.utils import jitVsGlow

# Test of the PyTorch t (transpose) on Glow.
def test_transpose():

  def transpose(a):
        b = a + a
        return b.t()

  x = torch.randn(7, 4)
  
  jitVsGlow(transpose, x)