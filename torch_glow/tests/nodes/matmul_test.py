import torch
import torch_glow
import pytest

from tests.utils import jitVsGlow

# Test of the PyTorch matmul Node on Glow with 1d x 1d inputs.
# NOTE: this should be equivalent to the dotproduct.
def test_matmul_1d_1d():

  def matmul_1d_1d(a, b):
        return a.matmul(b+b)

  x = torch.randn(4)
  y = torch.randn(4)

  jitVsGlow(matmul_1d_1d, x, y)

# Test of the PyTorch matmul Node on Glow with 2d x 2d inputs.
def test_matmul_2d_2d():

  def matmul_2d_2d(a, b):
        return a.matmul(b+b)

  x = torch.randn(7, 4)
  y = torch.randn(4, 9)

  jitVsGlow(matmul_2d_2d, x, y)

# Test of the PyTorch matmul Node on Glow with 1d x 2d inputs.
def test_matmul_1d_2d():

  def matmul_1d_2d(a, b):
        return a.matmul(b+b)

  x = torch.randn(4)
  y = torch.randn(4, 9)

  jitVsGlow(matmul_1d_2d, x, y)

# Test of the PyTorch matmul Node on Glow with 2d x 1d inputs.
def test_matmul_2d_1d():

  def matmul_2d_1d(a, b):
        return a.matmul(b+b)

  x = torch.randn(9, 4)
  y = torch.randn(4)

  jitVsGlow(matmul_2d_1d, x, y)

# Test of the PyTorch matmul Node on Glow with 3d x 3d inputs.
def test_matmul_3d_3d():

  def matmul_3d_3d(a, b):
        return a.matmul(b+b)

  x = torch.randn(2, 3, 4)
  y = torch.randn(2, 4, 5)

  jitVsGlow(matmul_3d_3d, x, y)

# Test of the PyTorch matmul Node on Glow with 3d x 3d inputs with second input
# batch size 1.
def test_matmul_3d_3d_leading_1():

  def matmul_3d_3d(a, b):
        return a.matmul(b+b)

  x = torch.randn(2, 3, 4)
  y = torch.randn(1, 4, 5)

  jitVsGlow(matmul_3d_3d, x, y)

# Test of the PyTorch matmul Node on Glow with 3d x 2d inputs.
def test_matmul_3d_2d():

  def matmul_3d_2d(a, b):
        return a.matmul(b+b)

  x = torch.randn(2, 3, 4)
  y = torch.randn(4, 5)

  jitVsGlow(matmul_3d_2d, x, y)
