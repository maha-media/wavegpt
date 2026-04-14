import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.spectral_linear import SpectralLinear


def test_residual_dtype_fp32():
    torch.manual_seed(0)
    W = torch.randn(64, 128, dtype=torch.bfloat16)
    linear = nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(W)
    spec = SpectralLinear.from_linear(
        linear, rank=32, keep_residual=True, residual_dtype=torch.float32,
    )
    assert spec.residual is not None
    assert spec.residual.dtype == torch.float32, f"got {spec.residual.dtype}"
    assert spec.U.dtype == torch.bfloat16
    assert spec.V.dtype == torch.bfloat16


def test_residual_dtype_default_matches_orig():
    linear = nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)
    spec = SpectralLinear.from_linear(linear, rank=32, keep_residual=True)
    assert spec.residual.dtype == torch.bfloat16
