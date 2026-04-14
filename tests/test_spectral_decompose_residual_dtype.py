import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.spectral_surgery import spectral_decompose
from wavegpt.spectral_linear import SpectralLinear


def test_decompose_fp32_residual():
    model = nn.Sequential(
        nn.Linear(128, 64, dtype=torch.bfloat16),
        nn.Linear(64, 32, dtype=torch.bfloat16),
    )
    spectral_decompose(
        model, rank=16, keep_residual=True, residual_dtype=torch.float32,
    )
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            assert m.residual is not None
            assert m.residual.dtype == torch.float32
