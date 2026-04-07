"""Tests for harmonic priors — regularization + adaptive rank."""
import torch
import torch.nn as nn


def test_harmonic_reg_zero_at_prior():
    """If spectrum exactly matches k^{-1/φ}, regularization loss is 0."""
    from wavegpt.harmonic_prior import harmonic_regularization, INV_PHI
    from wavegpt.spectral_linear import SpectralLinear
    k = torch.arange(1, 17, dtype=torch.float)
    s = 5.0 * k.pow(-INV_PHI)
    U = torch.randn(32, 16)
    V = torch.randn(32, 16)
    spec = SpectralLinear(U, s.clone(), V, mode='per_mode')
    loss = harmonic_regularization(spec)
    assert loss.item() < 1e-6


def test_harmonic_reg_nonzero_when_deviated():
    """Flat spectrum (far from power law) should have positive loss."""
    from wavegpt.harmonic_prior import harmonic_regularization
    from wavegpt.spectral_linear import SpectralLinear
    s = torch.ones(16)
    U = torch.randn(32, 16)
    V = torch.randn(32, 16)
    spec = SpectralLinear(U, s.clone(), V, mode='per_mode')
    loss = harmonic_regularization(spec)
    assert loss.item() > 0.1


def test_harmonic_reg_works_on_model():
    """Regularization should work when passed a full model."""
    from wavegpt.harmonic_prior import harmonic_regularization
    from wavegpt.spectral_linear import SpectralLinear
    from wavegpt.spectral_surgery import spectral_decompose

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(32, 64, bias=False)
            self.linear2 = nn.Linear(64, 16, bias=False)
        def forward(self, x):
            return self.linear2(self.linear1(x))

    model = TinyModel()
    spectral_decompose(model, rank=8, mode='per_mode')
    loss = harmonic_regularization(model)
    assert loss.item() >= 0.0
    # Should be differentiable
    loss.backward()


def test_harmonic_reg_skips_sigma1():
    """sigma1 mode layers should be skipped (no spectrum to regularize)."""
    from wavegpt.harmonic_prior import harmonic_regularization
    from wavegpt.spectral_linear import SpectralLinear
    k = torch.arange(1, 17, dtype=torch.float)
    s = 5.0 * k.pow(-0.618)
    U = torch.randn(32, 16)
    V = torch.randn(32, 16)
    spec = SpectralLinear(U, s.clone(), V, mode='sigma1')
    loss = harmonic_regularization(spec)
    assert loss.item() == 0.0


def test_adaptive_rank_increases_with_deviation():
    """Layers further from 1/φ should get higher rank."""
    from wavegpt.harmonic_prior import compute_adaptive_rank
    r_close = compute_adaptive_rank(alpha=0.62, base_rank=192)
    r_far = compute_adaptive_rank(alpha=1.0, base_rank=192)
    assert r_far > r_close


def test_adaptive_rank_at_golden_ratio():
    """Layer exactly at 1/φ should get base_rank."""
    from wavegpt.harmonic_prior import compute_adaptive_rank, INV_PHI
    r = compute_adaptive_rank(alpha=INV_PHI, base_rank=192)
    assert r == 192


def test_adaptive_rank_respects_max():
    """Max rank should cap the allocation."""
    from wavegpt.harmonic_prior import compute_adaptive_rank
    r = compute_adaptive_rank(alpha=2.0, base_rank=192, max_rank=256)
    assert r <= 256
