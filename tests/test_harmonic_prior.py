"""Tests for harmonic priors — bent power law, regularization, adaptive rank."""
import torch
import torch.nn as nn
import numpy as np


def test_harmonic_reg_zero_at_bent_prior():
    """If spectrum matches bent power law (k+k0)^{-1/φ}, loss is ~0."""
    from wavegpt.harmonic_prior import harmonic_regularization, INV_PHI
    from wavegpt.spectral_linear import SpectralLinear
    k0 = 100.0
    k = torch.arange(1, 17, dtype=torch.float)
    A = 50.0
    s = A * (k + k0).pow(-INV_PHI)
    U = torch.randn(32, 16)
    V = torch.randn(32, 16)
    spec = SpectralLinear(U, s.clone(), V, mode='per_mode', k0=k0)
    loss = harmonic_regularization(spec)
    assert loss.item() < 1e-4, f"Expected ~0, got {loss.item()}"


def test_harmonic_reg_zero_at_simple_prior():
    """If spectrum matches k^{-1/φ} (no k0), loss is ~0."""
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


def test_harmonic_reg_uses_k0_when_present():
    """With k0, prior should use bent power law, not simple."""
    from wavegpt.harmonic_prior import harmonic_regularization, INV_PHI
    from wavegpt.spectral_linear import SpectralLinear

    k = torch.arange(1, 65, dtype=torch.float)
    k0 = 200.0

    # Spectrum matches bent law
    A = 50.0
    s_bent = A * (k + k0).pow(-INV_PHI)
    U = torch.randn(128, 64)
    V = torch.randn(128, 64)

    # With correct k0 → low loss
    spec_right = SpectralLinear(U, s_bent.clone(), V, mode='per_mode', k0=k0)
    loss_right = harmonic_regularization(spec_right)

    # Without k0 (uses simple prior) → high loss (because s_bent is flat)
    spec_wrong = SpectralLinear(U, s_bent.clone(), V, mode='per_mode')
    loss_wrong = harmonic_regularization(spec_wrong)

    assert loss_right.item() < loss_wrong.item(), \
        f"Bent prior should fit better: {loss_right.item()} vs {loss_wrong.item()}"


def test_adaptive_rank_increases_with_poor_fit():
    """Layers with worse R² should get higher rank."""
    from wavegpt.harmonic_prior import compute_adaptive_rank
    r_good = compute_adaptive_rank(r2=0.95, base_rank=192)
    r_bad = compute_adaptive_rank(r2=0.60, base_rank=192)
    assert r_bad > r_good


def test_adaptive_rank_at_perfect_fit():
    """Layer with perfect R²=1 should get base_rank."""
    from wavegpt.harmonic_prior import compute_adaptive_rank
    r = compute_adaptive_rank(r2=1.0, base_rank=192)
    assert r == 192


def test_adaptive_rank_respects_max():
    """Max rank should cap the allocation."""
    from wavegpt.harmonic_prior import compute_adaptive_rank
    r = compute_adaptive_rank(r2=0.0, base_rank=192, max_rank=256)
    assert r <= 256


def test_fit_bent_power_law_recovers_k0():
    """Fitting a bent spectrum should recover the true k0."""
    from wavegpt.harmonic_prior import fit_bent_power_law, INV_PHI

    # Generate synthetic spectrum with known k0
    true_k0 = 150.0
    A = 100.0
    k = np.arange(1, 513)
    S = A * (k + true_k0) ** (-INV_PHI)
    S_tensor = torch.tensor(S, dtype=torch.float)

    result = fit_bent_power_law(S_tensor)
    assert abs(result['k0'] - true_k0) < 10.0, \
        f"Expected k0≈{true_k0}, got {result['k0']}"
    assert result['r2'] > 0.99, f"R² should be near 1, got {result['r2']}"


def test_fit_bent_power_law_small_k0():
    """For spectra close to simple power law, k0 should be small."""
    from wavegpt.harmonic_prior import fit_bent_power_law, INV_PHI

    # Generate simple power law (k0 ≈ 0)
    k = np.arange(1, 101)
    S = 10.0 * k ** (-INV_PHI)
    S_tensor = torch.tensor(S, dtype=torch.float)

    result = fit_bent_power_law(S_tensor)
    assert result['k0'] < 5.0, f"Expected small k0, got {result['k0']}"
    assert result['r2'] > 0.95, f"R² should be high, got {result['r2']}"


def test_spectral_linear_stores_k0():
    """SpectralLinear should store and expose k0 as buffer."""
    from wavegpt.spectral_linear import SpectralLinear
    U = torch.randn(32, 8)
    S = torch.randn(8).abs()
    V = torch.randn(16, 8)
    spec = SpectralLinear(U, S, V, mode='per_mode', k0=150.0)
    assert spec.k0 is not None
    assert abs(spec.k0.item() - 150.0) < 1e-4
    # k0 should be a buffer (not a parameter)
    param_names = [n for n, _ in spec.named_parameters()]
    assert 'k0' not in param_names


def test_from_linear_fits_k0():
    """from_linear should fit and store k0."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 128, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='per_mode')
    # Should have k0 (may be small for random weights)
    assert spec.k0 is not None
    assert isinstance(spec.k0.item(), float)
