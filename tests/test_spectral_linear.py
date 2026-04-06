"""Tests for SpectralLinear — post-training spectral decomposition."""
import torch
import torch.nn as nn
import pytest


def test_from_linear_output_match():
    """Decompose nn.Linear, output should match within tolerance."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 128, bias=False)
    # rank=None → auto-select for 95% energy
    spec = SpectralLinear.from_linear(linear, rank=None)
    x = torch.randn(2, 10, 64)
    y_orig = linear(x)
    y_spec = spec(x)
    assert y_spec.shape == y_orig.shape
    rel_err = (y_orig - y_spec).norm() / y_orig.norm()
    assert rel_err < 0.25, f"Relative error {rel_err:.4f} too high"


def test_full_rank_exact_match():
    """Full rank decomposition should be lossless."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(32, 48, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=32)  # full rank = min(in,out)
    x = torch.randn(2, 5, 32)
    y_orig = linear(x)
    y_spec = spec(x)
    torch.testing.assert_close(y_orig, y_spec, atol=1e-5, rtol=1e-4)


def test_sigma1_mode_learnable():
    """In sigma1 mode, only one param per layer is learnable."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='sigma1')
    learnable = [p for p in spec.parameters() if p.requires_grad]
    assert len(learnable) == 1
    assert learnable[0].numel() == 1


def test_per_mode_learnable():
    """In per_mode, one amplitude per singular value is learnable."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='per_mode')
    learnable = [p for p in spec.parameters() if p.requires_grad]
    assert len(learnable) == 1
    assert learnable[0].numel() == 16


def test_spectrum_report():
    """Report fitted alpha, sigma1, energy captured."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 128, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=32)
    report = spec.spectral_report()
    assert 'alpha' in report
    assert 'sigma1' in report
    assert 'energy_captured' in report
    assert 0 < report['energy_captured'] <= 1.0


def test_sigma1_gradient_frozen_geometry():
    """In sigma1 mode, gradients flow only to sigma1, not U or V."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='sigma1')
    x = torch.randn(2, 5, 64)
    y = spec(x)
    loss = y.sum()
    loss.backward()
    assert spec.sigma1.grad is not None
    assert not spec.U.requires_grad
    assert not spec.V.requires_grad


def test_per_mode_gradient_frozen_geometry():
    """In per_mode, gradients flow to spectrum vector, not U or V."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='per_mode')
    x = torch.randn(2, 5, 64)
    y = spec(x)
    loss = y.sum()
    loss.backward()
    assert spec.spectrum.grad is not None
    assert spec.spectrum.grad.shape == (16,)
