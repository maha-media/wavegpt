"""Tests for SpectralLinear — post-training spectral decomposition."""
import math
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
    assert spec.log_spectrum.grad is not None
    assert spec.log_spectrum.grad.shape == (16,)


def test_to_linear_roundtrip():
    """Decompose → modify spectrum → merge back → output matches new spectrum."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(48, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=24, mode='per_mode')
    # Modify the spectrum (simulate fine-tuning)
    with torch.no_grad():
        spec.log_spectrum += math.log(1.1)  # multiply spectrum by 1.1 in log-space
    x = torch.randn(2, 5, 48)
    y_spec = spec(x).detach()
    # Merge back
    merged = spec.to_linear()
    y_merged = merged(x).detach()
    torch.testing.assert_close(y_spec, y_merged, atol=1e-5, rtol=1e-4)


def test_save_load_spectral_params():
    """Save only spectral params, load into fresh decomposition."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 64, bias=False)
    spec1 = SpectralLinear.from_linear(linear, rank=16, mode='per_mode')
    # Simulate fine-tuning
    with torch.no_grad():
        spec1.log_spectrum += math.log(0.9)  # multiply spectrum by 0.9 in log-space
    # Save just the learnable params
    spectral_state = {k: v for k, v in spec1.state_dict().items()
                      if 'log_spectrum' in k or 'sigma1' in k}
    # Load into fresh decomposition of same layer
    spec2 = SpectralLinear.from_linear(linear, rank=16, mode='per_mode')
    spec2.load_state_dict(spec2.state_dict() | spectral_state)
    x = torch.randn(2, 5, 64)
    torch.testing.assert_close(spec1(x), spec2(x), atol=1e-6, rtol=1e-5)


def test_residual_preservation_exact():
    """With residual, decompose + reconstruct is lossless at any rank."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 128, bias=False)
    x = torch.randn(2, 10, 64)
    y_orig = linear(x)
    spec = SpectralLinear.from_linear(linear, rank=8, mode='per_mode', keep_residual=True)
    y_spec = spec(x)
    torch.testing.assert_close(y_orig, y_spec, atol=1e-5, rtol=1e-4)


def test_residual_stored_as_buffer():
    """Residual should be a frozen buffer, not a parameter."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 128, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=8, mode='per_mode', keep_residual=True)
    assert hasattr(spec, 'residual')
    assert not spec.residual.requires_grad
    learnable = [p for p in spec.parameters() if p.requires_grad]
    assert len(learnable) == 1
    assert learnable[0].numel() == 8


def test_residual_with_bias():
    """Residual preservation works with biased layers too."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 128, bias=True)
    x = torch.randn(2, 10, 64)
    y_orig = linear(x)
    spec = SpectralLinear.from_linear(linear, rank=8, mode='per_mode', keep_residual=True)
    y_spec = spec(x)
    torch.testing.assert_close(y_orig, y_spec, atol=1e-5, rtol=1e-4)


def test_no_residual_by_default():
    """Default from_linear should NOT store residual (backward compat)."""
    from wavegpt.spectral_linear import SpectralLinear
    linear = nn.Linear(64, 128, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=8, mode='per_mode')
    assert spec.residual is None
