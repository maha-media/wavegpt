"""Tests for HarmonicLinear — the power-law parameterized layer."""
import torch
import pytest
import math
from wavegpt.harmonic_linear import HarmonicLinear


def test_harmonic_linear_output_shape():
    """HarmonicLinear produces correct output shape."""
    layer = HarmonicLinear(in_dim=64, out_dim=128, rank=16)
    x = torch.randn(2, 10, 64)
    y = layer(x)
    assert y.shape == (2, 10, 128)


def test_spectrum_follows_power_law():
    """The constructed weight matrix has power-law singular values."""
    layer = HarmonicLinear(in_dim=64, out_dim=128, rank=32)
    W = layer.get_weight()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    S = S.detach()

    # First rank singular values should be non-trivial
    assert S[0] > 0
    # Should decay — later values smaller than earlier
    assert S[0] > S[15]
    assert S[15] > S[31]


def test_alpha_controls_decay_rate():
    """Higher alpha = faster decay = more compressed spectrum."""
    layer_slow = HarmonicLinear(in_dim=64, out_dim=64, rank=16)
    layer_fast = HarmonicLinear(in_dim=64, out_dim=64, rank=16)

    with torch.no_grad():
        layer_slow.alpha.fill_(0.3)
        layer_fast.alpha.fill_(1.5)

    W_slow = layer_slow.get_weight()
    W_fast = layer_fast.get_weight()

    _, S_slow, _ = torch.linalg.svd(W_slow)
    _, S_fast, _ = torch.linalg.svd(W_fast)

    # Fast decay should have more energy concentrated in top modes
    energy_slow = S_slow[0]**2 / (S_slow**2).sum()
    energy_fast = S_fast[0]**2 / (S_fast**2).sum()
    assert energy_fast > energy_slow


def test_parameter_count_reduction():
    """HarmonicLinear uses far fewer parameters than nn.Linear."""
    in_dim, out_dim, rank = 384, 1536, 30
    full_params = in_dim * out_dim  # 589,824
    harmonic = HarmonicLinear(in_dim, out_dim, rank)
    harmonic_params = sum(p.numel() for p in harmonic.parameters())
    
    # Should be roughly (in_dim + out_dim) * rank + 2
    expected = in_dim * rank + out_dim * rank + 2
    assert harmonic_params == expected
    assert harmonic_params < full_params / 5  # At least 5x reduction


def test_gradient_flows():
    """Gradients flow through sigma1, alpha, U, V."""
    layer = HarmonicLinear(in_dim=32, out_dim=64, rank=8)
    x = torch.randn(2, 5, 32)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert layer.sigma1.grad is not None
    assert layer.alpha.grad is not None
    assert layer.U.grad is not None
    assert layer.V.grad is not None
    # All grads should be non-zero
    assert layer.sigma1.grad.abs() > 0
    assert layer.alpha.grad.abs() > 0


def test_from_pretrained_reconstruction():
    """Can reconstruct a weight matrix from its SVD with power-law fit."""
    # Create a known weight matrix with power-law structure
    d = 64
    k = 16
    U = torch.randn(d, k)
    U, _ = torch.linalg.qr(U)  # orthogonalize
    V = torch.randn(d, k)
    V, _ = torch.linalg.qr(V)
    sigma = torch.tensor([10.0 * i**(-0.7) for i in range(1, k+1)])
    W_orig = U @ torch.diag(sigma) @ V.T

    # Reconstruct
    layer = HarmonicLinear.from_weight(W_orig, rank=k)
    W_recon = layer.get_weight()

    # Should be close (not exact due to power-law fitting)
    error = (W_orig - W_recon).norm() / W_orig.norm()
    assert error < 0.5  # within 50% relative error (power law is approximate)
