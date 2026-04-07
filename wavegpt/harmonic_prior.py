"""
Harmonic priors for spectral fine-tuning.

The key insight: trained weights converge to W = σ₁ · Σ k^{-1/φ} · u_k · v_k^T.
We use this as a prior for:
  1. Rank allocation — layers deviating from 1/φ need more spectral freedom
  2. Regularization — spectral weight decay toward the power-law equilibrium

This is what we add on top of vanilla SVFit/SVFT (NeurIPS 2024).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .spectral_linear import SpectralLinear

PHI = (1 + 5**0.5) / 2
INV_PHI = 1 / PHI  # 0.6180339887...


def harmonic_regularization(
    module_or_model: nn.Module,
    lambda_h: float = 1.0,
) -> torch.Tensor:
    """
    Spectral weight decay toward k^{-1/φ} prior.

    For each per_mode SpectralLinear, penalizes deviation of the
    learned spectrum from the power-law shape anchored at σ₁.

    L_harmonic = λ · mean_layers[ mean_k[ (s_k - σ₁ · k^{-1/φ})² ] ]

    Args:
        module_or_model: a SpectralLinear or any nn.Module containing them
        lambda_h: regularization strength (applied as multiplier)

    Returns:
        Scalar loss tensor (differentiable through spectrum params).
    """
    modules = (
        [module_or_model]
        if isinstance(module_or_model, SpectralLinear)
        else [m for m in module_or_model.modules() if isinstance(m, SpectralLinear)]
    )

    loss = torch.tensor(0.0)
    count = 0

    for m in modules:
        if m.mode != 'per_mode':
            continue
        s = m.spectrum
        device = s.device
        k = torch.arange(1, len(s) + 1, device=device, dtype=s.dtype)
        # Prior: power law anchored at current σ₁ (detached so prior
        # doesn't push σ₁ itself — only the shape matters)
        prior = s[0].detach() * k.pow(-INV_PHI)
        loss = loss.to(device) + ((s - prior) ** 2).mean()
        count += 1

    return lambda_h * loss


def compute_adaptive_rank(
    alpha: float,
    base_rank: int,
    beta: float = 2.0,
    max_rank: int | None = None,
) -> int:
    """
    Allocate rank proportional to deviation from 1/φ.

    Layers close to the golden ratio prior are well-described by the
    power law — fewer free modes needed. Layers that deviate (e.g.
    projection layers at α ≈ 1.0) need more spectral freedom.

    rank_k = base_rank × (1 + β × |α_k - 1/φ|)

    Args:
        alpha: fitted power-law exponent for this layer
        base_rank: rank for a layer exactly at 1/φ
        beta: sensitivity to deviation (default 2.0)
        max_rank: hard cap on rank allocation

    Returns:
        Integer rank for this layer.
    """
    deviation = abs(alpha - INV_PHI)
    rank = int(base_rank * (1.0 + beta * deviation))
    if max_rank is not None:
        rank = min(rank, max_rank)
    return rank


def fit_alpha(weight: torch.Tensor) -> float:
    """
    Quick power-law fit on a weight matrix (no decomposition stored).

    Returns the fitted α from log(σ_k) = log(σ₁) - α·log(k).
    """
    import numpy as np

    W = weight.data.float().cpu()
    S = torch.linalg.svdvals(W)
    s_np = S.numpy()
    n_fit = max(int(len(s_np) * 0.9), 4)
    log_k = np.log(np.arange(1, n_fit + 1))
    log_s = np.log(s_np[:n_fit] + 1e-10)
    coeffs = np.polyfit(log_k, log_s, 1)
    return float(-coeffs[0])
