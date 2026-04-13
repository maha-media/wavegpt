"""
Harmonic priors for spectral fine-tuning.

The key insight: trained weights converge to a bent power law:

    σ_k = A · (k + k₀)^{-1/φ}

where 1/φ = 0.618... is a universal constant and k₀ is a per-layer
spectral offset. For small models (d≈768), k₀ ≈ 0 and the simple
power law k^{-1/φ} suffices. For large models (d≥5120), k₀ ranges
from ~100 to ~1000, creating a flat top that steepens to 1/φ slope.

We use this as a prior for:
  1. Rank allocation — layers with poor bent-fit R² need more freedom
  2. Regularization — spectral weight decay toward the bent power law

This is what we add on top of vanilla SVFit/SVFT (NeurIPS 2024).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from .spectral_linear import SpectralLinear

PHI = (1 + 5**0.5) / 2
INV_PHI = 1 / PHI  # 0.6180339887...

# F/L spectral exponents by layer type
# α = (1/φ)^p where p = F(a)/L(b)
# NOTE: Only attn_o = 1/3 is universal. Other fractions are model-specific.
# These are per-model profiles with (target_alpha, natural_std).
# The std defines a dead zone — no penalty within ±std of target.

FL_EXPONENTS_QWEN = {
    'attn_o': (INV_PHI ** (1/3), 0.048),      # 0.8518, tight
    'attn_q': (INV_PHI ** (5/4), 0.137),       # 0.5480
    'attn_k': (INV_PHI ** (2/11), 0.094),      # 0.9162
    'attn_v': (INV_PHI ** (3/7), 0.143),        # 0.8136
    'mlp_gate': (INV_PHI ** (4/7), 0.098),      # 0.7596
    'mlp_up': (INV_PHI ** (8/11), 0.062),       # 0.7047
    'mlp_down': (INV_PHI ** (5/7), 0.061),      # 0.7091
}

FL_EXPONENTS_GEMMA = {
    'attn_o': (INV_PHI ** (1/3), 0.125),       # 0.8518, universal
    'attn_q': (INV_PHI ** (2/7), 0.056),        # 0.8715
    'attn_k': (INV_PHI ** (4/18), 0.068),       # 0.8986 (L(3)/L(6))
    'attn_v': (INV_PHI ** (1/18), 0.036),       # 0.9736
    'mlp_gate': (INV_PHI ** (1/1), 0.239),      # 0.6180
    'mlp_up': (INV_PHI ** (3/4), 0.128),        # 0.6970
    'mlp_down': (INV_PHI ** (3/4), 0.170),      # 0.6970
}

# Legacy flat dict for backwards compatibility
FL_EXPONENTS = {k: v[0] for k, v in FL_EXPONENTS_QWEN.items()}

# Model profile lookup
FL_PROFILES = {
    'qwen': FL_EXPONENTS_QWEN,
    'gemma': FL_EXPONENTS_GEMMA,
}


def _get_fl_profile(model_name: str | None) -> dict:
    """Get the F/L exponent profile for a model."""
    if model_name:
        name_lower = model_name.lower()
        for key, profile in FL_PROFILES.items():
            if key in name_lower:
                return profile
    return FL_EXPONENTS_QWEN  # default


def _classify_layer_type(name: str) -> str | None:
    """Classify a module name into a layer type for harmonic priors."""
    name_lower = name.lower()
    if 'o_proj' in name_lower or 'out_proj' in name_lower:
        return 'attn_o'
    if 'q_proj' in name_lower:
        return 'attn_q'
    if 'k_proj' in name_lower:
        return 'attn_k'
    if 'v_proj' in name_lower:
        return 'attn_v'
    if 'gate_proj' in name_lower or 'gate' in name_lower:
        return 'mlp_gate'
    if 'up_proj' in name_lower:
        return 'mlp_up'
    if 'down_proj' in name_lower:
        return 'mlp_down'
    return None


def harmonic_regularization(
    module_or_model: nn.Module,
    lambda_h: float = 1.0,
    type_aware: bool = False,
    attn_o_weight: float = 10.0,
    model_name: str | None = None,
    soft_band: bool = True,
) -> torch.Tensor:
    """
    Spectral weight decay toward bent power-law prior.

    For each per_mode SpectralLinear, penalizes deviation of the
    learned spectrum from (k + k₀)^{-α} anchored at σ₁.

    Three modes:
      type_aware=False (legacy): α = 1/φ for all layers.
      type_aware=True, soft_band=False: hard pull to F/L target per type.
      type_aware=True, soft_band=True: dead zone within ±σ of target.
        Only penalizes drift BEYOND the natural pre-trained variance.
        Model-specific profiles via model_name ('qwen', 'gemma', etc).

    L_harmonic = λ · Σ w_type · mean_k[ (s_k - prior_k)² ]

    With soft_band: prior is only enforced when the effective α
    drifts more than σ_type away from the target. Within the band,
    the spectrum is free to adapt.

    Args:
        module_or_model: a SpectralLinear or any nn.Module containing them
        lambda_h: regularization strength (applied as multiplier)
        type_aware: use F/L exponents per layer type
        attn_o_weight: extra weight on attn_o layers (default: 10.0)
        model_name: model identifier for profile lookup ('qwen', 'gemma')
        soft_band: allow natural variance (dead zone within ±σ)

    Returns:
        Scalar loss tensor (differentiable through spectrum params).
    """
    if isinstance(module_or_model, SpectralLinear):
        named_modules = [('', module_or_model)]
    else:
        named_modules = [
            (name, m) for name, m in module_or_model.named_modules()
            if isinstance(m, SpectralLinear)
        ]

    profile = _get_fl_profile(model_name) if type_aware else None

    loss = torch.tensor(0.0)
    count = 0

    for name, m in named_modules:
        if m.mode != 'per_mode':
            continue
        s = m.get_spectrum()
        device = s.device
        k = torch.arange(1, len(s) + 1, device=device, dtype=s.dtype)

        # Use bent power law if k₀ available, else simple
        k0 = getattr(m, 'k0', None)
        if k0 is not None:
            k_shifted = k + k0
        else:
            k_shifted = k

        # Determine exponent and band
        if type_aware and profile:
            ltype = _classify_layer_type(name)
            entry = profile.get(ltype)
            if entry:
                alpha, nat_std = entry
            else:
                alpha, nat_std = INV_PHI, 0.1
            weight = attn_o_weight if ltype == 'attn_o' else 1.0
        else:
            alpha = INV_PHI
            nat_std = 0.1
            weight = 1.0

        # Prior anchored at σ₁ (detached so prior doesn't push σ₁)
        A = s[0].detach() / k_shifted[0].pow(-alpha)
        prior = A * k_shifted.pow(-alpha)
        deviation = (s - prior) ** 2

        if soft_band and type_aware:
            # Dead zone: scale penalty by how far the MEAN deviation
            # exceeds the natural variance band. Within band → no penalty.
            mean_dev = deviation.mean()
            # Approximate: the natural std in α maps to a std in the spectrum.
            # A rough proxy: allow the mean squared deviation to be up to
            # (nat_std * σ₁)² before penalizing.
            sigma1 = s[0].detach()
            band_sq = (nat_std * sigma1) ** 2
            # Soft hinge: only penalize excess beyond band
            excess = torch.clamp(mean_dev - band_sq, min=0.0)
            loss = loss.to(device) + weight * excess
        else:
            loss = loss.to(device) + weight * deviation.mean()
        count += 1

    return lambda_h * loss


def compute_adaptive_rank(
    r2: float,
    base_rank: int,
    beta: float = 2.0,
    max_rank: int | None = None,
) -> int:
    """
    Allocate rank proportional to how poorly the bent power law fits.

    Layers with high R² are well-described by the prior — fewer free
    modes needed. Layers with low R² need more spectral freedom.

    rank_k = base_rank × (1 + β × (1 - R²))

    Args:
        r2: R² of bent power-law fit for this layer (0 to 1)
        base_rank: rank for a perfectly-fitted layer (R²=1)
        beta: sensitivity to poor fit (default 2.0)
        max_rank: hard cap on rank allocation

    Returns:
        Integer rank for this layer.
    """
    deviation = max(0.0, 1.0 - r2)
    rank = int(base_rank * (1.0 + beta * deviation))
    if max_rank is not None:
        rank = min(rank, max_rank)
    return rank


def fit_bent_power_law(
    S: np.ndarray | torch.Tensor,
) -> dict:
    """
    Fit the bent power law: σ_k = A · (k + k₀)^{-1/φ}.

    Exponent is FIXED at 1/φ. Only A and k₀ are fitted.
    Uses scipy curve_fit on the top 90% of singular values.

    Args:
        S: singular values (1D array, descending order)

    Returns:
        dict with keys: A, k0, r2, n_fit
    """
    from scipy.optimize import curve_fit

    if isinstance(S, torch.Tensor):
        s_np = S.detach().cpu().float().numpy()
    else:
        s_np = np.asarray(S, dtype=np.float64)

    n_fit = max(int(len(s_np) * 0.9), 4)
    k = np.arange(1, n_fit + 1, dtype=np.float64)
    y = s_np[:n_fit].astype(np.float64)

    def bent(k, A, k0):
        return A * (k + k0) ** (-INV_PHI)

    try:
        popt, _ = curve_fit(
            bent, k, y,
            p0=[y[0] * 50, max(len(s_np) * 0.1, 10)],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000,
        )
        pred = bent(k, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {'A': float(popt[0]), 'k0': float(popt[1]), 'r2': r2, 'n_fit': n_fit}
    except Exception:
        # Fallback: return simple fit results with k0=0
        return {'A': float(s_np[0]), 'k0': 0.0, 'r2': 0.0, 'n_fit': n_fit}


def fit_alpha(weight: torch.Tensor) -> float:
    """
    Quick power-law fit on a weight matrix (no decomposition stored).

    Returns the fitted α from log(σ_k) = log(σ₁) - α·log(k).

    Note: For large models (d≥5120), prefer fit_bent_power_law()
    which uses the correct shifted equation with fixed 1/φ exponent.
    """
    W = weight.data.float().cpu()
    S = torch.linalg.svdvals(W)
    s_np = S.numpy()
    n_fit = max(int(len(s_np) * 0.9), 4)
    log_k = np.log(np.arange(1, n_fit + 1))
    log_s = np.log(s_np[:n_fit] + 1e-10)
    coeffs = np.polyfit(log_k, log_s, 1)
    return float(-coeffs[0])
