"""
Spectral Surgery — decompose any model's nn.Linear layers into SpectralLinear.

Walk the module tree, replace each nn.Linear with a SpectralLinear that
has frozen geometry (U, V) and learnable spectral amplitudes.

Usage:
    from wavegpt.spectral_surgery import spectral_decompose, spectral_report

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    decomposed = spectral_decompose(model, rank=64, mode='per_mode')
    report = spectral_report(decomposed)
"""
from __future__ import annotations

import re

import torch.nn as nn

from .spectral_linear import SpectralLinear
from .harmonic_prior import fit_alpha, fit_bent_power_law, compute_adaptive_rank

import torch


def spectral_decompose(
    model: nn.Module,
    rank: int | str | None = None,
    mode: str = 'per_mode',
    skip_patterns: list[str] | None = None,
    keep_residual: bool = False,
    base_rank: int = 192,
    adaptive_beta: float = 2.0,
    max_rank: int | None = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with SpectralLinear (in-place).

    Args:
        model: Any nn.Module
        rank: SVD truncation rank. Options:
            - int: fixed rank for all layers
            - None: auto 95% energy per layer
            - 'adaptive': theory-guided rank via bent power-law R²
        mode: 'sigma1' or 'per_mode'
        skip_patterns: list of regex patterns for layer names to skip
        keep_residual: if True, store frozen W_residual (Pythagorean comma)
        base_rank: base rank when rank='adaptive' (rank at R²=1)
        adaptive_beta: sensitivity to poor fit when rank='adaptive'
        max_rank: hard cap on per-layer rank when adaptive

    Returns:
        The same model with nn.Linear layers replaced.
    """
    skip_patterns = skip_patterns or []
    adaptive = (rank == 'adaptive')

    def should_skip(name: str) -> bool:
        return any(re.search(p, name) for p in skip_patterns)

    # Collect names of nn.Linear modules to replace
    replacements: list[str] = []
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_skip(full_name):
                continue
            replacements.append(full_name)

    # If adaptive, first pass: fit bent power law per layer for rank
    layer_ranks: dict[str, int] = {}
    if adaptive:
        for full_name in replacements:
            linear = _get_submodule(model, full_name)
            W = linear.weight.data.float().cpu()
            S = torch.linalg.svdvals(W)
            bent = fit_bent_power_law(S)
            dim_max = min(linear.weight.shape)
            lr = compute_adaptive_rank(
                bent['r2'], base_rank, beta=adaptive_beta, max_rank=max_rank,
            )
            layer_ranks[full_name] = min(lr, dim_max)

    # Replace each linear layer
    import time as _time
    _t0 = _time.time()
    _total = len(replacements)
    for _i, full_name in enumerate(replacements):
        linear = _get_submodule(model, full_name)

        if adaptive:
            layer_rank = layer_ranks[full_name]
        else:
            layer_rank = rank  # int or None

        spec = SpectralLinear.from_linear(
            linear, rank=layer_rank, mode=mode, keep_residual=keep_residual,
        )

        # Progress logging
        _elapsed = _time.time() - _t0
        if _i > 0:
            _eta = _elapsed / _i * (_total - _i)
            print(f"  [{_i+1}/{_total}] {full_name} "
                  f"{tuple(linear.weight.shape)} → rank {spec.rank} "
                  f"k₀={spec.k0.item():.0f} "
                  f"({_elapsed:.0f}s elapsed, ETA {_eta:.0f}s)",
                  flush=True)
        else:
            print(f"  [{_i+1}/{_total}] {full_name} "
                  f"{tuple(linear.weight.shape)} → rank {spec.rank}",
                  flush=True)

        # Set the SpectralLinear on the parent
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        attr_name = parts[-1]
        if attr_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr_name)] = spec
        else:
            setattr(parent, attr_name, spec)

    return model


def _get_submodule(model: nn.Module, full_name: str) -> nn.Module:
    """Navigate dotted name to get a submodule."""
    parts = full_name.split('.')
    module = model
    for part in parts:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def spectral_scaffold(
    model: nn.Module,
    rank: int = 256,
    mode: str = 'per_mode',
    skip_patterns: list[str] | None = None,
) -> nn.Module:
    """
    Replace nn.Linear layers with empty SpectralLinear shells (no SVD).

    Creates the correct architecture for loading a saved state_dict.
    5 seconds instead of 3 hours.

    Usage:
        spectral_scaffold(model, rank=256, mode='per_mode')
        model.load_state_dict(torch.load('decomposed.pt'))
    """
    skip_patterns = skip_patterns or []

    def should_skip(name: str) -> bool:
        return any(re.search(p, name) for p in skip_patterns)

    replacements = []
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_skip(full_name):
                continue
            replacements.append(full_name)

    for full_name in replacements:
        linear = _get_submodule(model, full_name)
        out_dim, in_dim = linear.weight.shape
        has_bias = linear.bias is not None
        dtype = linear.weight.dtype

        spec = SpectralLinear.from_shape(
            out_dim, in_dim, rank=rank, mode=mode,
            has_bias=has_bias, dtype=dtype,
        )

        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        attr_name = parts[-1]
        if attr_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr_name)] = spec
        else:
            setattr(parent, attr_name, spec)

    print(f"  Scaffolded {len(replacements)} layers → SpectralLinear (no SVD)")
    return model


def spectral_report(model: nn.Module) -> dict:
    """
    Generate spectral report for all SpectralLinear layers.

    Returns:
        dict mapping layer name → spectral info (alpha, sigma1, rank, etc.)
    """
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, SpectralLinear):
            report[name] = module.spectral_report()
    return report
