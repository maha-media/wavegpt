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
from .harmonic_prior import fit_alpha, compute_adaptive_rank


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
            - 'adaptive': theory-guided rank via 1/φ deviation
        mode: 'sigma1' or 'per_mode'
        skip_patterns: list of regex patterns for layer names to skip
        keep_residual: if True, store frozen W_residual (Pythagorean comma)
        base_rank: base rank when rank='adaptive' (rank at α = 1/φ)
        adaptive_beta: sensitivity to α deviation when rank='adaptive'
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

    # If adaptive, first pass: fit α per layer
    layer_ranks: dict[str, int] = {}
    if adaptive:
        for full_name in replacements:
            linear = _get_submodule(model, full_name)
            alpha = fit_alpha(linear.weight)
            dim_max = min(linear.weight.shape)
            lr = compute_adaptive_rank(
                alpha, base_rank, beta=adaptive_beta, max_rank=max_rank,
            )
            layer_ranks[full_name] = min(lr, dim_max)

    # Replace each linear layer
    for full_name in replacements:
        linear = _get_submodule(model, full_name)

        if adaptive:
            layer_rank = layer_ranks[full_name]
        else:
            layer_rank = rank  # int or None

        spec = SpectralLinear.from_linear(
            linear, rank=layer_rank, mode=mode, keep_residual=keep_residual,
        )

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
