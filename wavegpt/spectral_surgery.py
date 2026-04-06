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


def spectral_decompose(
    model: nn.Module,
    rank: int | None = None,
    mode: str = 'per_mode',
    skip_patterns: list[str] | None = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with SpectralLinear (in-place).

    Args:
        model: Any nn.Module
        rank: SVD truncation rank (None = auto 95% energy)
        mode: 'sigma1' or 'per_mode'
        skip_patterns: list of regex patterns for layer names to skip

    Returns:
        The same model with nn.Linear layers replaced.
    """
    skip_patterns = skip_patterns or []

    def should_skip(name: str) -> bool:
        return any(re.search(p, name) for p in skip_patterns)

    # Collect all (parent, attr_name, full_name) for nn.Linear modules
    replacements: list[tuple[nn.Module, str, str]] = []
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if should_skip(full_name):
                continue
            replacements.append(full_name)

    # Replace each linear layer
    for full_name in replacements:
        # Navigate to the parent module
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        attr_name = parts[-1]
        linear = getattr(parent, attr_name) if not attr_name.isdigit() else parent[int(attr_name)]

        spec = SpectralLinear.from_linear(linear, rank=rank, mode=mode)

        if attr_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(attr_name)] = spec
        else:
            setattr(parent, attr_name, spec)

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
