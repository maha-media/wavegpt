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
    k0_mult: float = 0.0,
    k0_pad: int = 0,
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

        # For k0-adaptive: first decompose at max rank to get k0, then re-decompose
        if k0_mult > 0:
            # Quick SVD to get k₀
            W = linear.weight.data.float().cpu()
            S_vals = torch.linalg.svdvals(W)
            try:
                bent = fit_bent_power_law(S_vals)
                k0_val = bent['k0']
            except Exception:
                k0_val = 0
            min_dim = min(linear.weight.shape)
            layer_rank = min(int(k0_val * k0_mult) + k0_pad, min_dim)
            layer_rank = max(layer_rank, 32)  # minimum rank

        spec = SpectralLinear.from_linear(
            linear, rank=layer_rank, mode=mode, keep_residual=keep_residual,
        )

        # Progress logging
        _elapsed = _time.time() - _t0
        k0_str = f"k₀={spec.k0.item():.0f} " if spec.k0 is not None else ""
        if _i > 0:
            _eta = _elapsed / _i * (_total - _i)
            print(f"  [{_i+1}/{_total}] {full_name} "
                  f"{tuple(linear.weight.shape)} → rank {spec.rank} "
                  f"{k0_str}"
                  f"({_elapsed:.0f}s elapsed, ETA {_eta:.0f}s)",
                  flush=True)
        else:
            print(f"  [{_i+1}/{_total}] {full_name} "
                  f"{tuple(linear.weight.shape)} → rank {spec.rank} {k0_str}",
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
    state_dict: dict | None = None,
) -> nn.Module:
    """
    Replace nn.Linear layers with empty SpectralLinear shells (no SVD).

    Creates the correct architecture for loading a saved state_dict.
    5 seconds instead of 3 hours.

    If state_dict is provided, infers per-layer rank from the saved
    spectrum/U tensor shapes (needed for adaptive-rank models).

    Usage:
        # Fixed rank:
        spectral_scaffold(model, rank=256, mode='per_mode')
        model.load_state_dict(torch.load('decomposed.pt'), strict=False)

        # Variable rank (from adaptive decomposition):
        sd = torch.load('decomposed.pt', map_location='cpu')
        spectral_scaffold(model, mode='per_mode', state_dict=sd)
        model.load_state_dict(sd, strict=False)
    """
    skip_patterns = skip_patterns or []

    def should_skip(name: str) -> bool:
        return any(re.search(p, name) for p in skip_patterns)

    # Build rank map and residual set from state dict if provided
    rank_map: dict[str, int] = {}
    residual_set: set[str] = set()
    migrated_count = 0
    if state_dict is not None:
        # Migrate old .spectrum keys → .log_spectrum (log-space parameterization)
        keys_to_migrate = [k for k in state_dict if k.endswith('.spectrum') and state_dict[k].dim() == 1]
        for key in keys_to_migrate:
            raw_spectrum = state_dict.pop(key)
            log_key = key.rsplit('.spectrum', 1)[0] + '.log_spectrum'
            state_dict[log_key] = torch.log(raw_spectrum.clamp(min=1e-12))
            # Also create the frozen init buffer
            init_key = key.rsplit('.spectrum', 1)[0] + '.log_spectrum_init'
            state_dict[init_key] = state_dict[log_key].clone()
            migrated_count += 1

        for key, tensor in state_dict.items():
            # log_spectrum tensor: <layer_name>.log_spectrum with shape (rank,)
            if key.endswith('.log_spectrum') and tensor.dim() == 1:
                layer_name = key.rsplit('.log_spectrum', 1)[0]
                rank_map[layer_name] = tensor.shape[0]
            # Fallback: U_basis tensor: <layer_name>.U with shape (out, rank)
            elif key.endswith('.U') and tensor.dim() == 2:
                layer_name = key.rsplit('.U', 1)[0]
                if layer_name not in rank_map:
                    rank_map[layer_name] = tensor.shape[1]
            # Track which layers have residual buffers
            elif key.endswith('.residual'):
                layer_name = key.rsplit('.residual', 1)[0]
                residual_set.add(layer_name)
        if migrated_count > 0:
            print(f"  Migrated {migrated_count} layers: .spectrum → .log_spectrum (log-space)")
        if rank_map:
            ranks = list(rank_map.values())
            print(f"  Rank map from state_dict: {len(rank_map)} layers, "
                  f"rank range [{min(ranks)}, {max(ranks)}]")
        if residual_set:
            print(f"  Residual buffers: {len(residual_set)} layers")

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

        # Determine rank: from state_dict if available, else fixed
        layer_rank = rank_map.get(full_name, rank)
        has_residual = full_name in residual_set

        spec = SpectralLinear.from_shape(
            out_dim, in_dim, rank=layer_rank, mode=mode,
            has_bias=has_bias, has_residual=has_residual, dtype=dtype,
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
