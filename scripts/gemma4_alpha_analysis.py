"""
Free-α analysis for Gemma 4 31B.

Adapted from free_alpha_analysis.py with Gemma-specific handling:
  - Filters out vision_tower layers (only analyzes language model)
  - Sub-classifies attention layers by sliding vs full attention
  - Handles attention_k_eq_v (K=V sharing on full attention layers)
  - Depth bands adjusted for 60 layers

Usage:
    python gemma4_alpha_analysis.py --local-dir /workspace/gemma4-31b
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

INV_PHI = 2.0 / (1.0 + np.sqrt(5))  # 0.6180339887...

# Gemma 4 31B: every 6th layer (5,11,17,23,29,35,41,47,53,59) is full attention
FULL_ATTN_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}


def bent_power_law(k, A, k0, alpha):
    """σ_k = A · (k + k₀)^{-α} — all three parameters free."""
    return A * (k + k0) ** (-alpha)


def fit_free_alpha(S: np.ndarray):
    """Fit bent power law with α free. Returns dict with A, k0, alpha, r2."""
    S = S[S > 1e-10]
    n = len(S)
    if n < 10:
        return None

    k = np.arange(1, n + 1, dtype=np.float64)
    s = S.astype(np.float64)

    A0 = float(s[0])
    k0_guess = max(1.0, n * 0.1)
    alpha_guess = INV_PHI

    try:
        popt, _ = curve_fit(
            bent_power_law, k, s,
            p0=[A0, k0_guess, alpha_guess],
            bounds=([0, 0, 0.01], [A0 * 100, n * 5, 2.0]),
            maxfev=10000,
        )
        A_fit, k0_fit, alpha_fit = popt

        s_pred = bent_power_law(k, *popt)
        ss_res = np.sum((s - s_pred) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'A': float(A_fit),
            'k0': float(k0_fit),
            'alpha': float(alpha_fit),
            'r2': float(r2),
        }
    except Exception:
        return None


def fit_fixed_phi(S: np.ndarray):
    """Fit bent power law with α FIXED at 1/φ. Returns dict with A, k0, r2."""
    S = S[S > 1e-10]
    n = len(S)
    if n < 10:
        return None

    k = np.arange(1, n + 1, dtype=np.float64)
    s = S.astype(np.float64)

    def model(k, A, k0):
        return A * (k + k0) ** (-INV_PHI)

    A0 = float(s[0])
    k0_guess = max(1.0, n * 0.1)

    try:
        popt, _ = curve_fit(
            model, k, s,
            p0=[A0, k0_guess],
            bounds=([0, 0], [A0 * 100, n * 5]),
            maxfev=10000,
        )
        A_fit, k0_fit = popt

        s_pred = model(k, *popt)
        ss_res = np.sum((s - s_pred) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'A': float(A_fit),
            'k0': float(k0_fit),
            'alpha': float(INV_PHI),
            'r2': float(r2),
        }
    except Exception:
        return None


def classify_layer(name, depth):
    """
    Classify layer by type, sub-classifying attention by sliding vs full.

    Gemma 4 architecture:
      - 50 sliding attention layers (GQA 32:16, head_dim=256)
      - 10 full attention layers (GQA 32:4, global_head_dim=512)
      - attention_k_eq_v on full attention → no separate v_proj
      - MLP: gate_proj + up_proj + down_proj (GELU, not SiLU)
    """
    is_full = depth in FULL_ATTN_LAYERS
    attn_suffix = '_full' if is_full else '_slide'

    if 'gate_proj' in name: return 'mlp_gate'
    if 'up_proj' in name: return 'mlp_up'
    if 'down_proj' in name: return 'mlp_down'
    if 'q_proj' in name: return 'attn_q' + attn_suffix
    if 'k_proj' in name: return 'attn_k' + attn_suffix
    if 'v_proj' in name: return 'attn_v' + attn_suffix
    if 'o_proj' in name: return 'attn_o' + attn_suffix
    if 'embed' in name: return 'embedding'
    if 'lm_head' in name: return 'lm_head'
    return 'other'


def get_depth(name):
    """Extract layer depth from name."""
    m = re.search(r'layers\.(\d+)', name)
    return int(m.group(1)) if m else -1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir', default='/workspace/gemma4-31b',
                        help='Local directory with safetensors + config.json')
    parser.add_argument('--output', default='gemma4-free-alpha.json')
    parser.add_argument('--min-dim', type=int, default=64,
                        help='Skip layers with min(shape) < this')
    args = parser.parse_args()

    from safetensors import safe_open

    model_dir = Path(args.local_dir)

    # Find all safetensor files
    st_files = sorted(model_dir.glob('*.safetensors'))
    print(f"Found {len(st_files)} safetensor shards")

    # Collect all weight tensor names and shapes — LANGUAGE MODEL ONLY
    tensor_map = {}
    for f in st_files:
        with safe_open(str(f), framework='pt') as sf:
            for name in sf.keys():
                # Skip vision tower entirely
                if 'vision_tower' in name or 'embed_vision' in name:
                    continue
                shape = sf.get_slice(name).get_shape()
                if len(shape) == 2 and 'weight' in name:
                    tensor_map[name] = (str(f), shape)

    print(f"Found {len(tensor_map)} 2D weight tensors (language model only)")

    # Show what we're analyzing
    type_counts = {}
    for name in tensor_map:
        depth = get_depth(name)
        ltype = classify_layer(name, depth)
        type_counts[ltype] = type_counts.get(ltype, 0) + 1
    print("Layer types:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    results = []
    skipped = 0
    t0 = time.time()

    for i, (name, (filepath, shape)) in enumerate(sorted(tensor_map.items())):
        min_dim = min(shape)
        if min_dim < args.min_dim:
            skipped += 1
            continue

        with safe_open(filepath, framework='pt') as sf:
            W = sf.get_tensor(name).float().cpu()

        # SVD
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        S_np = S.numpy()

        free_fit = fit_free_alpha(S_np)
        fixed_fit = fit_fixed_phi(S_np)

        if free_fit is None:
            skipped += 1
            continue

        depth = get_depth(name)
        layer_type = classify_layer(name, depth)

        result = {
            'name': name,
            'shape': list(shape),
            'min_dim': min_dim,
            'depth': depth,
            'type': layer_type,
            'free_alpha': free_fit['alpha'],
            'free_k0': free_fit['k0'],
            'free_A': free_fit['A'],
            'free_r2': free_fit['r2'],
            'fixed_r2': fixed_fit['r2'] if fixed_fit else None,
            'delta_alpha': free_fit['alpha'] - INV_PHI,
        }
        results.append(result)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1 - skipped) * (len(tensor_map) - i - 1) if (i + 1 - skipped) > 0 else 0
        print(f"  [{len(results)}/{len(tensor_map)-skipped}] {name} ({shape[0]}x{shape[1]}) "
              f"type={layer_type} "
              f"alpha={free_fit['alpha']:.4f} delta={free_fit['alpha']-INV_PHI:+.4f} "
              f"k0={free_fit['k0']:.0f} R2={free_fit['r2']:.4f} "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("GEMMA 4 31B — FREE-ALPHA SPECTRAL ANALYSIS")
    print("=" * 70)

    alphas = np.array([r['free_alpha'] for r in results])
    deltas = np.array([r['delta_alpha'] for r in results])
    r2_free = np.array([r['free_r2'] for r in results])
    r2_fixed = np.array([r['fixed_r2'] for r in results if r['fixed_r2'] is not None])

    print(f"\n  Layers analyzed:  {len(results)}")
    print(f"  Layers skipped:   {skipped}")
    print(f"  mean(alpha):      {np.mean(alphas):.6f}")
    print(f"  std(alpha):       {np.std(alphas):.6f}")
    print(f"  median(alpha):    {np.median(alphas):.6f}")
    print(f"  1/phi:            {INV_PHI:.6f}")

    # R2 comparison
    print(f"\n  R2 comparison (free alpha vs fixed 1/phi):")
    print(f"    mean R2(free):   {np.mean(r2_free):.4f}")
    print(f"    mean R2(fixed):  {np.mean(r2_fixed):.4f}")

    # === KEY TABLE: alpha by layer type ===
    print("\n" + "=" * 70)
    print("ALPHA BY LAYER TYPE")
    print("=" * 70)

    by_type = {}
    for r in results:
        by_type.setdefault(r['type'], []).append(r)

    # Also compute combined attention types (sliding + full merged)
    combined = {}
    for t, rs in by_type.items():
        base = t.replace('_slide', '').replace('_full', '')
        combined.setdefault(base, []).extend(rs)

    print("\n  Detailed (sliding vs full attention):")
    print(f"  {'Type':<16s}  {'mean alpha':>10s}  {'std':>7s}  {'n':>4s}  {'mean k0':>8s}  {'mean R2':>8s}")
    print("  " + "-" * 60)
    for t in sorted(by_type, key=lambda x: np.mean([r['free_alpha'] for r in by_type[x]])):
        rs = by_type[t]
        as_ = [r['free_alpha'] for r in rs]
        k0s = [r['free_k0'] for r in rs]
        r2s = [r['free_r2'] for r in rs]
        print(f"  {t:<16s}  {np.mean(as_):>10.4f}  {np.std(as_):>7.4f}  {len(rs):>4d}  {np.mean(k0s):>8.0f}  {np.mean(r2s):>8.4f}")

    print("\n  Combined (attention types merged):")
    print(f"  {'Type':<16s}  {'mean alpha':>10s}  {'std':>7s}  {'n':>4s}  {'mean k0':>8s}")
    print("  " + "-" * 50)
    for t in sorted(combined, key=lambda x: np.mean([r['free_alpha'] for r in combined[x]])):
        rs = combined[t]
        as_ = [r['free_alpha'] for r in rs]
        k0s = [r['free_k0'] for r in rs]
        print(f"  {t:<16s}  {np.mean(as_):>10.4f}  {np.std(as_):>7.4f}  {len(rs):>4d}  {np.mean(k0s):>8.0f}")

    # === THE BIG QUESTION: attn_o ===
    print("\n" + "=" * 70)
    print("THE BIG QUESTION: IS attn_o = 1/3?")
    print("=" * 70)
    attn_o_all = [r['free_alpha'] for r in results if r['type'].startswith('attn_o')]
    if attn_o_all:
        mean_o = np.mean(attn_o_all)
        predicted = INV_PHI ** (1/3)  # (1/phi)^(1/3) = 0.8517...
        error_pct = abs(mean_o - predicted) / predicted * 100
        print(f"  attn_o mean alpha:  {mean_o:.4f}")
        print(f"  predicted (1/phi)^(1/3): {predicted:.4f}")
        print(f"  error: {error_pct:.2f}%")
        if error_pct < 1.0:
            print(f"  >>> WITHIN 1% — HOLDS ON GEMMA 4 <<<")
        elif error_pct < 3.0:
            print(f"  >>> Within 3% — close but not exact <<<")
        else:
            print(f"  >>> DOES NOT HOLD (>{error_pct:.1f}% off) <<<")

        # Sliding vs full attn_o
        o_slide = [r['free_alpha'] for r in results if r['type'] == 'attn_o_slide']
        o_full = [r['free_alpha'] for r in results if r['type'] == 'attn_o_full']
        if o_slide:
            print(f"  attn_o sliding (n={len(o_slide)}): {np.mean(o_slide):.4f} +/- {np.std(o_slide):.4f}")
        if o_full:
            print(f"  attn_o full    (n={len(o_full)}):  {np.mean(o_full):.4f} +/- {np.std(o_full):.4f}")

    # Histogram
    print(f"\n  Histogram of alpha (all types):")
    hist, edges = np.histogram(alphas, bins=20)
    max_count = max(hist)
    for j in range(len(hist)):
        lo, hi = edges[j], edges[j + 1]
        bar = '#' * int(40 * hist[j] / max_count) if max_count > 0 else ''
        marker = ' <-- 1/phi' if lo <= INV_PHI <= hi else ''
        print(f"    {lo:.3f}-{hi:.3f} | {bar} {hist[j]}{marker}")

    # By depth band
    print("\n  Alpha by depth band:")
    by_depth = {}
    for r in results:
        if r['depth'] < 0: continue
        band = 'early(0-19)' if r['depth'] < 20 else 'mid(20-39)' if r['depth'] < 40 else 'late(40-59)'
        by_depth.setdefault(band, []).append(r['free_alpha'])
    for band in sorted(by_depth):
        vals = by_depth[band]
        print(f"    {band:<16s}  mean alpha={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        'model': 'google/gemma-4-31B',
        'architecture': 'Gemma4ForConditionalGeneration',
        'n_text_layers': 60,
        'n_full_attention': 10,
        'n_sliding_attention': 50,
        'gqa_sliding': '32:16',
        'gqa_full': '32:4',
        'activation': 'gelu_pytorch_tanh',
        'k_eq_v': True,
        'n_analyzed': len(results),
        'mean_alpha': float(np.mean(alphas)),
        'std_alpha': float(np.std(alphas)),
        'median_alpha': float(np.median(alphas)),
        'inv_phi': float(INV_PHI),
        'mean_r2_free': float(np.mean(r2_free)),
        'mean_r2_fixed': float(np.mean(r2_fixed)) if len(r2_fixed) > 0 else None,
        'layers': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == '__main__':
    main()
