"""
Free-α analysis: fit σ_k = A · (k + k₀)^{-α} with α FREE (not fixed at 1/φ).
Loads individual weight tensors from safetensors — no full model load needed.
Reports: mean(α), std(α), histogram, regression of Δα against type/depth/dim.

The number that matters: std(α). If < 0.01, 1/φ is exact. If > 0.05, it's narrative.
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


def bent_power_law(k, A, k0, alpha):
    """σ_k = A · (k + k₀)^{-α} — all three parameters free."""
    return A * (k + k0) ** (-alpha)


def fit_free_alpha(S: np.ndarray):
    """Fit bent power law with α free. Returns dict with A, k0, alpha, r2."""
    S = S[S > 1e-10]  # drop near-zero
    n = len(S)
    if n < 10:
        return None

    k = np.arange(1, n + 1, dtype=np.float64)
    s = S.astype(np.float64)

    # Initial guesses
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
    except Exception as e:
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


def classify_layer(name):
    """Classify layer by type."""
    if 'gate_proj' in name: return 'mlp_gate'
    if 'up_proj' in name: return 'mlp_up'
    if 'down_proj' in name: return 'mlp_down'
    if 'in_proj_qkv' in name: return 'delta_qkv'
    if 'in_proj_z' in name: return 'delta_z'
    if 'in_proj_a' in name or 'in_proj_b' in name: return 'delta_bias'
    if 'out_proj' in name and 'linear_attn' in name: return 'delta_out'
    if 'q_proj' in name: return 'attn_q'
    if 'k_proj' in name: return 'attn_k'
    if 'v_proj' in name: return 'attn_v'
    if 'o_proj' in name: return 'attn_o'
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
    parser.add_argument('--hf-model', default='Qwen/Qwen3.5-27B')
    parser.add_argument('--output', default='runs/free-alpha-analysis.json')
    parser.add_argument('--min-dim', type=int, default=64,
                        help='Skip layers with min(shape) < this')
    args = parser.parse_args()

    from safetensors import safe_open
    from transformers import AutoConfig
    from huggingface_hub import snapshot_download

    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)
    model_path = snapshot_download(args.hf_model)
    model_dir = Path(model_path)

    # Find all safetensor files
    st_files = sorted(model_dir.glob('*.safetensors'))
    print(f"Found {len(st_files)} safetensor shards")

    # Collect all weight tensor names and shapes
    tensor_map = {}  # name -> (file, shape)
    for f in st_files:
        with safe_open(str(f), framework='pt') as sf:
            for name in sf.keys():
                shape = sf.get_slice(name).get_shape()
                if len(shape) == 2 and 'weight' in name:
                    tensor_map[name] = (str(f), shape)

    print(f"Found {len(tensor_map)} 2D weight tensors")

    results = []
    skipped = 0
    t0 = time.time()

    for i, (name, (filepath, shape)) in enumerate(sorted(tensor_map.items())):
        min_dim = min(shape)
        if min_dim < args.min_dim:
            skipped += 1
            continue

        # Load single tensor
        with safe_open(filepath, framework='pt') as sf:
            W = sf.get_tensor(name).float().cpu()

        # SVD
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        S_np = S.numpy()

        # Fit with free α
        free_fit = fit_free_alpha(S_np)
        # Fit with fixed 1/φ
        fixed_fit = fit_fixed_phi(S_np)

        if free_fit is None:
            skipped += 1
            continue

        depth = get_depth(name)
        layer_type = classify_layer(name)

        result = {
            'name': name,
            'shape': list(shape),
            'min_dim': min_dim,
            'depth': depth,
            'type': layer_type,
            'free_alpha': free_fit['alpha'],
            'free_k0': free_fit['k0'],
            'free_r2': free_fit['r2'],
            'fixed_r2': fixed_fit['r2'] if fixed_fit else None,
            'delta_alpha': free_fit['alpha'] - INV_PHI,
        }
        results.append(result)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1 - skipped) * (len(tensor_map) - i - 1) if (i + 1 - skipped) > 0 else 0
        print(f"  [{len(results)}/{len(tensor_map)-skipped}] {name} ({shape[0]}×{shape[1]}) "
              f"α={free_fit['alpha']:.4f} Δα={free_fit['alpha']-INV_PHI:+.4f} "
              f"k₀={free_fit['k0']:.0f} R²={free_fit['r2']:.4f} "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    # === ANALYSIS ===
    alphas = np.array([r['free_alpha'] for r in results])
    deltas = np.array([r['delta_alpha'] for r in results])
    r2_free = np.array([r['free_r2'] for r in results])
    r2_fixed = np.array([r['fixed_r2'] for r in results if r['fixed_r2'] is not None])

    print("\n" + "=" * 70)
    print("THE NUMBER THAT MATTERS")
    print("=" * 70)
    print(f"  Layers analyzed:  {len(results)}")
    print(f"  Layers skipped:   {skipped} (dim < {args.min_dim})")
    print(f"  mean(α):          {np.mean(alphas):.6f}")
    print(f"  std(α):           {np.std(alphas):.6f}")
    print(f"  median(α):        {np.median(alphas):.6f}")
    print(f"  1/φ:              {INV_PHI:.6f}")
    print(f"  mean(Δα):         {np.mean(deltas):+.6f}")
    print(f"  |mean(Δα)|:       {abs(np.mean(deltas)):.6f}")
    print()

    # Verdict
    std_alpha = np.std(alphas)
    if std_alpha < 0.005:
        verdict = "1/φ IS EXACT. Write the theorem."
    elif std_alpha < 0.02:
        verdict = "1/φ is the leading term with small structured corrections. Strong paper."
    elif std_alpha < 0.05:
        verdict = "Power-law real, ~0.618 cluster. Suggestive, not proven. Need more models."
    else:
        verdict = "Broad basin. 1/φ may be narrative imposed on flexible fit."
    print(f"  VERDICT: {verdict}")

    # Histogram
    print("\n  Histogram of α:")
    hist, edges = np.histogram(alphas, bins=20)
    max_count = max(hist)
    for j in range(len(hist)):
        lo, hi = edges[j], edges[j + 1]
        bar = '█' * int(40 * hist[j] / max_count) if max_count > 0 else ''
        marker = ' ◄── 1/φ' if lo <= INV_PHI <= hi else ''
        print(f"    {lo:.3f}-{hi:.3f} | {bar} {hist[j]}{marker}")

    # R² comparison: free vs fixed
    print(f"\n  R² comparison (free α vs fixed 1/φ):")
    print(f"    mean R²(free):   {np.mean(r2_free):.4f}")
    print(f"    mean R²(fixed):  {np.mean(r2_fixed):.4f}")
    print(f"    mean ΔR²:        {np.mean(r2_free) - np.mean(r2_fixed):+.4f}")
    wins_free = sum(1 for r in results if r['fixed_r2'] and r['free_r2'] > r['fixed_r2'] + 0.001)
    wins_fixed = sum(1 for r in results if r['fixed_r2'] and r['fixed_r2'] > r['free_r2'] + 0.001)
    ties = len(results) - wins_free - wins_fixed
    print(f"    Free α wins:     {wins_free}")
    print(f"    Fixed 1/φ wins:  {wins_fixed}")
    print(f"    Ties (±0.001):   {ties}")

    # By type
    print("\n  α by layer type:")
    by_type = {}
    for r in results:
        by_type.setdefault(r['type'], []).append(r['free_alpha'])
    for t in sorted(by_type, key=lambda x: np.mean(by_type[x])):
        vals = by_type[t]
        print(f"    {t:<12s}  mean α={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"n={len(vals):>3d}  range=[{min(vals):.4f}, {max(vals):.4f}]")

    # By depth
    print("\n  α by depth band:")
    by_depth = {}
    for r in results:
        if r['depth'] < 0: continue
        band = 'early(0-15)' if r['depth'] < 16 else 'mid(16-47)' if r['depth'] < 48 else 'late(48-63)'
        by_depth.setdefault(band, []).append(r['free_alpha'])
    for band in sorted(by_depth):
        vals = by_depth[band]
        print(f"    {band:<16s}  mean α={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")

    # REGRESSION: Δα ~ type + depth + log(dim)
    print("\n  Regression: Δα ~ depth + log(dim) + type")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import OneHotEncoder

        depths = np.array([r['depth'] for r in results if r['depth'] >= 0]).reshape(-1, 1)
        log_dims = np.log(np.array([r['min_dim'] for r in results if r['depth'] >= 0])).reshape(-1, 1)
        types = np.array([r['type'] for r in results if r['depth'] >= 0]).reshape(-1, 1)
        y = np.array([r['delta_alpha'] for r in results if r['depth'] >= 0])

        enc = OneHotEncoder(sparse_output=False, drop='first')
        type_encoded = enc.fit_transform(types)

        X = np.hstack([depths / 64.0, log_dims, type_encoded])
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        reg_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"    R² of regression: {reg_r2:.4f}")
        print(f"    depth coef:       {reg.coef_[0]:+.4f}")
        print(f"    log(dim) coef:    {reg.coef_[1]:+.4f}")
        print(f"    intercept:        {reg.intercept_:+.4f}")

        if reg_r2 > 0.5:
            print("    → Deviations are STRUCTURED. Missing terms exist.")
        elif reg_r2 > 0.2:
            print("    → Weak structure in deviations. Possible corrections.")
        else:
            print("    → Deviations are mostly noise. 1/φ captures the signal.")
    except ImportError:
        print("    (sklearn not available — skipping regression)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        'model': args.hf_model,
        'n_layers': len(results),
        'mean_alpha': float(np.mean(alphas)),
        'std_alpha': float(np.std(alphas)),
        'median_alpha': float(np.median(alphas)),
        'inv_phi': float(INV_PHI),
        'mean_delta': float(np.mean(deltas)),
        'mean_r2_free': float(np.mean(r2_free)),
        'mean_r2_fixed': float(np.mean(r2_fixed)),
        'layers': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == '__main__':
    main()
