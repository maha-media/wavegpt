"""
Fit the k₀ equation from decomposition logs.

Tests: log(k₀) = β₀ + β₁·log(d_eff) + β₂·(depth/L) + β₃·(depth/L)² + type_offset

Usage:
    python scripts/fit_k0_equation.py --log runs/Q-B-vanilla/train.log
    python scripts/fit_k0_equation.py --log log1.txt --log log2.txt
"""
import argparse
import re
import sys
import numpy as np
from collections import defaultdict


def parse_log(path):
    """Extract layer info from decomposition log."""
    layers = []
    for line in open(path):
        m = re.search(
            r'model\.layers\.(\d+)\.(\S+)\s+\((\d+),\s*(\d+)\).*?rank\s+(\d+)(?:\s+k₀=(\d+))?',
            line
        )
        if not m:
            continue
        depth = int(m.group(1))
        name = m.group(2)
        d_out, d_in = int(m.group(3)), int(m.group(4))
        k0 = int(m.group(6)) if m.group(6) else None
        if k0 is None:
            continue

        # Classify
        if 'gate_proj' in name: t = 'mlp_gate'
        elif 'up_proj' in name: t = 'mlp_up'
        elif 'down_proj' in name: t = 'mlp_down'
        elif 'in_proj_qkv' in name: t = 'delta_qkv'
        elif 'in_proj_z' in name: t = 'delta_z'
        elif 'in_proj_a' in name or 'in_proj_b' in name: t = 'delta_bias'
        elif 'out_proj' in name and 'linear_attn' in name: t = 'delta_out'
        elif 'q_proj' in name: t = 'attn_q'
        elif 'k_proj' in name: t = 'attn_k'
        elif 'v_proj' in name: t = 'attn_v'
        elif 'o_proj' in name: t = 'attn_o'
        else: t = 'other'

        layers.append({
            'depth': depth, 'type': t, 'name': name,
            'd_out': d_out, 'd_in': d_in, 'k0': k0,
        })
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='append', required=True)
    parser.add_argument('--n-layers', type=int, default=64,
                        help='Total layers in model (for depth normalization)')
    args = parser.parse_args()

    # Parse all logs, deduplicate
    all_layers = []
    for path in args.log:
        all_layers.extend(parse_log(path))

    seen = set()
    unique = []
    for r in all_layers:
        key = (r['depth'], r['type'], r['d_out'], r['d_in'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Filter: skip tiny bias layers and k₀=0
    data = [r for r in unique if r['type'] != 'delta_bias' and r['k0'] > 0]
    print(f"Total layers: {len(data)} (from {len(unique)} unique, {len(all_layers)} raw)")

    L = args.n_layers
    types = sorted(set(r['type'] for r in data))
    print(f"Layer types: {types}")
    print(f"Depth range: {min(r['depth'] for r in data)}–{max(r['depth'] for r in data)}")

    # ===================================================================
    # 1. Raw statistics
    # ===================================================================
    print(f"\n{'='*70}")
    print("RAW k₀ BY TYPE")
    print(f"{'='*70}")
    by_type = defaultdict(list)
    for r in data:
        by_type[r['type']].append(r)

    for t in sorted(by_type, key=lambda x: np.mean([r['k0'] for r in by_type[x]])):
        k0s = [r['k0'] for r in by_type[t]]
        dims = [(r['d_in'], r['d_out']) for r in by_type[t]]
        geo = np.sqrt(dims[0][0] * dims[0][1])
        print(f"  {t:<12s}  n={len(k0s):>3d}  mean={np.mean(k0s):>7.0f}  "
              f"std={np.std(k0s):>6.0f}  dims=({dims[0][0]}×{dims[0][1]})  "
              f"γ=k₀/√(d₁d₂)={np.mean(k0s)/geo:.4f}")

    # ===================================================================
    # 2. Simple model: k₀ = γ_type · √(d_in·d_out)
    # ===================================================================
    print(f"\n{'='*70}")
    print("MODEL 1: k₀ = γ_type · √(d_in·d_out)")
    print(f"{'='*70}")
    gamma = {}
    for t in types:
        rs = by_type[t]
        geos = [np.sqrt(r['d_in'] * r['d_out']) for r in rs]
        k0s = [r['k0'] for r in rs]
        gamma[t] = np.mean([k/g for k, g in zip(k0s, geos)])

    preds = [gamma[r['type']] * np.sqrt(r['d_in'] * r['d_out']) for r in data]
    actuals = [r['k0'] for r in data]
    ss_res = sum((a - p) ** 2 for a, p in zip(actuals, preds))
    ss_tot = sum((a - np.mean(actuals)) ** 2 for a in actuals)
    r2 = 1 - ss_res / ss_tot
    mape = np.mean([abs(a - p) / a * 100 for a, p in zip(actuals, preds)])
    print(f"  R² = {r2:.4f}")
    print(f"  MAPE = {mape:.1f}%")
    for t in sorted(gamma, key=lambda x: gamma[x]):
        print(f"    {t:<12s}  γ = {gamma[t]:.5f}")

    # ===================================================================
    # 3. Log-linear regression (manual, no sklearn needed)
    # ===================================================================
    print(f"\n{'='*70}")
    print("MODEL 2: log(k₀) = β₀ + β₁·log(d_eff) + β₂·(d/L) + β₃·(d/L)² + type_offset")
    print(f"{'='*70}")

    # Build design matrix
    y = np.array([np.log(r['k0']) for r in data])
    n = len(data)

    # Features: intercept, log(sqrt(d_in*d_out)), depth/L, (depth/L)²
    X_base = np.column_stack([
        np.ones(n),
        np.array([np.log(np.sqrt(r['d_in'] * r['d_out'])) for r in data]),
        np.array([r['depth'] / L for r in data]),
        np.array([(r['depth'] / L) ** 2 for r in data]),
    ])

    # Type dummy variables (drop first for identifiability)
    type_to_idx = {t: i for i, t in enumerate(types)}
    type_dummies = np.zeros((n, len(types) - 1))
    for i, r in enumerate(data):
        idx = type_to_idx[r['type']]
        if idx > 0:
            type_dummies[i, idx - 1] = 1.0

    X = np.column_stack([X_base, type_dummies])

    # OLS: β = (X^T X)^{-1} X^T y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        print("  FAILED: singular matrix")
        return

    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Back-transform to k₀ space for interpretable error
    k0_pred = np.exp(y_pred)
    k0_actual = np.array([r['k0'] for r in data])
    mape = np.mean(np.abs(k0_actual - k0_pred) / k0_actual * 100)

    print(f"  R² (log space) = {r2:.4f}")
    print(f"  MAPE (k₀ space) = {mape:.1f}%")
    print(f"\n  Coefficients:")
    print(f"    β₀ (intercept)     = {beta[0]:+.4f}")
    print(f"    β₁ (log d_eff)     = {beta[1]:+.4f}  {'← √ scaling confirmed' if 0.8 < beta[1] < 1.2 else '← NOT √'}")
    print(f"    β₂ (depth/L)       = {beta[2]:+.4f}")
    print(f"    β₃ (depth/L)²      = {beta[3]:+.4f}")
    print(f"\n  Type offsets (vs {types[0]}):")
    for i, t in enumerate(types[1:]):
        print(f"    {t:<12s}  = {beta[4 + i]:+.4f}  (γ ratio = {np.exp(beta[4+i]):.3f})")

    # ===================================================================
    # 4. Depth profile per type
    # ===================================================================
    print(f"\n{'='*70}")
    print("DEPTH PROFILES (k₀ vs depth for each type)")
    print(f"{'='*70}")
    for t in sorted(by_type, key=lambda x: np.mean([r['k0'] for r in by_type[x]])):
        rs = sorted(by_type[t], key=lambda r: r['depth'])
        if len(rs) < 5:
            continue
        depths = [r['depth'] for r in rs]
        k0s = [r['k0'] for r in rs]
        # Linear fit within type
        p = np.polyfit(depths, k0s, 1)
        print(f"\n  {t} (n={len(rs)}):")
        print(f"    depth range: {min(depths)}–{max(depths)}")
        print(f"    k₀ range:    {min(k0s)}–{max(k0s)}")
        print(f"    slope:       {p[0]:+.1f} per layer")
        # Show every 8th layer
        for d, k in zip(depths[::max(1, len(depths)//8)], k0s[::max(1, len(k0s)//8)]):
            print(f"      d={d:>2d}  k₀={k:>5d}")

    # ===================================================================
    # 5. Residual analysis
    # ===================================================================
    print(f"\n{'='*70}")
    print("RESIDUALS (actual vs predicted k₀)")
    print(f"{'='*70}")
    residuals = k0_actual - k0_pred
    print(f"  Mean residual:     {np.mean(residuals):+.1f}")
    print(f"  Std residual:      {np.std(residuals):.1f}")
    print(f"  Max |residual|:    {np.max(np.abs(residuals)):.1f}")

    # Worst predictions
    worst_idx = np.argsort(np.abs(residuals))[-5:]
    print(f"\n  Worst 5 predictions:")
    for i in worst_idx:
        r = data[i]
        print(f"    depth={r['depth']:>2d} {r['type']:<12s} actual={r['k0']:>5d} "
              f"pred={k0_pred[i]:>7.0f} err={residuals[i]:+.0f}")


if __name__ == '__main__':
    main()
