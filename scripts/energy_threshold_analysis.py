"""
Energy concentration analysis: do transformer weight matrices show
φ-power energy thresholds like C. elegans gap junctions?

C. elegans gap junctions showed:
  90% energy at k/n = 0.237 ≈ 1/φ³ (0.3% error)
  95% energy at k/n = 0.363 ≈ 1/φ² (4.9% error)
  99% energy at k/n = 0.604 ≈ 1/φ  (2.3% error)

This script reconstructs singular value spectra from fitted parameters
(A, k₀, α) and tests for the same pattern across transformer layers.
"""

import json
import sys
from math import sqrt, log
from collections import defaultdict

import numpy as np

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

# φ-power targets for energy concentration
PHI_TARGETS = {
    '1/φ⁴': INV_PHI**4,   # 0.1459
    '1/φ³': INV_PHI**3,   # 0.2361
    '1/φ²': INV_PHI**2,   # 0.3820
    '1/φ':  INV_PHI,      # 0.6180
}

# Standard energy thresholds to test
THRESHOLDS = [0.50, 0.75, 0.80, 0.90, 0.95, 0.99]


def reconstruct_spectrum(A, k0, alpha, n):
    """Reconstruct singular values from bent power law fit."""
    k = np.arange(1, n + 1, dtype=np.float64)
    return A * (k + k0) ** (-alpha)


def compute_energy_thresholds(S):
    """Compute k/n at which cumulative energy crosses thresholds."""
    energy = S ** 2
    total = energy.sum()
    if total == 0:
        return {}
    cumulative = np.cumsum(energy) / total
    n = len(S)

    results = {}
    for t in THRESHOLDS:
        k = np.searchsorted(cumulative, t) + 1
        k = min(k, n)
        results[t] = k / n
    return results


def find_best_phi_target(ratio):
    """Find the closest φ-power target for a k/n ratio."""
    best_name = None
    best_err = float('inf')
    for name, target in PHI_TARGETS.items():
        err = abs(ratio - target) / target * 100
        if err < best_err:
            best_err = err
            best_name = name
            best_target = target
    return best_name, best_target, best_err


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'runs/gemma4-free-alpha.json'
    with open(data_path) as f:
        data = json.load(f)

    layers = data['layers']
    model_name = data.get('model', 'unknown')
    print(f"Model: {model_name}")
    print(f"Layers: {len(layers)}")

    # Collect energy thresholds per layer type
    type_thresholds = defaultdict(lambda: defaultdict(list))
    all_thresholds = defaultdict(list)

    for layer in layers:
        ltype = layer['type']
        n = layer['min_dim']
        A = layer['free_A']
        k0 = layer['free_k0']
        alpha = layer['free_alpha']
        r2 = layer['free_r2']

        # Skip poor fits
        if r2 < 0.8 or n < 10:
            continue

        S = reconstruct_spectrum(A, k0, alpha, n)
        thresholds = compute_energy_thresholds(S)

        for t, ratio in thresholds.items():
            type_thresholds[ltype][t].append(ratio)
            all_thresholds[t].append(ratio)

    # ═══════════════════════════════════════════════
    # GLOBAL ANALYSIS (all layers combined)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("GLOBAL ENERGY CONCENTRATION (all layers)")
    print("=" * 80)
    print(f"\n  {'Threshold':<12} {'Mean k/n':<12} {'Std':<10} {'Best φ target':<12} {'Target':<10} {'Error':<8} {'n'}")
    print(f"  {'-'*75}")

    for t in THRESHOLDS:
        if t in all_thresholds:
            ratios = np.array(all_thresholds[t])
            mean_r = np.mean(ratios)
            std_r = np.std(ratios)
            name, target, err = find_best_phi_target(mean_r)
            marker = ' ◄◄' if err < 3 else (' ◄' if err < 8 else '')
            print(f"  {t:<12.2f} {mean_r:<12.4f} {std_r:<10.4f} {name:<12} {target:<10.4f} {err:<7.1f}% {len(ratios)}{marker}")

    # ═══════════════════════════════════════════════
    # PER-TYPE ANALYSIS
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PER-TYPE ENERGY CONCENTRATION")
    print("=" * 80)

    # Sort types for consistent output
    type_order = sorted(type_thresholds.keys())

    for ltype in type_order:
        thresholds = type_thresholds[ltype]
        n_layers = len(next(iter(thresholds.values())))
        print(f"\n  --- {ltype} (n={n_layers}) ---")
        print(f"  {'Threshold':<10} {'Mean k/n':<10} {'Std':<8} {'φ target':<10} {'Target':<8} {'Error':<8}")
        print(f"  {'-'*58}")

        for t in THRESHOLDS:
            if t in thresholds:
                ratios = np.array(thresholds[t])
                mean_r = np.mean(ratios)
                std_r = np.std(ratios)
                name, target, err = find_best_phi_target(mean_r)
                marker = ' ◄◄' if err < 3 else (' ◄' if err < 8 else '')
                print(f"  {t:<10.2f} {mean_r:<10.4f} {std_r:<8.4f} {name:<10} {target:<8.4f} {err:<7.1f}%{marker}")

    # ═══════════════════════════════════════════════
    # DIRECT COMPARISON: C. elegans vs Transformer
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("C. ELEGANS GAP JUNCTIONS vs TRANSFORMER WEIGHT MATRICES")
    print("=" * 80)

    celegans_gap = {
        0.90: 0.2367,
        0.95: 0.3633,
        0.99: 0.6041,
    }

    print(f"\n  {'Threshold':<12} {'C.elegans gap':<14} {'Transformer':<14} {'φ target':<10} {'CE err':<8} {'TF err':<8}")
    print(f"  {'-'*68}")
    for t in [0.90, 0.95, 0.99]:
        ce = celegans_gap.get(t, None)
        tf_ratios = all_thresholds.get(t, [])
        tf = np.mean(tf_ratios) if tf_ratios else None
        name, target, _ = find_best_phi_target(ce) if ce else ('', 0, 0)

        ce_err = abs(ce - target) / target * 100 if ce else 0
        tf_err = abs(tf - target) / target * 100 if tf else 0

        ce_str = f"{ce:.4f}" if ce else "N/A"
        tf_str = f"{tf:.4f}" if tf else "N/A"
        print(f"  {t:<12.2f} {ce_str:<14} {tf_str:<14} {name:<10} {ce_err:<7.1f}% {tf_err:<7.1f}%")

    # ═══════════════════════════════════════════════
    # α-DEPENDENT ANALYSIS
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ENERGY THRESHOLDS vs SPECTRAL EXPONENT α")
    print("=" * 80)
    print("  (Does the energy concentration pattern depend on the power-law steepness?)")

    # Bin layers by alpha
    alpha_bins = [(0.3, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.2)]
    for lo, hi in alpha_bins:
        matching = []
        for layer in layers:
            alpha = layer['free_alpha']
            r2 = layer['free_r2']
            n = layer['min_dim']
            if lo <= alpha < hi and r2 >= 0.8 and n >= 10:
                S = reconstruct_spectrum(layer['free_A'], layer['free_k0'], alpha, n)
                matching.append(compute_energy_thresholds(S))

        if not matching:
            continue

        print(f"\n  α ∈ [{lo}, {hi}) — {len(matching)} layers")
        print(f"  {'Threshold':<10} {'Mean k/n':<10} {'φ target':<10} {'Error':<8}")
        print(f"  {'-'*40}")
        for t in THRESHOLDS:
            ratios = [m[t] for m in matching if t in m]
            if ratios:
                mean_r = np.mean(ratios)
                name, target, err = find_best_phi_target(mean_r)
                marker = ' ◄◄' if err < 3 else (' ◄' if err < 8 else '')
                print(f"  {t:<10.2f} {mean_r:<10.4f} {name:<10} {err:<7.1f}%{marker}")

    # ═══════════════════════════════════════════════
    # THEORETICAL PREDICTION
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("THEORETICAL: ENERGY THRESHOLDS FOR PURE POWER LAWS")
    print("=" * 80)
    print("  For σ_k = k^(-α), what energy thresholds emerge?")
    print("  (Closed-form for comparison — no k₀ offset)")

    for alpha in [0.5, INV_PHI, 0.7, 0.85, 1.0, 1.5]:
        n = 5000  # large n limit
        k = np.arange(1, n + 1, dtype=np.float64)
        S = k ** (-alpha)
        thresholds = compute_energy_thresholds(S)
        name_90, tgt_90, err_90 = find_best_phi_target(thresholds.get(0.90, 0))
        name_95, tgt_95, err_95 = find_best_phi_target(thresholds.get(0.95, 0))
        name_99, tgt_99, err_99 = find_best_phi_target(thresholds.get(0.99, 0))
        phi_label = " (= 1/φ)" if abs(alpha - INV_PHI) < 0.001 else ""
        print(f"\n  α = {alpha:.4f}{phi_label}")
        print(f"    90% at k/n = {thresholds.get(0.90, 0):.4f} → {name_90} err={err_90:.1f}%")
        print(f"    95% at k/n = {thresholds.get(0.95, 0):.4f} → {name_95} err={err_95:.1f}%")
        print(f"    99% at k/n = {thresholds.get(0.99, 0):.4f} → {name_99} err={err_99:.1f}%")

    # Save
    output = {
        'model': model_name,
        'global_thresholds': {str(t): float(np.mean(all_thresholds[t])) for t in THRESHOLDS if t in all_thresholds},
        'per_type_thresholds': {
            ltype: {str(t): float(np.mean(thresholds[t])) for t in THRESHOLDS if t in thresholds}
            for ltype, thresholds in type_thresholds.items()
        },
    }
    out_path = data_path.replace('.json', '-energy.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == '__main__':
    main()
