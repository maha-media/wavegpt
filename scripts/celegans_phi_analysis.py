"""
C. elegans connectome: search for φ^(F/L) spectral structure.

Takes the free-alpha fits from celegans-spectral.json and systematically
tests whether each spectral exponent α can be expressed as:

    α = φ^p   or equivalently   α = (1/φ)^(-p)

where p = F(a)/L(b) is a Fibonacci/Lucas fraction — the same structure
found in transformer weight matrices (where α = (1/φ)^(F/L) < 1).

The biological system may operate in the "inverse regime": steeper spectral
decay (α > 1) means information is concentrated in fewer modes, as expected
from evolution optimizing wiring efficiency over millions of generations
vs gradient descent distributing representations over training.
"""

import json
import sys
from math import log, sqrt, exp
from itertools import product

PHI = (1 + sqrt(5)) / 2      # 1.6180339887...
INV_PHI = 1 / PHI            # 0.6180339887...
LN_PHI = log(PHI)            # 0.4812118250...

# Fibonacci numbers F(1)..F(10)
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
FIB_IDX = list(range(1, 11))

# Lucas numbers L(1)..L(10)
LUC = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123]
LUC_IDX = list(range(1, 11))


def generate_fl_fractions():
    """Generate all unique F(a)/L(b) fractions with labels."""
    fracs = {}
    # F(a)/L(b)
    for i, fi in enumerate(FIB):
        for j, lj in enumerate(LUC):
            p = fi / lj
            label = f"F({FIB_IDX[i]})/L({LUC_IDX[j]}) = {fi}/{lj}"
            if p not in fracs:
                fracs[p] = label
    # Also L(a)/L(b) — used in transformer mlp_gate (4/7 = L(3)/L(4))
    for i, li in enumerate(LUC):
        for j, lj in enumerate(LUC):
            if i != j:
                p = li / lj
                label = f"L({LUC_IDX[i]})/L({LUC_IDX[j]}) = {li}/{lj}"
                if p not in fracs:
                    fracs[p] = label
    # F(a)/F(b)
    for i, fi in enumerate(FIB):
        for j, fj in enumerate(FIB):
            if i != j and fj > 0:
                p = fi / fj
                label = f"F({FIB_IDX[i]})/F({FIB_IDX[j]}) = {fi}/{fj}"
                if p not in fracs:
                    fracs[p] = label
    return fracs


def find_best_match(alpha, fracs, max_error_pct=5.0):
    """
    Find best F/L fraction match for observed alpha.

    Tests:  α = φ^p  (for α > 1)  and  α = (1/φ)^p  (for α < 1)
    where p is an F/L fraction.

    Returns list of (error_pct, predicted_alpha, p_value, p_label, formula).
    """
    matches = []
    for p, label in fracs.items():
        # Test α = φ^p
        predicted = PHI ** p
        if predicted > 0:
            err = abs(alpha - predicted) / alpha * 100
            if err < max_error_pct:
                formula = f"φ^({label})" if p >= 0 else f"φ^({label})"
                matches.append((err, predicted, p, label, f"φ^p"))

        # Test α = (1/φ)^p
        predicted = INV_PHI ** p
        if predicted > 0:
            err = abs(alpha - predicted) / alpha * 100
            if err < max_error_pct:
                matches.append((err, predicted, p, label, f"(1/φ)^p"))

    matches.sort(key=lambda x: x[0])
    return matches


def solve_p(alpha):
    """Solve α = φ^p for p."""
    if alpha <= 0:
        return None
    return log(alpha) / LN_PHI


def main():
    # Load results
    results_path = sys.argv[1] if len(sys.argv) > 1 else "runs/celegans-spectral.json"
    with open(results_path) as f:
        data = json.load(f)

    fracs = generate_fl_fractions()
    print(f"Generated {len(fracs)} unique F/L fractions\n")

    # Collect all observed alphas
    observed = {}

    # Full matrices
    for mname, mdata in data['full_matrix'].items():
        if mdata.get('free_fit'):
            observed[f"full:{mname}"] = mdata['free_fit']['alpha']

    # Type-specific
    for tname, tdata in data['type_specific'].items():
        if tdata.get('free_fit'):
            observed[f"type:{tname}"] = tdata['free_fit']['alpha']

    # Print raw exponents first
    print("=" * 80)
    print("OBSERVED SPECTRAL EXPONENTS")
    print("=" * 80)
    print(f"{'Matrix':<40} {'α':>8} {'p = ln(α)/ln(φ)':>16} {'Regime':<12}")
    print("-" * 80)
    for name, alpha in sorted(observed.items(), key=lambda x: x[1]):
        p = solve_p(alpha)
        regime = "α < 1 (spread)" if alpha < 1 else "α > 1 (concentrated)"
        print(f"{name:<40} {alpha:>8.4f} {p:>16.4f} {regime}")

    # Compare with transformer results
    print("\n" + "=" * 80)
    print("TRANSFORMER COMPARISON (for reference)")
    print("=" * 80)
    transformer_alphas = {
        'attn_o (Qwen)': 0.853,
        'attn_o (Mistral)': 0.845,
        'attn_o (Gemma4)': 0.8524,
        'attn_v (Qwen)': 0.811,
        'attn_q (Qwen)': 0.550,
        'attn_k (Qwen)': 0.910,
        'mlp_up (Qwen)': 0.703,
        'mlp_down (Qwen)': 0.714,
        'mlp_gate (Qwen)': 0.763,
    }
    for name, alpha in sorted(transformer_alphas.items(), key=lambda x: x[1]):
        p = solve_p(alpha)
        print(f"  {name:<30} α = {alpha:.4f}  p = {p:.4f}")

    # Now find best F/L matches
    print("\n" + "=" * 80)
    print("BEST φ^(F/L) MATCHES")
    print("=" * 80)
    print("Testing: α = φ^p and α = (1/φ)^p where p = F(a)/L(b), L(a)/L(b), or F(a)/F(b)")
    print()

    summary = []
    for name, alpha in sorted(observed.items(), key=lambda x: x[1]):
        p_exact = solve_p(alpha)
        matches = find_best_match(alpha, fracs, max_error_pct=3.0)

        print(f"\n  {name}")
        print(f"  α = {alpha:.4f}, exact p = {p_exact:.4f}")

        if matches:
            best = matches[0]
            err, pred, p_val, p_label, formula = best
            print(f"  BEST: α ≈ {formula} where p = {p_label} = {p_val:.4f}")
            print(f"        predicted = {pred:.4f}, error = {err:.2f}%")
            summary.append((name, alpha, p_label, p_val, pred, err, formula))

            # Show top 3
            if len(matches) > 1:
                print(f"  Other close matches:")
                for m in matches[1:4]:
                    print(f"    {m[4]} p={m[3]} ({m[2]:.4f}) → {m[1]:.4f}, err={m[0]:.2f}%")
        else:
            print(f"  NO MATCH within 3%")
            # Try wider search
            wider = find_best_match(alpha, fracs, max_error_pct=10.0)
            if wider:
                best = wider[0]
                print(f"  Nearest (>3%): {best[4]} p={best[3]} ({best[2]:.4f}) → {best[1]:.4f}, err={best[0]:.2f}%")
                summary.append((name, alpha, best[3], best[2], best[1], best[0], best[4]))

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: C. ELEGANS φ^(F/L) SPECTRAL STRUCTURE")
    print("=" * 80)
    print(f"\n{'Matrix':<40} {'α':>7} {'Fraction':<25} {'Predicted':>9} {'Error':>7}")
    print("-" * 90)
    for name, alpha, p_label, p_val, pred, err, formula in sorted(summary, key=lambda x: x[1]):
        print(f"{name:<40} {alpha:>7.4f} {p_label:<25} {pred:>9.4f} {err:>6.2f}%")

    mean_err = sum(x[5] for x in summary) / len(summary) if summary else 0
    print(f"\n  Mean error: {mean_err:.2f}%")

    # === KEY BIOLOGICAL PARALLELS ===
    print("\n" + "=" * 80)
    print("BIOLOGICAL PARALLELS TO TRANSFORMER ARCHITECTURE")
    print("=" * 80)
    parallels = [
        ("command_interneuron sending", "attn_o (consensus/integration)", "1/3"),
        ("sensory sending", "attn_v (value/input routing)", "3/7"),
        ("motor receiving", "mlp layers (transformation)", "3/4"),
    ]
    for bio, transformer, expected_frac in parallels:
        bio_key = f"type:{bio}"
        if bio_key in observed:
            bio_alpha = observed[bio_key]
            bio_p = solve_p(bio_alpha)
            print(f"\n  {bio}")
            print(f"    Biological α = {bio_alpha:.4f} (p = {bio_p:.4f})")
            print(f"    Transformer analog: {transformer} (fraction {expected_frac})")

    # Save results
    output = {
        'organism': 'C. elegans',
        'analysis': 'phi_FL_fraction_matching',
        'observed_alphas': observed,
        'matches': [
            {
                'matrix': name,
                'alpha': alpha,
                'fraction_label': p_label,
                'fraction_value': p_val,
                'predicted_alpha': pred,
                'error_pct': err,
                'formula': formula,
            }
            for name, alpha, p_label, p_val, pred, err, formula in summary
        ],
        'mean_error_pct': mean_err,
    }
    out_path = results_path.replace('.json', '-phi-analysis.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == '__main__':
    main()
