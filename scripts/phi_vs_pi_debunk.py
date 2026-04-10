"""
Debunk check: can π produce the same spectral patterns as φ?

If (1/π)^p with "nice" rational fractions fits the observed alphas
as well as (1/φ)^(F/L), then our finding is numerology.

Tests:
1. Best rational fraction p for each observed α under both bases
2. Are the π-fractions as "clean" (small numerator/denominator)?
3. Does the same fraction persist across models? (attn_o = 1/3 on φ)
4. Error comparison: φ vs π vs e vs √2 vs arbitrary bases
5. C. elegans: does the biological data also prefer φ?
"""

import json
import sys
from math import sqrt, log, pi, e
from itertools import product

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

# Bases to test
BASES = {
    'φ': INV_PHI,           # 0.6180 — our claim
    'π': 1/pi,              # 0.3183
    'e': 1/e,               # 0.3679
    '√2': 1/sqrt(2),        # 0.7071
    '√3': 1/sqrt(3),        # 0.5774
    '2': 1/2,               # 0.5000
    '3': 1/3,               # 0.3333
    'plastic': 1/1.3247,    # plastic ratio ≈ 1.3247
}

# Generate "clean" fractions: a/b where a,b ∈ {1,...,20}, gcd(a,b)=1
from math import gcd
def clean_fractions(max_val=20):
    fracs = set()
    for a in range(1, max_val + 1):
        for b in range(1, max_val + 1):
            if gcd(a, b) == 1:
                fracs.add((a, b, a/b))
    return sorted(fracs, key=lambda x: x[2])

# Fibonacci/Lucas fractions specifically
FIB = [1, 1, 2, 3, 5, 8, 13, 21]
LUC = [1, 3, 4, 7, 11, 18, 29, 47]

def fl_fractions():
    fracs = set()
    for f in FIB:
        for l in LUC:
            fracs.add((f, l, f/l, f'F/L={f}/{l}'))
    for l1 in LUC:
        for l2 in LUC:
            if l1 != l2:
                fracs.add((l1, l2, l1/l2, f'L/L={l1}/{l2}'))
    return sorted(fracs, key=lambda x: x[2])


def best_fraction_match(alpha, base, fracs, max_err=5.0):
    """Find best rational fraction p such that base^p ≈ alpha."""
    if base <= 0 or base >= 1:
        return None

    # Solve: base^p = alpha → p = log(alpha)/log(base)
    if alpha <= 0:
        return None
    exact_p = log(alpha) / log(base)

    best = None
    best_err = float('inf')
    for frac in fracs:
        a, b, p_val = frac[0], frac[1], frac[2]
        # Test both base^p and base^(-p)
        for p in [p_val, -p_val]:
            predicted = base ** p
            if predicted > 0:
                err = abs(alpha - predicted) / alpha * 100
                complexity = a + b  # lower = "cleaner" fraction
                if err < best_err:
                    best_err = err
                    sign = '+' if p >= 0 else '-'
                    best = {
                        'p': p,
                        'frac': f"{a}/{b}" if p >= 0 else f"-{a}/{b}",
                        'predicted': predicted,
                        'error': err,
                        'complexity': complexity,
                    }
    return best


def main():
    # Observed alphas from transformers (Qwen, using the best-measured values)
    transformer_alphas = {
        'attn_q': 0.550,
        'mlp_up': 0.703,
        'mlp_down': 0.714,
        'mlp_gate': 0.763,
        'attn_v': 0.811,
        'attn_o': 0.853,
        'attn_k': 0.910,
    }

    # Cross-model attn_o values
    attn_o_cross = {
        'Qwen': 0.853,
        'Mistral': 0.845,
        'Gemma4': 0.8524,
    }

    # C. elegans alphas
    celegans_alphas = {
        'cmd_inter_recv': 0.6841,
        'motor_recv': 0.6999,
        'sensory_send': 0.8091,
        'gap_junction': 0.9200,
        'cmd_inter_send': 1.1774,
    }

    all_fracs = clean_fractions(20)
    fl_fracs = fl_fractions()

    # ═══════════════════════════════════════════════
    print("=" * 90)
    print("TEST 1: BEST FRACTION FIT — φ vs π vs e vs √2 vs others")
    print("=" * 90)
    print("For each observed α, find the simplest fraction p where base^p ≈ α")
    print()

    for dataset_name, alphas in [("TRANSFORMER (Qwen)", transformer_alphas),
                                   ("C. ELEGANS", celegans_alphas)]:
        print(f"\n  --- {dataset_name} ---")
        print(f"  {'Type':<18} {'α':>6} ", end='')
        for bname in BASES:
            print(f"| {bname:>8} err  frac  ", end='')
        print()
        print(f"  {'-'*18} {'-'*6} " + "| " + "-"*18 + " " * (len(BASES)-1) * 20)

        base_errors = {bname: [] for bname in BASES}

        for ltype, alpha in sorted(alphas.items(), key=lambda x: x[1]):
            print(f"  {ltype:<18} {alpha:>6.4f} ", end='')
            for bname, base in BASES.items():
                match = best_fraction_match(alpha, base, all_fracs)
                if match and match['error'] < 5:
                    print(f"| {match['error']:>6.2f}% {match['frac']:>6s} ", end='')
                    base_errors[bname].append(match['error'])
                else:
                    # Try with base > 1 (inverse)
                    match_inv = best_fraction_match(alpha, 1/base if base < 1 else base, all_fracs)
                    if match_inv and match_inv['error'] < 5:
                        print(f"| {match_inv['error']:>6.2f}% {match_inv['frac']:>6s} ", end='')
                        base_errors[bname].append(match_inv['error'])
                    else:
                        print(f"|    >5%        ", end='')
            print()

        print(f"\n  Mean errors (< 5% matches only):")
        for bname in BASES:
            errs = base_errors[bname]
            if errs:
                print(f"    {bname:<8}: {sum(errs)/len(errs):.3f}% ({len(errs)}/{len(alphas)} matched)")
            else:
                print(f"    {bname:<8}: no matches < 5%")

    # ═══════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 2: FRACTION CLEANLINESS — How 'nice' are the required fractions?")
    print("=" * 90)
    print("Lower complexity (a+b) = cleaner fraction. F/L fractions are privileged for φ.")
    print()

    for bname, base in [('φ', INV_PHI), ('π', 1/pi), ('e', 1/e), ('√2', 1/sqrt(2))]:
        print(f"\n  --- Base = 1/{bname} = {base:.4f} ---")
        print(f"  {'Type':<18} {'α':>6} {'Best p':>10} {'Predicted':>10} {'Error':>8} {'Complexity':>11}")
        total_complexity = 0
        total_err = 0
        n = 0
        for ltype, alpha in sorted(transformer_alphas.items(), key=lambda x: x[1]):
            match = best_fraction_match(alpha, base, all_fracs)
            if match:
                print(f"  {ltype:<18} {alpha:>6.3f} {match['frac']:>10} {match['predicted']:>10.4f} {match['error']:>7.2f}% {match['complexity']:>8d}")
                total_complexity += match['complexity']
                total_err += match['error']
                n += 1
        if n:
            print(f"  {'TOTAL':<18} {'':>6} {'':>10} {'':>10} {total_err/n:>7.2f}% {total_complexity/n:>8.1f}")

    # ═══════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 3: CROSS-MODEL UNIVERSALITY — Does the SAME fraction work across models?")
    print("=" * 90)
    print("The φ claim: attn_o = (1/φ)^(1/3) on ALL three models.")
    print("Can any other base match this with a SINGLE clean fraction?")
    print()

    for bname, base in BASES.items():
        # Find best fraction for each model's attn_o
        fracs_found = {}
        for model, alpha in attn_o_cross.items():
            match = best_fraction_match(alpha, base, all_fracs)
            if match:
                fracs_found[model] = match

        if len(fracs_found) == 3:
            # Check if same fraction
            frac_strs = [m['frac'] for m in fracs_found.values()]
            same = len(set(frac_strs)) == 1
            errors = [m['error'] for m in fracs_found.values()]
            mean_err = sum(errors) / len(errors)

            label = "✓ SAME" if same else "✗ DIFFERENT"
            print(f"  Base 1/{bname:<6}: {label} fraction across models", end='')
            if same:
                print(f" → p = {frac_strs[0]}, mean error = {mean_err:.2f}%")
            else:
                print(f" → {', '.join(f'{m}: p={f}' for m, f in zip(fracs_found.keys(), frac_strs))}")
                print(f"              mean error = {mean_err:.2f}%")

    # ═══════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 4: STATISTICAL SIGNIFICANCE — Random base comparison")
    print("=" * 90)
    print("Generate 1000 random bases and check: how often does a random base")
    print("match all 7 transformer alphas within 1% using fractions a/b ≤ 20?")
    print()

    import random
    random.seed(42)

    n_trials = 10000
    n_all_match_1pct = {bname: 0 for bname in BASES}
    n_all_match_1pct['random'] = 0
    n_all_match_2pct = {bname: 0 for bname in BASES}
    n_all_match_2pct['random'] = 0

    # First check our known bases
    for bname, base in BASES.items():
        all_under_1 = True
        all_under_2 = True
        for alpha in transformer_alphas.values():
            match = best_fraction_match(alpha, base, all_fracs)
            if not match or match['error'] > 1.0:
                all_under_1 = False
            if not match or match['error'] > 2.0:
                all_under_2 = False
        if all_under_1:
            n_all_match_1pct[bname] = 1
        if all_under_2:
            n_all_match_2pct[bname] = 1

    # Random bases
    random_1pct = 0
    random_2pct = 0
    for _ in range(n_trials):
        base = random.uniform(0.1, 0.95)
        all_under_1 = True
        all_under_2 = True
        for alpha in transformer_alphas.values():
            match = best_fraction_match(alpha, base, all_fracs)
            if not match or match['error'] > 1.0:
                all_under_1 = False
            if not match or match['error'] > 2.0:
                all_under_2 = False
        if all_under_1:
            random_1pct += 1
        if all_under_2:
            random_2pct += 1

    print(f"  Results (fractions a/b with a,b ≤ 20):")
    print(f"  {'Base':<12} {'All 7 within 1%':<20} {'All 7 within 2%':<20}")
    print(f"  {'-'*52}")
    for bname in BASES:
        y1 = "YES" if n_all_match_1pct[bname] else "no"
        y2 = "YES" if n_all_match_2pct[bname] else "no"
        print(f"  {bname:<12} {y1:<20} {y2:<20}")
    print(f"  {'random':<12} {random_1pct}/{n_trials} ({random_1pct/n_trials*100:.1f}%)   {random_2pct}/{n_trials} ({random_2pct/n_trials*100:.1f}%)")

    # ═══════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 5: THE REAL DISCRIMINATOR — Fibonacci/Lucas fractions ONLY")
    print("=" * 90)
    print("φ's claim is specific: the fractions are F(a)/L(b), not arbitrary a/b.")
    print("Test: using ONLY F/L fractions, how well does each base fit?")
    print()

    for bname, base in [('φ', INV_PHI), ('π', 1/pi), ('e', 1/e), ('√2', 1/sqrt(2)), ('2', 0.5)]:
        total_err = 0
        n_matched = 0
        print(f"  Base = 1/{bname} = {base:.4f}, F/L fractions only:")
        for ltype, alpha in sorted(transformer_alphas.items(), key=lambda x: x[1]):
            # Only use F/L fractions
            best_err = float('inf')
            best_match = None
            for f, l, p_val, label in fl_fracs:
                for p in [p_val, -p_val]:
                    predicted = base ** p
                    err = abs(alpha - predicted) / alpha * 100
                    if err < best_err:
                        best_err = err
                        best_match = (label, predicted, err, p)

            if best_match and best_match[2] < 10:
                print(f"    {ltype:<15} α={alpha:.3f} → {best_match[0]:<15} pred={best_match[1]:.4f} err={best_match[2]:.2f}%")
                total_err += best_match[2]
                n_matched += 1
            else:
                print(f"    {ltype:<15} α={alpha:.3f} → best err = {best_err:.1f}% (>10%)")

        if n_matched > 0:
            print(f"    Mean error: {total_err/n_matched:.2f}% ({n_matched}/{len(transformer_alphas)} matched <10%)")
        print()


if __name__ == '__main__':
    main()
