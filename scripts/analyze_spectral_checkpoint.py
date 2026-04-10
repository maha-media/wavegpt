"""
Analyze a spectral fine-tuning checkpoint.

The checkpoint contains learned singular value spectra (S vectors)
for every SpectralLinear layer. We can:
1. Fit bent power laws to each spectrum
2. Classify by layer type and check α against F/L predictions
3. Compare to pre-trained baseline (Qwen α values)
4. Look for spectral drift — did fine-tuning change the exponents?
"""

import json
import sys
from math import sqrt
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

# Qwen pre-trained α values (from free_alpha_analysis)
PRETRAINED_ALPHAS = {
    'attn_q': 0.550,
    'mlp_up': 0.703,
    'mlp_down': 0.714,
    'mlp_gate': 0.763,
    'attn_v': 0.811,
    'attn_o': 0.853,
    'attn_k': 0.910,
}

# F/L predictions
FL_PREDICTIONS = {
    'attn_q': ('5/4', INV_PHI**(5/4)),
    'mlp_up': ('8/11', INV_PHI**(8/11)),
    'mlp_down': ('5/7', INV_PHI**(5/7)),
    'mlp_gate': ('4/7', INV_PHI**(4/7)),
    'attn_v': ('3/7', INV_PHI**(3/7)),
    'attn_o': ('1/3', INV_PHI**(1/3)),
    'attn_k': ('2/11', INV_PHI**(2/11)),
}


def bent_power_law(k, A, k0, alpha):
    return A * (k + k0) ** (-alpha)


def fit_spectrum(S):
    """Fit bent power law to a spectrum."""
    S = np.sort(S)[::-1]  # descending
    S = S[S > 1e-10]
    n = len(S)
    if n < 4:
        return None

    k = np.arange(1, n + 1, dtype=np.float64)
    s = S.astype(np.float64)

    try:
        popt, _ = curve_fit(
            bent_power_law, k, s,
            p0=[s[0], max(1.0, n * 0.1), INV_PHI],
            bounds=([0, 0, 0.01], [s[0] * 100, n * 5, 3.0]),
            maxfev=20000,
        )
        A, k0, alpha = popt
        s_pred = bent_power_law(k, *popt)
        ss_res = np.sum((s - s_pred) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {'A': float(A), 'k0': float(k0), 'alpha': float(alpha), 'r2': float(r2), 'n': n}
    except Exception:
        return None


def classify_layer(name):
    """Classify a layer name by type."""
    if 'out_proj' in name:
        return 'attn_o'
    elif 'in_proj_qkv' in name:
        return 'attn_qkv'
    elif 'in_proj_z' in name:
        return 'attn_z'
    elif 'in_proj_a' in name:
        return 'attn_a'
    elif 'in_proj_b' in name:
        return 'attn_b'
    elif 'q_proj' in name:
        return 'attn_q'
    elif 'k_proj' in name:
        return 'attn_k'
    elif 'v_proj' in name:
        return 'attn_v'
    elif 'o_proj' in name:
        return 'attn_o'
    elif 'gate_proj' in name:
        return 'mlp_gate'
    elif 'up_proj' in name:
        return 'mlp_up'
    elif 'down_proj' in name:
        return 'mlp_down'
    else:
        return 'other'


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/best_spectral.pt'

    print(f"Loading checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    print(f"  {len(sd)} spectra\n")

    # Fit each spectrum
    type_fits = defaultdict(list)
    all_fits = []

    for name, tensor in sd.items():
        S = tensor.numpy()
        ltype = classify_layer(name)
        fit = fit_spectrum(S)
        if fit and fit['r2'] > 0.7:
            fit['name'] = name
            fit['type'] = ltype
            type_fits[ltype].append(fit)
            all_fits.append(fit)

    print(f"Fitted {len(all_fits)} spectra (R² > 0.7)\n")

    # ═══════════════════════════════════════════════
    print("=" * 80)
    print("FINE-TUNED SPECTRAL EXPONENTS BY TYPE")
    print("=" * 80)
    print(f"\n{'Type':<15} {'n':>4} {'Mean α':>8} {'Std':>6} {'Pre-trained α':>14} {'Δα':>8} {'F/L pred':>10} {'Pred err':>9}")
    print("-" * 80)

    for ltype in sorted(type_fits.keys()):
        fits = type_fits[ltype]
        alphas = [f['alpha'] for f in fits]
        mean_a = np.mean(alphas)
        std_a = np.std(alphas)

        pretrained = PRETRAINED_ALPHAS.get(ltype, None)
        fl = FL_PREDICTIONS.get(ltype, None)

        delta = f"{mean_a - pretrained:+.4f}" if pretrained else "   -"
        pre_str = f"{pretrained:.4f}" if pretrained else "   -"
        fl_str = ""
        pred_err = ""
        if fl:
            fl_str = fl[0]
            pred_err = f"{abs(mean_a - fl[1])/fl[1]*100:.1f}%"

        print(f"{ltype:<15} {len(fits):>4} {mean_a:>8.4f} {std_a:>6.3f} {pre_str:>14} {delta:>8} {fl_str:>10} {pred_err:>9}")

    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SPECTRAL DRIFT: DID FINE-TUNING CHANGE THE EXPONENTS?")
    print("=" * 80)

    for ltype in sorted(type_fits.keys()):
        if ltype not in PRETRAINED_ALPHAS:
            continue
        fits = type_fits[ltype]
        alphas = np.array([f['alpha'] for f in fits])
        pretrained = PRETRAINED_ALPHAS[ltype]
        fl_pred = FL_PREDICTIONS[ltype][1]

        mean_a = np.mean(alphas)
        drift = mean_a - pretrained
        drift_pct = abs(drift) / pretrained * 100

        # Did it drift TOWARD or AWAY from the F/L prediction?
        pre_err = abs(pretrained - fl_pred)
        post_err = abs(mean_a - fl_pred)
        direction = "→ TOWARD F/L" if post_err < pre_err else "← AWAY from F/L"

        print(f"\n  {ltype}:")
        print(f"    Pre-trained: α = {pretrained:.4f}")
        print(f"    Fine-tuned:  α = {mean_a:.4f} (Δ = {drift:+.4f}, {drift_pct:.1f}%)")
        print(f"    F/L target:  α = {fl_pred:.4f}")
        print(f"    {direction} (pre-err={pre_err:.4f}, post-err={post_err:.4f})")

    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SPECTRUM SHAPE: EXAMPLE LAYERS")
    print("=" * 80)

    # Show a few example spectra
    for ltype in ['attn_o', 'mlp_down', 'mlp_gate']:
        if ltype in type_fits and type_fits[ltype]:
            fit = type_fits[ltype][0]
            name = fit['name']
            S = sd[name].numpy()
            S = np.sort(S)[::-1]
            n = len(S)
            print(f"\n  {ltype} ({name}):")
            print(f"    n={n}, α={fit['alpha']:.4f}, k₀={fit['k0']:.1f}, R²={fit['r2']:.4f}")
            print(f"    Top 10 SVs: {S[:10].round(3)}")
            print(f"    σ₁/σ₂ = {S[0]/S[1]:.3f}, σ₁/σ_n = {S[0]/S[-1]:.1f}")

            # Energy concentration
            energy = S ** 2
            total = energy.sum()
            cum = np.cumsum(energy) / total
            for thresh in [0.5, 0.9, 0.95]:
                k = np.searchsorted(cum, thresh) + 1
                print(f"    {thresh*100:.0f}% energy at k/n = {k/n:.4f} ({k}/{n})")

    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    all_alphas = [f['alpha'] for f in all_fits]
    all_r2 = [f['r2'] for f in all_fits]
    print(f"\n  Total spectra analyzed: {len(all_fits)}")
    print(f"  Mean α: {np.mean(all_alphas):.4f} ± {np.std(all_alphas):.4f}")
    print(f"  Mean R²: {np.mean(all_r2):.4f}")
    print(f"  Min R²: {np.min(all_r2):.4f}")

    # How many are within 2% of an F/L prediction?
    n_close = 0
    for f in all_fits:
        if f['type'] in FL_PREDICTIONS:
            _, pred = FL_PREDICTIONS[f['type']]
            err = abs(f['alpha'] - pred) / pred * 100
            if err < 2:
                n_close += 1
    n_testable = sum(1 for f in all_fits if f['type'] in FL_PREDICTIONS)
    print(f"  Within 2% of F/L prediction: {n_close}/{n_testable} ({n_close/max(n_testable,1)*100:.0f}%)")


if __name__ == '__main__':
    main()
