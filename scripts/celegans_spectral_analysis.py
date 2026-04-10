"""
C. elegans connectome spectral analysis.

Tests whether biological neural circuits exhibit the same φ-based
spectral structure found in transformer weight matrices.

302 neurons, ~7000 chemical synapses, ~600 gap junctions.
Neuron types: sensory (routing/input), interneuron (integration),
motor (output/transformation).

The hypothesis: if φ-harmonic spectral structure is a property of
information processing itself (not just gradient descent), then:
  1. The connectome weight matrix should show bent power-law decay
  2. Different neuron types should have different spectral exponents
  3. Hub interneurons (the "attn_o" of biology — consensus/integration)
     might show a specific spectral signature

Data: Varshney et al. (2011), CSV from ivan-ea/celegans_connectome
"""

import json
import sys
from pathlib import Path
from io import StringIO

import numpy as np
from scipy.optimize import curve_fit

INV_PHI = 2.0 / (1.0 + np.sqrt(5))  # 0.6180339887...
PHI = (1.0 + np.sqrt(5)) / 2.0


def bent_power_law(k, A, k0, alpha):
    return A * (k + k0) ** (-alpha)


def fit_free_alpha(S):
    """Fit bent power law with free alpha."""
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
        A_fit, k0_fit, alpha_fit = popt
        s_pred = bent_power_law(k, *popt)
        ss_res = np.sum((s - s_pred) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {'A': float(A_fit), 'k0': float(k0_fit), 'alpha': float(alpha_fit), 'r2': float(r2)}
    except Exception:
        return None


def fit_fixed_phi(S):
    """Fit bent power law with alpha fixed at 1/phi."""
    S = S[S > 1e-10]
    n = len(S)
    if n < 4:
        return None

    k = np.arange(1, n + 1, dtype=np.float64)
    s = S.astype(np.float64)

    def model(k, A, k0):
        return A * (k + k0) ** (-INV_PHI)

    try:
        popt, _ = curve_fit(
            model, k, s,
            p0=[s[0], max(1.0, n * 0.1)],
            bounds=([0, 0], [s[0] * 100, n * 5]),
            maxfev=20000,
        )
        s_pred = model(k, *popt)
        ss_res = np.sum((s - s_pred) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {'A': float(popt[0]), 'k0': float(popt[1]), 'alpha': float(INV_PHI), 'r2': float(r2)}
    except Exception:
        return None


# C. elegans neuron type classification
# Based on WormAtlas canonical classification
# S = sensory, I = interneuron, M = motor
NEURON_TYPES = {
    # Sensory neurons
    'ADE': 'sensory', 'ADF': 'sensory', 'ADL': 'sensory', 'AFD': 'sensory',
    'ALA': 'sensory', 'ALM': 'sensory', 'ALN': 'sensory', 'AQR': 'sensory',
    'ASE': 'sensory', 'ASG': 'sensory', 'ASH': 'sensory', 'ASI': 'sensory',
    'ASJ': 'sensory', 'ASK': 'sensory', 'AVM': 'sensory', 'AWA': 'sensory',
    'AWB': 'sensory', 'AWC': 'sensory', 'BAG': 'sensory', 'CEP': 'sensory',
    'FLP': 'sensory', 'IL1': 'sensory', 'IL2': 'sensory', 'OLL': 'sensory',
    'OLQ': 'sensory', 'PDE': 'sensory', 'PHA': 'sensory', 'PHB': 'sensory',
    'PHC': 'sensory', 'PLM': 'sensory', 'PLN': 'sensory', 'PQR': 'sensory',
    'PVD': 'sensory', 'SDQ': 'sensory', 'URB': 'sensory', 'URX': 'sensory',
    'URY': 'sensory',
    # Interneurons
    'ADA': 'interneuron', 'AIA': 'interneuron', 'AIB': 'interneuron',
    'AIM': 'interneuron', 'AIN': 'interneuron', 'AIY': 'interneuron',
    'AIZ': 'interneuron', 'AVA': 'interneuron', 'AVB': 'interneuron',
    'AVD': 'interneuron', 'AVE': 'interneuron', 'AVF': 'interneuron',
    'AVG': 'interneuron', 'AVH': 'interneuron', 'AVJ': 'interneuron',
    'AVK': 'interneuron', 'AVL': 'interneuron', 'BDU': 'interneuron',
    'CAN': 'interneuron', 'DVA': 'interneuron', 'DVB': 'interneuron',
    'DVC': 'interneuron', 'LUA': 'interneuron', 'PVC': 'interneuron',
    'PVN': 'interneuron', 'PVP': 'interneuron', 'PVQ': 'interneuron',
    'PVR': 'interneuron', 'PVT': 'interneuron', 'PVW': 'interneuron',
    'RIA': 'interneuron', 'RIB': 'interneuron', 'RIC': 'interneuron',
    'RID': 'interneuron', 'RIF': 'interneuron', 'RIG': 'interneuron',
    'RIH': 'interneuron', 'RIM': 'interneuron', 'RIP': 'interneuron',
    'RIR': 'interneuron', 'RIS': 'interneuron', 'RIV': 'interneuron',
    # Command interneurons (hub/consensus — analogous to attn_o)
    # AVA, AVB, AVD, AVE, PVC are command interneurons that integrate
    # sensory input into forward/backward locomotion decisions
    # Motor neurons
    'AS': 'motor', 'DA': 'motor', 'DB': 'motor', 'DD': 'motor',
    'PDB': 'motor', 'VA': 'motor', 'VB': 'motor', 'VC': 'motor',
    'VD': 'motor', 'RMD': 'motor', 'RME': 'motor', 'RMF': 'motor',
    'RMG': 'motor', 'RMH': 'motor', 'SAA': 'motor', 'SAB': 'motor',
    'SIA': 'motor', 'SIB': 'motor', 'SMB': 'motor', 'SMD': 'motor',
    'URA': 'motor', 'HSN': 'motor',
}

# Command interneurons — the "consensus" neurons
COMMAND_INTERNEURONS = {'AVA', 'AVB', 'AVD', 'AVE', 'PVC'}


def classify_neuron(name):
    """Classify a neuron by stripping L/R/D/V suffix and looking up type."""
    # Strip left/right/dorsal/ventral suffixes
    base = name.rstrip('LRDV0123456789')
    if not base:
        base = name

    ntype = NEURON_TYPES.get(base)
    if ntype:
        is_command = base in COMMAND_INTERNEURONS
        if is_command:
            return 'command_interneuron'
        return ntype
    return 'unknown'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='celegans-spectral.json')
    parser.add_argument('--fetch', action='store_true', help='Download data from GitHub')
    parser.add_argument('--chem-csv', default=None, help='Local path to chem.csv')
    parser.add_argument('--gap-csv', default=None, help='Local path to gap.csv')
    parser.add_argument('--labels-csv', default=None, help='Local path to labels.csv')
    args = parser.parse_args()

    # Load data
    if args.fetch:
        print("Cloning C. elegans connectome from GitHub...")
        import subprocess, tempfile
        tmpdir = tempfile.mkdtemp()
        subprocess.run(['git', 'clone', '--depth=1',
                       'https://github.com/ivan-ea/celegans_connectome.git',
                       tmpdir + '/repo'], check=True, capture_output=True)
        args.chem_csv = tmpdir + '/repo/results/Chem_headless.csv'
        args.gap_csv = tmpdir + '/repo/results/Gap_headless.csv'
        args.labels_csv = tmpdir + '/repo/results/labels.csv'

    chem = np.genfromtxt(args.chem_csv, delimiter=',')
    gap = np.genfromtxt(args.gap_csv, delimiter=',')
    labels = [l.strip().strip('"') for l in open(args.labels_csv).read().strip().split('\n') if l.strip()]

    n = chem.shape[0]
    print(f"Loaded: {n} neurons, chem shape {chem.shape}, gap shape {gap.shape}")
    print(f"Labels: {len(labels)} (first 5: {labels[:5]})")

    # Classify neurons
    types = [classify_neuron(l) for l in labels[:n]]
    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nNeuron types:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # === FULL MATRIX ANALYSIS ===
    print("\n" + "=" * 70)
    print("FULL CONNECTOME SPECTRAL ANALYSIS")
    print("=" * 70)

    matrices = {
        'chemical': np.abs(chem),  # absolute weights (some are negative/inhibitory)
        'gap_junction': gap,
        'combined': np.abs(chem) + gap,
    }

    results = {}
    for mname, W in matrices.items():
        print(f"\n--- {mname} ({W.shape}) ---")
        print(f"  Non-zero: {np.count_nonzero(W)} / {W.size} ({100*np.count_nonzero(W)/W.size:.1f}%)")
        print(f"  Max weight: {W.max():.0f}, Mean non-zero: {W[W>0].mean():.2f}")

        S = np.linalg.svd(W, compute_uv=False)
        S = S[S > 1e-10]
        print(f"  Singular values: {len(S)} non-zero")
        print(f"  Top 5: {S[:5].round(2)}")

        free = fit_free_alpha(S)
        fixed = fit_fixed_phi(S)

        if free:
            print(f"  Free alpha: {free['alpha']:.4f} (k0={free['k0']:.1f}, R2={free['r2']:.4f})")
        if fixed:
            print(f"  Fixed 1/phi: R2={fixed['r2']:.4f}")
        if free and fixed:
            print(f"  Delta R2 (free vs fixed): {free['r2'] - fixed['r2']:+.4f}")

        results[mname] = {
            'shape': list(W.shape),
            'nnz': int(np.count_nonzero(W)),
            'n_sv': len(S),
            'top_sv': S[:10].tolist(),
            'free_fit': free,
            'fixed_fit': fixed,
        }

    # === TYPE-SPECIFIC SUBMATRICES ===
    print("\n" + "=" * 70)
    print("TYPE-SPECIFIC SPECTRAL ANALYSIS")
    print("=" * 70)
    print("(SVD of submatrices: connections FROM type-X neurons TO all neurons)")

    W = np.abs(chem) + gap  # combined
    type_results = {}

    for ntype in ['sensory', 'interneuron', 'command_interneuron', 'motor']:
        idx = [i for i, t in enumerate(types) if t == ntype]
        if len(idx) < 3:
            continue

        # Submatrix: rows = neurons of this type, cols = all neurons
        # This captures "what this type sends" — its output signature
        W_sub = W[idx, :]

        S = np.linalg.svd(W_sub, compute_uv=False)
        S = S[S > 1e-10]

        free = fit_free_alpha(S)
        fixed = fit_fixed_phi(S)

        print(f"\n  {ntype} (n={len(idx)}, sending to all)")
        print(f"    Shape: {W_sub.shape}, non-zero SV: {len(S)}")
        if free:
            print(f"    alpha = {free['alpha']:.4f}, k0 = {free['k0']:.1f}, R2 = {free['r2']:.4f}")
            # Check against known F/L fractions
            predicted_13 = INV_PHI ** (1/3)
            err_13 = abs(free['alpha'] - predicted_13) / predicted_13 * 100
            print(f"    vs (1/phi)^(1/3) = {predicted_13:.4f}: error = {err_13:.1f}%")

        type_results[ntype] = {
            'n_neurons': len(idx),
            'submatrix_shape': list(W_sub.shape),
            'n_sv': len(S),
            'free_fit': free,
            'fixed_fit': fixed,
        }

    # Also: receiving submatrices (connections TO type-X from all)
    print("\n  --- Receiving submatrices (all → type) ---")
    for ntype in ['sensory', 'interneuron', 'command_interneuron', 'motor']:
        idx = [i for i, t in enumerate(types) if t == ntype]
        if len(idx) < 3:
            continue

        W_sub = W[:, idx]
        S = np.linalg.svd(W_sub, compute_uv=False)
        S = S[S > 1e-10]

        free = fit_free_alpha(S)
        print(f"\n  {ntype} receiving (n={len(idx)})")
        print(f"    Shape: {W_sub.shape}, non-zero SV: {len(S)}")
        if free:
            print(f"    alpha = {free['alpha']:.4f}, k0 = {free['k0']:.1f}, R2 = {free['r2']:.4f}")

        type_results[f'{ntype}_receiving'] = {
            'n_neurons': len(idx),
            'submatrix_shape': list(W_sub.shape),
            'n_sv': len(S),
            'free_fit': free,
            'fixed_fit': fixed if fixed else None,
        }

    # === COMMAND INTERNEURON FOCUS ===
    print("\n" + "=" * 70)
    print("COMMAND INTERNEURONS (the 'attn_o' of C. elegans)")
    print("=" * 70)

    cmd_idx = [i for i, t in enumerate(types) if t == 'command_interneuron']
    cmd_names = [labels[i] for i in cmd_idx]
    print(f"  Command neurons: {cmd_names}")

    # Their full connectivity profile
    for ci in cmd_idx:
        name = labels[ci]
        out_weights = W[ci, :]
        in_weights = W[:, ci]
        n_out = np.count_nonzero(out_weights)
        n_in = np.count_nonzero(in_weights)
        print(f"  {name}: {n_in} inputs, {n_out} outputs, "
              f"total_in={in_weights.sum():.0f}, total_out={out_weights.sum():.0f}")

    # === ENERGY DISTRIBUTION ===
    print("\n" + "=" * 70)
    print("ENERGY DISTRIBUTION (variance explained by top modes)")
    print("=" * 70)

    W_full = np.abs(chem) + gap
    S_full = np.linalg.svd(W_full, compute_uv=False)
    total_energy = (S_full ** 2).sum()
    cumulative = np.cumsum(S_full ** 2) / total_energy

    print(f"  Mode 1:     {cumulative[0]*100:.1f}% variance")
    if len(cumulative) > 9:
        print(f"  Modes 1-10: {cumulative[9]*100:.1f}% variance")
    if len(cumulative) > 23:
        print(f"  Modes 1-24: {cumulative[23]*100:.1f}% variance")
    if len(cumulative) > 49:
        print(f"  Modes 1-50: {cumulative[49]*100:.1f}% variance")

    # Compare to transformer pattern: 76%, 88%, 98%, 100%
    print(f"\n  Transformer pattern: 76% (fundamental), 88% (+themes), 98% (+details), 100%")
    print(f"  C. elegans:          {cumulative[0]*100:.0f}% (mode 1), "
          f"{cumulative[min(9,len(cumulative)-1)]*100:.0f}% (top 10), "
          f"{cumulative[min(23,len(cumulative)-1)]*100:.0f}% (top 24), "
          f"{cumulative[min(49,len(cumulative)-1)]*100:.0f}% (top 50)")

    # Save
    output = {
        'organism': 'C. elegans',
        'n_neurons': n,
        'source': 'Varshney et al. 2011 via ivan-ea/celegans_connectome',
        'neuron_type_counts': type_counts,
        'full_matrix': results,
        'type_specific': type_results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {args.output}")


if __name__ == '__main__':
    main()
