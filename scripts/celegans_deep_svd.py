"""
Deep SVD analysis of C. elegans connectome.

Goes beyond singular value decay to analyze:
1. Consecutive SV ratios: σ_k / σ_{k+1} — convergence behavior
2. Energy concentration: cumulative variance thresholds
3. U-matrix clustering: do neurons of the same type cluster in singular vector space?
4. Angular spacing in U: angles between consecutive left singular vectors
5. Mode-type alignment: which neuron types dominate which singular modes?

Looking for φ-related structure beyond the power-law exponents.
"""

import json
import sys
from math import sqrt, log, pi, acos
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI
GOLDEN_ANGLE_DEG = 360 / (PHI ** 2)  # ~137.507°
GOLDEN_ANGLE_RAD = 2 * pi / (PHI ** 2)


# Neuron classification (same as celegans_spectral_analysis.py)
NEURON_TYPES = {
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
    'AS': 'motor', 'DA': 'motor', 'DB': 'motor', 'DD': 'motor',
    'PDB': 'motor', 'VA': 'motor', 'VB': 'motor', 'VC': 'motor',
    'VD': 'motor', 'RMD': 'motor', 'RME': 'motor', 'RMF': 'motor',
    'RMG': 'motor', 'RMH': 'motor', 'SAA': 'motor', 'SAB': 'motor',
    'SIA': 'motor', 'SIB': 'motor', 'SMB': 'motor', 'SMD': 'motor',
    'URA': 'motor', 'HSN': 'motor',
}
COMMAND_INTERNEURONS = {'AVA', 'AVB', 'AVD', 'AVE', 'PVC'}


def classify_neuron(name):
    base = name.rstrip('LRDV0123456789')
    if not base:
        base = name
    ntype = NEURON_TYPES.get(base)
    if ntype:
        if base in COMMAND_INTERNEURONS:
            return 'command_interneuron'
        return ntype
    return 'unknown'


def analyze_sv_ratios(S, label):
    """Analyze consecutive singular value ratios σ_k / σ_{k+1}."""
    S = S[S > 1e-10]
    n = len(S)
    if n < 5:
        return

    ratios = S[:-1] / S[1:]

    print(f"\n  {'='*60}")
    print(f"  CONSECUTIVE SV RATIOS: {label}")
    print(f"  {'='*60}")
    print(f"  n = {n} singular values")
    print("  First 20 ratios (σ_k / σ_{k+1}):")
    for i in range(min(20, len(ratios))):
        r = ratios[i]
        # Check proximity to φ, 1/φ, φ², 1
        errs = {
            'φ': abs(r - PHI) / PHI * 100,
            '1/φ': abs(r - INV_PHI) / INV_PHI * 100,
            'φ²': abs(r - PHI**2) / PHI**2 * 100,
            '1': abs(r - 1.0) * 100,
        }
        closest = min(errs, key=errs.get)
        print(f"    σ_{i+1}/σ_{i+2} = {r:.4f}  (closest: {closest}, err={errs[closest]:.1f}%)")

    # Look at ratio convergence in tail
    if n > 30:
        tail_ratios = ratios[-20:]
        mean_tail = np.mean(tail_ratios)
        std_tail = np.std(tail_ratios)
        print(f"\n  Tail ratios (last 20): mean={mean_tail:.4f}, std={std_tail:.4f}")

        # Test: does the mean tail ratio relate to φ?
        for target_name, target in [('1/φ', INV_PHI), ('φ', PHI), ('1', 1.0),
                                     ('φ^(1/3)', PHI**(1/3)), ('(1/φ)^(1/3)', INV_PHI**(1/3))]:
            err = abs(mean_tail - target) / target * 100
            if err < 10:
                print(f"    → Near {target_name} = {target:.4f} (err={err:.1f}%)")

    # Overall distribution
    print(f"\n  Ratio statistics:")
    print(f"    Mean: {np.mean(ratios):.4f}")
    print(f"    Median: {np.median(ratios):.4f}")
    print(f"    Std: {np.std(ratios):.4f}")

    # Histogram of ratios
    bins = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0, 3.0]
    hist, _ = np.histogram(ratios, bins=bins)
    print(f"  Distribution:")
    for i in range(len(hist)):
        bar = '#' * hist[i]
        print(f"    [{bins[i]:.2f}-{bins[i+1]:.2f}): {hist[i]:3d} {bar}")

    return ratios


def analyze_energy_thresholds(S, label):
    """Analyze cumulative energy and where key thresholds fall."""
    S = S[S > 1e-10]
    n = len(S)
    energy = S ** 2
    total = energy.sum()
    cumulative = np.cumsum(energy) / total

    print(f"\n  {'='*60}")
    print(f"  ENERGY CONCENTRATION: {label}")
    print(f"  {'='*60}")

    # Key thresholds
    thresholds = [0.5, 0.618, 0.75, 0.80, 0.90, 0.95, 0.99]
    print(f"  {'Threshold':<12} {'Rank k':<10} {'k/n':<10} {'k/n vs φ targets'}")
    print(f"  {'-'*55}")
    for t in thresholds:
        k = np.searchsorted(cumulative, t) + 1
        ratio = k / n
        # Check ratio against φ-related values
        phi_targets = {
            '1/φ²': 1/PHI**2,    # 0.382
            '1/φ': INV_PHI,       # 0.618
            '1/φ³': 1/PHI**3,    # 0.236
            '2/φ²': 2/PHI**2,    # 0.764
        }
        matches = []
        for name, val in phi_targets.items():
            err = abs(ratio - val) / val * 100
            if err < 15:
                matches.append(f"{name}={val:.3f} ({err:.1f}%)")
        match_str = ', '.join(matches) if matches else '-'
        print(f"  {t:<12.3f} {k:<10d} {ratio:<10.4f} {match_str}")

    # The "knee" — where does the biggest drop in marginal energy happen?
    marginal = energy / total
    # Find the mode where marginal contribution drops below 1/n (uniform baseline)
    uniform_baseline = 1.0 / n
    knee_candidates = np.where(marginal < uniform_baseline)[0]
    if len(knee_candidates) > 0:
        knee = knee_candidates[0] + 1
        knee_energy = cumulative[knee - 1]
        print(f"\n  Knee point (marginal < 1/n): mode {knee}, cumulative energy = {knee_energy:.4f}")
        print(f"    knee/n = {knee/n:.4f}")

    # φ-partition test: does mode floor(n/φ) capture exactly 1/φ of energy?
    phi_mode = int(n / PHI)
    if phi_mode < n:
        phi_energy = cumulative[phi_mode - 1]
        print(f"\n  φ-partition test:")
        print(f"    Mode k = floor(n/φ) = {phi_mode}")
        print(f"    Cumulative energy at k: {phi_energy:.4f}")
        print(f"    vs 1/φ = {INV_PHI:.4f}: error = {abs(phi_energy - INV_PHI)/INV_PHI*100:.1f}%")
        print(f"    vs 1/φ² = {INV_PHI**2:.4f}: error = {abs(phi_energy - INV_PHI**2)/INV_PHI**2*100:.1f}%")

    return cumulative


def analyze_u_clustering(U, types, labels, label):
    """Analyze whether neurons of the same type cluster in U-space."""
    print(f"\n  {'='*60}")
    print(f"  U-MATRIX CLUSTERING: {label}")
    print(f"  {'='*60}")

    n = U.shape[0]
    # Use top-k singular vectors for embedding
    for k in [3, 5, 10, 20]:
        if k > U.shape[1]:
            continue
        U_k = U[:, :k]  # n x k embedding

        # Compute pairwise distances
        dists = squareform(pdist(U_k, metric='cosine'))

        # For each type pair, compute mean intra-type vs inter-type distance
        type_set = sorted(set(types))
        print(f"\n  Top-{k} singular vectors:")
        print(f"    {'Type Pair':<45} {'Mean Dist':>10} {'vs Overall':>12}")

        overall_mean = dists[np.triu_indices(n, k=1)].mean()

        for t in type_set:
            idx = [i for i, tt in enumerate(types) if tt == t]
            if len(idx) < 3:
                continue
            # Intra-type distance
            intra_dists = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    intra_dists.append(dists[idx[i], idx[j]])
            if intra_dists:
                mean_intra = np.mean(intra_dists)
                ratio = mean_intra / overall_mean
                marker = '◄ CLUSTERED' if ratio < 0.8 else ('◄ DISPERSED' if ratio > 1.2 else '')
                print(f"    {t + ' (intra)':<45} {mean_intra:>10.4f} {ratio:>10.2f}x  {marker}")

        # Inter-type distances
        for i, t1 in enumerate(type_set):
            for j, t2 in enumerate(type_set):
                if j <= i:
                    continue
                idx1 = [ii for ii, tt in enumerate(types) if tt == t1]
                idx2 = [ii for ii, tt in enumerate(types) if tt == t2]
                if len(idx1) < 3 or len(idx2) < 3:
                    continue
                inter_dists = [dists[a][b] for a in idx1 for b in idx2]
                mean_inter = np.mean(inter_dists)
                ratio = mean_inter / overall_mean
                if ratio > 1.15 or ratio < 0.85:
                    marker = '◄ SEPARATED' if ratio > 1.15 else '◄ CLOSE'
                    print(f"    {t1 + ' ↔ ' + t2:<45} {mean_inter:>10.4f} {ratio:>10.2f}x  {marker}")


def analyze_mode_type_alignment(U, types, S, label):
    """Which neuron types dominate which SVD modes?"""
    print(f"\n  {'='*60}")
    print(f"  MODE-TYPE ALIGNMENT: {label}")
    print(f"  {'='*60}")
    print(f"  (Which neuron types have strongest loading on each mode)")

    type_set = sorted(set(t for t in types if t != 'unknown'))
    n_modes = min(20, U.shape[1])

    print(f"\n  {'Mode':<6} {'σ':>8} {'Dominant Type':<25} {'Loading':>8} {'2nd Type':<25} {'Loading':>8}")
    print(f"  {'-'*85}")

    for m in range(n_modes):
        u_col = U[:, m]
        # Compute mean absolute loading per type
        type_loadings = {}
        for t in type_set:
            idx = [i for i, tt in enumerate(types) if tt == t]
            if idx:
                type_loadings[t] = np.mean(np.abs(u_col[idx]))

        sorted_types = sorted(type_loadings.items(), key=lambda x: -x[1])
        if len(sorted_types) >= 2:
            t1, l1 = sorted_types[0]
            t2, l2 = sorted_types[1]
            sv = S[m] if m < len(S) else 0
            print(f"  {m+1:<6} {sv:>8.2f} {t1:<25} {l1:>8.4f} {t2:<25} {l2:>8.4f}")


def analyze_angular_spacing(U, label):
    """Analyze angles between consecutive left singular vectors."""
    print(f"\n  {'='*60}")
    print(f"  ANGULAR SPACING IN U: {label}")
    print(f"  {'='*60}")

    n_modes = min(50, U.shape[1])
    angles = []
    for i in range(n_modes - 1):
        u1 = U[:, i]
        u2 = U[:, i + 1]
        # Angle between consecutive singular vectors
        cos_theta = np.clip(np.dot(u1, u2), -1, 1)
        theta_deg = np.degrees(acos(abs(cos_theta)))
        angles.append(theta_deg)

    angles = np.array(angles)
    print(f"  Angles between consecutive singular vectors (degrees):")
    print(f"  First 20:")
    for i, a in enumerate(angles[:20]):
        note = ''
        if abs(a - GOLDEN_ANGLE_DEG) / GOLDEN_ANGLE_DEG < 0.05:
            note = ' ◄ GOLDEN ANGLE!'
        elif abs(a - 90) < 5:
            note = ' (≈ orthogonal)'
        print(f"    θ_{i+1},{i+2} = {a:.2f}°{note}")

    print(f"\n  Statistics:")
    print(f"    Mean: {np.mean(angles):.2f}°")
    print(f"    Median: {np.median(angles):.2f}°")
    print(f"    Std: {np.std(angles):.2f}°")
    print(f"    Golden angle = {GOLDEN_ANGLE_DEG:.2f}°")
    print(f"    Mean vs golden: error = {abs(np.mean(angles) - GOLDEN_ANGLE_DEG)/GOLDEN_ANGLE_DEG*100:.1f}%")

    # Distribution relative to golden angle
    n_near_golden = np.sum(np.abs(angles - GOLDEN_ANGLE_DEG) < 10)
    n_near_90 = np.sum(np.abs(angles - 90) < 5)
    print(f"    Within 10° of golden angle: {n_near_golden}/{len(angles)}")
    print(f"    Within 5° of 90° (orthogonal): {n_near_90}/{len(angles)}")

    return angles


def main():
    # Load connectome data
    import subprocess, tempfile
    tmpdir = tempfile.mkdtemp()

    # Check if already cloned
    repo_path = Path('/tmp/celegans_connectome')
    if not repo_path.exists():
        print("Cloning C. elegans connectome...")
        subprocess.run(['git', 'clone', '--depth=1',
                       'https://github.com/ivan-ea/celegans_connectome.git',
                       str(repo_path)], check=True, capture_output=True)

    chem = np.genfromtxt(repo_path / 'results' / 'Chem_headless.csv', delimiter=',')
    gap = np.genfromtxt(repo_path / 'results' / 'Gap_headless.csv', delimiter=',')
    labels_raw = open(repo_path / 'results' / 'labels.csv').read().strip().split('\n')
    labels = [l.strip().strip('"') for l in labels_raw if l.strip()]

    n = chem.shape[0]
    types = [classify_neuron(l) for l in labels[:n]]
    W = np.abs(chem) + gap  # combined connectome

    print(f"Loaded: {n} neurons, {np.count_nonzero(W)} non-zero connections")
    print(f"Types: {dict(sorted([(t, types.count(t)) for t in set(types)]))}")

    # Full SVD (need U and V this time)
    print("\nComputing full SVD...")
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    V = Vt.T
    print(f"  U: {U.shape}, S: {S.shape}, V: {V.shape}")

    # ═══════════════════════════════════════════════
    # 1. CONSECUTIVE SV RATIOS
    # ═══════════════════════════════════════════════
    ratios = analyze_sv_ratios(S, "Combined connectome")

    # Also for chemical-only and gap-only
    S_chem = np.linalg.svd(np.abs(chem), compute_uv=False)
    S_gap = np.linalg.svd(gap, compute_uv=False)
    analyze_sv_ratios(S_chem, "Chemical synapses")
    analyze_sv_ratios(S_gap, "Gap junctions")

    # ═══════════════════════════════════════════════
    # 2. ENERGY CONCENTRATION
    # ═══════════════════════════════════════════════
    analyze_energy_thresholds(S, "Combined connectome")
    analyze_energy_thresholds(S_chem, "Chemical synapses")
    analyze_energy_thresholds(S_gap, "Gap junctions")

    # ═══════════════════════════════════════════════
    # 3. U-MATRIX CLUSTERING
    # ═══════════════════════════════════════════════
    analyze_u_clustering(U, types, labels[:n], "Combined connectome (rows=sending)")

    # Also: receiving perspective (V matrix)
    analyze_u_clustering(V, types, labels[:n], "Combined connectome (cols=receiving)")

    # ═══════════════════════════════════════════════
    # 4. MODE-TYPE ALIGNMENT
    # ═══════════════════════════════════════════════
    analyze_mode_type_alignment(U, types, S, "U (sending)")
    analyze_mode_type_alignment(V, types, S, "V (receiving)")

    # ═══════════════════════════════════════════════
    # 5. ANGULAR SPACING
    # ═══════════════════════════════════════════════
    angles_u = analyze_angular_spacing(U, "U (left singular vectors)")
    angles_v = analyze_angular_spacing(V, "V (right singular vectors)")

    # ═══════════════════════════════════════════════
    # 6. BONUS: SV RATIO SEQUENCE φ-CONVERGENCE TEST
    # ═══════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  RATIO-OF-RATIOS (second-order structure)")
    print(f"  {'='*60}")
    if ratios is not None and len(ratios) > 5:
        ratio_of_ratios = ratios[:-1] / ratios[1:]
        print(f"  r_k = (σ_k/σ_{'{k+1}'}) / (σ_{'{k+1}'}/σ_{'{k+2}'})")
        print(f"  First 15:")
        for i in range(min(15, len(ratio_of_ratios))):
            rr = ratio_of_ratios[i]
            err_phi = abs(rr - PHI) / PHI * 100
            err_1 = abs(rr - 1.0) * 100
            note = ''
            if err_phi < 5:
                note = f' ◄ near φ ({err_phi:.1f}%)'
            elif err_1 < 5:
                note = f' ◄ near 1 ({err_1:.1f}%)'
            print(f"    r_{i+1} = {rr:.4f}{note}")
        print(f"  Mean: {np.mean(ratio_of_ratios):.4f}")
        print(f"  vs φ = {PHI:.4f}: error = {abs(np.mean(ratio_of_ratios) - PHI)/PHI*100:.1f}%")

    # Save results
    output = {
        'sv_ratios_combined': ratios.tolist() if ratios is not None else None,
        'angles_u': angles_u.tolist() if angles_u is not None else None,
        'angles_v': angles_v.tolist() if angles_v is not None else None,
    }
    out_path = 'runs/celegans-deep-svd.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == '__main__':
    main()
