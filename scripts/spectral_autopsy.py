"""
Spectral autopsy: load a trained model checkpoint, SVD every linear layer,
fit the power law W = σ₁ · Σ k^{-α} · u_k · v_k^T, report how close α is to 1/φ.

Usage:
    python scripts/spectral_autopsy.py --checkpoint path/to/best.pt
    python scripts/spectral_autopsy.py --hf-model gpt2
"""
import argparse
import json
import sys
import math

import torch
import numpy as np

PHI = (1 + 5**0.5) / 2
INV_PHI = 1 / PHI


def autopsy_state_dict(state_dict: dict) -> list[dict]:
    """SVD every 2D weight matrix, fit power law, report."""
    results = []
    for name, W in sorted(state_dict.items()):
        if W.ndim != 2:
            continue
        if W.shape[0] < 4 or W.shape[1] < 4:
            continue
        if 'mask' in name or 'wpe' in name:
            continue

        W = W.float()
        out_dim, in_dim = W.shape
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S = S.numpy()

        # Fit power law on top 90% of modes (tail is noise)
        n_fit = max(int(len(S) * 0.9), 4)
        log_k = np.log(np.arange(1, n_fit + 1))
        log_s = np.log(S[:n_fit] + 1e-10)
        coeffs = np.polyfit(log_k, log_s, 1)
        alpha = float(-coeffs[0])
        sigma1_fit = float(np.exp(coeffs[1]))

        # R² of fit
        predicted = coeffs[0] * log_k + coeffs[1]
        ss_res = np.sum((log_s - predicted) ** 2)
        ss_tot = np.sum((log_s - log_s.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Energy at various ranks
        total_energy = (S ** 2).sum()

        def energy_at(r):
            return float((S[:r] ** 2).sum() / total_energy)

        deviation = abs(alpha - INV_PHI)

        # Classify layer type
        layer_type = 'unknown'
        if 'c_attn' in name or 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            layer_type = 'attention'
        elif 'c_proj' in name or 'o_proj' in name:
            layer_type = 'projection'
        elif 'c_fc' in name or 'up_proj' in name or 'gate_proj' in name:
            layer_type = 'mlp_up'
        elif 'mlp' in name and 'proj' in name:
            layer_type = 'mlp_down'
        elif 'lm_head' in name or 'wte' in name or 'embed' in name:
            layer_type = 'embedding'

        results.append({
            'name': name,
            'shape': f"{out_dim}×{in_dim}",
            'layer_type': layer_type,
            'sigma1_fit': round(sigma1_fit, 4),
            'sigma1_actual': round(float(S[0]), 4),
            'alpha': round(alpha, 4),
            'deviation_from_inv_phi': round(deviation, 4),
            'r2': round(r2, 4),
            'energy_r32': round(energy_at(32), 4) if len(S) >= 32 else None,
            'energy_r64': round(energy_at(64), 4) if len(S) >= 64 else None,
            'energy_r128': round(energy_at(128), 4) if len(S) >= 128 else None,
            'energy_r256': round(energy_at(256), 4) if len(S) >= 256 else None,
        })

    return results


def print_report(results: list[dict]):
    """Pretty-print the autopsy report."""
    alphas = [r['alpha'] for r in results]
    r2s = [r['r2'] for r in results]
    devs = [r['deviation_from_inv_phi'] for r in results]

    print(f"\n{'=' * 90}")
    print(f"  SPECTRAL AUTOPSY — {len(results)} weight matrices")
    print(f"{'=' * 90}\n")
    print(f"  1/φ = {INV_PHI:.10f}\n")
    print(f"  {'Layer':<50s} {'Shape':>10s} {'Type':>10s} {'α':>7s} {'Δ(1/φ)':>7s} {'R²':>6s} {'σ₁':>8s}")
    print(f"  {'-' * 50} {'-' * 10} {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 8}")

    for r in results:
        marker = '✓' if r['deviation_from_inv_phi'] < 0.1 else ' '
        print(
            f"  {r['name']:<50s} {r['shape']:>10s} {r['layer_type']:>10s} "
            f"{r['alpha']:>7.4f} {r['deviation_from_inv_phi']:>7.4f} "
            f"{r['r2']:>6.3f} {r['sigma1_actual']:>8.2f} {marker}"
        )

    print(f"\n  {'─' * 60}")
    print(f"  Summary:")
    print(f"    Mean α:          {np.mean(alphas):>8.4f}  (1/φ = {INV_PHI:.4f})")
    print(f"    Median α:        {np.median(alphas):>8.4f}")
    print(f"    Std α:           {np.std(alphas):>8.4f}")
    print(f"    Mean R²:         {np.mean(r2s):>8.4f}")
    print(f"    Mean |α - 1/φ|:  {np.mean(devs):>8.4f}")
    print(f"    Layers ≈ 1/φ:    {sum(1 for d in devs if d < 0.1)}/{len(devs)}")

    # By layer type
    types = set(r['layer_type'] for r in results)
    if len(types) > 1:
        print(f"\n  By layer type:")
        for t in sorted(types):
            t_alphas = [r['alpha'] for r in results if r['layer_type'] == t]
            if t_alphas:
                print(f"    {t:<15s}: mean α = {np.mean(t_alphas):.4f} ± {np.std(t_alphas):.4f} (n={len(t_alphas)})")


def main():
    parser = argparse.ArgumentParser(description="Spectral autopsy of trained weights")
    parser.add_argument('--checkpoint', type=str, help='Path to .pt checkpoint')
    parser.add_argument('--hf-model', type=str, help='HuggingFace model name')
    parser.add_argument('--output', type=str, help='Save JSON report to file')
    args = parser.parse_args()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if isinstance(ckpt, dict):
            for key in ['model_state_dict', 'model', 'state_dict']:
                if key in ckpt:
                    ckpt = ckpt[key]
                    break
        state_dict = ckpt
    elif args.hf_model:
        print(f"Loading HuggingFace model: {args.hf_model}")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.hf_model)
        state_dict = model.state_dict()
    else:
        print("Provide --checkpoint or --hf-model")
        sys.exit(1)

    results = autopsy_state_dict(state_dict)
    print_report(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {args.output}")


if __name__ == '__main__':
    main()
