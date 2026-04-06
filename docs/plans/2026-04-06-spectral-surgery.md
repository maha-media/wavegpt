# Spectral Surgery Implementation Plan

**Goal:** Decompose trained models into spectral form, verify the power-law equation, enable fine-tuning via spectral amplitudes.
**Architecture:** SpectralLinear (frozen U,V + learnable σ) wraps any nn.Linear. SpectralSurgery walks a model, replaces layers, preserves outputs.
**Design Doc:** `docs/plans/2026-04-06-spectral-surgery-design.md`
**Estimated Tasks:** 12 tasks
**Complexity:** Large

---

## Batch 1: SpectralLinear + Decomposition (foundation)

### Task 1: SpectralLinear — frozen geometry, learnable spectrum

**Files:**
- Create: `wavegpt/spectral_linear.py`
- Test: `tests/test_spectral_linear.py`

**Step 1: Write failing tests**
```python
# tests/test_spectral_linear.py
import torch
from wavegpt.spectral_linear import SpectralLinear

def test_from_linear_output_match():
    """Decompose nn.Linear, output should match within tolerance."""
    linear = torch.nn.Linear(64, 128, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=32)
    x = torch.randn(2, 10, 64)
    y_orig = linear(x)
    y_spec = spec(x)
    # Rank-32 truncation loses some energy, but should be close
    assert y_spec.shape == y_orig.shape
    # Relative error < 10% (rank truncation)
    rel_err = (y_orig - y_spec).norm() / y_orig.norm()
    assert rel_err < 0.1, f"Relative error {rel_err:.4f} too high"

def test_full_rank_exact_match():
    """Full rank decomposition should be lossless."""
    linear = torch.nn.Linear(32, 48, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=32)  # full rank = min(in,out)
    x = torch.randn(2, 5, 32)
    y_orig = linear(x)
    y_spec = spec(x)
    torch.testing.assert_close(y_orig, y_spec, atol=1e-5, rtol=1e-4)

def test_sigma1_mode_learnable():
    """In sigma1 mode, only one param per layer is learnable."""
    linear = torch.nn.Linear(64, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='sigma1')
    learnable = [p for p in spec.parameters() if p.requires_grad]
    assert len(learnable) == 1  # just sigma1
    assert learnable[0].numel() == 1

def test_per_mode_learnable():
    """In per_mode, one amplitude per singular value is learnable."""
    linear = torch.nn.Linear(64, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='per_mode')
    learnable = [p for p in spec.parameters() if p.requires_grad]
    assert len(learnable) == 1  # spectrum vector
    assert learnable[0].numel() == 16  # one per mode

def test_spectrum_report():
    """Report fitted alpha, sigma1, energy captured."""
    linear = torch.nn.Linear(64, 128, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=32)
    report = spec.spectral_report()
    assert 'alpha' in report
    assert 'sigma1' in report
    assert 'energy_captured' in report
    assert 0 < report['energy_captured'] <= 1.0
```

**Step 2: Verify they fail**
Run: `pytest tests/test_spectral_linear.py -v`
Expected: FAIL — "No module named 'wavegpt.spectral_linear'"

**Step 3: Implement**
```python
# wavegpt/spectral_linear.py
"""
SpectralLinear — post-training spectral decomposition of nn.Linear.

Unlike HarmonicLinear (which trains from scratch), SpectralLinear
decomposes a TRAINED weight matrix into (U, S, V) and freezes
the geometry (U, V). Only the spectral amplitudes are learnable.

Three modes:
  - sigma1: W = σ₁ · Σ_k k^{-α_fit} · u_k · v_k^T  (1 learnable param)
  - per_mode: W = Σ_k s_k · u_k · v_k^T               (rank learnable params)
  - spectral_lora: per_mode + small trainable corrections to U, V

The equation: gradient descent converges to W = σ₁ · Σ k^{-1/φ} · u_k · v_k^T.
We observe the converged structure, then fine-tune the amplitudes.
"""
import math
import torch
import torch.nn as nn
import numpy as np


PHI = (1 + 5**0.5) / 2
INV_PHI = 1 / PHI


class SpectralLinear(nn.Module):
    def __init__(self, U, S, V, mode='per_mode', alpha_fit=None, bias=None):
        super().__init__()
        self.mode = mode
        self.rank = S.shape[0]
        self.out_dim = U.shape[0]
        self.in_dim = V.shape[0]

        # Geometry: FROZEN
        self.register_buffer('U', U)   # (out_dim, rank)
        self.register_buffer('V', V)   # (in_dim, rank)

        # Fitted alpha from power-law regression
        self.alpha_fit = alpha_fit or INV_PHI

        if mode == 'sigma1':
            # One scalar — reconstruct spectrum as σ₁ · k^{-α_fit}
            self.sigma1 = nn.Parameter(torch.tensor(S[0].item()))
            self.register_buffer('k_indices',
                torch.arange(1, self.rank + 1, dtype=torch.float))
        elif mode == 'per_mode':
            # One amplitude per mode — free spectral shape
            self.spectrum = nn.Parameter(S.clone())
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Bias (frozen if present)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def get_spectrum(self):
        if self.mode == 'sigma1':
            return self.sigma1 * self.k_indices.pow(-self.alpha_fit)
        else:
            return self.spectrum

    def forward(self, x):
        spectrum = self.get_spectrum()
        # W = U · diag(s) · V^T → x @ W^T = x @ V · diag(s) · U^T
        # More efficient: (x @ V) * s @ U^T
        xV = x @ self.V                    # (..., rank)
        xVs = xV * spectrum.unsqueeze(0)   # broadcast spectrum
        out = xVs @ self.U.t()             # (..., out_dim)
        if self.bias is not None:
            out = out + self.bias
        return out

    def spectral_report(self):
        with torch.no_grad():
            s = self.get_spectrum()
            s_np = s.cpu().numpy()
            log_k = np.log(np.arange(1, self.rank + 1))
            log_s = np.log(np.abs(s_np) + 1e-10)
            coeffs = np.polyfit(log_k, log_s, 1)
            alpha = float(-coeffs[0])
            sigma1 = float(np.exp(coeffs[1]))

            # Energy captured (vs full matrix if we had it)
            total_energy = (s ** 2).sum().item()

        return {
            'alpha': alpha,
            'sigma1': sigma1,
            'rank': self.rank,
            'energy_captured': 1.0,  # We don't have the tail; set by from_linear
            'mode': self.mode,
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
        }

    @classmethod
    def from_linear(cls, linear, rank=None, mode='per_mode'):
        W = linear.weight.data.float()  # (out, in)
        out_dim, in_dim = W.shape
        max_rank = min(out_dim, in_dim)

        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
        total_energy = (S_full ** 2).sum()

        if rank is None:
            energy_ratio = torch.cumsum(S_full ** 2, 0) / total_energy
            rank = int((energy_ratio < 0.95).sum().item()) + 1
            rank = max(rank, 2)
        rank = min(rank, max_rank)

        U = U_full[:, :rank].contiguous()
        S = S_full[:rank].contiguous()
        V = Vh_full[:rank, :].t().contiguous()  # (in_dim, rank)

        energy_captured = (S ** 2).sum() / total_energy

        # Fit alpha
        s_np = S.numpy()
        log_k = np.log(np.arange(1, rank + 1))
        log_s = np.log(s_np + 1e-10)
        coeffs = np.polyfit(log_k, log_s, 1)
        alpha_fit = float(-coeffs[0])

        bias = linear.bias.data.clone() if linear.bias is not None else None

        layer = cls(U, S, V, mode=mode, alpha_fit=alpha_fit, bias=bias)
        layer._energy_captured = energy_captured.item()

        # Patch the report
        orig_report = layer.spectral_report
        def patched_report():
            r = orig_report()
            r['energy_captured'] = layer._energy_captured
            return r
        layer.spectral_report = patched_report

        return layer
```

**Step 4: Verify tests pass**
Run: `pytest tests/test_spectral_linear.py -v`
Expected: 5 passed

**Step 5: Commit**
```bash
git add -A && git commit -m "feat: SpectralLinear — post-training spectral decomposition"
```

---

### Task 2: SpectralSurgery — walk model, replace nn.Linear → SpectralLinear

**Files:**
- Create: `wavegpt/spectral_surgery.py`
- Test: `tests/test_spectral_surgery.py`

**Step 1: Write failing tests**
```python
# tests/test_spectral_surgery.py
import torch
import torch.nn as nn
from wavegpt.spectral_surgery import spectral_decompose, spectral_report

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 64, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 16, bias=False)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def test_decompose_replaces_linears():
    """All nn.Linear layers should be replaced with SpectralLinear."""
    model = TinyModel()
    decomposed = spectral_decompose(model, rank=16)
    from wavegpt.spectral_linear import SpectralLinear
    assert isinstance(decomposed.linear1, SpectralLinear)
    assert isinstance(decomposed.linear2, SpectralLinear)

def test_decompose_output_close():
    """Decomposed model output should approximate original."""
    model = TinyModel()
    x = torch.randn(2, 5, 32)
    y_orig = model(x)
    decomposed = spectral_decompose(model, rank=16)
    y_dec = decomposed(x)
    rel_err = (y_orig - y_dec).norm() / y_orig.norm()
    assert rel_err < 0.15

def test_spectral_report_all_layers():
    """Report should have one entry per linear layer."""
    model = TinyModel()
    decomposed = spectral_decompose(model, rank=16)
    report = spectral_report(decomposed)
    assert len(report) == 2  # linear1 and linear2
    for name, info in report.items():
        assert 'alpha' in info
        assert 'sigma1' in info

def test_skip_pattern():
    """Layers matching skip pattern should not be decomposed."""
    model = TinyModel()
    decomposed = spectral_decompose(model, rank=16, skip_patterns=['linear1'])
    from wavegpt.spectral_linear import SpectralLinear
    assert isinstance(decomposed.linear1, nn.Linear)  # skipped
    assert isinstance(decomposed.linear2, SpectralLinear)  # decomposed
```

**Step 2: Verify they fail**
Run: `pytest tests/test_spectral_surgery.py -v`
Expected: FAIL — "No module named 'wavegpt.spectral_surgery'"

**Step 3: Implement**
```python
# wavegpt/spectral_surgery.py
"""
Spectral Surgery — decompose any model's nn.Linear layers into SpectralLinear.

Usage:
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    decomposed = spectral_decompose(model, rank=64)
    report = spectral_report(decomposed)
"""
import re
import torch.nn as nn
from .spectral_linear import SpectralLinear


def spectral_decompose(
    model: nn.Module,
    rank: int | None = None,
    mode: str = 'per_mode',
    skip_patterns: list[str] | None = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with SpectralLinear.

    Args:
        model: Any nn.Module
        rank: SVD truncation rank (None = auto 95% energy)
        mode: 'sigma1' or 'per_mode'
        skip_patterns: list of regex patterns for layer names to skip
    """
    skip_patterns = skip_patterns or []

    def should_skip(name):
        return any(re.search(p, name) for p in skip_patterns)

    def replace_linears(module, prefix=''):
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if child is None:
                continue
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name

            if isinstance(child, nn.Linear):
                if should_skip(full_name):
                    continue
                spec = SpectralLinear.from_linear(child, rank=rank, mode=mode)
                setattr(module, attr_name, spec)

        # Also handle named children in ModuleList/ModuleDict/Sequential
        for child_name, child in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if should_skip(full_name):
                    continue
                spec = SpectralLinear.from_linear(child, rank=rank, mode=mode)
                # Need to set via parent
                _set_submodule(module, child_name, spec)
            else:
                replace_linears(child, full_name)

    replace_linears(model)
    return model


def _set_submodule(parent, name, new_module):
    """Set a child module by name, handling ModuleList indices."""
    if hasattr(parent, name):
        setattr(parent, name, new_module)
    elif name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(name)] = new_module


def spectral_report(model: nn.Module) -> dict:
    """Generate spectral report for all SpectralLinear layers."""
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, SpectralLinear):
            report[name] = module.spectral_report()
    return report
```

**Step 4: Verify tests pass**
Run: `pytest tests/test_spectral_surgery.py -v`
Expected: 4 passed

**Step 5: Commit**
```bash
git add -A && git commit -m "feat: spectral_surgery — decompose any model to spectral form"
```

---

### Task 3: Autopsy script — verify power law on G2-A checkpoint

**Files:**
- Create: `scripts/spectral_autopsy.py`

**Step 1: No test needed (script, not library)**

**Step 2: Implement**
```python
# scripts/spectral_autopsy.py
"""
Spectral autopsy: load a trained model checkpoint, SVD every linear layer,
fit the power law W = σ₁ · Σ k^{-α} · u_k · v_k^T, report how close α is to 1/φ.

Usage:
    python scripts/spectral_autopsy.py --checkpoint path/to/best.pt
    python scripts/spectral_autopsy.py --hf-model gpt2
"""
import argparse, json, sys, math
import torch
import numpy as np

PHI = (1 + 5**0.5) / 2
INV_PHI = 1 / PHI

def autopsy_state_dict(state_dict):
    results = []
    for name, W in sorted(state_dict.items()):
        if W.ndim != 2:
            continue
        if W.shape[0] < 4 or W.shape[1] < 4:
            continue
        # Skip masks, embeddings by name
        if 'mask' in name or 'wpe' in name:
            continue

        W = W.float()
        out_dim, in_dim = W.shape
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S = S.numpy()

        # Fit power law on top modes (skip last 10% which are noise)
        n_fit = max(int(len(S) * 0.9), 4)
        log_k = np.log(np.arange(1, n_fit + 1))
        log_s = np.log(S[:n_fit] + 1e-10)
        coeffs = np.polyfit(log_k, log_s, 1)
        alpha = float(-coeffs[0])
        sigma1 = float(np.exp(coeffs[1]))

        # R² of fit
        predicted = coeffs[0] * log_k + coeffs[1]
        ss_res = np.sum((log_s - predicted) ** 2)
        ss_tot = np.sum((log_s - log_s.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Energy at various ranks
        total_energy = (S ** 2).sum()
        def energy_at(r):
            return (S[:r] ** 2).sum() / total_energy

        # Distance from 1/φ
        deviation = abs(alpha - INV_PHI)

        results.append({
            'name': name,
            'shape': f"{out_dim}×{in_dim}",
            'sigma1': round(sigma1, 4),
            'alpha': round(alpha, 4),
            'deviation_from_inv_phi': round(deviation, 4),
            'r2': round(r2, 4),
            'actual_s1': round(float(S[0]), 4),
            'energy_r64': round(energy_at(64), 4) if len(S) >= 64 else None,
            'energy_r128': round(energy_at(128), 4) if len(S) >= 128 else None,
        })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to .pt checkpoint')
    parser.add_argument('--hf-model', type=str, help='HuggingFace model name')
    parser.add_argument('--output', type=str, help='Save JSON report to file')
    args = parser.parse_args()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        # Handle both raw state_dict and wrapped checkpoints
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

    # Summary
    alphas = [r['alpha'] for r in results]
    r2s = [r['r2'] for r in results]
    devs = [r['deviation_from_inv_phi'] for r in results]

    print(f"\n{'='*80}")
    print(f"  SPECTRAL AUTOPSY — {len(results)} weight matrices")
    print(f"{'='*80}\n")
    print(f"  1/φ = {INV_PHI:.10f}\n")
    print(f"  {'Layer':<50s} {'Shape':>10s} {'α':>8s} {'Δ(1/φ)':>8s} {'R²':>6s} {'σ₁':>8s}")
    print(f"  {'-'*50} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    for r in results:
        marker = '✓' if r['deviation_from_inv_phi'] < 0.1 else '✗'
        print(f"  {r['name']:<50s} {r['shape']:>10s} {r['alpha']:>8.4f} {r['deviation_from_inv_phi']:>8.4f} {r['r2']:>6.3f} {r['sigma1']:>8.2f} {marker}")

    print(f"\n  Summary:")
    print(f"    Mean α:        {np.mean(alphas):.4f}")
    print(f"    Median α:      {np.median(alphas):.4f}")
    print(f"    Std α:         {np.std(alphas):.4f}")
    print(f"    Mean R²:       {np.mean(r2s):.4f}")
    print(f"    Mean |α-1/φ|:  {np.mean(devs):.4f}")
    print(f"    Layers ≈ 1/φ:  {sum(1 for d in devs if d < 0.1)}/{len(devs)}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {args.output}")

if __name__ == '__main__':
    main()
```

**Step 3: Run on G2-A checkpoint (on pod 2)**
```bash
ssh pod2 "cd /workspace/wavegpt && PYTHONPATH=. python3 scripts/spectral_autopsy.py \
  --checkpoint runs/G2-A-standard-random/best.pt \
  --output runs/G2-A-spectral-autopsy.json"
```
Expected: α values reported per layer, mean α near 0.618

**Step 4: Commit**
```bash
git add -A && git commit -m "feat: spectral_autopsy — verify power law on any checkpoint"
```

---

## Batch 2: Fine-tuning Pipeline

### Task 4: SpectralLinear gradient test — verify only spectrum gets gradients

**Files:**
- Test: `tests/test_spectral_linear.py` (add tests)

```python
def test_sigma1_gradient_frozen_geometry():
    """In sigma1 mode, gradients flow only to sigma1, not U or V."""
    linear = torch.nn.Linear(64, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='sigma1')
    x = torch.randn(2, 5, 64)
    y = spec(x)
    loss = y.sum()
    loss.backward()
    assert spec.sigma1.grad is not None
    assert not spec.U.requires_grad
    assert not spec.V.requires_grad

def test_per_mode_gradient_frozen_geometry():
    """In per_mode, gradients flow to spectrum vector, not U or V."""
    linear = torch.nn.Linear(64, 64, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=16, mode='per_mode')
    x = torch.randn(2, 5, 64)
    y = spec(x)
    loss = y.sum()
    loss.backward()
    assert spec.spectrum.grad is not None
    assert spec.spectrum.grad.shape == (16,)
```

---

### Task 5: Spectral fine-tune training script

**Files:**
- Create: `scripts/finetune_spectral.py`

Trains ONLY spectral params (σ₁ or per-mode amplitudes) on a corpus.
Accepts any checkpoint we've decomposed. Logs val loss, spectral drift.

---

### Task 6: Merge-back — export decomposed model with tuned spectrum as standard weights

**Files:**
- Add to: `wavegpt/spectral_linear.py`

```python
def to_linear(self) -> nn.Linear:
    """Reconstruct a standard nn.Linear from spectral params."""
    spectrum = self.get_spectrum()
    W = (self.U * spectrum.unsqueeze(0)) @ self.V.t()
    linear = nn.Linear(self.in_dim, self.out_dim, bias=self.bias is not None)
    linear.weight.data = W
    if self.bias is not None:
        linear.bias.data = self.bias.clone()
    return linear
```

Test: decompose → fine-tune spectrum → merge back → output matches new spectrum.

---

## Batch 3: Real Model Integration

### Task 7: HuggingFace model support — from_pretrained decomposition

Wrap any `AutoModelForCausalLM` with spectral surgery. Handle:
- Tied weights (lm_head = wte)
- Embedding skip (don't decompose embeddings)
- Architecture-specific skip patterns (layernorm, etc.)

### Task 8: Autopsy on HuggingFace GPT-2

Run `spectral_autopsy.py --hf-model gpt2` to verify 1/φ on OpenAI's original weights.

### Task 9: Autopsy on Qwen3 (smallest available)

Pick the smallest open-weight Qwen model, verify the equation holds.

---

## Batch 4: Praxis Fine-Tune

### Task 10: Export RAI corpus for spectral fine-tuning

Use existing `export_rich_corpus.py` or `export_corpus.py` output.
Format: tokenized sequences with loss masks.

### Task 11: Fine-tune target model on RAI corpus

- Decompose model → SpectralLinear (per_mode)
- Freeze everything except spectrum params
- Train on RAI corpus
- Compare: spectral fine-tune (768 params/layer) vs LoRA (hundreds of thousands)

### Task 12: Evaluation + demo

- Perplexity on held-out RAI data
- Generate samples — does it sound like Ray?
- Compare to base model and LoRA-finetuned
