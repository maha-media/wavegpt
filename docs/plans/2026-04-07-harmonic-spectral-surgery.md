# Harmonic Spectral Surgery — Implementation Plan

**Goal:** Apply theory-guided spectral fine-tuning to Qwen3.5-27B with harmonic priors, compare against vanilla SVFit and LoRA.
**Architecture:** SpectralLinear + residual preservation + harmonic regularization + adaptive rank. HuggingFace integration for Qwen3.5-27B.
**Design Doc:** `docs/plans/2026-04-07-harmonic-spectral-surgery-design.md`
**Estimated Tasks:** 16 tasks, 5 batches
**Complexity:** Large

---

## Batch 1: Harmonic Extensions to SpectralLinear (local, TDD)

### Task 1: Residual Preservation in SpectralLinear

**Files:**
- Modify: `wavegpt/spectral_linear.py`
- Test: `tests/test_spectral_linear.py`

**Step 1: Write failing test**
```python
def test_residual_preservation_exact():
    """With residual, decompose + reconstruct is lossless at any rank."""
    linear = torch.nn.Linear(64, 128, bias=False)
    x = torch.randn(2, 10, 64)
    y_orig = linear(x)
    spec = SpectralLinear.from_linear(linear, rank=8, mode='per_mode', keep_residual=True)
    y_spec = spec(x)
    torch.testing.assert_close(y_orig, y_spec, atol=1e-5, rtol=1e-4)

def test_residual_stored_as_buffer():
    """Residual should be a frozen buffer, not a parameter."""
    linear = torch.nn.Linear(64, 128, bias=False)
    spec = SpectralLinear.from_linear(linear, rank=8, mode='per_mode', keep_residual=True)
    assert hasattr(spec, 'residual')
    assert not spec.residual.requires_grad
    # Learnable params should still be just the spectrum
    learnable = [p for p in spec.parameters() if p.requires_grad]
    assert len(learnable) == 1
    assert learnable[0].numel() == 8
```

**Step 2: Verify fails**
Run: `pytest tests/test_spectral_linear.py::test_residual_preservation_exact -v`
Expected: FAIL — `from_linear() got unexpected keyword argument 'keep_residual'`

**Step 3: Implement**
Add `keep_residual` parameter to `from_linear()` and `__init__()`. In forward:
```python
# After spectral reconstruction
out = xVs @ self.U.t()
if self.residual is not None:
    out = out + x @ self.residual.t()  # residual correction
```
Residual = `W_original - U_r @ diag(S_r) @ V_r^T`, stored as frozen buffer.

**Step 4: Verify passes**
Run: `pytest tests/test_spectral_linear.py -v`
Expected: all pass (old + 2 new)

**Step 5: Commit**
```bash
git add -A && git commit -m "feat: residual preservation in SpectralLinear (Pythagorean comma)"
```

---

### Task 2: Harmonic Prior and Regularization

**Files:**
- Modify: `wavegpt/spectral_linear.py`
- Create: `wavegpt/harmonic_prior.py`
- Test: `tests/test_harmonic_prior.py`

**Step 1: Write failing tests**
```python
# tests/test_harmonic_prior.py
import torch
from wavegpt.harmonic_prior import harmonic_regularization, compute_adaptive_rank, INV_PHI

def test_harmonic_regularization_zero_at_prior():
    """If spectrum exactly matches k^{-1/φ}, regularization loss is 0."""
    from wavegpt.spectral_linear import SpectralLinear
    k = torch.arange(1, 17, dtype=torch.float)
    s = 5.0 * k.pow(-INV_PHI)
    U = torch.eye(32, 16)
    V = torch.eye(32, 16)
    spec = SpectralLinear(U, s, V, mode='per_mode')
    loss = harmonic_regularization(spec)
    assert loss.item() < 1e-6

def test_harmonic_regularization_nonzero_when_deviated():
    """Spectrum deviating from power law should have positive loss."""
    from wavegpt.spectral_linear import SpectralLinear
    s = torch.ones(16)  # flat spectrum, far from power law
    U = torch.eye(32, 16)
    V = torch.eye(32, 16)
    spec = SpectralLinear(U, s, V, mode='per_mode')
    loss = harmonic_regularization(spec)
    assert loss.item() > 0.1

def test_adaptive_rank_increases_with_deviation():
    """Layers further from 1/φ should get higher rank."""
    r_close = compute_adaptive_rank(alpha=0.62, base_rank=192)  # near 1/φ
    r_far = compute_adaptive_rank(alpha=1.0, base_rank=192)     # far from 1/φ
    assert r_far > r_close

def test_adaptive_rank_at_golden_ratio():
    """Layer exactly at 1/φ should get base_rank."""
    r = compute_adaptive_rank(alpha=INV_PHI, base_rank=192)
    assert r == 192
```

**Step 2: Verify fails**
Run: `pytest tests/test_harmonic_prior.py -v`
Expected: FAIL — "No module named 'wavegpt.harmonic_prior'"

**Step 3: Implement**
```python
# wavegpt/harmonic_prior.py
"""
Harmonic priors for spectral fine-tuning.

The key insight: trained weights converge to W = σ₁ · Σ k^{-1/φ} · u_k · v_k^T.
We use this as a prior for rank allocation and regularization.
"""
import torch
from .spectral_linear import SpectralLinear

PHI = (1 + 5**0.5) / 2
INV_PHI = 1 / PHI


def harmonic_regularization(module_or_model, lambda_h=1.0):
    """Spectral weight decay toward k^{-1/φ} prior."""
    loss = torch.tensor(0.0)
    modules = [module_or_model] if isinstance(module_or_model, SpectralLinear) else [
        m for m in module_or_model.modules() if isinstance(m, SpectralLinear)
    ]
    for m in modules:
        if m.mode != 'per_mode':
            continue
        s = m.spectrum
        device = s.device
        k = torch.arange(1, len(s) + 1, device=device, dtype=s.dtype)
        prior = s[0].detach() * k.pow(-INV_PHI)
        loss = loss.to(device) + ((s - prior) ** 2).mean()
    return lambda_h * loss


def compute_adaptive_rank(alpha, base_rank, beta=2.0, max_rank=None):
    """Allocate rank proportional to deviation from 1/φ."""
    deviation = abs(alpha - INV_PHI)
    rank = int(base_rank * (1.0 + beta * deviation))
    if max_rank is not None:
        rank = min(rank, max_rank)
    return rank
```

**Step 4: Verify passes**
Run: `pytest tests/test_harmonic_prior.py -v`
Expected: 4 passed

**Step 5: Commit**
```bash
git add -A && git commit -m "feat: harmonic_prior — regularization + adaptive rank allocation"
```

---

### Task 3: Adaptive Rank in spectral_decompose()

**Files:**
- Modify: `wavegpt/spectral_surgery.py`
- Test: `tests/test_spectral_surgery.py`

**Step 1: Write failing test**
```python
def test_adaptive_rank_decomposition():
    """With adaptive rank, layers get different ranks based on their α."""
    model = TinyModel()
    decomposed = spectral_decompose(model, rank='adaptive', base_rank=8, mode='per_mode')
    from wavegpt.spectral_linear import SpectralLinear
    r1 = decomposed.linear1.rank
    r2 = decomposed.linear2.rank
    # Ranks should exist and be > 0 (exact values depend on random init)
    assert r1 > 0 and r2 > 0
    # With adaptive, they may differ (random init means α varies)
    # At minimum, verify both are SpectralLinear
    assert isinstance(decomposed.linear1, SpectralLinear)
    assert isinstance(decomposed.linear2, SpectralLinear)
```

**Step 2: Verify fails**
Expected: FAIL — rank='adaptive' not handled

**Step 3: Implement**
In `spectral_decompose()`, when `rank='adaptive'`:
1. First pass: SVD each layer, fit α (quick, no replacement)
2. `compute_adaptive_rank()` per layer
3. Second pass: decompose with per-layer rank

Add `base_rank` and `adaptive_beta` parameters.

**Step 4: Verify passes**
**Step 5: Commit**
```bash
git add -A && git commit -m "feat: adaptive rank decomposition guided by harmonic prior"
```

---

## Batch 2: HuggingFace Integration (local + pod)

### Task 4: Generic HF model loading in finetune script

**Files:**
- Modify: `scripts/finetune_spectral.py`

Add `--hf-model` flag that loads any `AutoModelForCausalLM`. Remove hard dependency on WaveGPT model class. The script should work with both:
- `--checkpoint` (WaveGPT .pt file, requires --n-layer etc.)
- `--hf-model Qwen/Qwen3.5-27B` (auto-detects architecture)

Use HF tokenizer for data loading when `--hf-model` is used.

**Test:** Run on pod with `--hf-model gpt2` (small, quick sanity check).

**Commit:**
```bash
git add -A && git commit -m "feat: HuggingFace model support in finetune_spectral.py"
```

---

### Task 5: HF tokenizer-aware data export for RAI corpus

**Files:**
- Create: `scripts/export_rai_for_hf.py`

Export the RAI corpus tokenized with Qwen3.5's tokenizer (248K vocab). Reads the existing enriched text from MongoDB, tokenizes with HF tokenizer, writes train.bin / val.bin in our binary format.

**Commit:**
```bash
git add -A && git commit -m "feat: export RAI corpus for any HF tokenizer"
```

---

### Task 6: Harmonic regularization + anti-collapse in finetune loop

**Files:**
- Modify: `scripts/finetune_spectral.py`

Add flags:
- `--harmonic-lambda 0.01` — harmonic regularization strength
- `--collapse-alpha 0.05` — anti-collapse variance penalty
- `--keep-residual` — preserve Pythagorean comma
- `--adaptive-rank` — use theory-guided rank allocation

These compose additively in the loss:
```python
total_loss = ce_loss + harmonic_regularization(model, lambda_h) + collapse_penalty
```

**Test:** Run on pod with GPT-2, verify harmonic loss logged, compare with/without.

**Commit:**
```bash
git add -A && git commit -m "feat: harmonic regularization + anti-collapse in spectral fine-tuning"
```

---

## Batch 3: Qwen3.5-27B Setup (pod)

### Task 7: Pod environment setup

On pod (RTX PRO 6000, 96GB):
```bash
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"
pip install accelerate sentencepiece
# Download Qwen3.5-27B (BF16, ~56GB)
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-27B', torch_dtype='bfloat16')"
```

Verify model loads and fits in VRAM. Report actual nn.Linear count.

---

### Task 8: Qwen3.5-27B Spectral Autopsy (Exp Q-A)

Run spectral_autopsy.py on Qwen3.5-27B pretrained weights:
```bash
python scripts/spectral_autopsy.py --hf-model Qwen/Qwen3.5-27B --output runs/Q-A-autopsy.json
```

This is the key universality test:
- Does α ≈ 1/φ in a model trained by Alibaba on trillions of tokens?
- Do DeltaNet layers vs full attention layers show different α?
- Does the two-constant model (representation vs projection) hold?

Save results. If 1/φ confirmed → proceed. If not → reassess theory.

**Commit:**
```bash
git add -A && git commit -m "data: Qwen3.5-27B spectral autopsy results"
```

---

### Task 9: Export RAI corpus with Qwen3.5 tokenizer

Run `export_rai_for_hf.py` on the RAI corpus with Qwen3.5 tokenizer.
Upload tokenized data to pod.

---

## Batch 4: Fine-Tuning Experiments (pod)

### Task 10: Exp Q-B — Vanilla SVFit Baseline

```bash
python scripts/finetune_spectral.py \
  --hf-model Qwen/Qwen3.5-27B \
  --data-dir data/rai-qwen \
  --run-name Q-B-vanilla-svfit \
  --rank 256 --mode per_mode \
  --batch-size 2 --grad-accum 8 --max-steps 3000 \
  --lr 1e-3 --warmup-steps 200
```

No harmonic priors. This is the "what SVFit would do" baseline.

---

### Task 11: Exp Q-C — Harmonic Spectral Surgery (full)

```bash
python scripts/finetune_spectral.py \
  --hf-model Qwen/Qwen3.5-27B \
  --data-dir data/rai-qwen \
  --run-name Q-C-harmonic-full \
  --adaptive-rank --base-rank 192 \
  --mode per_mode --keep-residual \
  --harmonic-lambda 0.01 --collapse-alpha 0.05 \
  --batch-size 2 --grad-accum 8 --max-steps 3000 \
  --lr 1e-3 --warmup-steps 200
```

Full harmonic treatment: adaptive rank, regularization, residual, anti-collapse.

---

### Task 12: Exp Q-D — LoRA r-16 Baseline

Use PEFT/Unsloth for standard LoRA fine-tuning:
```bash
python scripts/finetune_lora.py \
  --hf-model Qwen/Qwen3.5-27B \
  --data-dir data/rai-qwen \
  --run-name Q-D-lora-r16 \
  --lora-rank 16 --lora-alpha 16 \
  --batch-size 2 --grad-accum 8 --max-steps 3000 \
  --lr 2e-4
```

~111M trainable params, 212MB adapter. The head-to-head comparison.

---

## Batch 5: Evaluation + Paper Data (pod + local)

### Task 13: Perplexity comparison table

Evaluate all runs on held-out RAI val set:
| Experiment | Params | File Size | Val PPL | vs Base |
|-----------|--------|-----------|---------|---------|
| Base (no fine-tune) | 0 | 0 | ? | — |
| Q-B Vanilla SVFit | ~115K | ~450KB | ? | ? |
| Q-C Harmonic | ~115K | ~450KB | ? | ? |
| Q-D LoRA | ~111M | ~212MB | ? | ? |

---

### Task 14: Ablations (if Q-C > Q-B)

Run Q-C with each harmonic component removed:
- Q-C1: flat rank (no adaptive)
- Q-C2: no harmonic regularization
- Q-C3: no residual
- Q-C4: no anti-collapse
- Q-C5: random data order

Each isolates the contribution of one harmonic prior.

---

### Task 15: Generation samples

Generate text from each model variant on prompts like:
- "The future of artificial intelligence will..."
- "Consciousness in machines requires..."
- "The singularity represents..."

Compare: does the harmonic fine-tuned model sound more like Ray?

---

### Task 16: Paper draft outline

Write `docs/paper-outline.md` with:
- Abstract
- Key figures (autopsy heatmap, 1/φ convergence, perplexity comparison, ablation table)
- Contributions list
- Related work positioning vs SVFit/SVFT/Spectral Adapter
