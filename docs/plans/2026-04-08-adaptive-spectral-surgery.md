# Adaptive Spectral Surgery — Plan
**Date**: 2026-04-08
**Status**: Active

## Context

Rank-256 uniform decomposition of Qwen3.5-27B (8.6GB, 107K params) converges in loss but produces punctuation soup — it can't speak. The truncation cut through the flat top of MLP spectra (k₀≈1000), destroying language coherence while preserving token statistics.

The fix: **k₀-adaptive rank**. Each layer's rank is set to clear its spectral flat top: `rank = k₀ × 1.5 + 128`. Attention (k₀≈100) gets rank ~278, MLP (k₀≈1000) gets rank ~1700. Result: ~13.6GB model (67% compression), 350K learnable params, 1.4MB spectral file.

## The Equation

```
σ_k = A(type, depth) · (k + k₀(type, depth, d_eff))^{-1/φ}
```

- **1/φ = 0.6180339887** — universal constant (confirmed 338/338 layers Qwen3.5-27B + GPT-2 124M)
- **k₀** — per-layer spectral offset, approximated by `γ(type) · √(d_in·d_out) · h(depth)`
- **A** — per-layer amplitude (not yet characterized)

### What's confirmed
- 1/φ exponent: every layer of two very different models (GPT-2 124M, Qwen3.5-27B hybrid DeltaNet)
- k₀ hierarchy by type: R²=0.73 with simple model (36 layers)
- k₀ scales with dimension and depth

### What's pending
- Free-α analysis: is 1/φ exact (std < 0.005) or approximate (std > 0.02)?
- Full k₀ regression with 496 layers: β₁, depth function h(d), refined γ table
- A(type, depth) characterization
- Cross-model γ transfer (GPT-2, Llama, Mistral)

## Experiments

### Running
| ID | Pod | Config | Status | Notes |
|----|-----|--------|--------|-------|
| Q-C | Pod 2 | r256, harmonic (λ=0.01, α=0.05) | Step ~150/2000 | PPL curve useful even though can't speak |
| Adaptive decomp | Pod 1 | k₀×1.5+128, per_mode | Layer ~2/496 | ~12 hours ETA |

### Planned

#### Batch 1 — Adaptive Training (after decomposition)

**Q-D: Adaptive vanilla** (Pod 1)
- Load adaptive decomposed.pt
- Vanilla spectral fine-tuning (no priors)
- batch=8, block=512, grad_accum=2, lr=1e-3, 2000 steps
- **Key metric**: Can it speak at step 200?
- ~3 hours

**Q-E: Adaptive harmonic** (Pod 2, after Q-C finishes or is killed)
- Same adaptive decomposed.pt
- Harmonic priors: λ=0.01, collapse-α=0.05
- Same hyperparams as Q-D
- Head-to-head: does the harmonic prior help at correct rank?

#### Batch 2 — Baselines

**Q-F: LoRA r-16 baseline** (either pod)
- Standard LoRA (r=16, alpha=32) on Qwen3.5-27B
- Same data, same steps, same eval
- Head-to-head: spectral file (1.4MB) vs LoRA adapter (~200MB)

**Q-G: Full-rank baseline eval**
- No fine-tuning. Measure base Qwen3.5-27B on RAI corpus
- Already have: PPL ~26K, baseline responses saved

#### Batch 3 — Equation Validation

**Free-α analysis** (either pod, ~1 hour)
- Run `scripts/free_alpha_analysis.py` on full model
- The one number: std(α) across all 496 layers
- < 0.005 = universal constant proven
- > 0.05 = just a good approximation

**k₀ regression** (local, from decomposition log)
- Run `scripts/fit_k0_equation.py` on full 496-layer decompose.log
- Fit: `log(k₀) = β₀ + β₁·log(d_eff) + β₂·(d/L) + β₃·(d/L)² + type_offset`
- Get: refined γ table, depth function h(d), R² on full data
- Predict k₀ for unseen architectures

**Cross-model validation** (after k₀ regression)
- Download GPT-2-XL (1.5B) or Llama-3-8B
- Run spectral autopsy
- Test: does the γ table transfer? Does 1/φ hold?

#### Batch 4 — Generation Quality

**RAI 25-prompt eval** (after Q-D)
- Run `scripts/test_rai_baseline.py` against fine-tuned model
- Compare voice, accuracy, repetition against baseline
- The real test: does it sound like Ray?

**Spectral personality swap demo**
- Save Q-D spectral file (~1.4MB)
- Load base model + spectral file
- Generate side-by-side: base vs personality
- Measure swap time (should be < 1 second)

## Success Criteria

| Metric | Rank-256 | Target (Adaptive) |
|--------|----------|-------------------|
| Can speak | ✗ | ✓ |
| Model size | 8.6 GB | ~14 GB |
| Spectral file | 430 KB | 1.4 MB |
| Learnable params | 107K | 350K |
| PPL on RAI corpus | 39K (step 100) | < 1000 |
| Coherent generation | No | Yes |
| Voice quality | N/A | Recognizably Ray |

## Architecture

```
Base Qwen3.5-27B (54GB BF16, frozen)
    ↓ SVD per layer
Adaptive Decomposed (13.6GB)
    ├── U, V buffers (frozen, BF16) — geometry = knowledge
    ├── S parameters (learnable, float32) — spectrum = personality  
    └── Embeddings, LayerNorm (frozen)
    ↓ Fine-tune S on Ray's corpus
Spectral File (1.4MB)
    └── 350K floats — Ray's spectral fingerprint
```

## Risks

1. **Adaptive rank still can't speak**: k₀×1.5 might not be enough. Fix: k₀×2 or k₀×2.5. More aggressive padding.
2. **Decomposition OOM**: Higher ranks = bigger SVDs. All on CPU so unlikely, but slow (~12 hours).
3. **Training OOM at higher rank**: More parameters in U, V buffers. Should be ~14GB model vs 96GB card — plenty of headroom.
4. **k₀ equation doesn't generalize**: γ table may be Qwen-specific. Cross-model validation needed.
5. **Harmonic prior hurts at correct rank**: Might only help when the model is starved (rank-256). At proper rank, the prior may over-constrain.

## Timeline

- **Hour 0-12**: Adaptive decomposition (Pod 1). Q-C finishes or is killed (Pod 2).
- **Hour 12-15**: Q-D adaptive vanilla training (Pod 1). Transfer decomposed.pt to Pod 2, launch Q-E.
- **Hour 15-18**: Step 200 samples — can it speak? RAI eval if yes.
- **Hour 18-24**: Free-α analysis. k₀ regression. Results for paper.
- **Day 2**: LoRA baseline. Cross-model validation. Paper outline.
