# Harmonic Spectral Surgery — Design Doc

**Date:** 2026-04-07
**Prior art:** SVFit (2024), SVFT (NeurIPS 2024), SVDiff (ICCV 2023), Spectral Adapter (NeurIPS 2024), PiSSA (NeurIPS 2024)
**Our addition:** Harmonic priors from the 1/φ discovery — theory-guided rank allocation, spectral regularization, residual preservation, anti-collapse, spectral data ordering.
**Target model:** Qwen3.5-27B (27B dense, 64 layers, hidden 5120, Apache 2.0)

## What Exists (Prior Art)

The mechanism of SVD-decompose → freeze U,V → train singular values is established:

| Method | Venue | What trains | Notes |
|--------|-------|-------------|-------|
| SVDiff | ICCV 2023 | Diagonal S | Diffusion models, 2200× fewer params |
| SVFit | arXiv 2024 | Top-r singular values | 16× fewer params than LoRA |
| SVFT | NeurIPS 2024 | Sparse M in U·M·V^T basis | Diagonal M = our per_mode. Off-diagonal adds expressivity |
| Spectral Adapter | NeurIPS 2024 | Additive/rotational tuning of top U,V columns | Cayley rotation preserves SVD |
| PiSSA | NeurIPS 2024 | Principal SVD components via LoRA architecture | Better init for LoRA |

**None of them have a theory for the spectral structure.** They decompose, tune, and measure. We know *why* the spectrum looks the way it does (1/φ power law), *what it means* (representation vs projection layers), and *how to use that knowledge*.

## What We Add (Novel Contributions)

### 1. Adaptive Spectral Budget via Harmonic Prior

Prior art: flat rank across all layers.
Our approach: allocate rank proportional to each layer's deviation from 1/φ.

**Insight:** Layers close to α = 1/φ are well-described by the power law — fewer free modes needed, the prior already captures their shape. Layers that deviate (projection layers at α ≈ 1.0) need more freedom.

```python
def compute_adaptive_rank(layer_alpha, base_rank, beta=2.0):
    """More rank for layers that deviate from the harmonic prior."""
    deviation = abs(layer_alpha - INV_PHI)
    return int(base_rank * (1.0 + beta * deviation))
```

From G2-A autopsy: attention layers α ≈ 0.53 (Δ = 0.09), MLP-up α ≈ 0.57 (Δ = 0.05), projection α ≈ 0.66 (Δ = 0.04). With β = 2.0 and base_rank = 192:
- Attention: rank ≈ 227 (needs more room, furthest from prior)
- MLP-up: rank ≈ 211
- Projection: rank ≈ 207

This is rank-adaptive per layer, guided by theory.

### 2. Harmonic Regularization (Spectral Weight Decay)

Prior art: no regularization on singular values during fine-tuning.
Our approach: penalize departure from the power-law equilibrium.

```python
def harmonic_regularization(model, lambda_h=0.01):
    """Pull spectral amplitudes toward k^{-1/φ} prior."""
    loss = 0.0
    for module in model.modules():
        if isinstance(module, SpectralLinear) and module.mode == 'per_mode':
            s = module.spectrum
            k = torch.arange(1, len(s) + 1, device=s.device, dtype=s.dtype)
            # Power-law prior: s_k ∝ s_1 · k^{-1/φ}
            prior = s[0].detach() * k.pow(-INV_PHI)
            loss = loss + ((s - prior) ** 2).mean()
    return lambda_h * loss
```

This is spectral weight decay — L2 toward the natural equilibrium, not toward zero. The model can deviate from 1/φ if the data demands it, but there's a cost. Prevents degenerate spectral shapes.

### 3. Residual Preservation (The Pythagorean Comma)

Prior art: SVFit/SVFT discard modes below rank-r.
Our approach: keep the residual as a frozen correction.

```
W = U_r · diag(s_trainable) · V_r^T + W_residual_frozen
```

NeurIPS 2025 confirmed small singular values matter. Our theory quantifies the loss: at rank-256 on 768-dim, we lose 13% energy. At rank-256 on 5120-dim (Qwen3.5), we'd lose more. The frozen residual `W_residual = W_full - U_r · diag(s_original) · V_r^T` costs no extra learnable params but preserves fine structure.

### 4. Anti-Collapse Variance Penalty

Prior art: no mode collapse prevention during spectral fine-tuning.
Our approach: proven variance penalty on hidden states.

```python
collapse_loss = -alpha * torch.log(hidden_states.var(dim=-1).mean() + 1e-8)
```

When adjusting 256 amplitudes, it's easy to accidentally suppress modes. The variance penalty keeps the representation diverse. Confirmed to help in experiments M/N (PPL 105 → 93).

### 5. Spectral Data Ordering

Prior art: random data ordering.
Our approach: spectral curriculum ordering via SVD of TF-IDF matrix.

Proven: harmonic ordering PPL 6.3 vs random PPL 13.5 (experiment S-B vs S-A). 2x advantage from data ordering alone. We apply the same spectral ordering to the RAI fine-tuning corpus.

## Architecture

### SpectralLinear Extensions

Add to existing `SpectralLinear`:
- `residual` buffer: frozen W_residual for comma preservation
- `harmonic_prior()`: return k^{-1/φ} scaled to this layer's σ₁
- `from_linear()`: optionally compute and store residual

### HarmonicSpectralSurgery

Wraps `spectral_decompose()` with harmonic priors:
- Autopsy each layer first (fit α)
- Compute adaptive rank per layer
- Decompose with per-layer rank
- Store residuals
- Report harmonic budget allocation

### finetune_spectral.py Generalization

Current script is WaveGPT-only. Generalize to:
- Any HuggingFace model via `--hf-model`
- Harmonic regularization via `--harmonic-lambda`
- Adaptive rank via `--adaptive-rank`
- Residual preservation via `--keep-residual`
- Anti-collapse via `--collapse-alpha`
- LoRA comparison mode via `--lora-baseline`

## Qwen3.5-27B Specifics

```
Architecture:
  64 layers (48 Gated DeltaNet + 16 full attention, 3:1 pattern)
  hidden_size = 5120
  FFN intermediate = 17408
  vocab = 248,320
  ~448 decomposable nn.Linear modules
  tie_word_embeddings = false (skip both embed + lm_head)
  BF16: ~56GB VRAM (fits on 96GB RTX PRO 6000)
  Requires transformers >= 5.2

Layer types per block:
  DeltaNet: in_proj_qkvz, in_proj_ba, out_proj, gate + gate_proj, up_proj, down_proj
  Full attn: q_proj, k_proj, v_proj, o_proj + gate_proj, up_proj, down_proj
```

### Spectral Param Budget

| Config | Params | File Size (fp32) | vs LoRA r-16 |
|--------|--------|-------------------|-------------|
| Flat rank-256 | 114,688 | 448 KB | 969× fewer |
| Adaptive rank (avg 256) | ~115K | ~450 KB | ~970× fewer |
| LoRA r-16 | ~111M | ~212 MB | baseline |

### Memory Budget (96GB pod)

| Component | VRAM |
|-----------|------|
| Qwen3.5-27B BF16 | ~56 GB |
| Activation memory (bs=2, seq=1024) | ~8 GB |
| Optimizer states (115K params) | < 1 MB |
| Residuals (frozen, same dtype) | ~8 GB |
| Headroom | ~24 GB |

SVD decomposition happens on CPU — no VRAM cost.

## Experiment Plan

### Exp Q-A: Qwen3.5-27B Autopsy
- Load pretrained weights, SVD all layers
- Fit α per layer, report deviation from 1/φ
- Confirm/deny universality on a model trained by Alibaba
- Compare layer types: DeltaNet projections vs full attention vs MLP

### Exp Q-B: Vanilla SVFit Baseline
- Flat rank-256, per_mode, no regularization, no residual
- Fine-tune on RAI corpus (4.7M tokens)
- This is the "prior art" baseline

### Exp Q-C: Harmonic Spectral Surgery (full)
- Adaptive rank per layer (guided by autopsy α)
- Harmonic regularization λ_h = 0.01
- Residual preservation
- Anti-collapse α = 0.05
- Spectral data ordering of RAI corpus
- Same total param budget as Q-B

### Exp Q-D: LoRA r-16 Baseline
- Standard LoRA on same model, same corpus
- ~111M trainable params, 212MB adapter file
- The comparison: does 450KB compete with 212MB?

### Ablations (if Q-C beats Q-B)
- Q-C1: no adaptive rank (flat) but keep everything else
- Q-C2: no harmonic regularization
- Q-C3: no residual preservation
- Q-C4: no anti-collapse
- Q-C5: random data ordering

## Success Criteria

1. **Equation confirmed on Qwen3.5** — α ≈ 1/φ ± 0.1 for majority of layers
2. **Harmonic beats vanilla** — Q-C val loss < Q-B val loss (same param count)
3. **Competitive with LoRA** — Q-C within 20% of Q-D quality with 970× fewer params
4. **RAI personality transfer** — generated text sounds like Ray Kurzweil
5. **450KB personality file** — one file per voice, hot-swappable

## Paper Framing

*"Harmonic Priors for Spectral Fine-Tuning: Theory-Guided Adaptation of Large Language Models"*

Contributions:
1. Discovery: trained weight spectra converge to k^{-1/φ} (golden ratio exponent)
2. Theory-guided rank allocation outperforms flat rank
3. Harmonic regularization prevents spectral drift during fine-tuning
4. Residual preservation recovers the Pythagorean comma
5. 450KB personality files on 27B model, competitive with 212MB LoRA
