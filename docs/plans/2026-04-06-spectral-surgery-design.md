# Spectral Surgery — Design Doc

**Date:** 2026-04-06  
**Insight:** "The double-slit experiment. When you measure it, particles. In flux, waves. We cannot train it like a wave."

## The Shift

We proved that trained weight matrices converge to:

```
W = σ₁ · Σₖ k^{-1/φ} · uₖ · vₖᵀ
```

We tried training *inside* this parameterization (HarmonicGPT). It worked at 30M (PPL 7.0, 6.4× compression) but diverged at 124M — five times, every configuration. The wave collapses when you constrain it.

The equation is **Planck's law, not a blueprint**. You don't build a black body by constraining photons. You heat the cavity, the distribution emerges, then you use the law.

New approach:
1. **Train standard** — let SGD find the power law naturally
2. **Decompose** — SVD reveals the spectral structure  
3. **Compress** — truncate modes below the Pythagorean comma
4. **Fine-tune** — adjust spectral amplitudes on target corpus

## What This Enables for Praxis

Take any pretrained model. Decompose it into spectral form. Fine-tune the **spectral amplitudes** on Ray's corpus. The geometry (U, V) carries general knowledge. The amplitudes (σ) carry personality.

This is **Spectral LoRA** — but principled. Standard LoRA adds arbitrary low-rank corrections. Spectral fine-tuning adjusts the energy distribution across existing modes. It says: "this model already knows how to think — make it think *louder* at mode 7 and *quieter* at mode 43."

### Fine-Tuning Levels

| Level | Params | What Changes | Analogy |
|-------|--------|--------------|---------|
| **σ₁-only** | 1 per layer (~72) | Layer-level volume | Master EQ |
| **Per-mode σ** | rank per layer (~18K) | Individual mode amplitudes | Parametric EQ |
| **Spectral LoRA** | 2×rank per layer (~36K) | Mode amplitudes + small basis corrections | EQ + mic repositioning |
| **Standard LoRA** | 2×rank×d per layer (~1.2M) | Arbitrary correction | Rebuilding the instrument |

The first two levels are novel. Nobody has fine-tuned by adjusting singular value distributions.

## Architecture

### Core Module: `SpectralModel`

Wraps any HuggingFace model. Decomposes linear layers, freezes geometry, exposes spectral knobs.

```python
class SpectralModel(nn.Module):
    """
    Wraps a pretrained model with spectral decomposition.
    
    Every nn.Linear → frozen (U, V) + learnable spectrum.
    Forward pass reconstructs W from spectrum + bases, same outputs.
    """
    
    def __init__(self, base_model, rank='auto', mode='sigma1'):
        # mode: 'sigma1' | 'per_mode' | 'spectral_lora'
        ...
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        # Load HF model, decompose, wrap
        ...
    
    def spectral_report(self) -> dict:
        # Per-layer: fitted α, σ₁, rank, energy captured, deviation from 1/φ
        ...
    
    def save_spectral(self, path):
        # Save only the spectral params (tiny file)
        ...
    
    def load_spectral(self, path):
        # Load spectral params, reconstruct W
        ...
```

### Decomposition: `SpectralLinear`

Replaces `nn.Linear` during surgery. Simpler than HarmonicLinear — no training dynamics, just reconstruction.

```python
class SpectralLinear(nn.Module):
    """
    Post-training spectral layer.
    
    U, V are FROZEN (buffers). Only spectrum is learnable.
    Three modes:
      - sigma1: one scalar, spectrum = σ₁ · k^{-1/φ}
      - per_mode: one scalar per mode, free amplitudes
      - spectral_lora: free amplitudes + small U,V correction
    """
    
    def __init__(self, U, V, spectrum, mode='per_mode'):
        self.register_buffer('U', U)      # frozen geometry
        self.register_buffer('V', V)      # frozen geometry
        self.spectrum = nn.Parameter(spectrum)  # learnable amplitudes
```

### Pipeline

```
pretrained model ─→ decompose ─→ verify power law ─→ compress ─→ fine-tune ─→ export
                     (SVD)        (fit α, check      (truncate    (learn σ    (merge W
                                   ≈ 1/φ)            modes)       on corpus)  back)
```

## Verification Plan

Before any fine-tuning, we verify the equation holds on real models:
- **G2-A checkpoint** (our 124M, trained on SFT data) — already have this
- **GPT-2 124M from HuggingFace** — OpenAI's original weights
- **Qwen 2.5 0.5B / 1.5B** — modern model at useful scale

For each: SVD every linear layer, fit α, check if ≈ 1/φ. This validates the equation on models we didn't train.

## Implementation Plan

### Batch 1: Decompose + Verify (foundation)
1. `wavegpt/spectral_linear.py` — SpectralLinear module (frozen U,V + learnable spectrum)
2. `wavegpt/spectral_surgery.py` — decompose any model, replace nn.Linear → SpectralLinear
3. `scripts/spectral_autopsy.py` — CLI: load model, SVD all layers, report α per layer, verify 1/φ
4. Tests for decompose → reconstruct roundtrip (output equivalence)

### Batch 2: Compress + Fine-tune
5. Compression: truncate to rank r, measure quality loss on validation set
6. `scripts/spectral_finetune.py` — fine-tune spectral amplitudes on a corpus
7. Three fine-tuning modes: σ₁-only, per-mode, spectral-LoRA
8. Test: fine-tune on RAI corpus, measure perplexity shift

### Batch 3: HuggingFace Integration
9. `SpectralModel.from_pretrained()` — one-line decomposition of any HF model
10. `SpectralModel.save_pretrained()` / `push_to_hub()` — export merged or spectral-only
11. Support for Qwen, Llama, GPT-2 architectures

### Batch 4: Praxis Integration
12. Fine-tune a real model (Qwen 2.5 1.5B?) on RAI corpus
13. Compare against standard LoRA fine-tuning
14. Evaluate: perplexity, style matching, knowledge retention

## Why This Could Be Big

**Standard LoRA** (rank=16): ~1.2M params per layer, arbitrary correction  
**Spectral fine-tuning** (per-mode): ~768 params per layer, structured correction  

If spectral fine-tuning matches LoRA quality with 1000× fewer parameters, that's a publishable result and a practical tool. The hypothesis: personality lives in the amplitudes, not the geometry. The geometry is universal (learned from pretraining data). The amplitudes are individual (tuned to the target voice).

## What We're NOT Doing

- ~~Training from scratch with spectral parameterization~~ (the wave collapses)
- ~~HarmonicGPT at scale~~ (5 failures, same pattern)
- ~~Custom training frameworks~~ (standard PyTorch/HF training works)

## Success Criteria

1. **Equation verified** on ≥3 external models (α ≈ 1/φ ± 0.05)
2. **Roundtrip loss** < 0.1% (decompose → reconstruct ≈ original)
3. **Spectral fine-tuning** achieves ≥80% of LoRA quality with ≤5% of parameters
4. **RAI demo**: model fine-tuned on Ray's corpus produces Ray-like responses
