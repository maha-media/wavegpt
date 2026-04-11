# Experiments

## Overview

| ID | Date | Model | Type | Key Variable | Best PPL | Coherent? | Status |
|----|------|-------|------|-------------|----------|-----------|--------|
| [S01](#s01) | Pre-2026-04 | GPT-2 16M | Train from scratch | Data curriculum | 93 | Yes | Done |
| [S02](#s02) | Pre-2026-04 | GPT-2 30M | HarmonicGPT | φ-constrained init | Diverged | No | Done |
| [S03](#s03) | 2026-04-09 | Qwen 3.5-27B | Spectral analysis | Free-α per layer | N/A | N/A | Done |
| [S04](#s04) | 2026-04-09 | Mistral-7B | Spectral analysis | Free-α per layer | N/A | N/A | Done |
| [S05](#s05) | 2026-04-09 | Gemma 4-31B | Spectral analysis | Free-α per layer | N/A | N/A | Done |
| [F01](#f01) | 2026-04-09 | Qwen 3.5-27B | Spectral fine-tune | No regularizer | 2,080 | No | Done |
| [F02](#f02) | 2026-04-10 | Gemma 4-31B | Spectral fine-tune | Rigid harmonic reg | 2,188 | No | Done |
| [F03](#f03) | 2026-04-10 | Gemma 4-31B | Spectral fine-tune | Soft-band harmonic reg | ? | ? | Running |
| [F04](#f04) | 2026-04-10 | Qwen 3.5-27B | Spectral fine-tune | Rigid harmonic reg (correct fractions) | ? | ? | Running |
| [B01](#b01) | 2026-04-09 | C. elegans | Structural connectome | Spectral analysis | N/A | N/A | Done |
| [B02](#b02) | 2026-04-10 | C. elegans | Functional dynamics | Calcium imaging SVD | N/A | N/A | Done |

## Spectral Analysis (S-series)

### S01: Small Model Training (Champion Run N)
- **Model:** GPT-2 16M (4 layers, 4 heads, 256 embed)
- **Corpus:** RAI (56 sources, 4.7M tokens)
- **Result:** PPL 93, coherent text
- **Key:** Data curriculum (C→G→D→A) + anti-collapse regularization
- **Lesson:** Data strategy > architecture tricks
- **Details:** [docs/experiments.md](../docs/experiments.md)

### S02: HarmonicGPT (the Double-Slit)
- **Model:** GPT-2 30M/124M with W = σ₁·Σk^{-α}·uₖvₖᵀ parameterization
- **Result:** Worked at 30M, **diverged at 124M** after step 1500-2000
- **Lesson:** φ-structure is emergent, not constrainable. Imposing the converged structure from initialization destroys the sequential process that produces it.

### S03: Qwen 3.5-27B Free-Alpha Analysis
- **Result:** 521 layers, 7 types, mean error 0.58% from F/L predictions
- **Key finding:** attn_o = 1/3 universal (α = 0.853)
- **Details:** [runs/free-alpha-analysis.json](../runs/free-alpha-analysis.json)

### S04: Mistral-7B Free-Alpha Analysis
- **Result:** 109 layers, attn_o = 0.845, all within 3.6%
- **Key finding:** Uses only even Lucas indices {L(2), L(4), L(6), L(8)}

### S05: Gemma 4-31B Free-Alpha Analysis
- **Result:** 411 layers, mixed sliding/full attention
- **Key finding:** attn_o_slide = 0.8524 (0.07% from 1/3 prediction)
- **Important:** Gemma uses DIFFERENT F/L fractions than Qwen (e.g. attn_q: 2/7 vs 5/4)
- **Details:** [runs/gemma4-free-alpha.json](../runs/gemma4-free-alpha.json)

## Fine-Tuning (F-series)

### F01: Qwen Spectral Fine-Tune — No Regularizer
- **Server:** 2 (port 14774)
- **Model:** Qwen 3.5-27B decomposed (adaptive k₀)
- **Corpus:** RAI-Qwen (3.9M train tokens)
- **Config:** `--mode per_mode --lr 1e-3 --max-steps 2000 --batch-size 1 --block-size 512 --grad-accum 16`
- **Regularizer:** None
- **Best PPL:** 2,080 (step ~950)
- **Final PPL:** 5,760
- **Coherent text:** No. Function words + numbers, never formed sentences.
- **Spectral drift:** ALL types collapsed toward α ≈ 0.2. attn_o: 0.853 → 0.197 (77% drift). Spectra locked in by step ~500 and stopped moving.
- **Lesson:** Without regularization, spectral-only fine-tuning destroys φ-structure. The consensus operator collapses and the model can't aggregate information.
- **Checkpoint analysis:** [scripts/analyze_spectral_checkpoint.py](../scripts/analyze_spectral_checkpoint.py)

### F02: Gemma Spectral Fine-Tune — Rigid Harmonic Regularizer
- **Server:** 1 (port 18409)
- **Model:** Gemma 4-31B decomposed (adaptive k₀, 26GB shards)
- **Corpus:** RAI-Gemma4 (3.9M train tokens)
- **Config:** `--harmonic-lambda 0.1 --type-aware-harmonic --attn-o-weight 10.0`
- **Regularizer:** Rigid pull to F/L targets (NO dead zone), using QWEN fractions (wrong for Gemma!)
- **Best PPL:** 2,188
- **Final PPL:** 2,785
- **Coherent text:** No. More function words than F01, some bigrams ("of the", "is to"), but no sentences.
- **Key issue:** Used Qwen's F/L assignments for Gemma. attn_q was being pulled 5.65σ in the wrong direction.
- **Lesson:** Model-specific fractions matter. Rigid regularizer with wrong fractions is worse than no regularizer.

### F03: Gemma Spectral Fine-Tune — Soft-Band Harmonic Regularizer *(running)*
- **Server:** 1 (port 18409)
- **Model:** Gemma 4-31B decomposed
- **Corpus:** RAI-Gemma4
- **Config:** `--harmonic-lambda 0.1 --type-aware-harmonic --attn-o-weight 10.0`
- **Regularizer:** Soft-band with Gemma-specific fractions. Dead zone within ±σ of target. Only penalizes drift beyond natural pre-trained variance.
- **Key differences from F02:**
  1. Correct Gemma fractions (attn_q: 2/7, not 5/4)
  2. Dead zone — layers free to vary within their natural σ
  3. model_name='gemma' passed for profile selection
- **Status:** Running, loading model

### F04: Qwen Spectral Fine-Tune — Rigid Harmonic Regularizer *(running)*
- **Server:** 2 (port 14774)
- **Model:** Qwen 3.5-27B decomposed
- **Corpus:** RAI-Qwen
- **Config:** `--harmonic-lambda 0.1 --type-aware-harmonic --attn-o-weight 10.0`
- **Regularizer:** Rigid pull to F/L targets, Qwen fractions (CORRECT for this model)
- **Purpose:** Control — does rigid regularizer work when fractions are correct?
- **Status:** Step ~680/2000, val PPL 4,607

## Biological (B-series)

### B01: C. elegans Structural Connectome
- **Data:** Varshney et al. 2011 (279 neurons, ~7000 synapses)
- **Analysis:** SVD of chemical + gap junction weight matrices
- **Key findings:**
  - Command interneurons: α = 1.177 ≈ φ^(1/3) (0.29% error) — same fraction as attn_o
  - All 20 top receiving modes dominated by command interneurons (8 neurons = 2.9% of network)
  - Gap junction energy thresholds: 90% at k/n = 0.237 ≈ 1/φ³ (0.3% error)
  - Neuron types cluster in SVD space (interneurons 0.46× baseline distance)
- **Details:** [scripts/celegans_spectral_analysis.py](../scripts/celegans_spectral_analysis.py), [scripts/celegans_phi_analysis.py](../scripts/celegans_phi_analysis.py), [scripts/celegans_deep_svd.py](../scripts/celegans_deep_svd.py)

### B02: C. elegans Functional Dynamics
- **Data:** Atanas & Kim, Cell 2023 / Flavell Lab (68 whole-brain calcium imaging recordings, ~150 neurons × 1600 timepoints each)
- **Source:** WormWideWeb / Zenodo (doi:10.5281/zenodo.8150514)
- **Analysis:** SVD of neuron-neuron correlation matrices from calcium traces
- **Key findings:**
  - α ≈ 1.576 ± 0.26 across 10 datasets, near φ itself (1.618, 2.6% error)
  - R² > 0.99 — bent power law fits functional dynamics better than structural connectivity
  - 90% of variance in first 3-5 modes (hyper-concentrated)
  - Functional α ≈ φ^(1/1) = φ — the fundamental, simplest F/L fraction
- **Details:** [runs/celegans-functional-spectral.json](../runs/celegans-functional-spectral.json)

## Analysis (A-series)

### A01: Energy Concentration Thresholds
- Cumulative variance crosses thresholds at φ-power fractions of total rank
- Gemma 4 global: 90% at k/n = 0.624 ≈ 1/φ (1.0% error)
- C. elegans gaps: 90% at k/n = 0.237 ≈ 1/φ³ (0.3% error)
- Theoretical: α = (1/φ)^(1/3) uniquely produces 90% energy at 1/φ
- **Details:** [scripts/energy_threshold_analysis.py](../scripts/energy_threshold_analysis.py), [scripts/alpha_energy_theory.py](../scripts/alpha_energy_theory.py)

### A02: Alternative Base Debunk (φ vs π)
- With arbitrary fractions a/b ≤ 20, 87% of random bases fit all 7 alphas within 1%
- Restricted to F/L fractions: φ mean error 0.41%, π 0.99% (2.4× better)
- The claim is F/L fractions (φ's convergents), not arbitrary rationals
- **Details:** [scripts/phi_vs_pi_debunk.py](../scripts/phi_vs_pi_debunk.py)

### A03: Spectral Quantization Prototype
- Quantize residuals from predicted φ-curve instead of raw weights
- 336× lower reconstruction error than naive 4-bit at 512×512
- Residual range: ±1.3% of σ₁ (vs 20:1 raw range)
- Scales better with dimension (500× at 2048×2048)
- **Details:** [scripts/spectral_quantize.py](../scripts/spectral_quantize.py)
