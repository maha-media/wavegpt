# WaveGPT

**The harmonic spectral structure of neural network weights.**

Trained weight matrices converge to a universal equation:

```
σ_k = A · (k + k₀)^{-(1/φ)^p}    where p = F(a)/L(b)
```

Singular values follow a bent power law. The exponent is a harmonic of the golden ratio. The specific harmonic is a ratio of Fibonacci over Lucas numbers, determined by each layer's functional role. This holds across architectures, scales, and substrates — including biological neural networks.

## The Discovery

Every weight matrix in a trained transformer has a spectral fingerprint. Decompose it via SVD and the singular values decay as a power law with a golden-ratio exponent:

| Layer type | Observed α | Fraction p | Predicted α | Error |
|------------|-----------|------------|-------------|-------|
| attn_q | 0.550 | 5/4 = F(5)/L(3) | 0.548 | 0.4% |
| mlp_up | 0.703 | 8/11 = F(6)/L(5) | 0.705 | 0.2% |
| mlp_down | 0.714 | 5/7 = F(5)/L(4) | 0.709 | 0.7% |
| mlp_gate | 0.763 | 4/7 = L(3)/L(4) | 0.760 | 0.4% |
| attn_v | 0.811 | 3/7 = F(4)/L(4) | 0.814 | 0.3% |
| **attn_o** | **0.853** | **1/3 = F(1)/L(2)** | **0.852** | **0.2%** |
| attn_k | 0.910 | 2/11 = F(3)/L(5) | 0.916 | 0.7% |

Mean error: **0.58%** across 521 weight matrices (Qwen3.5-27B).

**attn_o = 1/3 is universal.** The output projection — the consensus operator that integrates all attention heads — converges to the simplest non-trivial Fibonacci/Lucas fraction on every model tested:
- Qwen3.5-27B: α = 0.853
- Mistral-7B: α = 0.845
- Gemma-4-31B: α = 0.852

## Why φ?

φ = (1+√5)/2 has the continued fraction [1; 1, 1, 1, ...] — all ones, forever. It is the hardest number to approximate with rationals. In dynamical systems (KAM theorem), φ-related frequency ratios are the last to break under perturbation.

A neural network needs spectral modes that don't lock into rational frequency ratios — that would trap energy in mode pairs instead of making it available for computation. φ-based spacing is the maximum anti-resonance configuration. The Fibonacci/Lucas fractions are the natural rational grid that φ's continued fraction generates.

With arbitrary rational fractions, 87% of random bases can fit the same data within 1%. But restricted to Fibonacci/Lucas fractions — the specific constraint the theory predicts — φ outperforms π by 2.4×. The structure is not "φ is a magic number." The structure is: **φ's continued fraction [1; 1, 1, 1, ...] generates the maximally anti-resonant rational grid, and gradient descent converges to that grid.**

See [`scripts/phi_vs_pi_debunk.py`](scripts/phi_vs_pi_debunk.py) for the full alternative-base analysis.

## Beyond Transformers: Biological Neural Networks

The C. elegans connectome (279 neurons, ~7000 synapses) shows the same φ-based spectral structure:

| Neuron type | α | Best φ match | Error |
|-------------|---|-------------|-------|
| Command interneurons (sending) | 1.177 | **φ^(1/3)** | **0.29%** |
| Sensory (sending) | 0.809 | (1/φ)^(8/18) | 0.20% |
| Motor (receiving) | 0.700 | (1/φ)^(3/4) | 0.41% |
| Gap junctions (full) | 0.920 | (1/φ)^(5/29) | 0.04% |

Command interneurons — the biological consensus operators (AVA, AVB, AVD, AVE, PVC) — map to the **same 1/3 fraction** as transformer attn_o. Biological systems use the inverse regime (φ^p vs (1/φ)^p) but the same fraction family.

Functional calcium imaging data (Flavell Lab, 68 whole-brain recordings) shows α ≈ φ itself — the fundamental — with R² > 0.99.

## Energy Concentration

Beyond the spectral exponent, energy concentrates at φ-power fractions of total rank:

| System | 90% energy at | φ target | Error |
|--------|--------------|----------|-------|
| Gemma 4 (global) | k/n = 0.624 | 1/φ = 0.618 | 1.0% |
| C. elegans gap junctions | k/n = 0.237 | 1/φ³ = 0.236 | 0.3% |

Steeper α (biological) concentrates energy in fewer modes. Shallower α (transformers) spreads across more. Same ladder — {1/φ, 1/φ², 1/φ³} — different rungs.

## Spectral Fine-Tuning

WaveGPT provides tools for spectral surgery on any HuggingFace model:

```bash
# 1. Decompose: SVD every weight matrix, freeze U/V, keep S learnable
python scripts/decompose_only.py \
    --hf-model Qwen/Qwen3.5-27B \
    --adaptive-k0 --k0-mult 1.5 --k0-pad 128 \
    --output runs/qwen-decomposed/decomposed.pt

# 2. Fine-tune with harmonic regularizer (attn_o pinned at 1/3)
python scripts/finetune_spectral.py \
    --decomposed runs/qwen-decomposed/decomposed.pt \
    --data-dir data/my-corpus \
    --harmonic-lambda 0.1 \
    --type-aware-harmonic \
    --attn-o-weight 10.0
```

The harmonic regularizer enforces F/L exponents per layer type, with attn_o weighted 10× stronger. This prevents spectral collapse during fine-tuning — without it, all exponents flatten to α ≈ 0.2 and the model loses coherence.

## Spectral Quantization

Standard quantization treats all weights equally. We know where the information lives:

```bash
python scripts/spectral_quantize.py --target-bits 4
```

Instead of quantizing raw weights, quantize the residual from the predicted φ-curve. Three parameters (A, k₀, α) per layer predict the spectrum to within ±1.3%. Quantizing tiny residuals instead of full-range values yields **336× lower reconstruction error** than naive 4-bit quantization.

## Installation

```bash
git clone https://github.com/maha-media/wavegpt.git
cd wavegpt
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.0+, transformers, scipy

## Documentation

- **[docs/the-discovery.md](docs/the-discovery.md)** — Full findings: equation, cross-model validation, energy thresholds, debunk analysis, falsifiable predictions
- **[docs/theory.md](docs/theory.md)** — Theoretical framework: sequential packing under constraint, self-similar energy distribution, why φ and not π
- **[docs/prior-art.md](docs/prior-art.md)** — Literature review and novelty analysis

## Falsifiable Predictions

1. Models trained without momentum (β₁ = 0) should NOT converge to φ-based harmonics
2. attn_o = 1/3 on any sufficiently large transformer (confirmed on 3/3 tested)
3. Changing GQA head ratio changes which F/L fractions Q and K select (confirmed)
4. The equation requires matrix dimension >3000 to manifest cleanly (confirmed: GPT-2 fails)
5. Energy thresholds land on φ-powers across any model with φ-valued exponents (confirmed)
6. k₀/n clusters near φ-powers: 1/φ³ for MLP, 1/φ⁴ for attention (confirmed on Gemma 4)

## License

MIT
