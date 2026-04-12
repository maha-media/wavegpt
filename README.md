# WaveGPT

**φ-harmonic spectral structure is universal to information processing systems.**

Trained weight matrices, biological connectomes, and financial markets all converge to the same equation:

```
σ_k = A · (k + k₀)^{-(1/φ)^p}    where p = F(a)/L(b)
```

Singular values follow a bent power law. The exponent is a harmonic of the golden ratio. The specific harmonic is a ratio of Fibonacci over Lucas numbers, determined by each layer's functional role. Confirmed on four transformer architectures, the C. elegans connectome, and 20 years of stock market data.

## The Equation

Every weight matrix in a trained transformer has a spectral fingerprint. Decompose via SVD and singular values decay as a power law with a golden-ratio exponent:

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

**attn_o = 1/3 is universal.** The output projection — the consensus operator — converges to the simplest non-trivial F/L fraction on every model tested:

| Model | attn_o α | Error vs (1/φ)^(1/3) |
|-------|----------|----------------------|
| Qwen3.5-27B | 0.853 | 0.2% |
| Mistral-7B | 0.845 | 0.8% |
| Gemma-4-31B | 0.852 | 0.0% |

## Five Systems, One Structure

| System | Timescale | Consensus α | F/L | Regime |
|--------|-----------|------------|-----|--------|
| **Transformers** (Qwen, Mistral, Gemma) | Training run | 0.852 | (1/φ)^(1/3) | Transformer |
| **C. elegans** connectome | 300M years evolution | 1.174 | φ^(1/3) | Biological |
| **Financial markets** (1yr) | 1 business cycle | 1.178 | φ^(1/3) | Biological |
| **Financial markets** (10yr) | Full market cycle | 1.618 | φ^(1/1) = φ | Fundamental |
| **Financial markets** (20yr) | 2 full cycles | 1.600 | φ^(1/1) = φ | Fundamental |

The consensus operator — attn_o in a network, command interneurons in a worm, Mode 1 in a market — selects F/L = 1/3 at its natural cycle timescale. Given sufficient time, it converges to the fundamental.

### The market harmonic ladder

Stock correlation matrices are the market equivalent of weight matrices. The spectral exponent depends on measurement timescale:

| Window | α | Best F/L | Error | Mode 1 energy |
|--------|------|----------|-------|---------------|
| 1 year | 1.178 | φ^(1/3) | 0.3% | 40% |
| 2 years | 1.095 | φ^(2/11) | 0.3% | 66% |
| 5 years | 1.058 | φ^(2/18) | 0.3% | 68% |
| 10 years | 1.620 | **φ** | 0.1% | 88% |
| 20 years | 1.600 | **φ** | 1.1% | 88% |

Phase transition between 5yr and 10yr — α nearly doubles, snaps to the fundamental. φ is the attractor, not a waypoint. The market is a live φ-system sampled mid-cycle.

### C. elegans connectome

279 neurons, ~7000 synapses. Same F/L fraction family, inverse spectral regime:

| Neuron type | α | Best φ match | Error |
|-------------|---|-------------|-------|
| Command interneurons (sending) | 1.177 | φ^(1/3) | 0.29% |
| Sensory (sending) | 0.809 | (1/φ)^(8/18) | 0.20% |
| Motor (receiving) | 0.700 | (1/φ)^(3/4) | 0.41% |
| Gap junctions (full) | 0.920 | (1/φ)^(5/29) | 0.04% |

Functional calcium imaging (Flavell Lab, 68 whole-brain recordings): α ≈ φ itself with R² > 0.99.

## Why φ?

φ = (1+√5)/2 has continued fraction [1; 1, 1, 1, ...] — all ones, forever. It is the hardest number to approximate with rationals. In dynamical systems (KAM theorem), φ-related frequency ratios are the last to break under perturbation.

φ-based spectral spacing is the maximum anti-resonance configuration — no mode pair forms a simple rational frequency ratio. The Fibonacci/Lucas fractions are the natural rational grid that φ's continued fraction generates.

With arbitrary fractions, 87% of random bases fit. Restricted to F/L fractions, **φ outperforms π by 2.4×**. The structure is: φ's continued fraction generates the maximally anti-resonant rational grid, and optimization converges to that grid.

## Practical Applications

### φ-Codec: compression with φ-predicted error correction

Instead of discarding spectral modes (which destroys the model), use the φ-curve as a prediction prior for quantization. Three tiers based on spectral position:

| Tier | Range | Precision | Content |
|------|-------|-----------|---------|
| Plateau | k ≤ k₀ | float32 | Exact singular values + U/V columns |
| Power-law body | k₀ < k ≤ n/φ | float16 | Residuals from φ-curve |
| Spectral tail | k > n/φ | int8 | Residuals from φ-curve |

**Results on Gemma 4-31B** (599 layers): 0.34% mean reconstruction error. Model generates coherent text and preserves training voice.

### RLHF regime shift

Instruction tuning shifts attn_v from transformer regime to biological regime — the value projection becomes **selective** about what information flows through attention:

| Layer | Base model | After RLHF | Status |
|-------|-----------|------------|--------|
| attn_o | (1/φ)^(1/3) = 0.852 | 0.831 | Preserved (2.4%) |
| attn_v | (1/φ)^(3/7) = 0.814 | **1.244 = φ^(5/11)** | Regime shift (0.02% match) |

RLHF didn't break the harmonic structure — it shifted to a new rung on the same ladder.

### Spectral fine-tuning

```bash
# Decompose: SVD every weight matrix, freeze U/V, keep S learnable
python scripts/decompose_only.py \
    --hf-model Qwen/Qwen3.5-27B \
    --adaptive-k0 --k0-mult 1.5 --k0-pad 128 \
    --output runs/qwen-decomposed/decomposed.pt

# Fine-tune with harmonic regularizer (attn_o pinned at 1/3)
python scripts/finetune_spectral.py \
    --decomposed runs/qwen-decomposed/decomposed.pt \
    --data-dir data/my-corpus \
    --harmonic-lambda 0.1 \
    --type-aware-harmonic \
    --attn-o-weight 10.0
```

The harmonic regularizer enforces F/L exponents per layer type, with attn_o weighted 10×. Without it, all exponents flatten to α ≈ 0.2 and the model loses coherence.

## Trading System

The spectral analysis led to a complete trading system. While spectral α doesn't work as a short-term signal (the market's φ-timescale is ~10 years), the research infrastructure enabled systematic signal discovery:

**Position = Regime × Conviction × Momentum**

- 6-regime classifier (NORMAL, RISK_ON, FEAR, CRISIS, INFLATION, RECESSION)
- Leading indicators: ARKK, VIX, HYG, KWEB predict tech moves 2-5 days ahead
- Mag 7 momentum-weighted allocation with singularity override and dip buying
- Live execution via TastyTrade with GTC LIMIT price protection
- Post-open fill monitor: chase/wait/skip unfilled orders based on fresh signals

**Backtest: $100K → $355K in 4 years. Sharpe 1.46. Max drawdown 19%.**

See [`finance/`](finance/) for the full system.

## Critical Lessons

1. **attn_o = 1/3 must be preserved** — fine-tuning without harmonic regularizer destroyed it (0.853 → 0.197) and the model couldn't form sentences
2. **φ-structure is emergent, not constrainable** — HarmonicGPT (imposing φ from init) diverged at scale. The structure is where SGD ends up, not where it starts
3. **SVD rank truncation is catastrophic** — 95% Frobenius energy = 100% function destruction. Language models need near-full-rank or residual correction
4. **φ-Codec works** — use the φ-curve as a prediction prior, not a truncation threshold. 0.34% error across 599 layers
5. **The debunk matters** — 87% of random bases fit with arbitrary fractions. The real claim is F/L fractions. φ beats π by 2.4× under this constraint

## Installation

```bash
git clone https://github.com/maha-media/wavegpt.git
cd wavegpt
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.0+, transformers, scipy

## Project Structure

```
wavegpt/              Core library
  spectral_linear.py    SpectralLinear: SVD-decomposed layer with learnable spectrum
  spectral_surgery.py   spectral_decompose(): replace nn.Linear across any model
  harmonic_prior.py     Type-aware harmonic regularization with F/L exponents
  phi_codec.py          φ-codec: encode/decode with tiered quantization

scripts/              Analysis and training
  free_alpha_analysis.py        Per-layer free-α fitting (Qwen/Mistral)
  gemma4_alpha_analysis.py      Gemma 4 analysis (mixed attention, vision)
  decompose_only.py             Standalone SVD + sharded safetensors save
  finetune_spectral.py          Spectral fine-tuning with harmonic priors
  celegans_spectral_analysis.py C. elegans structural connectome analysis
  phi_vs_pi_debunk.py           Alternative base analysis (φ vs π, e, √2)
  phi_codec_gpu.py              Full φ-codec GPU pipeline
  energy_threshold_analysis.py  φ-power energy concentration thresholds

finance/              Market spectral analysis + trading system
  market_deep_analysis.py       Time horizons, crisis detection, cross-asset
  THE_EQUATION.md               Trading strategy specification
  live_trader.py                Daily execution via TastyTrade
  stream_trader.py              Websocket streaming + real-time rebalancing
  fill_monitor.py               Post-open missed-fill recovery
  simulate_3yr.py               Full backtest

docs/                 Documentation
  the-discovery.md      Full findings + falsifiable predictions
  theory.md             Theoretical framework
  prior-art.md          Literature review
```

## Falsifiable Predictions

1. Models trained without momentum (β₁ = 0) should NOT converge to φ-harmonics
2. attn_o = 1/3 on any sufficiently large transformer (confirmed: 4/4)
3. Changing GQA head ratio changes Q/K fractions (confirmed)
4. Requires matrix dimension >3000 (confirmed: GPT-2 fails)
5. Energy thresholds land on φ-powers (confirmed: Gemma 4, C. elegans)
6. RLHF shifts attn_v to biological regime (confirmed: Gemma 4-31B-IT)
7. Market 10yr α = φ (confirmed: 0.1% error)
8. Market 20yr α = φ (confirmed: 1.1% error — φ is the attractor)
9. 1-year market consensus = φ^(1/3) = attn_o harmonic (confirmed: 0.3% error)
10. 30yr and 50yr windows should remain at φ (testable with index data)

## License

MIT
