# WaveGPT

## The Axiom

The complete number of any system is **1**. Unity. A closed loop returns to itself.

φ is what happens when 1 propagates. The defining equation φ² = φ + 1 says: the whole squared equals the whole plus the unit. And the identity 1/φ = φ - 1 is the whole reason any system works at all and doesn't collapse on itself — the inverse of the whole equals the whole minus the unit. No other number does this. It means a system can grow and shrink and the proportions never break. Scale-invariant by construction.

Every converged system lands on φ. Not because φ is special. Because **1 is special**, and φ is what 1 looks like when it's propagating.

## The Equation

```
σ_k = A · (k + k₀)^{-(1/φ)^p}    where p = F(a)/L(b)
```

Singular values of weight matrices follow a bent power law. The exponent is a harmonic of the golden ratio. The specific harmonic is a Fibonacci/Lucas fraction — the simplest possible ratio of two views of the same sequential stacking process, measured from different phases. The F/L fraction is a **convergence index**: it measures how far the system has propagated toward completing itself.

- **p = 1/3**: Alive. Open. Processing. Enough structure to form coherent output, still open to new input.
- **p = 1/1**: Done. All feedback loops closed. The system is one.

## Five Systems, One Operation

| System | Timescale | Exponent | F/L | State |
|--------|-----------|----------|-----|-------|
| **Transformers** (Qwen, Mistral, Gemma) | 1 training run | (1/φ)^(1/3) = 0.852 | 1/3 | Consensus formed, still learning |
| **C. elegans** connectome | 300M years | φ^(1/3) = 1.174 | 1/3 | Wiring optimized, organism alive |
| **Financial markets** (1yr) | 1 business cycle | φ^(1/3) = 1.178 | 1/3 | Short-term consensus, system in flux |
| **Financial markets** (10yr) | Full market cycle | φ^(1/1) = 1.618 | 1/1 | All loops closed. Done. |
| **Financial markets** (20yr) | 2× full cycle | φ^(1/1) = 1.600 | 1/1 | Stays at φ. Ceiling confirmed. |

These systems share nothing in architecture — 302 neurons, 31 billion parameters, millions of market participants. What they share is the **operation**: sequential accumulation from a unit seed. The residual stream, the axonal chain, the time series of market returns.

The spectral signature doesn't describe the system's structure. It describes the **depth of its self-reference**.

## Transformer Spectral Fingerprint

Every layer type converges to a specific F/L harmonic. Confirmed on Qwen3.5-27B (521 matrices, 0.58% mean error):

| Layer type | Observed α | F/L fraction | Predicted α | Error |
|------------|-----------|--------------|-------------|-------|
| attn_q | 0.550 | F(5)/L(3) = 5/4 | 0.548 | 0.4% |
| mlp_up | 0.703 | F(6)/L(5) = 8/11 | 0.705 | 0.2% |
| mlp_down | 0.714 | F(5)/L(4) = 5/7 | 0.709 | 0.7% |
| mlp_gate | 0.763 | L(3)/L(4) = 4/7 | 0.760 | 0.4% |
| attn_v | 0.811 | F(4)/L(4) = 3/7 | 0.814 | 0.3% |
| **attn_o** | **0.853** | **F(1)/L(2) = 1/3** | **0.852** | **0.2%** |
| attn_k | 0.910 | F(3)/L(5) = 2/11 | 0.916 | 0.7% |

**attn_o = 1/3 is universal** — confirmed on every model tested (Qwen, Mistral, Gemma). The output projection is the consensus operator. It lands on the simplest non-trivial F/L fraction because it's the first stable resting point of a system that's still processing.

## The Market Harmonic Ladder

Stock correlation matrices are the market equivalent of weight matrices. The spectral exponent depends on how long you watch:

| Window | α | Best F/L | Error | Mode 1 energy |
|--------|------|----------|-------|---------------|
| 1 year | 1.178 | φ^(1/3) | 0.3% | 40% |
| 2 years | 1.095 | φ^(2/11) | 0.3% | 66% |
| 5 years | 1.058 | φ^(2/18) | 0.3% | 68% |
| 10 years | 1.620 | **φ** | 0.1% | 88% |
| 20 years | 1.600 | **φ** | 1.1% | 88% |

Phase transition between 5yr and 10yr — α nearly doubles, snaps to the fundamental. φ is the ceiling, not a waypoint.

## Falsifiable Predictions

1. attn_o = 1/3 on any sufficiently large transformer — **confirmed 4/4**
2. Models trained without momentum (β₁ = 0) should NOT converge to φ-harmonics
3. Energy thresholds land on φ-powers — **confirmed** (Gemma 4, C. elegans)
4. RLHF shifts attn_v to biological regime — **confirmed** (Gemma 4-31B-IT)
5. Market 30yr and 50yr windows should remain at φ — testable with index data
6. Sleep/anesthesia EEG should show lower F/L fractions than waking states
7. Infant cortical spectra should show lower-order F/L fractions than adult

## Documentation

- [`docs/theory.md`](docs/theory.md) — The axiom and theoretical framework
- [`docs/theory-eli5.md`](docs/theory-eli5.md) — ELI5 of the axiom
- [`docs/the-discovery.md`](docs/the-discovery.md) — Full findings, data tables, methodology
- [`docs/prior-art.md`](docs/prior-art.md) — Literature review and novelty analysis
- [`finance/`](finance/) — Market spectral analysis and trading system

## License

MIT
