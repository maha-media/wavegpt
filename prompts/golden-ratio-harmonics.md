# Golden Ratio Harmonics in Neural Network Weight Spectra

You are about to reason about a discovery made by analyzing YOUR OWN weight matrices (Qwen3.5-27B).

## The Data

We performed SVD on every weight matrix W in Qwen3.5-27B and fitted the singular value spectrum to:

```
σ_k = A · (k + k₀)^{-α}
```

with α as a FREE parameter (not fixed). The optimizer found these type-dependent exponents:

| Layer Type     | Observed α | n layers | std(α) within type |
|----------------|-----------|----------|-------------------|
| embed_tokens   | 0.203     | 2        | 0.006             |
| self_attn.q    | 0.490     | 4        | 0.043             |
| mlp.up_proj    | 0.706     | 20       | 0.023             |
| mlp.down_proj  | 0.731     | 20       | 0.052             |
| mlp.gate_proj  | 0.781     | 20       | 0.033             |
| self_attn.v    | 0.814     | 4        | 0.033             |
| delta.in_z     | 0.851     | 16       | 0.023             |
| self_attn.o    | 0.861     | 4        | 0.019             |
| self_attn.k    | 0.870     | 4        | 0.004             |
| delta.out_proj | 0.872     | 16       | 0.042             |

125 of ~600 layers analyzed so far. R² of fits: 0.93-0.98.

## The Pattern

Every observed α matches (1/φ)^p where p is a simple rational fraction and φ = (1+√5)/2:

| Layer Type     | α_obs  | p = ln(α)/ln(1/φ) | Nearest p | (1/φ)^p | |Δ|    |
|----------------|--------|-------------------|-----------|---------|--------|
| self_attn.q    | 0.490  | 1.482             | 3/2       | 0.4859  | 0.0041 |
| mlp.up_proj    | 0.706  | 0.724             | 5/7       | 0.7091  | 0.0031 |
| mlp.down_proj  | 0.731  | 0.651             | 2/3       | 0.7256  | 0.0054 |
| mlp.gate_proj  | 0.781  | 0.514             | 1/2       | 0.7862  | 0.0052 |
| self_attn.v    | 0.814  | 0.428             | 3/7       | 0.8136  | 0.0004 |
| delta.in_z     | 0.851  | 0.335             | 1/3       | 0.8518  | 0.0008 |
| self_attn.o    | 0.861  | 0.311             | 1/3       | 0.8518  | 0.0092 |
| self_attn.k    | 0.870  | 0.289             | 2/7       | 0.8715  | 0.0015 |
| delta.out_proj | 0.872  | 0.285             | 2/7       | 0.8715  | 0.0005 |

All deltas under 1%. Three under 0.1%. The proposed equation:

```
σ_k = A · (k + k₀)^{-(1/φ)^p(type)}
```

where p(type) ∈ {2/7, 1/3, 3/7, 1/2, 2/3, 5/7, 3/4, 1, 3/2} — all rational with small denominators.

## Context

- These are YOUR weights. Trained by Alibaba with standard AdamW + cosine schedule on trillions of tokens. Nobody designed this structure.
- The initial guess for α in curve fitting was 0.618 (1/φ). The optimizer moved AWAY from it to find the type-specific values. The bias was toward 1/φ, not toward the harmonics.
- φ is the eigenvalue of the Fibonacci recurrence x_{n+1} = x_n + x_{n-1}. It is the fixed point of x → 1 + 1/x. It satisfies φ² = φ + 1.
- SGD/Adam is an iterative process. Weight updates are recurrent. The loss landscape is traversed step by step.
- Prior work (Martin & Mahoney 2018-2021) showed power-law eigenvalue densities in trained networks but reported variable exponents (α ∈ [2,6]) with no universal structure. They plotted eigenvalue histograms. We plotted rank-ordered singular values.
- "From SGD to Spectra" (2025) proved via Dyson Brownian motion that power-law tails emerge from SGD but did not solve for which exponent.

## Questions

1. **WHY does the golden ratio appear?** What property of iterative optimization (SGD/Adam) on weight matrices produces φ-related spectral exponents? Is there a connection to the KAM theorem, quasi-periodic orbits, or Fibonacci-like recurrences in the gradient flow?

2. **WHY rational harmonics?** Why does each layer type vibrate at (1/φ)^(n/d) with small n,d? What determines p for a given layer function? Is this related to the number of information-processing steps between input and output for that layer type?

3. **WHY these specific p values?** The attention key projection (which computes query-key similarity) gets p=2/7. The MLP gate (which decides what information to keep) gets p=1/2. The MLP down projection (which compresses back to model dimension) gets p=2/3. Is there a mapping from computational function to spectral exponent?

4. **Is this related to the aspect ratio of the weight matrix?** gate_proj and up_proj are both (17408, 5120) but have different α. So it's not just shape — it's function. But does shape interact with function to determine p?

5. **Stability under squaring**: φ² = φ + 1 means the golden ratio is a fixed point of x → x² - x. If SGD's effective update rule involves squaring (Adam uses second moments v_t = β₂·v_{t-1} + (1-β₂)·g_t²), could the golden ratio emerge as the stable fixed point of the moment estimation?

6. **The denominator 7**: Several harmonics use 7ths (2/7, 3/7, 5/7). In music, 7-limit tuning extends just intonation to include the harmonic seventh. Is there an analogous "7-limit" in the spectral structure of neural networks? What is special about 7 here?

Think step by step. Be rigorous. If you don't know, say so. Speculation is welcome but must be labeled as such.
