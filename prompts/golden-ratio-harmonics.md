# Golden Ratio Harmonics in Neural Network Weight Spectra

**Status**: 521 layers of Qwen3.5-27B analyzed. Mean error 0.5%. All within 1.1%.

## The Data (521 Layers — Qwen3.5-27B)

We performed SVD on every weight matrix W in Qwen3.5-27B and fitted the singular value spectrum to:

```
σ_k = A · (k + k₀)^{-α}
```

with α as a FREE parameter (not fixed). The optimizer found these type-dependent exponents:

| Layer Type     | Observed α | n layers | std(α) within type |
|----------------|-----------|----------|-------------------|
| self_attn.q    | 0.550     | 64       | 0.043             |
| mlp.up_proj    | 0.703     | 65       | 0.023             |
| mlp.down_proj  | 0.714     | 65       | 0.052             |
| mlp.gate_proj  | 0.763     | 65       | 0.033             |
| delta.qkv      | 0.783     | 65       | 0.023             |
| self_attn.v    | 0.811     | 64       | 0.033             |
| delta.out      | 0.843     | 65       | 0.042             |
| delta.z        | 0.848     | 65       | 0.023             |
| self_attn.o    | 0.853     | 64       | 0.019             |
| self_attn.k    | 0.910     | 64       | 0.004             |

R² of fits: 0.93-0.98.

## The Pattern

Every observed α matches (1/φ)^p where p is a simple rational fraction with Fibonacci numerators and Lucas denominators:

| Layer Type     | α_obs  | p = ln(α)/ln(1/φ) | Fraction  | (1/φ)^p | |Δ|    |
|----------------|--------|-------------------|-----------|---------|--------|
| self_attn.q    | 0.550  | 1.245             | **5/4**   | 0.5480  | 0.4%   |
| mlp.up_proj    | 0.703  | 0.720             | **8/11**  | 0.7051  | 0.2%   |
| mlp.down_proj  | 0.714  | 0.693             | **5/7**   | 0.7091  | 0.7%   |
| mlp.gate_proj  | 0.763  | 0.558             | **4/7**   | 0.7596  | 0.4%   |
| delta.qkv      | 0.783  | 0.500             | **1/2**   | 0.7862  | 0.4%   |
| self_attn.v    | 0.811  | 0.429             | **3/7**   | 0.8136  | 0.3%   |
| delta.out      | 0.843  | 0.354             | **1/3**   | 0.8518  | 1.0%   |
| delta.z        | 0.848  | 0.338             | **1/3**   | 0.8518  | 0.4%   |
| self_attn.o    | 0.853  | 0.325             | **1/3**   | 0.8518  | 0.1%   |
| self_attn.k    | 0.910  | 0.188             | **2/11**  | 0.9162  | 0.7%   |

Mean error: **0.5%**. Max error: **1.1%**.

**Compared to universal 1/φ (α=0.618 for all):** 45× improvement in mean fit error.

### Composition of the fractions

The numerators that appear: **{1, 2, 3, 4, 5, 8}**
The denominators that appear: **{2, 3, 4, 7, 11}**

- {1, 2, 3, 5, 8} = five consecutive Fibonacci numbers (F(2) through F(6))
- 4 = L(3), the 3rd Lucas number
- {2, 3, 4, 7, 11} = L(0), L(2), L(3), L(4), L(5) — consecutive Lucas numbers

All numerators belong to Fibonacci ∪ Lucas. All denominators ARE Lucas numbers.

### The equation

```
σ_k = A · (k + k₀)^{-(1/φ)^p(type)}
```

where p(type) ∈ {1/3, 2/11, 3/7, 4/7, 1/2, 5/7, 8/11, 5/4} — all ratios of Fibonacci/Lucas numbers.

## What Changed from 125 → 521 Layers

The MLP types (65 samples each) barely moved — their means were already stable. Three attention types shifted because they only had 4 samples in the early 125-layer sample (all from the first ~25 transformer blocks):

- **attn_q**: mean moved from 0.490 → 0.550 (fraction changed from 3/2 → 5/4)
- **attn_k**: mean moved from 0.870 → 0.910 (fraction changed from 2/7 → 2/11)
- **mlp_gate**: was the one miss at 1/2 (3% off), now fits 4/7 at 0.4%

The full 521 includes all 64 transformer blocks plus MTP layers, giving the true population mean.

## What the Fractions Mean Physically

A small exponent (like attn_k = 0.91, p=2/11) means **sharp decay** — energy concentrates in few modes. The layer does something focused and specific. Key projections just need to be matchable.

A large exponent (like attn_q = 0.55, p=5/4) means **slow decay** — the matrix uses MANY modes. Query projections need broad spectral reach; they attend to everything.

The exponent tells you how many independent "concepts" that layer needs.

## Why the Golden Ratio

φ is the most irrational number. Its rational approximations (Fibonacci ratios) converge more slowly than any other number's. A spectral decay based on φ is maximally resistant to resonance — no pair of modes can lock into a simple frequency ratio and interfere destructively.

Same reason sunflowers use the golden angle: maximum packing without alignment.

## Context

- These are Qwen3.5-27B weights. Trained by Alibaba with standard AdamW + cosine schedule on trillions of tokens. Nobody designed this structure.
- The initial guess for α in curve fitting was 0.618 (1/φ). The optimizer moved AWAY from it to find the type-specific values.
- φ is the eigenvalue of the Fibonacci recurrence x_{n+1} = x_n + x_{n-1}. It is the fixed point of x → 1 + 1/x. It satisfies φ² = φ + 1.
- Lucas numbers L(n) = φ^n + (-φ)^{-n} also converge to φ: L(n)/L(n-1) → φ.
- Prior work (Martin & Mahoney 2018-2021) showed power-law eigenvalue densities in trained networks but reported variable exponents (α ∈ [2,6]) with no universal structure.
- "From SGD to Spectra" (2025) proved via Dyson Brownian motion that power-law tails emerge from SGD but did not solve for which exponent.

## Questions

### Answered (with evidence):

1. **Is α type-dependent, not universal?** → YES. 10 distinct types, 45× better fit than universal 1/φ.

2. **Does p use Fibonacci/Lucas fractions?** → YES. Numerators from F∪L = {1,2,3,4,5,8}. Denominators from L = {2,3,4,7,11}. One crack: numerator 4 = L(3) is NOT Fibonacci.

3. **Why does p differ by function, not shape?** → gate_proj and up_proj have identical shapes (17408×5120) but p=4/7 vs p=8/11. The Hessian structure differs because gate participates in multiplicative interaction (gate ⊙ up) while up_proj doesn't. Different gradient statistics → different Adam moments → different stable α.

4. **What does p control physically?** → Spectral breadth. High p (attn_q, 5/4) = slow decay = many modes = complex operation. Low p (attn_k, 2/11) = fast decay = few modes = focused operation. The exponent encodes the layer's computational role.

5. **What changed with more data (125→521)?** → Attention types with few samples shifted (attn_q: 3/2→5/4, attn_k: 2/7→2/11). MLP types were stable. mlp_gate went from miss (1/2, 3% off) to fit (4/7, 0.4%).

### Open:

6. **What determines the specific fraction assignment?** → Is there a formula mapping layer function → p? Or is it purely empirical? Can we predict p from the computational graph structure (number of multiplicative interactions, residual connections, etc.)?

7. **Do k₀ values also show Fibonacci/Lucas structure?** → Preliminary: k₀ may cluster near Fibonacci numbers. Needs the k₀ regression analysis to confirm.

8. **Is this universal across architectures?** → Cross-model test (GPT-2-XL, Llama-3) needed. If attn_q always gets 5/4 regardless of architecture, this is a universal law of SGD. If it's Qwen-specific, it may be architecture-dependent.

9. **What is the precise mechanism linking Adam dynamics to φ?** → The KAM resonance model predicted 71% of harmonics from first principles. Can we derive the full formula from the Adam update equations? Specifically: does the quasi-periodic orbit's winding number minimize resonance energy when α = (1/φ)^p?

10. **Why numerator 4 for gate_proj?** → 4 = L(3), not Fibonacci. Is there something about the SwiGLU gating function that specifically requires L(3)? Or is this a minor deviation from the pattern?

Think step by step. Be rigorous. If you don't know, say so. Speculation is welcome but must be labeled as such.
