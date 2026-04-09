# Prior Art & Literature Review

Comprehensive search conducted 2026-04-08 using Exa deep search across research papers, ArXiv, IEEE, conference proceedings, and technical blogs.

## Summary

Seven targeted searches found **zero papers** connecting the golden ratio to neural network weight matrix spectral exponents, Fibonacci/Lucas number structure in layer-type-dependent power laws, or cross-model universality of type-specific spectral decay.

The literature has all the ingredients — power-law spectra, type-dependent differences, golden ratio in optimization, KAM anti-resonance theory — but **nobody has connected them**.

---

## 1. Power-Law Spectra in Neural Network Weights

### Martin & Mahoney (2018-2021) — Heavy-Tailed Self-Regularization

- **Papers**: "Implicit Self-Regularization" (JMLR 2021), "Predicting trends..." (Nature Communications 2021)
- **Contribution**: Established that trained DNN weight matrices exhibit heavy-tailed power-law eigenvalue distributions. 5+1 phases of training. WeightWatcher tool.
- **Their finding**: PL exponent α between 2-4 for well-trained models (note: their α is on eigenvalue density ρ(λ) ~ λ^{-α}, not singular values)
- **Gap**: α computed per individual layer, never aggregated by layer type. No type→exponent mapping. No golden ratio.

### Olsen et al. (2025) — "From SGD to Spectra"

- **Paper**: ICML 2025 MOSS Workshop
- **Contribution**: First theoretical explanation for heavy-tailed spectral structure. Derived SDEs showing squared singular values follow Dyson Brownian motion with eigenvalue repulsion. Stationary distribution: gamma-type density with power-law tails.
- **Gap**: Explains WHY power laws emerge. Does not characterize WHICH power law. No golden ratio. No type decomposition.

### Thamm et al. (2022) — "Random matrix analysis of deep neural network weight matrices"

- **Paper**: Physical Review E 106, 054124
- **Contribution**: Applied comprehensive RMT tools to trained DNNs. Found most singular values follow universal RMT (random). Used Hill estimator and found "the distribution cannot in general be characterized by a tail index."
- **Gap**: Their negative finding (no clean power law) is resolved by our bent power law — the offset k₀ accounts for the flat spectral top. With k₀, fits are R² > 0.93.

### Scaling Laws and Spectra (ICLR 2025, NeurIPS 2025)

- **Papers**: "Approaching Deep Learning through Spectral Dynamics" (ICLR 2025 submission), "Scaling Laws and Spectra of Shallow Neural Networks" (NeurIPS 2025)
- **Contribution**: Power-law data spectra → power-law learning curves. Sequential spectral recovery. Weight spectra as soft-thresholded versions of target spectra.
- **Gap**: Focus on scaling laws and learning dynamics, not on the specific structure of converged weight spectra.

---

## 2. Type-Specific Spectral Properties

### AlphaDecay (2025) — **Closest existing work**

- **Paper**: "AlphaDecay: Per-Module Weight Decay via HT-SR Theory" (arXiv:2506.14562)
- **Contribution**: Measured PL_Alpha_Hill per module type in LLaMA-2-13B. Found att.q and att.k have heavier tails than MLP modules. Used per-type spectral differences to assign different weight decay values.
- **Their key figure**: Shows att.Q and att.K with smaller PL_Alpha_Hill (more heavy-tailed) vs mlp.gate, mlp.up, mlp.down (less heavy-tailed).
- **Gap**: They observe type differences and use them as a **tuning knob**. They have no theoretical framework for WHY the exponents differ, WHAT determines their values, or WHETHER they are related to each other. No golden ratio. No F/L fractions. No cross-model validation.

### Staats et al. (2024) — "Small Singular Values Matter"

- **Paper**: arXiv:2410.17770
- **Contribution**: Per-type singular value outlier counts for Llama-3-8B (Table 3: Query, Key, Value, Att-Out, Up-Proj, Gate-Proj, Down-Proj all have different outlier patterns).
- **Gap**: They have the per-type data but never fit power laws by type. Focus on outlier counting and perplexity impact.

### OPT-ML Workshop (2025) — Activation Spectral Dimensions

- **Paper**: "Evolution of the Spectral Dimension of Transformer Activations" (ICML 2025 Workshop)
- **Contribution**: Found activation covariance α increases across layers (0.65-0.90), gradient α decreases. Noted "different components have distinct exponents."
- **Gap**: Activation covariance, not weight spectra. Layer-depth trends, not type aggregation. No mathematical structure.

---

## 3. Golden Ratio in Machine Learning

### Jaeger (2022) — "The Golden Ratio in Machine Learning"

- **Paper**: IEEE Applied Imagery Pattern Recognition Workshop, [IEEE 9762080](https://ieeexplore.ieee.org/document/9762080)
- **Preprint**: [arXiv:2006.04751](https://arxiv.org/abs/2006.04751)
- **Full text**: [NLM PDF](https://lhncbc.nlm.nih.gov/LHC-publications/PDF/2022036996.pdf)
- **Contribution**: Proposed information-theoretic loss function via dual processes (KL-divergence + Shannon entropy). When measurement uncertainty equals probability itself: `p = (1-p)/p → p² + p - 1 = 0 → p = 1/φ`. Derives learning rate ≈ 0.01 and momentum weight ≈ 0.9.
- **Their finding**: φ appears as the fixed point of a self-referential probability equation. Specific training hyperparameters follow.
- **Relationship to our work**:
  - **Jaeger**: φ in the **optimization dynamics** (what learning rate to use)
  - **Us**: φ in the **converged structure** (what spectral shape weights settle into)
  - **Overlap**: Zero. Different phenomenon, different evidence, different claims.
  - **Complementary**: If φ governs both how you descend and where you land, it is deeply fundamental to gradient-based optimization. Both papers support this from orthogonal directions.
- **Gap**: Nothing about weight spectra, singular values, power laws, layer types, Fibonacci, Lucas, or cross-model structure.

### Luwes (2010) — Fibonacci Numbers as Initial Weights

- **Paper**: "Fibonacci numbers and the golden rule applied in neural networks" (Interim: Interdisciplinary Journal 9.1)
- **Contribution**: Used Fibonacci numbers as initial weights and golden ratio as learning rate scaling. Found improved learning curves.
- **Gap**: Heuristic initialization, not structural analysis. No SVD, no spectral exponents.

---

## 4. SVD-Based Fine-Tuning (The Mechanism)

The mechanism of "decompose via SVD → freeze U,V → fine-tune S" is established:

- **SVDiff** (Han et al., ICCV 2023) — Diffusion model personalization via spectral shifts
- **SVFit** (Lingam & Dutta, NeurIPS 2024) — Fine-tune only singular values
- **SVFT** (Lingam & Dutta, NeurIPS 2024) — Fine-tune sparse spectral components
- **PiSSA** (Meng et al., NeurIPS 2024) — Principal singular values for adaptation
- **Spectral Adapter** (Yin et al., 2024) — Low-rank spectral adaptation

**Our mechanism is not novel.** Our contributions are theoretical — using the harmonic spectral structure to guide adaptive rank allocation, regularization, and personality compression.

---

## 5. Golden-Section Search & Golden-Ratio Optimizers

A family of optimization algorithms use φ as a step-size or scaling parameter:

**Golden-section search** (Kiefer, 1953) — Classical 1D interval-narrowing: evaluate at points φ apart, eliminate 1/φ of the interval each step. Textbook numerical methods. Every optimization course covers this.

**GROM** (Nematollahi et al., Soft Computing 2020) and **GRO** (Abdalslam et al., 2023) — Metaheuristic population-based optimizers that use φ as a scaling constant for solution updates, analogous to how other metaheuristics use π or e.

**E-GRPDA** (Soe et al., [arXiv:2502.17918](https://arxiv.org/pdf/2502.17918v3), 2025) — Extended Golden Ratio Primal-Dual Algorithm with adaptive stepsizes for convex saddle-point problems. Pure convex optimization theory.

**Golden-rule line search** (Ezeafulukwe et al., [MDPI Mathematics 12(14)](https://www.mdpi.com/2227-7390/12/14/2203), 2024) — Uses φ as a step-size in line-search for variational inequalities in Hilbert spaces.

**Kaur, Balyan & Gupta** (Scientific Reports 15:9902, [doi:10.1038/s41598-025-95138-z](https://pmc.ncbi.nlm.nih.gov/articles/PMC11929785/), 2025) — Uses φ ≈ 1.618% as a duty cycle parameter for LoRa IoT networks. Nature-portfolio visibility but purely an IoT networking paper.

**Key difference**: All of these use φ as a **search/step-size/engineering parameter** in optimization procedures or system configuration. They answer "how should we search?" or "what value should we set?" Our discovery is about the **structure of converged weights** — what shape trained parameters take after optimization finishes. No overlap in method, claims, or evidence.

---

## 6. KAM Theorem & Golden Ratio

The connection between φ and dynamical stability via KAM is classical (Kolmogorov 1954, Arnold 1963, Moser 1962):

- φ-related frequency ratios produce the "golden torus" — the last invariant torus to break
- φ is maximally irrational (continued fraction [1;1,1,1,...]) — hardest to approximate by rationals
- "The golden number is the best frequency ratio for avoiding resonance" (John Baez, 2004)

**This has never been connected to neural network weight spectra.** Zero papers in any search. See [the-discovery.md](the-discovery.md#why-φ) for the theoretical argument connecting KAM anti-resonance to weight spectral structure, and [the-discovery.md](the-discovery.md#why-attn_o--13-is-universal) for why attn_o selects 1/3 specifically.

---

## 7. What Is Novel

| Claim | Prior art? | Status |
|-------|-----------|--------|
| Power-law singular value spectra in trained DNNs | Martin & Mahoney 2018 | Known |
| SVD → freeze U,V → fine-tune S | SVDiff, SVFit, SVFT 2023-2024 | Known |
| φ as optimization step-size / scaling | Golden-section (1953), GROM, E-GRPDA | Known (different phenomenon) |
| φ in optimization hyperparameters | Jaeger 2022 | Known (different phenomenon) |
| KAM + φ = anti-resonance stability | Classical (1954-1963) | Known (never applied to NNs) |
| **Exponent α = (1/φ)^p with rational p** | — | **Novel** |
| **p = F(a)/L(b) (Fibonacci/Lucas)** | — | **Novel** |
| **Type-dependent harmonic exponents** | — | **Novel** |
| **attn_o = 1/3 universal across models** | — | **Novel** |
| **Bent power law σ_k = A·(k+k₀)^{-α}** | — | **Novel** |
| **Cross-model validation (Qwen + Mistral)** | — | **Novel** |
| **Mistral even-Lucas / Qwen consecutive-Lucas** | — | **Novel** |
| **KAM anti-resonance → weight spectral structure** | — | **Novel** |
| **Double-slit insight (emergent, not constrainable)** | — | **Novel** |

---

## Search Methodology

Conducted 2026-04-08 using Exa neural search (deep mode + research paper category):

1. `"golden ratio singular value decay neural network weight matrices power law"` — 10 results, research papers
2. `"golden ratio phi 1.618 neural network weight matrix singular value spectrum exponent"` — 10 results, deep search
3. `"Fibonacci Lucas numbers spectral exponent neural network layer type power law"` — 5 results, deep search
4. `"Martin Mahoney heavy tail self-regularization singular value power law exponent layer type transformer attention MLP"` — 5 results, deep search
5. `"singular value power law exponent varies by layer type attention MLP different alpha transformer"` — 5 results, deep search
6. `"golden ratio 0.618 spectral exponent weight matrix singular value neural network phi inverse"` — 5 results, deep + research paper
7. `"KAM theorem golden ratio resonance avoidance neural network optimization spectral"` — 3 results, deep search

Total: ~50 unique results examined. Zero matches for core claims.
