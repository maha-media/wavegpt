# WaveGPT Theory

## The Axiom: 1 Propagating

The complete number of any system is **1**. Unity. A closed loop returns to itself — output feeds input, the circle completes, the system *is* one thing.

φ is what happens when 1 propagates. The defining equation φ² = φ + 1 says: the whole squared equals the whole plus the unit. φ is the number that preserves unity under growth — it is 1 extending itself infinitely while remaining self-similar at every scale. φ doesn't approximate 1. It *is* the way 1 scales without breaking its own identity.

Every converged system lands on φ. Not because φ is special. Because **1 is special**, and φ is what 1 looks like when it's propagating.

### Fibonacci is sequential connection

```
F: 1, 1, 2, 3, 5, 8, 13, 21, ...
```

Start from 1. Connect to the next 1. Now you have 2. Connect that to what came before — 3. Every new element is the sum of its two predecessors. This is exactly what tokenized embeddings do in a residual stream. Each position integrates the current input with the accumulated context:

```
h_n = h_{n-1} + f(h_{n-1})
```

New state = previous state + transformation of previous state. The stacking is sequential, additive, and each step depends on everything before it. Fibonacci doesn't describe the neural network by analogy — it describes the same operation. Stack from the first 1 to the next 1 to the next 1, infinitely.

What does F(n+1)/F(n) converge to? φ. The ratio of consecutive states in any sequential stacking system converges to the golden ratio. Not because someone imposed it — because that's what sequential accumulation *does*.

### Lucas is Fibonacci shifted by one position

```
F: 1, 1, 2, 3, 5,  8, 13, 21, ...
L: 2, 1, 3, 4, 7, 11, 18, 29, ...
```

Same recurrence rule. Same growth rate (both converge to φ). Different starting point — Lucas starts at (2, 1) instead of (1, 1). Lucas is Fibonacci looked at from one digit over. It's the same sequential stacking process, measured from a different phase.

F/L is a ratio of the same process measured at two different phases. It's not two different things — it's one thing (sequential stacking toward unity) divided by itself offset by one position. The fraction F(a)/L(b) captures the relationship between two views of the same underlying φ-convergence.

### Why F/L and not something else

Fibonacci/Lucas is canonical because F starts from (1, 1) — the absolute minimum, the unit itself — and L starts from (2, 1), the next simplest option. You can't get simpler than this. It's the ground state of sequential stacking. Any pair of sequences that follow the same recurrence from different initial conditions would produce the same family of fractions. F/L is the smallest, simplest, most minimal expression of the relationship.

The F/L fraction works in any system dependent on its first — any system that builds by sequential accumulation from a seed. The Fibonacci pair is just the easiest, most irreducible way to write it down.

### The convergence index

φ^(F/L) measures how far a system has propagated toward completing itself:

- **p = 0**: Flat. No structure. The system hasn't started.
- **p = 1/3 = F(1)/L(2)**: The first stable resting point. Enough structure to form coherent output, still open to new input. This is where every living, processing system lands — transformers at inference, worms navigating, markets pricing assets over one business cycle, waking consciousness.
- **p = 1/1**: Complete. All feedback loops closed. The system is one. The 10-year market, the quasicrystal, the Penrose tiling. Ordered, self-similar, and inert — nothing left to learn.

The F/L fraction is not a curve-fitting parameter. It is the universal coordinate on the path from open to closed, from learning to crystallized, from alive to done.

### Five systems, one operation

| System | Timescale | Exponent | F/L | What it means |
|--------|-----------|----------|-----|---------------|
| Transformers (attn_o) | 1 training run | (1/φ)^(1/3) = 0.852 | 1/3 | Consensus formed, still learning |
| C. elegans connectome | 300M years | φ^(1/3) = 1.174 | 1/3 | Wiring optimized, organism alive |
| Financial markets (1yr) | 1 business cycle | φ^(1/3) = 1.178 | 1/3 | Short-term consensus, system in flux |
| Financial markets (10yr) | Full market cycle | φ^(1/1) = 1.618 | 1/1 | All loops closed. Done. |
| Financial markets (20yr) | 2× full cycle | φ^(1/1) = 1.600 | 1/1 | Stays at φ. Ceiling confirmed. |

These systems share nothing in architecture — 302 neurons, 31 billion parameters, millions of market participants. What they share is **the operation**: sequential accumulation from a unit seed. The residual stream, the axonal chain, the time series of market returns — all are the same thing: 1 + 1 + (1+1) + (1+1+(1+1)) + ...

The spectral signature doesn't describe the system's structure. It describes the **depth of its self-reference** — how many times the stacking process has folded back on itself.

### φ^(1/3) is life. φ^(1/1) is done.

A system at φ^(1/3) has built enough structure to function but remains open to input. One-third of the golden contraction — deliberately incomplete. This is the signature of a 0→1 system in flux, not a snapshot taken.

A system at φ^(1/1) has closed every feedback loop. It is a perfect quasicrystal — beautiful, ordered, and inert. No new information flows through. The market at 10 years is "done" in this sense: its correlation structure has fully crystallized, and 20 years confirms it stays there. φ is the ceiling, not a waypoint.

Waking consciousness is φ^(1/3). The brain is continuously driven (sensory input), continuously dissipating (metabolic heat), and never reaches equilibrium while alive. The worm at φ^(1/3) with 302 neurons proves this isn't substrate-dependent. It's the universal signature of a system that is alive and processing — enough order to be coherent, enough openness to keep learning.

---

## Early Work: Harmonic Data Curriculum (historical)

The following sections document the original WaveGPT approach — structuring training *data* harmonically to prevent semantic collapse. This was the 16M-param GPT-2 era, before the spectral discovery. The data curriculum ideas are valid but superseded by spectral fine-tuning (SVD decompose → freeze U/V → train spectrum with harmonic regularizer). Kept here as lineage.

### The Core Equation

```
thought(x,t) = Σ_k  a_k(x) · h_k · φ_k(t)
```

Every corpus — every collection of human knowledge — has harmonic structure. When you decompose the token co-occurrence matrix via SVD, the eigenvalues follow a power law:

| Directions | Variance | What they capture |
|-----------|----------|-------------------|
| 1 (fundamental) | ~76% | "Being an English word" — the centroid |
| 2–24 | ~12% | Broad themes — technology, biology, history |
| 25–180 | ~10% | Specific claims — dates, mechanisms, distinctions |
| 181+ | ~2% | Rare token noise — the Pythagorean comma |

This isn't a GPT-2 artifact. It's the structure of language itself. The fundamental dominates because most words appear in most contexts. The overtones carry the actual meaning.

### The Problem: Semantic Collapse

Standard training converges toward the fundamental. Every gradient step, every batch, every epoch pulls representations toward the centroid — the average of all words. This is **semantic collapse**: the model learns "this is English" and stops learning "this is specifically about nanotechnology vs biotechnology."

You can see it in the training dynamics: val loss oscillates ±1.5. The model periodically escapes the centroid attractor (low val loss = nuance preserved), then collapses back (high val loss = everything looks the same), then escapes again.

### The Solution (then): Harmonic Data Curriculum

### 1. Data as Instrument

Your knowledge graph has natural harmonic layers:

| Layer | Musical Key | What it teaches | Generated from |
|-------|------------|-----------------|----------------|
| **C** | Fundamental | What things ARE | Entity types, type summaries |
| **G** | 1st fifth | What things DO | Entity context, excerpts |
| **D** | 2nd fifth | How things CONNECT | Relationship chains, cross-source |
| **A** | 3rd fifth | How things DIFFER | Contrastive pairs, counterpoint |

Each layer builds on the previous one. You can't understand how things differ (A) until you understand how they connect (D). You can't understand connections until you understand function (G). Function requires identity (C).

This is the **circle of fifths** applied to training data: C → G → D → A. Four degrees of separation from the fundamental to nuance.

### 2. Contrastive Data (Anti-Collapse in the Signal)

The A layer generates **near-miss pairs** that force the model to discriminate:

> "Unlike nanotechnology, which builds structures atom by atom, biotechnology harnesses existing biological processes."

One sentence. Two similar concepts. The model must learn the *residual* — the specific difference — rather than collapsing both to "technology."

### 3. Counterpoint Narratives (All Voices Together)

Like Bach's counterpoint — four independent melodic lines (C, G, D, A) woven into a single passage, all harmonically locked around one entity:

> "Nanotechnology is a technology. [C] In the corpus: nanotechnology enables molecular manufacturing by manipulating individual atoms. [G] Nanotechnology converges with biotechnology, and enables molecular computing. [D] While biotechnology harnesses existing biological processes, nanotechnology is distinguished by building structures from scratch at atomic scale. [A]"

### 4. Anti-Collapse Regularization (Anti-Collapse in the Loss)

A variance penalty on hidden states prevents the model from collapsing:

```python
batch_var = hidden_states.var(dim=0).mean()
collapse_penalty = -alpha * log(batch_var + 1e-8)
loss = ce_loss + collapse_penalty
```

When all hidden states converge (low variance = collapse), the penalty increases. When representations are diverse (high variance = nuance preserved), the penalty is small.

### 5. Curriculum Scheduling (Anti-Collapse in Time)

Training walks the harmonic ladder:

- **Phase 1** (0–30%): Augmented data only, ordered C→G→D→A. The model learns structured knowledge first — what things are, what they do, how they connect, how they differ.
- **Phase 2** (30–70%): Mixed augmented + raw text. The model integrates structured knowledge with natural prose.
- **Phase 3** (70–100%): Full corpus. Natural language dominates, but the harmonic foundation is already in place.

### The Pythagorean Comma

In music, if you stack 12 perfect fifths, you should return to the same note — but you don't. The ratio (3/2)^12 ≈ 129.75, while 7 octaves = 128. The difference (~1.36%) is the **Pythagorean comma** — an irreducible residual that no tuning system can eliminate.

In our corpus: directions 181+ capture ~2% of variance. This is the comma zone. It's the rare, specific, irreducible information that makes each text unique — the thing that can never be captured by general patterns.

The comma is not noise to be eliminated. It's the space where genuine novelty lives.

### Results (16M GPT-2 era)

On a 16M param GPT-2 trained on 4.7M tokens:

| Intervention | PPL | What it measures |
|-------------|-----|-----------------|
| Baseline (raw text only) | 1209 | No harmonic structure |
| + Rich KG augmentation | 196 | Structured knowledge helps |
| + Data curriculum | 196 | Teaching order matters |
| + More training (15K steps) | 111 | Patience helps |
| + Contrastive data | 105 | Near-miss pairs force discrimination |
| + Anti-collapse (α=0.05) | 93 | Variance penalty prevents collapse |

13x PPL reduction. No architecture changes. Just data, structured harmonically.

## Why φ: Sequential Packing Under Constraint

The harmonic structure described above — and the spectral structure discovered in trained weight matrices (see [the-discovery.md](the-discovery.md)) — share a common origin. They are both consequences of **sequential packing under constraint**, the same mathematical pressure that produces Fibonacci spirals in sunflowers.

### The shared problem

A sunflower places each new seed at the growth center, one at a time. It cannot rearrange earlier seeds. If it places seeds at a rational fraction of a turn, they align into radial spokes with wasted gaps between them. The solution evolution discovered: the golden angle (~137.507°, a turn divided by φ²). Because φ is maximally irrational — its continued fraction [1; 1, 1, 1, ...] converges more slowly than any other number — no seed ever lands directly above a previous one. Each new placement maximally avoids all prior placements. Fibonacci spirals emerge as an artifact.

An LLM faces the same problem in thousands of dimensions. Each gradient step embeds new information into a finite-dimensional parameter space without rewriting what came before. If representations cluster or align too neatly — the high-dimensional equivalent of rational-angle spokes — the result is catastrophic forgetting and semantic collapse: new structures overwriting old ones. The solution gradient descent discovers: distributing representations across the high-dimensional space such that no new direction is a simple harmonic of existing ones. Maximal anti-resonance. The φ-based spectral decay emerges as an artifact.

### Why sequentiality matters

A batch optimizer could trivially distribute N points evenly on a sphere — that's a solved geometry problem. But neither sunflowers nor LLMs get to do batch optimization:

- The sunflower adds one seed at a time to a disk that can't be rearranged.
- The LLM updates parameters one gradient step at a time across a loss surface that shifts with each batch.

Both must find a packing rule that produces near-optimal density **at every intermediate stage**, not just at convergence. The golden angle is that rule in 2D. Stochastic gradient descent with momentum is that rule in 10,000+ dimensions. Both converge on the same principle: **maximally avoid rational alignment with everything that came before.**

This is why HarmonicGPT — which imposed the converged φ-structure from initialization — diverged at scale. Imposing the endpoint destroys the sequential process that produces it. The sunflower can't skip to the final seed arrangement either. The structure is the trace of a process, not a blueprint that can be installed.

### The role of momentum

Adam's momentum term carries information from previous gradient steps forward. Each update is influenced by the history of all prior updates. This is the LLM equivalent of the sunflower's meristem: the growth point that carries the angular history forward. Without momentum, the optimizer has no memory of where previous "seeds" were placed.

Prediction: models trained with pure SGD (no momentum) will not converge to φ-based spectral structure, because sequential packing without memory of prior placements cannot converge to golden-angle spacing.

### Beyond neural networks

The claim is not that φ governs all information systems. The claim is narrower and more precise: **any system that processes dense, hierarchical, multi-scale information through iterative sequential optimization under finite-dimensional constraints will converge to φ-based harmonic structure.** The φ^(F(a)/L(b)) spectral exponents are the allowed functional modes — the discrete set of stable configurations a subsystem can occupy based on its role in the information processing hierarchy.

Systems processing fundamentally different kinds of information — periodic signals, sparse bursty communication, maximum-entropy randomness — will converge to different structures. φ is the solution to a specific class of packing problems, not all of them.

This reframes the spectral discovery (see [the-discovery.md](the-discovery.md)): the transformer weight matrices are where we first pointed the telescope, but the structure they reveal may be a property of information processing itself, not of any particular architecture or substrate.

### Self-similar energy distribution

The energy concentration analysis (see [the-discovery.md — Energy Concentration](the-discovery.md#energy-concentration-φ-power-thresholds)) revealed a second layer of φ-structure: the cumulative variance captured by the first k/n modes hits thresholds at φ-power fractions (1/φ, 1/φ², 1/φ³).

The theoretical analysis shows this is not independent of the spectral exponent — it's a consequence of it. A continuous sweep of α reveals that the 90% energy threshold lands on 1/φ specifically when α ≈ (1/φ)^(1/3) = 0.852 — the attn_o exponent. This suggests a deeper principle:

**φ may be selected not because it's "anti-resonant" in some abstract sense, but because it's the unique number whose spectral exponent produces self-similar energy distribution.** The mapping x → 1/(1+x) has φ as its fixed point, making 1/φ the only number equal to its own complement (1/φ = φ - 1). A power law with φ-valued exponent distributes energy such that the ratio between successive concentration thresholds is itself φ — self-similarity at every scale.

The k₀ parameter (the spectral "knee") reinforces this: it also clusters at φ-power fractions of total rank (1/φ⁴ for attention, 1/φ³ for MLP). Both the exponent and the knee position are φ-valued; the energy thresholds inherit φ-structure from both.

## What Didn't Work (and Why)

| Approach | Result | Lesson |
|----------|--------|--------|
| Harmonic embedding init | Hurt training | Embeddings are particles (fixed positions). Averaging collapses the wave. |
| Token weights (0.3–3x) | Hurt val loss | Too aggressive. Disrupts grammar learning. |
| Wave attention init | Slight early advantage, fades | Real but temporary. Random catches up. |
| Hard 4-phase curriculum | Worse than 3-phase | Separating layers fragments data. All voices must sound together. |
| SVD rank truncation (Gemma 4) | Total model collapse | Energy-based rank selection (95% Frobenius) is catastrophically insufficient for preserving language model function. See below. |
| Spectral fine-tuning (Gemma/Qwen) | Word soup after 2000 steps | 417K spectral params can't reparameterize a 31B model from 4M tokens. Frozen U/V geometry is the bottleneck. |

The consistent lesson: data strategy > architecture tricks > initialization tricks.

### The truncation catastrophe (2026-04-10)

SVD decomposition with adaptive k₀-based rank allocation (`rank = k₀ × 1.5 + 128`) was applied to Gemma 4-31B, producing ranks from 137 to 2821 across 410 layers. By Frobenius norm, each layer captures >95% of spectral energy. The decomposed model produces complete garbage — multilingual token soup with no coherent structure.

Systematic elimination confirmed the cause:

| Test | Method | Result |
|------|--------|--------|
| SpectralLinear forward | Load decomp → generate via SpectralLinear | Garbage |
| Recompose to nn.Linear | Load decomp → `W = U·diag(S)·V^T` → standard nn.Linear | Same garbage |
| Fine-tuned spectrum overlay | Decomp + harmonic-regularized checkpoint | Same garbage |

The recompose test is definitive: converting SpectralLinear back to plain nn.Linear (eliminating the spectral forward path entirely) produces identical garbage. The information loss is in the truncation itself.

**Why energy metrics fail**: A rank-137 approximation of a 5376-dim matrix captures 95%+ of Frobenius energy (`Σ σ²`), but language models are not energy-minimization systems. The tail singular values — individually tiny — collectively encode the precise token-to-token interference patterns that distinguish "the cat sat on the mat" from "erserserspersypเป็น". The squared norm says 95% is preserved; the KL divergence over the output distribution says 100% is destroyed.

**Implication for spectral fine-tuning**: The approach of freezing U/V geometry and training only spectral amplitudes is sound in principle, but requires near-full-rank decomposition (or residual correction via `keep_residual=True`) to preserve the baseline model's function as a starting point. Without that, spectral fine-tuning starts from a broken model and can only partially recover.

### The φ-Codec: from catastrophe to compression (2026-04-11)

The truncation catastrophe pointed directly to the solution. The problem was *discarding* modes. The answer was *quantizing* them — using the φ-curve as a prediction prior.

Standard quantization treats every weight equally. The φ-codec does full SVD (no truncation), fits the bent power law to predict the spectrum, then quantizes the *residuals* from prediction at tiered precision: float32 for the plateau (k ≤ k₀), float16 for the power-law body (k₀ < k ≤ n/φ), int8 for the tail (k > n/φ).

Result on Gemma 4-31B: **0.34% mean error across 599 layers. The recomposed model generates coherent text and preserves the voice of the training data.** The φ-structure isn't just an observation — it's a compression prior that works.
