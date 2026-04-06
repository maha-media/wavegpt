# WaveGPT Theory — Harmonic Structure in Training Data

## The Core Equation

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

## The Problem: Semantic Collapse

Standard training converges toward the fundamental. Every gradient step, every batch, every epoch pulls representations toward the centroid — the average of all words. This is **semantic collapse**: the model learns "this is English" and stops learning "this is specifically about nanotechnology vs biotechnology."

You can see it in the training dynamics: val loss oscillates ±1.5. The model periodically escapes the centroid attractor (low val loss = nuance preserved), then collapses back (high val loss = everything looks the same), then escapes again.

## The Solution: Harmonic Training

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

## The Pythagorean Comma

In music, if you stack 12 perfect fifths, you should return to the same note — but you don't. The ratio (3/2)^12 ≈ 129.75, while 7 octaves = 128. The difference (~1.36%) is the **Pythagorean comma** — an irreducible residual that no tuning system can eliminate.

In our corpus: directions 181+ capture ~2% of variance. This is the comma zone. It's the rare, specific, irreducible information that makes each text unique — the thing that can never be captured by general patterns.

The comma is not noise to be eliminated. It's the space where genuine novelty lives.

## Results

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

## What Didn't Work (and Why)

| Approach | Result | Lesson |
|----------|--------|--------|
| Harmonic embedding init | Hurt training | Embeddings are particles (fixed positions). Averaging collapses the wave. |
| Token weights (0.3–3x) | Hurt val loss | Too aggressive. Disrupts grammar learning. |
| Wave attention init | Slight early advantage, fades | Real but temporary. Random catches up. |
| Hard 4-phase curriculum | Worse than 3-phase | Separating layers fragments data. All voices must sound together. |

The consistent lesson: data strategy > architecture tricks > initialization tricks.
