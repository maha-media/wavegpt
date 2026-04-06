# WaveGPT

**Harmonic training framework for small language models.**

Your data has structure. Your training should too.

WaveGPT treats training data as a harmonic system — knowledge graphs decompose into fundamental frequencies (what things *are*), overtones (what they *do*, how they *connect*), and nuance (how they *differ*). The training curriculum walks this harmonic ladder like the circle of fifths: **C → G → D → A**.

The result: a 16M parameter model trained on 4.7M tokens achieves PPL 93 — a 13x improvement over naive training. No architecture tricks. No exotic initialization. Just data, structured harmonically.

## The Equation

```
thought(x,t) = Σ_k  a_k(x) · h_k · φ_k(t)
               amplitude × harmonic × phase
```

Every corpus has this structure. The fundamentals dominate (~76% of variance), broad themes form the first overtones (~12%), specific claims the second (~10%), and rare nuance the residual (~2%). Training that respects this structure converges faster and generalizes better than training that ignores it.

## Quick Start

```bash
pip install wavegpt

# Prepare your knowledge graph as JSON
wavegpt export --entities entities.json --relationships rels.json --chunks chunks.json --output data/

# Train
wavegpt train --data-dir data/ --model small --steps 15000 --data-curriculum --collapse-alpha 0.05
```

## What WaveGPT Does

### 1. Harmonic Data Augmentation
Given a knowledge graph (entities, relationships, chunks of text), WaveGPT generates training narratives at four harmonic layers:

| Layer | Musical Key | What it teaches | Example |
|-------|------------|-----------------|---------|
| **C** | Fundamental | What things ARE | *"Nanotechnology is a technology."* |
| **G** | 1st fifth | What things DO | *"In the corpus: nanotechnology enables molecular manufacturing…"* |
| **D** | 2nd fifth | How things CONNECT | *"Nanotechnology converges with biotechnology through…"* |
| **A** | 3rd fifth | How things DIFFER | *"Unlike genetic engineering, nanotechnology builds from atoms up."* |

Plus **counterpoint narratives** — all four voices woven into a single passage per entity.

### 2. Anti-Collapse Regularization
Standard training collapses representations toward the centroid (the "fundamental"). WaveGPT adds a variance penalty to the loss function that prevents this:

```python
# Hidden state variance across batch — high = nuance preserved, low = collapse
batch_var = hidden_states.var(dim=0).mean()
collapse_penalty = -alpha * log(batch_var)
loss = ce_loss + collapse_penalty
```

### 3. Curriculum Scheduling
Training walks the harmonic ladder:
- **Phase 1** (0-30%): Augmented data only — structured knowledge
- **Phase 2** (30-70%): Mixed augmented + raw text
- **Phase 3** (70-100%): Full corpus — natural language dominates

The augmented data is ordered **C → G → D → A** within the file, so sequential reading naturally walks the circle of fifths.

## Results

Experiments on a 16M param GPT-2 (4 layers, 4 heads, 256 embed) trained on Ray Kurzweil's published works (56 sources, 22K entities, 17K relationships):

| Run | Config | PPL | Δ from baseline |
|-----|--------|-----|-----------------|
| D | Baseline (random init, raw text) | 1209 | — |
| J | + Rich KG augmentation + curriculum | 196 | 6.2x better |
| L | + 15K steps | 111 | 10.9x |
| M | + Contrastive data | 105 | 11.5x |
| **N** | **+ Anti-collapse (α=0.05)** | **93** | **13.0x** |

Every improvement came from data strategy, not architecture changes.

## Installation

```bash
pip install wavegpt
```

Or from source:

```bash
git clone https://github.com/maha-media/wavegpt.git
cd wavegpt
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.0+, tiktoken

## Data Format

WaveGPT expects your knowledge graph as JSON:

```json
// entities.json
[
  {"entity_id": "e1", "name": "nanotechnology", "type": "technology",
   "source_chunks": ["c1", "c2", "c3"]}
]

// relationships.json
[
  {"source_entity": "e1", "target_entity": "e2",
   "type": "converges with", "description": "..."}
]

// chunks.json
[
  {"chunk_id": "c1", "source_id": "s1", "text": "Nanotechnology enables..."}
]
```

Any knowledge graph works. The framework extracts harmonic structure from whatever entities and relationships you provide.

## Architecture

WaveGPT is a standard GPT-2 with one addition: the anti-collapse penalty on hidden state variance. No custom attention, no special embeddings. The innovation is entirely in how data is prepared and scheduled.

```
WaveGPTConfig(
    vocab_size=50257,   # GPT-2 tokenizer
    block_size=256,     # context window
    n_layer=4,          # transformer layers
    n_head=4,           # attention heads
    n_embd=256,         # embedding dimension
    dropout=0.1,
)
```

Model sizes: `small` (16M), `medium` (30M), `large` (125M).

## Theory

See [docs/theory.md](docs/theory.md) for the full harmonic framework:
- Why training data has wave structure
- The circle of fifths as a curriculum
- Anti-collapse: fighting the fundamental attractor
- The Pythagorean comma — the 2% residual where nuance lives

## License

MIT
