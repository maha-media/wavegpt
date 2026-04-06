# Experiment Log

All experiments on rai-5m (16M params, 4 layers, 4 heads, 256 embed dim).
Corpus: Ray Kurzweil's published works — 56 sources, 22K entities, 17K relationships.

## Run Index

| Run | Config | Steps | Best Val | PPL | Key Finding |
|-----|--------|-------|----------|-----|-------------|
| D | Random init, old data | 5K | 7.097 | 1209 | Baseline |
| E | + token weights (0.3-3x) | 5K | 7.476 | 1766 | **Weights hurt** |
| F | + rich data + weights | 5K | 6.586 | 725 | Rich data helps despite weights |
| G | + data curriculum | 5K | 5.880 | 358 | **Curriculum is huge** |
| H | Harmonic embedding init | 5K | 6.575 | 717 | Particle approach hurts |
| I | Wave attention init | 5K | 6.016 | 410 | Wins early, fades |
| J | G minus token weights | 5K | 5.276 | 196 | **Weights were hurting all along** |
| K | J + dropout 0.3 | 5K | 5.280 | 196 | Dropout 0.3 = no change |
| L | J + 15K steps | 15K | 4.711 | 111 | More training helps |
| M | + contrastive data | 15K | 4.657 | 105 | Anti-collapse in data |
| **N** | **+ anti-collapse α=0.05** | **15K** | **4.530** | **93** | **Champion** |
| O | Harmonic curriculum (hard) | 15K | 4.697 | 110 | Phase separation hurts |
| P | Harmonic curriculum (soft) | 15K | 4.669 | 107 | Better but still fragmented |
| Q | Ordered aug (C→G→D→A) | 15K | 4.570 | 97 | Best stability (std 0.638) |
| R | + counterpoint narratives | 15K | 4.572 | 97 | Right idea, noisy execution |

## Key Transitions

**D → J** (PPL 1209 → 196, 6.2x): Rich KG-augmented data + 3-phase curriculum.
The single biggest improvement. Structured knowledge from the entity graph
provides the overtones that raw text alone can't.

**J → L** (PPL 196 → 111, 1.8x): 15K steps instead of 5K.
At 0.23 tokens/param, the model is massively under-trained. More epochs help.

**L → N** (PPL 111 → 93, 1.2x): Contrastive data + anti-collapse regularization.
Two forms of nuance preservation: in the data (near-miss pairs) and in the loss
(variance penalty). Together they push the model past the collapse attractor.

## Oscillation Analysis

All runs show val loss oscillation — the model periodically collapses to the
fundamental and escapes. The oscillation range is the "breathing" of the model
between collapse and nuance.

| Run | Osc Std | Range | Interpretation |
|-----|---------|-------|----------------|
| D | 0.996 | 2.96 | Wild swings — no stability |
| J | 0.716 | 1.72 | Rich data calms it |
| N | 0.678 | 1.73 | Anti-collapse helps |
| Q | 0.638 | 1.71 | Ordered data = smoothest |
