# Spectral Personalities

## One Model, Infinite Voices

A 27B language model is ~56GB. A LoRA adapter is ~212MB. A spectral personality file is **~450KB**.

### Why It's Small

Every weight matrix W in a neural network can be decomposed:

```
W = U · diag(σ₁, σ₂, ..., σᵣ) · Vᵀ
```

U and V are the **geometry** — the directions the model learned during pretraining. These encode *what the model knows*: syntax, facts, reasoning patterns. They're large (millions of floats) and shared across all personalities.

The singular values σ₁...σᵣ are the **amplitudes** — how much each direction is activated. These encode *voice*: which knowledge to emphasize, what style to produce, how to weight competing patterns.

To make a model sound like Ray Kurzweil vs a legal assistant vs a poet, you don't need different geometry. You need different amplitudes.

For Qwen3.5-27B with rank-256 across 496 layers:
- **Base model**: 26.9B params, 56GB — loaded once, shared
- **Per personality**: 115K params, 450KB — swapped per request

That's a **127,000× compression ratio**.

### How Swapping Works

```
Base model in VRAM (56GB, loaded once)
  ↓
Request arrives: "Use personality: ray-kurzweil"
  ↓
Load ray-kurzweil.pt (450KB) from disk/cache
  ↓
For each SpectralLinear layer:
  layer.spectrum = personality_state_dict[layer_name]  # vector copy
  ↓
Generate response (forward pass uses updated amplitudes)
  ↓
Next request: "Use personality: legal-analyst"
  ↓
Load legal-analyst.pt (450KB), overwrite spectra
  ↓
Generate with new voice
```

No model reload. No VRAM allocation. Just overwriting ~115K floats across 496 layers — a few milliseconds of memcpy.

### Comparison

| Method | Trainable Params | File Size | Swap Time | VRAM per personality |
|--------|-----------------|-----------|-----------|---------------------|
| Full fine-tune | 26.9B | 56GB | Minutes (full reload) | 56GB |
| LoRA r-16 | 111M | 212MB | Seconds (adapter load) | ~400MB |
| **Spectral r-256** | **115K** | **450KB** | **Milliseconds (vector copy)** | **~0** |

### What This Enables

- **One GPU serves unlimited personalities**. Load Qwen3.5-27B once. Keep a directory of 450KB `.pt` files. Route requests to the right personality by name.

- **Personalities are trivially distributable**. Email a 450KB file. Put it in a git repo. Download over 2G mobile in under a second.

- **A/B testing is free**. Swap between personality A and B between requests. No infrastructure changes.

- **Personality arithmetic**. Since spectra are just vectors, you can interpolate: `spectrum = 0.7 * ray + 0.3 * legal` produces a blend. Or subtract: `spectrum = technical - casual` to find the "formality direction" in spectral space.

### The Math of Why This Works

Trained weight matrices converge to a universal structure:

```
W = σ₁ · Σₖ k^{-1/φ} · uₖ · vₖᵀ
```

The geometry (U, V) is **shared equilibrium** — all models trained on language converge to similar directional structure. The amplitudes following k^{-1/φ} decay are the **universal prior**. Personality is the deviation from that prior: which modes get boosted, which get damped, relative to the power law.

A personality file is literally a list of numbers saying "boost direction 17 by 3%, damp direction 42 by 8%, ..." across each layer. That's all voice is. Geometry is knowledge. Amplitude is personality.
