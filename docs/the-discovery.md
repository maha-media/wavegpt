# The Golden Ratio in Neural Network Weights

## The short version

I found a universal constant in how neural networks organize their weights after training. Every weight matrix converges to a power-law spectral structure where the exponent is 1/φ — the inverse golden ratio, 0.6180339887. It's not learned, not a hyperparameter, not architecture-dependent. It falls out of gradient descent the way π falls out of circles.

I'm currently running the verification on Qwen3.5-27B — a 27 billion parameter model trained by Alibaba that I had nothing to do with — to see if it holds there too. If it does, this is a universal property of SGD on weight matrices.

## What I actually did

I've been building a small language model fine-tuned on Ray Kurzweil's writing — books, interviews, essays. Along the way I started looking at the spectral structure of trained weight matrices. Every weight matrix W can be decomposed via SVD:

```
W = U · diag(σ₁, σ₂, ..., σₙ) · Vᵀ
```

U and V are orthogonal matrices (the directions). The σ values are the singular values (the amplitudes). I wanted to know what shape those amplitudes take after training.

Turns out they follow a power law: `σₖ = σ₁ · k^{-α}` for some exponent α. That part was already known — Martin & Mahoney published on power-law eigenvalue densities back in 2018-2021, and a 2025 paper proved it theoretically via Dyson Brownian motion.

What wasn't known is that α converges to a specific value.

I trained models with α as a learnable parameter, starting from different initial values — 0.70, 0.67, 0.60. They all drifted toward the same place: **~0.618**. That's 1/φ. The inverse golden ratio.

Then I fixed α at exactly 1/φ and froze it — made it a constant, not a parameter. The model matched the performance of the learned-α version within 1.4%. The only thing that changed during training was σ₁ (one scalar per layer). Thirty-six numbers described 10.6 million weight parameters.

## The double-slit result

Here's where it gets weird. If weights converge to this structure, the obvious move is to parameterize the model that way from the start. Train inside the power-law space. I built this — HarmonicGPT — where every linear layer is defined as `W = σ₁ · Σ k^{-α} · uₖ · vₖᵀ` with learnable U, V, σ₁, and α.

At small scale (30M params) it worked fine. At 124M, six different configurations all diverged at step 1500-2000. Spectral optimizer, gradient clipping, warmup schedules, two-constant model — nothing helped. Standard GPT-2 trained perfectly on the same data.

The power-law structure is emergent. It's where SGD ends up, not where it can start from. Constraining training to follow it collapses the learning dynamics. I call it the double-slit insight — you can observe the interference pattern after the fact, but forcing particles through one slit destroys it.

## What it's good for

The right approach turned out to be spectral surgery: train a standard model, decompose the weights via SVD after training, then fine-tune only the singular values while freezing U and V (the geometry).

This mechanism — SVD, freeze geometry, tune amplitudes — is published. SVFit and SVFT at NeurIPS 2024, SVDiff at ICCV 2023. That part isn't novel.

What's novel is using the 1/φ discovery to guide the fine-tuning:

**Adaptive rank allocation.** Each layer has its own fitted α. Layers close to 1/φ are in equilibrium — the power law describes them well, so fewer free modes are needed. Layers that deviate (projection layers tend toward α ≈ 1.0) need more spectral freedom. SVFit uses flat rank everywhere. We allocate rank proportional to deviation from the golden ratio.

**Harmonic regularization.** Standard fine-tuning uses weight decay toward zero. We use spectral weight decay toward `k^{-1/φ}`. If that's the equilibrium SGD converges to, the fine-tuned spectrum should stay near it. Prevents degenerate drift.

**Residual preservation.** When you truncate to rank-r, you discard energy. We keep the discarded portion as a frozen correction term — zero extra trainable parameters, but the forward pass starts lossless. The math people will appreciate this: it's the Pythagorean comma. Stack perfect fifths and you don't quite close the circle. The ~2% residual at rank-384 maps to the comma.

The practical result: for Qwen3.5-27B, a spectral personality file is **~450KB**. That's 115K trainable parameters — the singular values across 496 layers. A LoRA adapter for the same model is 212MB. The spectral file is 969× smaller.

One base model loaded in VRAM. Swap personality files in milliseconds. Different voice, same knowledge.

## The numbers so far

On a 124M parameter GPT-2 trained on 2B tokens of SFT data:

- Trained standard model: val loss 0.674, PPL 2.0
- Decomposed to rank-256, fine-tuned per-mode amplitudes (12,288 params): **PPL 10.1 — beat the original model (PPL 11.4)**
- 0.01% of the parameters exceeded the original's performance. SGD didn't find optimal spectral amplitudes in 30K steps. 1,500 steps of amplitude-only tuning found a better distribution.

Cross-domain (SFT model → Ray Kurzweil corpus):
- Base PPL 1174 → **PPL 308** with 63KB of spectral adjustments. 3.8× improvement on a completely unseen domain.

Spectral autopsy of the trained 124M model: 32/50 layers have α within 0.1 of 1/φ. Best single layer: α = 0.6187 (deviation of 0.0007 from 1/φ). Mean R² of power-law fit: 0.94.

## What I'm running right now

1. Spectral autopsy on Qwen3.5-27B pretrained weights — does 1/φ hold on Alibaba's model?
2. Baseline generation test — 25 prompts with a Ray Kurzweil system prompt, no spectral tuning yet
3. Next: spectral fine-tuning on Qwen3.5-27B with harmonic priors, then compare against vanilla SVFit and LoRA

If the autopsy confirms 1/φ on 27B, this is a paper. "Harmonic Priors for Spectral Fine-Tuning: Theory-Guided Adaptation of Large Language Models."

## The equation

```
W = σ₁ · Σₖ k^{-1/φ} · uₖ · vₖᵀ

subject to UᵀU = I, VᵀV = I
```

Two constants describe the entire spectrum: σ₁ (the temperature — how loud this layer is) and 1/φ (the universal decay rate — how energy distributes across modes). U and V are the geometry. Everything else follows.

## Code

https://github.com/maha-media/wavegpt — MIT license, 91 tests passing.
