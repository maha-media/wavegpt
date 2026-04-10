# WaveGPT

## What this is

A research framework proving that trained neural network weight matrices converge to φ-based harmonic spectral structure. The core equation:

```
σ_k = A · (k + k₀)^{-(1/φ)^p}    where p = F(a)/L(b)
```

Singular values follow a bent power law where the exponent is a harmonic of 1/φ, with the specific harmonic determined by Fibonacci/Lucas fractions that depend on the layer's functional type. Confirmed on Qwen3.5-27B, Mistral-7B, Gemma-4-31B, and C. elegans biological neural connectome.

## Key finding

attn_o (the output/consensus projection) has exponent p = 1/3 = F(1)/L(2) on every model tested. This is the only universal exponent. It must be preserved during fine-tuning — destroying it collapses the model's ability to form coherent output.

## Architecture

### Core library (`wavegpt/`)
- `spectral_linear.py` — SpectralLinear: SVD-decomposed linear layer with frozen U,V and learnable spectrum S
- `spectral_surgery.py` — spectral_decompose(): replace nn.Linear with SpectralLinear across any model
- `harmonic_prior.py` — Type-aware harmonic regularization with F/L exponents per layer type. **FL_EXPONENTS dict** maps layer types to their predicted α values. `harmonic_regularization()` supports `type_aware=True` for per-type priors with `attn_o_weight` multiplier
- `harmonic_linear.py` — HarmonicLinear: train-from-scratch spectral parameterization (experimental, diverges at scale — the "double-slit" lesson)
- `model.py` — Small WaveGPT model for training experiments
- `data_io.py` — Binary token file I/O (read_datafile, write_datafile)
- `dataloader.py` — Training data loader with curriculum support

### Analysis scripts (`scripts/`)
- `free_alpha_analysis.py` — Per-layer free-α fitting, type aggregation (Qwen/Mistral)
- `gemma4_alpha_analysis.py` — Gemma 4 analysis (handles mixed sliding/full attention, vision layers)
- `decompose_only.py` — Standalone SVD decomposition + sharded safetensors save. Supports `--delete-source` to free disk before saving, `--adaptive-k0` for per-layer rank allocation
- `finetune_spectral.py` — Spectral fine-tuning with `--harmonic-lambda`, `--type-aware-harmonic`, `--attn-o-weight` flags. Handles HF models via `from_config` (config-only, no weight download). Passes `mm_token_type_ids` for Gemma 4 multimodal
- `retokenize_for_gemma.py` — Re-tokenize corpus between tokenizers
- `celegans_spectral_analysis.py` — C. elegans structural connectome spectral analysis
- `celegans_phi_analysis.py` — F/L fraction matching for C. elegans
- `celegans_deep_svd.py` — Deep SVD: U-clustering, energy thresholds, mode-type alignment
- `energy_threshold_analysis.py` — φ-power energy concentration thresholds
- `alpha_energy_theory.py` — Theoretical analysis of α vs energy distribution
- `phi_vs_pi_debunk.py` — Alternative base analysis (φ vs π, e, √2, random bases)
- `spectral_quantize.py` — Spectral quantization prototype (φ-informed bit allocation)
- `analyze_spectral_checkpoint.py` — Compare spectral checkpoints (drift analysis)

### Docs (`docs/`)
- `the-discovery.md` — Main findings document: equation, cross-model validation, energy thresholds, debunk analysis, falsifiable predictions
- `theory.md` — Harmonic training theory, sequential packing thesis, self-similar energy distribution
- `prior-art.md` — Literature review and novelty analysis

## RunPod servers

Two GPU servers for large-model work:
- Server 1 (port 18409, 216.243.220.173): Gemma 4 decomposition + fine-tuning
- Server 2 (port 14774, 216.243.220.242): Qwen 3.5 decomposition + fine-tuning

SSH: `ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -p PORT root@HOST`

## Conventions

- Spectral exponents: α = (1/φ)^(F/L) for transformers, φ^(F/L) for biological systems (inverse regime)
- Layer type classification: `o_proj`/`out_proj`/`c_proj` → attn_o; `q_proj` → attn_q; etc.
- Sharded safetensors for models >5GB (4GB per shard)
- `python3 -u` (unbuffered) for long-running scripts on servers
- Logs go to `/root/*.log` on servers, NOT `/workspace/`

## Critical lessons

1. **attn_o = 1/3 is universal and must be preserved** — fine-tuning without harmonic regularizer destroyed attn_o's exponent (0.853 → 0.197) and the model couldn't form sentences
2. **φ-structure is emergent, not constrainable** — HarmonicGPT (imposing φ from init) diverged at scale. The structure is where SGD ends up, not where it starts
3. **The debunk matters** — with arbitrary fractions, 87% of random bases fit. The real claim is F/L fractions (φ's continued fraction convergents), not arbitrary rationals. φ beats π by 2.4× under this constraint
4. **Multimodal models need `mm_token_type_ids`** (Gemma 4) and flattened text_config (Qwen 3.5) for `from_config` loading
5. **Disk quota on RunPod** — use `--delete-source` when decomposing, clear hf_cache proactively

## Current state

- Gemma 4 decomposition complete (26GB, 7 shards)
- Harmonic fine-tuning running on Gemma (server 1) with type-aware regularizer
- Qwen harmonic fine-tuning being set up on server 2
- C. elegans structural AND functional spectral analysis complete
- Spectral quantization prototype working (336× lower error than naive 4-bit)
