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
6. **SVD rank truncation catastrophe** — Energy-based rank selection (95% Frobenius) completely destroys model function on Gemma 4-31B. Recomposing `W = U·S·V^T` back to nn.Linear produces identical garbage, confirming the loss is in truncation itself, not SpectralLinear. Language models need near-full-rank or residual correction (`keep_residual=True`).
7. **`from_config` for Gemma 4 takes ~10 min on CPU** — `torch.device('meta')` + `to_empty()` is instant but leaves non-persistent buffers (RoPE inv_freq, embed_scale) uninitialized. These buffers aren't in state_dict. Use `from_config` for correctness when testing inference.
8. **φ-Codec works** — Full SVD + φ-curve prediction + tiered quantization (32/16/8-bit) at 0.34% mean error across 599 layers of Gemma 4-31B. Recomposed model generates coherent text and preserves training voice. The φ-structure is a practical compression prior, not just a theoretical observation.
9. **SpectralLinear needs fp32 spectrum multiply** — The factored forward pass `(x @ V) * S @ U^T` splits one matmul into three bf16 ops, accumulating precision loss that the original fused `x @ W` kernel doesn't. With σ₁≈14 through 60 layers, certain inputs hit resonant paths that overflow bf16 max (65504) → NaN. Fix: keep spectrum in fp32, upcast `xV.float() * spectrum` then downcast. This is NOT bad data — same inputs work with nn.Linear. The NaN is an artifact of factored bf16 arithmetic.
10. **Generation with CPU offloading is prohibitively slow** — ~12s/token because each token requires swapping 410 layers (137.9GB) CPU↔GPU. Skip generation during offloaded training (`not args.offload` guard). Eval val PPL is sufficient for monitoring.

## Current state

- **φ-Codec validated on Gemma 4-31B** — 599 layers encoded→decoded at 0.34% mean error, model talks coherently
- Qwen harmonic fine-tuning running on server 2 (~step 900/2000, val PPL 4607 best at step 650)
- C. elegans structural AND functional spectral analysis complete
- **Next step**: Save φ-compressed format to disk, measure actual compression ratio, compare against GPTQ/GGUF
