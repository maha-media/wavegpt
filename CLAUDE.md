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
11. **Always use GPU SVD for decomposition** — CPU SVD on 31B model (410 layers, full-rank) takes ~11 hours. GPU SVD (`torch.linalg.svd` on CUDA) takes ~50 minutes — 13x speedup. `SpectralLinear.from_linear()` auto-detects CUDA. Never run `decompose_only.py` on a CPU-only machine.
12. **Gemma 4 decomposed + residual needs ≥2 GPUs for inference** — Gemma-4-31B bf16 is ~62 GB, plus per-layer residuals (`keep_residual=True`) push the resident footprint to ~137 GB. `model.to('cuda:0')` OOMs at ~80 GB. The watcher dispatches across two visible GPUs via `accelerate.dispatch_model` with `no_split_module_classes=['Gemma4TextDecoderLayer']`; split across GPUs 4+5 lands at ~69/68 GB.
13. **Gemma 4 generate() on torch < 2.6 needs `attn_implementation='eager'`** — the default attention path uses `or_mask_function` / `and_mask_function` from flex-attention which require torch ≥ 2.6. Pod has torch 2.4. Every `model.generate()` raises `GENERATION FAILED: ... require torch>=2.6`. Build the model via `from_config(..., attn_implementation='eager')` to bypass the flex-attention mask builder.

## Continuous eval watcher

`scripts/eval_watcher.py` runs alongside FSDP training to prove the Kurzweil voice is emerging (CLAUDE.md lesson #1: val PPL alone doesn't tell you the model is speaking).

- **Pinned to idle GPUs** via `CUDA_VISIBLE_DEVICES=4,5` — leaves 0-3 to the 4-way FSDP trainer
- **Boot (~18 min):** scaffold 410 SpectralLinear shells, stream 35 shards into U/V/residual (CPU), then `accelerate.dispatch_model` across both visible GPUs (~137 GB total)
- **Watch loop:** polls `best_spectral.pt` mtime every 30 s; on change, waits for file-size stability, `torch.load`s the 8 MB spectrum dict (`map_location='cpu'` so dispatched params keep their devices), reads latest `{step,val_ppl,val_loss}` from `training_log.json`, runs the 10 Kurzweil prompts (`torch.manual_seed(42+i)`, `temperature=0.7`, `top_p=0.9`, `max_new_tokens=512`) and appends a `## Step N` section to `runs/<RUN>/eval_samples.md`
- **Manual kill only** — `kill $(cat /root/eval_watcher.pid)`
- **Invocation:**
  ```
  CUDA_VISIBLE_DEVICES=4,5 nohup python3 -u scripts/eval_watcher.py \
      --run-dir runs/RAI-gemma4-lossless \
      --hf-model google/gemma-4-31b-it \
      --decomposed-path runs/RAI-gemma4-lossless/shards \
      --rank 0 --mode per_mode --trust-remote-code \
      > /root/eval_watcher.log 2>&1 &
  ```

## Current state

- **φ-Codec validated on Gemma 4-31B** — 599 layers encoded→decoded at 0.34% mean error, model talks coherently
- Qwen harmonic fine-tuning running on server 2 (~step 900/2000, val PPL 4607 best at step 650)
- C. elegans structural AND functional spectral analysis complete
- **Next step**: Save φ-compressed format to disk, measure actual compression ratio, compare against GPTQ/GGUF
