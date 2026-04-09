"""
Decompose a HuggingFace model and save. No training, no GPU needed.
Produces a decomposed.pt that can be loaded instantly for any experiment.

Usage:
    python scripts/decompose_only.py \
        --hf-model Qwen/Qwen3.5-27B \
        --rank 256 --mode per_mode \
        --output runs/qwen35-decomposed/decomposed.pt
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.spectral_surgery import spectral_decompose
from wavegpt.spectral_linear import SpectralLinear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--rank", type=int, default=256,
                        help="Fixed rank (ignored if --adaptive-k0)")
    parser.add_argument("--adaptive-k0", action="store_true",
                        help="Set rank per layer based on k₀: rank = k₀*mult + pad")
    parser.add_argument("--k0-mult", type=float, default=1.5,
                        help="Multiplier for k₀ in adaptive rank")
    parser.add_argument("--k0-pad", type=int, default=128,
                        help="Padding added to k₀*mult for safety margin")
    parser.add_argument("--mode", default="per_mode", choices=["sigma1", "per_mode"])
    parser.add_argument("--keep-residual", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load model to CPU
    print(f"Loading {args.hf_model} to CPU (BF16)...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"  {total:,} params")

    # 2. Decompose
    skip = ['embed_tokens', 'lm_head', 'visual', 'vision', 'wte', 'wpe']
    if args.adaptive_k0:
        print(f"\nDecomposing: adaptive k₀ (mult={args.k0_mult}, pad={args.k0_pad}), mode={args.mode}")
    else:
        print(f"\nDecomposing: rank={args.rank}, mode={args.mode}, skip={skip}")
    t0 = time.time()
    spectral_decompose(
        model, rank=args.rank, mode=args.mode,
        skip_patterns=skip, keep_residual=args.keep_residual,
        k0_mult=args.k0_mult if args.adaptive_k0 else 0.0,
        k0_pad=args.k0_pad if args.adaptive_k0 else 0,
    )
    elapsed = time.time() - t0
    print(f"\n  Decomposition complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # 3. Audit
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_spectral = sum(1 for m in model.modules() if isinstance(m, SpectralLinear))
    print(f"  Params:    {param_bytes/1e9:.2f} GB")
    print(f"  Buffers:   {buf_bytes/1e9:.2f} GB")
    print(f"  Total:     {(param_bytes+buf_bytes)/1e9:.2f} GB")
    print(f"  SpectralLinear layers: {n_spectral}")
    print(f"  Learnable params:      {learnable:,}")

    # 4. Save — use safetensors for large models (no zip overhead, crash-safe)
    print(f"\nSaving to {out_path}...")
    sd = model.state_dict()
    use_safetensors = (param_bytes + buf_bytes) > 5e9  # >5GB → safetensors
    if use_safetensors:
        from safetensors.torch import save_file
        import gc
        # Shard into ~4GB chunks to avoid write quota / memory issues
        shard_dir = out_path.parent / "shards"
        os.makedirs(shard_dir, exist_ok=True)

        shard_max = 4 * 1024**3  # 4GB per shard
        current_shard = {}
        current_bytes = 0
        shard_idx = 0
        shard_files = []

        for k, v in sd.items():
            t = v.contiguous().clone()
            t_bytes = t.nelement() * t.element_size()

            if current_bytes + t_bytes > shard_max and current_shard:
                shard_name = f"shard_{shard_idx:04d}.safetensors"
                shard_path = shard_dir / shard_name
                print(f"  Saving {shard_name} ({current_bytes/1e9:.2f} GB, {len(current_shard)} tensors)...")
                save_file(current_shard, str(shard_path))
                shard_files.append(shard_name)
                shard_idx += 1
                current_shard = {}
                current_bytes = 0
                gc.collect()

            current_shard[k] = t
            current_bytes += t_bytes

        if current_shard:
            shard_name = f"shard_{shard_idx:04d}.safetensors"
            shard_path = shard_dir / shard_name
            print(f"  Saving {shard_name} ({current_bytes/1e9:.2f} GB, {len(current_shard)} tensors)...")
            save_file(current_shard, str(shard_path))
            shard_files.append(shard_name)

        # Write index
        index = {"shards": shard_files, "total_size": int(param_bytes + buf_bytes)}
        with open(shard_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        total_saved = sum(os.path.getsize(shard_dir / s) for s in shard_files)
        print(f"  ✓ {total_saved/1e9:.2f} GB across {len(shard_files)} shards")
        del sd; gc.collect()
    else:
        torch.save(sd, out_path)
        size = os.path.getsize(out_path)
        print(f"  ✓ {size/1e9:.2f} GB")

    # 5. Save config
    config_path = out_path.parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'hf_model': args.hf_model,
            'rank': args.rank,
            'mode': args.mode,
            'keep_residual': args.keep_residual,
            'n_spectral': n_spectral,
            'learnable_params': learnable,
            'decomp_time_s': elapsed,
        }, f, indent=2)

    print(f"\nDone. Use with:")
    print(f"  python scripts/finetune_spectral.py --decomposed {out_path} ...")


if __name__ == '__main__':
    main()
