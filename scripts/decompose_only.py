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
    parser.add_argument("--rank", type=int, default=256)
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
    print(f"\nDecomposing: rank={args.rank}, mode={args.mode}, skip={skip}")
    t0 = time.time()
    spectral_decompose(
        model, rank=args.rank, mode=args.mode,
        skip_patterns=skip, keep_residual=args.keep_residual,
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

    # 4. Save
    print(f"\nSaving to {out_path}...")
    torch.save(model.state_dict(), out_path)
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
