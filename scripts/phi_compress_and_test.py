"""
Full φ-Codec pipeline: compress a model and test inference.

1. Load original model from HuggingFace
2. φ-encode every linear layer (full SVD → φ-curve → tiered quantization)
3. φ-decode back to nn.Linear (recompose W = U·diag(S)·V^T)
4. Generate text — does the recomposed model still talk?

Usage:
    python3 -u scripts/phi_compress_and_test.py \
        --hf-model google/gemma-4-12b-it --layers 5 \
        --trust-remote-code
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavegpt.phi_codec import PhiCodec, classify_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--layers", type=int, default=None,
                        help="Limit transformer layers (None = all)")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompts", nargs="+", default=[
        "The most important thing about the Singularity is",
        "When I think about my father, I remember",
        "The golden ratio appears in nature because",
    ])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    print("=" * 70)
    print("φ-CODEC: COMPRESS → RECOMPOSE → GENERATE")
    print("=" * 70)

    # Step 1: Load model
    print(f"\n[STEP 1] Loading {args.hf_model}...")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code,
    )
    if args.layers is not None:
        text_cfg = getattr(config, 'text_config', config)
        orig_layers = getattr(text_cfg, 'num_hidden_layers', 32)
        text_cfg.num_hidden_layers = min(args.layers, orig_layers)
        print(f"  Using {text_cfg.num_hidden_layers}/{orig_layers} layers")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Baseline generation (original model)
    print(f"\n[STEP 2] Baseline generation (original model)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.requires_grad_(False)

    for prompt in args.prompts[:1]:  # just one for speed
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  Q: {prompt}")
        print(f"  A: {text[len(prompt):][:200]}")

    # Move back to CPU for SVD
    model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: φ-encode + decode (recompose) every linear layer
    print(f"\n[STEP 3] φ-encode → decode all linear layers...")
    codec = PhiCodec()

    total_original = 0
    total_compressed = 0
    errors = []
    recomposed = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        ltype = classify_layer(name)
        m, n = module.weight.shape
        min_dim = min(m, n)

        if ltype is None or min_dim < 64:
            skipped += 1
            continue

        t0 = time.time()
        W = module.weight.data.float()  # fp32 for accurate SVD
        bias = module.bias.data.float() if module.bias is not None else None

        # Encode
        comp = codec.encode_layer(W, layer_type=ltype, bias=bias)

        # Decode (recompose)
        W_hat = codec.decode_layer(comp)

        # Error
        rel_err = float(np.sqrt(np.sum((W.numpy() - W_hat) ** 2)) /
                        np.sqrt(np.sum(W.numpy() ** 2)))
        errors.append(rel_err)
        total_original += comp.original_bytes()
        total_compressed += comp.storage_bytes()

        # Replace weight in-place
        module.weight.data = torch.from_numpy(W_hat).to(torch.bfloat16)
        elapsed = time.time() - t0

        recomposed += 1
        if recomposed <= 10 or recomposed % 50 == 0:
            print(f"  [{recomposed}] {name} ({ltype}) "
                  f"{m}×{n} err={rel_err*100:.4f}% [{elapsed:.1f}s]")

    print(f"\n  Recomposed: {recomposed} layers, skipped: {skipped}")
    print(f"  Mean error: {np.mean(errors)*100:.4f}%")
    print(f"  Max error:  {np.max(errors)*100:.4f}%")
    print(f"  Storage: {total_original/1e9:.2f}GB → {total_compressed/1e9:.2f}GB "
          f"({total_original/max(total_compressed,1):.2f}× compression)")

    # Step 4: Generate with recomposed model
    print(f"\n[STEP 4] Generation with φ-recomposed model...")
    model.to(device)

    for prompt in args.prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            top5 = torch.topk(logits[0, -1], 5)
            print(f"\n  Q: {prompt}")
            print(f"  Top-5: {[tokenizer.decode([t]) for t in top5.indices.tolist()]}")

            out = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  A: {text[len(prompt):][:200]}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
