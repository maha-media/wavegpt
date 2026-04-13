"""
Test fine-tuned Gemma 4 inference: load decomposed model + spectral checkpoint.

Memory-efficient: loads shards one at a time, uses spectral checkpoint for rank inference.

Usage:
    python3 -u scripts/test_gemma_inference.py \
        --decomposed-dir runs/gemma4-decomposed \
        --checkpoint runs/gemma4-rai-harmonic/best_spectral.pt \
        --config-dir /workspace/gemma4-config
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavegpt.spectral_surgery import spectral_scaffold
from wavegpt.spectral_linear import SpectralLinear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposed-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--prompts", nargs="+", default=[
        "The most important thing about the Singularity is",
        "When I think about my father, I remember",
        "The golden ratio appears in nature because",
    ])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="Skip fine-tuned checkpoint — test raw decomposed model")
    parser.add_argument("--recompose", action="store_true",
                        help="Convert SpectralLinear back to nn.Linear (test decomp quality)")
    args = parser.parse_args()

    decomp_dir = Path(args.decomposed_dir)
    shard_dir = decomp_dir / "shards"

    # Step 1: Get rank info for scaffold
    print("=" * 70)
    print("STEP 1: Load rank info for scaffold")
    print("=" * 70)
    if not args.skip_checkpoint:
        # Use fine-tuned checkpoint for rank inference (it has .spectrum keys)
        spectral_sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        print(f"  Spectral checkpoint: {len(spectral_sd)} keys")
        rank_sd = spectral_sd  # use for scaffold
    else:
        # Get rank info from decomposed shards (look at .log_spectrum/.spectrum keys)
        print(f"  --skip-checkpoint: extracting ranks from decomposed shards")
        from safetensors.torch import load_file
        with open(shard_dir / "index.json") as f:
            index = json.load(f)
        rank_sd = {}
        for shard_name in index["shards"]:
            shard = load_file(str(shard_dir / shard_name), device="cpu")
            for k, v in shard.items():
                if k.endswith(".log_spectrum") or k.endswith(".spectrum"):
                    rank_sd[k] = v
            del shard
        print(f"  Found {len(rank_sd)} spectrum keys in shards")
        spectral_sd = None
    rank_vals = [v.shape[0] for v in rank_sd.values()]
    print(f"  Rank range: [{min(rank_vals)}, {max(rank_vals)}], median={sorted(rank_vals)[len(rank_vals)//2]}")

    # Step 2: Create model from config (skip random weight init for speed)
    print("\n" + "=" * 70)
    print("STEP 2: Create model skeleton from config (skip random init)")
    print("=" * 70)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    text_cfg = getattr(config, 'text_config', config)
    hidden = getattr(text_cfg, 'hidden_size', '?')
    print(f"  Config: {config.model_type}, hidden={hidden}")
    t0 = time.time()
    # Get the actual model class to monkey-patch _init_weights
    from transformers import AutoModelForCausalLM as Auto
    model_cls = Auto._model_mapping[type(config)]
    orig_init_weights = model_cls._init_weights
    model_cls._init_weights = lambda self, module: None  # no-op: skip random fill
    try:
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
    finally:
        model_cls._init_weights = orig_init_weights  # restore
    elapsed = time.time() - t0
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created: {total_params:,} params ({total_params*2/1e9:.1f}GB bf16) [{elapsed:.1f}s]")

    # Step 3: Scaffold — replace nn.Linear with SpectralLinear shells
    print("\n" + "=" * 70)
    print("STEP 3: Scaffold SpectralLinear architecture")
    print("=" * 70)
    skip = ['embed_tokens', 'lm_head', 'visual', 'vision', 'wte', 'wpe']
    spectral_scaffold(model, rank=256, mode='per_mode',
                      skip_patterns=skip, state_dict=rank_sd)

    # Verify scaffold
    n_spectral = sum(1 for m in model.modules() if isinstance(m, SpectralLinear))
    print(f"  SpectralLinear layers: {n_spectral}")

    # Step 4: Load decomposed shards one by one
    print("\n" + "=" * 70)
    print("STEP 4: Load decomposed weights (shard by shard)")
    print("=" * 70)
    with open(shard_dir / "index.json") as f:
        index = json.load(f)

    total_loaded = 0
    total_skipped = 0
    model_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())

    for shard_name in index["shards"]:
        from safetensors.torch import load_file
        shard_path = str(shard_dir / shard_name)
        t0 = time.time()
        shard_sd = load_file(shard_path, device="cpu")
        elapsed = time.time() - t0

        # Check which keys match
        matched = set(shard_sd.keys()) & model_keys
        unmatched = set(shard_sd.keys()) - model_keys
        total_loaded += len(matched)
        total_skipped += len(unmatched)

        # Apply this shard
        model.load_state_dict(shard_sd, strict=False)
        print(f"  {shard_name}: {len(shard_sd)} keys ({len(matched)} matched, "
              f"{len(unmatched)} skipped) [{elapsed:.1f}s]")

        if unmatched and len(unmatched) <= 5:
            for k in sorted(unmatched):
                print(f"    SKIP: {k}")
        elif unmatched:
            sample = sorted(unmatched)[:3]
            print(f"    SKIP sample: {sample} (+ {len(unmatched)-3} more)")

        del shard_sd
        gc.collect()

    print(f"\n  TOTAL: {total_loaded} loaded, {total_skipped} skipped")

    # Step 5: Override spectrum with fine-tuned values (if not skipped)
    print("\n" + "=" * 70)
    if args.skip_checkpoint:
        print("STEP 5: SKIPPED — using original decomposed spectrum")
        print("=" * 70)
        print("  (testing raw decomposition quality, no fine-tuning applied)")
    else:
        print("STEP 5: Apply fine-tuned spectral checkpoint")
        print("=" * 70)
        applied = 0
        shape_mismatch = 0
        for key, val in spectral_sd.items():
            parts = key.split('.')
            obj = model
            try:
                for part in parts[:-1]:
                    obj = getattr(obj, part) if not part.isdigit() else obj[int(part)]
                param_name = parts[-1]
                current = getattr(obj, param_name)
                if current.shape == val.shape:
                    if isinstance(current, torch.nn.Parameter):
                        current.data.copy_(val)
                    else:
                        setattr(obj, param_name, val)
                    applied += 1
                else:
                    print(f"  SHAPE MISMATCH: {key}: model={current.shape} vs ckpt={val.shape}")
                    shape_mismatch += 1
            except (AttributeError, IndexError) as e:
                print(f"  MISSING: {key}: {e}")
        print(f"  Applied: {applied}/{len(spectral_sd)} spectral params")
        if shape_mismatch:
            print(f"  Shape mismatches: {shape_mismatch}")

    # Step 5b: Recompose SpectralLinear → nn.Linear (if requested)
    if args.recompose:
        print("\n" + "=" * 70)
        print("STEP 5b: RECOMPOSE SpectralLinear → nn.Linear")
        print("=" * 70)
        recomp_count = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, SpectralLinear):
                linear = module.to_linear()
                # Navigate to parent and replace
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
                setattr(parent, parts[-1], linear)
                recomp_count += 1
        n_remaining = sum(1 for m in model.modules() if isinstance(m, SpectralLinear))
        print(f"  Recomposed {recomp_count} layers back to nn.Linear")
        print(f"  SpectralLinear remaining: {n_remaining}")

    # Step 6: Move to GPU and generate
    print("\n" + "=" * 70)
    print("STEP 6: Generate text")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Moving to {device}...")
    model.to(device)
    model.requires_grad_(False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt in args.prompts:
        print(f"\n  Q: {prompt}")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            # Check initial logits
            kwargs = {}
            if 'gemma' in getattr(config, 'model_type', ''):
                kwargs['mm_token_type_ids'] = torch.zeros_like(input_ids)
            outputs = model(input_ids=input_ids, **kwargs)
            logits = outputs.logits
            top5 = torch.topk(logits[0, -1], 5)
            print(f"  Top-5 next tokens: {[tokenizer.decode([t]) for t in top5.indices.tolist()]}")
            print(f"  Top-5 logits: {top5.values.tolist()}")

            # Generate
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                mm_token_type_ids=torch.zeros_like(input_ids) if 'gemma' in getattr(config, 'model_type', '') else None,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  A: {text[len(prompt):][:300]}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
