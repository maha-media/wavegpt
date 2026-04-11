"""
Test φ-Codec on real model weights.

Loads a few layers from the original model, encodes with φ-codec,
decodes, and measures error vs naive quantization.

Usage:
    python3 -u scripts/test_phi_codec.py --hf-model google/gemma-4-12b-it --layers 5
    python3 -u scripts/test_phi_codec.py --hf-model Qwen/Qwen3.5-27B --layers 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavegpt.phi_codec import PhiCodec, classify_layer, quantize_uniform


def naive_quantize_error(W: np.ndarray, n_bits: int) -> float:
    """Relative Frobenius error from naive uniform quantization."""
    q = quantize_uniform(W, n_bits)
    W_hat = q.dequantize().reshape(W.shape)
    return float(np.sqrt(np.sum((W - W_hat) ** 2)) / np.sqrt(np.sum(W ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of transformer layers to test")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("φ-CODEC TEST: quantization with φ-predicted error correction")
    print("=" * 70)

    # Load model
    print(f"\nLoading {args.hf_model} (first {args.layers} layers)...")
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code,
    )
    # Limit layers for speed
    text_cfg = getattr(config, 'text_config', config)
    orig_layers = getattr(text_cfg, 'num_hidden_layers', 32)
    text_cfg.num_hidden_layers = min(args.layers, orig_layers)
    print(f"  Using {text_cfg.num_hidden_layers}/{orig_layers} layers")

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, config=config,
        torch_dtype=torch.float32,  # need fp32 for accurate SVD
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"  Loaded.\n")

    codec = PhiCodec()

    # Collect all linear layers
    results = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        ltype = classify_layer(name)
        if ltype is None:
            continue

        W = module.weight.data
        m, n = W.shape
        if min(m, n) < 64:
            continue

        print(f"  {name}")
        print(f"    shape: {m}×{n}, type: {ltype}")

        t0 = time.time()
        stats = codec.encode_decode_error(W, layer_type=ltype)
        elapsed = time.time() - t0

        # Compare against naive 4-bit
        W_np = W.detach().cpu().float().numpy()
        naive_4bit = naive_quantize_error(W_np, 4)
        naive_8bit = naive_quantize_error(W_np, 8)

        improvement_vs_4bit = naive_4bit / max(stats['rel_error'], 1e-10)
        improvement_vs_8bit = naive_8bit / max(stats['rel_error'], 1e-10)

        print(f"    φ-codec error:   {stats['rel_error']:.6f} ({stats['rel_error']*100:.4f}%)")
        print(f"    naive 4-bit:     {naive_4bit:.6f} ({naive_4bit*100:.4f}%)")
        print(f"    naive 8-bit:     {naive_8bit:.6f} ({naive_8bit*100:.4f}%)")
        print(f"    φ-codec vs 4bit: {improvement_vs_4bit:.1f}× better")
        print(f"    φ-codec vs 8bit: {improvement_vs_8bit:.1f}× better")
        print(f"    compression:     {stats['compression_ratio']:.2f}×")
        print(f"    storage:         {stats['original_mb']:.1f}MB → {stats['storage_mb']:.1f}MB")
        print(f"    tiers (T1/T2/T3): {stats['tiers']}")
        print(f"    curve (A, k₀):   {stats['curve'][0]:.2f}, {stats['curve'][1]:.1f}")
        print(f"    [{elapsed:.1f}s]")

        results.append({
            'name': name,
            'type': ltype,
            'shape': (m, n),
            **stats,
            'naive_4bit': naive_4bit,
            'naive_8bit': naive_8bit,
            'improvement_4bit': improvement_vs_4bit,
            'improvement_8bit': improvement_vs_8bit,
        })
        print()

    # Summary
    if results:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        mean_phi = np.mean([r['rel_error'] for r in results])
        mean_4bit = np.mean([r['naive_4bit'] for r in results])
        mean_8bit = np.mean([r['naive_8bit'] for r in results])
        mean_ratio = np.mean([r['compression_ratio'] for r in results])
        mean_improve = np.mean([r['improvement_4bit'] for r in results])

        print(f"  Layers tested:        {len(results)}")
        print(f"  Mean φ-codec error:   {mean_phi:.6f} ({mean_phi*100:.4f}%)")
        print(f"  Mean naive 4-bit:     {mean_4bit:.6f} ({mean_4bit*100:.4f}%)")
        print(f"  Mean naive 8-bit:     {mean_8bit:.6f} ({mean_8bit*100:.4f}%)")
        print(f"  Mean improvement:     {mean_improve:.1f}× vs 4-bit")
        print(f"  Mean compression:     {mean_ratio:.2f}×")

        total_orig = sum(r['original_mb'] for r in results)
        total_comp = sum(r['storage_mb'] for r in results)
        print(f"  Total storage:        {total_orig:.1f}MB → {total_comp:.1f}MB")

        # Per-type breakdown
        types_seen = sorted(set(r['type'] for r in results))
        if len(types_seen) > 1:
            print(f"\n  Per-type breakdown:")
            for t in types_seen:
                t_results = [r for r in results if r['type'] == t]
                t_err = np.mean([r['rel_error'] for r in t_results])
                t_imp = np.mean([r['improvement_4bit'] for r in t_results])
                print(f"    {t:12s}: error={t_err:.6f}, {t_imp:.1f}× vs 4-bit")


if __name__ == '__main__':
    main()
