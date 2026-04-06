"""
Export Step-3.5-Flash-SFT dataset to WaveGPT training format.

Downloads from HuggingFace, classifies conversations into harmonic layers
(C→G→D→A), tokenizes with loss masks, writes .bin + .npy files.

Usage:
    python scripts/export_sft.py --max-examples 50000 --output-dir data/sft-50k
    python scripts/export_sft.py --max-examples 50000 --output-dir data/sft-50k --harmonic-order
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import datasets

# Add parent to path for wavegpt imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavegpt.sft_dataloader import tokenize_conversation, classify_harmonic_layer
from wavegpt.spectral_order import compute_spectral_order
from wavegpt.data_io import write_datafile


def parse_conversations(example: dict) -> list[dict]:
    """Parse conversations from HF dataset example."""
    convos = example.get("conversations", [])
    turns = []
    for turn in convos:
        t = json.loads(turn) if isinstance(turn, str) else turn
        turns.append(t)
    return turns


def main():
    parser = argparse.ArgumentParser(description="Export SFT dataset for WaveGPT")
    parser.add_argument("--max-examples", type=int, default=50000,
                        help="Max conversations to export")
    parser.add_argument("--output-dir", type=str, default="data/sft-50k",
                        help="Output directory for .bin files")
    parser.add_argument("--harmonic-order", action="store_true",
                        help="Order by circle of fifths walk through embedding space")
    parser.add_argument("--spectral-order", action="store_true",
                        help="Order by spectral walk (SVD + circle of fifths)")
    parser.add_argument("--include-reasoning", action="store_true", default=True,
                        help="Include reasoning_content in tokenization")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="Exclude reasoning_content")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Fraction for validation (default 5%%)")
    args = parser.parse_args()

    include_reasoning = not args.no_reasoning
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming stepfun-ai/Step-3.5-Flash-SFT...")
    print(f"Max examples: {args.max_examples:,}")
    print(f"Include reasoning: {include_reasoning}")
    print(f"Harmonic ordering: {args.harmonic_order}")
    print(f"Spectral ordering: {args.spectral_order}")
    print()

    # ── Stream and classify ──
    ds = datasets.load_dataset(
        "stepfun-ai/Step-3.5-Flash-SFT",
        split="train",
        streaming=True,
    )

    all_conversations = []
    layer_counts = Counter()
    total_skipped = 0
    t0 = time.time()

    for i, example in enumerate(ds):
        if i >= args.max_examples:
            break

        turns = parse_conversations(example)
        if not turns:
            total_skipped += 1
            continue

        layer = classify_harmonic_layer(turns)
        all_conversations.append(turns)
        layer_counts[layer] += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.max_examples - i - 1) / rate
            counts_str = ' '.join(f"{k}={v}" for k, v in sorted(layer_counts.items()))
            print(f"  [{i+1:>7,}/{args.max_examples:,}] "
                  f"{counts_str} "
                  f"({rate:.0f} ex/s, ETA {eta:.0f}s)")

    total = len(all_conversations)
    elapsed = time.time() - t0
    print(f"\nLoaded {total:,} conversations in {elapsed:.1f}s (skipped {total_skipped})")
    from wavegpt.sft_dataloader import CIRCLE_OF_FIFTHS
    for layer in CIRCLE_OF_FIFTHS:
        n = layer_counts.get(layer, 0)
        if n > 0:
            print(f"  {layer:>3s}: {n:>6,} ({100*n/total:.1f}%)")

    # ── Build conversation ordering ──
    ordered_tokenized = None
    ordered = None
    if args.spectral_order:
        print("\nOrdering: spectral walk (SVD + circle of fifths)")
        print("  Pre-tokenizing for spectral analysis...")
        pre_tokenized = []
        for j, conv in enumerate(all_conversations):
            tokens, mask = tokenize_conversation(conv, include_reasoning=include_reasoning)
            pre_tokenized.append((tokens, mask))
            if (j + 1) % 10000 == 0:
                print(f"    [{j+1:,}/{total:,}]")
        token_seqs = [t for t, _ in pre_tokenized]
        order = compute_spectral_order(token_seqs)
        ordered_tokenized = [pre_tokenized[i] for i in order]
    elif args.harmonic_order:
        # Legacy: sort by classified layer
        print("\nOrdering: classified harmonic layers")
        classified = {}
        for conv in all_conversations:
            layer = classify_harmonic_layer(conv)
            classified.setdefault(layer, []).append(conv)
        ordered = []
        for note in CIRCLE_OF_FIFTHS:
            ordered.extend(classified.get(note, []))
    else:
        print("\nOrdering: shuffled (random baseline)")
        import random
        random.seed(42)
        ordered = list(all_conversations)
        random.shuffle(ordered)

    # ── Tokenize (or use pre-tokenized for spectral order) ──
    all_tokens = []
    all_masks = []
    t0 = time.time()

    if args.spectral_order and ordered_tokenized:
        print(f"\nFlattening {len(ordered_tokenized):,} pre-tokenized conversations...")
        for i, (tokens, mask) in enumerate(ordered_tokenized):
            all_tokens.extend(tokens)
            all_masks.extend(mask)
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:>7,}/{len(ordered_tokenized):,}] "
                      f"{len(all_tokens):,} tokens so far ({elapsed:.1f}s)")
    else:
        print(f"\nTokenizing {len(ordered):,} conversations...")
        for i, turns in enumerate(ordered):
            tokens, mask = tokenize_conversation(turns, include_reasoning=include_reasoning)
            all_tokens.extend(tokens)
            all_masks.extend(mask)
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:>7,}/{len(ordered):,}] "
                      f"{len(all_tokens):,} tokens so far ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    total_tokens = len(all_tokens)
    active_tokens = sum(all_masks)
    print(f"\nTokenized: {total_tokens:,} tokens in {elapsed:.1f}s")
    print(f"  Loss-active: {active_tokens:,} ({100*active_tokens/total_tokens:.1f}%)")
    print(f"  Masked: {total_tokens - active_tokens:,} ({100*(1-active_tokens/total_tokens):.1f}%)")

    # ── Train/val split ──
    val_size = int(total_tokens * args.val_split)
    train_size = total_tokens - val_size

    train_tokens = all_tokens[:train_size]
    train_masks = all_masks[:train_size]
    val_tokens = all_tokens[train_size:]
    val_masks = all_masks[train_size:]

    print(f"\nSplit: train={train_size:,} val={val_size:,}")

    # ── Write files ──
    print(f"\nWriting to {output_dir}/...")

    write_datafile(str(output_dir / "sft_train.bin"), train_tokens)
    write_datafile(str(output_dir / "sft_val.bin"), val_tokens)
    np.save(str(output_dir / "sft_train_mask.npy"),
            np.array(train_masks, dtype=np.uint8))
    np.save(str(output_dir / "sft_val_mask.npy"),
            np.array(val_masks, dtype=np.uint8))

    # ── Write metadata ──
    ordering = "spectral" if args.spectral_order else "harmonic" if args.harmonic_order else "random"
    meta = {
        "source": "stepfun-ai/Step-3.5-Flash-SFT",
        "max_examples": args.max_examples,
        "total_conversations": total,
        "ordering": ordering,
        "harmonic_order": args.harmonic_order,
        "spectral_order": args.spectral_order,
        "include_reasoning": include_reasoning,
        "layer_counts": dict(layer_counts),
        "total_tokens": total_tokens,
        "train_tokens": train_size,
        "val_tokens": val_size,
        "loss_active_pct": round(100 * active_tokens / total_tokens, 1),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Files:")
    for p in sorted(output_dir.iterdir()):
        size = p.stat().st_size
        unit = "MB" if size > 1e6 else "KB"
        val = size / 1e6 if size > 1e6 else size / 1e3
        print(f"  {p.name}: {val:.1f} {unit}")


if __name__ == "__main__":
    main()
