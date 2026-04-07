#!/usr/bin/env python3
"""
Export RAI corpus tokenized with any HuggingFace tokenizer.

Reads existing binary data (GPT-2 tokenized), decodes back to text,
re-tokenizes with target model's tokenizer, writes train/val .bin files.

Usage:
    python scripts/export_rai_for_hf.py \
        --source-dir data/rai-corpus \
        --output-dir data/rai-qwen \
        --tokenizer Qwen/Qwen3.5-27B \
        --val-ratio 0.1
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.data_io import read_datafile, write_datafile


def decode_gpt2_tokens(tokens):
    """Decode GPT-2 tokenized data back to text."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    # Decode in chunks to handle potential issues
    text_parts = []
    chunk_size = 100_000
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size].tolist()
        try:
            text_parts.append(enc.decode(chunk, errors='replace'))
        except Exception:
            text_parts.append(enc.decode(chunk, errors='replace'))
    return ''.join(text_parts)


def tokenize_with_hf(text, tokenizer_name, trust_remote_code=False):
    """Tokenize text with a HuggingFace tokenizer."""
    from transformers import AutoTokenizer
    print(f"  Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Tokenizing...")
    tokens = tokenizer.encode(text)
    print(f"  {len(tokens):,} tokens")
    return np.array(tokens, dtype=np.uint32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Dir with GPT-2 tokenized train/val .bin files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output dir for re-tokenized files")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HuggingFace tokenizer name or path")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source_dir)

    # Load and decode source tokens
    print("Loading source data...")
    for name in ["sft_train.bin", "train.bin"]:
        if (src / name).exists():
            train_tokens = read_datafile(str(src / name))
            print(f"  Train: {len(train_tokens):,} GPT-2 tokens from {name}")
            break
    else:
        raise FileNotFoundError(f"No train data in {src}")

    for name in ["sft_val.bin", "val.bin"]:
        if (src / name).exists():
            val_tokens = read_datafile(str(src / name))
            print(f"  Val: {len(val_tokens):,} GPT-2 tokens from {name}")
            break
    else:
        val_tokens = None
        print("  No val data found, will split from train")

    # Decode to text
    print("\nDecoding GPT-2 tokens to text...")
    train_text = decode_gpt2_tokens(train_tokens)
    print(f"  Train text: {len(train_text):,} chars")

    if val_tokens is not None:
        val_text = decode_gpt2_tokens(val_tokens)
        print(f"  Val text: {len(val_text):,} chars")
    else:
        # Split
        split_idx = int(len(train_text) * (1 - args.val_ratio))
        val_text = train_text[split_idx:]
        train_text = train_text[:split_idx]
        print(f"  Split: {len(train_text):,} train / {len(val_text):,} val chars")

    # Re-tokenize with target tokenizer
    print(f"\nRe-tokenizing with {args.tokenizer}...")
    train_new = tokenize_with_hf(train_text, args.tokenizer, args.trust_remote_code)
    val_new = tokenize_with_hf(val_text, args.tokenizer, args.trust_remote_code)

    # Write output
    print(f"\nWriting to {out}/...")
    write_datafile(str(out / "train.bin"), train_new.tolist())
    write_datafile(str(out / "val.bin"), val_new.tolist())

    # Stats
    meta = {
        "tokenizer": args.tokenizer,
        "train_tokens": len(train_new),
        "val_tokens": len(val_new),
        "source_dir": str(src),
    }
    import json
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Train: {len(train_new):,} tokens ({(out / 'train.bin').stat().st_size / 1024:.0f} KB)")
    print(f"  Val:   {len(val_new):,} tokens ({(out / 'val.bin').stat().st_size / 1024:.0f} KB)")
    print("  Done!")


if __name__ == "__main__":
    main()
