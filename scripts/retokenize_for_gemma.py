"""
Re-tokenize RAI corpus from Qwen tokens to Gemma 4 tokens.

Decodes existing Qwen-tokenized .bin → raw text → re-encodes with Gemma tokenizer.

Usage:
    python retokenize_for_gemma.py \
        --source-dir data/rai-qwen \
        --source-tokenizer Qwen/Qwen3.5-27B \
        --target-tokenizer google/gemma-4-31B \
        --output-dir data/rai-gemma4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.data_io import read_datafile, write_datafile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/rai-qwen")
    parser.add_argument("--source-tokenizer", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--target-tokenizer", default="google/gemma-4-31B")
    parser.add_argument("--target-tokenizer-local", default=None,
                        help="Local path to tokenizer files (skip download)")
    parser.add_argument("--output-dir", default="data/rai-gemma4")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizers
    from transformers import AutoTokenizer

    print(f"Loading source tokenizer: {args.source_tokenizer}")
    src_tok = AutoTokenizer.from_pretrained(args.source_tokenizer, trust_remote_code=True)

    target_path = args.target_tokenizer_local or args.target_tokenizer
    print(f"Loading target tokenizer: {target_path}")
    tgt_tok = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)

    print(f"Source vocab: {src_tok.vocab_size}, Target vocab: {tgt_tok.vocab_size}")

    for split in ["train", "val"]:
        src_path = source_dir / f"{split}.bin"
        if not src_path.exists():
            print(f"  Skipping {split} (not found)")
            continue

        print(f"\n  Processing {split}...")
        tokens = read_datafile(str(src_path))
        print(f"    Source: {len(tokens):,} tokens")

        # Decode to text in chunks (avoid OOM on large corpora)
        chunk_size = 100_000
        all_target_tokens = []

        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            text = src_tok.decode(chunk.tolist(), skip_special_tokens=True)
            new_tokens = tgt_tok.encode(text, add_special_tokens=False)
            all_target_tokens.extend(new_tokens)
            if (i // chunk_size) % 10 == 0:
                print(f"    Chunk {i // chunk_size}: {len(all_target_tokens):,} target tokens so far")

        out_path = output_dir / f"{split}.bin"
        write_datafile(str(out_path), all_target_tokens)
        print(f"    Target: {len(all_target_tokens):,} tokens → {out_path}")

    # Save metadata
    meta = {
        "source_tokenizer": args.source_tokenizer,
        "target_tokenizer": args.target_tokenizer,
        "source_dir": str(args.source_dir),
    }
    for split in ["train", "val"]:
        out_path = output_dir / f"{split}.bin"
        if out_path.exists():
            tokens = read_datafile(str(out_path))
            meta[f"{split}_tokens"] = len(tokens)

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved meta.json")
    print("Done.")


if __name__ == "__main__":
    main()
