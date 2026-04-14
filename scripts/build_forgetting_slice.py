#!/usr/bin/env python3
"""Forgetting-guard slice — held-out wikitext-103 val tokens for Phase 1 CPT.

Tokenizes contiguous examples from `wikitext-103-raw-v1` validation split until
`--target-tokens` is reached (overshoots the final example rather than
truncating it), writes a binary token file via `wavegpt.data_io.write_datafile`,
and emits a manifest with dataset/tokenizer provenance and a SHA of the bytes
written.

`--dry-run` fabricates a deterministic 50k-token buffer (`i % 60000`) and
skips all network calls — used by the offline test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavegpt.data_io import write_datafile


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-tokens", type=int, default=50000)
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "wikitext_val.bin"
    manifest_path = out_dir / "manifest.json"

    if args.dry_run:
        tokens = [i % 60000 for i in range(args.target_tokens)]
        n_examples = 1
        dataset_name = "dry-run"
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        print(f"Loading tokenizer {args.tokenizer}", flush=True)
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code
        )
        print(f"Streaming {args.dataset}/{args.config}:{args.split}", flush=True)
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

        tokens: list[int] = []
        n_examples = 0
        for ex in ds:
            text = ex.get("text", "")
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if not ids:
                continue
            tokens.extend(ids)
            n_examples += 1
            if len(tokens) >= args.target_tokens:
                break
        dataset_name = args.dataset

    write_datafile(str(bin_path), tokens)

    n_tokens = len(tokens)
    overshoot = n_tokens - args.target_tokens
    manifest = {
        "dataset": dataset_name,
        "config": args.config,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "n_tokens": n_tokens,
        "n_examples": n_examples,
        "sha256_bin": _sha256(bin_path),
        "target_tokens": args.target_tokens,
        "overshoot": overshoot,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote {n_tokens} tokens ({n_examples} examples, overshoot={overshoot}) "
        f"to {bin_path}",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
