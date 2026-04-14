#!/usr/bin/env python3
"""Phase 1 corpus prep — chunk biographical text into 2k windows, oversample gap categories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavegpt.data_io import write_datafile


def find_source_files(raw_dir: Path) -> list[Path]:
    files = []
    for ext in ("*.txt", "*.md"):
        files.extend(sorted(raw_dir.rglob(ext)))
    return files


def chunk_tokens(tokens: list[int], window: int, overlap: int) -> list[list[int]]:
    if len(tokens) < window:
        return [tokens] if tokens else []
    stride = window - overlap
    chunks = []
    i = 0
    while i + window <= len(tokens):
        chunks.append(tokens[i : i + window])
        i += stride
    if i < len(tokens) and len(tokens) - i > overlap:
        chunks.append(tokens[-window:])
    return chunks


def probe_matches(chunk_text: str, probe_expected: str) -> bool:
    expected = probe_expected.strip().lower()
    if not expected:
        return False
    chunk_lower = chunk_text.lower()
    for token in expected.split():
        token = token.strip(".,;:!?\"'()[]")
        if len(token) >= 3 and token in chunk_lower:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--tier-json", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-4-31b-it")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--oversample-factor", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tier = json.loads(Path(args.tier_json).read_text())
    gap_categories = list(tier.get("gap_categories", []))
    probes = json.loads(Path(args.probes).read_text())["probes"]

    source_files = find_source_files(raw_dir)
    print(f"Found {len(source_files)} source files", flush=True)

    texts = [p.read_text(encoding="utf-8", errors="replace") for p in source_files]
    full_text = "\n\n".join(texts)

    oversampled_categories = []
    for cat in gap_categories:
        cat_probes = [p for p in probes if p.get("category") == cat]
        if any(probe_matches(full_text, p.get("expected", "")) for p in cat_probes):
            oversampled_categories.append(cat)

    if args.dry_run:
        fake_chunks = max(1, len(full_text.split()) // (args.window - args.overlap))
        fake_tokens = fake_chunks * args.window
        manifest = {
            "n_chunks": fake_chunks,
            "n_tokens": fake_tokens,
            "oversampled_categories": oversampled_categories,
            "window": args.window,
            "overlap": args.overlap,
            "source_files": [str(p) for p in source_files],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[dry-run] manifest written to {out_dir / 'manifest.json'}", flush=True)
        return

    from transformers import AutoTokenizer

    print(f"Loading tokenizer {args.tokenizer}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )

    print("Tokenizing corpus", flush=True)
    tokens = tok.encode(full_text, add_special_tokens=False)
    print(f"Total tokens: {len(tokens)}", flush=True)

    chunks = chunk_tokens(tokens, args.window, args.overlap)
    print(f"Base chunks: {len(chunks)}", flush=True)

    extras: list[list[int]] = []
    for cat in oversampled_categories:
        cat_probes = [p for p in probes if p.get("category") == cat]
        for chunk in chunks:
            chunk_text = tok.decode(chunk)
            if any(probe_matches(chunk_text, p.get("expected", "")) for p in cat_probes):
                extras.extend([chunk] * args.oversample_factor)
    print(f"Oversample extras: {len(extras)} chunks", flush=True)

    n_val = max(1, len(chunks) // 20)
    val_chunks = chunks[-n_val:]
    train_chunks = chunks[:-n_val] + extras

    train_tokens: list[int] = [t for c in train_chunks for t in c]
    val_tokens: list[int] = [t for c in val_chunks for t in c]

    write_datafile(str(out_dir / "train.bin"), train_tokens)
    write_datafile(str(out_dir / "val.bin"), val_tokens)

    manifest = {
        "n_chunks": len(train_chunks) + len(val_chunks),
        "n_tokens": len(train_tokens) + len(val_tokens),
        "oversampled_categories": oversampled_categories,
        "window": args.window,
        "overlap": args.overlap,
        "source_files": [str(p) for p in source_files],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(train_tokens)} train / {len(val_tokens)} val tokens", flush=True)
    print(f"Manifest: {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
