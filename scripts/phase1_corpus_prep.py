#!/usr/bin/env python3
"""Phase 1 corpus prep — chunk biographical text into 2k windows, oversample gap categories.

Reads raw `.txt`/`.md` sources under `--raw-dir`, tokenizes with the HF tokenizer,
splits into overlapping chunks of `--window` tokens (stride = window - overlap),
and duplicates chunks that contain matches for any probe in a gap category (as
listed in `--tier-json`) `--oversample-factor` times. Base chunks are shuffled
with a fixed seed before the last 5% are reserved as val; extras are appended
to train only. Emits `train.bin`, `val.bin`, and `manifest.json`.

Usage:
    python3 scripts/phase1_corpus_prep.py \\
        --raw-dir data/ray/raw \\
        --tier-json runs/probes/ray_baseline/tier.json \\
        --probes runs/probes/ray_baseline/probe_baseline.json \\
        --output-dir data/ray/phase1

`--dry-run` skips tokenizer load and bin writes (for offline tests / CI); it
writes only a manifest whose `oversampled_categories` is computed by the same
chunk-based matcher the real path uses (word-group simulated chunks over the
raw text).
"""
from __future__ import annotations

import argparse
import json
import random
import re
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


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Drop stopwords + subject-name tokens so a Ray corpus doesn't match every
# chunk on "Kurzweil". English function words and a few geographic/temporal
# fillers that show up constantly in biographical prose.
_STOPWORDS = frozenset([
    "ray", "kurzweil", "raymond",
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
    "not", "any", "all", "our", "who", "was", "are", "but", "one", "two",
    "has", "had", "have", "been", "were", "what", "when", "where", "why",
    "how", "just", "only", "some", "more", "such", "also", "can", "will",
    "new", "york", "city", "year", "years", "time", "born", "his", "her",
    "him", "she", "they", "them", "their", "there", "about", "after",
    "before", "because", "which", "while", "would", "could", "should",
])


def _rare_tokens(text: str) -> set[str]:
    """Extract distinctive tokens from `text`: lower-cased alnum runs, with
    stopwords + subject-name tokens removed. Words must be ≥5 chars;
    all-digit tokens ≥4 chars (so "1948", "2045" survive but "1st" doesn't).
    """
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if tok.isdigit():
            if len(tok) >= 4:
                out.add(tok)
        elif len(tok) >= 5:
            out.add(tok)
    return out


def probe_matches(chunk_text: str, probe_expected: str) -> bool:
    """Return True if `chunk_text` contains ≥2 distinct rare tokens from
    `probe_expected`. Rare = not a stopword, not a subject-name token, word
    ≥5 chars or digit-run ≥4. The two-token floor prevents a single token
    like "Kurzweil" (stripped by the stoplist) or a generic word like "new"
    from triggering a whole-corpus match.
    """
    expected_tokens = _rare_tokens(probe_expected)
    if len(expected_tokens) < 2:
        return False
    chunk_lower = chunk_text.lower()
    hits = 0
    for tok in expected_tokens:
        if tok in chunk_lower:
            hits += 1
            if hits >= 2:
                return True
    return False


def detect_oversampled_categories(
    chunk_texts: list[str],
    gap_categories: list[str],
    probes: list[dict],
) -> list[str]:
    """Return the subset of `gap_categories` for which at least one chunk text
    matches at least one probe in that category. Single source of truth shared
    by the dry-run and real code paths.
    """
    result = []
    for cat in gap_categories:
        cat_probes = [p for p in probes if p.get("category") == cat]
        if not cat_probes:
            continue
        for chunk_text in chunk_texts:
            if any(probe_matches(chunk_text, p.get("expected", "")) for p in cat_probes):
                result.append(cat)
                break
    return result


def fake_word_chunks(full_text: str, window: int, overlap: int) -> list[str]:
    """Dry-run stand-in for tokenized chunks: split `full_text` on whitespace
    and group into window-sized word groups with the same stride the real path
    uses. Keeps dry-run and real path match-detection logic identical.
    """
    words = full_text.split()
    if not words:
        return []
    if len(words) < window:
        return [" ".join(words)]
    stride = max(1, window - overlap)
    chunks = []
    i = 0
    while i + window <= len(words):
        chunks.append(" ".join(words[i : i + window]))
        i += stride
    if i < len(words) and len(words) - i > overlap:
        chunks.append(" ".join(words[-window:]))
    return chunks


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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tier = json.loads(Path(args.tier_json).read_text())
    gap_categories = list(tier.get("gap_categories", []))
    probes = json.loads(Path(args.probes).read_text())["probes"]

    source_files = find_source_files(raw_dir)
    print(f"Found {len(source_files)} source files", flush=True)
    if not source_files:
        print(f"No .txt/.md files under {raw_dir}; aborting.", flush=True)
        sys.exit(1)

    texts = [p.read_text(encoding="utf-8", errors="replace") for p in source_files]
    full_text = "\n\n".join(texts)

    if args.dry_run:
        sim_chunks = fake_word_chunks(full_text, args.window, args.overlap)
        oversampled_categories = detect_oversampled_categories(
            sim_chunks, gap_categories, probes
        )
        fake_n_chunks = max(1, len(sim_chunks))
        fake_tokens = fake_n_chunks * args.window
        manifest = {
            "n_chunks": fake_n_chunks,
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

    chunk_texts = [tok.decode(c) for c in chunks]
    oversampled_categories = detect_oversampled_categories(
        chunk_texts, gap_categories, probes
    )

    extras: list[list[int]] = []
    for cat in oversampled_categories:
        cat_probes = [p for p in probes if p.get("category") == cat]
        for chunk, chunk_text in zip(chunks, chunk_texts):
            if any(probe_matches(chunk_text, p.get("expected", "")) for p in cat_probes):
                extras.extend([chunk] * args.oversample_factor)
    print(f"Oversample extras: {len(extras)} chunks", flush=True)

    # Shuffle base chunks with a fixed seed so val isn't tied to file order.
    # Extras are appended to train post-split — do NOT shuffle them in.
    rng = random.Random(args.seed)
    shuffled = list(chunks)
    rng.shuffle(shuffled)

    n_val = max(1, len(shuffled) // 20)
    val_chunks = shuffled[-n_val:]
    train_chunks = shuffled[:-n_val] + extras

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
