"""Generate bulk passage-elicit QA pairs from Ray Kurzweil's books.

Tasks 2-3 yield ~650 bio/conceptual pairs — too few for a stable training
signal. Passage elicitation fills in the volume: for every ~200-400-word
chunk of Ray's prose we emit one (question, answer) pair where the answer
is the passage verbatim and the question is a rotating prompt glued to a
topic seed drawn from the passage. The loss mask (Task 5) only penalizes
answer tokens, so the model is trained to produce Ray's prose when asked
about one of its core concepts.

Determinism:
- Books are processed in sorted filename order.
- Paragraphs are emitted in file order.
- Prompts rotate round-robin (no RNG, no sampling).
- Seed lookup scans the seed-list in declaration order; first hit wins.

Strategy:
1. Split each *.txt into paragraphs on blank-line boundaries.
2. Greedily glue consecutive paragraphs until the chunk lands in
   [MIN_WORDS, MAX_WORDS]; truncate any single oversize paragraph at
   MAX_WORDS word boundary.
3. Skip chunks under MIN_WORDS (too short to be coherent Ray prose).
4. For each chunk, find a "topic seed":
     a. Case-insensitive substring match against SEED_CONCEPTS (first hit
        in declaration order).
     b. Fallback: first noun-phrase-ish token run (capitalized word +
        up to 3 trailing lowercase words, OR a single >=6-char token).
     c. If no seed is found, SKIP the chunk (no garbage questions).
5. Pair prompt[i % len(PROMPTS)] + seed + "?" with the verbatim chunk
   as the answer.

Output schema (matches Task 2/3 pairs):
    {
        "question": str,
        "answer": str,              # verbatim passage
        "probe_id": "passage",
        "source_chunk_ids": ["<book-stem>#<paragraph-index>"],
        "confidence": 1.0,
        "category": "passage",
    }

Usage:
    python3 scripts/gen_passage_qa.py \\
        --input-dir data/ray/raw/books \\
        --output data/ray/bio_qa/passage_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---- Tunables ------------------------------------------------------------

MIN_WORDS = 200          # chunk word-count floor (drop below)
MAX_WORDS = 400          # chunk word-count ceiling (stop gluing)
HARD_TRUNCATE = 500      # if a single paragraph exceeds this, truncate
SHORT_SKIP = 100         # paragraphs with < this many words can't stand alone
MIN_SEED_CHARS = 5       # topic-seed length floor (Q must share >=5 chars w/ A)

# Rotating question prompts. Short, elicitation-style. Each ends with a
# space so we can append the seed + "?" cleanly.
PROMPTS: list[str] = [
    "Tell me about ",
    "What do you think about ",
    "Explain ",
    "Your view on ",
    "What is ",
    "Share your thoughts on ",
    "Give me your perspective on ",
    "Talk about ",
]

# Kurzweil's key concepts. Case-insensitive match. Order matters — the
# first hit wins, so we front-load the highest-signal terms.
SEED_CONCEPTS: list[str] = [
    "Singularity",
    "Law of Accelerating Returns",
    "longevity escape velocity",
    "radical life extension",
    "reverse-engineering the brain",
    "pattern recognition",
    "recursive self-improvement",
    "exponential growth",
    "brain uploading",
    "digital immortality",
    "merger with machines",
    "transhumanism",
    "molecular manufacturing",
    "matrioshka brain",
    "technological unemployment",
    "intelligent infrastructure",
    "augmented intelligence",
    "cellular automata",
    "virtual reality",
    "Turing test",
    "nanobots",
    "neocortex",
    "artificial general intelligence",
    "exponential",
    "consciousness",
    "emergence",
    "strong AI",
    "narrow AI",
    "genome",
    "proteome",
    "connectome",
    "computronium",
    "evolution",
    "longevity",
    "intelligence",
]

# Noun-phrase fallback: a capitalized word followed by 0-3 lowercase words,
# OR a single >=6-char token. Used only when no seed concept hits.
_NP_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{2,}(?:\s+[a-z][a-zA-Z]{1,}){0,3})\b"
)
_LONG_TOKEN_RE = re.compile(r"\b([A-Za-z]{6,})\b")

# Sentence-opener function words that should NOT serve as topical seeds.
# If the capitalized-NP fallback's first token is one of these, reject the
# match and keep searching (or fall through to the long-token last resort).
_SEED_STOPWORDS = frozenset({
    "After", "Although", "As", "Because", "Before", "But", "During",
    "However", "If", "Since", "Still", "Then", "Throughout", "Though",
    "Thus", "Until", "When", "While", "Yet",
})


# ---- Chunking ------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split on blank-line runs, strip, drop empties."""
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def _word_count(s: str) -> int:
    return len(s.split())


def _chunk_paragraphs(paragraphs: list[str]) -> list[tuple[str, list[int]]]:
    """Greedy-glue paragraphs into [MIN_WORDS, MAX_WORDS] chunks.

    Returns a list of (chunk_text, paragraph_indices). Chunks under
    MIN_WORDS are dropped. Single paragraphs that exceed MAX_WORDS are
    emitted alone but truncated at HARD_TRUNCATE words.
    """
    chunks: list[tuple[str, list[int]]] = []
    cur_parts: list[str] = []
    cur_idxs: list[int] = []
    cur_words = 0

    def flush():
        nonlocal cur_parts, cur_idxs, cur_words
        if cur_parts and cur_words >= MIN_WORDS:
            chunks.append(("\n\n".join(cur_parts), list(cur_idxs)))
        cur_parts = []
        cur_idxs = []
        cur_words = 0

    for i, p in enumerate(paragraphs):
        w = _word_count(p)

        # Adding this paragraph would overflow: flush first, then start new.
        if cur_words + w > MAX_WORDS and cur_parts:
            flush()

        # Fresh buffer + paragraph already too big: truncate to HARD_TRUNCATE
        # and emit alone. Unconditional — any oversize paragraph gets capped.
        if w > MAX_WORDS and not cur_parts:
            toks = p.split()
            if len(toks) > HARD_TRUNCATE:
                p = " ".join(toks[:HARD_TRUNCATE])
                w = HARD_TRUNCATE
            chunks.append((p, [i]))
            continue

        cur_parts.append(p)
        cur_idxs.append(i)
        cur_words += w

        # Hit the window — flush greedily.
        if cur_words >= MIN_WORDS:
            flush()

    flush()
    return chunks


# ---- Seed extraction -----------------------------------------------------

def _pick_seed(chunk: str) -> str | None:
    """Return the topic seed to use in the question, or None to skip."""
    low = chunk.lower()

    # 1. Seed-list lookup (case-insensitive). First hit in declaration order
    #    among concepts that meet the MIN_SEED_CHARS floor.
    for concept in SEED_CONCEPTS:
        if len(concept) < MIN_SEED_CHARS:
            continue
        if concept.lower() in low:
            # Find the actual-case occurrence in the chunk so the question
            # reads naturally (respecting the passage's own capitalization).
            idx = low.find(concept.lower())
            return chunk[idx:idx + len(concept)]

    # 2. Fallback: first decent noun-phrase (capitalized + tail). Skip any
    #    match whose first token is a sentence-opener function word.
    for m in _NP_RE.finditer(chunk):
        phrase = m.group(1).strip()
        if len(phrase) < MIN_SEED_CHARS:
            continue
        tokens = phrase.split()
        if tokens and tokens[0] in _SEED_STOPWORDS:
            continue
        return phrase

    # 3. Last resort: first >=6-char alpha token.
    m = _LONG_TOKEN_RE.search(chunk)
    if m:
        tok = m.group(1).strip()
        if len(tok) >= MIN_SEED_CHARS:
            return tok

    return None


# ---- Main generator ------------------------------------------------------

def generate_pairs(input_dir: Path) -> tuple[list[dict], Counter]:
    """Walk input_dir/*.txt, chunk, emit pairs. Returns (pairs, stats)."""
    pairs: list[dict] = []
    stats: Counter = Counter()

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"no .txt files found under {input_dir}")

    prompt_cursor = 0

    for path in files:
        stem = path.stem
        text = path.read_text(errors="replace")
        paragraphs = _split_paragraphs(text)
        stats[f"paragraphs:{stem}"] = len(paragraphs)

        chunks = _chunk_paragraphs(paragraphs)
        stats[f"chunks:{stem}"] = len(chunks)

        for chunk_text, para_idxs in chunks:
            seed = _pick_seed(chunk_text)
            if seed is None:
                stats["skipped_no_seed"] += 1
                continue

            prompt = PROMPTS[prompt_cursor % len(PROMPTS)]
            prompt_cursor += 1
            question = f"{prompt}{seed}?"

            chunk_id = f"{stem}#p{para_idxs[0]:04d}"
            if len(para_idxs) > 1:
                chunk_id += f"-p{para_idxs[-1]:04d}"

            pairs.append({
                "question": question,
                "answer": chunk_text,
                "probe_id": "passage",
                "source_chunk_ids": [chunk_id],
                "confidence": 1.0,
                "category": "passage",
            })
            stats["pairs_emitted"] += 1

    return pairs, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir with *.txt books")
    ap.add_argument("--output", required=True, help="output JSONL path")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"ERROR: input-dir not found or not a dir: {in_dir}",
              file=sys.stderr)
        return 2

    try:
        pairs, stats = generate_pairs(in_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    if not pairs:
        print("ERROR: no pairs emitted (all chunks lacked a topic seed?)",
              file=sys.stderr)
        return 4

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- Report ----
    print(f"input_dir: {in_dir}")
    print(f"files: {len(sorted(in_dir.glob('*.txt')))}")
    print(f"pairs_emitted: {stats.get('pairs_emitted', 0)}")
    print(f"skipped_no_seed: {stats.get('skipped_no_seed', 0)}")
    print("per-book paragraphs / chunks:")
    # Iterate the stats with the paragraphs:/chunks: prefixes in pairs.
    book_stems = sorted({k.split(":", 1)[1]
                         for k in stats
                         if k.startswith("paragraphs:")})
    for stem in book_stems:
        p = stats.get(f"paragraphs:{stem}", 0)
        c = stats.get(f"chunks:{stem}", 0)
        print(f"  {stem}: paragraphs={p} chunks={c}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
