"""Test for scripts/gen_passage_qa.py — runs in fixture mode against a tiny
books directory built in tmp_path.

Covers:
- >=6 pairs emitted (2 files x 3 paragraphs each, all >=200 words).
- Every `answer` appears verbatim as a paragraph body from one of the
  fixture files.
- Every `question` ends with '?' and contains a >=5-char substring that
  also appears in the corresponding `answer` (the topic seed).
- Output is JSONL: every line is valid JSON with the full required schema.
"""
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_KEYS = {
    "question",
    "answer",
    "probe_id",
    "source_chunk_ids",
    "confidence",
    "category",
}


# Each paragraph is ~220 words (well above the 100-word floor) and contains at
# least one Kurzweil seed-list concept (Singularity, neocortex, nanobots, etc.)
# so the seed-lookup path is exercised, not the fallback noun-phrase path.
FIXTURE_PARAGRAPHS = {
    "book_alpha.txt": [
        # Paragraph 1 — Singularity
        (
            "The Singularity is the point at which machine intelligence will "
            "exceed biological intelligence by every meaningful measure. "
        ) * 20,
        # Paragraph 2 — neocortex
        (
            "The neocortex is organized as a hierarchy of pattern recognizers "
            "stacked in predictable layers that learn from experience. "
        ) * 20,
        # Paragraph 3 — nanobots
        (
            "Nanobots will eventually travel through our capillaries to repair "
            "cellular damage and extend the healthy human lifespan indefinitely. "
        ) * 20,
    ],
    "book_beta.txt": [
        # Paragraph 1 — exponential
        (
            "Exponential growth in information technology is the engine behind "
            "every curve we track, from compute per dollar to sequencing cost. "
        ) * 20,
        # Paragraph 2 — AGI
        (
            "AGI, or artificial general intelligence, will eventually master "
            "any cognitive task a human can perform and do so at lower cost. "
        ) * 20,
        # Paragraph 3 — consciousness
        (
            "Consciousness remains the deepest puzzle of cognitive science, "
            "and any mature theory of mind must confront it without hand-waving. "
        ) * 20,
    ],
}


def _write_fixture_books(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for fname, paragraphs in FIXTURE_PARAGRAPHS.items():
        body = "\n\n".join(p.strip() for p in paragraphs)
        (root / fname).write_text(body)


def test_passage_qa_fixture(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    books = tmp_path / "books"
    _write_fixture_books(books)
    out = tmp_path / "passage_pairs.jsonl"

    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "gen_passage_qa.py"),
            "--input-dir", str(books),
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert out.exists(), "passage_pairs.jsonl missing"

    lines = [json.loads(L) for L in out.read_text().splitlines() if L.strip()]
    assert len(lines) >= 6, f"expected >=6 pairs, got {len(lines)}"

    # All fixture paragraphs collapsed into a set of verbatim answers we
    # expect the script to have emitted (after the same strip() normalization
    # the script applies). We don't want to overspecify — the script MAY
    # glue adjacent paragraphs into a single chunk if it wants to hit the
    # word-count window. So instead we require that each emitted answer is
    # made up ENTIRELY of whole fixture paragraphs joined with "\n\n".
    fixture_paragraphs: set[str] = set()
    for paragraphs in FIXTURE_PARAGRAPHS.values():
        for p in paragraphs:
            fixture_paragraphs.add(p.strip())

    for row in lines:
        missing = REQUIRED_KEYS - set(row.keys())
        assert not missing, f"row missing keys {missing}: {row}"
        assert row["category"] == "passage"
        assert isinstance(row["source_chunk_ids"], list)
        assert isinstance(row["confidence"], (int, float))

        ans = row["answer"]
        # Decompose answer into its \n\n-separated pieces; each piece must be
        # a verbatim fixture paragraph (after strip()).
        pieces = [p.strip() for p in ans.split("\n\n") if p.strip()]
        for piece in pieces:
            assert piece in fixture_paragraphs, (
                f"answer piece not verbatim from fixture: {piece[:120]!r}..."
            )

        q = row["question"]
        assert q.endswith("?"), f"question not terminated with '?': {q!r}"
        # Find a >=5-char substring shared between question and answer.
        # We can't assume the script uses any particular seed — so we just
        # check that SOME 5-grams of the question appear in the answer.
        shared = False
        for size in (20, 15, 10, 8, 5):
            for i in range(len(q) - size + 1):
                sub = q[i:i + size].strip()
                if len(sub) >= 5 and sub in ans:
                    shared = True
                    break
            if shared:
                break
        assert shared, (
            f"question shares no >=5-char substring with answer; "
            f"q={q!r} a={ans[:120]!r}"
        )


def test_passage_qa_missing_input_dir(tmp_path):
    """Script must fail loudly if --input-dir does not exist."""
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "out.jsonl"
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "gen_passage_qa.py"),
            "--input-dir", str(tmp_path / "does_not_exist"),
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode != 0, "expected nonzero exit for missing input dir"
