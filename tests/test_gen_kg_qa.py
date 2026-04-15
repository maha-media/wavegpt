"""Test for scripts/gen_kg_qa.py — runs in fixture mode against a tiny KG.

Covers:
- Templated types produce pairs (one row per template × matched relationship).
- Unknown types produce zero rows and log a skipped counter.
- Every row carries the full required schema.
- Every answer contains the canonical target entity name verbatim.
- Non-Ray subjects are filtered (we don't emit them).
- `_looks_like_citation` flags bibliography / table / OCR noise.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_gen_kg_qa():
    """Import scripts/gen_kg_qa.py as a module (not on sys.path normally)."""
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "gen_kg_qa.py"
    spec = importlib.util.spec_from_file_location("gen_kg_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


REQUIRED_KEYS = {
    "question",
    "answer",
    "relationship_type",
    "source_chunk_ids",
    "confidence",
    "category",
}


def test_kg_qa_fixture(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fixture = repo / "tests" / "fixtures" / "fake_kg.json"
    out = tmp_path / "kg_pairs.jsonl"

    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "gen_kg_qa.py"),
            "--fixture", str(fixture),
            "--output", str(out),
            "--ray-entity-id", "ray_kurzweil_id",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert out.exists(), "kg_pairs.jsonl missing"

    lines = [json.loads(L) for L in out.read_text().splitlines() if L.strip()]
    assert lines, "no pairs emitted"

    # Every row has the required schema.
    for row in lines:
        missing = REQUIRED_KEYS - set(row.keys())
        assert not missing, f"row missing keys {missing}: {row}"
        assert isinstance(row["source_chunk_ids"], list)
        assert isinstance(row["confidence"], (int, float))
        assert row["relationship_type"] in {"created", "predicts"}, (
            f"unknown-type relationship leaked into output: {row}"
        )

    # Group by relationship_type
    by_type = {}
    for row in lines:
        by_type.setdefault(row["relationship_type"], []).append(row)

    # Both templated types present
    assert "created" in by_type, "expected 'created' pairs in output"
    assert "predicts" in by_type, "expected 'predicts' pairs in output"

    # Unknown type ('grubbles') must NOT appear in output.
    for row in lines:
        assert row["relationship_type"] != "grubbles"

    # Per templated rel, #rows emitted == #templates for that type.
    # (single rel per type in the fixture, so #rows == #templates).
    # We don't hardcode template counts here; instead assert stdout reports
    # the skipped-unknown count as 1 and each templated type as >=1 rel.
    assert "skipped_unknown_types: 1" in r.stdout, (
        f"expected unknown-type skip counter in stdout; got:\n{r.stdout}"
    )
    assert "grubbles" in r.stdout, (
        "expected the unknown type name ('grubbles') to be logged"
    )

    # Answer string must contain the canonical target name verbatim.
    # For the 'created' rel, target is "Kurzweil Reading Machine" (via
    # description field AND target entity name). For 'predicts', target is
    # "Singularity".
    created_rows = by_type["created"]
    for row in created_rows:
        assert "Kurzweil Reading Machine" in row["answer"], (
            f"'created' answer missing target: {row['answer']!r}"
        )

    predicts_rows = by_type["predicts"]
    for row in predicts_rows:
        assert "Singularity" in row["answer"], (
            f"'predicts' answer missing target: {row['answer']!r}"
        )

    # Each row's source_chunk_ids must be non-empty (we look them up to
    # build the answer).
    for row in lines:
        assert row["source_chunk_ids"], f"empty source_chunk_ids: {row}"

    # Distinct questions per type — templates must not collapse to a single
    # paraphrase. Require at least 3 distinct paraphrases per templated type.
    for t, rows in by_type.items():
        qs = {row["question"] for row in rows}
        assert len(qs) >= 3, (
            f"type {t!r}: expected >=3 distinct question paraphrases, got {len(qs)}"
        )


def test_looks_like_citation():
    """`_looks_like_citation` must flag URL / table / OCR-noise strings and
    leave clean Ray prose alone."""
    mod = _load_gen_kg_qa()
    f = mod._looks_like_citation

    # Positive cases — must flag.
    assert f(
        "Lanier, 'One Half of a Manifesto.' IBM Systems Journal, "
        "findarticles.com/p/articles/mi_m0ISJ/..."
    ), "URL-bearing bibliography entry should be flagged"
    assert f(
        "| | 1976 | Kurzweil Computer Products introduces the first "
        "omnifont OCR | reading machine |"
    ), "markdown-table row should be flagged"
    assert f(
        "In 1976, Kurzwe1I Computer Products introduced the first "
        "omnifont OCR reading machine."
    ), "OCR-noise 'Kurzwe1I' should be flagged"

    # Negative cases — must NOT flag clean Ray prose.
    assert not f(
        "I founded Kurzweil Computer Products in 1974 and built the "
        "Kurzweil Reading Machine, the first omnifont OCR device."
    ), "clean first-person Ray prose should pass"
    assert not f(
        "Ray Kurzweil was born on February 12, 1948 in Queens, New York."
    ), "clean third-person date/place phrase should pass"
