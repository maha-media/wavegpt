"""Test for scripts/gen_probe_qa.py — runs in fixture mode against a tiny
authored YAML.

Covers:
- YAML with N templates × K substitution axes produces >= N*min(axes) pairs,
  capped at MAX_VARIANTS_PER_TEMPLATE per template.
- Every emitted answer contains >=1 expected_anchor for its probe.
- Every row has the full required schema (question, answer, probe_id,
  category, confidence, source_chunk_ids).
- Paraphrase expansion is deterministic (same YAML -> same output order).
- Multiple answer_bodies are rotated round-robin across variants so each
  answer appears at least once when #variants >= #answer_bodies.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_gen_probe_qa():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "gen_probe_qa.py"
    spec = importlib.util.spec_from_file_location("gen_probe_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


REQUIRED_KEYS = {
    "question",
    "answer",
    "probe_id",
    "source_chunk_ids",
    "confidence",
    "category",
}


def test_probe_qa_fixture(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fixture = repo / "tests" / "fixtures" / "fake_probe_manual.yaml"
    out = tmp_path / "probe_pairs.jsonl"

    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "gen_probe_qa.py"),
            "--manual", str(fixture),
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert out.exists(), "probe_pairs.jsonl missing"

    lines = [json.loads(L) for L in out.read_text().splitlines() if L.strip()]
    assert lines, "no pairs emitted"

    # Fixture: template 1 has axes {Ray|Ray Kurzweil|you} (3) and
    # {father|dad} (2) = 6 variants.
    # Template 2 has {father|dad} (2) = 2 variants.
    # Template 3 has {father|dad} (2) = 2 variants.
    # Total: >= 6+2+2 = 10 variants. Test task spec only requires >= 6.
    assert len(lines) >= 6, f"expected >=6 pairs, got {len(lines)}"

    # Every row has the required schema.
    for row in lines:
        missing = REQUIRED_KEYS - set(row.keys())
        assert not missing, f"row missing keys {missing}: {row}"
        assert row["probe_id"] == "bio_test_probe"
        assert row["category"] == "biographical"
        assert isinstance(row["source_chunk_ids"], list)
        assert isinstance(row["confidence"], (int, float))

    # Every answer contains >=1 expected_anchor (Fredric OR pianist).
    anchors = {"Fredric", "pianist"}
    for row in lines:
        hit = any(a in row["answer"] for a in anchors)
        assert hit, (
            f"answer missing all expected_anchors {anchors}: "
            f"{row['answer']!r}"
        )

    # Questions must actually have been expanded (no {...|...} left over).
    for row in lines:
        assert "{" not in row["question"], (
            f"unexpanded substitution in question: {row['question']!r}"
        )
        assert "|" not in row["question"], (
            f"unexpanded pipe in question: {row['question']!r}"
        )

    # Both answer_bodies should appear (round-robin).
    answers = {row["answer"] for row in lines}
    assert len(answers) >= 2, (
        f"expected both answer_body variants to appear; got {len(answers)}"
    )


def test_expand_template_cap():
    """Expander must cap variants at MAX_VARIANTS_PER_TEMPLATE."""
    mod = _load_gen_probe_qa()
    # 5 axes of 3 options each = 243 combos. Must be capped.
    template = "{a|b|c} {d|e|f} {g|h|i} {j|k|l} {m|n|o}"
    variants = mod.expand_template(template)
    assert len(variants) <= mod.MAX_VARIANTS_PER_TEMPLATE
    # And all variants must be fully expanded.
    for v in variants:
        assert "{" not in v and "|" not in v


def test_expand_template_no_subs():
    """Template with no substitution markers returns itself."""
    mod = _load_gen_probe_qa()
    variants = mod.expand_template("Tell me about your father.")
    assert variants == ["Tell me about your father."]


def test_expand_template_deterministic():
    """Same input -> same output (order and values)."""
    mod = _load_gen_probe_qa()
    t = "{Ray|you} {was|is} {born|made}"
    a = mod.expand_template(t)
    b = mod.expand_template(t)
    assert a == b
