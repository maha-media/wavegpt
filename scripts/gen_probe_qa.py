"""Expand an authored YAML of Ray-probe templates into a JSONL of Q/A pairs.

The 3 deterministic-miss probes (`bio_birthplace` / `bio_education` /
`bio_father`) are the success criterion for the persona pipeline. They
deserve overweight representation (~500-1000 pairs across the 3). This
script takes a small hand-authored YAML of
(probe, question_templates, answer_bodies, expected_anchors) and emits a
cartesian paraphrase expansion.

Input YAML shape (see data/ray/bio_qa/probe_qa_manual.yaml):
    probes:
      - id: bio_father
        expected_anchors: [Fredric, pianist, conductor, Austrian, Vienna]
        question_templates:
          - "Who was {Ray|Ray Kurzweil|you}'s {father|dad}?"
          - ...
        answer_bodies:
          - |
            My father, Fredric Kurzweil, was an Austrian-Jewish concert
            pianist and conductor. ...
          - |
            ...

Output JSONL (one object per line), schema matches Task 2 (gen_kg_qa.py)
with `probe_id` in place of `relationship_type`:
    {
        "question": str,
        "answer": str,
        "probe_id": str,
        "source_chunk_ids": [],          # probe-authored, no source
        "confidence": 1.0,
        "category": "biographical",
    }

Paraphrase expansion: each question_template may contain
substitution groups written `{opt1|opt2|opt3}`. We expand the cartesian
product across all groups in the template, cap at
MAX_VARIANTS_PER_TEMPLATE=20 (deterministic truncation — itertools.product
preserves insertion order), and rotate answer_bodies round-robin across
the expansions so each body gets ~equal representation.

Usage:
    python3 scripts/gen_probe_qa.py \\
        --manual data/ray/bio_qa/probe_qa_manual.yaml \\
        --output data/ray/bio_qa/probe_pairs.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml


# Cap expansion per template. Plan spec: <=20.
MAX_VARIANTS_PER_TEMPLATE = 20

# Matches {a|b|c} but NOT a bare {foo} (needs a pipe).
_SUB_RE = re.compile(r"\{([^{}]+\|[^{}]+)\}")


def expand_template(template: str) -> list[str]:
    """Expand a template containing `{a|b|c}` substitution groups into
    the cartesian product of concrete strings. Cap at
    MAX_VARIANTS_PER_TEMPLATE. Deterministic: uses itertools.product, no
    random sampling."""
    # Find all substitution groups, left-to-right.
    parts: list[list[str]] = []      # for each slot, list of options
    literal_chunks: list[str] = []   # literal text between slots

    pos = 0
    for m in _SUB_RE.finditer(template):
        literal_chunks.append(template[pos:m.start()])
        options = m.group(1).split("|")
        parts.append([o for o in options])
        pos = m.end()
    literal_chunks.append(template[pos:])

    if not parts:
        return [template]

    # Cartesian product of options.
    variants: list[str] = []
    for combo in itertools.product(*parts):
        out = [literal_chunks[0]]
        for i, opt in enumerate(combo):
            out.append(opt)
            out.append(literal_chunks[i + 1])
        variants.append("".join(out))
        if len(variants) >= MAX_VARIANTS_PER_TEMPLATE:
            break
    return variants


def generate_pairs(manual: dict) -> tuple[list[dict], Counter]:
    """Expand every probe's templates and emit pairs.

    Returns (pairs, per_probe_counts).
    """
    pairs: list[dict] = []
    per_probe = Counter()

    probes = manual.get("probes") or []
    if not probes:
        raise ValueError("manual YAML has no `probes:` list")

    for probe in probes:
        pid = probe.get("id")
        anchors = probe.get("expected_anchors") or []
        templates = probe.get("question_templates") or []
        bodies = probe.get("answer_bodies") or []
        category = probe.get("category", "biographical")

        if not pid:
            raise ValueError(f"probe missing id: {probe}")
        if not templates:
            raise ValueError(f"probe {pid!r} has no question_templates")
        if not bodies:
            raise ValueError(f"probe {pid!r} has no answer_bodies")
        if not anchors:
            raise ValueError(f"probe {pid!r} has no expected_anchors")

        # Clean bodies (collapse trailing newline, strip).
        cleaned_bodies = [b.strip() for b in bodies if b and b.strip()]
        if not cleaned_bodies:
            raise ValueError(f"probe {pid!r} has no non-empty answer_bodies")

        # Sanity: every body mentions >=1 anchor (fail fast; don't let a
        # bad-author YAML ship pairs that fail the probe).
        for i, body in enumerate(cleaned_bodies):
            if not any(a in body for a in anchors):
                raise ValueError(
                    f"probe {pid!r} answer_body[{i}] contains none of "
                    f"expected_anchors {anchors}"
                )

        # Round-robin counter for bodies, across ALL variants of this probe.
        body_cursor = 0

        for template in templates:
            variants = expand_template(template)
            for q in variants:
                if "{" in q or "}" in q or "|" in q:
                    raise ValueError(
                        f"probe {pid!r} template left unexpanded markers: {q!r}"
                    )
                body = cleaned_bodies[body_cursor % len(cleaned_bodies)]
                body_cursor += 1
                pairs.append({
                    "question": q,
                    "answer": body,
                    "probe_id": pid,
                    "source_chunk_ids": [],
                    "confidence": 1.0,
                    "category": category,
                })
                per_probe[pid] += 1

    return pairs, per_probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True, help="authored YAML path")
    ap.add_argument("--output", required=True, help="output JSONL path")
    args = ap.parse_args()

    manual_path = Path(args.manual)
    if not manual_path.exists():
        print(f"ERROR: manual not found: {manual_path}", file=sys.stderr)
        return 2

    manual = yaml.safe_load(manual_path.read_text())

    try:
        pairs, per_probe = generate_pairs(manual)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- Report ----
    print(f"manual: {manual_path}")
    print(f"probes: {len(manual.get('probes') or [])}")
    print(f"pairs_emitted: {len(pairs)}")
    for pid, n in per_probe.most_common():
        print(f"  {pid}: {n} pairs")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
