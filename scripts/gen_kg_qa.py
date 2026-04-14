"""Generate QA pairs from Ray Kurzweil KG relationships.

Emits one JSON object per line to --output, following the schema:
    {
        "question": str,
        "answer": str,
        "relationship_type": str,
        "source_chunk_ids": list[str],
        "confidence": float,
        "category": str,
    }

Scope: ONLY emits pairs where `source_entity` resolves to Ray Kurzweil
(by entity_id, exact name match, or alias match — case-insensitive).

Types templated: only those that actually exist in the live digital_rai KG
in meaningful volume AND plausibly map to biographical QA. After inspecting
the real collection, these are:
    - 'created'  (Ray founded/invented X)     -> category=biographical
    - 'predicts' (Ray predicts/forecasts X)   -> category=conceptual
    - 'influences' (Ray's work influenced X)  -> category=biographical

Every other `type` value is LOGGED and SKIPPED (no pairs emitted).

Plan-level types that the KG does NOT have (father_of, educated_at,
born_in, authored, worked_at, believes) are intentionally absent from the
template table — biographical facts like "father=Fredric", "born=Queens",
"attended=MIT" live in chunk text and will be extracted by Task 4 (passage
elicitation), not from explicit graph relations.

Usage:
    # live Mongo
    python3 scripts/gen_kg_qa.py --output data/ray/bio_qa/kg_pairs.jsonl

    # test fixture
    python3 scripts/gen_kg_qa.py --fixture tests/fixtures/fake_kg.json \\
        --output /tmp/out.jsonl --ray-entity-id ray_kurzweil_id
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


# ---- Question templates --------------------------------------------------

TEMPLATES: dict[str, list[str]] = {
    "created": [
        "What did Ray Kurzweil invent?",
        "What did Ray Kurzweil create?",
        "Tell me about something Ray Kurzweil built.",
        "What has Ray Kurzweil founded?",
        "Name a company or invention from Ray Kurzweil.",
        "What is one of your inventions, Ray?",
        "What did you create, Ray?",
        "Tell me about one of the companies you started.",
    ],
    "predicts": [
        "What does Ray Kurzweil predict?",
        "What is one of Ray Kurzweil's predictions?",
        "What has Ray Kurzweil forecast about the future?",
        "Tell me about a prediction Ray Kurzweil has made.",
        "What does Ray believe will happen in the future?",
        "What is one of your predictions, Ray?",
        "What do you forecast, Ray?",
    ],
    "influences": [
        "Whose work has Ray Kurzweil influenced?",
        "What has Ray Kurzweil influenced?",
        "Tell me about Ray Kurzweil's influence on technology.",
        "What field has Ray Kurzweil influenced?",
    ],
}

# Which category to stamp on each templated type.
CATEGORY_BY_TYPE: dict[str, str] = {
    "created": "biographical",
    "predicts": "conceptual",
    "influences": "biographical",
}

ANSWER_CHAR_CAP = 400


# ---- Helpers -------------------------------------------------------------

def _load_fixture(path: Path) -> dict[str, list[dict]]:
    return json.loads(path.read_text())


def _load_mongo(uri: str, db_name: str) -> dict[str, list[dict]]:
    from pymongo import MongoClient
    c = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = c[db_name]
    return {
        "entities": list(db.entities.find()),
        "relationships": list(db.relationships.find()),
        "chunks": list(db.chunks.find()),
    }


def _index_entities(entities: list[dict]) -> dict[str, dict]:
    """Index by entity_id for fast lookup."""
    idx = {}
    for e in entities:
        eid = e.get("entity_id")
        if eid:
            idx[eid] = e
    return idx


def _index_chunks(chunks: list[dict]) -> dict[str, dict]:
    idx = {}
    for ch in chunks:
        cid = ch.get("chunk_id")
        if cid:
            idx[cid] = ch
    return idx


RAY_NAME_PATTERNS = {"ray kurzweil", "raymond kurzweil", "ray"}


def _resolve_ray_ids(
    entities: list[dict],
    entities_by_id: dict[str, dict],
    explicit_id: str | None,
) -> set[str]:
    """Return set of entity_ids that represent Ray Kurzweil.

    Matches by entity_id (if provided) OR by name / aliases containing
    'Ray Kurzweil' / 'Raymond Kurzweil' (case-insensitive).
    """
    ids: set[str] = set()
    if explicit_id and explicit_id in entities_by_id:
        ids.add(explicit_id)

    for e in entities:
        name = (e.get("name") or "").strip().lower()
        aliases = [a.strip().lower() for a in (e.get("aliases") or [])]
        haystack = [name] + aliases
        if any(p in haystack for p in RAY_NAME_PATTERNS):
            # Require "Kurzweil" appears somewhere (defends against matching
            # e.g. the bare alias "Ray" belonging to some other entity).
            if "kurzweil" in name or any("kurzweil" in a for a in aliases):
                eid = e.get("entity_id")
                if eid:
                    ids.add(eid)
    return ids


def _build_answer(
    rel: dict,
    target_entity: dict | None,
    chunks_by_id: dict[str, dict],
) -> tuple[str, str] | None:
    """Return (answer_text, canonical_target_name) or None if the rel is
    unusable (no chunk text, or no meaningful target name).
    """
    # Prefer the relationship description as the canonical target — the live
    # KG often has noisy target entities (single-letter fragments) where the
    # description is the clean phrase (e.g. desc='Kurzweil Reading Machine'
    # with target.name='Reading').
    desc = (rel.get("description") or "").strip()
    tgt_name = (target_entity.get("name") if target_entity else "") or ""
    tgt_name = tgt_name.strip()

    candidates = [n for n in (desc, tgt_name) if n and len(n) >= 3]
    if not candidates:
        return None

    # Concat chunk text.
    chunk_texts = []
    for cid in rel.get("source_chunks", []) or []:
        ch = chunks_by_id.get(cid)
        if ch and ch.get("text"):
            chunk_texts.append(ch["text"].strip())
    raw = " ".join(chunk_texts).strip()
    if not raw:
        return None

    # Pick a canonical name that appears in the chunk text, if any.
    canonical = None
    for c in candidates:
        if c.lower() in raw.lower():
            canonical = c
            break
    if canonical is None:
        # None of the candidates appears — drop the rel, because the eval
        # would fail the verbatim-target assertion downstream anyway.
        return None

    answer = raw[:ANSWER_CHAR_CAP].rstrip()
    # Guarantee the canonical name appears in the emitted (possibly
    # truncated) answer. If truncation chopped it off, re-window around
    # the first occurrence.
    if canonical not in answer:
        idx = raw.lower().find(canonical.lower())
        start = max(0, idx - 100)
        end = min(len(raw), idx + len(canonical) + (ANSWER_CHAR_CAP - 100 - len(canonical)))
        answer = raw[start:end].strip()
        if canonical not in answer:
            # Last-ditch: just prepend the canonical.
            return None

    return answer, canonical


# ---- Main generator ------------------------------------------------------

def generate_pairs(
    relationships: list[dict],
    entities_by_id: dict[str, dict],
    chunks_by_id: dict[str, dict],
    ray_ids: set[str],
) -> tuple[list[dict], Counter, Counter, Counter]:
    """Returns (pairs, templated_counts, skipped_type_counts, reject_reasons)."""
    pairs: list[dict] = []
    templated = Counter()
    skipped_types = Counter()
    rejected = Counter()

    for rel in relationships:
        src = rel.get("source_entity")
        if src not in ray_ids:
            rejected["non_ray_subject"] += 1
            continue

        rtype = rel.get("type")
        templates = TEMPLATES.get(rtype)
        if not templates:
            skipped_types[rtype or "<none>"] += 1
            continue

        tgt = entities_by_id.get(rel.get("target_entity"))
        built = _build_answer(rel, tgt, chunks_by_id)
        if built is None:
            rejected[f"{rtype}:no_usable_answer"] += 1
            continue
        answer, canonical = built

        source_chunk_ids = list(rel.get("source_chunks") or [])
        category = CATEGORY_BY_TYPE.get(rtype, "biographical")

        for q in templates:
            pairs.append({
                "question": q,
                "answer": answer,
                "relationship_type": rtype,
                "source_chunk_ids": source_chunk_ids,
                "confidence": 1.0,
                "category": category,
            })
        templated[rtype] += 1

    return pairs, templated, skipped_types, rejected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo-uri", default="mongodb://172.23.208.1:27017")
    ap.add_argument("--db", default="digital_rai")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fixture", help="path to JSON fixture (bypasses Mongo)")
    ap.add_argument(
        "--ray-entity-id",
        default="e97852ef610267da",
        help="entity_id for Ray Kurzweil (default: live-KG id)",
    )
    ap.add_argument(
        "--sample-pairs",
        type=int,
        default=0,
        help="print this many random sampled pairs to stdout (qualitative check)",
    )
    args = ap.parse_args()

    if args.fixture:
        collections = _load_fixture(Path(args.fixture))
    else:
        collections = _load_mongo(args.mongo_uri, args.db)

    entities = collections["entities"]
    relationships = collections["relationships"]
    chunks = collections["chunks"]

    entities_by_id = _index_entities(entities)
    chunks_by_id = _index_chunks(chunks)
    ray_ids = _resolve_ray_ids(entities, entities_by_id, args.ray_entity_id)

    if not ray_ids:
        print("ERROR: could not resolve Ray Kurzweil entity in the KG.", file=sys.stderr)
        return 2

    pairs, templated, skipped_types, rejected = generate_pairs(
        relationships, entities_by_id, chunks_by_id, ray_ids,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- Report ----
    print(f"ray_entity_ids_resolved: {sorted(ray_ids)}")
    print(f"total_relationships_scanned: {len(relationships)}")
    print(f"pairs_emitted: {len(pairs)}")
    print(f"templated_relationships_by_type: {dict(templated.most_common())}")
    # Skipped unknown types — single aggregate count + top 15 breakdown.
    total_skipped = sum(skipped_types.values())
    print(f"skipped_unknown_types: {total_skipped}")
    for t, n in skipped_types.most_common(15):
        print(f"  skipped_type {t!r}: {n}")
    print(f"rejected_breakdown (non-Ray / missing chunk / etc.): "
          f"{dict(rejected.most_common())}")

    if args.sample_pairs and pairs:
        print(f"\n--- {args.sample_pairs} random sample pairs ---")
        rng = random.Random(42)
        for row in rng.sample(pairs, min(args.sample_pairs, len(pairs))):
            print(f"[{row['relationship_type']}] Q: {row['question']}")
            a = row["answer"]
            print(f"  A: {a[:200]}{'...' if len(a) > 200 else ''}")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
