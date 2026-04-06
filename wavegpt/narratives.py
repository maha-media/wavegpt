"""
Rich Knowledge Graph Narratives — deterministic augmentation from MongoDB.

Generates training text by walking the knowledge graph. No LLM needed.
All narratives are derived from entity/relationship/chunk data already in the DB.

Current augmented data (24.5% of corpus):
  - "X is a concept." (entity narratives) — low information
  - "X enables Y." (relationship narratives) — low information

Rich narratives (target 40-50% of corpus):
  - Entity context: "Nanotechnology appears across multiple works. In [book],
    Kurzweil discusses molecular manufacturing. In [book], ..."
  - Relationship chains: "A enables B, which enables C, leading to D."
  - Cross-source synthesis: entity's treatment across different books
  - Type summaries: "Key technologies: nanotech, AI, genetic engineering..."
"""
from __future__ import annotations
import random
from collections import defaultdict
from itertools import combinations


def _excerpt(text: str, max_len: int = 200) -> str:
    """Extract a clean excerpt from chunk text."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    # Try to cut at a sentence boundary
    cut = text[:max_len].rfind(". ")
    if cut > max_len // 2:
        return text[:cut + 1]
    # Cut at word boundary
    cut = text[:max_len].rfind(" ")
    if cut > 0:
        return text[:cut] + "..."
    return text[:max_len] + "..."


def generate_entity_context_narratives(
    entities: list[dict],
    chunks: dict[str, dict],
    min_chunks: int = 3,
    max_excerpts: int = 4,
    seed: int = 42,
) -> list[dict]:
    """
    Generate context-rich narratives for entities with multiple chunk appearances.

    For each entity with enough chunks, pulls excerpts and weaves them
    into a paragraph showing how the entity is discussed.

    Returns list of {"text": str, "category": "entity_context"}
    """
    rng = random.Random(seed)
    narratives = []

    for ent in entities:
        chunk_ids = ent.get("source_chunks", [])
        if len(chunk_ids) < min_chunks:
            continue

        name = ent.get("name", "")
        etype = ent.get("type", "concept")
        if not name:
            continue

        # Collect available chunk texts
        available = []
        for cid in chunk_ids:
            chunk = chunks.get(cid)
            if chunk and chunk.get("text", "").strip():
                available.append(chunk["text"])

        if len(available) < min_chunks:
            continue

        # Sample excerpts
        sampled = rng.sample(available, min(max_excerpts, len(available)))
        excerpts = [_excerpt(t, 200) for t in sampled]

        # Build narrative
        parts = [f"{name} is a {etype} discussed extensively in the corpus."]
        for i, exc in enumerate(excerpts):
            if i == 0:
                parts.append(f"In one context: {exc}")
            elif i == 1:
                parts.append(f"Elsewhere: {exc}")
            else:
                parts.append(f"Additionally: {exc}")

        narratives.append({
            "text": " ".join(parts),
            "category": "entity_context",
        })

    return narratives


def generate_relationship_chain_narratives(
    relationships: list[dict],
    entity_names: dict[str, str],
    max_chain_length: int = 3,
    max_narratives: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """
    Walk the relationship graph 2-3 hops and generate chain narratives.

    "A enables B. B enables C. Thus A enables C through B."

    Returns list of {"text": str, "category": "relationship_chain"}
    """
    rng = random.Random(seed)

    # Build adjacency list
    graph = defaultdict(list)
    for rel in relationships:
        src = rel.get("source_entity", "")
        tgt = rel.get("target_entity", "")
        rtype = rel.get("type", "related to")
        if src and tgt and src in entity_names and tgt in entity_names:
            graph[src].append((rtype, tgt))

    # Find 2-hop and 3-hop chains
    chains = []
    nodes = list(graph.keys())
    rng.shuffle(nodes)

    for start in nodes:
        if len(chains) >= max_narratives * 3:  # oversample then truncate
            break
        for rtype1, mid in graph[start]:
            for rtype2, end in graph.get(mid, []):
                if end == start:
                    continue
                src_name = entity_names[start]
                mid_name = entity_names[mid]
                end_name = entity_names[end]

                text = (
                    f"{src_name} {rtype1} {mid_name}. "
                    f"{mid_name} {rtype2} {end_name}. "
                    f"This connects {src_name} to {end_name} through {mid_name}."
                )
                chains.append({
                    "text": text,
                    "category": "relationship_chain",
                })

    # Deduplicate and limit
    seen = set()
    unique = []
    for c in chains:
        key = c["text"]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    rng.shuffle(unique)
    return unique[:max_narratives]


def generate_cross_source_narratives(
    entities: list[dict],
    chunks: dict[str, dict],
    source_names: dict[str, str],
    min_sources: int = 2,
    max_excerpts_per_source: int = 2,
    seed: int = 42,
) -> list[dict]:
    """
    Generate narratives for entities appearing across multiple sources.

    "Nanotechnology is discussed in The Age of Intelligent Machines (1990)
    where Kurzweil notes [excerpt]. In The Singularity Is Near (2005), [excerpt]."

    Returns list of {"text": str, "category": "cross_source"}
    """
    rng = random.Random(seed)
    narratives = []

    for ent in entities:
        name = ent.get("name", "")
        etype = ent.get("type", "concept")
        chunk_ids = ent.get("source_chunks", [])
        if not name or len(chunk_ids) < 2:
            continue

        # Group chunks by source
        source_chunks = defaultdict(list)
        for cid in chunk_ids:
            chunk = chunks.get(cid)
            if chunk and chunk.get("text", "").strip():
                sid = chunk.get("source_id", "unknown")
                source_chunks[sid].append(chunk["text"])

        # Filter to sources we have names for
        named_sources = {
            sid: texts for sid, texts in source_chunks.items()
            if sid in source_names
        }

        if len(named_sources) < min_sources:
            continue

        # Build narrative
        parts = [f"{name} is a {etype} that appears across multiple works."]
        for sid, texts in list(named_sources.items())[:4]:
            sname = source_names[sid]
            sampled = rng.sample(texts, min(max_excerpts_per_source, len(texts)))
            for txt in sampled:
                exc = _excerpt(txt, 200)
                parts.append(f"In {sname}: {exc}")

        narratives.append({
            "text": " ".join(parts),
            "category": "cross_source",
        })

    return narratives


def generate_entity_type_summaries(
    entities: list[dict],
    min_entities_per_type: int = 5,
    max_per_summary: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    Generate type-grouped summaries.

    "Key technologies discussed include: nanotechnology, artificial intelligence,
    genetic engineering, robotics, and virtual reality."

    Returns list of {"text": str, "category": "type_summary"}
    """
    rng = random.Random(seed)
    type_groups = defaultdict(list)
    for ent in entities:
        etype = ent.get("type", "concept")
        name = ent.get("name", "")
        count = len(ent.get("source_chunks", []))
        if name:
            type_groups[etype].append((count, name))

    narratives = []
    for etype, members in type_groups.items():
        if len(members) < min_entities_per_type:
            continue

        # Sort by mention count, take top N
        members.sort(reverse=True)
        top = [name for _, name in members[:max_per_summary]]

        if len(top) > 2:
            listing = ", ".join(top[:-1]) + f", and {top[-1]}"
        elif len(top) == 2:
            listing = f"{top[0]} and {top[1]}"
        else:
            listing = top[0]

        text = f"Key {etype}s discussed in the corpus include: {listing}."
        narratives.append({
            "text": text,
            "category": "type_summary",
        })

    return narratives


def generate_contrastive_narratives(
    entities: list[dict],
    chunks: dict[str, dict],
    min_chunks: int = 3,
    max_narratives: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate contrastive pairs — near-miss statements that force discrimination.

    This is the anti-collapse mechanism in the data. By placing similar-but-different
    concepts side by side with explicit distinguishing language, the model is forced
    to learn the RESIDUAL signal (nuance) rather than collapsing to the fundamental.

    Three types:
      a) Same-type pairs: "Unlike nanotechnology, biotechnology focuses on..."
      b) Same entity, different context: shows nuanced usage
      c) Specificity ladders: generic → specific

    Returns list of {"text": str, "category": "contrastive"}
    """
    rng = random.Random(seed)
    narratives = []

    # Index entities by type
    by_type = defaultdict(list)
    for ent in entities:
        name = ent.get("name", "")
        if name and len(ent.get("source_chunks", [])) >= min_chunks:
            by_type[ent.get("type", "concept")].append(ent)

    contrastive_connectors = [
        "Unlike {a}, {b} takes a different approach.",
        "While {a} and {b} are both {type}s, they differ significantly.",
        "In contrast to {a}, {b} addresses a distinct area.",
        "{a} and {b} are related {type}s, but {a} is distinct from {b}.",
    ]

    # ── Type a: Same-type contrastive pairs ──
    for etype, ents in by_type.items():
        if len(ents) < 2:
            continue
        pairs = list(combinations(ents, 2))
        rng.shuffle(pairs)
        budget = max(1, max_narratives // (3 * max(1, len(by_type))))
        for ent_a, ent_b in pairs[:budget]:
            exc_a = _get_excerpt_for(ent_a, chunks, rng)
            exc_b = _get_excerpt_for(ent_b, chunks, rng)
            if not exc_a or not exc_b:
                continue
            conn = rng.choice(contrastive_connectors).format(
                a=ent_a["name"], b=ent_b["name"], type=etype
            )
            text = f"{conn} {ent_a['name']}: {exc_a} {ent_b['name']}: {exc_b}"
            narratives.append({"text": text, "category": "contrastive"})

    # ── Type b: Same entity, different chunk contexts ──
    for etype, ents in by_type.items():
        for ent in ents:
            cids = [c for c in ent.get("source_chunks", [])
                    if c in chunks and chunks[c].get("text", "").strip()]
            if len(cids) < 2:
                continue
            pair = rng.sample(cids, 2)
            exc1 = _excerpt(chunks[pair[0]]["text"], 200)
            exc2 = _excerpt(chunks[pair[1]]["text"], 200)
            name = ent["name"]
            text = (
                f"{name} appears in multiple contexts with distinct meanings. "
                f"In one context: {exc1} "
                f"Whereas elsewhere: {exc2} "
                f"These usages of {name} illustrate its nuanced role."
            )
            narratives.append({"text": text, "category": "contrastive"})

    # ── Type c: Specificity ladders ──
    for etype, ents in by_type.items():
        for ent in ents:
            exc = _get_excerpt_for(ent, chunks, rng)
            if not exc:
                continue
            name = ent["name"]
            text = (
                f"{name} is a {etype}. "
                f"More specifically, {exc} "
                f"This distinguishes {name} from other {etype}s."
            )
            narratives.append({"text": text, "category": "contrastive"})

    rng.shuffle(narratives)
    return narratives[:max_narratives]


def _get_excerpt_for(ent: dict, chunks: dict, rng: random.Random) -> str:
    """Get a random excerpt from an entity's chunks."""
    cids = [c for c in ent.get("source_chunks", [])
            if c in chunks and chunks[c].get("text", "").strip()]
    if not cids:
        return ""
    return _excerpt(chunks[rng.choice(cids)]["text"], 200)


def generate_counterpoint_narratives(
    entities: list[dict],
    chunks: dict[str, dict],
    relationships: list[dict],
    entity_names: dict[str, str],
    min_chunks: int = 3,
    max_narratives: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate counterpoint narratives — four harmonic voices in one passage.

    Like Bach's counterpoint: independent melodic lines (C, G, D, A) that
    are individually coherent but harmonically locked around an anchor entity.
    Every voice is about THE SAME entity. No random tangents.

    Structure of each narrative:
      Voice C (identity):    "X is a <type>." — what it IS
      Voice G (function):    from a chunk mentioning X — what it DOES
      Voice D (connection):  relationships to named entities — how it CONNECTS
      Voice A (distinction): how it differs from a RELATED entity — nuance

    Returns list of {"text": str, "category": "counterpoint"}
    """
    rng = random.Random(seed)
    narratives = []

    # Build relationship index: entity_id → [(type, target_id), ...]
    rel_from = defaultdict(list)
    for rel in relationships:
        src = rel.get("source_entity", "")
        tgt = rel.get("target_entity", "")
        rtype = rel.get("type", "related to")
        if src and tgt:
            rel_from[src].append((rtype, tgt))
            rel_from[tgt].append((rtype, src))

    # Build entity index
    ent_by_id = {}
    by_type = defaultdict(list)
    for ent in entities:
        eid = ent.get("entity_id", "")
        name = ent.get("name", "")
        if name and len(ent.get("source_chunks", [])) >= min_chunks:
            ent_by_id[eid] = ent
            by_type[ent.get("type", "concept")].append(ent)

    for anchor in ent_by_id.values():
        aid = anchor.get("entity_id", "")
        aname = anchor["name"]
        atype = anchor.get("type", "concept")

        # Get chunks that actually MENTION this entity by name
        relevant_chunks = _get_relevant_chunks(anchor, chunks, aname)
        if len(relevant_chunks) < 2:
            continue

        # ── Voice C: Identity (the fundamental) ──
        voice_c = f"{aname} is a {atype}."

        # ── Voice G: Function (what it does — from a chunk about it) ──
        exc_g = _excerpt(rng.choice(relevant_chunks), 200)
        voice_g = f"In the corpus: {exc_g}"

        # ── Voice D: Connection (how it relates to other entities) ──
        connections = rel_from.get(aid, [])
        voice_d = ""
        if connections:
            named = [(rt, tid) for rt, tid in connections
                     if len(entity_names.get(tid, "")) >= 3]  # filter noise names
            if named:
                sample = rng.sample(named, min(3, len(named)))
                chain_parts = []
                for rtype, tid in sample:
                    chain_parts.append(f"{rtype} {entity_names[tid]}")
                voice_d = f"{aname} " + ", and ".join(chain_parts) + "."

        # ── Voice A: Distinction (contrast with a RELATED entity, not random) ──
        voice_a = ""
        # First try: contrast with an entity this one is related to
        related_ids = {tid for _, tid in connections} if connections else set()
        contrast_ent = None
        for rid in related_ids:
            if rid in ent_by_id and ent_by_id[rid]["name"] != aname:
                contrast_ent = ent_by_id[rid]
                break
        # Fallback: same type
        if not contrast_ent:
            same = [e for e in by_type.get(atype, []) if e["name"] != aname]
            if same:
                contrast_ent = rng.choice(same)

        if contrast_ent:
            cname = contrast_ent["name"]
            c_relevant = _get_relevant_chunks(contrast_ent, chunks, cname)
            if c_relevant:
                c_exc = _excerpt(rng.choice(c_relevant), 150)
                # Use a second excerpt from anchor for the contrast
                a_excs = [c for c in relevant_chunks if c != exc_g]
                a_exc2 = _excerpt(rng.choice(a_excs), 150) if a_excs else exc_g
                voice_a = (
                    f"While {cname} is characterized by: {c_exc}, "
                    f"{aname} is distinguished by: {a_exc2}"
                )

        # ── Weave — all voices into one flowing passage ──
        parts = [voice_c, voice_g]
        if voice_d:
            parts.append(voice_d)
        if voice_a:
            parts.append(voice_a)

        if len(parts) >= 3:
            narratives.append({
                "text": " ".join(parts),
                "category": "counterpoint",
            })

    rng.shuffle(narratives)
    return narratives[:max_narratives]


def _get_relevant_chunks(
    ent: dict, chunks: dict, name: str,
) -> list[str]:
    """Get chunk texts that actually mention the entity name."""
    name_lower = name.lower()
    results = []
    for cid in ent.get("source_chunks", []):
        chunk = chunks.get(cid)
        if chunk:
            text = chunk.get("text", "").strip()
            if text and name_lower in text.lower():
                results.append(text)
    # Fallback: if no chunk mentions the name, use any available
    if not results:
        for cid in ent.get("source_chunks", []):
            chunk = chunks.get(cid)
            if chunk and chunk.get("text", "").strip():
                results.append(chunk["text"].strip())
    return results
