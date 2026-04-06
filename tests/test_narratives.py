"""Tests for rich knowledge graph narrative generation."""
import pytest

from wavegpt.narratives import (
    generate_entity_context_narratives,
    generate_relationship_chain_narratives,
    generate_cross_source_narratives,
    generate_entity_type_summaries,
)


# ── Fixtures ──

ENTITIES = [
    {
        "entity_id": "e1",
        "name": "nanotechnology",
        "type": "technology",
        "source_chunks": ["c1", "c2", "c3"],
    },
    {
        "entity_id": "e2",
        "name": "Ray Kurzweil",
        "type": "person",
        "source_chunks": ["c1", "c4"],
    },
    {
        "entity_id": "e3",
        "name": "consciousness",
        "type": "concept",
        "source_chunks": ["c2", "c5"],
    },
    {
        "entity_id": "e4",
        "name": "IBM",
        "type": "organization",
        "source_chunks": ["c3"],
    },
]

CHUNKS = {
    "c1": {"chunk_id": "c1", "source_id": "s1", "text": "Nanotechnology will enable molecular manufacturing. Ray Kurzweil predicts this by 2030."},
    "c2": {"chunk_id": "c2", "source_id": "s2", "text": "Nanotechnology at the molecular scale could revolutionize medicine and consciousness research."},
    "c3": {"chunk_id": "c3", "source_id": "s1", "text": "IBM developed early nanotechnology research programs in the 1990s."},
    "c4": {"chunk_id": "c4", "source_id": "s2", "text": "Ray Kurzweil argues that the singularity is near."},
    "c5": {"chunk_id": "c5", "source_id": "s2", "text": "Consciousness remains the hard problem of philosophy of mind."},
}

SOURCE_NAMES = {
    "s1": "The Age of Intelligent Machines (1990)",
    "s2": "The Singularity Is Near (2005)",
}

RELATIONSHIPS = [
    {"source_entity": "e1", "target_entity": "e3", "type": "enables", "description": ""},
    {"source_entity": "e2", "target_entity": "e1", "type": "predicts", "description": ""},
    {"source_entity": "e1", "target_entity": "e4", "type": "part_of", "description": ""},
]


# ── Tests ──

def test_entity_context_generates_text():
    """Entity context narratives include entity name and chunk excerpts."""
    narratives = generate_entity_context_narratives(
        entities=ENTITIES,
        chunks=CHUNKS,
        min_chunks=2,
    )
    assert len(narratives) > 0
    # Should include entities with 2+ chunks
    texts = [n["text"] for n in narratives]
    combined = " ".join(texts)
    assert "nanotechnology" in combined.lower()
    assert "Ray Kurzweil" in combined


def test_entity_context_min_chunks_filter():
    """Entities below min_chunks threshold are excluded."""
    narratives = generate_entity_context_narratives(
        entities=ENTITIES,
        chunks=CHUNKS,
        min_chunks=5,  # No entity has 5+ chunks in test data
    )
    assert len(narratives) == 0


def test_relationship_chains():
    """Chain narratives walk 2+ hops through the graph."""
    ENTITY_MAP = {e["entity_id"]: e["name"] for e in ENTITIES}
    narratives = generate_relationship_chain_narratives(
        relationships=RELATIONSHIPS,
        entity_names=ENTITY_MAP,
        max_chain_length=3,
        max_narratives=100,
    )
    # Should find at least one chain: Kurzweil -predicts-> nanotech -enables-> consciousness
    assert len(narratives) > 0
    texts = [n["text"] for n in narratives]
    combined = " ".join(texts)
    # Chain should mention at least 3 entities
    assert "nanotechnology" in combined.lower() or "Kurzweil" in combined


def test_cross_source_narratives():
    """Cross-source narratives mention entities appearing in multiple sources."""
    narratives = generate_cross_source_narratives(
        entities=ENTITIES,
        chunks=CHUNKS,
        source_names=SOURCE_NAMES,
    )
    assert len(narratives) > 0
    # Nanotechnology appears in s1 and s2
    texts = [n["text"] for n in narratives]
    combined = " ".join(texts)
    assert "nanotechnology" in combined.lower()
    # Should mention source names
    assert "Intelligent Machines" in combined or "Singularity" in combined


def test_entity_type_summaries():
    """Type summaries list entities grouped by type."""
    narratives = generate_entity_type_summaries(
        entities=ENTITIES,
        min_entities_per_type=1,
    )
    assert len(narratives) > 0
    texts = [n["text"] for n in narratives]
    combined = " ".join(texts)
    assert "technology" in combined.lower()


def test_narratives_have_category():
    """All narratives include a category field for data mixing."""
    narratives = generate_entity_context_narratives(
        entities=ENTITIES, chunks=CHUNKS, min_chunks=2,
    )
    for n in narratives:
        assert "category" in n
        assert "text" in n
        assert len(n["text"]) > 10
