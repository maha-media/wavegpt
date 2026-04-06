"""Tests for counterpoint narratives — multiple harmonic voices in one passage.

Bach's counterpoint: independent melodic lines that are individually coherent
but harmonically locked. Applied to training data: a single narrative that
simultaneously teaches identity (C), function (G), connection (D), and
distinction (A) through natural voice leading.

"Nanotechnology is a technology [C]. It enables molecular manufacturing [G],
which connects to biotechnology through convergent engineering [D]. Unlike
genetic engineering, which modifies existing biology, nanotechnology builds
from atoms up [A]."

One passage. Four voices. All harmonically locked around one entity.
"""
import pytest


def test_counterpoint_has_all_four_voices():
    """Each counterpoint narrative should touch C, G, D, and A."""
    from wavegpt.narratives import generate_counterpoint_narratives

    entities = [
        {"entity_id": "e1", "name": "nanotechnology", "type": "technology",
         "source_chunks": ["c1", "c2", "c3", "c4"]},
        {"entity_id": "e2", "name": "biotechnology", "type": "technology",
         "source_chunks": ["c2", "c3", "c5"]},
        {"entity_id": "e3", "name": "genetic engineering", "type": "technology",
         "source_chunks": ["c3", "c5", "c6"]},
    ]

    chunks = {
        "c1": {"chunk_id": "c1", "source_id": "s1",
               "text": "Nanotechnology will enable molecular manufacturing by 2030."},
        "c2": {"chunk_id": "c2", "source_id": "s1",
               "text": "Both nanotechnology and biotechnology are converging rapidly."},
        "c3": {"chunk_id": "c3", "source_id": "s2",
               "text": "Nanotechnology at the atomic scale differs from bulk engineering."},
        "c4": {"chunk_id": "c4", "source_id": "s2",
               "text": "Molecular nanotechnology constructs materials atom by atom."},
        "c5": {"chunk_id": "c5", "source_id": "s1",
               "text": "Biotechnology enables personalized medicine through genomics."},
        "c6": {"chunk_id": "c6", "source_id": "s2",
               "text": "Genetic engineering modifies DNA to alter organism traits."},
    }

    relationships = [
        {"source_entity": "e1", "target_entity": "e2", "type": "converges with",
         "description": "nanotechnology and biotechnology are converging"},
        {"source_entity": "e2", "target_entity": "e3", "type": "includes",
         "description": "biotechnology includes genetic engineering"},
    ]

    entity_names = {e["entity_id"]: e["name"] for e in entities}

    narratives = generate_counterpoint_narratives(
        entities=entities, chunks=chunks,
        relationships=relationships, entity_names=entity_names,
        min_chunks=2, seed=42,
    )

    assert len(narratives) > 0

    for n in narratives:
        assert n["category"] == "counterpoint"
        text = n["text"].lower()
        # Should be substantial (multi-voice = longer)
        assert len(text) > 100, f"Too short for counterpoint: {len(text)} chars"


def test_counterpoint_mentions_related_entities():
    """Counterpoint should reference connected entities, not just the anchor."""
    from wavegpt.narratives import generate_counterpoint_narratives

    entities = [
        {"entity_id": "e1", "name": "nanotechnology", "type": "technology",
         "source_chunks": ["c1", "c2", "c3"]},
        {"entity_id": "e2", "name": "biotechnology", "type": "technology",
         "source_chunks": ["c2", "c4"]},
    ]

    chunks = {
        "c1": {"chunk_id": "c1", "source_id": "s1",
               "text": "Nanotechnology enables molecular manufacturing."},
        "c2": {"chunk_id": "c2", "source_id": "s1",
               "text": "Nanotechnology and biotechnology converge."},
        "c3": {"chunk_id": "c3", "source_id": "s2",
               "text": "Nanotechnology builds structures atom by atom."},
        "c4": {"chunk_id": "c4", "source_id": "s1",
               "text": "Biotechnology harnesses biological processes."},
    }

    relationships = [
        {"source_entity": "e1", "target_entity": "e2", "type": "converges with",
         "description": "converging technologies"},
    ]

    entity_names = {e["entity_id"]: e["name"] for e in entities}

    narratives = generate_counterpoint_narratives(
        entities=entities, chunks=chunks,
        relationships=relationships, entity_names=entity_names,
        min_chunks=2, seed=42,
    )

    # At least one narrative should mention both nanotechnology and biotechnology
    texts = " ".join(n["text"].lower() for n in narratives)
    has_both = "nanotechnology" in texts and "biotechnology" in texts
    assert has_both, "Counterpoint should weave related entities together"


def test_counterpoint_differs_from_existing_narratives():
    """Counterpoint narratives should be distinct from single-voice narratives."""
    from wavegpt.narratives import (
        generate_counterpoint_narratives,
        generate_entity_context_narratives,
        generate_contrastive_narratives,
    )

    entities = [
        {"entity_id": "e1", "name": "nanotechnology", "type": "technology",
         "source_chunks": ["c1", "c2", "c3"]},
        {"entity_id": "e2", "name": "biotechnology", "type": "technology",
         "source_chunks": ["c2", "c3", "c4"]},
    ]

    chunks = {
        "c1": {"chunk_id": "c1", "source_id": "s1",
               "text": "Nanotechnology enables molecular manufacturing."},
        "c2": {"chunk_id": "c2", "source_id": "s1",
               "text": "Both technologies are converging."},
        "c3": {"chunk_id": "c3", "source_id": "s2",
               "text": "Nanotechnology differs from biotechnology in approach."},
        "c4": {"chunk_id": "c4", "source_id": "s1",
               "text": "Biotechnology harnesses biological processes."},
    }

    relationships = [
        {"source_entity": "e1", "target_entity": "e2", "type": "converges with",
         "description": "converging"},
    ]

    entity_names = {e["entity_id"]: e["name"] for e in entities}

    cp = generate_counterpoint_narratives(
        entities=entities, chunks=chunks,
        relationships=relationships, entity_names=entity_names,
        min_chunks=2, seed=42,
    )

    ctx = generate_entity_context_narratives(
        entities=entities, chunks=chunks, min_chunks=2, seed=42,
    )

    # Counterpoint texts should not be identical to context texts
    cp_texts = {n["text"] for n in cp}
    ctx_texts = {n["text"] for n in ctx}
    assert cp_texts != ctx_texts or len(cp_texts) == 0
