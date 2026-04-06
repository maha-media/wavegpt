"""Tests for harmonic curriculum — circle of fifths data scheduling.

The harmonic curriculum walks C→G→D→A through 4 degrees of separation:
  C (fundamental): What things ARE — types, categories, abstracts
  G (1st fifth):   What things DO — entity context, properties
  D (2nd fifth):   How things CONNECT — chains, cross-source, relationships
  A (3rd fifth):   How things DIFFER — contrastive pairs, nuance

Each phase MOVES the tonic — the new layer is emphasized, previous layers
become maintenance. Like modulating keys in music.
"""
import numpy as np
import pytest


def test_harmonic_layers_assignment():
    """Each narrative category maps to exactly one harmonic layer."""
    from wavegpt.dataloader import HARMONIC_LAYERS

    # Every known category should map to a layer
    known_categories = [
        "abstract", "entity", "relationship", "temporal",
        "type_summary",          # C layer
        "entity_context",        # G layer
        "chain", "cross_source", # D layer
        "contrastive",           # A layer
    ]
    for cat in known_categories:
        assert cat in HARMONIC_LAYERS, f"Category '{cat}' not in HARMONIC_LAYERS"

    # Layers should be C, G, D, A
    layers = set(HARMONIC_LAYERS.values())
    assert layers == {"C", "G", "D", "A"}


def test_harmonic_phase_weights():
    """Phase weights shift emphasis forward through the harmonic chain."""
    from wavegpt.dataloader import get_harmonic_phase_weights

    # Phase 1 (early): C is tonic (highest weight)
    w1 = get_harmonic_phase_weights(0.0)
    assert w1["C"] > w1.get("G", 0), "C should be tonic in Phase 1"
    assert w1["C"] > w1.get("D", 0)

    # Phase 2: G becomes tonic, C drops
    w2 = get_harmonic_phase_weights(0.20)
    assert w2["G"] > w2["C"], f"G={w2['G']} should dominate over C={w2['C']}"

    # Phase 3: D becomes tonic
    w3 = get_harmonic_phase_weights(0.45)
    assert w3["D"] > w3["C"]
    assert w3["D"] > w3["G"]

    # Phase 4: A + raw enter, all layers present
    w4 = get_harmonic_phase_weights(0.80)
    assert w4.get("A", 0) > 0
    assert w4.get("raw", 0) > 0
    assert all(w4.get(k, 0) > 0 for k in ["C", "G", "D", "A", "raw"])

    # All weights sum to 1 at each phase
    for progress in [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0]:
        w = get_harmonic_phase_weights(progress)
        total = sum(w.values())
        assert abs(total - 1.0) < 0.01, f"Weights at {progress} sum to {total}"


def test_harmonic_curriculum_loader_init():
    """HarmonicCurriculumLoader loads layer files."""
    from wavegpt.dataloader import HarmonicCurriculumLoader
    from wavegpt.data_io import write_datafile
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake layer files
        for layer in ["C", "G", "D", "A"]:
            tokens = list(range(100 + ord(layer)))  # different sizes
            write_datafile(os.path.join(tmpdir, f"rai_layer_{layer}.bin"), tokens)

        # Raw file
        raw_tokens = list(range(500))
        write_datafile(os.path.join(tmpdir, "rai_train_000.bin"), raw_tokens)

        loader = HarmonicCurriculumLoader(
            data_dir=tmpdir, batch_size=2, block_size=16, device="cpu",
        )

        assert "C" in loader.layers
        assert "G" in loader.layers
        assert "raw" in loader.layers

        # Should be able to get batches
        x, y = loader.get_batch(step=0, total_steps=100)
        assert x.shape == (2, 16)
        assert y.shape == (2, 16)


def test_harmonic_curriculum_phase_transitions():
    """Loader serves different data at different phases."""
    from wavegpt.dataloader import HarmonicCurriculumLoader
    from wavegpt.data_io import write_datafile
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create layers with distinctive token values
        # C layer: all 1s, G: all 2s, D: all 3s, A: all 4s
        for layer, val in [("C", 1), ("G", 2), ("D", 3), ("A", 4)]:
            tokens = [val] * 500
            write_datafile(os.path.join(tmpdir, f"rai_layer_{layer}.bin"), tokens)

        raw_tokens = [5] * 500
        write_datafile(os.path.join(tmpdir, "rai_train_000.bin"), raw_tokens)

        loader = HarmonicCurriculumLoader(
            data_dir=tmpdir, batch_size=4, block_size=16, device="cpu",
        )

        # Phase 1 (step 0): should be mostly C (value=1)
        vals = set()
        for _ in range(20):
            x, _ = loader.get_batch(step=0, total_steps=1000)
            vals.add(x[0, 0].item())
        assert 1 in vals, f"Phase 1 should include C (val=1), got {vals}"

        # Phase 4 (step 900): should include raw (value=5)
        vals = set()
        for _ in range(50):
            x, _ = loader.get_batch(step=900, total_steps=1000)
            vals.add(x[0, 0].item())
        assert 5 in vals, f"Phase 4 should include raw (val=5), got {vals}"
