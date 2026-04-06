"""Tests for harmonic token initialization."""
import numpy as np
import pytest

from wavegpt.harmonics import compute_token_harmonics


def test_token_harmonics_shape():
    """Token harmonics should be (vocab_size, n_harmonics)."""
    harmonics = np.random.randn(8, 16).astype(np.float32)
    chunk_tokens = {
        "c1": [0, 1, 2, 3],
        "c2": [2, 3, 4, 5],
    }
    chunk_embeddings = {
        "c1": np.random.randn(16).astype(np.float32),
        "c2": np.random.randn(16).astype(np.float32),
    }
    result = compute_token_harmonics(
        harmonics=harmonics,
        chunk_tokens=chunk_tokens,
        chunk_embeddings=chunk_embeddings,
        vocab_size=10,
    )
    assert result.shape == (10, 8)
    assert result.dtype == np.float32


def test_token_harmonics_shared_tokens():
    """Tokens appearing in multiple chunks get averaged harmonic coords."""
    harmonics = np.eye(4, dtype=np.float32)  # identity basis
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    chunk_tokens = {"c1": [0, 1], "c2": [1, 2]}
    chunk_embeddings = {"c1": emb1, "c2": emb2}
    result = compute_token_harmonics(
        harmonics=harmonics,
        chunk_tokens=chunk_tokens,
        chunk_embeddings=chunk_embeddings,
        vocab_size=5,
    )
    # Token 1 appears in both chunks — average of projections
    expected = (emb1 + emb2) / 2
    np.testing.assert_allclose(result[1], expected, atol=1e-6)


def test_unseen_tokens_are_zero():
    """Tokens not in corpus get zero vectors."""
    harmonics = np.eye(4, dtype=np.float32)
    chunk_tokens = {"c1": [0]}
    chunk_embeddings = {"c1": np.ones(4, dtype=np.float32)}
    result = compute_token_harmonics(
        harmonics=harmonics,
        chunk_tokens=chunk_tokens,
        chunk_embeddings=chunk_embeddings,
        vocab_size=10,
    )
    np.testing.assert_array_equal(result[5], np.zeros(4))


def test_repeated_token_in_chunk_counted_once():
    """A token appearing multiple times in one chunk doesn't over-count."""
    harmonics = np.eye(4, dtype=np.float32)
    emb1 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 4.0, 0.0, 0.0], dtype=np.float32)
    # Token 0 appears 3x in c1 but should only count c1 once
    chunk_tokens = {"c1": [0, 0, 0], "c2": [0, 1]}
    chunk_embeddings = {"c1": emb1, "c2": emb2}
    result = compute_token_harmonics(
        harmonics=harmonics,
        chunk_tokens=chunk_tokens,
        chunk_embeddings=chunk_embeddings,
        vocab_size=5,
    )
    # Token 0: average of c1 projection and c2 projection
    expected = (emb1 + emb2) / 2
    np.testing.assert_allclose(result[0], expected, atol=1e-6)
