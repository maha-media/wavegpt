"""
Harmonic Token Initialization — map each token to its position in
the knowledge space's harmonic coordinate system.

For each token t:
  1. Find all chunks containing t
  2. Average their 1024-dim embeddings
  3. Project into 384-dim harmonic space: coords = avg_emb @ H^T

The 384 dimensions ARE the harmonic modes discovered by SVD.
384 = 3 × 2^7 — the perfect fifth enters the octave chain.
Tokens not in the corpus get zero vectors (filled with random init later).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import tiktoken


def compute_wave_lens(
    token_harmonics: np.ndarray,
    n_embd: int | None = None,
) -> np.ndarray:
    """
    Compute the wave lens — an orthogonal matrix that orders dimensions
    by harmonic variance for attention initialization.

    The wave lens is the eigenvector matrix of the token-harmonic covariance.
    When used to initialize Q/K projections, attention naturally decomposes
    hidden states into harmonic modes:
      - Head 1 dims → highest-variance directions (fundamentals)
      - Last head dims → lowest-variance directions (overtones)

    This is the WAVE approach: harmonics enter via dynamic projection
    (attention), not static position (embedding).

    Args:
        token_harmonics: (vocab_size, K) harmonic coordinates per token
        n_embd: model dimension. Truncates harmonics if K > n_embd.

    Returns:
        (n_embd, n_embd) orthogonal matrix — the wave lens
    """
    T = token_harmonics.copy()
    if n_embd is not None and T.shape[1] > n_embd:
        T = T[:, :n_embd]
    elif n_embd is not None and T.shape[1] < n_embd:
        # Pad with zeros if needed
        pad = np.zeros((T.shape[0], n_embd - T.shape[1]), dtype=T.dtype)
        T = np.concatenate([T, pad], axis=1)

    d = T.shape[1]

    # Use only tokens with nonzero harmonic coordinates
    mask = (T != 0).any(axis=1)
    T_active = T[mask]

    if len(T_active) < d:
        # Not enough data — return identity
        return np.eye(d, dtype=np.float32)

    # Covariance of harmonic coordinates across vocabulary
    T_centered = T_active - T_active.mean(axis=0)
    C = T_centered.T @ T_centered / len(T_centered)  # (d, d)

    # Eigendecomposition → principal harmonic directions
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort by descending eigenvalue (highest variance first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]  # columns = eigenvectors, sorted

    # The lens is eigenvectors.T — it rotates from harmonic space
    # to variance-ordered space
    lens = eigenvectors.T.astype(np.float32)  # (d, d)

    return lens


def compute_token_harmonics(
    harmonics: np.ndarray,
    chunk_tokens: dict[str, list[int]],
    chunk_embeddings: dict[str, np.ndarray],
    vocab_size: int = 50257,
) -> np.ndarray:
    """
    Compute per-token harmonic coordinates.

    Args:
        harmonics: (K, dim) harmonic basis from SVD
        chunk_tokens: {chunk_id: [token_ids]} for each chunk
        chunk_embeddings: {chunk_id: (dim,) embedding} for each chunk
        vocab_size: vocabulary size

    Returns:
        (vocab_size, K) array of harmonic coordinates per token
    """
    K = harmonics.shape[0]

    # Accumulate projections per token
    token_sum = np.zeros((vocab_size, K), dtype=np.float64)
    token_count = np.zeros(vocab_size, dtype=np.int64)

    for chunk_id, token_ids in chunk_tokens.items():
        emb = chunk_embeddings.get(chunk_id)
        if emb is None:
            continue
        # Project chunk embedding into harmonic space
        coords = (emb @ harmonics.T).astype(np.float64)  # (K,)
        # Each unique token in this chunk gets one vote
        unique_tokens = set(token_ids)
        for t in unique_tokens:
            if 0 <= t < vocab_size:
                token_sum[t] += coords
                token_count[t] += 1

    # Average
    mask = token_count > 0
    token_sum[mask] /= token_count[mask, None]

    return token_sum.astype(np.float32)


def build_token_harmonics_from_db(
    db,
    harmonics_path: str = "data/wave/wave_harmonics.npy",
    output_path: str = "data/wave/token_harmonics.npy",
) -> tuple[np.ndarray, dict]:
    """
    Full pipeline: load harmonics + DB chunks → token harmonic matrix.

    Returns:
        (token_harmonics, stats) where stats has coverage info
    """
    enc = tiktoken.get_encoding("gpt2")
    harmonics = np.load(harmonics_path)

    chunk_tokens = {}
    chunk_embeddings = {}
    total_tokens = 0

    for chunk in db.chunks.find(
        {"embedding": {"$exists": True}},
        {"chunk_id": 1, "text": 1, "context_prefix": 1, "embedding": 1},
    ):
        cid = chunk["chunk_id"]
        text = chunk.get("text", "")
        ctx = chunk.get("context_prefix", "")
        if ctx and ctx != "(untitled)":
            text = f"{ctx}\n{text}"
        tokens = enc.encode_ordinary(text)
        chunk_tokens[cid] = tokens
        chunk_embeddings[cid] = np.array(chunk["embedding"], dtype=np.float32)
        total_tokens += len(tokens)

    result = compute_token_harmonics(
        harmonics=harmonics,
        chunk_tokens=chunk_tokens,
        chunk_embeddings=chunk_embeddings,
        vocab_size=enc.n_vocab,
    )

    covered = int((result != 0).any(axis=1).sum())
    stats = {
        "vocab_size": enc.n_vocab,
        "covered_tokens": covered,
        "coverage_pct": round(100 * covered / enc.n_vocab, 1),
        "total_corpus_tokens": total_tokens,
        "n_chunks": len(chunk_tokens),
        "n_harmonics": harmonics.shape[0],
    }

    np.save(output_path, result)
    return result, stats
