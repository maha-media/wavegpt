"""
Spectral ordering via circle of fifths walk through embedding space.

The ordering isn't a semantic taxonomy — it's pure math:
- Embed all conversations into a shared space
- SVD to find the principal spectral directions (the harmonics)
- Compute phase angle in the (u₁, u₂) plane
- Walk the circle of fifths: stride by 7/12 of the circle per step

C is any starting point. G is the perfect fifth from C.
D is where the fifth of G lands. The circle generates itself
from one operation: multiply by 3/2, fold into the octave.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds


# Circle of fifths ordering for 12 segments
# Each step is 7 semitones (= perfect fifth) around the circle
# Segment indices: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
FIFTHS_NOTES = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']


def compute_spectral_order(
    token_sequences: list[list[int]],
    vocab_size: int = 50257,
    n_harmonics: int = 12,
) -> np.ndarray:
    """
    Compute circle-of-fifths ordering for a list of token sequences.

    Steps:
    1. Build sparse bag-of-tokens matrix (TF-IDF weighted)
    2. Truncated SVD → project into spectral space
    3. Phase angle in (PC1, PC2) plane → circular position
    4. Divide circle into 12 segments, interleave by fifths

    Args:
        token_sequences: List of token ID lists, one per conversation
        vocab_size: Tokenizer vocabulary size
        n_harmonics: Number of SVD components (default 12 = chromatic)

    Returns:
        order: np.ndarray of indices into token_sequences in fifths-walk order
    """
    N = len(token_sequences)
    if N < n_harmonics + 1:
        return np.arange(N)

    # ── Build sparse TF matrix ──
    print(f"  Building token matrix ({N:,} × {vocab_size:,})...")
    X = lil_matrix((N, vocab_size), dtype=np.float32)
    for i, tokens in enumerate(token_sequences):
        for t in tokens:
            if t < vocab_size:
                X[i, t] += 1.0

    X = X.tocsr()

    # ── TF-IDF weighting ──
    # IDF = log(N / df) where df = number of docs containing term
    print(f"  Computing TF-IDF...")
    df = np.array((X > 0).sum(axis=0)).flatten().astype(np.float64)
    df = np.maximum(df, 1)  # avoid div by zero
    idf = np.log(N / df)

    # Apply IDF (multiply each column by its IDF weight)
    from scipy.sparse import diags
    X = X @ diags(idf)

    # L2 normalize rows
    row_norms = np.sqrt(np.array(X.multiply(X).sum(axis=1)).flatten())
    row_norms = np.maximum(row_norms, 1e-10)
    X = X.multiply(1.0 / row_norms[:, np.newaxis])

    # ── Truncated SVD ──
    k = min(n_harmonics, N - 1, vocab_size - 1)
    print(f"  SVD (k={k})...")
    U, S, Vt = svds(X, k=k)

    # svds returns in ascending order — flip to descending
    idx = np.argsort(-S)
    U = U[:, idx]
    S = S[idx]

    # Report variance explained
    total_var = np.sum(np.array(X.multiply(X).sum()))
    explained = np.sum(S ** 2)
    print(f"  Top {k} components explain {100*explained/max(total_var,1e-10):.1f}% variance")
    print(f"  Singular values: {', '.join(f'{s:.1f}' for s in S[:6])}...")

    # ── Phase angle in (PC1, PC2) plane ──
    theta = np.arctan2(U[:, 1], U[:, 0])  # range [-π, π]

    # ── Divide into 12 segments and walk by fifths ──
    # Normalize to [0, 2π)
    theta_norm = (theta + np.pi) % (2 * np.pi)  # [0, 2π)

    # Assign each conversation to one of 12 segments
    segment_size = 2 * np.pi / 12
    segments = (theta_norm / segment_size).astype(int)
    segments = np.clip(segments, 0, 11)

    # Build segment lists, sort within each by phase
    segment_lists = [[] for _ in range(12)]
    for i in range(N):
        segment_lists[segments[i]].append((theta_norm[i], i))

    # Sort within each segment
    for seg in segment_lists:
        seg.sort(key=lambda x: x[0])

    # Report segment sizes
    sizes = [len(s) for s in segment_lists]
    print(f"  Segment sizes: {sizes}")
    print(f"  Walk order: {' → '.join(FIFTHS_NOTES)}")

    # ── Interleave by circle of fifths ──
    order = []
    for fifths_idx in FIFTHS_ORDER:
        for _, conv_idx in segment_lists[fifths_idx]:
            order.append(conv_idx)

    return np.array(order, dtype=np.int64)


def compute_spectral_order_simple(
    token_sequences: list[list[int]],
    vocab_size: int = 50257,
) -> np.ndarray:
    """
    Simplified spectral ordering: just sort by projection onto first
    singular vector. Linear walk through the fundamental.

    Use this as a baseline to compare against the full fifths walk.
    """
    N = len(token_sequences)
    if N < 3:
        return np.arange(N)

    X = lil_matrix((N, vocab_size), dtype=np.float32)
    for i, tokens in enumerate(token_sequences):
        for t in tokens:
            if t < vocab_size:
                X[i, t] += 1.0
    X = X.tocsr()

    U, S, Vt = svds(X, k=1)
    order = np.argsort(U[:, 0])
    return order
