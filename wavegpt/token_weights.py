"""
Token Weight Computation — focus loss on content words.

The first harmonic direction captures 75.9% of variance and encodes
"being an English word" (function words: the, and, in, for, with).
The residual directions (2–180) carry actual content.

Each token's weight = its residual norm after removing the fundamental.
High residual = content word → high loss weight.
Low residual = function word → low loss weight.

One multiply per token in the loss function, potentially 50% faster convergence
because the model stops wasting gradients on "the goes before nouns."
"""
from __future__ import annotations

import numpy as np


def compute_token_weights(
    token_harmonics: np.ndarray,
    min_weight: float = 0.3,
    max_weight: float = 3.0,
) -> np.ndarray:
    """
    Compute per-token loss weights from harmonic coordinates.

    Args:
        token_harmonics: (vocab_size, K) harmonic coordinates per token
        min_weight: minimum weight (for function words / absent tokens)
        max_weight: maximum weight (for rare content words)

    Returns:
        (vocab_size,) float32 array of loss weights, normalized to mean ≈ 1.0
    """
    V, K = token_harmonics.shape

    # The fundamental is dimension 0 (highest-variance SVD component)
    # Residual = everything except dimension 0
    residual = token_harmonics[:, 1:].copy()  # (V, K-1)

    # Residual norm per token
    norms = np.linalg.norm(residual, axis=1)  # (V,)

    # Tokens not in corpus (all zeros) → 0 norm → will get min_weight
    # Normalize: map to [0, 1] range based on percentiles (robust to outliers)
    nonzero_mask = norms > 0
    if nonzero_mask.sum() == 0:
        return np.full(V, 1.0, dtype=np.float32)

    nz_norms = norms[nonzero_mask]
    if len(nz_norms) > 10:
        lo = np.percentile(nz_norms, 10)
        hi = np.percentile(nz_norms, 90)
    else:
        lo = nz_norms.min()
        hi = nz_norms.max()

    # If all nonzero norms are identical, use [0, that_value] as range
    if hi - lo < 1e-8:
        hi = nz_norms.max()
        lo = 0.0

    if hi - lo < 1e-8:
        return np.full(V, 1.0, dtype=np.float32)

    # Linear map: lo → min_weight, hi → max_weight
    raw_weights = min_weight + (max_weight - min_weight) * (norms - lo) / (hi - lo)

    # Zero tokens get min_weight
    raw_weights[~nonzero_mask] = min_weight

    # Clamp
    raw_weights = np.clip(raw_weights, min_weight, max_weight)

    # Normalize to mean ≈ 1.0 so loss scale doesn't change
    mean_w = raw_weights.mean()
    if mean_w > 0:
        raw_weights = raw_weights / mean_w

    # Re-clamp after normalization
    raw_weights = np.clip(raw_weights, min_weight, max_weight)

    return raw_weights.astype(np.float32)
