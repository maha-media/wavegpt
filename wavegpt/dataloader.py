"""
Data loader for llm.c .bin format — feeds WaveGPT training.

Infinite iterator yielding (x, y) tensor batches where y = x shifted right by 1.
Reads the same binary format as Karpathy's llm.c (header + uint16 token stream).
"""
from __future__ import annotations

import os
import random as _random
from pathlib import Path

import numpy as np
import torch

from .data_io import read_datafile

# ── Harmonic layer assignments ──
# Maps narrative categories to their harmonic layer (circle of fifths):
#   C (fundamental): what things ARE — types, categories, abstracts
#   G (1st fifth):   what things DO — entity context, properties
#   D (2nd fifth):   how things CONNECT — chains, relationships, cross-source
#   A (3rd fifth):   how things DIFFER — contrastive pairs
HARMONIC_LAYERS = {
    # C layer — fundamental (identity)
    "abstract": "C",
    "type_summary": "C",
    "entity": "C",
    # G layer — first fifth (function)
    "entity_context": "G",
    "temporal": "G",
    # D layer — second fifth (connection)
    "relationship": "D",
    "chain": "D",
    "cross_source": "D",
    # A layer — third fifth (nuance)
    "contrastive": "A",
}


class WaveDataLoader:
    """
    Infinite-iteration data loader for llm.c .bin format.

    Yields (x, y) tensors where y[i] = x[i] shifted right by 1 token.
    Wraps around when data is exhausted.
    """

    def __init__(
        self,
        bin_path: str,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
    ):
        self.tokens = read_datafile(bin_path)
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.pos = 0
        self.n_tokens = len(self.tokens)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.block_size
        buf = np.zeros((B, T + 1), dtype=np.int64)

        for i in range(B):
            # Wrap around if we've exhausted the data
            start = self.pos % max(self.n_tokens - T - 1, 1)
            end = start + T + 1
            if end <= self.n_tokens:
                buf[i] = self.tokens[start:end].astype(np.int64)
            else:
                # Wrap: take what's left + start from beginning
                remaining = self.n_tokens - start
                buf[i, :remaining] = self.tokens[start:].astype(np.int64)
                buf[i, remaining:] = self.tokens[: T + 1 - remaining].astype(np.int64)
            self.pos += T

        x = torch.tensor(buf[:, :T], dtype=torch.long, device=self.device)
        y = torch.tensor(buf[:, 1 : T + 1], dtype=torch.long, device=self.device)
        return x, y

    def __len__(self) -> int:
        """Approximate number of batches per epoch."""
        return max(1, self.n_tokens // (self.batch_size * self.block_size))

    def reset(self):
        self.pos = 0


# ── Circle of Fifths Curriculum ──

def get_harmonic_phase_weights(progress: float) -> dict[str, float]:
    """
    Return sampling weights for each layer + raw at a given training progress.

    Walks C→G→D→A like the circle of fifths. Each phase MOVES the tonic
    while keeping all voices present. Like modulating keys — G becomes the
    new tonic, but C is still in the chord.

    Phase 1 (0.00-0.12): C is tonic — learn what things ARE
    Phase 2 (0.12-0.30): G is tonic — learn what things DO
    Phase 3 (0.30-0.55): D is tonic — learn how things CONNECT
    Phase 4 (0.55-1.00): A + raw    — learn nuance + natural text
    """
    if progress < 0.12:
        # Phase 1: C is tonic, others audible
        return {"C": 0.55, "G": 0.20, "D": 0.15, "A": 0.10}
    elif progress < 0.30:
        # Phase 2: Modulate to G. C drops to accompaniment.
        return {"C": 0.12, "G": 0.48, "D": 0.20, "A": 0.12, "raw": 0.08}
    elif progress < 0.55:
        # Phase 3: Modulate to D. C+G become background.
        return {"C": 0.08, "G": 0.12, "D": 0.45, "A": 0.15, "raw": 0.20}
    else:
        # Phase 4: Full spectrum. A + raw dominate. The nuance movement.
        return {"C": 0.05, "G": 0.07, "D": 0.13, "A": 0.30, "raw": 0.45}


class HarmonicCurriculumLoader:
    """
    Data loader that walks the circle of fifths through 4 harmonic layers.

    Expects files in data_dir:
      rai_layer_C.bin, rai_layer_G.bin, rai_layer_D.bin, rai_layer_A.bin
      rai_train_000.bin (raw + full for phase 4)

    At each step, probabilistically selects a layer based on the current
    phase weights, then samples a batch from that layer.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.rng = _random.Random(seed)

        self.layers: dict[str, np.ndarray] = {}

        # Load harmonic layer files
        for layer in ["C", "G", "D", "A"]:
            path = os.path.join(data_dir, f"rai_layer_{layer}.bin")
            if os.path.exists(path):
                self.layers[layer] = read_datafile(path)

        # Load raw/full for phase 4
        raw_path = os.path.join(data_dir, "rai_train_000.bin")
        if os.path.exists(raw_path):
            self.layers["raw"] = read_datafile(raw_path)

        # Track position per layer for sequential reading
        self.positions: dict[str, int] = {k: 0 for k in self.layers}

        # Total tokens across all layers
        self.n_tokens = sum(len(v) for v in self.layers.values())

    def get_batch(
        self, step: int, total_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch from the appropriate harmonic layer for this step."""
        progress = step / max(total_steps, 1)
        weights = get_harmonic_phase_weights(progress)

        # Probabilistically choose a layer
        layer = self._choose_layer(weights)
        data = self.layers[layer]

        return self._sample_from(data, layer)

    def _choose_layer(self, weights: dict[str, float]) -> str:
        """Weighted random selection of a layer."""
        # Filter to available layers
        available = {k: v for k, v in weights.items() if k in self.layers}
        if not available:
            # Fallback to any available layer
            return next(iter(self.layers))

        # Normalize weights
        total = sum(available.values())
        r = self.rng.random() * total
        cumulative = 0.0
        for layer, w in available.items():
            cumulative += w
            if r <= cumulative:
                return layer
        return list(available.keys())[-1]

    def _sample_from(
        self, data: np.ndarray, layer: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from a token array."""
        B, T = self.batch_size, self.block_size
        buf = np.zeros((B, T + 1), dtype=np.int64)
        n = len(data)

        for i in range(B):
            pos = self.positions.get(layer, 0) % max(n - T - 1, 1)
            end = pos + T + 1
            if end <= n:
                buf[i] = data[pos:end].astype(np.int64)
            else:
                remaining = n - pos
                buf[i, :remaining] = data[pos:].astype(np.int64)
                buf[i, remaining:] = data[: T + 1 - remaining].astype(np.int64)
            self.positions[layer] = pos + T

        x = torch.tensor(buf[:, :T], dtype=torch.long, device=self.device)
        y = torch.tensor(buf[:, 1 : T + 1], dtype=torch.long, device=self.device)
        return x, y
