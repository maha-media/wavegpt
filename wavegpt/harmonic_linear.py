"""
HarmonicLinear — a linear layer parameterized by a power-law spectrum.

Instead of storing a full (out_dim × in_dim) weight matrix, stores:
  - σ₁: scalar scale (1 param)
  - α:  spectral decay exponent (1 param)
  - U:  output basis vectors (out_dim × rank)
  - V:  input basis vectors (in_dim × rank)

The weight matrix is reconstructed as:
  W = Σ_k  σ₁ · k^{-α} · U[:,k] · V[:,k]^T

This is what gradient descent converges to anyway (harmonic autopsy proof).
We just skip the 15,000 steps of brute-force discovery.

Parameter count: (in_dim + out_dim) × rank + 2
vs full linear:  in_dim × out_dim

For rank=30, in=384, out=1536:
  HarmonicLinear:  57,602 params
  nn.Linear:      589,824 params
  Compression:    10.2x
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import numpy as np


class HarmonicLinear(nn.Module):
    """
    Linear layer with power-law spectral parameterization.

    W = U · diag(σ₁ · k^{-α}) · V^T

    where U ∈ R^{out×rank}, V ∈ R^{in×rank}, and the spectrum
    is fully determined by (σ₁, α).
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, init_alpha: float = 0.7, fix_alpha: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.fix_alpha = fix_alpha

        # The two spectral parameters
        self.sigma1 = nn.Parameter(torch.tensor(1.0))
        if fix_alpha:
            # Fixed constant — not learned
            self.register_buffer("alpha", torch.tensor(init_alpha))
        else:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))

        # Basis vectors (learned, initialized orthogonal)
        U = torch.randn(out_dim, rank)
        V = torch.randn(in_dim, rank)
        # QR for orthogonal init when rank <= dim
        if rank <= out_dim:
            U, _ = torch.linalg.qr(U)
        if rank <= in_dim:
            V, _ = torch.linalg.qr(V)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)

        # Pre-compute mode indices (not a parameter)
        self.register_buffer("k_indices", torch.arange(1, rank + 1, dtype=torch.float))

    def get_spectrum(self) -> torch.Tensor:
        """Compute the power-law singular value spectrum."""
        # σ_k = σ₁ · k^{-α}
        return self.sigma1 * self.k_indices.pow(-self.alpha)

    def get_weight(self) -> torch.Tensor:
        """Reconstruct the full weight matrix from harmonic parameters."""
        spectrum = self.get_spectrum()  # (rank,)
        # W = U · diag(spectrum) · V^T
        # = (U * spectrum) @ V^T
        return (self.U * spectrum.unsqueeze(0)) @ self.V.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x @ W^T (matches nn.Linear convention)."""
        W = self.get_weight()
        return x @ W.t()

    @classmethod
    def from_weight(
        cls,
        W: torch.Tensor,
        rank: int | None = None,
    ) -> "HarmonicLinear":
        """
        Construct HarmonicLinear from an existing weight matrix.

        Performs SVD, fits power law to singular values, stores
        the top-rank basis vectors.
        """
        out_dim, in_dim = W.shape
        U_full, S_full, Vh_full = torch.linalg.svd(W.float(), full_matrices=False)

        if rank is None:
            # Auto-select: modes needed for 95% energy
            energy = torch.cumsum(S_full**2, dim=0) / (S_full**2).sum()
            rank = int((energy < 0.95).sum().item()) + 1
            rank = max(rank, 2)

        rank = min(rank, len(S_full))

        # Fit power law: log(σ_k) = log(σ₁) - α·log(k)
        S_np = S_full[:rank].detach().cpu().numpy()
        log_k = np.log(np.arange(1, rank + 1))
        log_s = np.log(S_np + 1e-10)

        # Linear regression in log-log space
        coeffs = np.polyfit(log_k, log_s, 1)
        alpha_fit = float(-coeffs[0])
        sigma1_fit = float(np.exp(coeffs[1]))

        # Create layer
        layer = cls(in_dim, out_dim, rank)
        with torch.no_grad():
            layer.sigma1.fill_(sigma1_fit)
            layer.alpha.fill_(alpha_fit)
            layer.U.copy_(U_full[:, :rank])
            layer.V.copy_(Vh_full[:rank, :].t())

        return layer

    def compression_ratio(self) -> float:
        """How many fewer params vs full matrix."""
        full = self.in_dim * self.out_dim
        harmonic = sum(p.numel() for p in self.parameters())
        return full / harmonic

    def extra_repr(self) -> str:
        return (f"in={self.in_dim}, out={self.out_dim}, rank={self.rank}, "
                f"compression={self.compression_ratio():.1f}x")
