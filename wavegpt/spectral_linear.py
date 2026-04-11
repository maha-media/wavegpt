"""
SpectralLinear — post-training spectral decomposition of nn.Linear.

Unlike HarmonicLinear (which trains from scratch inside the spectral
parameterization), SpectralLinear decomposes a TRAINED weight matrix
into (U, S, V) and freezes the geometry (U, V). Only the spectral
amplitudes are learnable.

The double-slit insight: you can't train inside the wave equation
(it diverges at scale), but you can observe the converged structure
and fine-tune the amplitudes.

Two modes:
  - sigma1: W = σ₁ · Σ_k k^{-α_fit} · u_k · v_k^T  (1 learnable param)
  - per_mode: W = Σ_k s_k · u_k · v_k^T               (rank learnable params)

The equation: gradient descent converges to W = σ₁ · Σ k^{-1/φ} · u_k · v_k^T.
We observe the converged structure, then fine-tune the amplitudes.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import numpy as np


PHI = (1 + 5**0.5) / 2
INV_PHI = 1 / PHI


class SpectralLinear(nn.Module):
    """
    Post-training spectral layer.

    U, V are FROZEN (buffers). Only spectrum is learnable.
    Two modes:
      - sigma1: one scalar controls the whole spectrum via power law
      - per_mode: one amplitude per singular value, free spectral shape
    """

    def __init__(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        V: torch.Tensor,
        mode: str = 'per_mode',
        alpha_fit: float | None = None,
        bias: torch.Tensor | None = None,
        energy_captured: float = 1.0,
        residual: torch.Tensor | None = None,
        k0: float | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.rank = S.shape[0]
        self.out_dim = U.shape[0]
        self.in_dim = V.shape[0]
        self._energy_captured = energy_captured

        # Geometry: FROZEN
        self.register_buffer('U', U)   # (out_dim, rank)
        self.register_buffer('V', V)   # (in_dim, rank)

        # Residual: FROZEN (Pythagorean comma preservation)
        if residual is not None:
            self.register_buffer('residual', residual)  # (out_dim, in_dim)
        else:
            self.residual = None

        # Bent power law: k₀ (spectral offset, FROZEN)
        if k0 is not None:
            self.register_buffer('k0', torch.tensor(k0, dtype=torch.float))
        else:
            self.k0 = None

        # Fitted alpha from power-law regression
        self.alpha_fit = alpha_fit if alpha_fit is not None else INV_PHI

        if mode == 'sigma1':
            # One scalar — reconstruct spectrum as σ₁ · k^{-α_fit}
            self.sigma1 = nn.Parameter(torch.tensor(S[0].item()))
            self.register_buffer(
                'k_indices',
                torch.arange(1, self.rank + 1, dtype=torch.float),
            )
        elif mode == 'per_mode':
            # One amplitude per mode — free spectral shape
            self.spectrum = nn.Parameter(S.clone())
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Bias (frozen if present)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def get_spectrum(self) -> torch.Tensor:
        """Return the current spectral amplitudes."""
        if self.mode == 'sigma1':
            return self.sigma1 * self.k_indices.pow(-self.alpha_fit)
        else:
            return self.spectrum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct W from spectrum + frozen bases."""
        spectrum = self.get_spectrum().to(x.dtype)
        # W = U · diag(s) · V^T → x @ W^T = x @ V · diag(s) · U^T
        xV = x @ self.V                              # (..., rank)
        xVs = xV * spectrum                          # broadcast spectrum
        out = xVs @ self.U.t()                       # (..., out_dim)
        if self.residual is not None:
            out = out + x @ self.residual.t()         # frozen comma correction
        if self.bias is not None:
            out = out + self.bias
        return out

    def spectral_report(self) -> dict:
        """Report fitted alpha, sigma1, energy captured."""
        with torch.no_grad():
            s = self.get_spectrum()
            s_np = s.detach().cpu().numpy()
            log_k = np.log(np.arange(1, self.rank + 1))
            log_s = np.log(np.abs(s_np) + 1e-10)
            coeffs = np.polyfit(log_k, log_s, 1)
            alpha = float(-coeffs[0])
            sigma1 = float(np.exp(coeffs[1]))

        return {
            'alpha': alpha,
            'sigma1': sigma1,
            'rank': self.rank,
            'energy_captured': self._energy_captured,
            'mode': self.mode,
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
        }

    def to_linear(self) -> nn.Linear:
        """Reconstruct a standard nn.Linear from current spectral params."""
        with torch.no_grad():
            spectrum = self.get_spectrum().to(self.U.dtype)
            W = (self.U * spectrum.unsqueeze(0)) @ self.V.t()
        linear = nn.Linear(self.in_dim, self.out_dim, bias=self.bias is not None)
        linear.weight.data = W
        if self.bias is not None:
            linear.bias.data = self.bias.clone()
        return linear

    @classmethod
    def from_shape(
        cls,
        out_dim: int,
        in_dim: int,
        rank: int = 256,
        mode: str = 'per_mode',
        has_bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> 'SpectralLinear':
        """
        Create SpectralLinear with correct shapes but dummy data.
        Used for fast loading from saved state_dict (no SVD needed).
        Call model.load_state_dict() after to fill real values.
        """
        rank = min(rank, min(out_dim, in_dim))
        U = torch.zeros(out_dim, rank, dtype=dtype)
        V = torch.zeros(in_dim, rank, dtype=dtype)
        S = torch.ones(rank, dtype=torch.float32)
        bias = torch.zeros(out_dim, dtype=dtype) if has_bias else None
        return cls(U, S, V, mode=mode, alpha_fit=INV_PHI, bias=bias, k0=0.0)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int | None = None,
        mode: str = 'per_mode',
        keep_residual: bool = False,
    ) -> 'SpectralLinear':
        """
        Decompose a trained nn.Linear into SpectralLinear.

        Performs SVD on the weight matrix, keeps top-rank modes,
        fits the bent power law: σ_k = A · (k + k₀)^{-1/φ}.
        Falls back to simple power-law fit if scipy unavailable.
        """
        orig_dtype = linear.weight.data.dtype  # preserve original dtype (e.g. BF16)
        W = linear.weight.data.float().cpu()  # (out, in) — SVD needs float32
        out_dim, in_dim = W.shape
        max_rank = min(out_dim, in_dim)

        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
        total_energy = (S_full ** 2).sum()

        if rank is None:
            energy_ratio = torch.cumsum(S_full ** 2, 0) / total_energy
            rank = int((energy_ratio < 0.95).sum().item()) + 1
            rank = max(rank, 2)
        rank = min(rank, max_rank)

        U = U_full[:, :rank].contiguous()
        S = S_full[:rank].contiguous()
        V = Vh_full[:rank, :].t().contiguous()  # (in_dim, rank)

        energy_captured = float(((S ** 2).sum() / total_energy).item())

        # Fit bent power law: σ_k = A · (k + k₀)^{-1/φ}
        k0_val = None
        try:
            from .harmonic_prior import fit_bent_power_law
            bent_fit = fit_bent_power_law(S_full)
            k0_val = bent_fit['k0']
            alpha_fit = INV_PHI  # always 1/φ with bent model
        except Exception:
            # Fallback: simple log-log fit
            s_np = S.detach().numpy()
            log_k = np.log(np.arange(1, rank + 1))
            log_s = np.log(s_np + 1e-10)
            coeffs = np.polyfit(log_k, log_s, 1)
            alpha_fit = float(-coeffs[0])

        bias = linear.bias.data.cpu().clone() if linear.bias is not None else None

        # Compute frozen residual: W - U_r @ diag(S_r) @ V_r^T
        residual = None
        if keep_residual:
            W_approx = (U * S.unsqueeze(0)) @ V.t()
            residual = (W - W_approx).to(orig_dtype).contiguous()

        # Cast geometry to original dtype (e.g. BF16) to save memory
        # S (spectrum) stays float32 — it's the learnable parameter
        U = U.to(orig_dtype)
        V = V.to(orig_dtype)
        if bias is not None:
            bias = bias.to(orig_dtype)

        return cls(
            U, S, V,
            mode=mode,
            alpha_fit=alpha_fit,
            bias=bias,
            energy_captured=energy_captured,
            residual=residual,
            k0=k0_val,
        )

    def extra_repr(self) -> str:
        k0_str = f", k₀={self.k0.item():.1f}" if self.k0 is not None else ""
        return (
            f"in={self.in_dim}, out={self.out_dim}, rank={self.rank}, "
            f"mode={self.mode}, α_fit={self.alpha_fit:.4f}{k0_str}, "
            f"energy={self._energy_captured:.3f}"
        )
