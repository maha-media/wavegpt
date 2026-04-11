"""
φ-Codec: spectral quantization using the φ-power law as a prediction prior.

Standard quantization treats all weights equally.
We know: σ_k = A · (k + k₀)^{-(1/φ)^p}

Instead of quantizing raw values, we quantize RESIDUALS from the
predicted φ-curve. Residuals have ~100× smaller dynamic range → same
bits encode ~100× more precision.

Three tiers based on φ-informed spectral position:
  Tier 1 (k ≤ k₀):    plateau modes — highest precision (dominant information)
  Tier 2 (k₀ < k ≤ n/φ): power-law body — medium precision on residuals
  Tier 3 (k > n/φ):    spectral tail — low precision on residuals

Usage:
    from wavegpt.phi_codec import PhiCodec
    codec = PhiCodec()

    # Encode
    compressed = codec.encode_layer(W, layer_type='attn_o')

    # Decode
    W_hat = codec.decode_layer(compressed)

    # Full model
    codec.compress_model(model, output_dir='runs/gemma4-phi-quantized/')
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

# F/L exponents by layer type
FL_EXPONENTS = {
    'attn_o': INV_PHI ** (1/3),     # 0.8518
    'attn_q': INV_PHI ** (5/4),     # 0.5480
    'attn_k': INV_PHI ** (2/11),    # 0.9162
    'attn_v': INV_PHI ** (3/7),     # 0.8136
    'mlp_gate': INV_PHI ** (4/7),   # 0.7596
    'mlp_up': INV_PHI ** (8/11),    # 0.7047
    'mlp_down': INV_PHI ** (5/7),   # 0.7091
}

# Tier precision: standard hardware-aligned widths (32/16/8-bit)
# The φ-curve prediction is the innovation, not exotic bit packing.
DEFAULT_BITS = {
    'tier1_sv': 32,    # singular values: float32 (plateau — most critical)
    'tier1_uv': 32,    # U/V columns: float32
    'tier2_sv': 16,    # residuals from φ-curve: float16
    'tier2_uv': 16,    # U/V columns: float16
    'tier3_sv': 8,     # residuals from φ-curve: int8
    'tier3_uv': 8,     # U/V columns: int8 (σ_k is tiny → error is σ_k × ||δu||)
}


def classify_layer(name: str) -> str | None:
    """Classify a module name into a layer type."""
    name = name.lower()
    if 'o_proj' in name or 'out_proj' in name or 'c_proj' in name:
        return 'attn_o'
    if 'q_proj' in name:
        return 'attn_q'
    if 'k_proj' in name:
        return 'attn_k'
    if 'v_proj' in name:
        return 'attn_v'
    if 'gate' in name:
        return 'mlp_gate'
    if 'up_proj' in name:
        return 'mlp_up'
    if 'down_proj' in name:
        return 'mlp_down'
    return None


@dataclass
class QuantizedTensor:
    """A uniformly quantized tensor with scale + zero-point."""
    codes: np.ndarray    # uint8/uint16 quantization codes
    scale: float
    zero_point: float
    n_bits: int
    shape: tuple

    def dequantize(self) -> np.ndarray:
        return self.codes.astype(np.float32) * self.scale + self.zero_point

    def nbytes(self) -> int:
        """Storage size in bytes."""
        code_bits = self.codes.size * self.n_bits
        overhead = 8 + 8  # scale + zero_point floats
        return (code_bits + 7) // 8 + overhead


@dataclass
class CompressedLayer:
    """φ-compressed representation of a single weight matrix."""
    # Shape info
    out_dim: int
    in_dim: int
    n_sv: int

    # φ-curve parameters (the prediction prior — 3 floats)
    A: float
    k0: float
    alpha: float
    layer_type: str

    # Tier boundaries
    tier1_end: int   # k₀ boundary
    tier2_end: int   # n/φ boundary

    # Tier 1: plateau (full precision SVs, quantized U/V)
    S_tier1: np.ndarray          # float32 — exact
    U_tier1: QuantizedTensor     # quantized columns
    V_tier1: QuantizedTensor     # quantized columns

    # Tier 2: power-law body (quantized residuals + quantized U/V)
    residuals_tier2: QuantizedTensor   # δ = S - predicted
    U_tier2: QuantizedTensor
    V_tier2: QuantizedTensor

    # Tier 3: tail (quantized residuals + quantized U/V)
    residuals_tier3: QuantizedTensor
    U_tier3: QuantizedTensor
    V_tier3: QuantizedTensor

    # Optional bias
    bias: np.ndarray | None = None

    def storage_bytes(self) -> int:
        """Total compressed storage in bytes."""
        curve = 3 * 4  # A, k0, alpha as float32
        t1 = (self.S_tier1.nbytes +
              self.U_tier1.nbytes() + self.V_tier1.nbytes())
        t2 = (self.residuals_tier2.nbytes() +
              self.U_tier2.nbytes() + self.V_tier2.nbytes())
        t3 = (self.residuals_tier3.nbytes() +
              self.U_tier3.nbytes() + self.V_tier3.nbytes())
        bias = self.bias.nbytes if self.bias is not None else 0
        return curve + t1 + t2 + t3 + bias

    def original_bytes(self) -> int:
        """Original weight storage (bf16)."""
        return self.out_dim * self.in_dim * 2

    def compression_ratio(self) -> float:
        return self.original_bytes() / max(self.storage_bytes(), 1)


def quantize_uniform(values: np.ndarray, n_bits: int) -> QuantizedTensor:
    """Uniform quantization to n_bits."""
    shape = values.shape
    flat = values.flatten().astype(np.float32)

    if len(flat) == 0 or n_bits >= 16:
        return QuantizedTensor(
            codes=flat.astype(np.float32) if n_bits >= 16 else np.zeros(0, dtype=np.uint8),
            scale=1.0, zero_point=0.0, n_bits=n_bits, shape=shape,
        )

    vmin, vmax = flat.min(), flat.max()
    if vmax == vmin:
        return QuantizedTensor(
            codes=np.zeros(len(flat), dtype=np.uint16),
            scale=0.0, zero_point=float(vmin), n_bits=n_bits, shape=shape,
        )

    n_levels = 2 ** n_bits
    scale = (vmax - vmin) / (n_levels - 1)
    codes = np.round((flat - vmin) / scale).astype(np.uint16)
    codes = np.clip(codes, 0, n_levels - 1)

    return QuantizedTensor(
        codes=codes, scale=float(scale), zero_point=float(vmin),
        n_bits=n_bits, shape=shape,
    )


class PhiCodec:
    """
    φ-informed spectral quantization codec.

    Encodes weight matrices using the φ-power law as a prediction prior,
    quantizing only residuals from the predicted curve.
    """

    def __init__(self, bits: dict | None = None):
        self.bits = bits or DEFAULT_BITS

    def encode_layer(
        self,
        W: np.ndarray | torch.Tensor,
        layer_type: str | None = None,
        bias: np.ndarray | torch.Tensor | None = None,
    ) -> CompressedLayer:
        """
        Encode a weight matrix into φ-compressed representation.

        Full SVD (no truncation) → fit φ-curve → quantize residuals + U/V.
        """
        if isinstance(W, torch.Tensor):
            W = W.detach().cpu().float().numpy()
        if isinstance(bias, torch.Tensor):
            bias = bias.detach().cpu().float().numpy()

        m, n = W.shape
        alpha = FL_EXPONENTS.get(layer_type, INV_PHI)

        # Full SVD — no truncation
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        V = Vt.T  # (in_dim, n_sv)
        n_sv = len(S)

        # Fit φ-curve: σ_k = A · (k + k₀)^{-α}
        A_fit, k0_fit, predicted = self._fit_curve(S, alpha)

        # Residuals from prediction
        residuals = S - predicted

        # Tier boundaries
        k0_int = max(1, int(k0_fit))
        k_phi = int(n_sv / PHI)
        tier1_end = min(k0_int, n_sv)
        tier2_end = min(k_phi, n_sv)

        # === Tier 1: plateau (k ≤ k₀) ===
        S_t1 = S[:tier1_end].astype(np.float32)
        U_t1 = quantize_uniform(U[:, :tier1_end], self.bits['tier1_uv'])
        V_t1 = quantize_uniform(V[:, :tier1_end], self.bits['tier1_uv'])

        # === Tier 2: power-law body (k₀ < k ≤ n/φ) ===
        res_t2 = residuals[tier1_end:tier2_end]
        U_t2 = quantize_uniform(U[:, tier1_end:tier2_end], self.bits['tier2_uv'])
        V_t2 = quantize_uniform(V[:, tier1_end:tier2_end], self.bits['tier2_uv'])
        res_t2_q = quantize_uniform(res_t2, self.bits['tier2_sv'])

        # === Tier 3: tail (k > n/φ) ===
        res_t3 = residuals[tier2_end:]
        U_t3 = quantize_uniform(U[:, tier2_end:], self.bits['tier3_uv'])
        V_t3 = quantize_uniform(V[:, tier2_end:], self.bits['tier3_uv'])
        res_t3_q = quantize_uniform(res_t3, self.bits['tier3_sv'])

        return CompressedLayer(
            out_dim=m, in_dim=n, n_sv=n_sv,
            A=float(A_fit), k0=float(k0_fit), alpha=float(alpha),
            layer_type=layer_type or 'unknown',
            tier1_end=tier1_end, tier2_end=tier2_end,
            S_tier1=S_t1,
            U_tier1=U_t1, V_tier1=V_t1,
            residuals_tier2=res_t2_q,
            U_tier2=U_t2, V_tier2=V_t2,
            residuals_tier3=res_t3_q,
            U_tier3=U_t3, V_tier3=V_t3,
            bias=bias,
        )

    def decode_layer(self, comp: CompressedLayer) -> np.ndarray:
        """
        Decode a compressed layer back to a full weight matrix.

        Reconstructs spectrum from φ-curve + dequantized residuals,
        then W = U · diag(S) · V^T.
        """
        # Reconstruct spectrum
        k = np.arange(1, comp.n_sv + 1, dtype=np.float64)
        predicted = comp.A * (k + comp.k0) ** (-comp.alpha)

        # Tier 1: exact SVs
        S = np.zeros(comp.n_sv, dtype=np.float32)
        S[:comp.tier1_end] = comp.S_tier1

        # Tier 2: predicted + dequantized residual
        res_t2 = comp.residuals_tier2.dequantize()
        S[comp.tier1_end:comp.tier2_end] = (
            predicted[comp.tier1_end:comp.tier2_end].astype(np.float32) + res_t2
        )

        # Tier 3: predicted + dequantized residual
        res_t3 = comp.residuals_tier3.dequantize()
        S[comp.tier2_end:] = (
            predicted[comp.tier2_end:].astype(np.float32) + res_t3
        )

        # Reconstruct U and V from quantized tiers
        U = np.zeros((comp.out_dim, comp.n_sv), dtype=np.float32)
        V = np.zeros((comp.in_dim, comp.n_sv), dtype=np.float32)

        U[:, :comp.tier1_end] = comp.U_tier1.dequantize().reshape(comp.out_dim, -1)
        U[:, comp.tier1_end:comp.tier2_end] = comp.U_tier2.dequantize().reshape(comp.out_dim, -1)
        U[:, comp.tier2_end:] = comp.U_tier3.dequantize().reshape(comp.out_dim, -1)

        V[:, :comp.tier1_end] = comp.V_tier1.dequantize().reshape(comp.in_dim, -1)
        V[:, comp.tier1_end:comp.tier2_end] = comp.V_tier2.dequantize().reshape(comp.in_dim, -1)
        V[:, comp.tier2_end:] = comp.V_tier3.dequantize().reshape(comp.in_dim, -1)

        # W = U · diag(S) · V^T
        W = (U * S[np.newaxis, :]) @ V.T
        return W

    def encode_decode_error(
        self,
        W: np.ndarray | torch.Tensor,
        layer_type: str | None = None,
    ) -> dict:
        """Encode then decode, return error metrics."""
        if isinstance(W, torch.Tensor):
            W_np = W.detach().cpu().float().numpy()
        else:
            W_np = W

        comp = self.encode_layer(W_np, layer_type)
        W_hat = self.decode_layer(comp)

        frobenius_orig = np.sqrt(np.sum(W_np ** 2))
        frobenius_err = np.sqrt(np.sum((W_np - W_hat) ** 2))
        rel_error = frobenius_err / frobenius_orig if frobenius_orig > 0 else 0
        max_error = np.max(np.abs(W_np - W_hat))

        return {
            'rel_error': float(rel_error),
            'max_error': float(max_error),
            'compression_ratio': comp.compression_ratio(),
            'storage_mb': comp.storage_bytes() / 1e6,
            'original_mb': comp.original_bytes() / 1e6,
            'tiers': (comp.tier1_end,
                      comp.tier2_end - comp.tier1_end,
                      comp.n_sv - comp.tier2_end),
            'curve': (comp.A, comp.k0, comp.alpha),
        }

    def _fit_curve(
        self, S: np.ndarray, alpha: float,
    ) -> tuple[float, float, np.ndarray]:
        """Fit A and k₀ for the φ-curve with fixed α."""
        from scipy.optimize import curve_fit

        n_sv = len(S)
        k = np.arange(1, n_sv + 1, dtype=np.float64)
        s = S.astype(np.float64)

        def bent_pl(k, A, k0):
            return A * (k + k0) ** (-alpha)

        try:
            popt, _ = curve_fit(
                bent_pl, k, s,
                p0=[s[0] * 50, max(n_sv * 0.1, 10)],
                bounds=([0, 0], [s[0] * 1000, n_sv * 2]),
                maxfev=10000,
            )
            A_fit, k0_fit = popt
            predicted = bent_pl(k, A_fit, k0_fit)
        except Exception:
            A_fit, k0_fit = float(s[0]), 0.0
            predicted = s[0] * k ** (-alpha)

        return A_fit, k0_fit, predicted
