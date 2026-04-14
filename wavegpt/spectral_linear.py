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

Log-space parameterization (per_mode):
  Trains log(σ_k) instead of σ_k. Gradients are automatically
  scale-invariant across 7+ orders of magnitude (σ₁≈14 to σ_tail≈1e-7).
  Stolen from DoRA's insight that magnitude and direction need different
  update dynamics, extended to the full singular value spectrum.

Spectral tiers (from HFT/DoRA):
  Top modes (1..top_k): scaled LR — these carry most energy, gentle nudges
  Mid modes (top_k..tail_start): full LR — task adaptation lives here
  Tail modes (tail_start..rank): scaled LR — near-noise, keep quiet

Drift clamping (from LoRA's α/r):
  Optional max_log_drift caps how far any mode can deviate from its
  initial value in log-space. Like LoRA's scaling factor but explicit.

fp32 spectrum multiplication:
  The forward pass computes (x @ V) * S @ U^T. With bf16 throughout,
  the factored three-op chain accumulates more precision loss than the
  original fused x @ W CUDA kernel. σ₁≈14 compounded through 60
  transformer layers can overflow bf16 max (65504) on certain inputs
  that activate resonant paths. Fix: keep S in fp32, upcast xV to fp32
  for the spectrum multiply, then downcast back. One extra cast per
  layer, negligible cost, eliminates NaN from precision cascade.

  This is NOT "bad data" — the same inputs work fine with nn.Linear.
  The NaN is an artifact of splitting one matmul into three bf16 ops.

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
        max_log_drift: float | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.rank = S.shape[0]
        self.out_dim = U.shape[0]
        self.in_dim = V.shape[0]
        self._energy_captured = energy_captured
        self.max_log_drift = max_log_drift

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
            # Log-space: train log(σ_k) for scale-invariant gradients.
            # σ₁≈14 and σ_tail≈1e-7 get equal relative gradient treatment.
            log_s = torch.log(S.clamp(min=1e-12).clone())
            self.log_spectrum = nn.Parameter(log_s)
            # Frozen copy of initial log-spectrum for drift clamping
            self.register_buffer('log_spectrum_init', log_s.clone())
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
            log_s = self.log_spectrum
            # Drift clamping: don't let any mode drift more than max_log_drift
            # from its initial value (like LoRA's α/r scaling, but explicit)
            if self.max_log_drift is not None:
                log_s = torch.clamp(
                    log_s,
                    self.log_spectrum_init - self.max_log_drift,
                    self.log_spectrum_init + self.max_log_drift,
                )
            return torch.exp(log_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct W from spectrum + frozen bases."""
        spectrum = self.get_spectrum()               # keep fp32 for precision
        dev = x.device
        # Use prefetched GPU tensors if available (from PrefetchPipeline),
        # otherwise fall back to synchronous transfer.
        cache = getattr(self, '_prefetch_cache', None)
        if cache:
            V_pre = cache.pop('V', None)
            U_pre = cache.pop('U', None)
            res_prefetched = cache.pop('residual', None)
            # Handle dequantization for prefetched int8 buffers
            if V_pre is not None and V_pre.dtype == torch.int8:
                V_scale = cache.pop('V_scale', None) or getattr(self, 'V_scale').to(dev)
                V = V_pre.to(torch.bfloat16) * V_scale.unsqueeze(0)
            else:
                V = V_pre if V_pre is not None else self._dequant('V', dev)
            if U_pre is not None and U_pre.dtype == torch.int8:
                U_scale = cache.pop('U_scale', None) or getattr(self, 'U_scale').to(dev)
                U = U_pre.to(torch.bfloat16) * U_scale.unsqueeze(0)
            else:
                U = U_pre if U_pre is not None else self._dequant('U', dev)
        else:
            V = self._dequant('V', dev)
            U = self._dequant('U', dev)
            res_prefetched = None
        xV = x @ V                                   # (..., rank) in x.dtype
        # Full fp32 chain: bf16 matmul kernels accumulate in fp32 internally,
        # but our factored 3-matmul path doesn't get that — do it explicitly.
        out = ((xV.float() * spectrum) @ U.t().float()).to(x.dtype)
        if self.residual is not None:
            if res_prefetched is not None:
                res = res_prefetched
            else:
                res = self.residual.to(dev, non_blocking=True) if self.residual.device != dev else self.residual
            out = out + x @ res.t()
        if self.bias is not None:
            bias = self.bias.to(dev, non_blocking=True) if self.bias.device != dev else self.bias
            out = out + bias
        return out

    def spectral_report(self) -> dict:
        """Report fitted alpha, sigma1, energy captured, and drift from init."""
        with torch.no_grad():
            s = self.get_spectrum()
            s_np = s.detach().cpu().numpy()
            log_k = np.log(np.arange(1, self.rank + 1))
            log_s = np.log(np.abs(s_np) + 1e-10)
            coeffs = np.polyfit(log_k, log_s, 1)
            alpha = float(-coeffs[0])
            sigma1 = float(np.exp(coeffs[1]))

        report = {
            'alpha': alpha,
            'sigma1': sigma1,
            'rank': self.rank,
            'energy_captured': self._energy_captured,
            'mode': self.mode,
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
        }

        # Log-space drift: how far have we moved from init?
        if self.mode == 'per_mode':
            with torch.no_grad():
                drift = (self.log_spectrum - self.log_spectrum_init).abs()
                report['max_log_drift'] = float(drift.max().item())
                report['mean_log_drift'] = float(drift.mean().item())

        return report

    # -----------------------------------------------------------------
    # Spectral tier scaling (from DoRA + HFT)
    # -----------------------------------------------------------------

    def apply_tier_scaling(
        self,
        top_scale: float = 0.1,
        mid_scale: float = 1.0,
        tail_scale: float = 0.01,
        top_k: int = 50,
        tail_start: int = 500,
    ):
        """Scale gradients by spectral tier. Call after backward, before step."""
        if self.mode != 'per_mode' or self.log_spectrum.grad is None:
            return
        grad = self.log_spectrum.grad
        rank = grad.shape[0]
        scales = torch.full((rank,), mid_scale, device=grad.device)
        scales[:min(top_k, rank)] = top_scale
        if tail_start < rank:
            scales[tail_start:] = tail_scale
        grad.mul_(scales)

    # -----------------------------------------------------------------
    # Alternating spectral freeze (from HFT)
    # -----------------------------------------------------------------

    def set_mode_mask(self, mask: torch.Tensor | None):
        """Set which modes are trainable. mask: bool tensor (rank,), True = active."""
        if mask is not None:
            self.register_buffer('_mode_mask', mask, persistent=False)
        elif hasattr(self, '_mode_mask'):
            del self._mode_mask

    def apply_mode_mask(self):
        """Zero gradients for masked modes. Call after backward, before step."""
        if self.mode != 'per_mode' or self.log_spectrum.grad is None:
            return
        mask = getattr(self, '_mode_mask', None)
        if mask is not None:
            self.log_spectrum.grad.mul_(mask.to(self.log_spectrum.grad.device,
                                                dtype=self.log_spectrum.grad.dtype))

    # -----------------------------------------------------------------
    # Buffer quantization (from QLoRA — 4x memory reduction)
    # -----------------------------------------------------------------

    def quantize_buffers(self, bits: int = 8):
        """Quantize frozen U, V to int8. Saves ~50% buffer memory.

        Per-column absmax quantization: each column gets its own scale
        factor, preserving relative magnitudes within each singular vector.
        Dequantization happens automatically in forward().
        """
        for name in ('U', 'V'):
            buf = getattr(self, name, None)
            if buf is None or buf.dtype == torch.int8:
                continue
            # Per-column absmax quantization
            amax = buf.abs().amax(dim=0, keepdim=True).clamp(min=1e-10)
            scale = amax / 127.0
            quantized = (buf / scale).round().clamp(-127, 127).to(torch.int8)
            # Replace buffer with quantized version + scale
            self.register_buffer(name, quantized)
            self.register_buffer(f'{name}_scale', scale.squeeze(0))  # (rank,)

    def _dequant(self, name: str, device: torch.device) -> torch.Tensor:
        """Dequantize a buffer on-the-fly during forward pass."""
        buf = getattr(self, name)
        scale_name = f'{name}_scale'
        if buf.dtype == torch.int8 and hasattr(self, scale_name):
            scale = getattr(self, scale_name)
            scale = scale.to(device, non_blocking=True) if scale.device != device else scale
            buf = buf.to(device, non_blocking=True) if buf.device != device else buf
            return buf.to(torch.bfloat16) * scale.unsqueeze(0)
        # Not quantized — normal path
        return buf.to(device, non_blocking=True) if buf.device != device else buf

    # -----------------------------------------------------------------
    # Multiple spectra (from LoRA adapter merging — but trivially simple)
    # -----------------------------------------------------------------

    def save_spectrum(self) -> dict:
        """Save only the learnable spectrum (tiny — e.g. 20KB per layer)."""
        if self.mode == 'sigma1':
            return {'sigma1': self.sigma1.data.cpu()}
        return {'log_spectrum': self.log_spectrum.data.cpu()}

    def load_spectrum(self, state: dict):
        """Load a saved spectrum into this layer."""
        if self.mode == 'sigma1' and 'sigma1' in state:
            self.sigma1.data.copy_(state['sigma1'])
        elif 'log_spectrum' in state:
            self.log_spectrum.data.copy_(state['log_spectrum'])
        elif 'spectrum' in state:
            # Backward compat: raw spectrum → convert to log
            self.log_spectrum.data.copy_(torch.log(state['spectrum'].clamp(min=1e-12)))

    @staticmethod
    def blend_spectra(spectra: list[dict], weights: list[float]) -> dict:
        """Blend multiple saved spectra with given weights.

        Linear interpolation in log-space = geometric mean of amplitudes.
        blend([rai, other], [0.7, 0.3]) → 70% RAI, 30% other personality.
        """
        assert len(spectra) == len(weights)
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        key = 'log_spectrum' if 'log_spectrum' in spectra[0] else 'sigma1'
        blended = sum(w * s[key] for w, s in zip(weights, spectra))
        return {key: blended}

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
        has_residual: bool = False,
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
        # S=ones → log(S)=zeros — placeholder, overwritten by load_state_dict
        S = torch.ones(rank, dtype=torch.float32)
        bias = torch.zeros(out_dim, dtype=dtype) if has_bias else None
        residual = torch.zeros(out_dim, in_dim, dtype=dtype) if has_residual else None
        return cls(U, S, V, mode=mode, alpha_fit=INV_PHI, bias=bias,
                   residual=residual, k0=0.0)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int | None = None,
        mode: str = 'per_mode',
        keep_residual: bool = False,
        residual_dtype: torch.dtype | None = None,
    ) -> 'SpectralLinear':
        """
        Decompose a trained nn.Linear into SpectralLinear.

        Performs SVD on the weight matrix, keeps top-rank modes,
        fits the bent power law: σ_k = A · (k + k₀)^{-1/φ}.
        Falls back to simple power-law fit if scipy unavailable.
        """
        orig_dtype = linear.weight.data.dtype  # preserve original dtype (e.g. BF16)
        W = linear.weight.data.float()  # (out, in) — SVD needs float32
        # GPU-accelerated SVD if CUDA available and matrix fits in VRAM
        svd_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        W_svd = W.to(svd_device)
        out_dim, in_dim = W.shape
        max_rank = min(out_dim, in_dim)

        U_full, S_full, Vh_full = torch.linalg.svd(W_svd, full_matrices=False)
        U_full, S_full, Vh_full = U_full.cpu(), S_full.cpu(), Vh_full.cpu()
        del W_svd
        W = W.cpu()
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
            residual_cast_dtype = residual_dtype if residual_dtype is not None else orig_dtype
            residual = (W - W_approx).to(residual_cast_dtype).contiguous()

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
        drift_str = f", max_drift={self.max_log_drift}" if self.max_log_drift is not None else ""
        quant_str = ", quantized=int8" if (hasattr(self, 'U_scale') and self.U.dtype == torch.int8) else ""
        return (
            f"in={self.in_dim}, out={self.out_dim}, rank={self.rank}, "
            f"mode={self.mode}, α_fit={self.alpha_fit:.4f}{k0_str}{drift_str}{quant_str}, "
            f"energy={self._energy_captured:.3f}"
        )
