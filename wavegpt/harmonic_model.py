"""
HarmonicGPT — GPT-2 with power-law spectral parameterization.

Every linear layer replaced with HarmonicLinear:
  W = U · diag(σ₁ · k^{-α}) · V^T

The model learns (σ₁, α, U, V) per layer instead of full W matrices.
This enforces the spectral structure that gradient descent converges to anyway,
but with 10x fewer parameters and built-in spectral regularization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .harmonic_linear import HarmonicLinear
from .model import _build_sinusoidal_pe


@dataclass
class HarmonicGPTConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    # Rank budget per layer type (from autopsy: R90 values)
    rank_attn: int = 30       # Q/K/V + output projection
    rank_mlp: int = 48        # MLP up/down projection
    collapse_alpha: float = 0.0


class HarmonicCausalSelfAttention(nn.Module):
    """Self-attention with HarmonicLinear projections."""

    def __init__(self, config: HarmonicGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Replace c_attn (3*d, d) with 3 separate HarmonicLinear
        # (can't easily do combined QKV with harmonic param)
        self.q_proj = HarmonicLinear(config.n_embd, config.n_embd, config.rank_attn)
        self.k_proj = HarmonicLinear(config.n_embd, config.n_embd, config.rank_attn)
        self.v_proj = HarmonicLinear(config.n_embd, config.n_embd, config.rank_attn)
        self.c_proj = HarmonicLinear(config.n_embd, config.n_embd, config.rank_attn)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class HarmonicMLP(nn.Module):
    def __init__(self, config: HarmonicGPTConfig):
        super().__init__()
        self.c_fc = HarmonicLinear(config.n_embd, 4 * config.n_embd, config.rank_mlp)
        self.gelu = nn.GELU()
        self.c_proj = HarmonicLinear(4 * config.n_embd, config.n_embd, config.rank_mlp)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class HarmonicBlock(nn.Module):
    def __init__(self, config: HarmonicGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = HarmonicCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = HarmonicMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class HarmonicGPT(nn.Module):
    """
    GPT-2 with every linear layer parameterized as W = σ₁·k^{-α}·U·V^T.

    Same architecture as WaveGPT, but with ~10x fewer weight parameters.
    The spectral decay is baked into the parameterization.
    """

    def __init__(self, config: HarmonicGPTConfig):
        super().__init__()
        self.config = config
        self.collapse_alpha = config.collapse_alpha

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([HarmonicBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        # lm_head is still full — it's the embedding transposed (weight tying)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Fixed sinusoidal PE
        pe = _build_sinusoidal_pe(config.block_size, config.n_embd)
        self.transformer.register_buffer("wpe", pe)

        # Standard init for embeddings and layer norms
        self._init_non_harmonic()

    def _init_non_harmonic(self):
        """Initialize embedding and LayerNorm (HarmonicLinear handles its own init)."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        step: int = 0,
        total_steps: int = 5000,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        B, T = idx.size()

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe[:T]
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Anti-collapse
        collapse_penalty = torch.tensor(0.0, device=x.device)
        if self.collapse_alpha > 0 and self.training:
            batch_var = x.var(dim=0).mean()
            collapse_penalty = -self.collapse_alpha * torch.log(batch_var + 1e-8)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            raw_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none'
            )
            if loss_mask is not None:
                raw_loss = raw_loss * loss_mask.view(-1)
                n_active = loss_mask.sum()
                loss = raw_loss.sum() / n_active.clamp(min=1.0)
            else:
                loss = raw_loss.mean()

            if self.collapse_alpha > 0 and self.training:
                loss = loss + collapse_penalty

        return logits, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_weight_params(self) -> int:
        """Count only HarmonicLinear params (not embeddings/norms)."""
        total = 0
        for m in self.modules():
            if isinstance(m, HarmonicLinear):
                total += sum(p.numel() for p in m.parameters())
        return total

    def spectral_summary(self) -> dict:
        """Report sigma1 and alpha for every HarmonicLinear layer."""
        summary = {}
        for name, m in self.named_modules():
            if isinstance(m, HarmonicLinear):
                summary[name] = {
                    "sigma1": m.sigma1.item(),
                    "alpha": m.alpha.item(),
                    "rank": m.rank,
                    "compression": m.compression_ratio(),
                }
        return summary
