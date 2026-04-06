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
    init_alpha: float = 0.7   # initial spectral decay exponent
    fix_alpha: bool = False   # if True, alpha is a fixed constant (not learned)
    # Two-constant model: α=1/φ for representation, α=1 for projection
    alpha_proj: float | None = None  # if set, c_proj layers use this α
    collapse_alpha: float = 0.0
    ortho_lambda: float = 0.0      # orthogonality regularization on U, V


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
        alpha_r = config.init_alpha   # representation α
        alpha_p = config.alpha_proj if config.alpha_proj is not None else config.init_alpha
        fix = config.fix_alpha
        self.q_proj = HarmonicLinear(config.n_embd, config.n_embd, config.rank_attn, alpha_r, fix)
        self.k_proj = HarmonicLinear(config.n_embd, config.n_embd, config.rank_attn, alpha_r, fix)
        self.v_proj = HarmonicLinear(config.n_embd, config.n_embd, config.rank_attn, alpha_r, fix)
        self.c_proj = HarmonicLinear(config.n_embd, config.n_embd, config.rank_attn, alpha_p, fix)
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
        alpha_r = config.init_alpha
        alpha_p = config.alpha_proj if config.alpha_proj is not None else config.init_alpha
        fix = config.fix_alpha
        self.c_fc = HarmonicLinear(config.n_embd, 4 * config.n_embd, config.rank_mlp, alpha_r, fix)
        self.gelu = nn.GELU()
        self.c_proj = HarmonicLinear(4 * config.n_embd, config.n_embd, config.rank_mlp, alpha_p, fix)
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
        self.ortho_lambda = config.ortho_lambda

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

            # Orthogonality regularization on U, V basis vectors
            if self.ortho_lambda > 0 and self.training:
                ortho_loss = torch.tensor(0.0, device=device)
                n_layers = 0
                for m in self.modules():
                    if isinstance(m, HarmonicLinear):
                        ortho_loss = ortho_loss + m.orthogonality_loss()
                        n_layers += 1
                if n_layers > 0:
                    loss = loss + self.ortho_lambda * ortho_loss / n_layers

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

    def spectral_param_groups(self, lr: float, weight_decay: float = 0.1) -> list[dict]:
        """
        Build optimizer param groups with per-mode learning rate scaling.

        The power-law spectrum creates a gradient imbalance: mode 1 gets
        30x more gradient than mode 256. This equalizes by giving each
        HarmonicLinear's U, V a higher learning rate compensating for
        the spectral attenuation.

        Groups:
          1. sigma1 params: base lr (these are scalars, fine as-is)
          2. alpha params: base lr (if learnable)
          3. U, V params: lr * sqrt(mean_k / k_eff) — boosted
          4. Everything else (embeddings, norms): base lr + weight_decay
        """
        import math

        harmonic_uv = []     # U, V basis vectors — need boosted lr
        harmonic_sigma = []  # sigma1 — scalar, base lr
        harmonic_alpha = []  # alpha — scalar, base lr (if learnable)
        other_decay = []     # embeddings, dense layers with decay
        other_nodecay = []   # layer norms, biases

        harmonic_param_ids = set()
        for m in self.modules():
            if isinstance(m, HarmonicLinear):
                harmonic_uv.append(m.U)
                harmonic_uv.append(m.V)
                harmonic_sigma.append(m.sigma1)
                harmonic_param_ids.add(id(m.U))
                harmonic_param_ids.add(id(m.V))
                harmonic_param_ids.add(id(m.sigma1))
                if not m.fix_alpha and isinstance(m.alpha, nn.Parameter):
                    harmonic_alpha.append(m.alpha)
                    harmonic_param_ids.add(id(m.alpha))

        for name, p in self.named_parameters():
            if id(p) in harmonic_param_ids:
                continue
            if p.ndim < 2 or 'ln' in name or 'bias' in name:
                other_nodecay.append(p)
            else:
                other_decay.append(p)

        # Compute effective lr boost for U, V
        # Mean mode index across all layers
        mean_ranks = []
        for m in self.modules():
            if isinstance(m, HarmonicLinear):
                mean_ranks.append(m.rank)
        avg_rank = sum(mean_ranks) / len(mean_ranks) if mean_ranks else 64
        # Geometric mean of mode indices 1..rank: exp(mean(log(k)))
        # ≈ rank / e for large rank
        geo_mean_k = math.exp(sum(math.log(k) for k in range(1, int(avg_rank)+1)) / avg_rank)
        # Boost factor: compensate for average spectral attenuation
        # Modes see gradients scaled by k^{-α}. We boost lr by k^{α} on average.
        alpha_val = 1.0 / ((1 + 5**0.5) / 2)  # 1/φ
        boost = geo_mean_k ** alpha_val

        groups = [
            {"params": harmonic_uv, "lr": lr * boost, "weight_decay": weight_decay,
             "label": f"U,V (lr×{boost:.1f})"},
            {"params": harmonic_sigma, "lr": lr, "weight_decay": 0.0,
             "label": "sigma1"},
        ]
        if harmonic_alpha:
            groups.append(
                {"params": harmonic_alpha, "lr": lr, "weight_decay": 0.0,
                 "label": "alpha"}
            )
        groups.extend([
            {"params": other_decay, "lr": lr, "weight_decay": weight_decay,
             "label": "other (decay)"},
            {"params": other_nodecay, "lr": lr, "weight_decay": 0.0,
             "label": "other (no decay)"},
        ])

        # Filter empty groups
        groups = [g for g in groups if g["params"]]

        for g in groups:
            n = sum(p.numel() for p in g["params"])
            print(f"  param group: {g.get('label','?'):20s} | {n:>10,} params | lr={g['lr']:.2e} | wd={g['weight_decay']}")

        return groups

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
