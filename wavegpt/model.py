"""
WaveGPT — GPT-2 whose dimensions are harmonic modes.

Three innovations over standard GPT-2:
  1. Token embeddings initialized from harmonic coordinates (SVD of corpus)
  2. Fixed sinusoidal positional encoding (not learned)
  3. Progressive harmonic curriculum (fundamentals first, overtones later)

Architecture is standard GPT-2. The intelligence is in the coordinate system.

384 = 3 × 2^7. The perfect fifth enters the octave chain.
1024 = 2^10. Pure octaves — redundant.
The ratio 384:1024 = 3:8 — a fifth dropped two octaves.
The 2% residual variance is the Pythagorean comma.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WaveGPTConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384  # = 3 × 2^7 harmonic modes
    dropout: float = 0.1
    bias: bool = False  # no bias in linear layers (GPT-2 style)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    With harmonic-ordered dimensions and 6 heads of 64 dims each:
      Head 1: dims 1-64   = fundamental modes
      Head 6: dims 321-384 = highest overtones
    """

    def __init__(self, config: WaveGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: WaveGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: WaveGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------


def _build_sinusoidal_pe(block_size: int, n_embd: int) -> torch.Tensor:
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Low dimensions get low frequencies → slow variation over position.
    High dimensions get high frequencies → fast variation.

    With harmonic-ordered dims, this means:
      Fundamental modes (low dims) → smooth positional signal
      Overtone modes (high dims) → rapid positional signal

    The positional frequencies match the semantic frequencies.
    """
    pe = torch.zeros(block_size, n_embd)
    position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, n_embd, 2, dtype=torch.float) * -(math.log(10000.0) / n_embd)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: n_embd // 2])
    return pe


# ---------------------------------------------------------------------------
# WaveGPT
# ---------------------------------------------------------------------------


class WaveGPT(nn.Module):
    """
    GPT-2 in harmonic coordinates.

    The architecture is identical to GPT-2. The difference:
    - Token embeddings initialized from SVD harmonic projections
    - Positional encoding is fixed sinusoidal (not learned)
    - Progressive harmonic gate during training (fundamentals → overtones)
    """

    def __init__(
        self,
        config: WaveGPTConfig,
        token_harmonics: np.ndarray | None = None,
        token_weights: torch.Tensor | None = None,
        wave_lens: np.ndarray | None = None,
        collapse_alpha: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.collapse_alpha = collapse_alpha

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Fixed sinusoidal positional encoding (registered as buffer, not parameter)
        pe = _build_sinusoidal_pe(config.block_size, config.n_embd)
        self.transformer.register_buffer("wpe", pe)

        # Token-weighted loss: per-token multipliers for cross-entropy
        if token_weights is not None:
            self.register_buffer("token_weights", token_weights.float())
        else:
            self.token_weights = None

        # Initialize weights
        self.apply(self._init_weights)

        # Harmonic token embedding initialization (particle approach)
        if token_harmonics is not None:
            self._init_from_harmonics(token_harmonics)

        # Wave attention initialization (wave approach)
        # Sets Q/K projections so attention computes harmonic similarity
        if wave_lens is not None:
            self._init_wave_attention(wave_lens)

    def _init_weights(self, module):
        """Standard GPT-2 weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _init_from_harmonics(self, token_harmonics: np.ndarray):
        """
        Initialize token embeddings from harmonic coordinates.
        (Particle approach — sets fixed positions per token.)

        Tokens with non-zero harmonic coords get their SVD position.
        Tokens with zero coords (not in corpus) keep random init.
        """
        th = torch.from_numpy(token_harmonics)
        mask = (th != 0).any(dim=1)  # which tokens have harmonic coords
        with torch.no_grad():
            self.transformer.wte.weight[mask] = th[mask]

    def _init_wave_attention(self, wave_lens: np.ndarray):
        """
        Initialize attention Q/K projections from harmonic wave lens.
        (Wave approach — attention computes harmonic similarity dynamically.)

        The wave lens is an orthogonal (n_embd, n_embd) matrix that orders
        dimensions by harmonic variance. Head 1 sees fundamentals, last head
        sees overtones.

        This makes attention compute interference in harmonic space from step 0,
        regardless of how the embeddings are initialized.

        Leaves V projection and all other weights at standard init.
        """
        L = torch.from_numpy(wave_lens).float()  # (n_embd, n_embd)
        d = self.config.n_embd
        scale = 0.02  # GPT-2 init scale

        with torch.no_grad():
            for block in self.transformer.h:
                W = block.attn.c_attn.weight  # (3*d, d)
                # W[:d] = Q part, W[d:2d] = K part, W[2d:3d] = V part
                # Initialize Q and K with scaled wave lens
                # Each row of Q/K projects input to one harmonic direction
                W[:d] = L * scale  # Q = wave lens
                W[d:2*d] = L * scale  # K = wave lens
                # V stays at random init (standard GPT-2)

    # ── Progressive Harmonic Curriculum ──

    def harmonic_gate(self, step: int, total_steps: int) -> torch.Tensor:
        """
        Compute per-dimension gate values for progressive curriculum.

        Low dims (fundamentals) activate early.
        High dims (overtones) activate late.
        All dims fully active by ~60% of training.

        Returns: (n_embd,) tensor with values in [0, 1]
        """
        d = self.config.n_embd
        # Each dim activates at a different step
        # dim 0 activates at step 0, dim d-1 at ~60% of training
        activate_steps = torch.linspace(0, total_steps * 0.6, d)
        temperature = max(total_steps * 0.05, 1.0)  # smooth ramp
        gate = torch.sigmoid((step - activate_steps) / temperature)
        return gate

    # ── Forward ──

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        step: int = 0,
        total_steps: int = 5000,
        use_curriculum: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        # Token + position embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        pos_emb = self.transformer.wpe[:T]   # (T, C)
        x = tok_emb + pos_emb

        # Apply harmonic gate only if curriculum is enabled
        if use_curriculum:
            gate = self.harmonic_gate(step, total_steps).to(device)  # (C,)
            x = x * gate

        x = self.transformer.drop(x)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Anti-collapse: measure representation diversity BEFORE lm_head
        # This is where nuance lives or dies
        collapse_penalty = torch.tensor(0.0, device=x.device)
        if self.collapse_alpha > 0 and self.training:
            # Variance of hidden states across batch dimension
            # High variance = diverse representations = nuance preserved
            # Low variance = everything looks the same = semantic collapse
            batch_var = x.var(dim=0).mean()  # average variance across positions & dims
            # Penalty: -log(var) → high when var is low (collapse), low when var is high
            collapse_penalty = -self.collapse_alpha * torch.log(batch_var + 1e-8)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            if self.token_weights is not None:
                # Weighted cross-entropy: content words get higher loss
                raw_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none'
                )
                weights = self.token_weights[targets.view(-1)]
                loss = (raw_loss * weights).mean()
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Add anti-collapse penalty
            if self.collapse_alpha > 0 and self.training:
                loss = loss + collapse_penalty

        return logits, loss

    # ── Generation ──

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation. Gate fully open (step=total_steps)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, step=999999, total_steps=1)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # ── Utilities ──

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
