"""
SFT DataLoader — conversation-aware batching with loss masks.

Tokenizes multi-turn conversations with role markers, creates loss masks
(0 for user/system/tool tokens, 1 for assistant tokens), and packs
into fixed-length sequences for training.

Conversations classify into harmonic layers:
  C: Simple factual Q&A (no reasoning, short)
  G: Explanations with reasoning (single-turn)
  D: Multi-step workflows (tool use, agent patterns)
  A: Deep reasoning (long thinking traces)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tiktoken
import torch

from .data_io import write_datafile, read_datafile, GPT2_EOT

# Special tokens for role markers (using rare GPT-2 token IDs)
# These are above the normal vocabulary to avoid collisions
ROLE_TOKENS = {
    "user": 50257 - 4,       # <|user|>
    "assistant": 50257 - 3,  # <|assistant|>
    "system": 50257 - 2,     # <|system|>
    "tool": 50257 - 1,       # <|tool|>
}

_enc = None

def _get_enc():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


def tokenize_conversation(
    turns: list[dict],
    include_reasoning: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Tokenize a multi-turn conversation into tokens + loss mask.

    Returns:
        tokens: list of token IDs
        loss_mask: list of 0/1 ints (same length as tokens)
                   1 = compute loss here (assistant response)
                   0 = don't compute loss (user/system/tool)
    """
    enc = _get_enc()
    tokens = []
    loss_mask = []

    for turn in turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        reasoning = turn.get("reasoning_content", "") or ""

        # Role marker token
        role_tok = ROLE_TOKENS.get(role, ROLE_TOKENS["user"])
        tokens.append(role_tok)
        loss_mask.append(0)  # role marker is never loss-active

        # For assistant with reasoning, prepend reasoning to content
        if role == "assistant" and include_reasoning and reasoning:
            full_text = reasoning + "\n" + content
        else:
            full_text = content

        # Tokenize content
        content_tokens = enc.encode_ordinary(full_text) if full_text.strip() else []

        # Loss mask: 1 for assistant tokens, 0 for everything else
        is_assistant = 1 if role == "assistant" else 0
        tokens.extend(content_tokens)
        loss_mask.extend([is_assistant] * len(content_tokens))

    # End of conversation
    tokens.append(GPT2_EOT)
    loss_mask.append(0)

    return tokens, loss_mask


def classify_harmonic_layer(turns: list[dict]) -> str:
    """
    Classify a conversation into a harmonic layer: C, G, D, or A.

    C (fundamental): Simple Q&A, no reasoning, short answer
    G (1st fifth):   Explanation with reasoning, moderate depth
    D (2nd fifth):   Multi-step, tool use, agent workflows
    A (3rd fifth):   Deep reasoning (>5K chars thinking)
    """
    roles = set(t.get("role", "") for t in turns)
    n_turns = len(turns)
    total_reasoning = sum(len(t.get("reasoning_content", "") or "") for t in turns)
    total_content = sum(len(t.get("content", "")) for t in turns)

    # D: Multi-step with tools or many turns
    if "tool" in roles and n_turns > 3:
        return "D"

    # A: Deep reasoning
    if total_reasoning > 5000:
        return "A"

    # G: Has reasoning but not deep
    if total_reasoning > 0 and n_turns <= 4:
        return "G"

    # C: Simple, short, no reasoning
    if n_turns <= 2 and total_content < 2000 and total_reasoning == 0:
        return "C"

    # G: Longer single-turn explanations
    if n_turns <= 2:
        return "G"

    # D: Multi-turn without tools
    return "D"


class SFTDataLoader:
    """
    Data loader for SFT training with loss masks.

    Stores a flat token stream + loss mask stream, yields (x, y, mask) batches
    where mask indicates which positions contribute to the loss.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        loss_mask: np.ndarray,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
    ):
        self.tokens = tokens
        self.loss_mask = loss_mask
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.pos = 0
        self.n_tokens = len(tokens)

    @classmethod
    def from_conversations(
        cls,
        conversations: list[list[dict]],
        output_dir: str,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
        include_reasoning: bool = True,
    ) -> "SFTDataLoader":
        """Build loader from a list of conversations."""
        all_tokens = []
        all_masks = []

        for conv in conversations:
            toks, mask = tokenize_conversation(conv, include_reasoning)
            all_tokens.extend(toks)
            all_masks.extend(mask)

        tokens_np = np.array(all_tokens, dtype=np.uint16)
        masks_np = np.array(all_masks, dtype=np.uint8)

        # Save to disk
        os.makedirs(output_dir, exist_ok=True)
        write_datafile(os.path.join(output_dir, "sft_tokens.bin"), all_tokens)
        np.save(os.path.join(output_dir, "sft_loss_mask.npy"), masks_np)

        return cls(tokens_np, masks_np, batch_size, block_size, device)

    @classmethod
    def from_files(
        cls,
        data_dir: str,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
    ) -> "SFTDataLoader":
        """Load from pre-saved .bin + .npy files."""
        tokens = read_datafile(os.path.join(data_dir, "sft_tokens.bin"))
        masks = np.load(os.path.join(data_dir, "sft_loss_mask.npy"))
        return cls(tokens, masks, batch_size, block_size, device)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of (x, y, loss_mask) tensors."""
        B, T = self.batch_size, self.block_size
        x_buf = np.zeros((B, T), dtype=np.int64)
        y_buf = np.zeros((B, T), dtype=np.int64)
        m_buf = np.zeros((B, T), dtype=np.float32)
        n = self.n_tokens

        for i in range(B):
            start = self.pos % max(n - T - 1, 1)
            end = start + T + 1

            if end <= n:
                seq = self.tokens[start:end].astype(np.int64)
                mask_seq = self.loss_mask[start:end].astype(np.float32)
            else:
                remaining = n - start
                seq = np.concatenate([
                    self.tokens[start:].astype(np.int64),
                    self.tokens[:T + 1 - remaining].astype(np.int64),
                ])
                mask_seq = np.concatenate([
                    self.loss_mask[start:].astype(np.float32),
                    self.loss_mask[:T + 1 - remaining].astype(np.float32),
                ])

            x_buf[i] = seq[:T]
            y_buf[i] = seq[1:T + 1]
            # Loss mask aligns with y (targets), not x (inputs)
            m_buf[i] = mask_seq[1:T + 1]
            self.pos += T

        x = torch.tensor(x_buf, dtype=torch.long, device=self.device)
        y = torch.tensor(y_buf, dtype=torch.long, device=self.device)
        mask = torch.tensor(m_buf, dtype=torch.float32, device=self.device)
        return x, y, mask

    def __len__(self) -> int:
        return max(1, self.n_tokens // (self.batch_size * self.block_size))
