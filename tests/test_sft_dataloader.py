"""Tests for SFT dataloader — conversation-aware batching with loss masks."""
import json
import os
import tempfile
import numpy as np
import torch
import pytest

from wavegpt.sft_dataloader import (
    tokenize_conversation,
    SFTDataLoader,
    classify_harmonic_layer,
    ROLE_TOKENS,
)
from wavegpt.data_io import write_datafile


def test_tokenize_simple_conversation():
    """Basic user/assistant conversation tokenizes with role markers."""
    conversation = [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence."},
    ]
    tokens, loss_mask = tokenize_conversation(conversation)

    assert len(tokens) == len(loss_mask)
    assert len(tokens) > 0

    # Loss mask should be 0 for user tokens, 1 for assistant tokens
    assert 0 in loss_mask  # user tokens masked
    assert 1 in loss_mask  # assistant tokens active

    # Role tokens should be present
    assert ROLE_TOKENS["user"] in tokens
    assert ROLE_TOKENS["assistant"] in tokens


def test_tokenize_with_reasoning():
    """Reasoning content is included in assistant response."""
    conversation = [
        {"role": "user", "content": "Solve: 2+2"},
        {"role": "assistant", "content": "4",
         "reasoning_content": "Let me think. 2 plus 2 equals 4."},
    ]
    tokens, loss_mask = tokenize_conversation(conversation)

    # Should be longer than without reasoning
    conv_no_reason = [
        {"role": "user", "content": "Solve: 2+2"},
        {"role": "assistant", "content": "4"},
    ]
    tokens_nr, _ = tokenize_conversation(conv_no_reason)

    assert len(tokens) > len(tokens_nr)


def test_tokenize_multiturn():
    """Multi-turn conversations maintain correct loss masking."""
    conversation = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": "2"},
    ]
    tokens, loss_mask = tokenize_conversation(conversation)

    # System and user tokens should have loss_mask=0
    # Assistant tokens should have loss_mask=1
    assert sum(loss_mask) > 0  # some assistant tokens
    assert sum(1 - m for m in loss_mask) > 0  # some masked tokens


def test_classify_harmonic_layer():
    """Conversations classify into C, G, D, A layers."""
    # C: Simple Q&A, no reasoning, short
    c_conv = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    assert classify_harmonic_layer(c_conv) == "C"

    # G: Has reasoning, single turn
    g_conv = [
        {"role": "user", "content": "Explain photosynthesis"},
        {"role": "assistant", "content": "Plants convert light to energy.",
         "reasoning_content": "The user wants an explanation of the process."},
    ]
    assert classify_harmonic_layer(g_conv) == "G"

    # D: Multi-turn with tool use
    d_conv = [
        {"role": "user", "content": "Search for X"},
        {"role": "assistant", "content": "I'll search for that."},
        {"role": "tool", "content": "Result: X is Y"},
        {"role": "assistant", "content": "Based on my search, X is Y."},
    ]
    assert classify_harmonic_layer(d_conv) == "D"

    # A: Deep reasoning (long reasoning_content)
    a_conv = [
        {"role": "user", "content": "Prove the Riemann hypothesis"},
        {"role": "assistant", "content": "Here's my analysis...",
         "reasoning_content": "x" * 6000},  # >5K chars
    ]
    assert classify_harmonic_layer(a_conv) == "A"


def test_sft_dataloader_batches():
    """SFT dataloader yields batches with tokens and loss masks."""
    conversations = []
    for i in range(20):
        conversations.append([
            {"role": "user", "content": f"Question {i}: What is {i}?"},
            {"role": "assistant", "content": f"The answer is {i}."},
        ])

    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SFTDataLoader.from_conversations(
            conversations=conversations,
            output_dir=tmpdir,
            batch_size=2,
            block_size=64,
            device="cpu",
        )

        x, y, mask = loader.get_batch()
        assert x.shape == (2, 64)
        assert y.shape == (2, 64)
        assert mask.shape == (2, 64)
        assert mask.dtype == torch.float32
        # Some positions should be masked (0) and some active (1)
        assert mask.sum() > 0
        assert mask.sum() < mask.numel()
