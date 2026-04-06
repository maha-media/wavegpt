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
    """Conversations classify across the full circle of fifths."""
    from wavegpt.sft_dataloader import CIRCLE_OF_FIFTHS

    # C (score=0): Simple Q&A, no reasoning, short content
    c_conv = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    assert classify_harmonic_layer(c_conv) == "C"

    # G (score~1): Short content + light reasoning
    g_conv = [
        {"role": "user", "content": "Explain photosynthesis"},
        {"role": "assistant", "content": "Plants convert light to energy.",
         "reasoning_content": "The user wants an explanation."},
    ]
    assert classify_harmonic_layer(g_conv) == "G"

    # Higher layers with more complexity
    # D (score=2): Reasoning 1K-5K range (2pt) + short turns (0pt) + short content (0pt)
    d_conv = [
        {"role": "user", "content": "Explain photosynthesis."},
        {"role": "assistant", "content": "Plants use light.",
         "reasoning_content": "x" * 1500},
    ]
    assert classify_harmonic_layer(d_conv) == "D"

    # A (score~3): Multi-step reasoning
    a_conv = [
        {"role": "user", "content": "Step 1"},
        {"role": "assistant", "content": "Done 1", "reasoning_content": "x" * 3000},
        {"role": "user", "content": "Step 2"},
        {"role": "assistant", "content": "Done 2", "reasoning_content": "x" * 3000},
    ]
    result = classify_harmonic_layer(a_conv)
    # Multi-turn (4 turns = 1pt) + reasoning 6K (3pt) = score 4 → E or higher
    assert result in CIRCLE_OF_FIFTHS
    assert CIRCLE_OF_FIFTHS.index(result) >= 3  # at least A-level complexity

    # F (score=11): Maximum complexity — deep reasoning + tools + many turns + long content
    f_conv = [
        {"role": "system", "content": "You are a complex agent."},
        {"role": "user", "content": "Complex task... " * 500},
        {"role": "assistant", "content": "Working...", "reasoning_content": "x" * 20000},
        {"role": "tool", "content": "Result 1"},
        {"role": "assistant", "content": "Continuing..."},
        {"role": "tool", "content": "Result 2"},
        {"role": "tool", "content": "Result 3"},
        {"role": "tool", "content": "Result 4"},
        {"role": "tool", "content": "Result 5"},
        {"role": "assistant", "content": "Final answer... " * 500},
    ]
    result = classify_harmonic_layer(f_conv)
    assert CIRCLE_OF_FIFTHS.index(result) >= 8  # high complexity end

    # All results must be valid circle of fifths notes
    for conv in [c_conv, g_conv, d_conv, a_conv, f_conv]:
        assert classify_harmonic_layer(conv) in CIRCLE_OF_FIFTHS


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
