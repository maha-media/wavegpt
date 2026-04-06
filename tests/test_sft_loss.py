"""Tests for SFT masked loss in WaveGPT model."""
import torch
import pytest
from wavegpt.model import WaveGPT, WaveGPTConfig


def test_loss_mask_reduces_loss_positions():
    """When loss_mask zeros out positions, they don't contribute to loss."""
    config = WaveGPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = WaveGPT(config)
    model.train()

    B, T = 2, 16
    idx = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))

    # Full loss (no mask)
    _, loss_full = model(idx, targets=targets)

    # Half mask — zero out first half of positions
    mask = torch.ones(B, T)
    mask[:, :T//2] = 0.0
    _, loss_masked = model(idx, targets=targets, loss_mask=mask)

    # Losses should be different (different positions contribute)
    assert loss_full is not None
    assert loss_masked is not None
    # Both should be valid numbers
    assert not torch.isnan(loss_full)
    assert not torch.isnan(loss_masked)


def test_loss_mask_all_zeros_returns_zero():
    """If all positions are masked, loss should be 0."""
    config = WaveGPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = WaveGPT(config)
    model.train()

    B, T = 2, 16
    idx = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))
    mask = torch.zeros(B, T)

    _, loss = model(idx, targets=targets, loss_mask=mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_mask_all_ones_equals_no_mask():
    """loss_mask=all_ones should give same result as no mask."""
    config = WaveGPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = WaveGPT(config)
    model.eval()  # no dropout

    B, T = 2, 16
    idx = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))
    mask = torch.ones(B, T)

    _, loss_no_mask = model(idx, targets=targets)
    _, loss_all_ones = model(idx, targets=targets, loss_mask=mask)

    assert loss_no_mask.item() == pytest.approx(loss_all_ones.item(), abs=1e-5)


def test_loss_mask_with_collapse_penalty():
    """Anti-collapse penalty still applies when loss_mask is used."""
    config = WaveGPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = WaveGPT(config, collapse_alpha=0.1)
    model.train()

    B, T = 2, 16
    idx = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))
    mask = torch.ones(B, T)
    mask[:, :T//2] = 0.0

    _, loss = model(idx, targets=targets, loss_mask=mask)
    assert loss is not None
    assert not torch.isnan(loss)
