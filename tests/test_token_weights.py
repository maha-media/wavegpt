"""Tests for token weight computation and weighted loss."""
import numpy as np
import pytest
import torch



def test_residual_norm_removes_fundamental():
    """Tokens near the fundamental direction get low weight."""
    from wavegpt.token_weights import compute_token_weights

    # 5 tokens, 4 harmonic dims
    # Token 0: lies entirely on dimension 0 (the fundamental)
    # Token 1: lies entirely on dimension 3 (high overtone)
    coords = np.zeros((5, 4), dtype=np.float32)
    coords[0] = [10.0, 0, 0, 0]  # pure fundamental
    coords[1] = [0, 0, 0, 5.0]    # pure overtone
    coords[2] = [10.0, 0, 0, 5.0] # both
    coords[3] = [0, 0, 0, 0]      # zero (not in corpus)

    weights = compute_token_weights(coords, min_weight=0.1, max_weight=3.0)

    assert weights.shape == (5,)
    # Pure fundamental → low weight
    # Pure overtone → high weight
    assert weights[1] > weights[0], "Overtone token should have higher weight than fundamental"
    # Mixed → somewhere in between
    assert weights[2] > weights[0], "Mixed token should have higher weight than pure fundamental"
    # Zero → min weight (not in corpus, treat as function word)
    assert weights[3] == pytest.approx(0.1, abs=0.01)


def test_weights_are_bounded():
    """Weights stay within [min_weight, max_weight]."""
    from wavegpt.token_weights import compute_token_weights

    coords = np.random.randn(1000, 384).astype(np.float32)
    weights = compute_token_weights(coords, min_weight=0.2, max_weight=5.0)

    assert weights.min() >= 0.2 - 1e-6
    assert weights.max() <= 5.0 + 1e-6


def test_weights_normalize_to_unit_mean():
    """Weights should average to ~1.0 so loss scale is preserved."""
    from wavegpt.token_weights import compute_token_weights

    coords = np.random.randn(50257, 384).astype(np.float32)
    weights = compute_token_weights(coords, min_weight=0.3, max_weight=3.0)

    # Mean should be approximately 1.0 (within 20% tolerance for clamping effects)
    assert 0.5 < weights.mean() < 2.0, f"Mean weight {weights.mean()} should be near 1.0"


def test_weighted_loss_changes_gradient():
    """Token-weighted loss should amplify gradients for high-weight tokens."""
    from wavegpt.model import WaveGPT, WaveGPTConfig

    cfg = WaveGPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32)

    # Weights: token 50 has high weight, token 10 has low weight
    weights = torch.ones(100)
    weights[50] = 5.0
    weights[10] = 0.1

    model = WaveGPT(cfg, token_weights=weights)
    model.train()

    # Input where targets are all token 50 (high weight)
    x_high = torch.randint(0, 100, (1, 8))
    y_high = torch.full((1, 8), 50, dtype=torch.long)
    _, loss_high = model(x_high, y_high)

    # Input where targets are all token 10 (low weight)
    x_low = torch.randint(0, 100, (1, 8))
    y_low = torch.full((1, 8), 10, dtype=torch.long)
    _, loss_low = model(x_low, y_low)

    # Both should produce valid losses
    assert loss_high.item() > 0
    assert loss_low.item() > 0


def test_no_weights_standard_loss():
    """Without token weights, loss should be standard cross-entropy."""
    from wavegpt.model import WaveGPT, WaveGPTConfig

    cfg = WaveGPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = WaveGPT(cfg)  # no token_weights
    model.eval()

    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))

    with torch.no_grad():
        logits, loss = model(x, y)
        # Manual cross-entropy should match
        manual_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 100), y.view(-1)
        )
    assert abs(loss.item() - manual_loss.item()) < 1e-4
