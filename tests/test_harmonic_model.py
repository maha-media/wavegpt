"""Tests for HarmonicGPT — the spectral-parameterized transformer."""
import torch
import pytest
from wavegpt.harmonic_model import HarmonicGPT, HarmonicGPTConfig
from wavegpt.harmonic_linear import HarmonicLinear


def test_harmonic_gpt_forward():
    """HarmonicGPT produces correct output shape."""
    config = HarmonicGPTConfig(vocab_size=100, block_size=32, n_layer=2,
                                n_head=2, n_embd=64, rank_attn=8, rank_mlp=16)
    model = HarmonicGPT(config)
    x = torch.randint(0, 100, (2, 16))
    logits, _ = model(x)
    assert logits.shape == (2, 16, 100)


def test_harmonic_gpt_loss():
    """Forward with targets produces a loss."""
    config = HarmonicGPTConfig(vocab_size=100, block_size=32, n_layer=2,
                                n_head=2, n_embd=64, rank_attn=8, rank_mlp=16)
    model = HarmonicGPT(config)
    model.train()
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    _, loss = model(x, targets=y)
    assert loss is not None
    assert not torch.isnan(loss)
    loss.backward()


def test_harmonic_gpt_fewer_params():
    """HarmonicGPT uses far fewer weight params than standard."""
    config = HarmonicGPTConfig(vocab_size=50257, block_size=512, n_layer=6,
                                n_head=6, n_embd=384, rank_attn=30, rank_mlp=48)
    model = HarmonicGPT(config)
    weight_params = model.count_weight_params()
    total_params = model.count_params()

    # Standard model weight params:
    # Per layer: c_attn(1152×384) + c_proj(384×384) + mlp_fc(1536×384) + mlp_proj(384×1536)
    # = 442,368 + 147,456 + 589,824 + 589,824 = 1,769,472 per layer × 6 = 10,616,832
    standard_weight = 10_616_832

    print(f"HarmonicGPT weight params: {weight_params:,}")
    print(f"Standard weight params:    {standard_weight:,}")
    print(f"Compression: {standard_weight / weight_params:.1f}x")
    print(f"Total model params: {total_params:,}")

    assert weight_params < standard_weight / 3  # At least 3x reduction


def test_harmonic_gpt_with_loss_mask():
    """SFT masked loss works with HarmonicGPT."""
    config = HarmonicGPTConfig(vocab_size=100, block_size=32, n_layer=2,
                                n_head=2, n_embd=64, rank_attn=8, rank_mlp=16)
    model = HarmonicGPT(config)
    model.train()
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    mask = torch.ones(2, 16)
    mask[:, :8] = 0  # mask first half
    _, loss = model(x, targets=y, loss_mask=mask)
    assert loss is not None
    loss.backward()


def test_spectral_summary():
    """Can inspect sigma1 and alpha for all layers."""
    config = HarmonicGPTConfig(vocab_size=100, block_size=32, n_layer=2,
                                n_head=2, n_embd=64, rank_attn=8, rank_mlp=16)
    model = HarmonicGPT(config)
    summary = model.spectral_summary()
    assert len(summary) > 0
    for name, info in summary.items():
        assert "sigma1" in info
        assert "alpha" in info
        assert "rank" in info
        assert info["sigma1"] > 0


def test_alpha_is_learnable():
    """Alpha changes during training — the model learns its own decay rate."""
    config = HarmonicGPTConfig(vocab_size=100, block_size=32, n_layer=2,
                                n_head=2, n_embd=64, rank_attn=8, rank_mlp=16)
    model = HarmonicGPT(config)
    model.train()

    # Record initial alphas
    initial_alphas = {}
    for name, m in model.named_modules():
        if isinstance(m, HarmonicLinear):
            initial_alphas[name] = m.alpha.item()

    # One training step
    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4, 16))
    _, loss = model(x, targets=y)
    loss.backward()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()

    # Check alphas changed
    changed = 0
    for name, m in model.named_modules():
        if isinstance(m, HarmonicLinear):
            if abs(m.alpha.item() - initial_alphas[name]) > 1e-8:
                changed += 1

    assert changed > 0, "No alpha values changed during training"
