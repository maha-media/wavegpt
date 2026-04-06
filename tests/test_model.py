"""Tests for WaveGPT — GPT-2 with harmonic initialization."""
import torch
import numpy as np
import pytest

from wavegpt.model import WaveGPT, WaveGPTConfig


def test_forward_shape():
    """Forward pass produces correct logits shape."""
    cfg = WaveGPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = WaveGPT(cfg)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, 256)
    assert loss is None


def test_forward_with_targets():
    """Forward with targets produces scalar loss."""
    cfg = WaveGPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = WaveGPT(cfg)
    x = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, targets)
    assert loss is not None
    assert loss.dim() == 0
    assert loss.item() > 0


def test_harmonic_init():
    """Model initializes token embeddings from harmonic matrix."""
    cfg = WaveGPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=1, n_embd=4)
    token_harmonics = np.random.randn(10, 4).astype(np.float32)
    token_harmonics[8:] = 0  # tokens 8,9 are "unseen"
    model = WaveGPT(cfg, token_harmonics=token_harmonics)
    wte = model.transformer.wte.weight.detach().numpy()
    # Seen tokens match harmonic init
    np.testing.assert_allclose(wte[0], token_harmonics[0], atol=1e-6)
    np.testing.assert_allclose(wte[5], token_harmonics[5], atol=1e-6)
    # Unseen tokens get random init (not zero)
    assert not np.allclose(wte[8], np.zeros(4), atol=1e-6)


def test_sinusoidal_positions():
    """Positional encoding is fixed sinusoidal, not learned."""
    cfg = WaveGPTConfig(vocab_size=10, block_size=16, n_layer=1, n_head=1, n_embd=4)
    model = WaveGPT(cfg)
    # wpe should be a registered buffer, not a parameter
    assert "wpe" not in dict(model.transformer.named_parameters())
    # But should exist as buffer
    assert hasattr(model.transformer, "wpe")


def test_progressive_curriculum():
    """Harmonic gate masks higher dimensions early in training."""
    cfg = WaveGPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=1, n_embd=8)
    model = WaveGPT(cfg)
    # At step 0, low dims more active than high dims
    gate = model.harmonic_gate(step=0, total_steps=1000)
    assert gate.shape == (8,)
    assert gate[0].item() > gate[-1].item()
    # At final step, all dims ~1.0
    gate_end = model.harmonic_gate(step=1000, total_steps=1000)
    assert gate_end.min().item() > 0.9


def test_gate_affects_output():
    """Different curriculum steps produce different outputs when curriculum enabled."""
    cfg = WaveGPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = WaveGPT(cfg)
    model.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        logits_early, _ = model(x, step=0, total_steps=5000, use_curriculum=True)
        logits_late, _ = model(x, step=5000, total_steps=5000, use_curriculum=True)
    # Should be different because gate changes
    assert not torch.allclose(logits_early, logits_late)


def test_no_curriculum_same_output():
    """Without curriculum, step number doesn't affect output."""
    cfg = WaveGPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = WaveGPT(cfg)
    model.eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        logits_a, _ = model(x, step=0, total_steps=5000, use_curriculum=False)
        logits_b, _ = model(x, step=5000, total_steps=5000, use_curriculum=False)
    assert torch.allclose(logits_a, logits_b)


def test_param_count_rai15m():
    """Parameter count for rai-15m is in expected range."""
    cfg = WaveGPTConfig(vocab_size=50257, block_size=512, n_layer=6, n_head=6, n_embd=384)
    model = WaveGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    # ~30M params. Sinusoidal pos saves ~197K vs learned.
    assert 28_000_000 < n_params < 32_000_000, f"Got {n_params:,}"


def test_generate():
    """Model can generate tokens autoregressively."""
    cfg = WaveGPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = WaveGPT(cfg)
    model.eval()
    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=10)
    assert generated.shape == (1, 13)  # 3 prompt + 10 new
