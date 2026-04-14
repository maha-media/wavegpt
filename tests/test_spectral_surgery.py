"""Tests for spectral_surgery — model-level decomposition."""
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 64, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 16, bias=False)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def test_decompose_replaces_linears():
    """All nn.Linear layers should be replaced with SpectralLinear."""
    from wavegpt.spectral_surgery import spectral_decompose
    from wavegpt.spectral_linear import SpectralLinear
    model = TinyModel()
    decomposed = spectral_decompose(model, rank=16)
    assert isinstance(decomposed.linear1, SpectralLinear)
    assert isinstance(decomposed.linear2, SpectralLinear)


def test_decompose_output_close():
    """Decomposed model output should approximate original."""
    from wavegpt.spectral_surgery import spectral_decompose
    model = TinyModel()
    x = torch.randn(2, 5, 32)
    y_orig = model(x).detach()
    decomposed = spectral_decompose(model, rank=16)
    y_dec = decomposed(x).detach()
    rel_err = (y_orig - y_dec).norm() / y_orig.norm()
    assert rel_err < 0.75, f"Relative error {rel_err:.4f}"


def test_spectral_report_all_layers():
    """Report should have one entry per linear layer."""
    from wavegpt.spectral_surgery import spectral_decompose, spectral_report
    model = TinyModel()
    decomposed = spectral_decompose(model, rank=16)
    report = spectral_report(decomposed)
    assert len(report) == 2
    for name, info in report.items():
        assert 'alpha' in info
        assert 'sigma1' in info


def test_layer_filter_selects_subset():
    """layer_filter controls which indexed layers get decomposed."""
    from wavegpt.spectral_surgery import spectral_decompose
    from wavegpt.spectral_linear import SpectralLinear

    class FourLinears(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(16, 16, bias=False)
            self.b = nn.Linear(16, 16, bias=False)
            self.c = nn.Linear(16, 16, bias=False)
            self.d = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.d(self.c(self.b(self.a(x))))

    model = FourLinears()
    spectral_decompose(
        model, rank=8, mode='per_mode',
        layer_filter=lambda name, idx: idx % 2 == 0,
    )
    assert isinstance(model.a, SpectralLinear)
    assert isinstance(model.b, nn.Linear)
    assert isinstance(model.c, SpectralLinear)
    assert isinstance(model.d, nn.Linear)


def test_layer_filter_by_name():
    """layer_filter can filter by layer name string."""
    from wavegpt.spectral_surgery import spectral_decompose
    from wavegpt.spectral_linear import SpectralLinear
    model = TinyModel()
    spectral_decompose(
        model, rank=8, mode='per_mode',
        layer_filter=lambda name, idx: 'linear2' in name,
    )
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.linear2, SpectralLinear)


def test_skip_pattern():
    """Layers matching skip pattern should not be decomposed."""
    from wavegpt.spectral_surgery import spectral_decompose
    from wavegpt.spectral_linear import SpectralLinear
    model = TinyModel()
    decomposed = spectral_decompose(model, rank=16, skip_patterns=['linear1'])
    assert isinstance(decomposed.linear1, nn.Linear)
    assert isinstance(decomposed.linear2, SpectralLinear)


def test_adaptive_rank_decomposition():
    """With adaptive rank, layers get per-layer ranks based on their α."""
    from wavegpt.spectral_surgery import spectral_decompose
    from wavegpt.spectral_linear import SpectralLinear
    model = TinyModel()
    decomposed = spectral_decompose(
        model, rank='adaptive', base_rank=8, mode='per_mode',
    )
    assert isinstance(decomposed.linear1, SpectralLinear)
    assert isinstance(decomposed.linear2, SpectralLinear)
    # Ranks should be > 0 and possibly different
    assert decomposed.linear1.rank > 0
    assert decomposed.linear2.rank > 0


def test_adaptive_rank_with_residual():
    """Adaptive rank + residual = lossless output."""
    from wavegpt.spectral_surgery import spectral_decompose
    model = TinyModel()
    x = torch.randn(2, 5, 32)
    y_orig = model(x).detach()
    decomposed = spectral_decompose(
        model, rank='adaptive', base_rank=8, mode='per_mode',
        keep_residual=True,
    )
    y_dec = decomposed(x).detach()
    torch.testing.assert_close(y_orig, y_dec, atol=1e-4, rtol=1e-3)


def test_keep_residual_passthrough():
    """keep_residual flag should propagate to SpectralLinear."""
    from wavegpt.spectral_surgery import spectral_decompose
    model = TinyModel()
    decomposed = spectral_decompose(
        model, rank=8, mode='per_mode', keep_residual=True,
    )
    assert decomposed.linear1.residual is not None
    assert decomposed.linear2.residual is not None


def test_scaffold_fixed_rank():
    """Scaffold creates SpectralLinear shells at fixed rank."""
    from wavegpt.spectral_surgery import spectral_scaffold
    from wavegpt.spectral_linear import SpectralLinear
    model = TinyModel()
    spectral_scaffold(model, rank=8, mode='per_mode')
    assert isinstance(model.linear1, SpectralLinear)
    assert isinstance(model.linear2, SpectralLinear)
    assert model.linear1.rank == 8
    assert model.linear2.rank == 8


def test_scaffold_from_state_dict_variable_rank():
    """Scaffold infers per-layer rank from saved state_dict."""
    from wavegpt.spectral_surgery import spectral_decompose, spectral_scaffold
    from wavegpt.spectral_linear import SpectralLinear
    # First, decompose with different ranks
    model1 = TinyModel()
    decomposed = spectral_decompose(model1, rank=12, mode='per_mode')
    sd = decomposed.state_dict()
    # Verify the saved spectrum shapes
    assert sd['linear1.log_spectrum'].shape[0] == 12
    # Now scaffold a fresh model using that state_dict
    model2 = TinyModel()
    spectral_scaffold(model2, rank=999, mode='per_mode', state_dict=sd)
    # Should use rank from state_dict (12), not the passed rank (999)
    assert model2.linear1.rank == 12
    assert model2.linear2.rank == 12
    # Load should succeed
    model2.load_state_dict(sd, strict=False)
    x = torch.randn(2, 5, 32)
    out = model2(x)
    assert out.shape == (2, 5, 16)
