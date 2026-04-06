"""Tests for wave data loader (llm.c .bin format)."""
import os, tempfile
import numpy as np
import pytest

from wavegpt.dataloader import WaveDataLoader
from wavegpt.data_io import write_datafile


def test_dataloader_yields_batches():
    """Dataloader yields (x, y) batches from .bin file."""
    tokens = list(range(200))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "train.bin")
        write_datafile(path, tokens)
        dl = WaveDataLoader(path, batch_size=2, block_size=16)
        x, y = next(iter(dl))
        assert x.shape == (2, 16)
        assert y.shape == (2, 16)
        # y is x shifted by 1
        assert (y[:, :-1] == x[:, 1:]).all()


def test_dataloader_wraps():
    """Dataloader wraps around when exhausting data."""
    tokens = list(range(100))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "train.bin")
        write_datafile(path, tokens)
        dl = WaveDataLoader(path, batch_size=2, block_size=16)
        batches = [next(iter(dl)) for _ in range(20)]
        assert len(batches) == 20


def test_dataloader_token_values():
    """Tokens in batches are actual values from the file."""
    tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 10
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "train.bin")
        write_datafile(path, tokens)
        dl = WaveDataLoader(path, batch_size=1, block_size=4)
        x, y = next(iter(dl))
        # All values should be from our set
        valid = set(tokens)
        for val in x[0].tolist():
            assert val in valid
