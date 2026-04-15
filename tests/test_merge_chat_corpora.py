"""Tests for scripts/merge_chat_corpora.py.

Builds two tiny tokenized corpora in a tmp dir, runs ``merge_corpora``,
and asserts the merged ratio, mask alignment, determinism, and manifest.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.merge_chat_corpora import compute_oversample_k, merge_corpora
from wavegpt.data_io import read_datafile, write_datafile


def _write_corpus(dir_path: Path, train_tokens, val_tokens, train_mask_val, val_mask_val):
    dir_path.mkdir(parents=True, exist_ok=True)
    write_datafile(str(dir_path / "train.bin"), list(train_tokens))
    write_datafile(str(dir_path / "val.bin"), list(val_tokens))
    np.save(
        dir_path / "train_mask.npy",
        np.full(len(train_tokens), train_mask_val, dtype=np.float32),
    )
    np.save(
        dir_path / "val_mask.npy",
        np.full(len(val_tokens), val_mask_val, dtype=np.float32),
    )


def test_compute_oversample_k():
    # B=3000, I=500 -> ceil(9000/3500) = 3
    assert compute_oversample_k(3000, 500, 0.30) == 3
    # B=1000, I=1000 -> ceil(3000/7000) = 1 (inject already plenty)
    assert compute_oversample_k(1000, 1000, 0.30) == 1
    # Inject bigger than base: k clamped to 1
    assert compute_oversample_k(100, 10_000, 0.30) == 1
    # Empty inject: still returns 1 (we never divide by zero)
    assert compute_oversample_k(1000, 0, 0.30) == 1


def test_merge_corpora_ratio_alignment_determinism(tmp_path):
    # Base: 3000 tokens, mask all 0.8
    # Inject: 500 tokens, mask all 1.0
    base_dir = tmp_path / "base"
    inject_dir = tmp_path / "inject"
    out_dir = tmp_path / "merged"

    base_train = list(range(100, 100 + 3000))
    base_val = list(range(100, 100 + 600))
    inject_train = list(range(9000, 9000 + 500))
    inject_val = list(range(9000, 9000 + 100))

    _write_corpus(base_dir, base_train, base_val, 0.8, 0.8)
    _write_corpus(inject_dir, inject_train, inject_val, 1.0, 1.0)

    manifest = merge_corpora(
        base_dir=base_dir,
        inject_dir=inject_dir,
        output_dir=out_dir,
        target_ratio=0.30,
        chunk_size=256,
        seed=42,
    )

    # k for the train split: B=3000, I=500 -> 3
    assert manifest["k"] == 3

    # Files exist
    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    train_mask_path = out_dir / "train_mask.npy"
    val_mask_path = out_dir / "val_mask.npy"
    manifest_path = out_dir / "manifest.json"
    assert train_bin.exists() and val_bin.exists()
    assert train_mask_path.exists() and val_mask_path.exists()
    assert manifest_path.exists()

    train_tokens = read_datafile(str(train_bin))
    val_tokens = read_datafile(str(val_bin))
    train_mask = np.load(train_mask_path)
    val_mask = np.load(val_mask_path)

    # Mask alignment: parallel to token stream
    assert len(train_mask) == len(train_tokens)
    assert len(val_mask) == len(val_tokens)

    # Train ratio within +/- 5 percentage points of target 0.30
    assert abs(manifest["train_inject_ratio"] - 0.30) <= 0.05
    assert abs(manifest["val_inject_ratio"] - 0.30) <= 0.05

    # Total tokens matches B + I*k per split
    assert manifest["train_total_tokens"] == 3000 + 500 * manifest["k"]
    assert len(train_tokens) == manifest["train_total_tokens"]

    # Mask values in merged stream are a subset of the input mask values
    # (compare with tolerance — float32 round-trip of 0.8 is 0.80000001192)
    unique_vals = np.unique(train_mask)
    assert len(unique_vals) == 2, f"expected 2 unique mask values, got {unique_vals}"
    assert any(np.isclose(unique_vals, 0.8)), unique_vals
    assert any(np.isclose(unique_vals, 1.0)), unique_vals

    # Manifest sanity
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == manifest
    for key in (
        "base_dir", "inject_dir", "target_inject_ratio", "k",
        "train_base_tokens", "train_inject_tokens", "train_inject_ratio",
        "train_total_tokens",
        "val_base_tokens", "val_inject_tokens", "val_inject_ratio",
        "val_total_tokens",
        "chunk_size", "seed", "generated_at",
    ):
        assert key in on_disk, f"manifest missing {key}"

    # Determinism: run again into a fresh dir, compare bytes
    out_dir2 = tmp_path / "merged2"
    merge_corpora(
        base_dir=base_dir,
        inject_dir=inject_dir,
        output_dir=out_dir2,
        target_ratio=0.30,
        chunk_size=256,
        seed=42,
    )
    assert (out_dir2 / "train.bin").read_bytes() == train_bin.read_bytes()
    assert (out_dir2 / "val.bin").read_bytes() == val_bin.read_bytes()
    assert np.array_equal(np.load(out_dir2 / "train_mask.npy"), train_mask)
    assert np.array_equal(np.load(out_dir2 / "val_mask.npy"), val_mask)


def test_merge_corpora_rejects_misaligned_mask(tmp_path):
    base_dir = tmp_path / "base"
    inject_dir = tmp_path / "inject"
    out_dir = tmp_path / "merged"

    _write_corpus(base_dir, list(range(100, 100 + 500)),
                  list(range(100, 200)), 0.8, 0.8)
    _write_corpus(inject_dir, list(range(9000, 9000 + 200)),
                  list(range(9000, 9050)), 1.0, 1.0)

    # Corrupt base train mask length
    np.save(base_dir / "train_mask.npy",
            np.zeros(499, dtype=np.float32))

    import pytest
    with pytest.raises(AssertionError):
        merge_corpora(
            base_dir=base_dir,
            inject_dir=inject_dir,
            output_dir=out_dir,
            target_ratio=0.30,
            chunk_size=64,
            seed=42,
        )
