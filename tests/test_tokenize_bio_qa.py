"""Tests for scripts/tokenize_bio_qa.py.

Uses a stubbed tokenizer so tests do not download Gemma-4 weights.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.tokenize_bio_qa import tokenize_pairs
from wavegpt.data_io import read_datafile


class StubTok:
    """Word-count-based tokenizer stub.

    Each whitespace-delimited word becomes one token (id assigned in
    insertion order). BOS is added when ``add_special_tokens=True``.
    """

    def __init__(self):
        self.vocab = {}
        self.next_id = 100  # reserve <100 for specials

    def _id(self, word):
        if word not in self.vocab:
            self.vocab[word] = self.next_id
            self.next_id += 1
        return self.vocab[word]

    def encode(self, text, add_special_tokens=False):
        ids = []
        if add_special_tokens:
            ids.append(1)  # bos
        for word in text.split():
            ids.append(self._id(word))
        return ids


def _write_pairs(path, pairs):
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")


def test_tokenize_pairs_writes_bins_masks_and_manifest(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    kg = [
        {"question": "What did Ray invent?",
         "answer": "I founded Kurzweil Computer Products in 1974.",
         "category": "biographical", "confidence": 1.0,
         "source_chunk_ids": ["a"]},
        {"question": "What did Ray create at MIT?",
         "answer": "I built reading machines for the blind.",
         "category": "biographical", "confidence": 1.0,
         "source_chunk_ids": ["b"]},
    ]
    probe = [
        {"question": "Who was Ray's father?",
         "answer": "My father, Fredric Kurzweil, was a concert pianist.",
         "probe_id": "bio_father", "category": "biographical",
         "confidence": 1.0, "source_chunk_ids": []},
        {"question": "Where did Ray go to college?",
         "answer": "I attended MIT starting in 1965.",
         "probe_id": "bio_college", "category": "biographical",
         "confidence": 1.0, "source_chunk_ids": []},
    ]
    _write_pairs(input_dir / "kg_pairs.jsonl", kg)
    _write_pairs(input_dir / "probe_pairs.jsonl", probe)

    out_dir = tmp_path / "output"
    tok = StubTok()

    # Force a 50/50 split so both bins get content with only 4 pairs
    manifest = tokenize_pairs(
        pairs=None,  # exercise the input-dir path
        tokenizer=tok,
        output_dir=out_dir,
        train_frac=0.5,
        seed=42,
        input_dir=input_dir,
        tokenizer_name="stub",
    )

    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    train_mask_path = out_dir / "train_mask.npy"
    val_mask_path = out_dir / "val_mask.npy"
    manifest_path = out_dir / "manifest.json"

    assert train_bin.exists()
    assert val_bin.exists()
    assert train_mask_path.exists()
    assert val_mask_path.exists()
    assert manifest_path.exists()

    train_tokens = read_datafile(str(train_bin))
    train_mask = np.load(train_mask_path)
    val_tokens = read_datafile(str(val_bin))
    val_mask = np.load(val_mask_path)

    # Mask is parallel to token stream
    assert len(train_mask) == len(train_tokens)
    assert len(val_mask) == len(val_tokens)

    # Mask has both 0s (scaffolding) and 1s (answer)
    assert 0.0 < train_mask.sum() < len(train_mask)
    assert 0.0 < val_mask.sum() < len(val_mask)

    # Manifest sanity
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == manifest
    assert on_disk["total_pairs"] == 4
    assert on_disk["train_pairs"] + on_disk["val_pairs"] == 4
    assert on_disk["train_tokens"] == len(train_tokens)
    assert on_disk["val_tokens"] == len(val_tokens)
    assert "train_mask_coverage" in on_disk
    assert "val_mask_coverage" in on_disk
    assert on_disk["tokenizer"] == "stub"
    assert on_disk["seed"] == 42
    assert on_disk["train_frac"] == 0.5
    assert "generated_at" in on_disk
    sources = {s["filename"]: s["pair_count"] for s in on_disk["source_files"]}
    assert sources == {"kg_pairs.jsonl": 2, "probe_pairs.jsonl": 2}


def test_format_pair_mask_layout():
    """Stand-alone sanity check on the per-pair format function."""
    from scripts.tokenize_bio_qa import format_pair_with_mask

    tok = StubTok()
    tokens, mask = format_pair_with_mask(
        question="What is X?",
        answer="X is the answer.",
        tokenizer=tok,
    )
    assert len(tokens) == len(mask)
    # Some scaffolding (0s) on either side, some 1s in the middle
    assert mask[0] == 0.0
    assert mask[-1] == 0.0
    ones = sum(1 for m in mask if m == 1.0)
    # answer has 4 words: "X is the answer."
    assert ones == 4
