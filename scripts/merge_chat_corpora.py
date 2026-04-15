"""Merge two tokenized+masked chat corpora, oversampling the ``inject``
corpus so it reaches a target fraction of total tokens (default 30%).

Input layout (per corpus directory, same for ``--base`` and ``--inject``)::

    <dir>/train.bin        # wavegpt binary token stream
    <dir>/train_mask.npy   # float32 loss mask, same length as train.bin
    <dir>/val.bin
    <dir>/val_mask.npy

Output (under ``--output``)::

    train.bin, train_mask.npy
    val.bin,   val_mask.npy
    manifest.json

Recipe: for each split, replicate the inject stream ``k`` times with
``k = ceil(3B / (7I))`` (derived from ``(I*k)/(B+I*k) = 0.30``),
concatenate with the base stream, then chunk-shuffle the result at
``--chunk-size`` granularity with a fixed seed so mask alignment is
preserved and long runs of pure inject don't appear in any one epoch.

The controller runs this pod-side against
``data/rai-gemma4-chat-v2`` (base) and ``data/ray/bio_qa_chat`` (inject)
writing to ``data/ray-v5-merged``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.data_io import read_datafile, write_datafile


def compute_oversample_k(base_tokens: int, inject_tokens: int,
                         target_ratio: float) -> int:
    """Return the integer ``k`` such that ``(I*k) / (B + I*k) ~= target``.

    Derivation: (I*k)/(B + I*k) = r  =>  k = r*B / ((1-r)*I).
    For r=0.30 this is ``3B/(7I)``. We round *up* so the ratio is at
    least the target. If inject is already large enough (k <= 1) or
    inject is empty, return 1 (caller still emits one copy).
    """
    if inject_tokens <= 0:
        return 1
    num = target_ratio * base_tokens
    den = (1.0 - target_ratio) * inject_tokens
    if den <= 0:
        return 1
    k = math.ceil(num / den)
    return max(k, 1)


def _load_split(corpus_dir: Path, split: str):
    bin_path = corpus_dir / f"{split}.bin"
    mask_path = corpus_dir / f"{split}_mask.npy"
    assert bin_path.exists(), f"missing {bin_path}"
    assert mask_path.exists(), f"missing {mask_path}"
    tokens = read_datafile(str(bin_path))
    mask = np.load(mask_path)
    assert len(tokens) == len(mask), (
        f"{corpus_dir.name}/{split}: tokens={len(tokens)} mask={len(mask)}"
    )
    assert mask.dtype == np.float32, (
        f"{corpus_dir.name}/{split}_mask.npy dtype is {mask.dtype}, expected float32"
    )
    return np.asarray(tokens), mask


def _merge_split(
    base_tokens: np.ndarray, base_mask: np.ndarray,
    inject_tokens: np.ndarray, inject_mask: np.ndarray,
    k: int, chunk_size: int, seed: int,
):
    """Concatenate ``base`` with ``k`` copies of ``inject`` then chunk-
    shuffle the stream deterministically. Mask and tokens are shuffled
    with identical indices so alignment is preserved.
    """
    parts_tokens = [base_tokens] + [inject_tokens] * k
    parts_mask = [base_mask] + [inject_mask] * k
    merged_tokens = np.concatenate(parts_tokens)
    merged_mask = np.concatenate(parts_mask).astype(np.float32, copy=False)
    assert len(merged_tokens) == len(merged_mask)

    n = len(merged_tokens)
    if chunk_size <= 0 or n <= chunk_size:
        return merged_tokens, merged_mask

    # Split into chunks (last chunk may be shorter) then permute chunk order
    n_full = n // chunk_size
    tail = n - n_full * chunk_size
    chunk_idx = np.arange(n_full)
    rng = np.random.default_rng(seed)
    rng.shuffle(chunk_idx)

    out_tokens = np.empty_like(merged_tokens)
    out_mask = np.empty_like(merged_mask)
    cursor = 0
    for new_pos, old_pos in enumerate(chunk_idx):
        src_start = old_pos * chunk_size
        src_end = src_start + chunk_size
        out_tokens[cursor:cursor + chunk_size] = merged_tokens[src_start:src_end]
        out_mask[cursor:cursor + chunk_size] = merged_mask[src_start:src_end]
        cursor += chunk_size
    # Tail chunk (if any) stays at the end, unshuffled
    if tail:
        out_tokens[cursor:] = merged_tokens[n_full * chunk_size:]
        out_mask[cursor:] = merged_mask[n_full * chunk_size:]
    return out_tokens, out_mask


def merge_corpora(
    base_dir: Path | str,
    inject_dir: Path | str,
    output_dir: Path | str,
    *,
    target_ratio: float = 0.30,
    chunk_size: int = 4096,
    seed: int = 42,
):
    base_dir = Path(base_dir)
    inject_dir = Path(inject_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_train, base_train_mask = _load_split(base_dir, "train")
    base_val, base_val_mask = _load_split(base_dir, "val")
    inject_train, inject_train_mask = _load_split(inject_dir, "train")
    inject_val, inject_val_mask = _load_split(inject_dir, "val")

    # A single k computed from train-split totals. Val ratio follows if the
    # val split size tracks train; if val is disproportionately small or large
    # (rare) the val inject ratio may drift a few percent from --target-inject-ratio.
    k = compute_oversample_k(len(base_train), len(inject_train), target_ratio)

    train_tokens, train_mask = _merge_split(
        base_train, base_train_mask, inject_train, inject_train_mask,
        k=k, chunk_size=chunk_size, seed=seed,
    )
    val_tokens, val_mask = _merge_split(
        base_val, base_val_mask, inject_val, inject_val_mask,
        k=k, chunk_size=chunk_size, seed=seed + 1,
    )

    # write_datafile truthy-checks `tokens` (ndarray raises ValueError), so .tolist().
    write_datafile(str(output_dir / "train.bin"), train_tokens.tolist())
    # write_datafile truthy-checks `tokens` (ndarray raises ValueError), so .tolist().
    write_datafile(str(output_dir / "val.bin"), val_tokens.tolist())
    np.save(output_dir / "train_mask.npy", train_mask.astype(np.float32))
    np.save(output_dir / "val_mask.npy", val_mask.astype(np.float32))

    train_inject = int(len(inject_train)) * k
    val_inject = int(len(inject_val)) * k
    train_total = int(len(base_train)) + train_inject
    val_total = int(len(base_val)) + val_inject

    manifest = {
        "base_dir": str(base_dir),
        "inject_dir": str(inject_dir),
        "target_inject_ratio": target_ratio,
        "k": int(k),
        "train_base_tokens": int(len(base_train)),
        "train_inject_tokens": train_inject,
        "train_inject_ratio": (train_inject / train_total) if train_total else 0.0,
        "train_total_tokens": train_total,
        "val_base_tokens": int(len(base_val)),
        "val_inject_tokens": val_inject,
        "val_inject_ratio": (val_inject / val_total) if val_total else 0.0,
        "val_total_tokens": val_total,
        "chunk_size": int(chunk_size),
        "seed": int(seed),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="base corpus dir (train/val .bin + _mask.npy)")
    parser.add_argument("--inject", required=True,
                        help="inject corpus dir (will be oversampled)")
    parser.add_argument("--output", required=True,
                        help="where to write merged train/val .bin + _mask.npy + manifest.json")
    parser.add_argument("--target-inject-ratio", type=float, default=0.30,
                        help="target fraction of total tokens from inject (default 0.30)")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="chunk granularity for deterministic shuffle (default 4096)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = merge_corpora(
        base_dir=args.base,
        inject_dir=args.inject,
        output_dir=args.output,
        target_ratio=args.target_inject_ratio,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    print(json.dumps(manifest, indent=2))
    print(f"\nk = {manifest['k']}")
    print(f"train inject ratio: {manifest['train_inject_ratio']:.3f}")
    print(f"val   inject ratio: {manifest['val_inject_ratio']:.3f}")


if __name__ == "__main__":
    main()
