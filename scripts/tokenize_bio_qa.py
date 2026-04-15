"""Tokenize + chat-wrap the bio-QA pair corpus into train/val bins.

Reads every ``*_pairs.jsonl`` under ``--input-dir``, formats each pair
as a single-turn Gemma-style chat exchange::

    <|turn>user\n{question}<turn|>\n<|turn>model\n{answer}<turn|>

and emits:

  * ``train.bin`` / ``val.bin``      — token streams (wavegpt binary format)
  * ``train_mask.npy`` / ``val_mask.npy``  — float32 loss masks (1.0 on
    answer content tokens, 0.0 on the user-prefix + ``<turn|>`` end marker)
  * ``manifest.json``                — pair + token counts and mask coverage

The mask layout mirrors ``scripts/format_rai_chat.py::format_chat_with_mask``
(answer-only loss) so the output drops straight into ``scripts/finetune_fsdp.py``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.data_io import write_datafile


def format_pair_with_mask(question: str, answer: str, tokenizer):
    """Tokenize a single (Q, A) pair and return (tokens, mask).

    Layout: ``<|turn>user\n{Q}<turn|>\n<|turn>model\n`` (mask=0.0)
         ++ ``{A}``                                 (mask=1.0)
         ++ ``<turn|>``                             (mask=0.0)

    BOS is attached to the very first chunk (prefix) so each pair is
    tokenized as a self-contained example. Concatenating across pairs
    gives a standard packed corpus.
    """
    user_prefix = f"<|turn>user\n{question}<turn|>\n<|turn>model\n"
    model_suffix = "<turn|>"

    prefix_tokens = tokenizer.encode(user_prefix, add_special_tokens=True)
    content_tokens = tokenizer.encode(answer, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(model_suffix, add_special_tokens=False)

    tokens = prefix_tokens + content_tokens + suffix_tokens
    mask = (
        [0.0] * len(prefix_tokens)
        + [1.0] * len(content_tokens)
        + [0.0] * len(suffix_tokens)
    )
    return tokens, mask


def _load_pairs_from_dir(input_dir: Path):
    """Load all ``*_pairs.jsonl`` files under ``input_dir`` in sorted order."""
    source_files = []
    pairs = []
    for path in sorted(input_dir.glob("*_pairs.jsonl")):
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pairs.append(json.loads(line))
                count += 1
        source_files.append({"filename": path.name, "pair_count": count})
    return pairs, source_files


def _emit_split(
    split_pairs: Iterable[dict],
    tokenizer,
    output_dir: Path,
    split_name: str,
):
    all_tokens: list[int] = []
    all_mask: list[float] = []
    for pair in split_pairs:
        tokens, mask = format_pair_with_mask(
            question=pair["question"],
            answer=pair["answer"],
            tokenizer=tokenizer,
        )
        all_tokens.extend(tokens)
        all_mask.extend(mask)

    bin_path = output_dir / f"{split_name}.bin"
    mask_path = output_dir / f"{split_name}_mask.npy"
    write_datafile(str(bin_path), all_tokens)
    np.save(mask_path, np.array(all_mask, dtype=np.float32))
    return len(all_tokens), float(np.mean(all_mask)) if all_mask else 0.0


def tokenize_pairs(
    pairs,
    tokenizer,
    output_dir: Path | str,
    train_frac: float = 0.95,
    seed: int = 42,
    *,
    input_dir: Path | str | None = None,
    tokenizer_name: str = "",
):
    """Core pipeline: load (if needed), shuffle, split, tokenize, write.

    Either ``pairs`` (a list of dicts) or ``input_dir`` (a directory of
    ``*_pairs.jsonl`` files) must be supplied. When ``input_dir`` is
    given, ``pairs`` should be None and the files are loaded in sorted
    filename order so the manifest's ``source_files`` is deterministic.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pairs is None:
        assert input_dir is not None, "must supply pairs or input_dir"
        pairs, source_files = _load_pairs_from_dir(Path(input_dir))
    else:
        source_files = [{"filename": "<in-memory>", "pair_count": len(pairs)}]

    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(round(n_total * train_frac))
    # Keep at least one val pair when we have more than one pair
    if n_total > 1 and n_train >= n_total:
        n_train = n_total - 1
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    train_tokens, train_cov = _emit_split(train_pairs, tokenizer, output_dir, "train")
    val_tokens, val_cov = _emit_split(val_pairs, tokenizer, output_dir, "val")

    manifest = {
        "total_pairs": n_total,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_mask_coverage": train_cov,
        "val_mask_coverage": val_cov,
        "source_files": source_files,
        "tokenizer": tokenizer_name,
        "seed": seed,
        "train_frac": train_frac,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="directory containing *_pairs.jsonl files")
    parser.add_argument("--output-dir", required=True,
                        help="where to write train/val .bin + _mask.npy + manifest.json")
    parser.add_argument("--tokenizer", required=True,
                        help="HF tokenizer id or local path")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    manifest = tokenize_pairs(
        pairs=None,
        tokenizer=tok,
        output_dir=args.output_dir,
        train_frac=0.95,
        seed=42,
        input_dir=args.input_dir,
        tokenizer_name=args.tokenizer,
    )

    print(json.dumps(manifest, indent=2))
    cov = manifest["train_mask_coverage"] * 100
    print(f"\nTrain mask coverage: {cov:.1f}%")
    if cov < 50.0:
        print("WARNING: mask coverage <50% — prompt scaffolding dominates; investigate.")


if __name__ == "__main__":
    main()
