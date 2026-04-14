"""
Re-tokenize corpus to Gemma 4 tokens.

Two modes:

1) Qwen→Gemma (existing):
   Decodes existing Qwen-tokenized .bin → raw text → re-encodes with Gemma tokenizer.

   Usage:
       python retokenize_for_gemma.py \
           --source-dir data/rai-qwen \
           --source-tokenizer Qwen/Qwen3.5-27B \
           --target-tokenizer google/gemma-4-31B \
           --output-dir data/rai-gemma4

2) Chat JSONL → Gemma + assistant-only mask (Task 16):
   Consumes a chat JSONL (one `{"messages": [...]}` per line), renders each
   conversation with the target tokenizer's chat template, and emits
   `{split}.bin` plus `{split}_mask.bin` where the mask is 1 on assistant
   tokens and 0 everywhere else (system/user/chat scaffolding).

   Usage:
       python retokenize_for_gemma.py \
           --chat-jsonl data/ray.jsonl \
           --emit-mask \
           --target-tokenizer google/gemma-4-31B \
           --output-dir data/rai-gemma4-chat
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.data_io import read_datafile, write_datafile


def _load_tokenizer(name_or_path: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)


def render_messages_to_tokens_and_mask(tokenizer, messages):
    """Render one chat conversation to (tokens, mask).

    Mask is a list[int] (0/1) the same length as tokens. Assistant turns map
    to 1, everything else (system/user/chat-template scaffolding, including
    the assistant-close marker) to 0.

    Strategy: render the full sequence with the chat template, then for each
    assistant message, encode its content text alone and locate it as a
    contiguous subsequence inside the section added between
    `apply_chat_template(messages[:i])` and `apply_chat_template(messages[:i+1])`.
    Only the content span is marked 1; template scaffolding around it is 0.
    """
    full = list(tokenizer.apply_chat_template(messages, tokenize=True))
    mask = [0] * len(full)

    prev_render = []
    for i, msg in enumerate(messages):
        cur_render = list(tokenizer.apply_chat_template(messages[: i + 1], tokenize=True))
        # Common prefix with previous render is untouched template content.
        j = 0
        while j < len(prev_render) and j < len(cur_render) and prev_render[j] == cur_render[j]:
            j += 1
        span_start = j
        span_end = len(cur_render)
        prev_render = cur_render

        if msg["role"] != "assistant":
            continue

        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        if not content_ids:
            continue

        # Find the content subsequence inside [span_start, span_end) of `full`.
        # Fall back to masking the whole span if a clean subsequence match isn't
        # found (keeps behavior sane for exotic tokenizers that fold whitespace).
        found = -1
        for k in range(span_start, span_end - len(content_ids) + 1):
            if full[k:k + len(content_ids)] == content_ids:
                found = k
                break

        if found >= 0:
            for k in range(found, found + len(content_ids)):
                mask[k] = 1
        else:
            for k in range(span_start, span_end):
                mask[k] = 1

    return full, mask


def _retokenize_qwen_to_gemma(args):
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading source tokenizer: {args.source_tokenizer}")
    src_tok = _load_tokenizer(args.source_tokenizer)

    target_path = args.target_tokenizer_local or args.target_tokenizer
    print(f"Loading target tokenizer: {target_path}")
    tgt_tok = _load_tokenizer(target_path)

    print(f"Source vocab: {src_tok.vocab_size}, Target vocab: {tgt_tok.vocab_size}")

    for split in ["train", "val"]:
        src_path = source_dir / f"{split}.bin"
        if not src_path.exists():
            print(f"  Skipping {split} (not found)")
            continue

        print(f"\n  Processing {split}...")
        tokens = read_datafile(str(src_path))
        print(f"    Source: {len(tokens):,} tokens")

        chunk_size = 100_000
        all_target_tokens = []

        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            text = src_tok.decode(chunk.tolist(), skip_special_tokens=True)
            new_tokens = tgt_tok.encode(text, add_special_tokens=False)
            all_target_tokens.extend(new_tokens)
            if (i // chunk_size) % 10 == 0:
                print(f"    Chunk {i // chunk_size}: {len(all_target_tokens):,} target tokens so far")

        out_path = output_dir / f"{split}.bin"
        write_datafile(str(out_path), all_target_tokens)
        print(f"    Target: {len(all_target_tokens):,} tokens → {out_path}")

    meta = {
        "source_tokenizer": args.source_tokenizer,
        "target_tokenizer": args.target_tokenizer,
        "source_dir": str(args.source_dir),
    }
    for split in ["train", "val"]:
        out_path = output_dir / f"{split}.bin"
        if out_path.exists():
            tokens = read_datafile(str(out_path))
            meta[f"{split}_tokens"] = len(tokens)

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved meta.json")


def _retokenize_chat_jsonl(args):
    if not args.emit_mask:
        raise SystemExit("--emit-mask is required when using --chat-jsonl")

    jsonl_path = Path(args.chat_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_path = args.target_tokenizer_local or args.target_tokenizer
    print(f"Loading target tokenizer: {target_path}")
    tgt_tok = _load_tokenizer(target_path)

    conversations = []
    with jsonl_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"line {line_no}: missing 'messages' key")
            conversations.append(obj["messages"])

    n = len(conversations)
    if n == 0:
        raise SystemExit(f"No conversations found in {jsonl_path}")

    # Deterministic 95/5 split by line order.
    n_val = max(1, n // 20) if n >= 2 else 0
    n_train = n - n_val
    train_msgs = conversations[:n_train]
    val_msgs = conversations[n_train:]
    print(f"  Conversations: {n} (train={n_train}, val={n_val})")

    meta = {
        "target_tokenizer": args.target_tokenizer,
        "chat_jsonl": str(jsonl_path),
        "n_conversations": n,
    }

    for split, msgs in [("train", train_msgs), ("val", val_msgs)]:
        if not msgs:
            continue
        all_tokens = []
        all_mask = []
        for conv in msgs:
            tokens, mask = render_messages_to_tokens_and_mask(tgt_tok, conv)
            all_tokens.extend(tokens)
            all_mask.extend(mask)

        tok_path = output_dir / f"{split}.bin"
        mask_path = output_dir / f"{split}_mask.bin"
        write_datafile(str(tok_path), all_tokens)
        write_datafile(str(mask_path), all_mask)
        pct = 100.0 * sum(all_mask) / max(len(all_mask), 1)
        print(f"  {split}: {len(all_tokens):,} tokens, {pct:.1f}% assistant → {tok_path}, {mask_path}")
        meta[f"{split}_tokens"] = len(all_tokens)
        meta[f"{split}_mask_coverage"] = pct / 100.0

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved meta.json")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--source-dir", default=None,
                      help="Qwen-tokenized source dir (Qwen→Gemma mode)")
    mode.add_argument("--chat-jsonl", default=None,
                      help="Chat JSONL file (emits {split}.bin + {split}_mask.bin)")

    parser.add_argument("--source-tokenizer", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--target-tokenizer", default="google/gemma-4-31B")
    parser.add_argument("--target-tokenizer-local", default=None,
                        help="Local path to tokenizer files (skip download)")
    parser.add_argument("--output-dir", default="data/rai-gemma4")
    parser.add_argument("--emit-mask", action="store_true",
                        help="Required in --chat-jsonl mode; emits {split}_mask.bin")
    args = parser.parse_args()

    if args.chat_jsonl is not None:
        _retokenize_chat_jsonl(args)
    elif args.source_dir is not None:
        _retokenize_qwen_to_gemma(args)
    else:
        # Default to the legacy Qwen→Gemma path on the default --source-dir.
        args.source_dir = "data/rai-qwen"
        _retokenize_qwen_to_gemma(args)

    print("Done.")


if __name__ == "__main__":
    main()
