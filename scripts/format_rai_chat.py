"""
Reformat Ray's raw text corpus into Gemma IT chat format.

Splits text into paragraphs, wraps each as a model response to
topical prompts. Generates training data that aligns with the
instruction-tuning format the IT model expects.

Usage:
    python3 scripts/format_rai_chat.py \
        --input data/rai-gemma4/train.bin \
        --output data/rai-gemma4-chat/train.bin \
        --tokenizer /path/to/gemma-4-31B-it
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.data_io import read_datafile, write_datafile


# Prompts that elicit Ray-style responses
PROMPTS = [
    "Share your perspective on this topic.",
    "Tell me more about this.",
    "What are your thoughts on this?",
    "Explain this concept.",
    "How do you see this?",
    "Continue your explanation.",
    "What's important to understand here?",
    "Elaborate on this idea.",
    "What does this mean for the future?",
    "Help me understand this.",
    "What's your take on this?",
    "Walk me through this.",
    "Why does this matter?",
    "How would you describe this?",
    "What should people know about this?",
]


def split_into_chunks(text, min_len=200, max_len=1500):
    """Split text into paragraph-based chunks of reasonable size."""
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) < max_len:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if len(current) >= min_len:
                chunks.append(current)
            current = para

    if current and len(current) >= min_len:
        chunks.append(current)

    return chunks


def format_chat(chunk, prompt_idx):
    """Format a text chunk as a Gemma IT chat turn."""
    prompt = PROMPTS[prompt_idx % len(PROMPTS)]
    # Gemma IT format: <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n{response}<turn|>
    return f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n{chunk}<turn|>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-input", default=None)
    parser.add_argument("--val-output", default=None)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--min-chunk", type=int, default=200,
                        help="Minimum characters per chunk")
    parser.add_argument("--max-chunk", type=int, default=1500,
                        help="Maximum characters per chunk")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for split, inp, outp in [
        ("train", args.input, args.output),
        ("val", args.val_input, args.val_output),
    ]:
        if inp is None or outp is None:
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split}: {inp}")
        print(f"{'='*60}")

        # Read and decode original tokens
        raw_tokens = read_datafile(inp)
        print(f"  Original: {len(raw_tokens):,} tokens")

        text = tok.decode(raw_tokens.tolist(), skip_special_tokens=True)
        print(f"  Decoded: {len(text):,} characters")

        # Split into chunks
        chunks = split_into_chunks(text, args.min_chunk, args.max_chunk)
        print(f"  Chunks: {len(chunks)} (avg {sum(len(c) for c in chunks)//max(len(chunks),1)} chars)")

        # Format as chat and tokenize
        all_tokens = []
        for i, chunk in enumerate(chunks):
            chat_text = format_chat(chunk, i)
            tokens = tok.encode(chat_text, add_special_tokens=True)
            all_tokens.extend(tokens)

        print(f"  Chat tokens: {len(all_tokens):,} ({len(all_tokens)/len(raw_tokens):.2f}x original)")

        # Save
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        write_datafile(outp, all_tokens)
        print(f"  Saved: {outp}")

        # Show sample
        sample = tok.decode(all_tokens[:300], skip_special_tokens=False)
        print(f"\n  Sample:\n  {sample[:500]}")


if __name__ == "__main__":
    main()
