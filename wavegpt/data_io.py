"""
Data I/O — binary format for token streams.

Compatible with Karpathy's llm.c GPT-2 .bin format:
  Header: 256 × int32 (1024 bytes)
    [0] = 20240520 (magic)
    [1] = 1 (version)
    [2] = num_tokens
  Data: num_tokens × uint16 (GPT-2 tokenizer token IDs)
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import tiktoken

# GPT-2 tokenizer
_enc: Optional[tiktoken.Encoding] = None
GPT2_EOT = 50256  # <|endoftext|> token ID


def _get_encoder() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


def write_datafile(filename: str, tokens: list[int]) -> None:
    """Write tokens in llm.c GPT-2 .bin format."""
    assert len(tokens) < 2**31, f"Token count too large: {len(tokens)}"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(tokens)
    toks_np = np.array(tokens, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def read_datafile(filename: str) -> np.ndarray:
    """Read tokens from llm.c GPT-2 .bin format."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, f"Bad magic: {header[0]}"
        assert header[1] == 1, f"Unsupported version: {header[1]}"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, f"Expected {ntok} tokens, got {len(tokens)}"
    return tokens


def tokenize_text(text: str, add_eot: bool = False) -> list[int]:
    """Tokenize text with GPT-2 tokenizer. Optionally prepend EOT."""
    enc = _get_encoder()
    tokens = enc.encode_ordinary(text)
    if add_eot:
        tokens = [GPT2_EOT] + tokens
    return tokens
