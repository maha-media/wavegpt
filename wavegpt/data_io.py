"""
Data I/O -- binary format for token streams.

Version 1 (llm.c GPT-2 compatible):
  Header: 256 x int32 (1024 bytes)
    [0] = 20240520 (magic)
    [1] = 1 (version)
    [2] = num_tokens
  Data: num_tokens x uint16 (GPT-2 tokenizer, vocab <= 65535)

Version 2 (extended vocab):
  Header: 256 x int32 (1024 bytes)
    [0] = 20240520 (magic)
    [1] = 2 (version)
    [2] = num_tokens
  Data: num_tokens x uint32 (any tokenizer, vocab <= 4B)
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


def write_datafile(filename: str, tokens: list[int], version: int | None = None) -> None:
    """Write tokens in binary format. Auto-selects version based on max token ID."""
    assert len(tokens) < 2**31, f"Token count too large: {len(tokens)}"
    max_id = max(tokens) if tokens else 0
    if version is None:
        version = 1 if max_id <= 65535 else 2
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = version
    header[2] = len(tokens)
    dtype = np.uint16 if version == 1 else np.uint32
    toks_np = np.array(tokens, dtype=dtype)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def read_datafile(filename: str) -> np.ndarray:
    """Read tokens from binary format (v1 uint16 or v2 uint32)."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, f"Bad magic: {header[0]}"
        version = header[1]
        assert version in (1, 2), f"Unsupported version: {version}"
        ntok = header[2]
        dtype = np.uint16 if version == 1 else np.uint32
        tokens = np.frombuffer(f.read(), dtype=dtype)
    assert len(tokens) == ntok, f"Expected {ntok} tokens, got {len(tokens)}"
    return tokens


def tokenize_text(text: str, add_eot: bool = False) -> list[int]:
    """Tokenize text with GPT-2 tokenizer. Optionally prepend EOT."""
    enc = _get_encoder()
    tokens = enc.encode_ordinary(text)
    if add_eot:
        tokens = [GPT2_EOT] + tokens
    return tokens
