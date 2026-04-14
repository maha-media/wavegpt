"""Task 16 — assistant-only loss mask for spectral SFT.

Verifies the retokenizer's chat-JSONL mode emits a parallel mask where
positions inside assistant turns are 1 and everything else (system/user
prompts, chat-template scaffolding) is 0.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavegpt.data_io import read_datafile
from scripts.retokenize_for_gemma import render_messages_to_tokens_and_mask


class FakeChatTokenizer:
    """Minimal tokenizer that supports apply_chat_template + encode.

    One token per character; role scaffolding is deterministic.
    """
    SYS_OPEN = "<|sys|>"
    SYS_CLOSE = "</|sys|>"
    USER_OPEN = "<|user|>"
    USER_CLOSE = "</|user|>"
    ASSISTANT_OPEN = "<|assistant|>"
    ASSISTANT_CLOSE = "</|assistant|>"

    def __init__(self):
        self.vocab = {}

    def _id(self, ch):
        if ch not in self.vocab:
            self.vocab[ch] = len(self.vocab) + 1
        return self.vocab[ch]

    def encode(self, text, add_special_tokens=False):
        return [self._id(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)

    def _render(self, messages):
        parts = []
        for m in messages:
            role = m["role"]
            if role == "system":
                parts.append(self.SYS_OPEN + m["content"] + self.SYS_CLOSE)
            elif role == "user":
                parts.append(self.USER_OPEN + m["content"] + self.USER_CLOSE)
            elif role == "assistant":
                parts.append(self.ASSISTANT_OPEN + m["content"] + self.ASSISTANT_CLOSE)
            else:
                raise ValueError(f"unknown role: {role}")
        return "".join(parts)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = self._render(messages)
        if not tokenize:
            return text
        return self.encode(text)


MESSAGES = [
    {"role": "system", "content": "You are Ray Kurzweil."},
    {"role": "user", "content": "What is the Singularity?"},
    {"role": "assistant", "content": "The Singularity is 2045."},
    {"role": "user", "content": "Expand on that."},
    {"role": "assistant", "content": "Exponential tech growth will converge."},
]


def test_pure_function_mask_shape_and_coverage():
    tok = FakeChatTokenizer()
    tokens, mask = render_messages_to_tokens_and_mask(tok, MESSAGES)

    assert len(tokens) == len(mask), "tokens and mask must have same length"
    assert sum(mask) > 0, "at least one assistant token must be masked-in"
    assert sum(mask) < len(mask), "not every token can be assistant (scaffold exists)"
    assert all(v in (0, 1) for v in mask), "mask values must be 0 or 1"


def test_pure_function_mask_aligns_with_assistant_spans():
    tok = FakeChatTokenizer()
    tokens, mask = render_messages_to_tokens_and_mask(tok, MESSAGES)

    # Build a ground-truth role label per character position from the rendered
    # text, then compare to the mask (1 token per char in our fake tokenizer).
    rendered = tok.apply_chat_template(MESSAGES, tokenize=False)
    roles = []
    i = 0
    current_role = None
    role_markers = [
        (tok.SYS_OPEN, "system", False),
        (tok.SYS_CLOSE, None, True),
        (tok.USER_OPEN, "user", False),
        (tok.USER_CLOSE, None, True),
        (tok.ASSISTANT_OPEN, "assistant", False),
        (tok.ASSISTANT_CLOSE, None, True),
    ]
    while i < len(rendered):
        matched = None
        for marker, next_role, is_close in role_markers:
            if rendered.startswith(marker, i):
                matched = (marker, next_role, is_close)
                break
        if matched is not None:
            marker, next_role, is_close = matched
            # Scaffold tokens are always 0 (non-assistant)
            for _ in range(len(marker)):
                roles.append("scaffold")
            i += len(marker)
            if is_close:
                current_role = None
            else:
                current_role = next_role
        else:
            roles.append(current_role if current_role else "scaffold")
            i += 1

    assert len(roles) == len(mask)
    for idx, (role, m) in enumerate(zip(roles, mask)):
        if role == "assistant":
            assert m == 1, f"pos {idx} role={role} expected mask=1 got {m}"
        else:
            assert m == 0, f"pos {idx} role={role} expected mask=0 got {m}"


def test_pure_function_decoded_assistant_text_present_in_masked_region():
    tok = FakeChatTokenizer()
    tokens, mask = render_messages_to_tokens_and_mask(tok, MESSAGES)

    masked_ids = [t for t, m in zip(tokens, mask) if m == 1]
    masked_text = tok.decode(masked_ids)
    assert "The Singularity is 2045." in masked_text
    assert "Exponential tech growth will converge." in masked_text
    # Non-assistant text must NOT appear in the masked region
    assert "You are Ray Kurzweil." not in masked_text
    assert "What is the Singularity?" not in masked_text


def test_retokenizer_chat_jsonl_cli_emits_mask(tmp_path):
    """End-to-end smoke test: invoke the retokenizer on a chat JSONL file and
    verify {split}.bin + {split}_mask.bin land in the output dir.

    Uses the fake tokenizer via a monkey-patched import; the subprocess path
    tests CLI wiring, the pure-function tests above cover mask correctness.
    """
    jsonl_path = tmp_path / "chat.jsonl"
    # 20 copies so the 95/5 split produces a non-empty val
    with jsonl_path.open("w") as f:
        for _ in range(20):
            f.write(json.dumps({"messages": MESSAGES}) + "\n")

    out_dir = tmp_path / "out"
    repo = Path(__file__).resolve().parents[1]

    # Stub tokenizer loader via a shim script that injects the fake tokenizer
    # before invoking main(). This avoids downloading any HF model in tests.
    shim = tmp_path / "shim.py"
    shim.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_assistant_mask import FakeChatTokenizer\n"
        "import scripts.retokenize_for_gemma as R\n"
        "R._load_tokenizer = lambda *a, **k: FakeChatTokenizer()\n"
        "R.main()\n"
    )

    r = subprocess.run(
        [
            sys.executable,
            str(shim),
            "--chat-jsonl", str(jsonl_path),
            "--emit-mask",
            "--target-tokenizer", "fake",
            "--output-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    for split in ("train", "val"):
        tok_path = out_dir / f"{split}.bin"
        mask_path = out_dir / f"{split}_mask.bin"
        assert tok_path.exists(), f"{split}.bin missing"
        assert mask_path.exists(), f"{split}_mask.bin missing"
        toks = read_datafile(str(tok_path))
        mask = read_datafile(str(mask_path))
        assert len(toks) == len(mask)
        assert mask.sum() > 0
        assert mask.sum() < len(mask)
        assert set(np.unique(mask).tolist()).issubset({0, 1})
