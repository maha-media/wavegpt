import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.format_rai_chat import format_chat_with_mask


class FakeTok:
    """Minimal tokenizer stub: every character is one token, specials are length-1 integers."""
    def __init__(self):
        self.next_id = 1000
        self.cache = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        if add_special_tokens:
            ids.append(1)  # bos
        for c in text:
            if c not in self.cache:
                self.cache[c] = self.next_id
                self.next_id += 1
            ids.append(self.cache[c])
        return ids


def test_mask_covers_only_chunk_tokens():
    tok = FakeTok()
    chunk = "hello world"
    tokens, mask = format_chat_with_mask(chunk, prompt_idx=0, tokenizer=tok)
    assert len(tokens) == len(mask)
    # Check that the 1.0 span is exactly len(chunk) tokens
    ones = sum(1 for m in mask if m == 1.0)
    assert ones == len(chunk), f"Expected {len(chunk)} 1.0s, got {ones}"
    # Check that first token is masked 0 (it's the BOS or user-prefix)
    assert mask[0] == 0.0
    # Check that last token is masked 0 (it's the <turn|> end marker)
    assert mask[-1] == 0.0
