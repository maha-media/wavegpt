"""Deep analysis of Step-3.5-Flash-SFT — harmonic layer classification."""
import json
import datasets
import tiktoken
from collections import Counter

print("Loading dataset (streaming)...")
ds = datasets.load_dataset("stepfun-ai/Step-3.5-Flash-SFT", split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")

samples = []
for i, ex in enumerate(ds):
    if i >= 500:
        break
    samples.append(ex)

print(f"Sampled {len(samples)} examples\n")

# Classify each conversation by harmonic layer
# C: Simple factual (1-2 turns, no reasoning, short answers)
# G: Explanatory (reasoning present, moderate length)
# D: Multi-step/tool-use (tool turns, agent patterns, multi-hop)
# A: Nuanced (long reasoning, comparisons, edge cases)

layer_counts = Counter()
layer_tokens = Counter()
has_tool = 0
has_system = 0
has_reasoning = 0
turn_dist = Counter()
token_counts = []

for ex in samples:
    convos = ex.get("conversations", [])
    turns = []
    for turn in convos:
        t = json.loads(turn) if isinstance(turn, str) else turn
        turns.append(t)

    n_turns = len(turns)
    turn_dist[min(n_turns, 20)] += 1

    roles = set(t.get("role", "") for t in turns)
    any_reasoning = any(t.get("reasoning_content") for t in turns)
    any_tool = "tool" in roles
    any_system = "system" in roles

    if any_tool:
        has_tool += 1
    if any_system:
        has_system += 1
    if any_reasoning:
        has_reasoning += 1

    # Total tokens in this conversation
    total_chars = sum(len(t.get("content", "")) for t in turns)
    total_reasoning = sum(len(t.get("reasoning_content", "") or "") for t in turns)
    total_tok = (total_chars + total_reasoning) // 4  # rough
    token_counts.append(total_tok)

    # Classify
    if any_tool and n_turns > 4:
        layer = "D"  # Connection — multi-step tool use, agent behavior
    elif any_reasoning and total_reasoning > 5000:
        layer = "A"  # Nuance — deep reasoning required
    elif any_reasoning and n_turns <= 4:
        layer = "G"  # Function — explanation with reasoning
    elif n_turns <= 2 and total_chars < 2000:
        layer = "C"  # Fundamental — simple Q&A
    elif n_turns <= 2:
        layer = "G"  # Longer single-turn = explanation
    else:
        layer = "D"  # Multi-turn = connection

    layer_counts[layer] += 1
    layer_tokens[layer] += total_tok

print("=== Harmonic Layer Distribution (500 samples) ===")
for layer in ["C", "G", "D", "A"]:
    n = layer_counts[layer]
    tok = layer_tokens[layer]
    labels = {"C": "Fundamental (simple Q&A)",
              "G": "Function (explanation)",
              "D": "Connection (multi-step/tool)",
              "A": "Nuance (deep reasoning)"}
    print(f"  {layer}: {n:>4d} ({100*n/500:.0f}%) | ~{tok:>8,} tokens | {labels[layer]}")

print(f"\n  Has tool use: {has_tool} ({100*has_tool/500:.0f}%)")
print(f"  Has system prompt: {has_system} ({100*has_system/500:.0f}%)")
print(f"  Has reasoning: {has_reasoning} ({100*has_reasoning/500:.0f}%)")

print(f"\n=== Token Distribution ===")
token_counts.sort()
print(f"  Min: {min(token_counts):,}")
print(f"  Median: {token_counts[len(token_counts)//2]:,}")
print(f"  Mean: {sum(token_counts)//len(token_counts):,}")
print(f"  P95: {token_counts[int(len(token_counts)*0.95)]:,}")
print(f"  Max: {max(token_counts):,}")
print(f"  Total (500 samples): {sum(token_counts):,} tokens")

print(f"\n=== Turn Distribution ===")
for n in sorted(turn_dist.keys()):
    label = f"{n}+" if n == 20 else str(n)
    print(f"  {label:>3s} turns: {turn_dist[n]:>4d}")

# Extrapolate
total_tok_est = sum(token_counts) / 500 * 1_620_000
print(f"\n=== Extrapolated Full Dataset ===")
print(f"  ~{total_tok_est/1e9:.1f}B tokens (content + reasoning)")
print(f"  ~{total_tok_est/1e9 * 0.6:.1f}B tokens (content only, ~60%)")

# Subset planning
for subset_k in [10, 50, 100, 500]:
    subset_tok = sum(token_counts) / 500 * subset_k * 1000
    print(f"  {subset_k}K examples: ~{subset_tok/1e6:.0f}M tokens")
