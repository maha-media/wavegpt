"""Analyze the Step-3.5-Flash-SFT dataset structure."""
import json
import datasets

print("Loading dataset (streaming)...")
ds = datasets.load_dataset("stepfun-ai/Step-3.5-Flash-SFT", split="train", streaming=True)

samples = []
for i, ex in enumerate(ds):
    if i >= 50:
        break
    samples.append(ex)

print(f"Sampled {len(samples)} examples\n")

# Show first 3 in detail
for i, ex in enumerate(samples[:3]):
    convos = ex.get("conversations", [])
    print(f"--- Example {i} ({len(convos)} turns) ---")
    for j, turn in enumerate(convos):
        t = json.loads(turn) if isinstance(turn, str) else turn
        role = t.get("role", "?")
        content = t.get("content", "")
        loss = t.get("loss_mask", "?")
        rc = t.get("reasoning_content", "") or ""
        clen = len(content)
        rlen = len(rc)
        preview = content[:200].replace("\n", " ")
        print(f"  [{j}] {role} | loss_mask={loss} | content={clen} chars | reasoning={rlen} chars")
        print(f"      {preview}")
    print()

# Aggregate stats
print("=== Aggregate Stats (50 samples) ===")
turn_counts = []
content_lens = []
reasoning_count = 0
total_turns = 0
roles = {}
domains = set()

for ex in samples:
    convos = ex.get("conversations", [])
    turn_counts.append(len(convos))
    for turn in convos:
        t = json.loads(turn) if isinstance(turn, str) else turn
        role = t.get("role", "?")
        roles[role] = roles.get(role, 0) + 1
        total_turns += 1
        content = t.get("content", "")
        content_lens.append(len(content))
        if t.get("reasoning_content"):
            reasoning_count += 1

print(f"Turns/example: min={min(turn_counts)} max={max(turn_counts)} avg={sum(turn_counts)/len(turn_counts):.1f}")
print(f"Content chars: min={min(content_lens)} max={max(content_lens)} avg={sum(content_lens)/len(content_lens):.0f}")
print(f"Roles: {roles}")
print(f"Reasoning turns: {reasoning_count}/{total_turns} ({100*reasoning_count/total_turns:.1f}%)")
print(f"Total chars in 50 samples: {sum(content_lens):,}")

# Estimate total dataset size
avg_chars = sum(content_lens) / len(content_lens)
avg_turns = sum(turn_counts) / len(turn_counts)
est_total_chars = avg_chars * avg_turns * 1_620_000
est_total_tokens = est_total_chars / 4  # rough char-to-token ratio
print(f"\nEstimated total dataset:")
print(f"  ~{est_total_chars/1e9:.1f}B chars")
print(f"  ~{est_total_tokens/1e9:.1f}B tokens (rough)")
