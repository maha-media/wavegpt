# WaveGPT × Step-3.5-Flash-SFT — Experiment Plan

## Goal
Apply harmonic training to general-domain SFT data. Prove that WaveGPT's
data curriculum works beyond single-author corpora.

## Dataset
- Source: `stepfun-ai/Step-3.5-Flash-SFT` (1.62M examples, ~11B tokens)
- Subset: 50K examples (~340M tokens) for proof of concept
- Format: Multi-turn conversations with reasoning traces
- 99% have `reasoning_content`, 14% have tool use, 11% have system prompts

## Harmonic Layer Classification

The conversations naturally stratify:
- **C (1%)**: Simple factual Q&A — barely present (this is advanced SFT)
- **G (42%)**: Single-turn explanations with reasoning
- **D (17%)**: Multi-step tool use, agent workflows, multi-hop
- **A (40%)**: Deep reasoning (>5K chars thinking), nuanced analysis

Token distribution is even more skewed:
- G: 13% of tokens (short explanations)
- D: 29% of tokens (multi-turn = more text per example)
- A: 58% of tokens (deep reasoning = longest)

## WaveGPT Changes Needed

### 1. SFT DataLoader (`wavegpt/sft_dataloader.py`)
- Tokenize conversations with simple role markers: `<|user|>`, `<|assistant|>`, `<|system|>`, `<|tool|>`
- Create loss masks: 0 for user/system/tool tokens, 1 for assistant tokens
- Include `reasoning_content` as part of assistant response (or separate marker)
- Pack conversations into fixed-length sequences
- Support harmonic curriculum (order by C→G→D→A classification)

### 2. SFT Export (`scripts/export_sft.py`)
- Stream from HuggingFace dataset
- Classify each conversation into harmonic layer
- Write layer-ordered .bin files with loss mask companion files
- Support subset selection (--max-examples)

### 3. Masked Loss in Model
- `WaveGPT.forward()` accepts optional `loss_mask` tensor
- Cross-entropy computed only on masked positions
- Anti-collapse penalty still computed on ALL positions (we want diverse
  representations everywhere, not just where loss is applied)

### 4. Reasoning Content Handling
Two options:
  a) Include reasoning as part of assistant response (simple, more tokens)
  b) Separate `<|thinking|>...<|/thinking|>` markers (structured)

Start with (a) — it's what StepFun's own training does.

## Model Sizing

| Model | Params | Tokens | Tok/Param | Notes |
|-------|--------|--------|-----------|-------|
| small | 16M | 340M | 21 | Reasonable — close to Chinchilla |
| medium | 30M | 340M | 11 | Slightly over-parameterized |

Start with `small` (16M) on 50K examples. Chinchilla-optimal for ~340M tokens
would be ~17M params — our small model is perfect.

## Experiment Matrix

| Run | Config | Hypothesis |
|-----|--------|-----------|
| S-A | Random order, no curriculum | Baseline |
| S-B | Harmonic order (C→G→D→A) | Does ordering matter for SFT? |
| S-C | Harmonic + data curriculum (3-phase) | Structured → mixed → full |
| S-D | S-C + anti-collapse (α=0.05) | Does anti-collapse help in SFT? |

## Success Criteria
- S-D beats S-A on val loss (shows harmonic training helps for SFT)
- Qualitative: generated responses show better reasoning structure
- Oscillation metrics: harmonic ordering reduces val loss variance

## Infrastructure
- RunPod A40 for training
- 340M tokens ÷ 72K tok/s = ~78 min per 15K steps
- 4 experiments × ~80 min = ~5.3 hours total
