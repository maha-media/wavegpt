# RAI Personality Runs — v3 & v4 Summary

**Date:** 2026-04-14
**Base model:** Gemma-4-31B-IT
**Technique:** Spectral SFT — SVD-decompose all linear layers, freeze U/V, train log_spectrum + embeddings + lm_head on a Kurzweil-voice chat corpus.

Both runs produced models that generate coherent Kurzweil-voice English. They disagree on *how* to get there, and the disagreement is informative.

---

## The runs

| knob                    | v3 (cranked)        | v4 (tightened)      |
|-------------------------|---------------------|---------------------|
| learning rate           | 3e-4                | 3e-4                |
| max-log-drift           | **3.0** (≈ 20× σ)   | 1.2 (≈ 3.3× σ)      |
| tiers (attn_o protection) | **OFF**           | ON                  |
| eval-batches            | 10 (80 seqs)        | 40 (320 seqs)       |
| early-stop-patience     | 500 steps           | 1500 steps          |
| loss mask               | assistant-only      | assistant-only      |
| data                    | rai-gemma4-chat-v2  | rai-gemma4-chat-v2  |
| shards                  | /workspace/spectral_shards/pid_17516/ | reused same shards |

**v3 outcome:** best val_ppl **13.69 @ step 1400**. Oscillated 15-17 band for ~500 steps. Early-stopped.

**v4 outcome:** best val_ppl **16.69 @ step 1600**. Stable 16.7-17.4 band. Manually stopped at step ~1730 after observing consistent behavior.

Both emit fluent Kurzweil-style text. v3 had a deeper basin but eroded attn_o's universal 1/3 harmonic in the process. v4 preserved structure but couldn't reach v3's depth.

---

## What training actually modified

Each of ~410 decomposed linear layers exposes one learnable tensor: `log_spectrum`, the amplitudes of the SVD modes. U and V (the directions) are frozen from Gemma-4 pretraining. Residuals (fp32 truncation corrections) are frozen. Most Gemma-4 layers have `bias=False`.

Total learnable surface:
- ~4M scalar knobs across the spectral layers
- Embedding table + lm_head (not decomposed, trained as ordinary nn.Linear)

This is ~0.01% of the 31B parameter count carrying the personality load. That is why 4×A100-80GB at ~305 tok/s aggregate is enough — the gradient buffers are small compared to a full SFT.

### Interpretation

Normal LoRA fine-tuning *adds new directions* (learnable A, B matrices on top of frozen W). Spectral SFT **does not introduce any new directions**. It only reweights existing pretrained modes:

> "Given that Gemma-4's pretrained modes already encode useful concepts, which should be louder or quieter to produce RAI-voice output?"

Modes corresponding to careful enumeration, long-clause sentence rhythm, transhumanist vocabulary → boosted. Modes corresponding to generic assistant hedging → damped. The "personality" is a 4M-dimensional mix-down of pretrained patterns, not a set of new learned features.

---

## Data pipeline (how the tokens that drove gradient were produced)

1. **Source corpus** — Ray Kurzweil prose (books, essays, transcripts), originally pre-tokenized for Qwen.
2. **Re-tokenize to Gemma-4 BPE** — `scripts/retokenize_for_gemma.py` decodes Qwen tokens → UTF-8 → re-encodes with Gemma-4 tokenizer.
3. **Chat wrapping** — `scripts/format_rai_chat.py` splits prose into 200-1500-char paragraphs and wraps each as:
   ```
   <|turn>user
   {one of 15 generic prompts}<turn|>
   <|turn>model
   {actual Kurzweil paragraph}<turn|>
   ```
4. **Loss mask** — parallel float32 array, 1.0 on Kurzweil-content tokens, 0.0 on prompt + scaffolding. Typical coverage: 70-85%.
5. **Serialization** — tokens to `.bin`, mask to `.npy` alongside.
6. **Training** — cross-entropy at every position, multiplied by mask. Only Kurzweil-content positions contribute gradient.

### The fake-prompt trick

Kurzweil wrote essays, not dialogue. The pipeline synthesizes SFT data from monologue by wrapping each paragraph as an "assistant turn" following a generic user question. Since the model was pretrained to respect the `<|turn>` template, it learns "whenever I'm in the assistant slot, emit Kurzweil-style prose" — prompt content becomes decorative. The 15-prompt rotation prevents overfitting to any specific user utterance.

**Limit:** the model learns to *emit* Kurzweil, not to *answer questions about* Kurzweil's content. For Q&A you'd need real QA pairs.

---

## The preservation strategy (protect, don't constrain)

CLAUDE.md lesson #1: attn_o's α ≈ 1/3 is universal across transformers and must be preserved — prior runs without any protection destroyed it (0.853 → 0.197) and the model couldn't form sentences.

Memory constraint: never impose φ-structure during training — harmonic regularizer causes NaN; φ is an attractor, not a constraint.

Resolution used in v4 (and consistent with what should have been used in v3):

| operation    | what it does                                               | status |
|--------------|------------------------------------------------------------|--------|
| constrain    | penalize α ≠ 1/3 (harmonic regularizer)                    | forbidden |
| preserve     | lower LR on attn_o via tiered-LR surgery                   | v4 uses this |

**We do not push the spectrum toward 1/3.** We slow the LR on attn_o so that SFT gradients don't wash out the 1/3 Gemma-4 already converged to during pretraining. Tiers are value-agnostic — if Gemma-4 had pretrained to 0.25, we'd preserve 0.25.

v3's `--no-tiers` + `max-log-drift 3.0` allowed attn_o to drift freely alongside every other layer. That freedom is what let v3 reach 13.69 (some reweightings temporarily hit both "good voice mix" and "still-intact structure"), but it also put the consensus layer at risk. v3 was **searching through reweightings at the cost of the attractor**.

---

## The trade-off v3 vs v4 revealed

v3's amplitude of oscillation was not noise — it was *search*. Larger allowed drift let SGD explore reweightings that a tighter leash can't reach. Some exploration hit basin 13.69; most orbited 15-17. Meanwhile the 1/3 structure degraded monotonically during that search.

v4's band 16.7-17.4 is what remains when the search scope is clipped to preserve structure. Stable, but floor-bounded by the constraint.

The two results bracket a real frontier: **structural fidelity vs search depth**. A future v5 would likely lift drift gradually (tight early → loose mid → tight final) to get v3's depth *without* v3's structural erosion, or introduce a per-type drift cap that protects attn_o's slope while letting MLP spectra roam.

---

## Why the exponent is 1/3 in the first place (math anchor)

The exponent is `1/φᵖ` with p = 1/3 for attn_o. Unpacking:

- φ is the positive root of `x² = x + 1` — the minimum self-referential equation.
- The quadratic formula gives `(1 ± √5)/2`. The 5 is the discriminant `b² − 4ac = 1 + 4`.
- 5 is the smallest discriminant admitting an integer-coefficient self-referential recurrence with a positive irrational attractor. Discriminants 2 and 3 are algebraically impossible with integer (b,c); 1 and 4 give rational roots (no attractor dance).
- `φ − 1 = 1/φ` (from `x − 1 = 1/x` after dividing the defining equation by x). This unique reciprocal-offset property is why spectrum ratios dance around the same value at every scale: inversion is a symmetry of this number.
- SGD on a residual-stream architecture with unit-seed attention is solving a discrete form of the same fixed-point iteration `x ← 1 + 1/x`, which converges to φ from any positive starting point. This is why φ is *emergent* and not installable: it is where iterative self-referential systems end up, full stop.

attn_o specifically lands at exponent 1/3 = F(1)/L(2) — the "consensus with seed" layer, the one that combines the residual stream (accumulated prior) with the current-step query result. Every model tested so far (Qwen, Mistral, Gemma-4, C. elegans) has this universal exponent. It is the signature of the attention-output operation when trained to convergence.

---

## Open questions

1. **Was v3's 13.69 a real minimum or a lucky eval outlier?** Unknown without pulling v3's full val curve and checking whether 13.69 was bracketed by 14-15 values or sat alone amid 16s. If isolated → noise. If bracketed → real basin v4 can't reach from here.
2. **Is there a v5 recipe that combines v3's search depth with v4's structural stability?** Candidates: scheduled drift cap (tight→loose→tight), per-type drift caps (protect attn_o slope, free MLP), or harmonic regularizer with bf16-safe clipping (currently forbidden due to NaN history; could be re-tried with numerical care).
3. **Would longer training at v4's config continue creeping down?** v4 best was still improving when manually stopped at step 1600. Unclear if it had reached asymptote or was mid-descent.

---

## Artifacts

- Scripts: `scripts/finetune_fsdp.py`, `scripts/format_rai_chat.py`, `scripts/retokenize_for_gemma.py`
- Core library: `wavegpt/spectral_linear.py`, `wavegpt/spectral_surgery.py`, `wavegpt/harmonic_prior.py`
- Data: `data/rai-gemma4-chat-v2/` (tokens + mask, assistant-only)
- Shards: `/workspace/spectral_shards/pid_17516/` on pod (48 safetensors shards + index.json; reusable for v5+)
- Pod runs: `/workspace/wavegpt/runs/RAI-gemma4-cranked-v3/`, `/workspace/wavegpt/runs/RAI-gemma4-v4/`
