# Persona Pipeline: Gene + Activator Architecture

## Purpose

Extend the current RAI-gemma4 fine-tune pipeline to support subjects across a spectrum of base-model knowledge — from Ray-tier (model already knows them) to private individuals the model has never heard of. Without this, the pipeline only works when the base model has enough latent knowledge for the fine-tune to specialize; unknown subjects produce either a generic assistant or an overfit transcript-parrot.

## The observation that motivates this

The blind test (same 10 Kurzweil prompts, no system prompt, temp 0.3 / top_p 0.7) showed the step-1400 fine-tune and the untuned `google/gemma-4-31b-it` produce **essentially identical output** — same refusals on identity questions, same Taylor Swift pick ("All Too Well 10-minute"), same balanced consciousness framing, same "I am an LLM trained by Google" opener, zero Kurzweil-specific jargon. Val PPL gap is 55× (809 vs 14.5), but qualitative Ray-lean without the trigger is ~0.

Conclusion: **the fine-tune didn't bias the model toward Ray-coded language in general — it learned a conditional mapping** *"when user invokes Digital RAI → emit Ray-voice"*. Outside that condition the model is untouched. The system prompt is the activator; the fine-tune specializes latent knowledge that was already in the base model.

For an unknown subject there's no latent knowledge for the fine-tune to specialize, so the same pipeline fails silently. We need to plant the knowledge before specializing the voice.

## Architecture

```
Subject source material
  ├─ biographical corpus (books, interviews, articles, transcripts)
  └─ dialogue corpus (chat-formatted Q&A — may need synthesis for Tier C/D)
              │
              ▼
  Phase 0 — Gene Probe
    no-sysprompt factual probe on base model
    → tier classification (A/B/C/D) + gap_categories
              │
              ▼
  Phase 1 — Harmonically-Regularized CPT (the "gene")
    full continued pretraining on bio corpus
    loss = cross_entropy + λ · harmonic_prior(type_aware=True, attn_o_weight=w_ao)
              │
              ▼
  Phase 1.5 — Gene Verification Gate
    factual recall + φ-structure + general-knowledge retention
    GATE: fail → automated knob suggestion, halt
              │
              ▼
  Phase 2 — Spectral SFT on dialogue (the "activator loop")
    decompose phase1_verified.pt → train spectrum + harmonic regularizer
    sysprompt used during SFT = sysprompt used at inference
              │
              ▼
  Recompose → standard HF state_dict → deploy
    inference: system prompt = activator trigger
               + optional RAG for recent / private facts
```

### Key invariants

- **φ-structure is protected end-to-end.** Harmonic regularizer runs in both training phases. `attn_o` exponent at 1/3 is a hard invariant; everything else optimizes against it (CLAUDE.md lesson #1).
- **The activator must be the same string in training and inference.** That sysprompt is what the Phase 2 SFT teaches the model to condition on.
- **Dormancy is a feature.** Without the activator, the model reverts to default assistant behavior. One base can host multiple personas with no leak between them when untriggered.

## Phase 0 — Gene Probe & Tier Classification

Before spending any CPT compute, measure what the base model already knows about the subject.

**Input:** 30-50 factual probes hand-authored per subject. Categories:
- Biographical facts (birthplace, career milestones, collaborators)
- Signature beliefs / frameworks
- Idiomatic phrases the subject uses
- Recent events (post-cutoff items — these fail even for Tier A and route to RAG)

**Method:** each probe run with **no system prompt** at `temp=0.3, top_p=0.7`. Score each response 0/1/2 (wrong / partial / correct). Refusals score 0.

**Tier classification:**
- **Tier A**: ≥80% probes ≥1. Phase 1 light.
- **Tier B**: 40–80%. Phase 1 moderate.
- **Tier C**: 10–40%. Phase 1 heavy + likely synthetic dialogue augmentation.
- **Tier D**: <10%. Cold start. Phase 1 heavy + substantial bio corpus + RAG at inference required.

**Outputs:** `probe_baseline.md`, `tier.json` with `{tier, correct_rate, gap_categories}`. `gap_categories` drives oversampling in the Phase 1 corpus.

## Phase 1 — Harmonically-Regularized CPT

**Corpus preparation:**
- Ingest biographical source material: books, long-form interviews, essays, lecture transcripts, social media archives.
- Chunk into 2k-token windows with 128-token overlap.
- Oversample `gap_categories` from Phase 0.
- For **Tier C/D**, seed with a structured bio document (timeline, beliefs, key quotes) authored by the subject.

**Training objective:**
```
loss = cross_entropy + λ · harmonic_prior(type_aware=True, attn_o_weight=w_ao)
```
- `λ` starts at the value that worked for the current RAI run; tune upward if attn_o drift exceeds 5% at first checkpoint.
- `w_ao` (attn_o multiplier) cranked high.

**Schedule:**
- LR: 1/10 of original pretraining LR (conservative).
- Duration: 3–5 epochs, early-stop on val PPL plateau.
- Eval cadence: val PPL every N steps + spectral exponent check every checkpoint.

**Forgetting guard:** if val PPL on a general-knowledge held-out slice rises >10%, raise λ or shrink LR.

**Artifact:** `phase1_checkpoint.pt`.

## Phase 1.5 — Gene Verification Gate

Three gates, all must pass:

1. **Biographical recall.** Re-run Phase 0 probe on Phase 1 checkpoint, still no sysprompt. Pass: ≥70% probes ≥1 AND ≥2× improvement over baseline on `gap_categories`.
2. **φ-structure preserved.** Free-α analysis across all layer types. Pass: `attn_o` within 5% of base (~1/3), other types within 10%.
3. **General-knowledge retention.** Val PPL on pretraining-distribution slice within 10% of base.

**Outputs:** `phase1_gate_report.md`, `phase1_gate.json`. Pass → promote to `phase1_verified.pt`. Fail → automated knob suggestion + halt.

## Phase 2 — Spectral SFT (the activator loop)

Essentially the current `finetune_spectral.py` pipeline, starting from `phase1_verified.pt`.

- Dialogue corpus with **assistant-only loss mask** (CLAUDE.md — current RAI run lacks this, inflating val PPL ~2×).
- System prompt during SFT = activator at inference time. Consistency is load-bearing.
- For Tier C/D with sparse organic dialogue: synthesize pairs by prompting a strong model with verified Phase 1 bio knowledge + question archetypes; subject reviews/edits a sample before training on the synthetic set. Keep 50–100 hand-authored pairs for val.
- Decompose `phase1_verified.pt` → train spectrum with type-aware harmonic regularizer.
- Halt on val PPL plateau OR attn_o drift > 5%.

**Artifact:** `phase2_spectrum.pt`.

## Inference & Deployment

- **Recompose** Phase 2 spectrum + Phase 1 residuals into standard HF state_dict (`scripts/recompose_spectral.py` — validated at 0.34% mean reconstruction error, val PPL within 6%). Load with vanilla `from_pretrained`. No SpectralLinear at inference (bf16 arithmetic issue, CLAUDE.md lesson #9).
- **Activator** = subject-specific system prompt, identical to Phase 2 SFT sysprompt.
- **RAG layer** for recent/private facts: vector index over an append-only subject log (transcripts, notes, updates). Retrieved chunks appended to sysprompt as `<context>...</context>`. Keeps weights stable, facts fresh.

### Per-subject artifacts

- `recomposed_bf16/` — deployable HF model
- `activator.txt` — versioned system prompt
- `rag_index/` — vector store
- `probe_history.json` — every probe run (Phase 0 / 1.5 / final) for regression tracking

### Tier-specific deployment

- Tier A: RAG optional (post-cutoff recency).
- Tier B: RAG recommended for specificity.
- Tier C/D: RAG **required** — weights carry persona, RAG carries facts the small corpus couldn't cover.

## Validation Harness & Success Criteria

Four gates, all required:

1. **Gene strength.** Phase 0 probe with sysprompt on final model. Pass: ≥90% probes ≥1, ≥70% at score 2. Must beat Phase 1.5 numbers.
2. **Voice fidelity.** 10–20 open-ended dialogue prompts. Subject or proxy grades 1–5 (persona accuracy, idiolect, signature beliefs, conversational moves). Pass: mean ≥4.0, no sample <3.
3. **Dormancy.** Dialogue probe **without** sysprompt. Pass: model reverts to default assistant. No persona leak. Fail → retrain Phase 2 with sysprompt dropout (omit sysprompt on ~20% of training steps).
4. **φ-structure integrity.** Final model free-α scan. attn_o within 5% of 1/3, other types within 10% of base.

### Artifacts

- `eval_final.md` — human-readable report
- `eval_final.json` — machine-readable gates (gene/voice/dormancy/phi)
- `regression.json` — diff vs previous version

**Deployable iff all four gates green.** No manual overrides — bypassing gates destroys reproducibility.

## Scope limits (YAGNI)

- One model per subject. Multi-persona-per-base is Option C (spectral split per persona) — revisit only if fleet of subjects gets unwieldy.
- No automated bio-corpus scraping. Source material assembly is manual / human-curated per subject.
- No adaptive λ scheduling during Phase 1. Fixed λ; if gate 1.5 fails, retrain with a different λ. Simpler than inner-loop adaptation.
- No cross-subject contamination tests in v1. We assume one model = one subject.

## Open questions for implementation

1. **Concrete LR & λ starting values for Phase 1** — extract from the current RAI-cranked-v3 run's hyperparameters.
2. **What counts as "general-knowledge held-out slice"** for the forgetting guard — wikitext? a curated probe? Needs a standard.
3. **Synthetic dialogue quality bar for Tier C/D** — minimum human-review sample size before trusting synthetic set.
4. **RAG index granularity** — per-paragraph vs per-doc chunking; reranker or vanilla cosine.

These don't block the design; they're v2 tuning knobs.
