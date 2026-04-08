# WaveGPT Pipeline

**From open source model to chat window in five phases.**

One base model loaded once. Infinite personalities as 1.4MB spectral files. Confidence-gated RAG for grounding. Self-distillation for precision. Sub-second personality swap.

## The Equation

Every trained weight matrix converges to:

```
σ_k = A · (k + k₀)^{-1/φ}
```

- **1/φ = 0.6180339887** — universal constant, the inverse golden ratio. Not learned, not a hyperparameter. It falls out of gradient descent the way π falls out of circles.
- **k₀** — spectral offset, per-layer. Describes how many modes carry near-equal energy before the power-law decay begins. Predicted by layer type and dimensions.
- **A** — amplitude, per-layer. The one free parameter that encodes personality.

Knowledge lives in the geometry (U, V — the directions). Voice lives in the amplitudes (S — the spectrum). Change the amplitudes, change the voice. Keep the geometry, keep the knowledge.

→ Deep dive: [the-discovery.md](the-discovery.md)

## Architecture Overview

```
═══════════════════════════════════════════════════════════════
                    PHASE 1: DECOMPOSE
                  Done once per base model
═══════════════════════════════════════════════════════════════

                ┌─────────────────────┐
                │  Qwen3.5-27B (HF)   │  Any open source model.
                │  54GB BF16          │  Downloaded from HuggingFace.
                └──────────┬──────────┘
                           │
                    SVD every layer
                    Fit bent power law
                    Set rank by k₀
                           │
                ┌──────────▼──────────┐
                │   decomposed.pt     │  Frozen geometry (U, V) in BF16.
                │   ~14GB on disk     │  Learnable spectrum (S) in float32.
                │   496 layers        │  Adaptive rank per layer.
                │   350K params free  │  The rest: frozen forever.
                └──────────┬──────────┘
                           │
═══════════════════════════╪══════════════════════════════════
                           │
                    PHASE 2: CORPUS
              Same data serves two purposes
                           │
          ┌────────────────┼────────────────┐
          │                                 │
          ▼                                 ▼
  ┌───────────────┐               ┌─────────────────┐
  │ Tokenized for │               │ Chunked for     │
  │ training      │               │ retrieval       │
  │               │               │                 │
  │ Ray's books,  │               │ FAISS vectors   │
  │ interviews,   │               │ Knowledge graph │
  │ transcripts   │               │ Entity store    │
  │ → train.bin   │               │ Cross-encoder   │
  └───────┬───────┘               └────────┬────────┘
          │                                │
══════════╪════════════════════════════════╪══════════════════
          │                                │
          │    PHASE 3: SPECTRAL FT        │
          │   Done once per personality    │
          │                                │
  ┌───────▼───────┐                        │
  │ Load          │                        │
  │ decomposed.pt │                        │
  │               │                        │
  │ Train only S  │  Freeze U, V.          │
  │ on Ray's      │  Learn amplitudes.     │
  │ corpus.       │  Voice emerges.        │
  │               │                        │
  │ Harmonic reg: │  Stay near k^{-1/φ}.   │
  │ Anti-collapse:│  Preserve diversity.   │
  └───────┬───────┘                        │
          │                                │
  ┌───────▼───────┐                        │
  │spectral_ray.pt│  350K floats.          │
  │ 1.4MB         │  Ray's voice.          │
  └───────┬───────┘                        │
          │                                │
══════════╪════════════════════════════════╪══════════════════
          │                                │
          │    PHASE 4: SELF-DISTILLATION  │
          │    (SSD — arXiv:2604.01193)    │
          │    Done once per personality   │
          │                                │
  ┌───────▼────────────────────────────────▼──┐
  │                                           │
  │  The precision-exploration compound:      │
  │                                           │
  │  1. Generate RAG-style prompts            │
  │     (questions about Ray's ideas)         │
  │                                           │
  │  2. Retrieve chunks from the RAG index    │
  │     (same index used in production)       │
  │                                           │
  │  3. Feed [chunks + question] to model     │
  │     with spectral_ray.pt loaded           │
  │                                           │
  │  4. Sample N responses at T=1.2           │
  │     top_k=50, top_p=0.95                  │
  │     Higher temp → explore voice options   │
  │     Truncation → cut distractor tokens    │
  │                                           │
  │  5. Fine-tune spectrum on self-generated  │
  │     responses (standard cross-entropy)    │
  │                                           │
  │  Result: the spectrum learns WHEN to be   │
  │  precise (facts, dates, quotes from RAG)  │
  │  and WHEN to be exploratory (Ray's voice, │
  │  analogies, connections between ideas)    │
  │                                           │
  └─────────────────────┬─────────────────────┘
                        │
              ┌─────────▼─────────┐
              │spectral_ray_ssd.pt│  Same 1.4MB.
              │ Voice + precision │  Better grounding.
              └─────────┬─────────┘
                        │
════════════════════════╪═════════════════════════════════════
                        │
                  PHASE 5: DEPLOY
              Runs forever, serves users
                        │
              ┌─────────▼─────────┐
              │   GPU SERVER       │
              │                   │
              │ decomposed.pt     │  Loaded once into VRAM.
              │ + spectral_ray.pt │  1.4MB, hot-swapped.
              │                   │
              │ Swap personality: │
              │  ray.pt → bob.pt  │  < 1 second.
              │  350K floats      │  memcpy, no reload.
              │                   │
              └─────────┬─────────┘
                        │
════════════════════════╪═════════════════════════════════════
                        │
                  PHASE 6: QUERY
              Confidence-gated RAG
                        │
              ┌─────────▼─────────┐
              │   User asks a     │
              │   question        │
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │   PROBE PASS      │  ~100ms
              │                   │
              │ Generate 20 tokens│
              │ Measure mean      │
              │ token probability │
              └────┬─────────┬────┘
                   │         │
              HIGH │         │ LOW
           mean(p) > τ    mean(p) < τ
                   │         │
        ┌──────────▼┐   ┌───▼──────────────┐
        │  DIRECT   │   │  RAG + GENERATE   │
        │  GENERATE │   │                   │
        │           │   │ Vector search     │
        │ Voice     │   │ + KG traversal    │
        │ alone.    │   │ + Cross-encoder   │
        │           │   │ → Top 5 chunks    │
        │ ~500ms    │   │ → Generate with   │
        │           │   │   grounding       │
        │           │   │ ~2s               │
        └─────┬─────┘   └────────┬──────────┘
              │                  │
              └────────┬─────────┘
                       │
              ┌────────▼────────┐
              │                 │
              │  "Look, I've   │
              │   been thinking │
              │   about this   │
              │   for decades  │
              │   ..."         │
              │                 │
              │  → Chat window │
              └─────────────────┘
```

## Phase 1: Spectral Decomposition

**What**: SVD every linear layer of the base model, fit the bent power law, set rank adaptively by k₀.

**Why**: Separates knowledge (frozen geometry U, V) from voice (learnable spectrum S). The k₀-adaptive rank ensures each layer keeps enough modes to maintain coherent language while compressing maximally.

**How**:
```bash
python scripts/decompose_only.py \
    --hf-model Qwen/Qwen3.5-27B \
    --adaptive-k0 --k0-mult 1.5 --k0-pad 128 \
    --mode per_mode \
    --output decomposed.pt
```

**Time**: ~6 hours on CPU (no GPU needed). Done once. All future work loads this file.

**Output**: `decomposed.pt` (~14GB)
- 496 SpectralLinear layers with frozen U, V buffers and learnable spectrum parameters
- Attention layers: rank ~200-800 (low k₀, sharp spectrum, few active modes)
- MLP layers: rank ~1300-2000 (high k₀, broad spectrum, many active modes)

**Why adaptive rank matters**: Uniform rank-256 produces punctuation soup. The MLP layers have k₀≈1000, meaning ~1000 modes carry near-equal energy before decay begins. Truncating to 256 cuts through this flat top and destroys language coherence. Adaptive rank clears each layer's flat top: `rank = k₀ × 1.5 + 128`.

→ Deep dive: [spectral-personalities.md](spectral-personalities.md)

## Phase 2: Corpus Preparation

**What**: The author's complete body of work — books, interviews, transcripts, essays. Processed into two parallel formats from the same source material.

**For training** (spectral fine-tuning):
- Tokenized with the base model's tokenizer
- Stored as binary token arrays: `train.bin`, `val.bin`
- Used to teach the model the author's voice

**For retrieval** (RAG):
- Chunked into ~300 token passages with metadata
- Embedded with a sentence transformer (e.g. all-MiniLM-L6-v2)
- Indexed in FAISS for vector similarity search
- Entities and relationships extracted into a knowledge graph
- Cross-encoder (e.g. ms-marco-MiniLM) for reranking

**The dual-purpose insight**: The same corpus that teaches the model to SOUND like Ray also provides the factual grounding when users ask specific questions. The voice comes from the spectrum. The facts come from retrieval. Same data, different representations.

## Phase 3: Spectral Fine-Tuning

**What**: Train only the spectral amplitudes (S) on the author's corpus. Everything else is frozen.

**Why**: The base model already knows how to speak, reason, follow instructions. All of that lives in U and V (frozen). We only need to adjust HOW MUCH each knowledge direction is activated — which is the spectrum.

**How**:
```bash
python scripts/finetune_spectral.py \
    --decomposed decomposed.pt \
    --data-dir data/ray-corpus \
    --run-name ray-voice \
    --mode per_mode \
    --batch-size 4 --block-size 512 --grad-accum 4 \
    --lr 1e-3 --max-steps 2000 \
    --harmonic-lambda 0.01 --collapse-alpha 0.05
```

**Harmonic priors** (what makes this different from vanilla SVFit):
- **Harmonic regularization** (λ=0.01): Penalizes deviation from the k^{-1/φ} power law. The spectrum should stay near the universal prior — personality is a small perturbation, not a wholesale restructuring.
- **Anti-collapse** (α=0.05): Variance penalty on hidden states. Prevents the model from collapsing all outputs to the same representation (semantic collapse).

**Time**: ~3 hours on one GPU. Produces a 1.4MB spectral file.

**Output**: `spectral_ray.pt` — 350K floats encoding Ray's voice.

## Phase 4: Self-Distillation (SSD)

**What**: The model generates RAG-style responses, then fine-tunes its own spectrum on those responses. No external labels, no teacher model, no verification.

**Why** (arXiv:2604.01193): Language generation has a precision-exploration conflict. Some positions in a response need precision — quoting a retrieved fact, getting a date right, maintaining grammatical structure. Other positions need exploration — choosing which analogy to reach for, how to frame an insight, what Ray would emphasize. A single temperature can't serve both. SSD resolves this by reshaping the probability distribution context-dependently.

**The compound effect**:
- Phase 3 (spectral FT) taught the model WHO to be → Ray's voice
- Phase 4 (SSD) teaches the model WHEN to be precise → RAG grounding

**How**:
```bash
python scripts/finetune_spectral.py \
    --decomposed decomposed.pt \
    --data-dir data/ray-corpus \
    --run-name ray-ssd \
    --mode per_mode \
    --ssd --ssd-temperature 1.2 --ssd-top-k 50 --ssd-top-p 0.95 \
    --ssd-samples 8 --ssd-steps 500
```

**The math** (from the SSD paper, Eq. 15):
```
L(θ) = -log(KeptMass)              ← support compression
     + (1-T)·H_{1/T}(p|S)          ← Rényi reshaping within support
     + T·KL(q ∥ Temper_T[p|S])     ← alignment with teacher
```

Spectral surgery does the first two in weight space (rank truncation = support compression, amplitude adjustment = reshaping). SSD does them in output space (top-k truncation = support compression, temperature = reshaping). Orthogonal domains, compound result.

**Time**: ~1 hour. Same spectral file, refined.

**Output**: `spectral_ray_ssd.pt` — same 1.4MB, better precision.

## Phase 5: Deployment

**What**: Load the decomposed base model once. Swap spectral files per request.

**Model in VRAM**:
```
decomposed.pt (14GB)
  ├── 496 SpectralLinear layers
  │   ├── U buffers (frozen, BF16) — knowledge geometry
  │   ├── V buffers (frozen, BF16) — knowledge geometry
  │   └── S parameters (float32) — current personality
  ├── Embeddings (frozen)
  └── LayerNorm (frozen)
```

**Personality swap**:
```python
# Load spectral file (1.4MB, from disk or cache)
personality = torch.load("spectral_ray_ssd.pt")

# Overwrite spectrum parameters (memcpy, ~1ms)
for name, param in model.named_parameters():
    if name in personality:
        param.data.copy_(personality[name])

# Next forward pass uses Ray's voice
```

No model reload. No VRAM reallocation. Just overwriting 350K floats.

**Scaling**:
| Personalities | VRAM | Disk |
|--------------|------|------|
| 1 | 14.001 GB | 1.4 MB |
| 10 | 14.014 GB | 14 MB |
| 100 | 14.14 GB | 140 MB |
| 1000 | 15.4 GB | 1.4 GB |
| LoRA (1000) | 14 GB + 200 GB | 200 GB |

## Phase 6: Query Time

**What**: Confidence-gated RAG. The model decides whether it needs retrieved context.

### The probe

Before committing to a full response, generate 20 tokens and measure confidence:

```python
with torch.no_grad():
    outputs = model.generate(
        prompt_ids, max_new_tokens=20,
        output_scores=True, return_dict_in_generate=True,
    )
    probs = [F.softmax(s, dim=-1).max().item() for s in outputs.scores]
    confidence = sum(probs) / len(probs)
```

**High confidence** (mean token probability > τ): The model knows this. Let it speak directly. No RAG overhead. ~500ms response.

**Low confidence** (mean token probability < τ): The model is uncertain. Retrieve context, rebuild prompt with grounding chunks, generate with RAG. ~2s response.

### When each path fires

**Direct generation** — philosophy, worldview, frequently discussed ideas:
> "What does Ray believe about consciousness?"

The spectral voice carries this. These are the ideas Ray repeats across every book. The model internalized them during spectral FT. RAG would over-anchor on one specific quote instead of synthesizing Ray's broader view.

**RAG-grounded generation** — specific facts, dates, quotes, predictions:
> "What prediction did Ray make about solar energy costs in chapter 4?"

The model can't hallucinate this. It needs the actual chunk. RAG retrieves the passage, the model grounds its answer in it, the spectral voice frames the delivery.

### RAG retrieval stack

When retrieval triggers:
1. **Vector search**: Query embedding → FAISS → top 20 candidates
2. **Knowledge graph traversal**: Extract entities from query → find connected chunks
3. **Reciprocal rank fusion**: Merge vector + KG candidate lists
4. **Cross-encoder rerank**: Score each candidate against the query → top 5

### Prompt construction

```
System: You are Ray Kurzweil. Speak in first person. Ground your
answers in the provided context. If the context doesn't contain
relevant information, draw on your general knowledge.

Context:
[Chunk 1: "In The Singularity Is Near, I argued that..."]
[Chunk 2: "The exponential growth of solar energy..."]
[Chunk 3: ...]

User: What did Ray think about longevity?
```

The system prompt sets the frame. The chunks provide grounding. The spectral file provides the voice. The base model provides the reasoning.

## Cost Summary

| Phase | Time | Cost | Frequency |
|-------|------|------|-----------|
| Download base model | 30 min | Free | Once ever |
| Spectral decomposition | 6 hours | ~$10 (GPU rental) | Once per base model |
| Corpus tokenization | 10 min | Free | Once per personality |
| RAG index build | 30 min | Free | Once per personality |
| Spectral fine-tuning | 3 hours | ~$5 | Once per personality |
| Self-distillation (SSD) | 1 hour | ~$2 | Once per personality |
| **Total per new personality** | **~4 hours** | **~$7** | |

**Deployment**: One GPU with 16+ GB VRAM. Serves unlimited personalities from the same model instance.

## Comparison with Alternatives

| | Full Fine-Tune | LoRA r-16 | **WaveGPT Spectral** |
|---|---|---|---|
| Trainable params | 26.9B | 111M | **350K** |
| Adapter size | 54GB | 212MB | **1.4MB** |
| Training time | Days | Hours | **Hours** |
| Swap time | Minutes (reload) | Seconds | **Milliseconds** |
| VRAM per personality | 54GB | ~400MB | **~0** |
| Theoretical basis | None | Random low-rank | **k^{-1/φ} universal law** |
| RAG precision (with SSD) | N/A | N/A | **Compound** |
| 1000 personalities disk | 54TB | 200GB | **1.4GB** |

## References

- **The equation**: [the-discovery.md](the-discovery.md) — how we found 1/φ, the double-slit insight, spectral surgery
- **Spectral files**: [spectral-personalities.md](spectral-personalities.md) — file format, swapping mechanism, personality arithmetic
- **Harmonic theory**: [theory.md](theory.md) — data harmonics, anti-collapse, the Pythagorean comma
- **SSD paper**: [arXiv:2604.01193](https://arxiv.org/abs/2604.01193) — "Embarrassingly Simple Self-Distillation Improves Code Generation" (Apple, 2026)
- **Prior art (mechanism)**: SVFit/SVFT (NeurIPS 2024), SVDiff (ICCV 2023), PiSSA (2024)
- **Prior art (theory)**: Martin & Mahoney (2018-2021), "From SGD to Spectra" (2025)
- **Experiments**: [experiments.md](experiments.md) — full experiment log from GPT-2 30M through Qwen3.5-27B
