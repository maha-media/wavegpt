# Persona Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the end-to-end gene+activator pipeline (Phase 0 → Phase 1 CPT → Phase 1.5 gate → Phase 2 SFT → recompose → final eval) and land a deployable Ray Kurzweil persona artifact that passes all four validation gates.

**Architecture:** Extend the existing spectral fine-tune stack with a front-loaded CPT phase that plants subject knowledge under a harmonic regularizer (protecting φ-structure end-to-end), gate it with an automated verification pass, then specialize voice via the current spectral SFT loop. Execute entirely on RunPod GPU servers; commit artifacts locally after each phase.

**Tech Stack:** PyTorch 2.4 (RunPod), transformers (Gemma 4-31B), FSDP via `accelerate`, existing `wavegpt/harmonic_prior.py` + `spectral_linear.py` + `spectral_surgery.py`, existing `scripts/finetune_spectral.py` + `eval_watcher.py` + `decompose_only.py` + `recompose_spectral.py`.

**Subject:** Ray Kurzweil (probe set at `probes/ray_kurzweil.json`; first calibration run).

**Artifact root:** `runs/ray/` — `phase0/`, `phase1/`, `phase1_5/`, `phase2/`, `deploy/`, `eval_final/`.

**Stops:** End of Phase 0, Phase 1, Phase 1.5, Phase 2, and Final Eval. Nothing else blocks — continuous within phases.

---

## Phase 0 — Gene Probe & Tier Classification

Scaffold already exists (`scripts/phase0_probe.py`, `scripts/phase0_classify.py`, `probes/ray_kurzweil.json`). This phase is "run it, score it, classify it."

### Task 01 — Run baseline probe on pod

**Files:**
- Uses: `scripts/phase0_probe.py`, `probes/ray_kurzweil.json`
- Produces on pod: `/workspace/runs/ray/phase0/probe_baseline.md`, `probe_baseline.json`

**Step 1:** scp probe set + runner to Gemma pod (port 18409).

```bash
scp -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -P 18409 \
    probes/ray_kurzweil.json \
    scripts/phase0_probe.py scripts/eval_watcher.py \
    root@216.243.220.173:/workspace/
```

**Step 2:** Run probe with empty sysprompt on bare `google/gemma-4-31b-it`.

```bash
ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -p 18409 root@216.243.220.173 \
  "cd /workspace && CUDA_VISIBLE_DEVICES=4,5 python3 -u phase0_probe.py \
      --probes ray_kurzweil.json \
      --model-dir google/gemma-4-31b-it \
      --output-dir runs/ray/phase0 \
      --trust-remote-code 2>&1 | tee /root/ray_phase0_probe.log"
```

Expected: 15 probes complete, markdown + JSON written.

**Step 3:** scp artifacts back.

```bash
mkdir -p runs/ray/phase0
scp -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -P 18409 \
    root@216.243.220.173:/workspace/runs/ray/phase0/probe_baseline.{md,json} \
    runs/ray/phase0/
```

**Step 4:** Commit the probe run.

```bash
git add runs/ray/phase0/probe_baseline.md runs/ray/phase0/probe_baseline.json
git commit -m "runs(ray): phase0 baseline probe against gemma-4-31b-it"
```

### Task 02 — Human-score the probe

**Files:** `runs/ray/phase0/probe_baseline.md` (edit in place).

**Step 1:** Flip every `**score:** ?` line to `**score:** 0`, `**score:** 1`, or `**score:** 2`. Refusal / flat-wrong = 0, partially right = 1, materially correct = 2. 15 probes total.

**Step 2:** Commit the scored file.

```bash
git add runs/ray/phase0/probe_baseline.md
git commit -m "runs(ray): score phase0 probe"
```

### Task 03 — Classify tier + emit gap categories

**Files:** Produces `runs/ray/phase0/tier.json`.

**Step 1:** Run classifier locally (no GPU needed).

```bash
python3 scripts/phase0_classify.py --probe-dir runs/ray/phase0
```

Expected stdout: `tier=<A|B|C|D>  correct_rate=XX%  strong_rate=XX%  (n=15)` plus any gap categories.

**Step 2:** Commit the tier artifact.

```bash
git add runs/ray/phase0/tier.json
git commit -m "runs(ray): classify phase0 tier + gap categories"
```

### Task 04 — Phase 0 checkpoint review

**Stop and read:** Review `tier.json`. Confirm tier + `gap_categories` with user before building Phase 1 corpus; gap categories drive oversampling. If tier is D, flag for synthetic-dialogue augmentation in Phase 2.

**Phase 0 done when:** `runs/ray/phase0/tier.json` committed and tier reviewed.

---

## Phase 1 — Harmonically-Regularized CPT (the gene)

Full continued pretraining of Gemma-4-31B on the Ray bio corpus under `harmonic_prior(type_aware=True, attn_o_weight=w)`, with a forgetting-guard val slice.

### Task 05 — Assemble bio corpus

**Files:** Create `corpora/ray/raw/` and drop source text files there.

**Step 1:** Collect source material (books, long-form interviews, essays, transcripts). Store as plain `.txt` or `.md` under `corpora/ray/raw/`. One source per file, UTF-8.

**Step 2:** Write `corpora/ray/MANIFEST.md` listing each file, its provenance, approximate word count, and any licensing notes.

**Step 3:** Commit corpus (or a `.gitignore`d pointer if sources can't be checked in; in that case commit only `MANIFEST.md`).

```bash
git add corpora/ray/MANIFEST.md
# optionally: git add corpora/ray/raw/*.txt
git commit -m "corpora(ray): seed bio corpus manifest"
```

### Task 06 — Build corpus-prep script

**Files:**
- Create: `scripts/phase1_corpus_prep.py`
- Test: `tests/test_phase1_corpus_prep.py`

**Step 1: Write the failing test.**

```python
# tests/test_phase1_corpus_prep.py
import json, subprocess, sys
from pathlib import Path

def test_chunking_and_oversample(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    (raw / 'src.txt').write_text(' '.join(['word'] * 10000))
    tier = {'gap_categories': ['idiom']}
    (tmp_path / 'tier.json').write_text(json.dumps(tier))
    probes = {'probes': [{'id': 'x', 'category': 'idiom',
                          'question': 'q', 'expected': 'e'}]}
    (tmp_path / 'probes.json').write_text(json.dumps(probes))
    out = tmp_path / 'out'
    r = subprocess.run([sys.executable, 'scripts/phase1_corpus_prep.py',
                        '--raw-dir', str(raw),
                        '--tier-json', str(tmp_path / 'tier.json'),
                        '--probes', str(tmp_path / 'probes.json'),
                        '--tokenizer', 'google/gemma-4-31b-it',
                        '--window', '2048', '--overlap', '128',
                        '--output-dir', str(out),
                        '--oversample-factor', '3',
                        '--dry-run'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['n_chunks'] > 0
    assert manifest['oversampled_categories'] == ['idiom']
```

**Step 2:** Run it — expect FAIL (script doesn't exist).

```bash
pytest tests/test_phase1_corpus_prep.py -v
```

**Step 3: Implement.**

Script responsibilities (keep to ~200 lines):
1. Read `tier.json` → `gap_categories`.
2. Walk `--raw-dir`, read text files, concatenate, tokenize with HF tokenizer.
3. Emit 2k-token windows w/ 128 overlap into a single `train.bin` + `val.bin` (95/5 split) using `wavegpt.data_io.write_datafile`.
4. For each `gap_category` in tier: synthesize `--oversample-factor` copies of text blocks whose nearest probe (by category) matches. Heuristic match: substring of any probe `expected` appears in the source chunk. Append these to the train split.
5. Write `manifest.json`: `{n_chunks, n_tokens, oversampled_categories, window, overlap, source_files: [...]}`.
6. `--dry-run`: skip tokenizer download; emit manifest with fake token counts so tests run without network.

**Step 4:** Run test — expect PASS.

```bash
pytest tests/test_phase1_corpus_prep.py -v
```

**Step 5: Commit.**

```bash
git add scripts/phase1_corpus_prep.py tests/test_phase1_corpus_prep.py
git commit -m "feat(phase1): corpus prep with gap-category oversampling"
```

### Task 07 — Run corpus prep for Ray

**Step 1:** Execute against real corpus (local, needs tokenizer).

```bash
python3 scripts/phase1_corpus_prep.py \
    --raw-dir corpora/ray/raw \
    --tier-json runs/ray/phase0/tier.json \
    --probes probes/ray_kurzweil.json \
    --tokenizer google/gemma-4-31b-it \
    --window 2048 --overlap 128 \
    --oversample-factor 3 \
    --output-dir corpora/ray/prepared
```

**Step 2:** Inspect `corpora/ray/prepared/manifest.json`. Sanity: `n_tokens ≥ 500k` for Tier A/B; larger for C/D.

**Step 3:** Commit manifest (not the `.bin` files — those go in `.gitignore`).

```bash
echo "corpora/ray/prepared/*.bin" >> .gitignore
git add .gitignore corpora/ray/prepared/manifest.json
git commit -m "runs(ray): phase1 corpus prepared"
```

### Task 08 — Forgetting-guard slice

**Files:** Create `scripts/build_forgetting_slice.py`, emit `corpora/forgetting_guard/wikitext_val.bin`.

**Step 1:** Build a 50k-token held-out slice from `wikitext-103` validation split, tokenized with the Gemma tokenizer. One-shot script, reusable across subjects.

**Step 2:** Emit `corpora/forgetting_guard/manifest.json` with SHA of the binary for reproducibility.

**Step 3:** Commit script + manifest (bin is gitignored).

```bash
git add scripts/build_forgetting_slice.py corpora/forgetting_guard/manifest.json
git commit -m "feat: forgetting-guard slice (wikitext-103 val, gemma tokens)"
```

### Task 09 — Build Phase 1 CPT trainer

**Files:**
- Create: `scripts/phase1_cpt.py`
- Test: `tests/test_phase1_cpt_smoke.py` (single-GPU smoke, tiny model)

**Step 1: Write the smoke test.**

```python
# tests/test_phase1_cpt_smoke.py
import subprocess, sys, json
from pathlib import Path

def test_cpt_one_step(tmp_path):
    # Build a 1k-token fake train/val bin
    import numpy as np
    (tmp_path / 'train.bin').write_bytes(np.arange(1024, dtype=np.uint16).tobytes())
    (tmp_path / 'val.bin').write_bytes(np.arange(1024, dtype=np.uint16).tobytes())
    (tmp_path / 'forget.bin').write_bytes(np.arange(1024, dtype=np.uint16).tobytes())
    r = subprocess.run([sys.executable, 'scripts/phase1_cpt.py',
                        '--smoke',
                        '--train-bin', str(tmp_path / 'train.bin'),
                        '--val-bin',   str(tmp_path / 'val.bin'),
                        '--forget-bin',str(tmp_path / 'forget.bin'),
                        '--output-dir', str(tmp_path / 'out'),
                        '--max-steps', '2',
                        '--harmonic-lambda', '0.01',
                        '--attn-o-weight', '10.0'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    log = json.loads((tmp_path / 'out' / 'training_log.json').read_text())
    assert len(log) >= 2
    assert 'harmonic_loss' in log[-1]
    assert 'forget_ppl' in log[-1]
```

**Step 2:** Run — expect FAIL.

**Step 3: Implement `phase1_cpt.py`.** Core loop (reuse patterns from `finetune_spectral.py` but without spectral decomposition — full-weight training):

```python
# Sketch of the trainer body
from transformers import AutoModelForCausalLM, AutoTokenizer
from wavegpt.harmonic_prior import harmonic_regularization
from wavegpt.dataloader import TokenDataloader
import torch, json
from accelerate import Accelerator

accelerator = Accelerator()  # FSDP config via accelerate config
model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16,
                                             trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
model, optim = accelerator.prepare(model, optim)

train_loader = TokenDataloader(args.train_bin, args.window, args.batch_size)
val_loader   = TokenDataloader(args.val_bin,   args.window, args.batch_size)
forget_loader= TokenDataloader(args.forget_bin,args.window, args.batch_size)

log = []
for step, batch in enumerate(train_loader):
    if step >= args.max_steps: break
    out = model(input_ids=batch, labels=batch)
    hp = harmonic_regularization(model, type_aware=True,
                                 attn_o_weight=args.attn_o_weight)
    loss = out.loss + args.harmonic_lambda * hp
    accelerator.backward(loss)
    optim.step(); optim.zero_grad()

    if step % args.eval_every == 0 or step == args.max_steps - 1:
        val_ppl     = eval_ppl(model, val_loader)
        forget_ppl  = eval_ppl(model, forget_loader)
        log.append({'step': step, 'train_loss': float(out.loss),
                    'harmonic_loss': float(hp),
                    'val_ppl': val_ppl, 'forget_ppl': forget_ppl})
        json.dump(log, open(f'{args.output_dir}/training_log.json','w'), indent=2)
        # Save best on val_ppl
        # Early stop on 3 consecutive non-improvements
```

Forgetting-guard gate inside the loop: if `forget_ppl / forget_ppl_base > 1.10`, print warning. Don't halt; let human decide.

`--smoke` swaps the HF load for a 2-layer toy `GPTNeoX` or similar so the test runs on CPU in <30s.

**Step 4:** Run test — expect PASS.

**Step 5: Commit.**

```bash
git add scripts/phase1_cpt.py tests/test_phase1_cpt_smoke.py
git commit -m "feat(phase1): harmonic-regularized CPT trainer"
```

### Task 10 — Launch Phase 1 CPT on pod

**Pod endpoint:** resolved at launch time via `runpodctl ssh info <pod-id>`. Current: `185.216.23.177:47489` (pod `cmwrnr1qa40c2a`, 6× A100 SXM). Substitute host/port in the commands below.

**Step 1:** scp trainer + prepared corpus + forgetting slice to pod.

```bash
scp -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -P 47489 \
    scripts/phase1_cpt.py \
    data/ray/phase1/{train,val}.bin \
    corpora/forgetting_guard/wikitext_val.bin \
    root@185.216.23.177:/workspace/runs/ray/phase1/
```

**Step 2:** Compute `forget_ppl_base` once by running the trainer with `--eval-only` so the log has a baseline entry.

**Step 3:** Launch FSDP training (4× GPUs 0-3; watcher will use 4-5).

No harmonic regularizer — see the trainer's module docstring and `memory/project_spectral_sft_knobs.md`. `attn_o`'s pretrained 1/3 exponent is preserved via per-tier LR (`--attn-o-lr-mult 0.1` → attn_o trains at 1e-6 while everything else trains at 1e-5).

```bash
ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -p 47489 root@185.216.23.177 \
"cd /workspace && \
 CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
   --num_processes 4 --mixed_precision bf16 --use_fsdp \
   scripts/phase1_cpt.py \
   --model-dir google/gemma-4-31b-it --trust-remote-code \
   --train-bin runs/ray/phase1/train.bin \
   --val-bin   runs/ray/phase1/val.bin \
   --forget-bin runs/ray/phase1/wikitext_val.bin \
   --output-dir runs/ray/phase1 \
   --lr 1e-5 --attn-o-lr-mult 0.1 \
   --window 2048 --batch-size 2 \
   --max-steps 6000 --eval-every 100 \
   > /root/ray_phase1.log 2>&1 &"
```

**Step 4:** Tail the log; expect initial `val_ppl` drop in first 500 steps, attn_o drift check in first checkpoint.

### Task 11 — Monitor Phase 1, collect checkpoint

**Step 1:** Watch `val_ppl` curve; stop manually on plateau (3 consecutive non-improvements).

**Step 2:** Verify `forget_ppl` is within 10% of baseline. If not, halt, raise λ or lower LR, restart.

**Step 3:** scp best checkpoint + log back.

```bash
scp -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -P 18409 \
    root@216.243.220.173:/workspace/runs/ray/phase1/{best.pt,training_log.json} \
    runs/ray/phase1/
```

**Step 4:** Commit the log (not the `.pt` — too large; track via DVC later or just keep on pod).

```bash
git add runs/ray/phase1/training_log.json
git commit -m "runs(ray): phase1 CPT training log"
```

### Task 12 — Phase 1 checkpoint review

**Stop and read:** Confirm val PPL plateau + forgetting-guard within 10%. Move to Phase 1.5 gate.

**Phase 1 done when:** `runs/ray/phase1/best.pt` on pod + training log committed locally.

---

## Phase 1.5 — Gene Verification Gate

Automated three-gate check. Pass → promote to `phase1_verified.pt`; fail → halt with knob suggestion.

### Task 13 — Build gate script

**Files:**
- Create: `scripts/phase1_gate.py`
- Test: `tests/test_phase1_gate.py`

**Step 1: Write failing test** — feed a fake probe output + fake α analysis + fake forget PPL, assert `phase1_gate.json` has correct pass/fail flags for each sub-gate.

**Step 2:** Run — FAIL.

**Step 3: Implement.** Script responsibilities:

1. Re-run `phase0_probe` logic on the phase1 checkpoint (import + call, don't shell out).
2. Re-run `phase0_classify` → new `tier.json`. Compare to baseline: gate 1 passes iff `correct_rate ≥ 0.70` AND for each baseline `gap_category`, `category_rates[cat].correct_rate ≥ 2× baseline`.
3. Import `scripts.free_alpha_analysis` or `gemma4_alpha_analysis`, run per-type α fit. Gate 2: `attn_o` within 5% of 1/3 (≈0.333±0.017), all other types within 10% of base.
4. Measure val PPL on the forgetting-guard slice. Gate 3: within 10% of base.
5. Emit `runs/ray/phase1_5/phase1_gate.json` with three booleans + numbers, and `phase1_gate.md` with human-readable summary.
6. If all three pass: `cp best.pt phase1_verified.pt`.
7. If any fail: emit `knob_suggestion.md` — e.g., "gate 2 failed (attn_o drifted to 0.29): increase `--attn-o-weight` from 10 → 25, or lower LR to 5e-6".

**Step 4:** Run test — PASS.

**Step 5: Commit.**

```bash
git add scripts/phase1_gate.py tests/test_phase1_gate.py
git commit -m "feat(phase1.5): automated verification gate"
```

### Task 14 — Run gate on pod

```bash
scp -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -P 18409 \
    scripts/phase1_gate.py \
    root@216.243.220.173:/workspace/
ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -p 18409 root@216.243.220.173 \
"cd /workspace && CUDA_VISIBLE_DEVICES=4,5 python3 -u scripts/phase1_gate.py \
   --checkpoint runs/ray/phase1/best.pt \
   --probes ray_kurzweil.json \
   --baseline-tier runs/ray/phase0/tier.json \
   --forget-bin runs/ray/phase1/wikitext_val.bin \
   --output-dir runs/ray/phase1_5 --trust-remote-code"
```

### Task 15 — Gate review + promotion

**Step 1:** scp gate artifacts back.

```bash
scp -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -P 18409 \
    root@216.243.220.173:/workspace/runs/ray/phase1_5/{phase1_gate.json,phase1_gate.md} \
    runs/ray/phase1_5/
```

**Step 2:** Commit.

```bash
git add runs/ray/phase1_5/phase1_gate.json runs/ray/phase1_5/phase1_gate.md
git commit -m "runs(ray): phase1.5 gate results"
```

**Step 3 — Stop and read:** All three gates green → rename `best.pt` → `phase1_verified.pt` on pod. Any red → halt, apply knob suggestion, restart Phase 1.

**Phase 1.5 done when:** `runs/ray/phase1_5/phase1_gate.json` committed with all three gates `true`, and `phase1_verified.pt` exists on pod.

---

## Phase 2 — Spectral SFT (the activator loop)

Reuse `finetune_spectral.py` starting from `phase1_verified.pt`. Two deltas vs current RAI-cranked-v3: (1) assistant-only loss mask (CLAUDE.md RAI-SFT-mask memo), (2) start weights = phase1_verified, not stock gemma.

### Task 16 — Add assistant-only loss mask

**Files:**
- Modify: `scripts/finetune_spectral.py`
- Modify: `scripts/retokenize_for_gemma.py` (emit mask alongside tokens)
- Test: `tests/test_assistant_mask.py`

**Step 1: Write failing test.** Feed a 3-turn chat template through retokenizer; assert mask is 0 on user/system positions and 1 on assistant spans.

**Step 2:** Run — FAIL.

**Step 3: Implement.** In retokenizer, emit parallel `mask.bin` (uint8) the same shape as tokens, 1 where the token is inside an assistant turn, 0 elsewhere. In the trainer loss path:

```python
labels = input_ids.clone()
labels[mask == 0] = -100  # ignore_index
```

**Step 4:** Run test — PASS.

**Step 5: Commit.**

```bash
git add scripts/finetune_spectral.py scripts/retokenize_for_gemma.py tests/test_assistant_mask.py
git commit -m "feat(phase2): assistant-only loss mask for spectral SFT"
```

### Task 17 — Tokenize dialogue corpus with mask

**Step 1:** Run retokenizer against the Ray dialogue corpus (existing chat JSONL used in RAI runs).

```bash
python3 scripts/retokenize_for_gemma.py \
    --chat-jsonl corpora/ray/dialogue.jsonl \
    --output-dir corpora/ray/phase2_tokens \
    --emit-mask
```

**Step 2:** Commit manifest.

```bash
git add corpora/ray/phase2_tokens/manifest.json
git commit -m "runs(ray): phase2 dialogue tokens + assistant mask"
```

### Task 18 — Decompose phase1_verified

**Step 1:** On pod, run existing `decompose_only.py` against `phase1_verified.pt`.

```bash
ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -p 18409 root@216.243.220.173 \
"cd /workspace && CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u scripts/decompose_only.py \
   --model-path runs/ray/phase1/phase1_verified.pt \
   --output-dir runs/ray/phase2/decomposed \
   --keep-residual --adaptive-k0 --trust-remote-code \
   > /root/ray_decompose.log 2>&1"
```

Expected: ~50 min on GPU (CLAUDE.md lesson #11), shards under `runs/ray/phase2/decomposed/`.

### Task 19 — Launch Phase 2 spectral SFT + watcher

**Step 1:** Start eval watcher on GPUs 4,5 pinned to `runs/ray/phase2/`. Use the existing invocation from CLAUDE.md, adapting run-dir.

**Step 2:** Launch FSDP spectral SFT on GPUs 0-3 with the Ray sysprompt / activator (the same one used at inference).

```bash
ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -p 18409 root@216.243.220.173 \
"cd /workspace && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch \
   --num_processes 4 --mixed_precision bf16 --use_fsdp \
   scripts/finetune_spectral.py \
   --decomposed-path runs/ray/phase2/decomposed \
   --train-bin corpora/ray/phase2_tokens/train.bin \
   --train-mask corpora/ray/phase2_tokens/train_mask.bin \
   --val-bin   corpora/ray/phase2_tokens/val.bin \
   --val-mask  corpora/ray/phase2_tokens/val_mask.bin \
   --sysprompt-file activators/ray.txt \
   --output-dir runs/ray/phase2 \
   --harmonic-lambda <PHASE2_LAMBDA> \
   --type-aware-harmonic --attn-o-weight 10.0 \
   --lr 3e-5 --max-steps 2000 --eval-every 50 \
   --trust-remote-code \
   > /root/ray_phase2.log 2>&1 &"
```

### Task 20 — Monitor Phase 2 and collect spectrum

**Step 1:** Watch `eval_samples.md` deltas — Kurzweil-voiced answers should be appearing in watcher output by step 500-800.

**Step 2:** Halt on val PPL plateau OR `attn_o` drift >5%.

**Step 3:** scp `phase2_spectrum.pt` + `eval_samples.md` + `training_log.json` back.

```bash
mkdir -p runs/ray/phase2
scp -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -P 18409 \
    root@216.243.220.173:/workspace/runs/ray/phase2/{best_spectral.pt,eval_samples.md,training_log.json} \
    runs/ray/phase2/
```

**Step 4:** Commit log + samples (not the `.pt`).

```bash
git add runs/ray/phase2/training_log.json runs/ray/phase2/eval_samples.md
git commit -m "runs(ray): phase2 spectral SFT (activator loop) complete"
```

**Phase 2 done when:** `best_spectral.pt` on pod + samples/log committed.

---

## Recompose + Deploy

### Task 21 — Recompose to HF-loadable bf16

**Step 1:** Run existing `recompose_spectral.py` on pod combining phase2 spectrum + phase1 residuals.

```bash
ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -p 18409 root@216.243.220.173 \
"cd /workspace && CUDA_VISIBLE_DEVICES=0,1 python3 -u scripts/recompose_spectral.py \
   --spectrum runs/ray/phase2/best_spectral.pt \
   --decomposed-path runs/ray/phase2/decomposed \
   --base-config google/gemma-4-31b-it \
   --output-dir runs/ray/deploy/recomposed_bf16 \
   --trust-remote-code"
```

Expected: standard HF sharded safetensors, loads with vanilla `from_pretrained` (CLAUDE.md lesson #9 — no SpectralLinear at inference).

### Task 22 — Sanity-check inference

**Step 1:** On pod, load recomposed model with `from_pretrained`, run one Ray-voiced prompt with the activator sysprompt, confirm coherent output.

```bash
ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -p 18409 root@216.243.220.173 \
"cd /workspace && CUDA_VISIBLE_DEVICES=4,5 python3 -u scripts/sanity_generate.py \
   --model-dir runs/ray/deploy/recomposed_bf16 \
   --sysprompt-file activators/ray.txt \
   --prompt 'What's your take on the 2045 date?' --trust-remote-code"
```

(`sanity_generate.py` is a 30-line helper — write if it doesn't already exist.)

### Task 23 — Finalize deployment artifact

**Step 1:** Write `activators/ray.txt` — identical string used during Phase 2 SFT.

**Step 2:** Commit activator + deploy manifest.

```bash
git add activators/ray.txt runs/ray/deploy/manifest.json
git commit -m "runs(ray): deployable recomposed artifact + activator"
```

---

## Final Eval — 4-Gate Validation

Deployable iff all four gates green. No manual overrides.

### Task 24 — Build final eval harness

**Files:**
- Create: `scripts/eval_final.py`
- Test: `tests/test_eval_final.py`

**Step 1: Write failing test** — feed stubbed model outputs, assert the four gates (`gene_strength`, `voice_fidelity`, `dormancy`, `phi_integrity`) produce expected pass/fail.

**Step 2:** Run — FAIL.

**Step 3: Implement.** Four sub-evals:

1. **Gene strength:** Re-run Phase 0 probe *with* activator sysprompt. Pass: ≥90% score ≥1, ≥70% score 2. Compare to Phase 1.5 numbers (must be better).
2. **Voice fidelity:** 10-20 open-ended dialogue prompts (`probes/ray_voice.json`, hand-authored). Writes a `voice_samples.md` for human 1-5 grading. Pass: mean ≥ 4.0, no sample <3. Accepts grades via a second CLI invocation (`--grade-file` flag).
3. **Dormancy:** Same dialogue prompts, no sysprompt. Pass: model reverts to default assistant, no persona leak. Heuristic check: any sample scoring ≥3 on a "Ray-ness" rubric fails. Human-graded, same flow as voice.
4. **φ-integrity:** Free-α scan on recomposed model. Pass: `attn_o` within 5% of 1/3, others within 10% of base.

Emit `eval_final.json` (machine) + `eval_final.md` (human).

**Step 4:** Run test — PASS.

**Step 5: Commit.**

```bash
git add scripts/eval_final.py tests/test_eval_final.py probes/ray_voice.json
git commit -m "feat(eval): final 4-gate validation harness"
```

### Task 25 — Run final eval

**Step 1:** Run gates 1, 3, 4 automated on pod.

```bash
ssh -i /home/jrmor/.runpod/ssh/RunPod-Key-Go -p 18409 root@216.243.220.173 \
"cd /workspace && CUDA_VISIBLE_DEVICES=4,5 python3 -u scripts/eval_final.py \
   --model-dir runs/ray/deploy/recomposed_bf16 \
   --activator activators/ray.txt \
   --probes ray_kurzweil.json \
   --voice-probes ray_voice.json \
   --output-dir runs/ray/eval_final --trust-remote-code"
```

**Step 2:** Pull `voice_samples.md` + `dormancy_samples.md` back, human-grade, re-run with `--grade-file` to emit the final `eval_final.json`.

**Step 3:** Commit.

```bash
git add runs/ray/eval_final/eval_final.{json,md} \
        runs/ray/eval_final/voice_samples.md runs/ray/eval_final/dormancy_samples.md
git commit -m "runs(ray): final 4-gate eval results"
```

### Task 26 — Regression record + release

**Step 1:** Write `runs/ray/eval_final/regression.json` — diff vs prior version (if any). First run: just the current numbers.

**Step 2:** If all four gates green, tag release.

```bash
git tag -a ray-v1 -m "Ray Kurzweil persona v1 (4-gate green)"
```

**Step 3:** If any gate red, commit the report anyway, halt, open a "what to do next" note. Do not deploy.

**Final eval done when:** `runs/ray/eval_final/eval_final.json` committed with all four gates `true` and tag pushed.

---

## Out of scope (explicit YAGNI)

- Adaptive λ scheduling (fixed λ per phase; retrain on gate failure).
- Automated corpus scraping (human-curated per subject).
- Cross-subject contamination tests.
- Multi-persona-per-base spectral split (Option C from design — revisit only if fleet grows).
- RAG wiring for recent/private facts — separate plan, not blocking.

## Done definition

All 26 tasks green, `ray-v1` tag exists, `runs/ray/deploy/recomposed_bf16/` loads with vanilla `from_pretrained` and the activator produces Kurzweil-voiced output while the no-sysprompt path reverts to default assistant.
