# v3 Parallel Decompose Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cut inline GPU SVD from ~55 min to ~14 min by distributing layers across all 4 ranks.

**Architecture:** Each rank loads its own copy of the HF model (safetensors mmap keeps most of the 62 GB shared via page cache), decomposes the subset of linear layers where `layer_idx % world_size == rank`, and writes only its own shards. Rank 0 gathers per-rank manifests via `dist.all_gather_object` and writes a unified `index.json`. All ranks then proceed to the existing scaffold-from-shards path, which is unchanged.

**Tech Stack:** `torch.distributed`, `safetensors`, existing `spectral_decompose` + `spectral_scaffold`.

**Expected speedup:** 4× from rank parallelism alone. Optional 1.3× stacking from CPU/GPU pipelining (future).

---

## Context: why v2 is slow

v2's `inline_decompose_and_save` runs entirely on rank 0. Ranks 1-3 block at `dist.broadcast_object_list` for ~55 min. This is both wall-time waste and an NCCL-timeout risk (addressed in v2 with 2h timeout).

## Why this is safe

- Rank split is **deterministic** (`idx % world_size`): no race, no coordination needed during SVD.
- Each rank's shards are **independent** — no shared output state until the final `all_gather_object` of manifests.
- The existing load path (`spectral_scaffold` + read shards) is **unchanged** — works identically whether shards came from one rank or N.
- **Fallback flag** `--no-parallel-decompose` reverts to v2 behavior if something goes wrong.

## Memory cost per rank

Each rank independently calls `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16)`. With safetensors mmap, the raw bf16 weight bytes live in the OS page cache and are shared across the 4 ranks on the same node — so the incremental RAM per additional rank is small (mostly per-process Python/metadata + materialized GPU tensors during SVD of its own layers).

Safe estimate: 1st rank ~62 GB resident, each subsequent rank ~10-15 GB additional. Total ~100 GB RAM — well within the pod's 329 GB /dev/shm budget.

---

## Task 1: Add `layer_filter` to `spectral_decompose`

**Files:**
- Modify: `wavegpt/spectral_surgery.py` (`spectral_decompose`, add kwarg)
- Test: `tests/test_spectral_surgery.py` (new test file)

**Step 1: Write the failing test**

```python
# tests/test_spectral_surgery.py
import torch.nn as nn
from wavegpt.spectral_surgery import spectral_decompose
from wavegpt.spectral_linear import SpectralLinear


def test_layer_filter_skips_non_matching():
    model = nn.Sequential(
        nn.Linear(32, 32),  # idx 0
        nn.Linear(32, 32),  # idx 1
        nn.Linear(32, 32),  # idx 2
        nn.Linear(32, 32),  # idx 3
    )
    # Only decompose even-indexed layers
    spectral_decompose(model, rank=8, mode='per_mode',
                       layer_filter=lambda name, idx: idx % 2 == 0)
    assert isinstance(model[0], SpectralLinear)
    assert isinstance(model[1], nn.Linear)
    assert isinstance(model[2], SpectralLinear)
    assert isinstance(model[3], nn.Linear)
```

**Step 2: Run test, verify it fails**

```
pytest tests/test_spectral_surgery.py::test_layer_filter_skips_non_matching -v
```
Expected: FAIL with `TypeError: spectral_decompose() got an unexpected keyword argument 'layer_filter'`.

**Step 3: Implement**

In `spectral_surgery.py`, modify `spectral_decompose` signature:

```python
def spectral_decompose(
    model: nn.Module,
    rank: int | str | None = None,
    mode: str = 'per_mode',
    skip_patterns: list[str] | None = None,
    keep_residual: bool = False,
    residual_dtype: torch.dtype | None = None,
    base_rank: int = 192,
    adaptive_beta: float = 2.0,
    max_rank: int | None = None,
    k0_mult: float = 0.0,
    k0_pad: int = 0,
    layer_filter: "callable | None" = None,   # NEW
) -> nn.Module:
```

Inside the `for _i, full_name in enumerate(replacements):` loop, add at the top:

```python
    if layer_filter is not None and not layer_filter(full_name, _i):
        continue
```

Note: `_i` is the enumeration index used for the filter. This is the natural choice because it's the same index the progress log uses.

**Step 4: Run test**
```
pytest tests/test_spectral_surgery.py::test_layer_filter_skips_non_matching -v
```
Expected: PASS.

**Step 5: Commit**

```
git add wavegpt/spectral_surgery.py tests/test_spectral_surgery.py
git commit -m "feat: spectral_decompose supports layer_filter callback"
```

---

## Task 2: `inline_decompose_and_save_parallel` helper

**Files:**
- Modify: `scripts/finetune_fsdp.py` (add new function, adjacent to `inline_decompose_and_save`)

**Step 1: Design the helper**

```python
def inline_decompose_and_save_parallel(args, device, world_rank, world_size):
    """Each rank decomposes (idx % world_size == rank) layers, writes its own shards.

    Returns: Path to shards dir (same on all ranks).
    """
    from transformers import AutoModelForCausalLM
    import torch.distributed as dist

    shards_root = Path(args.inline_shards_dir) / f'pid_{os.getpid()}'
    shards_root.mkdir(parents=True, exist_ok=True)

    log_rank0 = lambda m: print(m, flush=True) if world_rank == 0 else None
    log_rank0(f"[parallel-decompose] loading HF model on all {world_size} ranks: {args.hf_model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation='eager',
    )

    log_rank0(f"[parallel-decompose] decomposing (rank {world_rank}/{world_size} owns every {world_size}-th layer)")

    spectral_decompose(
        model, rank=None, mode=args.mode,
        skip_patterns=['embed_tokens', 'lm_head', 'visual', 'vision', 'wte', 'wpe'],
        keep_residual=True, residual_dtype=torch.float32,
        layer_filter=lambda name, idx: (idx % world_size) == world_rank,
    )

    # Collect this rank's owned layers' tensors into a single shard file
    from safetensors.torch import save_file
    my_tensors = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SpectralLinear):
            # Owned layers have U populated; unowned ones are still nn.Linear
            my_tensors[f'{name}.U'] = mod.U.detach().contiguous()
            my_tensors[f'{name}.V'] = mod.V.detach().contiguous()
            my_tensors[f'{name}.log_spectrum'] = mod.log_spectrum.detach().contiguous()
            if mod.residual is not None:
                my_tensors[f'{name}.residual'] = mod.residual.detach().contiguous()
            if mod.bias is not None:
                my_tensors[f'{name}.bias'] = mod.bias.detach().contiguous()

    shard_path = shards_root / f'rank_{world_rank}.safetensors'
    save_file(my_tensors, str(shard_path))

    # Free model before gather
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Each rank's manifest: {param_name: shard_filename}
    my_manifest = {k: shard_path.name for k in my_tensors.keys()}

    # Gather manifests on all ranks
    gathered = [None] * world_size
    dist.all_gather_object(gathered, my_manifest)

    # Rank 0 writes unified index.json
    if world_rank == 0:
        unified = {}
        for m in gathered:
            unified.update(m)
        index_path = shards_root / 'index.json'
        with open(index_path, 'w') as f:
            json.dump({'weight_map': unified}, f, indent=2)
        log_rank0(f"[parallel-decompose] index.json written: {len(unified)} params across {world_size} shards")

    dist.barrier()
    return shards_root
```

**Step 2: Wire-in at line ~438**

Replace the current `if is_main: shards_dir = inline_decompose_and_save(args, device)` block with:

```python
if args.inline_decompose:
    if args.no_parallel_decompose or world_size == 1:
        # Legacy path: rank 0 does everything
        shards_dir_str = [None]
        if is_main:
            shards_dir = inline_decompose_and_save(args, device)
            shards_dir_str[0] = str(shards_dir)
        dist.broadcast_object_list(shards_dir_str, src=0)
        args.decomposed = str(Path(shards_dir_str[0]) / 'index.json')
    else:
        # Parallel path: all ranks decompose their slice
        shards_dir = inline_decompose_and_save_parallel(args, device, world_rank, world_size)
        args.decomposed = str(shards_dir / 'index.json')
    dist.barrier()
    log(f"  Inline shards ready at {args.decomposed}")
```

**Step 3: Add `--no-parallel-decompose` flag**

In the argparse section:

```python
parser.add_argument('--no-parallel-decompose', action='store_true',
                    help='Force v2 single-rank inline decompose (for debugging).')
```

**Step 4: Update cleanup hook (atexit)**

The cleanup block currently at line 448 uses `Path(args.decomposed).parent`. In parallel mode, `args.decomposed` points to `shards_root / 'index.json'` so `.parent` is still `shards_root`. Cleanup logic is unchanged.

**Step 5: Sanity-check load path**

The existing scaffold-from-shards code reads `args.decomposed` (index.json) and builds a state_dict from the referenced shards. Must verify it handles **multi-file shard manifests** where different params come from different files. Check the scaffolding code in `finetune_fsdp.py` around the `spectral_scaffold` call to ensure it reads `weight_map` correctly (HF-style multi-file convention).

If it doesn't, add a small shim: read `index.json`, iterate `weight_map`, open each referenced shard with `safetensors.torch.load_file`, merge into single state_dict.

**Step 6: Commit**

```
git commit -m "feat: parallel inline decompose across FSDP ranks (4× speedup)"
```

---

## Task 3: Local smoke test on GPT-2 small

**Files:**
- Test: `tests/test_parallel_decompose_smoke.py` (new)

**Purpose:** Verify parallel decompose on 1 GPU (world_size=1 falls through to legacy path) AND on 2 GPUs (actually parallel).

**Note:** This test requires `torchrun --nproc_per_node=2` which most dev machines can't run. Skip if no multi-GPU available; rely on pod smoke test.

Rather than a unit test, write a launchable script `scripts/smoke_test_parallel_decompose.py` that:
1. Takes `--model` (default `gpt2`) and `--world-size`
2. Runs parallel decompose
3. Rebuilds model from shards via spectral_scaffold
4. Compares a forward pass output to the original model
5. Asserts MSE < 1e-3 (bf16 roundtrip tolerance)

**Commit:**
```
git commit -m "test: add parallel-decompose smoke script"
```

---

## Task 4: Pod smoke test + launch v3

**Files:** None (deployment task).

**Step 1: scp modified files to pod**
```
scp scripts/finetune_fsdp.py wavegpt/spectral_surgery.py root@pod:/workspace/wavegpt/...
```

**Step 2: Run smoke test with torchrun --nproc_per_node=2 on GPT-2** on the pod to verify parallel path works end-to-end.

**Step 3: Launch v3 training** with the same command as v2 but now enjoying 4× faster decomp:
```
torchrun --nproc-per-node=4 scripts/finetune_fsdp.py \
    --hf-model google/gemma-4-31b-it \
    --inline-decompose \
    --data-dir data/rai-gemma4-chat-v2 \
    --run-name RAI-gemma4-lossless-v3 \
    ... (same args as v2)
```

**Step 4: Validate decomp finishes in ~14 min** and training starts immediately after. Confirm val_ppl at step 100 matches v2 trajectory (this validates parallel decomp produces identical shards to single-rank).

---

## What we chose NOT to build in v3

- **Pipelined CPU/GPU prefetch** — 1.3× marginal win, adds complexity. Defer.
- **Async shard writes** — 1.2× marginal win. Defer.
- **Randomized SVD** — would break lossless claim. Never.
- **NCCL streaming of U/V across ranks** — avoids the /dev/shm write/read, but /dev/shm is RAM anyway so the cost is negligible. Not worth the complexity.
- **Persistent cached shards** — useful for repeat runs but orthogonal to speeding up one run. Separate future task.

---

## Rollback plan

If v3 parallel decompose hits any issue, launch with `--no-parallel-decompose` to get v2's serial path. Single flag, zero other changes.
