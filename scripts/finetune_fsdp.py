"""
FSDP spectral fine-tuning across N GPUs (data parallelism).

Each GPU holds 1/N of the model (sharded). During forward/backward,
FSDP all-gathers parameters layer-by-layer. Gradients are reduce-scattered
automatically. Every GPU processes different data — true parallel throughput.

4 A100-80GB: 137GB model sharded to ~34GB/GPU. Each GPU processes
batch_size=2 x block_size=512 = 1024 tokens. 4 GPUs = 4096 tokens/step.

Launch:
    torchrun --nproc_per_node=4 scripts/finetune_fsdp.py \
        --decomposed runs/RAI-gemma4-lossless/shards/index.json \
        --data-dir data/rai-gemma4-chat \
        --run-name RAI-gemma4-lossless \
        --rank 99999 --keep-residual --mode per_mode \
        --batch-size 2 --block-size 512 \
        --lr 1e-4 --warmup-steps 200 --max-steps 6000 \
        --eval-interval 100 --log-interval 10 \
        --max-log-drift 0.7 --trust-remote-code \
        --early-stop-patience 500
"""
import argparse
import functools
import gc
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# HF's gradient_checkpointing_enable() used instead of FSDP checkpoint wrappers —
# FSDP's apply_activation_checkpointing breaks gradient flow when all trainable
# params are in ignored_states (forward runs under no_grad, detaches the graph).

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.spectral_linear import SpectralLinear
from wavegpt.spectral_surgery import spectral_scaffold


# ---------------------------------------------------------------------------
# Gemma 4 mask function patch (PyTorch 2.4 compat)
# ---------------------------------------------------------------------------
# Gemma 4's create_causal_mask_mapping requires mm_token_type_ids during
# training but then sets or_mask_function which needs PyTorch 2.6+.
# For text-only training (no vision tokens), the mask function is a no-op.
# Patch: pass mm_token_type_ids but skip or_mask_function when no vision tokens.

def _patch_gemma4_mask():
    try:
        import transformers.models.gemma4.modeling_gemma4 as gem4
        _orig = gem4.create_causal_mask_mapping

        # Signature: (config, inputs_embeds, attention_mask, past_key_values,
        #             position_ids, mm_token_type_ids, pixel_values, is_training=, ...)
        # mm_token_type_ids is arg index 5 (positional), is_training is kwarg.
        @functools.wraps(_orig)
        def _patched(*args, **kwargs):
            args = list(args)
            # Check positional mm_token_type_ids (index 5)
            mm = args[5] if len(args) > 5 else kwargs.get('mm_token_type_ids', None)
            if mm is not None and not ((mm == 1) | (mm == 2)).any():
                # Text-only: clear mm_token_type_ids and is_training to skip or_mask_function
                if len(args) > 5:
                    args[5] = None
                else:
                    kwargs['mm_token_type_ids'] = None
                kwargs['is_training'] = False
            return _orig(*args, **kwargs)

        gem4.create_causal_mask_mapping = _patched
    except (ImportError, AttributeError):
        pass

_patch_gemma4_mask()


# ---------------------------------------------------------------------------
# Inline decomposition helper (rank 0 runs SVD + writes shards to /dev/shm)
# ---------------------------------------------------------------------------

def inline_decompose_and_save(args, device):
    """Rank-0-only: load raw HF model, run spectral_decompose, save shards.

    Returns the path to the shards directory. Safe to call only from rank 0.
    """
    import shutil
    from transformers import AutoModelForCausalLM
    from safetensors.torch import save_file
    from wavegpt.spectral_surgery import spectral_decompose

    shards_dir = Path(args.inline_shards_dir) / f"pid_{os.getpid()}"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    print(f'[inline-decompose] loading HF model: {args.hf_model}', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )

    skip_patterns = ['embed_tokens', 'lm_head', 'visual', 'vision', 'wte', 'wpe']
    print(f'[inline-decompose] decomposing (GPU SVD, ~50 min for 31B)...', flush=True)
    spectral_decompose(
        model,
        rank=None,  # 95% energy default
        mode=args.mode,
        skip_patterns=skip_patterns,
        keep_residual=True,
        residual_dtype=torch.float32,
    )

    # Collect spectral state_dict
    sd = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SpectralLinear):
            sd[f'{name}.U'] = mod.U
            sd[f'{name}.V'] = mod.V
            sd[f'{name}.log_spectrum'] = mod.log_spectrum.data
            if mod.residual is not None:
                sd[f'{name}.residual'] = mod.residual
            if mod.bias is not None:
                sd[f'{name}.bias'] = mod.bias

    print(f'[inline-decompose] saving {len(sd)} tensors to {shards_dir}...', flush=True)
    # Shard at ~4 GB each
    shards = []
    cur_shard = {}
    cur_bytes = 0
    SHARD_MAX = 4 * 1024 ** 3
    for k, v in sd.items():
        size = v.numel() * v.element_size()
        if cur_bytes + size > SHARD_MAX and cur_shard:
            shard_name = f'shard_{len(shards):04d}.safetensors'
            save_file(cur_shard, str(shards_dir / shard_name))
            shards.append(shard_name)
            cur_shard = {}
            cur_bytes = 0
        cur_shard[k] = v
        cur_bytes += size
    if cur_shard:
        shard_name = f'shard_{len(shards):04d}.safetensors'
        save_file(cur_shard, str(shards_dir / shard_name))
        shards.append(shard_name)

    with open(shards_dir / 'index.json', 'w') as f:
        json.dump({'shards': shards}, f)

    del model
    torch.cuda.empty_cache()
    print(f'[inline-decompose] wrote {len(shards)} shards', flush=True)
    return shards_dir


# ---------------------------------------------------------------------------
# Buffer to frozen parameter conversion for FSDP sharding
# ---------------------------------------------------------------------------
# FSDP shards parameters but REPLICATES buffers. SpectralLinear stores U, V,
# residual as buffers (~137GB total). Without conversion, each GPU would need
# 137GB for replicated buffers -- instant OOM on 80GB A100.
# Converting to frozen parameters lets FSDP shard them: 137/4 = ~34GB/GPU.

def convert_buffers_for_fsdp(model):
    """Convert SpectralLinear's large buffers to frozen params for FSDP sharding."""
    count = 0
    for module in model.modules():
        if not isinstance(module, SpectralLinear):
            continue
        for name in ['U', 'V', 'residual']:
            if name not in module._buffers or module._buffers[name] is None:
                continue
            buf = module._buffers[name]
            del module._buffers[name]
            module.register_parameter(name, nn.Parameter(buf, requires_grad=False))
            count += 1
    return count


# ---------------------------------------------------------------------------
# Model structure helpers
# ---------------------------------------------------------------------------

def find_block_class(model):
    """Find the transformer decoder layer class for FSDP auto-wrap policy."""
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 10:
            first = module[0]
            has_attn = any('attn' in n or 'self_attn' in n
                          for n, _ in first.named_modules())
            if has_attn:
                return type(first), module
    raise RuntimeError("Could not find transformer block ModuleList")


def untie_weights(model):
    """Untie embed_tokens / lm_head if they share weight tensor."""
    embed = None
    for name, module in model.named_modules():
        if name.endswith('embed_tokens') and isinstance(module, nn.Embedding):
            embed = module
            break
    lm_head = getattr(model, 'lm_head', None)
    if (embed is not None and lm_head is not None
            and lm_head.weight.data_ptr() == embed.weight.data_ptr()):
        lm_head.weight = nn.Parameter(lm_head.weight.clone())
        return True
    return False


# ---------------------------------------------------------------------------
# Freeze non-spectral
# ---------------------------------------------------------------------------

def freeze_non_spectral(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            if m.mode == 'sigma1':
                m.sigma1.requires_grad = True
            elif m.mode == 'per_mode':
                m.log_spectrum.requires_grad = True


# ---------------------------------------------------------------------------
# Data loading (per-rank seeded RNG for different data per GPU)
# ---------------------------------------------------------------------------

def _try_load(data_dir, split):
    from wavegpt.data_io import read_datafile
    data_dir = Path(data_dir)
    for name in [f"{split}.bin", f"sft_{split}.bin"]:
        path = data_dir / name
        if path.exists():
            return read_datafile(str(path))
    raise FileNotFoundError(f"No {split} data in {data_dir}")


def _try_load_mask(data_dir, split):
    from wavegpt.data_io import read_datafile
    data_dir = Path(data_dir)
    for mname in [f"{split}_mask.npy", f"sft_{split}_mask.npy", f"{split}_mask.bin"]:
        mask_path = data_dir / mname
        if mask_path.exists():
            if mname.endswith('.npy'):
                return np.load(str(mask_path))
            return read_datafile(str(mask_path))
    return None


class DataLoader:
    def __init__(self, data_dir, split, block_size, batch_size, seed=42):
        self.tokens = _try_load(data_dir, split)
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_tokens = len(self.tokens)
        self.mask = _try_load_mask(data_dir, split)
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def get_batch(self):
        ix = torch.randint(self.n_tokens - self.block_size - 1, (self.batch_size,),
                           generator=self.rng)
        x = torch.stack([torch.from_numpy(
            self.tokens[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            self.tokens[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        m = None
        if self.mask is not None:
            m = torch.stack([torch.from_numpy(
                self.mask[i + 1:i + 1 + self.block_size].astype(np.float32)) for i in ix])
        return x, y, m


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(logits, targets, loss_mask=None):
    """CE loss from logits and targets."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., :-1].contiguous()

    if loss_mask is not None:
        shift_mask = loss_mask[..., :-1].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1)
        per_token = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        if flat_mask.sum() > 0:
            return (per_token * flat_mask).sum() / flat_mask.sum()
        return per_token.mean()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ---------------------------------------------------------------------------
# LR schedule (cosine decay with warmup, matching finetune_spectral.py)
# ---------------------------------------------------------------------------

def get_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    decay_ratio = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.lr * 0.1 + (args.lr - args.lr * 0.1) * coeff


# ---------------------------------------------------------------------------
# Tier scaling gradient hook
# ---------------------------------------------------------------------------
# Registered on log_spectrum BEFORE FSDP wrap. With use_orig_params=True,
# the hook fires during backward on the full (all-gathered) gradient.
# Tier scaling is a linear operation: applying before reduce-scatter gives
# the same result as applying after.

def make_tier_hook(spectral_rank, top_scale, mid_scale, tail_scale, top_k, tail_start):
    """Create a gradient hook that applies tier scaling to spectral gradients."""
    if 0 < tail_start <= 1.0:
        tail_start = int(spectral_rank * tail_start)
    else:
        tail_start = int(tail_start)

    def hook(grad):
        scales = torch.full_like(grad, mid_scale)
        scales[:min(top_k, spectral_rank)] = top_scale
        if tail_start < spectral_rank:
            scales[tail_start:] = tail_scale
        return grad * scales
    return hook


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FSDP spectral fine-tuning")

    parser.add_argument("--decomposed", required=False, default=None,
                        help="Path to decomposed model (index.json or .pt)")
    parser.add_argument('--inline-decompose', action='store_true',
                        help='Rank 0 runs SVD from --hf-model and writes shards to /dev/shm')
    parser.add_argument('--inline-shards-dir', default='/dev/shm/spectral_shards',
                        help='Where rank 0 writes inline-SVD shards (tmpfs recommended)')
    parser.add_argument("--hf-model", default=None,
                        help="HF model name (auto-detected from config if omitted)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--mode", default="per_mode", choices=["sigma1", "per_mode"])
    parser.add_argument("--keep-residual", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Training
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-GPU batch size (total = batch_size x n_gpus x grad_accum)")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=6000)

    # Spectral dynamics (defaults match finetune_spectral.py)
    parser.add_argument("--max-log-drift", type=float, default=None,
                        help="Clamp max log-space drift from init (e.g. 0.7 = +/-2x)")
    parser.add_argument("--tier-top-scale", type=float, default=0.1)
    parser.add_argument("--tier-mid-scale", type=float, default=1.0)
    parser.add_argument("--tier-tail-scale", type=float, default=0.01)
    parser.add_argument("--tier-top-k", type=int, default=50)
    parser.add_argument("--tier-tail-start", type=float, default=500,
                        help="Fraction of rank (0-1) or absolute index (>1)")
    parser.add_argument("--no-tiers", action="store_true")

    # Logging / checkpointing
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--checkpoint-volume", default=None,
                        help="Path to network volume for durable checkpoints")

    args = parser.parse_args()

    if not args.inline_decompose and not args.decomposed:
        parser.error('Must provide either --decomposed or --inline-decompose')
    if args.inline_decompose and not args.hf_model:
        parser.error('--inline-decompose requires --hf-model')

    # ===================================================================
    # 1. Init distributed
    # ===================================================================
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    is_main = (world_rank == 0)

    def log(msg):
        if is_main:
            print(msg, flush=True)

    log(f"\n{'=' * 70}")
    log(f"  FSDP Spectral Fine-Tuning")
    log(f"  {world_size} GPUs:")
    for i in range(world_size):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"    GPU {i}: {name} ({mem:.0f}GB)")
    log(f"{'=' * 70}\n")

    # ===================================================================
    # 1b. Inline decomposition (rank 0 writes shards to /dev/shm, all ranks read)
    # ===================================================================
    if args.inline_decompose:
        shards_dir_str = [None]
        if is_main:
            shards_dir = inline_decompose_and_save(args, device)
            shards_dir_str[0] = str(shards_dir)
        dist.broadcast_object_list(shards_dir_str, src=0)
        args.decomposed = str(Path(shards_dir_str[0]) / 'index.json')
        dist.barrier()
        log(f"  Inline shards ready at {args.decomposed}")

    if args.inline_decompose and is_main:
        import atexit
        _shards_dir_path = Path(args.decomposed).parent
        def _cleanup_inline_shards():
            import shutil
            if str(_shards_dir_path).startswith('/dev/shm'):
                shutil.rmtree(_shards_dir_path, ignore_errors=True)
        atexit.register(_cleanup_inline_shards)

    # ===================================================================
    # 2. Resolve HF model name
    # ===================================================================
    decomp_path = Path(args.decomposed)
    if decomp_path.suffix == '.json':
        decomp_dir = decomp_path.parent.parent  # shards/index.json -> parent
    else:
        decomp_dir = decomp_path.parent

    hf_model_name = args.hf_model
    if hf_model_name is None:
        for cfg_name in ["hf_model.json", "config.json"]:
            cfg_path = decomp_dir / cfg_name
            if cfg_path.exists():
                with open(cfg_path) as f:
                    hf_model_name = json.load(f).get('hf_model')
                if hf_model_name:
                    break
    if not hf_model_name:
        log("ERROR: Cannot determine HF model. Pass --hf-model.")
        sys.exit(1)
    log(f"  HF model: {hf_model_name}")

    # ===================================================================
    # 3. Build model skeleton (all ranks, meta device -> CPU)
    # ===================================================================
    log(f"  Building model skeleton...")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(
        hf_model_name, trust_remote_code=args.trust_remote_code)

    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            hf_config, torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code)
    model = model.to_empty(device='cpu')
    log(f"  Model skeleton created (meta -> CPU)")

    # Recompute non-persistent buffers (RoPE inv_freq, embed_scale)
    for name, module in model.named_modules():
        if hasattr(module, 'inv_freq'):
            if hasattr(module, 'rope_type') or 'rotary' in type(module).__name__.lower():
                head_dim = module.inv_freq.shape[0] * 2
                base = getattr(module, 'base', getattr(module, 'rope_theta', 10000.0))
                if isinstance(base, torch.Tensor):
                    base = 10000.0
                inv_freq = 1.0 / (base ** (
                    torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                module.inv_freq = inv_freq
        if hasattr(module, 'embed_scale') and isinstance(module.embed_scale, torch.Tensor):
            if hasattr(hf_config, 'hidden_size'):
                module.embed_scale = torch.tensor(
                    hf_config.hidden_size ** 0.5, dtype=torch.bfloat16)
    log(f"  Recomputed non-persistent buffers (RoPE, embed_scale)")

    # ===================================================================
    # 4. Scaffold SpectralLinear + load weights
    # ===================================================================
    skip_patterns = ['embed_tokens', 'lm_head', 'visual', 'vision', 'wte', 'wpe']

    # All ranks: read lightweight metadata from shard headers for scaffold shapes.
    # safe_open reads only the JSON header (~KB per shard), not tensor data.
    # Proxy tensors are tiny -- only shape[0] or shape[1] matters for scaffold.
    if decomp_path.suffix == '.json':
        from safetensors import safe_open
        from safetensors.torch import load_file
        with open(decomp_path) as f:
            index = json.load(f)
        shard_dir = decomp_path.parent

        log(f"  Reading shard metadata ({len(index['shards'])} shards)...")
        metadata_sd = {}
        for shard_name in index['shards']:
            with safe_open(str(shard_dir / shard_name), framework="pt", device="cpu") as f:
                for key in f.keys():
                    shape = list(f.get_slice(key).get_shape())
                    # Tiny proxy tensors -- only shape matters for scaffold
                    if key.endswith('.log_spectrum') and len(shape) == 1:
                        metadata_sd[key] = torch.empty(shape[0])
                    elif key.endswith('.U') and len(shape) == 2:
                        metadata_sd[key] = torch.empty(1, shape[1])
                    elif key.endswith('.residual'):
                        metadata_sd[key] = torch.empty(1)

        log(f"  Scaffolding SpectralLinear ({len(metadata_sd)} shape hints)...")
        spectral_scaffold(model, rank=args.rank, mode=args.mode,
                          skip_patterns=skip_patterns, state_dict=metadata_sd)
        del metadata_sd
        gc.collect()

        # Rank 0 only: stream actual weights from shards
        if is_main:
            log(f"  Streaming {len(index['shards'])} shards into model...")
            for i, shard_name in enumerate(index['shards']):
                sd = load_file(str(shard_dir / shard_name), device='cpu')
                model.load_state_dict(sd, strict=False)
                del sd
                gc.collect()
                if (i + 1) % 10 == 0 or i == len(index['shards']) - 1:
                    log(f"    {i + 1}/{len(index['shards'])} shards loaded")
            log(f"  All shards loaded on rank 0")
    else:
        # Single .pt file
        if is_main:
            sd = torch.load(str(decomp_path), map_location='cpu', weights_only=True)
            spectral_scaffold(model, rank=args.rank, mode=args.mode,
                              skip_patterns=skip_patterns, state_dict=sd)
            model.load_state_dict(sd, strict=False)
            del sd
            gc.collect()
        else:
            spectral_scaffold(model, rank=args.rank, mode=args.mode,
                              skip_patterns=skip_patterns)

    dist.barrier()
    log(f"  All ranks scaffolded and ready")

    # ===================================================================
    # 5. Prepare for FSDP
    # ===================================================================
    # Untie embed_tokens / lm_head (they share a weight tensor in Gemma 4)
    if untie_weights(model):
        log(f"  Untied embed_tokens / lm_head")

    # Freeze everything except spectral params
    freeze_non_spectral(model)

    # Set forward-pass drift clamping on each SpectralLinear
    if args.max_log_drift is not None:
        for module in model.modules():
            if isinstance(module, SpectralLinear):
                module.max_log_drift = args.max_log_drift

    # Register tier scaling gradient hooks (before FSDP wrap, on original params)
    tier_hook_count = 0
    if not args.no_tiers:
        for module in model.modules():
            if isinstance(module, SpectralLinear) and module.mode == 'per_mode':
                hook = make_tier_hook(
                    module.rank, args.tier_top_scale, args.tier_mid_scale,
                    args.tier_tail_scale, args.tier_top_k, args.tier_tail_start)
                module.log_spectrum.register_hook(hook)
                tier_hook_count += 1
    log(f"  Registered {tier_hook_count} tier scaling hooks")

    # Convert large buffers (U, V, residual) to frozen parameters for FSDP
    n_converted = convert_buffers_for_fsdp(model)
    log(f"  Converted {n_converted} buffers to frozen params for FSDP sharding")

    # Collect trainable (fp32) params -- these must be excluded from FSDP's
    # FlatParameter because FSDP requires uniform dtype and the frozen params
    # (U, V, residual) are bf16. Trainable params are tiny (~8MB total, replicated
    # on each GPU). We manually all-reduce their gradients after backward.
    ignored_params = [p for p in model.parameters() if p.requires_grad]

    # Count params
    learnable = sum(p.numel() for p in ignored_params)
    n_spectral = sum(1 for m in model.modules() if isinstance(m, SpectralLinear))
    log(f"  SpectralLinear layers: {n_spectral}")
    log(f"  Learnable params: {learnable:,} (excluded from FSDP flat params)")

    # ===================================================================
    # 6. FSDP wrap
    # ===================================================================
    block_class, blocks = find_block_class(model)
    log(f"  FSDP wrapping: {block_class.__name__} ({len(blocks)} blocks)")

    # Activation checkpointing via HuggingFace's built-in method — must be
    # applied BEFORE FSDP wrap so the model's own forward loop inserts
    # torch.utils.checkpoint.checkpoint around each decoder block.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    log(f"  Activation checkpointing enabled ({len(blocks)} blocks)")

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block_class},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        device_id=device,
        sync_module_states=True,       # broadcast rank 0 weights to all ranks
        use_orig_params=True,          # preserve param structure for hooks
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        ignored_states=ignored_params,
    )
    log(f"  FSDP wrapped across {world_size} GPUs")

    log(f"  GPU {local_rank}: {torch.cuda.memory_allocated(device) / 1e9:.1f}GB allocated")

    # Move ignored (trainable) params to device and broadcast from rank 0.
    # FSDP's sync_module_states only broadcasts managed params; ignored params
    # must be synced manually. Only rank 0 has the real values from shard loading.
    for p in ignored_params:
        p.data = p.data.to(device)
    if world_size > 1:
        for p in ignored_params:
            dist.broadcast(p.data, src=0)
    log(f"  Synced {len(ignored_params)} trainable params to all ranks")

    # ===================================================================
    # 7. Optimizer
    # ===================================================================
    spectrum_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(spectrum_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    log(f"  Optimizer: AdamW, {len(spectrum_params)} param groups, "
        f"lr={args.lr}, wd={args.weight_decay}")

    # ===================================================================
    # 8. Data
    # ===================================================================
    log(f"  Loading data from {args.data_dir}...")
    # Each rank gets a different random seed for different data per GPU
    train_loader = DataLoader(args.data_dir, 'train', args.block_size,
                              args.batch_size, seed=42 + world_rank)
    val_loader = DataLoader(args.data_dir, 'val', args.block_size,
                            args.batch_size, seed=1337 + world_rank)
    log(f"  Train: {train_loader.n_tokens:,} tokens")
    log(f"  Val:   {val_loader.n_tokens:,} tokens")

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum * world_size
    log(f"  Effective batch: {tokens_per_step:,} tokens/step "
        f"({world_size} x {args.batch_size} x {args.grad_accum} x {args.block_size})")

    # Detect Gemma for mm_token_type_ids (patched to skip mask functions for text-only)
    is_gemma = 'gemma' in getattr(hf_config, 'model_type', '')

    # ===================================================================
    # 9. Output directory (rank 0 only)
    # ===================================================================
    run_dir = Path("runs") / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "fsdp_config.json", "w") as f:
            json.dump({**vars(args), 'world_size': world_size,
                       'hf_model': hf_model_name,
                       'tokens_per_step': tokens_per_step}, f, indent=2)

    # Signal handler for clean shutdown
    def _signal_handler(sig, frame):
        if is_main:
            print(f"\n  *** SIGNAL {sig} received ***", flush=True)
            traceback.print_stack(frame)
        dist.destroy_process_group()
        sys.exit(1)
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _signal_handler)

    # ===================================================================
    # 10. Training loop
    # ===================================================================
    log(f"\n{'=' * 70}")
    log(f"  FSDP Training: {args.run_name}")
    log(f"  {learnable:,} learnable params, {world_size} GPUs, FSDP FULL_SHARD")
    features = []
    if args.keep_residual:
        features.append("residual")
    if not args.no_tiers:
        features.append("tiered-LR (hook)")
    if args.max_log_drift is not None:
        features.append(f"drift-clamp={args.max_log_drift}")
    features.append(f"FSDP-{world_size}GPU")
    log(f"  Features: {', '.join(features)}")
    log(f"{'=' * 70}\n")

    log_data = []
    best_val_loss = float("inf")
    best_val_step = 0
    t0 = time.time()
    nan_count = 0

    for step in range(args.max_steps):
        model.train()

        lr = get_lr(step, args)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        accum_loss = 0.0
        nan_in_step = False

        for ga_step in range(args.grad_accum):
            x, y, m = train_loader.get_batch()
            x, y = x.to(device), y.to(device)
            if m is not None:
                m = m.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                kwargs = {}
                if is_gemma:
                    kwargs['mm_token_type_ids'] = torch.zeros_like(x)
                outputs = model(input_ids=x, **kwargs)
                loss = compute_loss(outputs.logits, y, m)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_in_step = True
                break

            scaled = loss / args.grad_accum
            scaled.backward()
            accum_loss += loss.item() / args.grad_accum

        if nan_in_step:
            nan_count += 1
            log(f"  Warning: NaN/Inf at step {step} -- skip #{nan_count}")
            optimizer.zero_grad(set_to_none=True)
            if nan_count >= 10:
                log(f"  {nan_count} consecutive NaN -- aborting")
                break
            continue

        nan_count = 0

        # All-reduce gradients for ignored (trainable) params -- FSDP doesn't
        # handle these since they're excluded from FlatParameter sharding.
        # Each GPU computed gradients on its own data shard; average them.
        if world_size > 1:
            for p in ignored_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # Gradient clipping (FSDP-aware: computes global norm across shards)
        # Note: clip_grad_norm_ won't see ignored params. Clip them separately.
        torch.nn.utils.clip_grad_norm_(ignored_params, 1.0)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # --- Log ---
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            tps = (step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            mem = torch.cuda.memory_allocated(device) / 1e9

            # All-reduce loss for accurate reporting
            loss_tensor = torch.tensor([accum_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

            log(f"  step {step:>5d} | loss {loss_tensor.item():.4f} "
                f"| lr {lr:.2e} | {tps:.0f} tok/s | mem {mem:.0f}GB")

        # --- Eval ---
        if step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            for _ in range(args.eval_batches):
                x, y, m = val_loader.get_batch()
                x, y = x.to(device), y.to(device)
                if m is not None:
                    m = m.to(device)
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    kwargs = {}
                    if is_gemma:
                        kwargs['mm_token_type_ids'] = torch.zeros_like(x)
                    outputs = model(input_ids=x, **kwargs)
                    vloss = compute_loss(outputs.logits, y, m)
                val_losses.append(vloss.item())

            val_loss = sum(val_losses) / len(val_losses)
            # All-reduce val loss across ranks for consistent reporting
            val_tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_tensor.item()

            val_ppl = math.exp(min(val_loss, 20))
            train_ppl = math.exp(min(accum_loss, 20))

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = step
                marker = " *"
                # Save spectrum checkpoint (gather full state on rank 0)
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                          save_policy):
                    full_sd = model.state_dict()
                if is_main:
                    spectral_sd = {k: v for k, v in full_sd.items()
                                   if 'log_spectrum' in k and 'init' not in k}
                    torch.save(spectral_sd, run_dir / "best_spectral.pt")
                    if args.checkpoint_volume:
                        vol = Path(args.checkpoint_volume) / args.run_name
                        vol.mkdir(parents=True, exist_ok=True)
                        torch.save(spectral_sd, vol / "best_spectral.pt")
                    del spectral_sd
                del full_sd

            log(f"  >> val {step:>5d} | val {val_loss:.4f} | "
                f"ppl {val_ppl:.1f} | train_ppl {train_ppl:.1f}{marker}")

            if is_main:
                log_entry = {
                    "step": step, "train_loss": accum_loss,
                    "val_loss": val_loss, "val_ppl": val_ppl, "lr": lr,
                }
                log_data.append(log_entry)
                with open(run_dir / "training_log.json", "w") as f:
                    json.dump(log_data, f, indent=2)
                if args.checkpoint_volume:
                    import shutil
                    vol = Path(args.checkpoint_volume) / args.run_name
                    vol.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(run_dir / "training_log.json",
                                 vol / "training_log.json")

            # Early stopping
            if (args.early_stop_patience > 0
                    and step - best_val_step >= args.early_stop_patience):
                log(f"  Early stop: no improvement for "
                    f"{args.early_stop_patience} steps (best at step {best_val_step})")
                break

    # ===================================================================
    # 11. Final save
    # ===================================================================
    elapsed = time.time() - t0
    log(f"\n{'=' * 70}")
    log(f"  Training complete: {elapsed / 3600:.1f}h ({elapsed / 60:.0f}min)")
    log(f"  Best val loss: {best_val_loss:.4f} at step {best_val_step}")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        full_sd = model.state_dict()
    if is_main:
        final_sd = {k: v for k, v in full_sd.items()
                    if 'log_spectrum' in k and 'init' not in k}
        torch.save(final_sd, run_dir / "final_spectral.pt")
        log(f"  Saved: {run_dir / 'final_spectral.pt'}")
        log(f"  Best:  {run_dir / 'best_spectral.pt'}")
    del full_sd

    log(f"{'=' * 70}")

    if args.inline_decompose and is_main:
        import shutil
        shards_dir = Path(args.decomposed).parent
        if shards_dir.exists() and str(shards_dir).startswith('/dev/shm'):
            log(f"  [cleanup] removing inline shards at {shards_dir}")
            shutil.rmtree(shards_dir, ignore_errors=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
