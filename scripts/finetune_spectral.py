"""
Spectral fine-tuning: decompose a trained model, freeze geometry,
train only spectral amplitudes on target corpus.

Supports both WaveGPT checkpoints and any HuggingFace model.

Usage (WaveGPT):
    python scripts/finetune_spectral.py \
        --checkpoint runs/G2-A/best.pt \
        --data-dir data/sft-200k-spectral \
        --run-name FT-per-mode-r256 \
        --rank 256 --mode per_mode \
        --n-layer 12 --n-head 12 --n-embd 768

Usage (HuggingFace — memory-safe for 27B):
    python scripts/finetune_spectral.py \
        --hf-model Qwen/Qwen3.5-27B \
        --data-dir data/rai-qwen \
        --run-name Q-B-vanilla \
        --rank 256 --mode per_mode \
        --batch-size 1 --block-size 512 --grad-accum 16 \
        --lr 1e-3 --max-steps 2000 --eval-interval 50
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavegpt.spectral_surgery import spectral_decompose, spectral_report, spectral_scaffold
from wavegpt.spectral_linear import SpectralLinear
from wavegpt.harmonic_prior import harmonic_regularization


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def gpu_mem():
    """Return (allocated_GB, reserved_GB, total_GB)."""
    if not torch.cuda.is_available():
        return 0, 0, 0
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    props = torch.cuda.get_device_properties(0)
    t = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    return a, r, t


def log_mem(label=""):
    a, r, t = gpu_mem()
    print(f"  [MEM] {label}: {a:.1f}GB alloc / {r:.1f}GB reserved / {t:.1f}GB total")


def model_tensor_audit(model):
    """Count ALL tensors attached to model, not just params/buffers."""
    seen = set()
    total_bytes = 0
    by_dtype = {}
    for name, module in model.named_modules():
        for attr_name in list(module.__dict__.keys()):
            obj = getattr(module, attr_name, None)
            if isinstance(obj, torch.Tensor) and id(obj) not in seen:
                seen.add(id(obj))
                b = obj.numel() * obj.element_size()
                total_bytes += b
                dt = str(obj.dtype)
                by_dtype[dt] = by_dtype.get(dt, 0) + b
    # Also count _parameters and _buffers explicitly
    for p in model.parameters():
        if id(p.data) not in seen:
            seen.add(id(p.data))
            b = p.data.numel() * p.data.element_size()
            total_bytes += b
            dt = str(p.data.dtype)
            by_dtype[dt] = by_dtype.get(dt, 0) + b
    for buf in model.buffers():
        if id(buf) not in seen:
            seen.add(id(buf))
            b = buf.numel() * buf.element_size()
            total_bytes += b
            dt = str(buf.dtype)
            by_dtype[dt] = by_dtype.get(dt, 0) + b
    print(f"  Model tensor audit: {total_bytes/1e9:.2f} GB total")
    for dt, b in sorted(by_dtype.items(), key=lambda x: -x[1]):
        print(f"    {dt}: {b/1e9:.2f} GB")
    return total_bytes


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_wavegpt(args):
    """Load a WaveGPT checkpoint."""
    from wavegpt.model import WaveGPT, WaveGPTConfig
    config = WaveGPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, dropout=0.0,
    )
    model = WaveGPT(config)
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    return model, 'wavegpt'


def load_hf_model(args):
    """Load a HuggingFace causal LM."""
    from transformers import AutoModelForCausalLM
    kwargs = {'torch_dtype': torch.bfloat16, 'low_cpu_mem_usage': True}
    if args.trust_remote_code:
        kwargs['trust_remote_code'] = True
    print(f"  Loading HF model: {args.hf_model}")
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **kwargs)
    return model, 'hf'


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(model, model_type, x, y, loss_mask=None):
    """Compute CE loss, handling both WaveGPT and HF model APIs."""
    if model_type == 'wavegpt':
        _, loss = model(x, targets=y, loss_mask=loss_mask)
        return loss

    # Gemma 4 (multimodal) requires mm_token_type_ids; zeros = all text tokens
    kwargs = {}
    if hasattr(model.config, 'model_type') and 'gemma' in getattr(model.config, 'model_type', ''):
        kwargs['mm_token_type_ids'] = torch.zeros_like(x)
    outputs = model(input_ids=x, **kwargs)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = y[..., 1:].contiguous()

    if loss_mask is not None:
        shift_mask = loss_mask[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1)
        per_token = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        if flat_mask.sum() > 0:
            loss = (per_token * flat_mask).sum() / flat_mask.sum()
        else:
            loss = per_token.mean()
    else:
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    return loss


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_samples(model, model_type, tokenizer, device, prompts, max_new=200):
    """Generate sample completions for monitoring."""
    model.eval()
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            if model_type == 'hf':
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            else:
                # WaveGPT — simple autoregressive
                out = input_ids
                for _ in range(max_new):
                    logits, _ = model(out[:, -model.config.block_size:])
                    logits = logits[:, -1, :] / 0.7
                    probs = F.softmax(logits, dim=-1)
                    next_tok = torch.multinomial(probs, 1)
                    out = torch.cat([out, next_tok], dim=1)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({'prompt': prompt, 'response': text[len(prompt):]})
    return results


# ---------------------------------------------------------------------------
# Data loading
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
            else:
                return read_datafile(str(mask_path))
    return None


class DataLoader:
    def __init__(self, data_dir, split, block_size, batch_size, device):
        self.tokens = _try_load(data_dir, split)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.n_tokens = len(self.tokens)
        self.mask = _try_load_mask(data_dir, split)

    def get_batch(self):
        ix = torch.randint(self.n_tokens - self.block_size - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy(self.tokens[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.tokens[i+1:i+1+self.block_size].astype(np.int64)) for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        if self.mask is not None:
            m = torch.stack([torch.from_numpy(self.mask[i+1:i+1+self.block_size].astype(np.float32)) for i in ix])
            m = m.to(self.device)
        else:
            m = None
        return x, y, m


# ---------------------------------------------------------------------------
# Spectral param utilities
# ---------------------------------------------------------------------------

def count_params(model):
    learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    frozen += sum(b.numel() for b in model.buffers())
    return learnable, frozen


def freeze_non_spectral(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            if m.mode == 'sigma1':
                m.sigma1.requires_grad = True
            elif m.mode == 'per_mode':
                m.spectrum.requires_grad = True


def get_skip_patterns(model_type):
    if model_type == 'wavegpt':
        return ['wte', 'lm_head']
    return ['embed_tokens', 'lm_head', 'visual', 'vision', 'wte', 'wpe']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spectral fine-tuning")

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=str)
    g.add_argument("--hf-model", type=str)
    g.add_argument("--decomposed", type=str,
                   help="Pre-decomposed model state_dict (skip SVD)")

    # WaveGPT arch
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)

    # Spectral surgery
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--adaptive-rank", action="store_true")
    parser.add_argument("--base-rank", type=int, default=192)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--mode", type=str, default='per_mode',
                        choices=['sigma1', 'per_mode'])
    parser.add_argument("--keep-residual", action="store_true")

    # Harmonic priors
    parser.add_argument("--harmonic-lambda", type=float, default=0.0)
    parser.add_argument("--type-aware-harmonic", action="store_true",
                        help="Use F/L exponents per layer type (attn_o pinned at 1/3)")
    parser.add_argument("--attn-o-weight", type=float, default=10.0,
                        help="Extra regularization weight on attn_o layers (default 10x)")
    parser.add_argument("--collapse-alpha", type=float, default=0.0)

    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=512)

    # Training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--generate-interval", type=int, default=200,
                        help="Generate sample completions every N steps (0=off)")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, required=True)

    # SSD (Simple Self-Distillation) — arXiv:2604.01193
    parser.add_argument("--ssd", action="store_true",
                        help="Run SSD phase after corpus training")
    parser.add_argument("--ssd-temperature", type=float, default=1.2,
                        help="Sampling temperature for SSD data generation")
    parser.add_argument("--ssd-top-k", type=int, default=50,
                        help="Top-k truncation for SSD sampling")
    parser.add_argument("--ssd-top-p", type=float, default=0.95,
                        help="Top-p truncation for SSD sampling")
    parser.add_argument("--ssd-samples", type=int, default=8,
                        help="Number of SSD samples per prompt")
    parser.add_argument("--ssd-max-tokens", type=int, default=512,
                        help="Max tokens per SSD sample")
    parser.add_argument("--ssd-steps", type=int, default=500,
                        help="Training steps for SSD phase")
    parser.add_argument("--ssd-prompts", type=str, default=None,
                        help="Path to prompts JSONL file for SSD (one per line)")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        log_mem("startup")

    # ===================================================================
    # 1. Load model
    # ===================================================================
    hf_model_name = args.hf_model or (args.decomposed.split('/')[0] if args.decomposed else None)

    if args.decomposed:
        # Fast path: load pre-decomposed model (skip 3hr SVD)
        print(f"\nLoading pre-decomposed model: {args.decomposed}")
        from transformers import AutoModelForCausalLM, AutoConfig
        # Need the HF model name to reconstruct architecture
        decomp_dir = Path(args.decomposed).parent
        config_path = decomp_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)
            hf_model_name = saved_config.get('hf_model', hf_model_name)
        if not hf_model_name:
            print("ERROR: --decomposed requires HF model name in config.json or directory name")
            sys.exit(1)
        print(f"  Architecture from: {hf_model_name}")
        # Build model from config (random init — weights come from decomposed shards)
        hf_config = AutoConfig.from_pretrained(
            hf_model_name, trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_config(
            hf_config, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print(f"  ✓ Model skeleton created (random init, will load decomposed weights)")
        # Load state dict first to read variable ranks
        print(f"  Loading saved state_dict...")
        decomp_path = Path(args.decomposed)
        # Support: .pt, .safetensors, or shards/ directory
        shard_dir = decomp_path.parent / "shards"
        if shard_dir.exists() and (shard_dir / "index.json").exists():
            # Sharded safetensors
            from safetensors.torch import load_file
            import json as _json
            with open(shard_dir / "index.json") as f:
                index = _json.load(f)
            sd = {}
            for shard_name in index["shards"]:
                shard_sd = load_file(str(shard_dir / shard_name), device='cpu')
                sd.update(shard_sd)
                print(f"  Loaded shard {shard_name}: {len(shard_sd)} tensors")
            print(f"  Total: {len(sd)} tensors from {len(index['shards'])} shards")
        elif decomp_path.suffix == '.safetensors' or not decomp_path.exists():
            st_path = decomp_path.with_suffix('.safetensors') if decomp_path.suffix != '.safetensors' else decomp_path
            if st_path.exists():
                from safetensors.torch import load_file
                sd = load_file(str(st_path), device='cpu')
                print(f"  Loaded safetensors: {len(sd)} tensors")
            else:
                sd = torch.load(args.decomposed, map_location='cpu', weights_only=True)
        else:
            try:
                sd = torch.load(args.decomposed, map_location='cpu', weights_only=True)
            except RuntimeError:
                # Corrupted .pt? Try .safetensors fallback
                st_path = decomp_path.with_suffix('.safetensors')
                if st_path.exists():
                    from safetensors.torch import load_file
                    sd = load_file(str(st_path), device='cpu')
                    print(f"  .pt corrupted, loaded safetensors fallback")
                else:
                    raise
        # Scaffold with state_dict to infer per-layer rank (adaptive-rank support)
        skip = get_skip_patterns('hf')
        rank = args.rank or 256
        print(f"  Scaffolding SpectralLinear architecture (no SVD)...")
        spectral_scaffold(model, rank=rank, mode=args.mode,
                          skip_patterns=skip, state_dict=sd)
        model.load_state_dict(sd, strict=False)
        model_type = 'hf'
        print(f"  ✓ Loaded decomposed model ({len(sd)} tensors) — no SVD needed")
    elif args.checkpoint:
        print(f"\nLoading WaveGPT checkpoint: {args.checkpoint}")
        model, model_type = load_wavegpt(args)
    else:
        print(f"\nLoading HuggingFace model: {args.hf_model}")
        model, model_type = load_hf_model(args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Load tokenizer for HF models (needed for sample generation)
    tokenizer = None
    if model_type == 'hf' and hf_model_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

    # ===================================================================
    # 2. Load data
    # ===================================================================
    print("\nLoading data...")
    train_loader = DataLoader(args.data_dir, "train", args.block_size,
                              args.batch_size, device)
    val_loader = DataLoader(args.data_dir, "val", args.block_size,
                            args.batch_size, device)
    print(f"  Train: {train_loader.n_tokens:,} tokens")
    print(f"  Val:   {val_loader.n_tokens:,} tokens")

    # ===================================================================
    # 3. Spectral decomposition (on CPU — no GPU needed)
    # ===================================================================
    if not args.decomposed:
        skip = get_skip_patterns(model_type)
        print(f"\nSpectral decomposition (on CPU):")
        print(f"  Skip patterns: {skip}")

        rank = args.rank or 256
        if args.adaptive_rank:
            print(f"  Mode: adaptive (base={args.base_rank}, max={args.max_rank})")
            spectral_decompose(
                model, rank='adaptive', mode=args.mode, skip_patterns=skip,
                keep_residual=args.keep_residual,
                base_rank=args.base_rank, max_rank=args.max_rank,
            )
        else:
            print(f"  Mode: fixed rank={rank}")
            spectral_decompose(
                model, rank=rank, mode=args.mode, skip_patterns=skip,
                keep_residual=args.keep_residual,
            )

    # Audit model size BEFORE moving to GPU
    print("\n  Pre-GPU audit:")
    model_bytes = model_tensor_audit(model)
    model_gb = model_bytes / 1e9
    _, _, total_vram = gpu_mem()
    print(f"  Model: {model_gb:.1f} GB, GPU total: {total_vram:.1f} GB")

    if model_gb > total_vram * 0.85:
        print(f"  ⚠ WARNING: Model ({model_gb:.1f}GB) may not fit on GPU ({total_vram:.1f}GB)")
        print(f"  Consider: --rank {rank // 2} or smaller model")

    # Save decomposed model for instant reuse (skip 3hr SVD next time)
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.decomposed:
        decomp_path = run_dir / "decomposed.pt"
        print(f"\n  Saving decomposed model to {decomp_path}...")
        torch.save(model.state_dict(), decomp_path)
        decomp_size = os.path.getsize(decomp_path)
        print(f"  ✓ Saved ({decomp_size / 1e9:.2f} GB) — reuse with --decomposed {decomp_path}")

    # ===================================================================
    # 4. Move to GPU + enable memory optimizations
    # ===================================================================
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  Moving model to {device}...")
    model.to(device)
    log_mem("after model.to(device)")

    # Gradient checkpointing for HF models
    if model_type == 'hf' and hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        print("  ✓ Gradient checkpointing enabled")

    # ===================================================================
    # 5. Freeze + count
    # ===================================================================
    freeze_non_spectral(model)
    learnable, frozen = count_params(model)
    print(f"  Learnable params: {learnable:,}")
    print(f"  Frozen params:    {frozen:,}")
    if args.harmonic_lambda > 0:
        mode = "type-aware (F/L per type)" if args.type_aware_harmonic else "uniform (1/φ)"
        print(f"  Harmonic λ: {args.harmonic_lambda} [{mode}]")
        if args.type_aware_harmonic:
            print(f"  attn_o weight: {args.attn_o_weight}× (pinned at (1/φ)^(1/3))")
    if args.collapse_alpha > 0:
        print(f"  Anti-collapse α: {args.collapse_alpha}")

    # ===================================================================
    # 6. Post-decomposition eval
    # ===================================================================
    print("\n  Post-decomposition eval...")
    model.eval()
    dec_losses = []
    for i in range(args.eval_batches):
        x, y, m = val_loader.get_batch()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = compute_loss(model, model_type, x, y, m)
        dec_losses.append(loss.item())
    dec_val = sum(dec_losses) / len(dec_losses)
    dec_ppl = math.exp(min(dec_val, 20))
    print(f"  Post-decomp val loss: {dec_val:.4f} (PPL {dec_ppl:.1f})")
    log_mem("after eval")

    # ===================================================================
    # 7. Test backward pass with single microbatch before committing
    # ===================================================================
    print("\n  Testing backward pass (1 microbatch)...")
    model.train()
    x, y, m = train_loader.get_batch()
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = compute_loss(model, model_type, x, y, m)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        log_mem("after test backward")
        print("  ✓ Backward pass OK")
    except torch.cuda.OutOfMemoryError:
        print("  ✗ OOM on backward pass!")
        log_mem("OOM")
        print("  Trying with block_size=256...")
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        args.block_size = 256
        train_loader = DataLoader(args.data_dir, "train", 256,
                                  args.batch_size, device)
        val_loader = DataLoader(args.data_dir, "val", 256,
                                args.batch_size, device)
        x, y, m = train_loader.get_batch()
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = compute_loss(model, model_type, x, y, m)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"  ✓ Backward pass OK at block_size=256")
            log_mem("after fallback backward")
        except torch.cuda.OutOfMemoryError:
            print("  ✗ Still OOM at block_size=256. Cannot train. Exiting.")
            log_mem("fatal OOM")
            sys.exit(1)

    # ===================================================================
    # 8. Optimizer + LR schedule
    # ===================================================================
    spectral_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(spectral_params, lr=args.lr,
                                  weight_decay=args.weight_decay)

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    print(f"\n  Effective batch: {tokens_per_step:,} tokens/step")
    print(f"  Batch: {args.batch_size} × {args.block_size} × {args.grad_accum} accum")

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        decay_ratio = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.lr * 0.1 + (args.lr - args.lr * 0.1) * coeff

    # ===================================================================
    # 9. Output directory + sample prompts
    # ===================================================================
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # RAI test prompts for live monitoring
    test_prompts = [
        "The most important thing about the Singularity is",
        "When I think about my father, I remember",
        "The exponential growth of technology means that",
        "People often ask me if I'm afraid of death. My answer is",
        "Pattern recognition is fundamental to intelligence because",
    ]

    # ===================================================================
    # 10. Training loop
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"  Spectral Fine-Tuning: {args.run_name}")
    print(f"  {learnable:,} learnable params ({args.mode})")
    features = []
    if args.adaptive_rank: features.append("adaptive-rank")
    if args.keep_residual: features.append("residual")
    if args.harmonic_lambda > 0: features.append(f"harmonic-λ={args.harmonic_lambda}")
    if args.collapse_alpha > 0: features.append(f"collapse-α={args.collapse_alpha}")
    if features:
        print(f"  Harmonic: {', '.join(features)}")
    print(f"{'=' * 70}\n")

    log_data = []
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(args.max_steps):
        model.train()
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accum_loss = 0.0
        accum_hreg = 0.0

        for micro in range(args.grad_accum):
            x, y, m = train_loader.get_batch()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = compute_loss(model, model_type, x, y, m)

                if args.harmonic_lambda > 0:
                    hreg = harmonic_regularization(
                        model, args.harmonic_lambda,
                        type_aware=args.type_aware_harmonic,
                        attn_o_weight=args.attn_o_weight,
                    )
                    loss = loss + hreg
                    accum_hreg += hreg.item() / args.grad_accum

            scaled = loss / args.grad_accum
            scaled.backward()
            accum_loss += scaled.item()

        torch.nn.utils.clip_grad_norm_(spectral_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # --- Log ---
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            tps = (step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            a, r, t = gpu_mem()
            hreg_str = f" | hreg {accum_hreg:.4f}" if args.harmonic_lambda > 0 else ""
            print(f"  step {step:>5d} | loss {accum_loss:.4f}{hreg_str} "
                  f"| lr {lr:.2e} | {tps:.0f} tok/s | mem {a:.0f}GB")

        # --- Eval ---
        if step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            for _ in range(args.eval_batches):
                x, y, m = val_loader.get_batch()
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = compute_loss(model, model_type, x, y, m)
                val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(min(val_loss, 20))
            train_ppl = math.exp(min(accum_loss, 20))

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                marker = " ★"
                spectral_sd = {name: p.data.cpu()
                               for name, p in model.named_parameters()
                               if p.requires_grad}
                torch.save(spectral_sd, run_dir / "best_spectral.pt")

            print(f"  ► eval {step:>5d} | val {val_loss:.4f} | "
                  f"ppl {val_ppl:.1f} | train_ppl {train_ppl:.1f}{marker}")

            log_entry = {
                "step": step, "train_loss": accum_loss,
                "val_loss": val_loss, "val_ppl": val_ppl, "lr": lr,
            }
            log_data.append(log_entry)

            # Save running log (so we can monitor from outside)
            with open(run_dir / "training_log.json", "w") as f:
                json.dump(log_data, f, indent=2)

        # --- Generate samples ---
        if (args.generate_interval > 0 and tokenizer is not None
                and step > 0 and step % args.generate_interval == 0):
            print(f"\n  --- Samples at step {step} ---")
            try:
                samples = generate_samples(
                    model, model_type, tokenizer, device,
                    test_prompts[:3], max_new=150,
                )
                for s in samples:
                    resp = s['response'][:300].replace('\n', ' ')
                    print(f"  Q: {s['prompt']}")
                    print(f"  A: {resp}")
                    print()
                # Save samples
                all_samples_path = run_dir / "samples.jsonl"
                with open(all_samples_path, "a") as f:
                    for s in samples:
                        f.write(json.dumps({"step": step, **s}) + "\n")
            except Exception as e:
                print(f"  Sample generation failed: {e}")
            print(f"  --- End samples ---\n")

    # ===================================================================
    # 11. SSD Phase (Simple Self-Distillation) — arXiv:2604.01193
    # ===================================================================
    if args.ssd and tokenizer is not None and model_type == 'hf':
        print(f"\n{'=' * 70}")
        print(f"  SSD Phase: Self-Distillation")
        print(f"  T={args.ssd_temperature}, top_k={args.ssd_top_k}, "
              f"top_p={args.ssd_top_p}, samples={args.ssd_samples}")
        print(f"{'=' * 70}")

        # Load prompts
        ssd_prompts = []
        if args.ssd_prompts and Path(args.ssd_prompts).exists():
            with open(args.ssd_prompts) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            ssd_prompts.append(json.loads(line)['prompt'])
                        except (json.JSONDecodeError, KeyError):
                            ssd_prompts.append(line)
        else:
            # Default: use RAI-style prompts
            ssd_prompts = test_prompts + [
                "The key insight about exponential growth is",
                "In my view, the future of artificial intelligence",
                "What I learned from studying technology trends is",
                "The relationship between biology and technology",
                "Looking at the history of computing, we can see",
            ]
        print(f"  Using {len(ssd_prompts)} prompts")

        # Phase 1: Generate self-distillation data
        print(f"  Generating SSD data...")
        model.eval()
        ssd_tokens = []  # list of token sequences
        for pi, prompt in enumerate(ssd_prompts):
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            for si in range(args.ssd_samples):
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=args.ssd_max_tokens,
                        do_sample=True,
                        temperature=args.ssd_temperature,
                        top_k=args.ssd_top_k,
                        top_p=args.ssd_top_p,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                ssd_tokens.append(out[0].cpu())
            if (pi + 1) % 10 == 0:
                print(f"    Generated {(pi+1)*args.ssd_samples} samples "
                      f"({pi+1}/{len(ssd_prompts)} prompts)")

        total_ssd_tokens = sum(len(t) for t in ssd_tokens)
        print(f"  Generated {len(ssd_tokens)} samples, {total_ssd_tokens:,} tokens")

        # Save SSD data for inspection
        ssd_data_path = run_dir / "ssd_samples.jsonl"
        with open(ssd_data_path, "w") as f:
            for t in ssd_tokens[:20]:  # save first 20 for inspection
                text = tokenizer.decode(t, skip_special_tokens=True)
                f.write(json.dumps({"text": text[:500]}) + "\n")

        # Phase 2: Fine-tune on self-generated data
        print(f"  SSD training ({args.ssd_steps} steps)...")
        model.train()

        # Concatenate all SSD tokens into a flat array
        ssd_flat = torch.cat(ssd_tokens, dim=0).numpy()
        ssd_n = len(ssd_flat)
        print(f"  SSD corpus: {ssd_n:,} tokens")

        ssd_optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr * 0.1,  # lower LR for SSD refinement
            weight_decay=args.weight_decay,
        )

        ssd_best_val = float('inf')
        for ssd_step in range(args.ssd_steps):
            # Get batch from SSD data
            ix = torch.randint(ssd_n - args.block_size - 1, (args.batch_size,))
            x = torch.stack([torch.from_numpy(
                ssd_flat[i:i+args.block_size].astype(np.int64)) for i in ix]).to(device)
            y = torch.stack([torch.from_numpy(
                ssd_flat[i+1:i+1+args.block_size].astype(np.int64)) for i in ix]).to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = compute_loss(model, model_type, x, y)
            loss.backward()

            if (ssd_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                ssd_optimizer.step()
                ssd_optimizer.zero_grad(set_to_none=True)

            if ssd_step % args.log_interval == 0:
                print(f"  ssd step {ssd_step:>4d} | loss {loss.item():.4f}")

            if ssd_step % args.eval_interval == 0:
                model.eval()
                vl = []
                for _ in range(args.eval_batches):
                    xv, yv, mv = val_loader.get_batch()
                    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        vloss = compute_loss(model, model_type, xv, yv, mv)
                    vl.append(vloss.item())
                sv = sum(vl) / len(vl)
                sp = math.exp(min(sv, 20))
                marker = " ★" if sv < ssd_best_val else ""
                if sv < ssd_best_val:
                    ssd_best_val = sv
                    spectral_sd = {name: p.data.cpu()
                                   for name, p in model.named_parameters()
                                   if p.requires_grad}
                    torch.save(spectral_sd, run_dir / "best_ssd_spectral.pt")
                print(f"  ► ssd eval {ssd_step:>4d} | val {sv:.4f} | ppl {sp:.1f}{marker}")
                model.train()

        print(f"  SSD phase complete. Best val: {ssd_best_val:.4f} "
              f"(PPL {math.exp(min(ssd_best_val, 20)):.1f})")

    # ===================================================================
    # 12. Final eval + save
    # ===================================================================
    model.eval()
    val_losses = []
    for _ in range(50):
        x, y, m = val_loader.get_batch()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = compute_loss(model, model_type, x, y, m)
        val_losses.append(loss.item())
    final_val = sum(val_losses) / len(val_losses)
    final_ppl = math.exp(min(final_val, 20))

    # Save final spectral params
    spectral_sd = {name: p.data.cpu()
                   for name, p in model.named_parameters()
                   if p.requires_grad}
    torch.save(spectral_sd, run_dir / "final_spectral.pt")

    # Spectral report
    try:
        report = spectral_report(model)
        with open(run_dir / "spectral_report.json", "w") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        print(f"  Warning: spectral report failed: {e}")

    # Final training log
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # Generate final samples
    if tokenizer is not None:
        print("\n  --- Final samples ---")
        try:
            samples = generate_samples(
                model, model_type, tokenizer, device,
                test_prompts, max_new=200,
            )
            for s in samples:
                resp = s['response'][:400].replace('\n', ' ')
                print(f"  Q: {s['prompt']}")
                print(f"  A: {resp}")
                print()
            with open(run_dir / "final_samples.json", "w") as f:
                json.dump(samples, f, indent=2)
        except Exception as e:
            print(f"  Final sample generation failed: {e}")
        print(f"  --- End final samples ---")

    sp_size = os.path.getsize(run_dir / "best_spectral.pt") if (run_dir / "best_spectral.pt").exists() else 0

    print(f"\n{'=' * 70}")
    print(f"  DONE: {args.run_name}")
    print(f"  Post-decomp loss:  {dec_val:.4f} (PPL {dec_ppl:.1f})")
    print(f"  Best val loss:     {best_val_loss:.4f} (PPL {math.exp(min(best_val_loss, 20)):.1f})")
    print(f"  Final val loss:    {final_val:.4f} (PPL {final_ppl:.1f})")
    print(f"  Learnable params:  {learnable:,}")
    print(f"  Spectral file:     {sp_size / 1024:.1f} KB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
