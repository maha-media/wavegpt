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

Usage (HuggingFace):
    python scripts/finetune_spectral.py \
        --hf-model Qwen/Qwen3.5-27B \
        --data-dir data/rai-qwen \
        --run-name Q-C-harmonic \
        --adaptive-rank --base-rank 192 \
        --mode per_mode --keep-residual \
        --harmonic-lambda 0.01 --collapse-alpha 0.05
"""
import argparse
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

from wavegpt.spectral_surgery import spectral_decompose, spectral_report
from wavegpt.spectral_linear import SpectralLinear
from wavegpt.harmonic_prior import harmonic_regularization


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_wavegpt(args, device='cpu'):
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


def load_hf_model(args, device='cpu'):
    """Load a HuggingFace causal LM."""
    from transformers import AutoModelForCausalLM
    kwargs = {'torch_dtype': torch.bfloat16, 'low_cpu_mem_usage': True}
    if args.trust_remote_code:
        kwargs['trust_remote_code'] = True
    print(f"  Loading HF model: {args.hf_model}")
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **kwargs)
    return model, 'hf'


def compute_loss(model, model_type, x, y, loss_mask=None):
    """Compute CE loss, handling both WaveGPT and HF model APIs."""
    if model_type == 'wavegpt':
        _, loss = model(x, targets=y, loss_mask=loss_mask)
        return loss

    # HuggingFace model — compute loss manually for mask support
    outputs = model(input_ids=x)
    logits = outputs.logits
    # Shift: predict token t+1 from position t
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = y[..., 1:].contiguous()

    if loss_mask is not None:
        shift_mask = loss_mask[..., 1:].contiguous()
        # Flatten
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1)
        # Per-token CE
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
# Data loading
# ---------------------------------------------------------------------------

def _try_load(data_dir, split):
    """Find and load token file in various naming conventions."""
    from wavegpt.data_io import read_datafile
    data_dir = Path(data_dir)
    for name in [f"{split}.bin", f"sft_{split}.bin"]:
        path = data_dir / name
        if path.exists():
            return read_datafile(str(path))
    raise FileNotFoundError(f"No {split} data in {data_dir}")


def _try_load_mask(data_dir, split):
    """Find and load mask file if present."""
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
    """Simple binary token data loader."""

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
    """Count learnable vs frozen params."""
    learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    frozen += sum(b.numel() for b in model.buffers())
    return learnable, frozen


def freeze_non_spectral(model):
    """Freeze everything except SpectralLinear's learnable params."""
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, SpectralLinear):
            if m.mode == 'sigma1':
                m.sigma1.requires_grad = True
            elif m.mode == 'per_mode':
                m.spectrum.requires_grad = True


def get_skip_patterns(model_type, hf_model_name=None):
    """Return skip patterns for embeddings/heads based on model type."""
    if model_type == 'wavegpt':
        return ['wte', 'lm_head']
    # HuggingFace models — skip embeddings, lm_head, vision encoder
    return ['embed_tokens', 'lm_head', 'visual', 'vision', 'wte', 'wpe']


def collapse_penalty(model, alpha=0.05):
    """Anti-collapse: variance penalty on last hidden layer's output."""
    # Applied externally on logits during training, not inside forward.
    # This is a placeholder — actual penalty computed in training loop
    # on hidden states / logits.
    return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spectral fine-tuning")

    # Model source (one required)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=str, help="WaveGPT .pt checkpoint")
    g.add_argument("--hf-model", type=str, help="HuggingFace model name/path")

    # WaveGPT-specific
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)

    # Spectral surgery
    parser.add_argument("--rank", type=int, default=None,
                        help="Fixed rank (omit for adaptive)")
    parser.add_argument("--adaptive-rank", action="store_true",
                        help="Theory-guided rank via 1/φ deviation")
    parser.add_argument("--base-rank", type=int, default=192,
                        help="Base rank for adaptive allocation")
    parser.add_argument("--max-rank", type=int, default=None,
                        help="Hard cap on adaptive rank")
    parser.add_argument("--mode", type=str, default='per_mode',
                        choices=['sigma1', 'per_mode'])
    parser.add_argument("--keep-residual", action="store_true",
                        help="Preserve Pythagorean comma (frozen residual)")

    # Harmonic priors
    parser.add_argument("--harmonic-lambda", type=float, default=0.0,
                        help="Harmonic regularization strength")
    parser.add_argument("--collapse-alpha", type=float, default=0.0,
                        help="Anti-collapse variance penalty strength")

    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=1024)

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, required=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    if args.checkpoint:
        print(f"\nLoading WaveGPT checkpoint: {args.checkpoint}")
        model, model_type = load_wavegpt(args)
    else:
        print(f"\nLoading HuggingFace model: {args.hf_model}")
        model, model_type = load_hf_model(args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # -----------------------------------------------------------------------
    # 2. Load data + evaluate base model
    # -----------------------------------------------------------------------
    print("\nLoading data...")
    train_loader = DataLoader(args.data_dir, "train", args.block_size, args.batch_size, device)
    val_loader = DataLoader(args.data_dir, "val", args.block_size, args.batch_size, device)
    print(f"  Train: {train_loader.n_tokens:,} tokens")
    print(f"  Val:   {val_loader.n_tokens:,} tokens")

    model.to(device)
    model.eval()
    print("\nEvaluating base model...")
    base_losses = []
    for _ in range(args.eval_batches):
        x, y, m = val_loader.get_batch()
        with torch.no_grad():
            loss = compute_loss(model, model_type, x, y, m)
        base_losses.append(loss.item())
    base_val = sum(base_losses) / len(base_losses)
    print(f"  Base val loss: {base_val:.4f} (PPL {math.exp(min(base_val, 20)):.1f})")

    # -----------------------------------------------------------------------
    # 3. Spectral decomposition
    # -----------------------------------------------------------------------
    model.cpu()
    torch.cuda.empty_cache()

    skip = get_skip_patterns(model_type, getattr(args, 'hf_model', None))
    print(f"\nSpectral decomposition:")
    print(f"  Skip patterns: {skip}")

    if args.adaptive_rank:
        print(f"  Mode: adaptive (base_rank={args.base_rank}, max_rank={args.max_rank})")
        spectral_decompose(
            model, rank='adaptive', mode=args.mode, skip_patterns=skip,
            keep_residual=args.keep_residual,
            base_rank=args.base_rank, max_rank=args.max_rank,
        )
    else:
        rank = args.rank or 256
        print(f"  Mode: fixed rank={rank}")
        spectral_decompose(
            model, rank=rank, mode=args.mode, skip_patterns=skip,
            keep_residual=args.keep_residual,
        )

    model.to(device)

    # -----------------------------------------------------------------------
    # 4. Freeze + count
    # -----------------------------------------------------------------------
    freeze_non_spectral(model)
    learnable, frozen = count_params(model)
    print(f"  Learnable params: {learnable:,}")
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Ratio:            {learnable / max(learnable + frozen, 1) * 100:.4f}%")
    if args.keep_residual:
        n_residual = sum(1 for m in model.modules()
                         if isinstance(m, SpectralLinear) and m.residual is not None)
        print(f"  Layers with residual: {n_residual}")
    if args.harmonic_lambda > 0:
        print(f"  Harmonic λ: {args.harmonic_lambda}")
    if args.collapse_alpha > 0:
        print(f"  Anti-collapse α: {args.collapse_alpha}")

    # Verify decomposition
    model.eval()
    dec_losses = []
    for _ in range(args.eval_batches):
        x, y, m = val_loader.get_batch()
        with torch.no_grad():
            loss = compute_loss(model, model_type, x, y, m)
        dec_losses.append(loss.item())
    dec_val = sum(dec_losses) / len(dec_losses)
    print(f"  Post-decomposition val loss: {dec_val:.4f} (PPL {math.exp(min(dec_val, 20)):.1f})")
    print(f"  Degradation: {dec_val - base_val:+.4f}")

    # -----------------------------------------------------------------------
    # 5. Optimizer
    # -----------------------------------------------------------------------
    spectral_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(spectral_params, lr=args.lr, weight_decay=args.weight_decay)

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    print(f"\n  Effective batch: {tokens_per_step:,} tokens/step")

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        decay_ratio = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.lr * 0.1 + (args.lr - args.lr * 0.1) * coeff

    # -----------------------------------------------------------------------
    # 6. Output directory
    # -----------------------------------------------------------------------
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # -----------------------------------------------------------------------
    # 7. Training loop
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Spectral Fine-Tuning: {args.run_name}")
    print(f"  {learnable:,} learnable params ({args.mode})")
    features = []
    if args.adaptive_rank: features.append("adaptive-rank")
    if args.keep_residual: features.append("residual")
    if args.harmonic_lambda > 0: features.append(f"harmonic-λ={args.harmonic_lambda}")
    if args.collapse_alpha > 0: features.append(f"collapse-α={args.collapse_alpha}")
    if features:
        print(f"  Harmonic: {', '.join(features)}")
    print(f"{'=' * 60}\n")

    log_data = []
    best_val_loss = float("inf")
    t0 = time.time()
    model.train()

    for step in range(args.max_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accum_loss = 0.0
        accum_hreg = 0.0

        for micro in range(args.grad_accum):
            x, y, m = train_loader.get_batch()
            loss = compute_loss(model, model_type, x, y, m)

            # Harmonic regularization
            if args.harmonic_lambda > 0:
                hreg = harmonic_regularization(model, args.harmonic_lambda)
                loss = loss + hreg
                accum_hreg += hreg.item() / args.grad_accum

            # Anti-collapse: variance penalty on logits
            if args.collapse_alpha > 0:
                # Get logits from last forward (already computed in loss)
                # Re-forward is expensive; apply on loss magnitude instead
                # Simple proxy: penalize low loss variance across batch
                pass  # TODO: implement efficiently for HF models

            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(spectral_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Log
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            tps = (step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            hreg_str = f" | hreg {accum_hreg:.4f}" if args.harmonic_lambda > 0 else ""
            print(f"  step {step:>5d} | loss {accum_loss:.4f}{hreg_str} | lr {lr:.2e} | {tps:.0f} tok/s")

        # Eval
        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            for _ in range(args.eval_batches):
                x, y, m = val_loader.get_batch()
                with torch.no_grad():
                    loss = compute_loss(model, model_type, x, y, m)
                val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            train_ppl = math.exp(min(accum_loss, 20))
            val_ppl = math.exp(min(val_loss, 20))

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                marker = " ★"
                # Save best spectral params (tiny file)
                spectral_sd = {name: p.data.cpu()
                               for name, p in model.named_parameters()
                               if p.requires_grad}
                torch.save(spectral_sd, run_dir / "best_spectral.pt")

            print(f"  ► eval {step:>5d} | val {val_loss:.4f} | ppl {val_ppl:.1f} | train_ppl {train_ppl:.1f}{marker}")

            log_data.append({
                "step": step, "train_loss": accum_loss,
                "val_loss": val_loss, "lr": lr,
            })
            model.train()

    # -----------------------------------------------------------------------
    # 8. Final eval + save
    # -----------------------------------------------------------------------
    model.eval()
    val_losses = []
    for _ in range(50):
        x, y, m = val_loader.get_batch()
        with torch.no_grad():
            loss = compute_loss(model, model_type, x, y, m)
        val_losses.append(loss.item())
    final_val = sum(val_losses) / len(val_losses)

    # Save final spectral params
    spectral_sd = {name: p.data.cpu()
                   for name, p in model.named_parameters()
                   if p.requires_grad}
    torch.save(spectral_sd, run_dir / "final_spectral.pt")

    # Spectral report + training log
    report = spectral_report(model)
    with open(run_dir / "spectral_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # Summary
    sp_size = os.path.getsize(run_dir / "best_spectral.pt")

    print(f"\n{'=' * 60}")
    print(f"  DONE: {args.run_name}")
    print(f"  Base val loss:     {base_val:.4f} (PPL {math.exp(min(base_val, 20)):.1f})")
    print(f"  Post-decomp loss:  {dec_val:.4f} (PPL {math.exp(min(dec_val, 20)):.1f})")
    print(f"  Best val loss:     {best_val_loss:.4f} (PPL {math.exp(min(best_val_loss, 20)):.1f})")
    print(f"  Final val loss:    {final_val:.4f} (PPL {math.exp(min(final_val, 20)):.1f})")
    print(f"  Learnable params:  {learnable:,}")
    print(f"  Spectral file:     {sp_size / 1024:.1f} KB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
