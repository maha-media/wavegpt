"""
Spectral fine-tuning: decompose a trained model, freeze geometry,
train only spectral amplitudes on target corpus.

Usage:
    python scripts/finetune_spectral.py \
        --checkpoint runs/G2-A-standard-random/best.pt \
        --data-dir data/sft-200k-spectral \
        --run-name FT-per-mode-r256 \
        --rank 256 --mode per_mode \
        --n-layer 12 --n-head 12 --n-embd 768 --block-size 1024 \
        --batch-size 8 --grad-accum 4 --max-steps 5000 \
        --lr 1e-3 --warmup-steps 200
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

from wavegpt.model import WaveGPT, WaveGPTConfig
from wavegpt.spectral_surgery import spectral_decompose, spectral_report
from wavegpt.spectral_linear import SpectralLinear
from wavegpt.data_io import read_datafile


def count_spectral_params(model):
    """Count only learnable spectral params (not embeddings/norms)."""
    total_learnable = 0
    total_frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_learnable += p.numel()
        else:
            total_frozen += p.numel()
    return total_learnable, total_frozen


def freeze_non_spectral(model):
    """Freeze everything except SpectralLinear's learnable params."""
    for name, p in model.named_parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, SpectralLinear):
            if m.mode == 'sigma1':
                m.sigma1.requires_grad = True
            elif m.mode == 'per_mode':
                m.spectrum.requires_grad = True


class DataLoader:
    """Simple binary token data loader."""

    def __init__(self, data_dir, split, block_size, batch_size, device):
        data_dir = Path(data_dir)
        # Try multiple naming conventions
        for name in [f"{split}.bin", f"sft_{split}.bin"]:
            path = data_dir / name
            if path.exists():
                break
        self.tokens = read_datafile(str(path))
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.n_tokens = len(self.tokens)
        # Check for loss mask (.npy or .bin)
        self.mask = None
        for mname in [f"{split}_mask.npy", f"sft_{split}_mask.npy", f"{split}_mask.bin"]:
            mask_path = data_dir / mname
            if mask_path.exists():
                if mname.endswith('.npy'):
                    self.mask = np.load(str(mask_path))
                else:
                    self.mask = read_datafile(str(mask_path))
                break

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


def main():
    parser = argparse.ArgumentParser(description="Spectral fine-tuning")
    # Model
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--block-size", type=int, default=1024)
    # Spectral
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--mode", type=str, default='per_mode', choices=['sigma1', 'per_mode'])
    parser.add_argument("--skip-embeddings", action="store_true", default=True)
    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=100)
    # Output
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # 1. Load base model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    config = WaveGPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, dropout=0.0,
    )
    model = WaveGPT(config)
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)

    # 2. Evaluate base model first
    print("\nLoading data...")
    train_loader = DataLoader(args.data_dir, "train", args.block_size, args.batch_size, device)
    val_loader = DataLoader(args.data_dir, "val", args.block_size, args.batch_size, device)
    print(f"  Train: {train_loader.n_tokens:,} tokens")
    print(f"  Val:   {val_loader.n_tokens:,} tokens")

    model.to(device)
    model.eval()
    print("\nEvaluating base model...")
    base_losses = []
    for _ in range(20):
        x, y, m = val_loader.get_batch()
        with torch.no_grad():
            _, loss = model(x, targets=y, loss_mask=m)
        base_losses.append(loss.item())
    base_val = sum(base_losses) / len(base_losses)
    print(f"  Base val loss: {base_val:.4f} (PPL {math.exp(base_val):.1f})")

    # 3. Spectral decomposition (on CPU, then move to device)
    model.cpu()
    print(f"\nSpectral decomposition: rank={args.rank}, mode={args.mode}")
    skip = ['wte', 'lm_head'] if args.skip_embeddings else []
    spectral_decompose(model, rank=args.rank, mode=args.mode, skip_patterns=skip)
    model.to(device)

    # 4. Freeze non-spectral params
    freeze_non_spectral(model)
    learnable, frozen = count_spectral_params(model)
    print(f"  Learnable params: {learnable:,}")
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Ratio:            {learnable / (learnable + frozen) * 100:.4f}%")

    # Verify decomposition didn't break outputs
    model.eval()
    dec_losses = []
    for _ in range(20):
        x, y, m = val_loader.get_batch()
        with torch.no_grad():
            _, loss = model(x, targets=y, loss_mask=m)
        dec_losses.append(loss.item())
    dec_val = sum(dec_losses) / len(dec_losses)
    print(f"  Post-decomposition val loss: {dec_val:.4f} (PPL {math.exp(dec_val):.1f})")
    print(f"  Degradation: {dec_val - base_val:+.4f}")

    # 5. Optimizer — only spectral params
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

    # 6. Output directory
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 7. Training loop
    print(f"\n{'=' * 60}")
    print(f"  Spectral Fine-Tuning: {args.run_name}")
    print(f"  {learnable:,} learnable params ({args.mode}, rank={args.rank})")
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
        for micro in range(args.grad_accum):
            x, y, m = train_loader.get_batch()
            _, loss = model(x, targets=y, loss_mask=m)
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
            print(f"  step {step:>5d} | loss {accum_loss:.4f} | lr {lr:.2e} | {tps:.0f} tok/s")

        # Eval
        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            for _ in range(20):
                x, y, m = val_loader.get_batch()
                with torch.no_grad():
                    _, loss = model(x, targets=y, loss_mask=m)
                val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            train_ppl = math.exp(accum_loss)
            val_ppl = math.exp(val_loss)

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                marker = " ★"
                # Save best spectral params
                spectral_sd = {}
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        spectral_sd[name] = p.data.cpu()
                torch.save(spectral_sd, run_dir / "best_spectral.pt")
                # Also save full model
                torch.save(model.state_dict(), run_dir / "best.pt")

            print(f"  ► eval step {step:>5d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.1f} | train_ppl {train_ppl:.1f}{marker}")

            log_data.append({
                "step": step,
                "train_loss": accum_loss,
                "val_loss": val_loss,
                "lr": lr,
            })
            model.train()

    # Final eval
    model.eval()
    val_losses = []
    for _ in range(50):
        x, y, m = val_loader.get_batch()
        with torch.no_grad():
            _, loss = model(x, targets=y, loss_mask=m)
        val_losses.append(loss.item())
    final_val = sum(val_losses) / len(val_losses)

    # Save final
    spectral_sd = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            spectral_sd[name] = p.data.cpu()
    torch.save(spectral_sd, run_dir / "final_spectral.pt")
    torch.save(model.state_dict(), run_dir / "final.pt")

    # Spectral report
    report = spectral_report(model)
    with open(run_dir / "spectral_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE: {args.run_name}")
    print(f"  Base val loss:     {base_val:.4f} (PPL {math.exp(base_val):.1f})")
    print(f"  Post-decomp loss:  {dec_val:.4f} (PPL {math.exp(dec_val):.1f})")
    print(f"  Best val loss:     {best_val_loss:.4f} (PPL {math.exp(best_val_loss):.1f})")
    print(f"  Final val loss:    {final_val:.4f} (PPL {math.exp(final_val):.1f})")
    print(f"  Learnable params:  {learnable:,}")
    print(f"  Spectral params saved: {run_dir / 'best_spectral.pt'}")

    # Report spectral sizes
    sp_size = os.path.getsize(run_dir / "best_spectral.pt")
    full_size = os.path.getsize(run_dir / "best.pt")
    print(f"  Spectral file:     {sp_size / 1024:.1f} KB")
    print(f"  Full model file:   {full_size / 1024 / 1024:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
