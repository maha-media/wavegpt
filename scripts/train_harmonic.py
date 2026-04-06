"""
Train HarmonicGPT — the power-law spectral transformer.

Same training loop as train_sft.py but using HarmonicGPT (10x fewer weight params).
Logs alpha values per layer to track spectral learning.

Usage:
    python scripts/train_harmonic.py --data-dir data/sft-50k-harmonic --run-name H-A
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavegpt.harmonic_model import HarmonicGPT, HarmonicGPTConfig
from wavegpt.harmonic_linear import HarmonicLinear
from wavegpt.sft_dataloader import SFTDataLoader
from wavegpt.data_io import read_datafile


def main():
    parser = argparse.ArgumentParser(description="Train HarmonicGPT")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="harmonic-run")
    parser.add_argument("--output-dir", type=str, default="runs")

    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank-attn", type=int, default=30)
    parser.add_argument("--rank-mlp", type=int, default=48)
    parser.add_argument("--init-alpha", type=float, default=0.7)
    parser.add_argument("--fix-alpha", action="store_true",
                        help="Fix alpha as constant (not learned)")
    parser.add_argument("--alpha-proj", type=float, default=None,
                        help="Separate alpha for projection layers (c_proj)")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--collapse-alpha", type=float, default=0.0)

    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--spectral-interval", type=int, default=1000)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load data
    data_dir = Path(args.data_dir)
    print(f"\nLoading data from {data_dir}...")
    train_tokens = read_datafile(str(data_dir / "sft_train.bin"))
    val_tokens = read_datafile(str(data_dir / "sft_val.bin"))
    train_masks = np.load(str(data_dir / "sft_train_mask.npy"))
    val_masks = np.load(str(data_dir / "sft_val_mask.npy"))
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val:   {len(val_tokens):,} tokens")

    train_loader = SFTDataLoader(train_tokens, train_masks,
                                  args.batch_size, args.block_size, device)
    val_loader = SFTDataLoader(val_tokens, val_masks,
                                args.batch_size, args.block_size, device)

    # Model
    config = HarmonicGPTConfig(
        vocab_size=50257, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        dropout=args.dropout, rank_attn=args.rank_attn, rank_mlp=args.rank_mlp,
        init_alpha=args.init_alpha, fix_alpha=args.fix_alpha,
        alpha_proj=args.alpha_proj, collapse_alpha=args.collapse_alpha,
    )
    model = HarmonicGPT(config).to(device)

    total_params = model.count_params()
    weight_params = model.count_weight_params()
    print(f"\nHarmonicGPT: {total_params/1e6:.1f}M total params")
    print(f"  Weight params (HarmonicLinear): {weight_params:,}")
    print(f"  rank_attn={args.rank_attn}, rank_mlp={args.rank_mlp}")

    # Standard GPT for comparison
    std_weight = 6 * (3 * 384 * 384 + 384 * 384 + 384 * 1536 + 1536 * 384)
    print(f"  Standard weight params: {std_weight:,}")
    print(f"  Compression: {std_weight / weight_params:.1f}x")

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    print(f"  Effective batch: {tokens_per_step:,} tokens/step")

    # Optimizer
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / args.warmup_steps
        decay_ratio = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.lr * 0.1 + (args.lr - args.lr * 0.1) * coeff

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Training
    print(f"\n{'='*60}")
    print(f"Training: {args.run_name}")
    print(f"{'='*60}\n")

    log_data = []
    spectral_log = []
    best_val_loss = float("inf")
    t0 = time.time()
    model.train()
    optimizer.zero_grad()

    for step in range(args.max_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accum_loss = 0.0
        for micro in range(args.grad_accum):
            x, y, mask = train_loader.get_batch()
            _, loss = model(x, targets=y, loss_mask=mask, step=step, total_steps=args.max_steps)
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            tok_s = (step + 1) * tokens_per_step / elapsed if elapsed > 0 else 0
            print(f"  step {step:>6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {tok_s:.0f} tok/s")

        # Spectral logging — track how alpha evolves
        if step % args.spectral_interval == 0:
            summary = model.spectral_summary()
            alphas = [v["alpha"] for v in summary.values()]
            sigmas = [v["sigma1"] for v in summary.values()]
            entry = {
                "step": step,
                "mean_alpha": round(np.mean(alphas), 4),
                "std_alpha": round(np.std(alphas), 4),
                "min_alpha": round(min(alphas), 4),
                "max_alpha": round(max(alphas), 4),
                "mean_sigma1": round(np.mean(sigmas), 4),
            }
            spectral_log.append(entry)
            print(f"  ♪ α: mean={entry['mean_alpha']:.3f} "
                  f"[{entry['min_alpha']:.3f}, {entry['max_alpha']:.3f}] "
                  f"σ₁={entry['mean_sigma1']:.3f}")

        # Eval
        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    x, y, mask = val_loader.get_batch()
                    _, vloss = model(x, targets=y, loss_mask=mask)
                    val_losses.append(vloss.item())

            val_loss = np.mean(val_losses)
            val_ppl = math.exp(min(val_loss, 20))
            train_ppl = math.exp(min(accum_loss, 20))

            log_data.append({
                "step": step,
                "train_loss": round(accum_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 1),
            })

            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), run_dir / "best.pt")
                improved = " ★"

            print(f"  ► eval step {step:>6d} | val_loss {val_loss:.4f} | "
                  f"val_ppl {val_ppl:.1f} | train_ppl {train_ppl:.1f}{improved}")

            with open(run_dir / "log.json", "w") as f:
                json.dump(log_data, f, indent=2)
            with open(run_dir / "spectral_log.json", "w") as f:
                json.dump(spectral_log, f, indent=2)

            model.train()

    # Final
    elapsed = time.time() - t0
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(args.eval_steps * 2):
            x, y, mask = val_loader.get_batch()
            _, vloss = model(x, targets=y, loss_mask=mask)
            val_losses.append(vloss.item())

    final_val = np.mean(val_losses)
    final_ppl = math.exp(min(final_val, 20))

    print(f"\n{'='*60}")
    print(f"Done: {args.run_name}")
    print(f"  Final val loss: {final_val:.4f} PPL: {final_ppl:.1f}")
    print(f"  Best val loss:  {best_val_loss:.4f} PPL: {math.exp(min(best_val_loss,20)):.1f}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Weight compression: {std_weight/weight_params:.1f}x")

    # Final spectral state
    print(f"\n  Final spectral state:")
    for name, info in model.spectral_summary().items():
        print(f"    {name}: σ₁={info['sigma1']:.4f} α={info['alpha']:.4f}")

    torch.save(model.state_dict(), run_dir / "final.pt")
    summary = {
        "run_name": args.run_name,
        "final_val_loss": round(final_val, 4),
        "final_val_ppl": round(final_ppl, 1),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_ppl": round(math.exp(min(best_val_loss, 20)), 1),
        "total_time_min": round(elapsed / 60, 1),
        "total_params": total_params,
        "weight_params": weight_params,
        "compression": round(std_weight / weight_params, 1),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
