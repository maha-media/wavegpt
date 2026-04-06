#!/usr/bin/env python3
"""
Train WaveGPT — harmonic training for small language models.

Usage:
    python scripts/train.py --data-dir data/my-corpus --steps 15000 --data-curriculum --collapse-alpha 0.05
    python scripts/train.py --data-dir data/my-corpus --steps 5000 --dry-run
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavegpt.model import WaveGPT, WaveGPTConfig
from wavegpt.dataloader import WaveDataLoader, HarmonicCurriculumLoader


# ── Model configs ──

MODELS = {
    "small":  WaveGPTConfig(block_size=256, n_layer=4, n_head=4, n_embd=256),
    "medium": WaveGPTConfig(block_size=512, n_layer=6, n_head=6, n_embd=384),
    "large":  WaveGPTConfig(block_size=512, n_layer=8, n_head=8, n_embd=512),
}


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, data_loader, n_batches: int, device: str) -> float:
    """Estimate standard (unweighted) val loss for fair comparison."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = next(data_loader)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_sample(model, enc, prompt: str, max_tokens: int = 100, device: str = "cpu") -> str:
    """Generate text from a prompt."""
    model.eval()
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
    model.train()
    return enc.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Train WaveGPT")
    parser.add_argument("--model", default="small", choices=list(MODELS.keys()))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--collapse-alpha", type=float, default=0.0,
                        help="Anti-collapse regularization strength (try 0.05)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--data-dir", default="data/corpus")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-curriculum", action="store_true",
                        help="Augmented-first data curriculum (3-phase)")
    parser.add_argument("--harmonic-curriculum", action="store_true",
                        help="Circle of fifths curriculum (4-phase C→G→D→A)")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Custom prompts for sample generation")
    args = parser.parse_args()

    # ── Config ──
    config = MODELS[args.model]
    if args.dropout is not None:
        config.dropout = args.dropout
    if not args.output_dir:
        args.output_dir = f"data/training/{args.model}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data paths ──
    train_path = str(Path(args.data_dir) / "rai_train_000.bin")
    val_path = str(Path(args.data_dir) / "rai_val_000.bin")

    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}")
        print("Run: wavegpt export --help")
        sys.exit(1)

    # ── Print config ──
    print("=" * 60)
    print(f"  WaveGPT Training")
    print("=" * 60)
    print(f"  Model:     {args.model} ({config.n_layer}L, {config.n_head}H, {config.n_embd}E)")
    if args.harmonic_curriculum:
        print(f"  Curricul.: HARMONIC (C→G→D→A)")
    elif args.data_curriculum:
        print(f"  Curricul.: DATA (3-phase aug-first)")
    else:
        print(f"  Curricul.: OFF")
    if args.collapse_alpha > 0:
        print(f"  Collapse:  alpha={args.collapse_alpha}")
    print(f"  Steps:     {args.steps}")
    print(f"  Batch:     {args.batch_size} × {config.block_size} = {args.batch_size * config.block_size:,} tok/step")
    print(f"  Dropout:   {config.dropout}")
    print(f"  LR:        {args.lr} → {args.min_lr} (cosine, {args.warmup} warmup)")
    print(f"  Device:    {args.device}")
    print(f"  Output:    {args.output_dir}")
    print()

    if args.dry_run:
        print("(dry run — not training)")
        return

    # ── Build model ──
    print("Building model...")
    model = WaveGPT(config, collapse_alpha=args.collapse_alpha)
    model = model.to(args.device)
    n_params = model.count_params()
    print(f"  Parameters: {n_params:,} ({n_params*4/1e6:.1f} MB fp32)")

    if args.compile and hasattr(torch, "compile"):
        print("  Compiling model...")
        model = torch.compile(model)

    # ── Data loaders ──
    print("Loading data...")
    train_loader = WaveDataLoader(train_path, args.batch_size, config.block_size, device=args.device)
    val_loader = WaveDataLoader(val_path, args.batch_size, config.block_size, device=args.device) if os.path.exists(val_path) else None
    print(f"  Train: {train_loader.n_tokens:,} tokens ({len(train_loader)} batches/epoch)")
    if val_loader:
        print(f"  Val:   {val_loader.n_tokens:,} tokens")

    # Harmonic curriculum loader
    harmonic_loader = None
    if args.harmonic_curriculum:
        harmonic_loader = HarmonicCurriculumLoader(
            data_dir=args.data_dir, batch_size=args.batch_size,
            block_size=config.block_size, device=args.device,
        )
        print(f"  Harmonic layers: {list(harmonic_loader.layers.keys())} ({harmonic_loader.n_tokens:,} tokens)")

    # Data curriculum: augmented-first loader
    aug_loader = None
    if args.data_curriculum and not args.harmonic_curriculum:
        aug_path = str(Path(args.data_dir) / "rai_aug_000.bin")
        if os.path.exists(aug_path):
            aug_loader = WaveDataLoader(aug_path, args.batch_size, config.block_size, device=args.device)
            print(f"  Aug:   {aug_loader.n_tokens:,} tokens (phases 1-2)")
        else:
            print(f"  WARNING: {aug_path} not found, data curriculum disabled")
            args.data_curriculum = False

    # ── Optimizer ──
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # ── Tokenizer ──
    enc = tiktoken.get_encoding("gpt2")

    # ── Training log ──
    log = {
        "config": {
            "model": args.model, "steps": args.steps,
            "batch_size": args.batch_size, "lr": args.lr, "n_params": n_params,
            "n_embd": config.n_embd, "n_layer": config.n_layer, "n_head": config.n_head,
            "data_curriculum": args.data_curriculum,
            "harmonic_curriculum": args.harmonic_curriculum,
            "collapse_alpha": args.collapse_alpha,
        },
        "train_loss": [], "val_loss": [], "samples": [], "timing": {},
    }

    # ── Training loop ──
    print(f"\nTraining for {args.steps} steps...")
    print(f"{'Step':>6s} | {'Train Loss':>10s} | {'Val Loss':>10s} | {'LR':>10s} | {'tok/s':>8s}")
    print("-" * 60)

    model.train()
    t0 = time.time()
    tokens_processed = 0
    best_val_loss = float("inf")

    prompts = args.prompts or ["The", "In the beginning", "Technology will"]

    for step in range(args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Data selection
        if args.harmonic_curriculum and harmonic_loader is not None:
            x, y = harmonic_loader.get_batch(step, args.steps)
        elif args.data_curriculum and aug_loader is not None:
            phase1_end = int(args.steps * 0.3)
            phase2_end = int(args.steps * 0.7)
            if step < phase1_end:
                active_loader = aug_loader
            elif step < phase2_end:
                active_loader = aug_loader if step % 2 == 0 else train_loader
            else:
                active_loader = train_loader
            x, y = next(active_loader)
        else:
            x, y = next(train_loader)

        x, y = x.to(args.device), y.to(args.device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        tokens_processed += args.batch_size * config.block_size
        train_loss = loss.item()

        if step % 10 == 0:
            log["train_loss"].append({"step": step, "loss": train_loss})

        if step % args.val_every == 0 or step == args.steps - 1:
            val_loss = 0.0
            if val_loader:
                val_loss = estimate_loss(model, val_loader, n_batches=20, device=args.device)
                log["val_loss"].append({"step": step, "loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            val_str = f"{val_loss:>10.4f}" if val_loss > 0 else f"{'—':>10s}"
            print(f"{step:>6d} | {train_loss:>10.4f} | {val_str} | {lr:>10.6f} | {tok_per_sec:>8.0f}")

        if step % args.sample_every == 0 or step == args.steps - 1:
            for prompt in prompts:
                text = generate_sample(model, enc, prompt, max_tokens=60, device=args.device)
                log["samples"].append({"step": step, "prompt": prompt, "text": text})
                if step % args.sample_every == 0:
                    print(f"  [{prompt}] → {text[:120]}...")

        if step > 0 and (step % args.save_every == 0 or step == args.steps - 1):
            ckpt_path = out_dir / f"wavegpt_step{step}.pt"
            torch.save({
                "model_state_dict": model.state_dict() if not hasattr(model, "_orig_mod") else model._orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss if val_loader else None,
            }, ckpt_path)

    # ── Final stats ──
    elapsed = time.time() - t0
    log["timing"] = {
        "total_seconds": elapsed,
        "tokens_processed": tokens_processed,
        "tok_per_sec": tokens_processed / elapsed,
    }

    print()
    print("=" * 60)
    print(f"  Training complete")
    print(f"  Time:       {elapsed:.1f}s ({tokens_processed/elapsed:,.0f} tok/s)")
    print(f"  Final train: {train_loss:.4f}")
    if val_loader:
        print(f"  Best val:    {best_val_loss:.4f} (perplexity {math.exp(best_val_loss):.1f})")
    print(f"  Saved to:    {args.output_dir}")
    print("=" * 60)

    log_path = out_dir / "loss_log.json"
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"  Log: {log_path}")

    final_path = out_dir / "wavegpt_final.pt"
    torch.save({
        "model_state_dict": model.state_dict() if not hasattr(model, "_orig_mod") else model._orig_mod.state_dict(),
        "config": config, "step": args.steps - 1,
        "train_loss": train_loss, "val_loss": best_val_loss if val_loader else None,
    }, final_path)
    print(f"  Final model: {final_path}")


if __name__ == "__main__":
    main()
