"""Phase 1 continued pretraining (CPT) with attn_o LR protection.

Full-weight bf16 training of an HF causal LM on a curated corpus, with a
forgetting-guard eval against a held-out slice (e.g. wikitext). FSDP is
supplied externally via ``accelerate launch``; this script just calls
``Accelerator()`` and ``accelerator.prepare(...)``.

**Design note — why no harmonic regularizer here.** An earlier version of
this trainer added ``harmonic_regularization()`` to the CE loss to pull
every layer's spectrum toward its φ-harmonic target (and ``attn_o`` toward
1/3 with a multiplier). We removed it. The universal 1/3 exponent on
``attn_o`` already exists in the pretrained weights — pretraining ran long
enough for SGD to find the φ-attractor. Our job during CPT is to **not
erase it**, not to pull it toward anything. A harmonic term in the loss
NaNs at scale (memory: ``feedback_no_phi_constraint.md``) and contradicts
the core thesis that φ is emergent, not constrainable.

Instead, ``attn_o`` is protected the value-agnostic way: a smaller LR on
those parameters. Other types train at ``--lr``; ``attn_o`` at
``--lr * --attn-o-lr-mult`` (default 0.1). This preserves whatever
exponent pretraining landed on. If Gemma-4's ``attn_o`` had pretrained to
α=0.25 for some reason, we'd be preserving 0.25.

The ``--smoke`` mode swaps the HF model for a tiny toy built from
SpectralLinear layers named q_proj/k_proj/v_proj/o_proj so the parameter
grouping (``o_proj`` → reduced LR) exercises the real code path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wavegpt.dataloader import WaveDataLoader
from wavegpt.spectral_linear import SpectralLinear


# Parameter names (any substring) whose owning module is an attn_o projection.
# Matches HF conventions: GPT-style `c_proj`, LLaMA/Gemma-style `o_proj`,
# Qwen/BERT-style `out_proj`. See CLAUDE.md layer-type conventions.
_ATTN_O_NAME_PARTS = (".o_proj.", ".out_proj.", ".c_proj.")


def _is_attn_o_param(qualified_name: str) -> bool:
    return any(part in qualified_name for part in _ATTN_O_NAME_PARTS)


def build_param_groups(model: nn.Module, base_lr: float, attn_o_mult: float):
    """Split parameters into two groups: attn_o (LR = base_lr * attn_o_mult)
    and everything else (LR = base_lr). Returns (groups, n_attn_o, n_other)
    for logging.
    """
    attn_o_params, other_params = [], []
    n_attn_o = n_other = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_attn_o_param(name):
            attn_o_params.append(p)
            n_attn_o += p.numel()
        else:
            other_params.append(p)
            n_other += p.numel()
    groups = []
    if attn_o_params:
        groups.append({"params": attn_o_params, "lr": base_lr * attn_o_mult,
                       "name": "attn_o"})
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "name": "other"})
    return groups, n_attn_o, n_other


# ---------------------------------------------------------------------------
# Toy model for --smoke (CPU-runnable, exercises parameter-group routing)
# ---------------------------------------------------------------------------

class _SmokeAttnBlock(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.q_proj = self._spectral(dim, rank)
        self.k_proj = self._spectral(dim, rank)
        self.v_proj = self._spectral(dim, rank)
        self.o_proj = self._spectral(dim, rank)

    @staticmethod
    def _spectral(dim: int, rank: int) -> SpectralLinear:
        W = torch.randn(dim, dim) * 0.1
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return SpectralLinear(U[:, :rank].contiguous(),
                              S[:rank].contiguous(),
                              Vh[:rank, :].t().contiguous(),
                              mode='per_mode')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        return self.o_proj(h)


class SmokeModel(nn.Module):
    """Tiny toy model used only in --smoke mode."""

    def __init__(self, vocab: int = 65536, dim: int = 32, rank: int = 8, n_blocks: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([_SmokeAttnBlock(dim, rank) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, vocab, bias=False)

    class _Out:
        __slots__ = ("loss", "logits")

        def __init__(self, loss, logits):
            self.loss = loss
            self.logits = logits

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x) + x
        logits = self.head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return SmokeModel._Out(loss, logits)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_ppl(model, loader: WaveDataLoader, n_batches: int, device: torch.device) -> tuple[float, float]:
    """Return (mean_loss, perplexity) over n_batches from loader."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = next(loader)
            x = x.to(device)
            out = model(input_ids=x, labels=x)
            if out.loss is not None and torch.isfinite(out.loss):
                total += float(out.loss.item())
                count += 1
    model.train()
    if count == 0:
        return float('inf'), float('inf')
    mean_loss = total / count
    return mean_loss, math.exp(min(mean_loss, 20))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(args):
    if args.smoke:
        return SmokeModel()
    from transformers import AutoModelForCausalLM
    kwargs = {'torch_dtype': torch.bfloat16, 'low_cpu_mem_usage': True}
    if args.trust_remote_code:
        kwargs['trust_remote_code'] = True
    return AutoModelForCausalLM.from_pretrained(args.model_dir, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=str, default=None)
    p.add_argument('--train-bin', type=str, required=True)
    p.add_argument('--val-bin', type=str, required=True)
    p.add_argument('--forget-bin', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--window', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--attn-o-lr-mult', type=float, default=0.1,
                   help='LR multiplier for attn_o parameters (o_proj / out_proj / c_proj). '
                        'Smaller = stronger preservation of the pretrained 1/3 exponent.')
    p.add_argument('--max-steps', type=int, default=6000)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--forget-ppl-base', type=float, default=None,
                   help='Baseline forget PPL; warn if current/base > 1.10.')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.window = min(args.window, 64)
        args.batch_size = 1
        device = torch.device('cpu')
        accelerator = None
        is_main = True
    else:
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        is_main = accelerator.is_main_process

    print(f"[phase1_cpt] device={device} smoke={args.smoke} eval_only={args.eval_only}", flush=True)

    model = build_model(args)
    if not args.smoke:
        model = model.to(device)

    train_loader = WaveDataLoader(args.train_bin, args.batch_size, args.window, device='cpu')
    val_loader = WaveDataLoader(args.val_bin, args.batch_size, args.window, device='cpu')
    forget_loader = WaveDataLoader(args.forget_bin, args.batch_size, args.window, device='cpu')

    param_groups, n_attn_o, n_other = build_param_groups(
        model, args.lr, args.attn_o_lr_mult
    )
    if is_main:
        print(
            f"[phase1_cpt] param groups: attn_o={n_attn_o:,} params @ lr={args.lr * args.attn_o_lr_mult:.2e} | "
            f"other={n_other:,} params @ lr={args.lr:.2e}",
            flush=True,
        )
    optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    if accelerator is not None:
        model, optim = accelerator.prepare(model, optim)

    log_path = out_dir / 'training_log.json'
    log: list[dict] = []

    def flush_log():
        if is_main:
            log_path.write_text(json.dumps(log, indent=2))

    if args.eval_only:
        val_loss, val_ppl = eval_ppl(model, val_loader, args.eval_batches, device)
        forget_loss, forget_ppl = eval_ppl(model, forget_loader, args.eval_batches, device)
        log.append({
            'step': 0,
            'train_loss': None,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'forget_loss': forget_loss,
            'forget_ppl': forget_ppl,
            'lr_other': args.lr,
            'lr_attn_o': args.lr * args.attn_o_lr_mult,
        })
        flush_log()
        print(f"[eval-only] val_ppl={val_ppl:.3f} forget_ppl={forget_ppl:.3f}", flush=True)
        return

    model.train()
    best_val_ppl = float('inf')
    non_improving = 0
    forget_base = args.forget_ppl_base

    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        x, _ = next(train_loader)
        x = x.to(device)
        out = model(input_ids=x, labels=x)
        loss = out.loss

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        optim.step()
        optim.zero_grad()

        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss, val_ppl = eval_ppl(model, val_loader, args.eval_batches, device)
            forget_loss, forget_ppl = eval_ppl(model, forget_loader, args.eval_batches, device)
            entry = {
                'step': step,
                'train_loss': float(loss.item()),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'forget_loss': forget_loss,
                'forget_ppl': forget_ppl,
                'lr_other': args.lr,
                'lr_attn_o': args.lr * args.attn_o_lr_mult,
                'elapsed_s': time.time() - t0,
            }
            log.append(entry)
            flush_log()
            print(
                f"[step {step}/{args.max_steps}] train={loss.item():.4f} "
                f"val_ppl={val_ppl:.3f} forget_ppl={forget_ppl:.3f}",
                flush=True,
            )

            if forget_base is None:
                forget_base = forget_ppl
            elif forget_base > 0 and forget_ppl / forget_base > 1.10:
                print(
                    f"  WARNING: forget_ppl {forget_ppl:.3f} > 1.10 * base "
                    f"{forget_base:.3f} — knowledge drift",
                    flush=True,
                )

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                non_improving = 0
                if is_main:
                    sd = model.state_dict() if accelerator is None else accelerator.unwrap_model(model).state_dict()
                    torch.save(sd, out_dir / 'best.pt')
                    print(f"  ★ best val_ppl={val_ppl:.3f} saved", flush=True)
            else:
                non_improving += 1
                if non_improving >= 3:
                    print(
                        f"  early-stop: {non_improving} non-improving evals "
                        f"(best val_ppl={best_val_ppl:.3f})",
                        flush=True,
                    )
                    break

    flush_log()
    print(f"[phase1_cpt] done. best val_ppl={best_val_ppl:.3f}", flush=True)


if __name__ == '__main__':
    main()
