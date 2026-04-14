"""Phase 1 continued pretraining (CPT) with harmonic regularization.

Full-weight bf16 training of an HF causal LM on a curated corpus, with a
forgetting-guard eval against a held-out slice (e.g. wikitext). FSDP is
supplied externally via `accelerate launch`; this script just calls
`Accelerator()` and `accelerator.prepare(...)`.

The `--smoke` mode swaps the HF model for a tiny toy built from
SpectralLinear layers named q_proj/k_proj/v_proj/o_proj so that
harmonic_regularization(type_aware=True) exercises the real code path
(not a stub).
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
from wavegpt.harmonic_prior import harmonic_regularization
from wavegpt.spectral_linear import SpectralLinear


# ---------------------------------------------------------------------------
# Toy model for --smoke (CPU-runnable, exercises harmonic_regularization)
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
    """Tiny toy model used only in --smoke mode.

    Returns an object with `.loss` so the training loop can treat it like
    an HF CausalLMOutput.
    """

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
    p.add_argument('--harmonic-lambda', type=float, default=0.01)
    p.add_argument('--attn-o-weight', type=float, default=10.0)
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

    # --smoke forces CPU + tiny window/batch; real path uses Accelerator (FSDP).
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

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    if accelerator is not None:
        model, optim = accelerator.prepare(model, optim)

    log_path = out_dir / 'training_log.json'
    log: list[dict] = []

    def flush_log():
        if is_main:
            log_path.write_text(json.dumps(log, indent=2))

    # ---------- eval-only shortcut ----------
    if args.eval_only:
        val_loss, val_ppl = eval_ppl(model, val_loader, args.eval_batches, device)
        forget_loss, forget_ppl = eval_ppl(model, forget_loader, args.eval_batches, device)
        log.append({
            'step': 0,
            'train_loss': None,
            'harmonic_loss': None,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'forget_loss': forget_loss,
            'forget_ppl': forget_ppl,
            'lr': args.lr,
        })
        flush_log()
        print(f"[eval-only] val_ppl={val_ppl:.3f} forget_ppl={forget_ppl:.3f}", flush=True)
        return

    # ---------- training ----------
    model.train()
    best_val_ppl = float('inf')
    non_improving = 0
    forget_base = args.forget_ppl_base

    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        x, _ = next(train_loader)
        x = x.to(device)
        out = model(input_ids=x, labels=x)
        ce = out.loss

        hp = harmonic_regularization(
            model,
            type_aware=True,
            attn_o_weight=args.attn_o_weight,
        )
        hp = hp.to(ce.device) if isinstance(hp, torch.Tensor) else torch.tensor(float(hp), device=ce.device)
        loss = ce + args.harmonic_lambda * hp

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        optim.step()
        optim.zero_grad()

        # eval boundary: every eval-every steps AND at the last step
        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss, val_ppl = eval_ppl(model, val_loader, args.eval_batches, device)
            forget_loss, forget_ppl = eval_ppl(model, forget_loader, args.eval_batches, device)
            entry = {
                'step': step,
                'train_loss': float(ce.item()),
                'harmonic_loss': float(hp.item()) if isinstance(hp, torch.Tensor) else float(hp),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'forget_loss': forget_loss,
                'forget_ppl': forget_ppl,
                'lr': args.lr,
                'elapsed_s': time.time() - t0,
            }
            log.append(entry)
            flush_log()
            print(
                f"[step {step}/{args.max_steps}] train={ce.item():.4f} "
                f"hp={entry['harmonic_loss']:.4e} val_ppl={val_ppl:.3f} "
                f"forget_ppl={forget_ppl:.3f}",
                flush=True,
            )

            # Soft forgetting-guard
            if forget_base is None:
                forget_base = forget_ppl
            elif forget_base > 0 and forget_ppl / forget_base > 1.10:
                print(
                    f"  WARNING: forget_ppl {forget_ppl:.3f} > 1.10 * base "
                    f"{forget_base:.3f} — knowledge drift",
                    flush=True,
                )

            # best-on-val checkpoint
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
