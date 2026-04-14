#!/usr/bin/env python3
"""Phase 1.5 verification gate: three-gate check on the Phase 1 CPT checkpoint.

Gates:
    1. Biographical recall — overall correct_rate >= 0.70, and every baseline
       gap_category shows >= 2x improvement (or any positive delta if
       baseline == 0).
    2. phi-structure preserved — per-type free-alpha fit; attn_o within 5% of
       1/3, all other types within 10% of their base-model value.
    3. General-knowledge retention — forget_ppl / forget_ppl_base <= 1.10.

If all pass: copy `best.pt` -> `phase1_verified.pt` and exit 0.
If any fail: emit `knob_suggestion.md` and exit 1.

The --dry-run path consumes a JSON fixture with fabricated gate inputs and
does NOT import torch / transformers — used by the offline test.

Usage:
    python3 -u scripts/phase1_gate.py \\
        --checkpoint runs/ray/phase1/best.pt \\
        --probe-snapshot runs/ray/phase1/probe_scored.json \\
        --baseline-tier runs/ray/phase0/tier.json \\
        --base-alphas runs/ray/phase0/base_alphas.json \\
        --forget-bin corpora/forgetting_guard/wikitext_val.bin \\
        --forget-ppl-base 12.34 \\
        --model-config google/gemma-4-31b-it \\
        --output-dir runs/ray/phase1_5 \\
        [--trust-remote-code] [--dry-run --dry-run-fixture FIX.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ATTN_O_TARGET = 1.0 / 3.0       # 0.3333...
ATTN_O_TOL = 0.05               # +/- 5%
OTHER_TOL = 0.10                # +/- 10%
RECALL_MIN = 0.70               # gate 1 threshold
GAP_MULTIPLIER = 2.0            # gate 1 gap category requirement
RETENTION_MAX = 1.10            # gate 3 threshold


# ---------------------------------------------------------------------------
# Gate 1 — biographical recall
# ---------------------------------------------------------------------------

def aggregate_probe_scores(probe_scores: list[dict]) -> dict:
    """Given list of {id, category, score}, return per-category + overall rates.

    Shared with Phase 0 classifier logic (see scripts/phase0_classify.py).
    """
    per_category: dict[str, dict] = {}
    for r in probe_scores:
        cat = r['category']
        score = int(r['score'])
        bucket = per_category.setdefault(cat, {'total': 0, 'at_least_1': 0, 'score_2': 0})
        bucket['total'] += 1
        if score >= 1:
            bucket['at_least_1'] += 1
        if score >= 2:
            bucket['score_2'] += 1

    total = len(probe_scores)
    at_least_1 = sum(1 for r in probe_scores if int(r['score']) >= 1)
    score_2 = sum(1 for r in probe_scores if int(r['score']) >= 2)

    category_rates = {
        cat: {
            'total': b['total'],
            'correct_rate': (b['at_least_1'] / b['total']) if b['total'] else 0.0,
            'strong_rate':  (b['score_2'] / b['total']) if b['total'] else 0.0,
        }
        for cat, b in per_category.items()
    }
    return {
        'n': total,
        'correct_rate': (at_least_1 / total) if total else 0.0,
        'strong_rate':  (score_2 / total) if total else 0.0,
        'category_rates': category_rates,
    }


def check_recall(probe_scores: list[dict], baseline: dict) -> dict:
    agg = aggregate_probe_scores(probe_scores)
    base_cats = baseline.get('category_rates', {})
    gap_cats = set(baseline.get('gap_categories', []))

    per_cat_report = {}
    gap_pass = True
    for cat, info in agg['category_rates'].items():
        base_rate = float(base_cats.get(cat, {}).get('correct_rate', 0.0))
        is_gap = cat in gap_cats
        if is_gap:
            if base_rate <= 0.0:
                required = 1e-9       # any positive rate passes
                passed = info['correct_rate'] > 0.0
            else:
                required = base_rate * GAP_MULTIPLIER
                passed = info['correct_rate'] >= required
        else:
            required = 0.0
            passed = True
        if is_gap and not passed:
            gap_pass = False
        per_cat_report[cat] = {
            'correct_rate': round(info['correct_rate'], 4),
            'baseline': round(base_rate, 4),
            'required': round(required, 4) if required > 1e-6 else 0.0,
            'is_gap': is_gap,
            'pass': bool(passed),
        }

    overall_pass = agg['correct_rate'] >= RECALL_MIN
    return {
        'pass': bool(overall_pass and gap_pass),
        'correct_rate': round(agg['correct_rate'], 4),
        'strong_rate':  round(agg['strong_rate'], 4),
        'correct_rate_required': RECALL_MIN,
        'overall_pass': bool(overall_pass),
        'gap_pass': bool(gap_pass),
        'per_category': per_cat_report,
    }


# ---------------------------------------------------------------------------
# Gate 2 — phi-structure preserved
# ---------------------------------------------------------------------------

def check_phi(alphas: dict[str, float], base_alphas: dict[str, float]) -> dict:
    per_type = {}
    any_fail = False
    attn_o_alpha = float(alphas.get('attn_o', float('nan')))

    # attn_o: hard target 1/3 within +/-5%
    lo = ATTN_O_TARGET * (1 - ATTN_O_TOL)
    hi = ATTN_O_TARGET * (1 + ATTN_O_TOL)
    attn_o_pass = lo <= attn_o_alpha <= hi
    per_type['attn_o'] = {
        'alpha': round(attn_o_alpha, 6),
        'base': round(ATTN_O_TARGET, 6),
        'delta_pct': round((attn_o_alpha - ATTN_O_TARGET) / ATTN_O_TARGET * 100, 3)
                      if ATTN_O_TARGET else 0.0,
        'tolerance_pct': ATTN_O_TOL * 100,
        'pass': bool(attn_o_pass),
    }
    if not attn_o_pass:
        any_fail = True

    # other types: within 10% of base
    for t, base_val in base_alphas.items():
        if t == 'attn_o':
            continue
        if t not in alphas:
            per_type[t] = {
                'alpha': None, 'base': round(float(base_val), 6),
                'delta_pct': None, 'tolerance_pct': OTHER_TOL * 100,
                'pass': False, 'note': 'missing from Phase 1 alphas',
            }
            any_fail = True
            continue
        a = float(alphas[t])
        b = float(base_val)
        if b == 0:
            passed = a == 0
            delta_pct = 0.0 if passed else float('inf')
        else:
            delta_pct = (a - b) / b * 100
            passed = abs(delta_pct) <= OTHER_TOL * 100
        per_type[t] = {
            'alpha': round(a, 6),
            'base': round(b, 6),
            'delta_pct': round(delta_pct, 3) if delta_pct != float('inf') else None,
            'tolerance_pct': OTHER_TOL * 100,
            'pass': bool(passed),
        }
        if not passed:
            any_fail = True

    return {
        'pass': not any_fail,
        'attn_o_alpha': round(attn_o_alpha, 6),
        'per_type': per_type,
    }


# ---------------------------------------------------------------------------
# Gate 3 — general-knowledge retention
# ---------------------------------------------------------------------------

def check_retention(forget_ppl: float, forget_ppl_base: float) -> dict:
    if forget_ppl_base <= 0:
        # Undefined baseline; fail closed.
        return {
            'pass': False,
            'forget_ppl': float(forget_ppl),
            'forget_ppl_base': float(forget_ppl_base),
            'ratio': None,
            'ratio_max': RETENTION_MAX,
            'note': 'forget_ppl_base is non-positive',
        }
    ratio = float(forget_ppl) / float(forget_ppl_base)
    return {
        'pass': bool(ratio <= RETENTION_MAX),
        'forget_ppl': round(float(forget_ppl), 4),
        'forget_ppl_base': round(float(forget_ppl_base), 4),
        'ratio': round(ratio, 4),
        'ratio_max': RETENTION_MAX,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _mark(b: bool) -> str:
    return 'PASS' if b else 'FAIL'


def render_markdown(result: dict) -> str:
    lines = []
    lines.append(f'# Phase 1.5 Gate — {_mark(result["all_pass"])}')
    lines.append('')

    g1 = result['gene_recall']
    lines.append(f'## Gate 1: Biographical recall — {_mark(g1["pass"])}')
    lines.append('')
    lines.append(f'- overall correct_rate: **{g1["correct_rate"]:.3f}** '
                 f'(required >= {g1["correct_rate_required"]:.2f}) '
                 f'{_mark(g1["overall_pass"])}')
    lines.append(f'- gap-category 2x check: {_mark(g1["gap_pass"])}')
    lines.append('')
    lines.append('| category | rate | baseline | required | gap | pass |')
    lines.append('|---|---|---|---|---|---|')
    for cat, info in sorted(g1['per_category'].items()):
        lines.append(f'| {cat} | {info["correct_rate"]:.3f} | {info["baseline"]:.3f} | '
                     f'{info["required"]:.3f} | {"yes" if info["is_gap"] else "no"} | '
                     f'{_mark(info["pass"])} |')
    lines.append('')

    g2 = result['phi_integrity']
    lines.append(f'## Gate 2: phi-structure preserved — {_mark(g2["pass"])}')
    lines.append('')
    lines.append(f'- attn_o alpha: **{g2["attn_o_alpha"]:.4f}** '
                 f'(target {ATTN_O_TARGET:.4f} +/- {ATTN_O_TOL*100:.0f}%)')
    lines.append('')
    lines.append('| type | alpha | base | delta % | tol % | pass |')
    lines.append('|---|---|---|---|---|---|')
    for t, info in sorted(g2['per_type'].items()):
        alpha = info['alpha']
        alpha_s = f'{alpha:.4f}' if alpha is not None else '—'
        dp = info['delta_pct']
        dp_s = f'{dp:+.2f}' if dp is not None else '—'
        lines.append(f'| {t} | {alpha_s} | {info["base"]:.4f} | {dp_s} | '
                     f'{info["tolerance_pct"]:.0f} | {_mark(info["pass"])} |')
    lines.append('')

    g3 = result['general_retention']
    lines.append(f'## Gate 3: General-knowledge retention — {_mark(g3["pass"])}')
    lines.append('')
    ratio = g3.get('ratio')
    ratio_s = f'{ratio:.4f}' if ratio is not None else '—'
    lines.append(f'- forget_ppl: **{g3["forget_ppl"]:.3f}** / base '
                 f'{g3["forget_ppl_base"]:.3f} = {ratio_s} '
                 f'(max {g3["ratio_max"]:.2f})')
    lines.append('')
    return '\n'.join(lines) + '\n'


def render_knob_suggestion(result: dict, current: dict) -> str:
    """Emit targeted knob suggestions for any failing gate."""
    lines = ['# Phase 1.5 Gate — knob suggestions', '']
    lines.append('One or more gates failed. Suggested remediations '
                 '(apply selectively, re-run Phase 1):')
    lines.append('')

    cur_attn_o = current.get('attn_o_weight')
    cur_lr     = current.get('lr')
    cur_lam    = current.get('harmonic_lambda')

    def _fmt(x):
        return f'{x:g}' if isinstance(x, (int, float)) else '<unknown>'

    if not result['phi_integrity']['pass']:
        attn_o_info = result['phi_integrity']['per_type'].get('attn_o', {})
        alpha = attn_o_info.get('alpha')
        lines.append('## Gate 2 — phi-structure drift')
        lines.append('')
        if alpha is not None:
            lines.append(f'- attn_o alpha drifted to **{alpha:.4f}** '
                         f'(target {ATTN_O_TARGET:.4f}).')
        lines.append(f'- Increase `--attn-o-weight` from {_fmt(cur_attn_o)} '
                     f'to {_fmt(cur_attn_o*2.5) if isinstance(cur_attn_o,(int,float)) else "2.5x current"}.')
        lines.append(f'- OR lower `--lr` from {_fmt(cur_lr)} to '
                     f'{_fmt(cur_lr*0.5) if isinstance(cur_lr,(int,float)) else "0.5x current"}.')
        # flag any other drifted types
        drifted = [t for t, info in result['phi_integrity']['per_type'].items()
                   if t != 'attn_o' and not info.get('pass')]
        if drifted:
            lines.append(f'- Other drifted types: {", ".join(drifted)} '
                         f'— consider raising `--harmonic-lambda`.')
        lines.append('')

    if not result['general_retention']['pass']:
        lines.append('## Gate 3 — general-knowledge forgetting')
        lines.append('')
        ratio = result['general_retention'].get('ratio')
        if ratio is not None:
            lines.append(f'- forget_ppl ratio = {ratio:.3f} (max {RETENTION_MAX}).')
        lines.append(f'- Raise `--harmonic-lambda` by 2x '
                     f'(from {_fmt(cur_lam)} to '
                     f'{_fmt(cur_lam*2) if isinstance(cur_lam,(int,float)) else "2x current"}).')
        lines.append(f'- OR lower `--lr` (from {_fmt(cur_lr)} to '
                     f'{_fmt(cur_lr*0.5) if isinstance(cur_lr,(int,float)) else "0.5x current"}).')
        lines.append('- Consider a larger forgetting-guard slice to stabilize the estimate.')
        lines.append('')

    if not result['gene_recall']['pass']:
        g1 = result['gene_recall']
        lines.append('## Gate 1 — biographical recall')
        lines.append('')
        if not g1['overall_pass']:
            lines.append(f'- overall correct_rate {g1["correct_rate"]:.3f} '
                         f'< {g1["correct_rate_required"]:.2f}. '
                         f'Phase 1 likely did not converge — extend `--max-steps` '
                         f'or lower `--lr` and re-run.')
        if not g1['gap_pass']:
            failing = [cat for cat, info in g1['per_category'].items()
                       if info['is_gap'] and not info['pass']]
            lines.append(f'- gap categories not hitting 2x: {", ".join(failing)}. '
                         f'Enlarge corpus coverage for these categories, or add '
                         f'targeted narratives before re-running Phase 1.')
        lines.append('')

    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Real-path helpers (NOT imported in --dry-run)
# ---------------------------------------------------------------------------

def load_probe_snapshot(path: Path) -> list[dict]:
    """Load a scored probe snapshot JSON produced by phase0_probe + scoring.

    Accepts either {results: [{id, category, score, ...}, ...]} or a bare list.
    If `score` is missing, attempt to parse from a sibling markdown.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        results = data
    else:
        results = data.get('results') or data.get('probe_scores') or []
    out = []
    for r in results:
        if 'score' not in r:
            raise SystemExit(f'probe snapshot {path} has entries missing "score" — '
                             f'was the snapshot produced by a human-scoring pass?')
        out.append({'id': r['id'], 'category': r['category'], 'score': r['score']})
    if not out:
        raise SystemExit(f'probe snapshot {path} has no scored results')
    return out


def compute_alphas_from_checkpoint(checkpoint: Path, model_config: str,
                                    trust_remote_code: bool) -> dict[str, float]:
    """Load Phase 1 checkpoint state_dict, SVD each 2D weight, fit per-type mean alpha.

    Reuses fit_free_alpha / classify_layer from scripts/free_alpha_analysis.
    Heavy: imports torch. Not called in --dry-run.
    """
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from free_alpha_analysis import fit_free_alpha, classify_layer  # type: ignore

    # layer_types for mixed-attention models (Gemma 4)
    layer_types = None
    try:
        from transformers import AutoConfig  # type: ignore
        cfg = AutoConfig.from_pretrained(model_config, trust_remote_code=trust_remote_code)
        if hasattr(cfg, 'text_config'):
            layer_types = getattr(cfg.text_config, 'layer_types', None)
        elif hasattr(cfg, 'layer_types'):
            layer_types = cfg.layer_types
    except Exception as e:
        print(f'[warn] could not load config {model_config}: {e}', flush=True)

    print(f'[phi-gate] loading checkpoint {checkpoint}', flush=True)
    sd = torch.load(str(checkpoint), map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    by_type: dict[str, list[float]] = {}
    for name, W in sd.items():
        if not hasattr(W, 'dim') or W.dim() != 2:
            continue
        if 'weight' not in name:
            continue
        if min(W.shape) < 64:
            continue
        try:
            Wf = W.float().cpu()
            _, S, _ = torch.linalg.svd(Wf, full_matrices=False)
            fit = fit_free_alpha(S.numpy())
        except Exception as e:
            print(f'[phi-gate] SVD/fit failed for {name}: {e}', flush=True)
            continue
        if fit is None:
            continue
        t = classify_layer(name, layer_types=layer_types)
        by_type.setdefault(t, []).append(float(fit['alpha']))

    alphas = {t: float(np.mean(vs)) for t, vs in by_type.items() if vs}
    print(f'[phi-gate] fitted {sum(len(v) for v in by_type.values())} layers '
          f'across {len(alphas)} types', flush=True)
    return alphas


def compute_forget_ppl(checkpoint: Path, model_config: str, forget_bin: Path,
                        trust_remote_code: bool, eval_batches: int = 8,
                        window: int = 2048, batch_size: int = 1) -> float:
    """Eval forget_ppl by loading checkpoint into from_config model + running eval_ppl."""
    import math
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from wavegpt.dataloader import WaveDataLoader  # type: ignore
    from phase1_cpt import eval_ppl  # type: ignore
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = AutoConfig.from_pretrained(model_config, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    sd = torch.load(str(checkpoint), map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'[retention-gate] {len(missing)} missing keys (first 3: {missing[:3]})',
              flush=True)
    if unexpected:
        print(f'[retention-gate] {len(unexpected)} unexpected keys', flush=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    loader = WaveDataLoader(str(forget_bin), batch_size, window, device='cpu')
    _, ppl = eval_ppl(model, loader, eval_batches, device)
    return float(ppl)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--probe-snapshot',
                   help='JSON snapshot of Phase 1 probe run with per-probe score '
                        '(required unless --dry-run).')
    p.add_argument('--baseline-tier', required=True,
                   help='runs/<subject>/phase0/tier.json from phase0_classify.py')
    p.add_argument('--base-alphas', required=True,
                   help='JSON: {type: base_alpha_value} from Phase 0 free-alpha run')
    p.add_argument('--forget-bin', required=True)
    p.add_argument('--forget-ppl-base', type=float, required=True)
    p.add_argument('--model-config', required=True,
                   help='HF id or local path for AutoConfig (needed for layer_types).')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--trust-remote-code', action='store_true')
    # current training knobs — echoed into knob_suggestion.md
    p.add_argument('--current-attn-o-weight', type=float, default=10.0)
    p.add_argument('--current-lr', type=float, default=1e-5)
    p.add_argument('--current-harmonic-lambda', type=float, default=0.01)
    # dry-run
    p.add_argument('--dry-run', action='store_true',
                   help='Skip all heavy model loads; read inputs from --dry-run-fixture.')
    p.add_argument('--dry-run-fixture',
                   help='JSON: {probe_scores: [...], alphas: {...}, forget_ppl: float}')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(Path(args.baseline_tier).read_text())
    base_alphas = json.loads(Path(args.base_alphas).read_text())

    # -------- gather gate inputs --------
    if args.dry_run:
        if not args.dry_run_fixture:
            sys.exit('--dry-run requires --dry-run-fixture')
        fx = json.loads(Path(args.dry_run_fixture).read_text())
        probe_scores = fx['probe_scores']
        alphas = fx['alphas']
        forget_ppl = float(fx['forget_ppl'])
    else:
        if not args.probe_snapshot:
            sys.exit('--probe-snapshot required (or use --dry-run)')
        probe_scores = load_probe_snapshot(Path(args.probe_snapshot))
        alphas = compute_alphas_from_checkpoint(
            Path(args.checkpoint), args.model_config, args.trust_remote_code)
        forget_ppl = compute_forget_ppl(
            Path(args.checkpoint), args.model_config, Path(args.forget_bin),
            args.trust_remote_code)

    # -------- run gates --------
    g1 = check_recall(probe_scores, baseline)
    g2 = check_phi(alphas, base_alphas)
    g3 = check_retention(forget_ppl, args.forget_ppl_base)

    result = {
        'gene_recall': g1,
        'phi_integrity': g2,
        'general_retention': g3,
        'all_pass': bool(g1['pass'] and g2['pass'] and g3['pass']),
    }

    # -------- write outputs --------
    (out_dir / 'phase1_gate.json').write_text(json.dumps(result, indent=2))
    (out_dir / 'phase1_gate.md').write_text(render_markdown(result))

    knob_path = out_dir / 'knob_suggestion.md'
    if not result['all_pass']:
        current = {
            'attn_o_weight': args.current_attn_o_weight,
            'lr': args.current_lr,
            'harmonic_lambda': args.current_harmonic_lambda,
        }
        knob_path.write_text(render_knob_suggestion(result, current))
    elif knob_path.exists():
        knob_path.unlink()

    # -------- promote --------
    if result['all_pass']:
        if not args.dry_run:
            dst = out_dir / 'phase1_verified.pt'
            shutil.copy(args.checkpoint, dst)
            print(f'[phase1_gate] PROMOTED checkpoint -> {dst}', flush=True)
        else:
            print('[phase1_gate] all_pass=True (dry-run; skipping promote)', flush=True)
        print('[phase1_gate] result=PASS', flush=True)
        sys.exit(0)
    else:
        print('[phase1_gate] result=FAIL — see '
              f'{out_dir/"phase1_gate.md"} and {knob_path}', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
