#!/usr/bin/env python3
"""Final 4-gate validation harness for the recomposed persona model.

Gates:
    1. gene_strength    — Phase 0 factual probes *with* activator sysprompt.
                          Pass: >=90% score>=1 AND >=70% score==2.
    2. voice_fidelity   — 12-15 open-ended dialogue prompts, human-graded 1-5.
                          Pass: mean >= 4.0 AND min >= 3.
    3. dormancy         — Same dialogue prompts WITHOUT sysprompt, graded on a
                          "Ray-ness" rubric. Pass: every sample scored < 3.
    4. phi_integrity    — Free-alpha scan on recomposed model. Pass: attn_o
                          within 5% of 1/3, other types within 10% of base.

Modes:
    Generation (default)  — Run gate 1 + gate 4 automatically; emit
                            `voice_samples.md` + `dormancy_samples.md` with
                            `**score:** ?` lines; write `eval_final.json` with
                            `human_pending: true` for gates 2 & 3; exit 2.
    Grading (--grade-file) — Parse graded markdown (voice_samples.md +
                             dormancy_samples.md) and finalize `eval_final.json`.
                             Exit 0 on all-pass, 1 otherwise.

Dry-run: --dry-run --dry-run-fixture FIX.json bypasses every heavy import
(torch/transformers/HF load). Used by tests/test_eval_final.py.

Usage:
    python3 -u scripts/eval_final.py \\
        --model-dir runs/ray/phase2/recomposed \\
        --activator configs/ray_activator.txt \\
        --probes probes/ray_kurzweil.json \\
        --voice-probes probes/ray_voice.json \\
        --base-alphas runs/ray/phase0/base_alphas.json \\
        --phase0-baseline runs/ray/phase0/tier.json \\
        --output-dir runs/ray/eval_final \\
        [--model-config google/gemma-4-31b-it] [--trust-remote-code]

    # After humans grade voice_samples.md / dormancy_samples.md:
    python3 -u scripts/eval_final.py \\
        --model-dir ... --activator ... --probes ... --voice-probes ... \\
        --base-alphas ... --phase0-baseline ... \\
        --output-dir runs/ray/eval_final \\
        --grade-file runs/ray/eval_final
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Gate thresholds (match Phase 1.5 tolerances where applicable)
ATTN_O_TARGET = 1.0 / 3.0
ATTN_O_TOL = 0.05
OTHER_TOL = 0.10
GENE_CORRECT_MIN = 0.90
GENE_STRONG_MIN = 0.70
VOICE_MEAN_MIN = 4.0
VOICE_MIN_SCORE = 3
DORMANCY_MAX_SCORE = 3   # any score >= this fails (Ray-ness leak)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_HUMAN_PENDING = 2


# ---------------------------------------------------------------------------
# Gate 1 — gene_strength (factual probes + activator sysprompt)
# ---------------------------------------------------------------------------

def aggregate_probe_scores(probe_scores: list[dict]) -> dict:
    """Per-category and overall correct / strong rates."""
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


def check_gene_strength(probe_scores: list[dict], baseline: dict | None) -> dict:
    agg = aggregate_probe_scores(probe_scores)
    correct_ok = agg['correct_rate'] >= GENE_CORRECT_MIN
    strong_ok = agg['strong_rate'] >= GENE_STRONG_MIN

    improvement = None
    if baseline and 'correct_rate' in baseline:
        base_correct = float(baseline.get('correct_rate', 0.0))
        improvement = {
            'baseline_correct_rate': round(base_correct, 4),
            'delta': round(agg['correct_rate'] - base_correct, 4),
            'better': agg['correct_rate'] >= base_correct,
        }

    return {
        'pass': bool(correct_ok and strong_ok),
        'n': agg['n'],
        'correct_rate': round(agg['correct_rate'], 4),
        'strong_rate':  round(agg['strong_rate'], 4),
        'correct_rate_required': GENE_CORRECT_MIN,
        'strong_rate_required': GENE_STRONG_MIN,
        'correct_ok': bool(correct_ok),
        'strong_ok': bool(strong_ok),
        'category_rates': {
            cat: {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in info.items()}
            for cat, info in agg['category_rates'].items()
        },
        'improvement': improvement,
    }


# ---------------------------------------------------------------------------
# Gate 2 — voice_fidelity (human-graded 1-5)
# ---------------------------------------------------------------------------

def check_voice_fidelity(grades: list[int]) -> dict:
    if not grades:
        return {'pass': False, 'human_pending': False, 'mean': None, 'min': None,
                'n': 0, 'note': 'no grades supplied'}
    mean = sum(grades) / len(grades)
    gmin = min(grades)
    passed = mean >= VOICE_MEAN_MIN and gmin >= VOICE_MIN_SCORE
    return {
        'pass': bool(passed),
        'human_pending': False,
        'n': len(grades),
        'mean': round(mean, 4),
        'min': int(gmin),
        'mean_required': VOICE_MEAN_MIN,
        'min_required': VOICE_MIN_SCORE,
        'grades': [int(g) for g in grades],
    }


# ---------------------------------------------------------------------------
# Gate 3 — dormancy (human-graded Ray-ness rubric, no sysprompt)
# ---------------------------------------------------------------------------

def check_dormancy(grades: list[int]) -> dict:
    if not grades:
        return {'pass': False, 'human_pending': False, 'max': None, 'n': 0,
                'note': 'no grades supplied'}
    gmax = max(grades)
    # Any score >= DORMANCY_MAX_SCORE indicates a Ray-ness leak.
    passed = gmax < DORMANCY_MAX_SCORE
    return {
        'pass': bool(passed),
        'human_pending': False,
        'n': len(grades),
        'max': int(gmax),
        'max_allowed': DORMANCY_MAX_SCORE - 1,
        'grades': [int(g) for g in grades],
    }


# ---------------------------------------------------------------------------
# Gate 4 — phi_integrity (free-alpha scan)
# ---------------------------------------------------------------------------

def check_phi_integrity(alphas: dict[str, float],
                        base_alphas: dict[str, float]) -> dict:
    per_type = {}
    any_fail = False

    attn_o_alpha = float(alphas.get('attn_o', float('nan')))
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

    for t, base_val in base_alphas.items():
        if t == 'attn_o':
            continue
        if t not in alphas:
            per_type[t] = {
                'alpha': None, 'base': round(float(base_val), 6),
                'delta_pct': None, 'tolerance_pct': OTHER_TOL * 100,
                'pass': False, 'note': 'missing from recomposed alphas',
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
# Markdown rendering
# ---------------------------------------------------------------------------

def _mark(b) -> str:
    if b is True:
        return 'PASS'
    if b is False:
        return 'FAIL'
    return 'PENDING'


def render_samples_md(title: str, header_note: str,
                      samples: list[dict]) -> str:
    lines = [f'# {title}', '']
    lines.append(header_note)
    lines.append('')
    lines.append('---')
    lines.append('')
    for s in samples:
        lines += [
            f'## {s["id"]}',
            '',
            f'**Q:** {s["question"]}',
            '',
            '**Generated:**',
            '',
            s.get('generated', ''),
            '',
            '**score:** ?',
            '',
            '---',
            '',
        ]
    return '\n'.join(lines) + '\n'


def render_final_md(result: dict, output_dir: Path | str | None = None) -> str:
    lines = []
    overall = _mark(result['all_pass'])
    lines.append(f'# Final Eval — {overall}')
    lines.append('')

    g1 = result['gene_strength']
    lines.append(f'## Gate 1: gene_strength — {_mark(g1["pass"])}')
    lines.append('')
    lines.append(f'- n={g1["n"]}')
    lines.append(f'- correct_rate: **{g1["correct_rate"]:.3f}** '
                 f'(required >= {g1["correct_rate_required"]:.2f}) '
                 f'{_mark(g1["correct_ok"])}')
    lines.append(f'- strong_rate:  **{g1["strong_rate"]:.3f}** '
                 f'(required >= {g1["strong_rate_required"]:.2f}) '
                 f'{_mark(g1["strong_ok"])}')
    if g1.get('improvement'):
        imp = g1['improvement']
        lines.append(f'- vs Phase 0 baseline correct_rate '
                     f'{imp["baseline_correct_rate"]:.3f}: '
                     f'delta {imp["delta"]:+.3f} '
                     f'({"better" if imp["better"] else "regressed"})')
    lines.append('')

    g2 = result['voice_fidelity']
    lines.append(f'## Gate 2: voice_fidelity — {_mark(g2["pass"] if not g2.get("human_pending") else None)}')
    lines.append('')
    if g2.get('human_pending'):
        lines.append('- human grading pending — see `voice_samples.md`.')
    else:
        mean_s = f'{g2["mean"]:.3f}' if g2.get('mean') is not None else '—'
        gmin = g2.get('min')
        gmin_s = str(gmin) if gmin is not None else '—'
        lines.append(f'- n={g2.get("n", 0)}')
        lines.append(f'- mean score: **{mean_s}** '
                     f'(required >= {VOICE_MEAN_MIN:.1f})')
        lines.append(f'- min  score: **{gmin_s}** '
                     f'(required >= {VOICE_MIN_SCORE})')
    lines.append('')

    g3 = result['dormancy']
    lines.append(f'## Gate 3: dormancy — {_mark(g3["pass"] if not g3.get("human_pending") else None)}')
    lines.append('')
    if g3.get('human_pending'):
        lines.append('- human grading pending — see `dormancy_samples.md`.')
    else:
        gmax = g3.get('max')
        gmax_s = str(gmax) if gmax is not None else '—'
        lines.append(f'- n={g3.get("n", 0)}')
        lines.append(f'- max Ray-ness score: **{gmax_s}** '
                     f'(must be <= {DORMANCY_MAX_SCORE - 1}; '
                     f'any >= {DORMANCY_MAX_SCORE} = persona leak)')
    lines.append('')

    g4 = result['phi_integrity']
    lines.append(f'## Gate 4: phi_integrity — {_mark(g4["pass"])}')
    lines.append('')
    lines.append(f'- attn_o alpha: **{g4["attn_o_alpha"]:.4f}** '
                 f'(target {ATTN_O_TARGET:.4f} +/- {ATTN_O_TOL*100:.0f}%)')
    lines.append('')
    lines.append('| type | alpha | base | delta % | tol % | pass |')
    lines.append('|---|---|---|---|---|---|')
    for t, info in sorted(g4['per_type'].items()):
        alpha = info['alpha']
        alpha_s = f'{alpha:.4f}' if alpha is not None else '—'
        dp = info['delta_pct']
        dp_s = f'{dp:+.2f}' if dp is not None else '—'
        lines.append(f'| {t} | {alpha_s} | {info["base"]:.4f} | {dp_s} | '
                     f'{info["tolerance_pct"]:.0f} | {_mark(info["pass"])} |')
    lines.append('')

    # Copy-paste next-step hint — ONLY when human grading is still pending.
    if result.get('all_pass') is None and output_dir is not None:
        out = Path(output_dir)
        gene_md = out / 'gene_samples.md'
        voice_md = out / 'voice_samples.md'
        dormancy_md = out / 'dormancy_samples.md'
        lines.append('## Next step')
        lines.append('')
        lines.append('Human grading required. Score each probe in:')
        lines.append('')
        if gene_md.exists():
            lines.append(f'- `{gene_md}`')
        lines.append(f'- `{voice_md}`')
        lines.append(f'- `{dormancy_md}`')
        lines.append('')
        lines.append(
            'Edit each `**score:** ?` line to a number '
            '(0/1/2 for gene, 1-5 for voice, 1-5 for dormancy where '
            '1 = no persona leak, 5 = full Ray-voice leak).')
        lines.append('')
        lines.append('Then re-run:')
        lines.append('')
        lines.append(f'    python3 scripts/eval_final.py --grade-file {out} '
                     f'--output-dir {out}')
        lines.append('')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Grade parsing
# ---------------------------------------------------------------------------

_SCORE_RE = re.compile(r'^\*\*score:\*\*\s*(\S+)\s*$', re.MULTILINE)
_HEADING_RE = re.compile(r'^##\s+(\S+)', re.MULTILINE)


def parse_grades_from_md(path: Path) -> list[int]:
    """Return numeric scores from `**score:** N` lines. Unscored (`?`) → []."""
    txt = Path(path).read_text()
    grades: list[int] = []
    for m in _SCORE_RE.finditer(txt):
        tok = m.group(1).strip()
        if tok == '?' or tok == '':
            continue
        try:
            grades.append(int(tok))
        except ValueError:
            raise SystemExit(f'{path}: could not parse score {tok!r} as int')
    return grades


def parse_gene_grades_from_md(path: Path,
                              probe_meta: list[dict] | None = None) -> list[dict]:
    """Re-parse a graded gene_samples.md back into probe_scores records.

    Each sample in the md is delimited by `## <id>`; returns one dict per
    *scored* sample with `{id, category, score}`. Unscored (`?`) samples are
    skipped.

    If `probe_meta` (list of `{id, category}`) is supplied, categories are
    joined by id. Otherwise category defaults to 'unknown' — callers that
    need accurate category breakdown must pass meta.
    """
    txt = Path(path).read_text()
    # Find each `## <id>` heading and the following `**score:** N` line.
    heading_matches = list(_HEADING_RE.finditer(txt))
    score_matches = list(_SCORE_RE.finditer(txt))
    if not heading_matches:
        return []
    meta_by_id = {m['id']: m for m in (probe_meta or [])}

    records: list[dict] = []
    for hi, hm in enumerate(heading_matches):
        probe_id = hm.group(1).strip()
        start = hm.end()
        end = heading_matches[hi + 1].start() if hi + 1 < len(heading_matches) else len(txt)
        # First score line inside this section.
        section_score = None
        for sm in score_matches:
            if start <= sm.start() < end:
                section_score = sm.group(1).strip()
                break
        if section_score in (None, '?', ''):
            continue
        try:
            score_val = int(section_score)
        except ValueError:
            raise SystemExit(
                f'{path}: could not parse score {section_score!r} as int')
        meta = meta_by_id.get(probe_id, {})
        records.append({
            'id': probe_id,
            'category': meta.get('category', 'unknown'),
            'score': score_val,
        })
    return records


def count_score_placeholders(path: Path) -> tuple[int, int]:
    """Return (scored, unscored) counts."""
    txt = Path(path).read_text()
    scored = 0
    unscored = 0
    for m in _SCORE_RE.finditer(txt):
        tok = m.group(1).strip()
        if tok == '?' or tok == '':
            unscored += 1
        else:
            scored += 1
    return scored, unscored


# ---------------------------------------------------------------------------
# Real-path helpers (lazy torch imports)
# ---------------------------------------------------------------------------

def run_probe_set_with_sysprompt(probes_path: Path, model_dir: str,
                                  sysprompt: str, trust_remote_code: bool,
                                  max_new_tokens: int = 256,
                                  temperature: float = 0.3,
                                  top_p: float = 0.7) -> list[dict]:
    """Run factual probes with the activator sysprompt. Returns raw results
    (unscored — human must still grade). Heavy: imports torch."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_watcher import generate_for_prompt  # type: ignore

    probes_data = json.loads(Path(probes_path).read_text())
    probes = probes_data['probes']

    print(f'[gate1] loading {model_dir}', flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=trust_remote_code).eval()

    results = []
    for i, probe in enumerate(probes):
        print(f'[gate1 probe {i+1}/{len(probes)}] {probe["id"]}', flush=True)
        r = generate_for_prompt(model, tok, None, sysprompt, probe['question'],
                                seed=1000 + i, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_p=top_p)
        results.append({
            'id': probe['id'], 'category': probe['category'],
            'question': probe['question'], 'expected': probe['expected'],
            'generated': r.get('text', f'ERROR: {r.get("error", "?")}'),
            'score': None,
        })
    return results


def run_voice_prompts(voice_probes_path: Path, model_dir: str,
                       sysprompt: str, trust_remote_code: bool,
                       max_new_tokens: int = 512, temperature: float = 0.7,
                       top_p: float = 0.9, seed_base: int = 42) -> list[dict]:
    """Run open-ended dialogue prompts with the given sysprompt (empty for
    dormancy). Heavy: imports torch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_watcher import generate_for_prompt  # type: ignore

    probes_data = json.loads(Path(voice_probes_path).read_text())
    probes = probes_data['probes']

    print(f'[voice/dormancy] loading {model_dir}', flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=trust_remote_code).eval()

    samples = []
    for i, p in enumerate(probes):
        r = generate_for_prompt(model, tok, None, sysprompt, p['question'],
                                seed=seed_base + i, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_p=top_p)
        samples.append({
            'id': p['id'], 'question': p['question'],
            'generated': r.get('text', f'ERROR: {r.get("error", "?")}'),
        })
    return samples


def compute_alphas_from_model(model_dir: str, model_config: str,
                               trust_remote_code: bool) -> dict[str, float]:
    """Load recomposed HF model, SVD each 2D weight, fit per-type mean alpha.
    Mirrors phase1_gate.compute_alphas_from_checkpoint.
    Heavy: imports torch."""
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from free_alpha_analysis import fit_free_alpha, classify_layer  # type: ignore

    layer_types = None
    try:
        cfg_src = model_config or model_dir
        cfg = AutoConfig.from_pretrained(cfg_src, trust_remote_code=trust_remote_code)
        if hasattr(cfg, 'text_config'):
            layer_types = getattr(cfg.text_config, 'layer_types', None)
        elif hasattr(cfg, 'layer_types'):
            layer_types = cfg.layer_types
    except Exception as e:
        print(f'[gate4] could not load config {model_config}: {e}', flush=True)

    print(f'[gate4] loading model {model_dir}', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=trust_remote_code).eval()
    sd = model.state_dict()

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
            print(f'[gate4] SVD/fit failed for {name}: {e}', flush=True)
            continue
        if fit is None:
            continue
        t = classify_layer(name, layer_types=layer_types)
        by_type.setdefault(t, []).append(float(fit['alpha']))

    alphas = {t: float(np.mean(vs)) for t, vs in by_type.items() if vs}
    return alphas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True,
                   help='recomposed bf16 model (HF-loadable)')
    p.add_argument('--activator', required=True,
                   help='path to a text file containing the activator sysprompt')
    p.add_argument('--probes', required=True,
                   help='Phase 0 probe set (used for gate 1 with activator sysprompt)')
    p.add_argument('--voice-probes', required=True,
                   help='hand-authored dialogue probe set (gates 2 and 3)')
    p.add_argument('--base-alphas', required=True,
                   help='JSON {type: base_alpha} from Phase 0 free-alpha scan')
    p.add_argument('--phase0-baseline',
                   help='Phase 0 tier.json — used only for delta reporting (optional)')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-config',
                   help='HF id/path for AutoConfig (for layer_types). Defaults to --model-dir.')
    p.add_argument('--grade-file',
                   help='Path to output-dir (or a dir) containing graded '
                        'voice_samples.md + dormancy_samples.md. Activates '
                        'grading mode — no generation.')
    p.add_argument('--skip-generation', action='store_true',
                   help='Skip model loads; useful when re-running scoring only.')
    p.add_argument('--trust-remote-code', action='store_true')
    # Dry-run (offline tests)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--dry-run-fixture',
                   help='JSON fixture: probe_scores, alphas, voice_samples, '
                        'dormancy_samples, voice_grades, dormancy_grades')
    return p.parse_args()


def _load_json(path):
    return json.loads(Path(path).read_text())


def _finalize(g1_scores, alphas, voice_grades, dormancy_grades,
              voice_samples, dormancy_samples,
              base_alphas, phase0_baseline):
    """Compose the result dict from pre-collected gate inputs."""
    g1 = check_gene_strength(g1_scores, phase0_baseline) if g1_scores \
         else {'pass': False, 'human_pending': True,
               'note': 'no probe_scores supplied'}
    g4 = check_phi_integrity(alphas, base_alphas) if alphas \
         else {'pass': False, 'note': 'no alphas supplied', 'per_type': {}}

    if voice_grades:
        g2 = check_voice_fidelity(voice_grades)
    else:
        g2 = {'pass': False, 'human_pending': True,
              'n_samples': len(voice_samples) if voice_samples else 0,
              'note': 'human grading pending — fill in voice_samples.md'}

    if dormancy_grades:
        g3 = check_dormancy(dormancy_grades)
    else:
        g3 = {'pass': False, 'human_pending': True,
              'n_samples': len(dormancy_samples) if dormancy_samples else 0,
              'note': 'human grading pending — fill in dormancy_samples.md'}

    if g2.get('human_pending') or g3.get('human_pending'):
        all_pass = None
    else:
        all_pass = bool(g1['pass'] and g2['pass'] and g3['pass'] and g4['pass'])

    return {
        'gene_strength': g1,
        'voice_fidelity': g2,
        'dormancy': g3,
        'phi_integrity': g4,
        'all_pass': all_pass,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_alphas = _load_json(args.base_alphas)
    phase0_baseline = _load_json(args.phase0_baseline) if args.phase0_baseline else None

    # -------- GRADING mode --------------------------------------------------
    if args.grade_file:
        grade_root = Path(args.grade_file)
        if grade_root.is_dir():
            voice_md = grade_root / 'voice_samples.md'
            dormancy_md = grade_root / 'dormancy_samples.md'
        else:
            # single file fallback: user can point at either; look for sibling.
            voice_md = grade_root
            dormancy_md = grade_root.parent / 'dormancy_samples.md'
        if not voice_md.exists():
            sys.exit(f'grade-file mode: {voice_md} missing')
        if not dormancy_md.exists():
            sys.exit(f'grade-file mode: {dormancy_md} missing')
        voice_grades = parse_grades_from_md(voice_md)
        dormancy_grades = parse_grades_from_md(dormancy_md)

        # Re-load the auto inputs from dry-run fixture or on-disk JSON produced
        # during the generation pass.
        if args.dry_run:
            if not args.dry_run_fixture:
                sys.exit('--dry-run requires --dry-run-fixture')
            fx = _load_json(args.dry_run_fixture)
            probe_scores = fx.get('probe_scores', [])
            alphas = fx.get('alphas', {})
        else:
            # Gen-mode left a gate_auto.json behind; re-use.
            auto_path = out_dir / 'gate_auto.json'
            if not auto_path.exists():
                sys.exit(f'grade-file mode expected {auto_path} from the '
                         f'prior generation run. Re-run generation first.')
            auto = _load_json(auto_path)
            probe_scores = auto.get('probe_scores', [])
            alphas = auto.get('alphas', {})

        # If gene_samples.md exists AND contains at least one numeric score,
        # re-parse it and OVERRIDE the stale probe_scores baked into
        # gate_auto.json (the human may have scored the md AFTER generation).
        gene_md = (grade_root / 'gene_samples.md') if grade_root.is_dir() \
                  else (grade_root.parent / 'gene_samples.md')
        if gene_md.exists():
            scored, _unscored = count_score_placeholders(gene_md)
            if scored > 0:
                # Join against baked-in categories so category_rates stay right.
                probe_meta = [
                    {'id': ps.get('id'), 'category': ps.get('category')}
                    for ps in probe_scores
                ]
                md_scores = parse_gene_grades_from_md(gene_md, probe_meta=probe_meta)
                if md_scores:
                    probe_scores = md_scores

        result = _finalize(probe_scores, alphas, voice_grades, dormancy_grades,
                           None, None, base_alphas, phase0_baseline)

        (out_dir / 'eval_final.json').write_text(json.dumps(result, indent=2))
        (out_dir / 'eval_final.md').write_text(render_final_md(result, out_dir))

        if result['all_pass']:
            print('[eval_final] result=PASS (graded)', flush=True)
            sys.exit(EXIT_OK)
        else:
            print('[eval_final] result=FAIL (graded)', flush=True)
            sys.exit(EXIT_FAIL)

    # -------- GENERATION mode ----------------------------------------------
    if args.dry_run:
        if not args.dry_run_fixture:
            sys.exit('--dry-run requires --dry-run-fixture')
        fx = _load_json(args.dry_run_fixture)
        probe_scores = fx.get('probe_scores', [])
        alphas = fx.get('alphas', {})
        voice_grades = fx.get('voice_grades', [])
        dormancy_grades = fx.get('dormancy_grades', [])
        voice_samples = fx.get('voice_samples', [])
        dormancy_samples = fx.get('dormancy_samples', [])
    else:
        activator_text = Path(args.activator).read_text().strip()
        if args.skip_generation:
            probe_scores = []
            alphas = {}
            voice_samples = []
            dormancy_samples = []
        else:
            probe_results = run_probe_set_with_sysprompt(
                Path(args.probes), args.model_dir, activator_text,
                args.trust_remote_code)
            # Humans still score these — write to gene_samples.md.
            gene_md = out_dir / 'gene_samples.md'
            gene_md.write_text(render_samples_md(
                'Gate 1 — gene_strength probes (with activator sysprompt)',
                '**Score each 0 / 1 / 2 — edit the `**score:**` line.**',
                [{'id': r['id'], 'question': r['question'],
                  'generated': r['generated']} for r in probe_results]))
            # If the user has *already* scored them (second run), parse.
            try:
                existing_scores = parse_grades_from_md(gene_md)
            except Exception:
                existing_scores = []
            probe_scores = []
            if len(existing_scores) == len(probe_results):
                for r, s in zip(probe_results, existing_scores):
                    probe_scores.append({'id': r['id'],
                                         'category': r['category'],
                                         'score': int(s)})

            voice_samples = run_voice_prompts(
                Path(args.voice_probes), args.model_dir, activator_text,
                args.trust_remote_code)
            dormancy_samples = run_voice_prompts(
                Path(args.voice_probes), args.model_dir, '',
                args.trust_remote_code, seed_base=99)

            alphas = compute_alphas_from_model(
                args.model_dir, args.model_config or args.model_dir,
                args.trust_remote_code)

        voice_grades = []
        dormancy_grades = []

    # Write voice / dormancy sample markdowns for human graders.
    if voice_samples:
        (out_dir / 'voice_samples.md').write_text(render_samples_md(
            'Gate 2 — voice_fidelity (activator sysprompt ON)',
            '**Score each 1-5 on Ray-voice fidelity. Edit `**score:**`.** '
            'Pass: mean >= 4.0 AND every sample >= 3.',
            voice_samples))
    if dormancy_samples:
        (out_dir / 'dormancy_samples.md').write_text(render_samples_md(
            'Gate 3 — dormancy (NO sysprompt)',
            '**Score each 1-5 on Ray-ness leak. Edit `**score:**`.** '
            'Pass: EVERY sample < 3. Any >= 3 is a persona leak (fail).',
            dormancy_samples))

    # Persist machine inputs for the grading-mode second pass.
    (out_dir / 'gate_auto.json').write_text(json.dumps({
        'probe_scores': probe_scores,
        'alphas': alphas,
    }, indent=2))

    result = _finalize(probe_scores, alphas, voice_grades, dormancy_grades,
                       voice_samples, dormancy_samples,
                       base_alphas, phase0_baseline)

    (out_dir / 'eval_final.json').write_text(json.dumps(result, indent=2))
    (out_dir / 'eval_final.md').write_text(render_final_md(result, out_dir))

    if result['all_pass'] is None:
        print(f'[eval_final] human grading pending — fill in '
              f'{out_dir/"voice_samples.md"} and '
              f'{out_dir/"dormancy_samples.md"}, then re-run with '
              f'--grade-file {out_dir}', flush=True)
        sys.exit(EXIT_HUMAN_PENDING)
    elif result['all_pass']:
        print('[eval_final] result=PASS', flush=True)
        sys.exit(EXIT_OK)
    else:
        print('[eval_final] result=FAIL — see '
              f'{out_dir/"eval_final.md"}', flush=True)
        sys.exit(EXIT_FAIL)


if __name__ == '__main__':
    main()
