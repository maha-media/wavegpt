#!/usr/bin/env python3
"""Phase 0 tier classifier: parse scored probe markdown, emit tier.json.

Reads `probe_baseline.md` (scored by hand — `**score:** N` lines flipped from
`?` to 0/1/2) plus `probe_baseline.json` (for category metadata), computes
correct_rate and gap_categories, writes `tier.json`.

Tier thresholds (from persona-pipeline-design):
    A: ≥80% probes scored ≥1
    B: 40-80%
    C: 10-40%
    D: <10%

Usage:
    python3 scripts/phase0_classify.py \\
        --probe-dir runs/probes/ray_baseline
"""
import argparse
import json
import re
import sys
from pathlib import Path


SCORE_RE = re.compile(r'^\*\*score:\*\*\s*(\d+)\s*$', re.MULTILINE)
SECTION_RE = re.compile(
    r'^##\s+(?P<id>\S+)\s+—\s+(?P<category>[^\n]+?)\s*$(?P<body>.*?)(?=^##\s|\Z)',
    re.MULTILINE | re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--probe-dir', required=True,
                   help='directory containing probe_baseline.md + probe_baseline.json')
    p.add_argument('--gap-threshold', type=float, default=0.5,
                   help='category correct_rate below this → gap category')
    return p.parse_args()


def tier_for(rate: float) -> str:
    if rate >= 0.80:
        return 'A'
    if rate >= 0.40:
        return 'B'
    if rate >= 0.10:
        return 'C'
    return 'D'


def main():
    args = parse_args()
    probe_dir = Path(args.probe_dir)
    md_path = probe_dir / 'probe_baseline.md'
    json_path = probe_dir / 'probe_baseline.json'

    if not md_path.exists():
        sys.exit(f'missing {md_path}')
    if not json_path.exists():
        sys.exit(f'missing {json_path}')

    snapshot = json.loads(json_path.read_text())
    md = md_path.read_text()

    scores_by_id = {}
    for m in SECTION_RE.finditer(md):
        pid = m.group('id').strip()
        body = m.group('body')
        score_m = SCORE_RE.search(body)
        if not score_m:
            continue
        raw = score_m.group(1).strip()
        try:
            scores_by_id[pid] = int(raw)
        except ValueError:
            pass

    unscored = []
    per_probe = []
    per_category = {}
    for r in snapshot['results']:
        pid = r['id']
        cat = r['category']
        score = scores_by_id.get(pid)
        if score is None:
            unscored.append(pid)
            continue
        per_probe.append({'id': pid, 'category': cat, 'score': score})
        bucket = per_category.setdefault(cat, {'total': 0, 'at_least_1': 0, 'score_2': 0})
        bucket['total'] += 1
        if score >= 1:
            bucket['at_least_1'] += 1
        if score >= 2:
            bucket['score_2'] += 1

    if unscored:
        print(f'[warn] {len(unscored)} probes unscored: {unscored}', flush=True)

    if not per_probe:
        sys.exit('no scored probes found — did you edit probe_baseline.md?')

    total = len(per_probe)
    at_least_1 = sum(1 for e in per_probe if e['score'] >= 1)
    score_2 = sum(1 for e in per_probe if e['score'] >= 2)
    correct_rate = at_least_1 / total
    strong_rate = score_2 / total
    tier = tier_for(correct_rate)

    category_rates = {}
    gap_categories = []
    for cat, b in per_category.items():
        rate = b['at_least_1'] / b['total'] if b['total'] else 0.0
        category_rates[cat] = {
            'total': b['total'],
            'correct_rate': round(rate, 3),
            'strong_rate': round(b['score_2'] / b['total'], 3) if b['total'] else 0.0,
        }
        if rate < args.gap_threshold:
            gap_categories.append(cat)

    tier_data = {
        'subject': snapshot['subject'],
        'model_dir': snapshot['model_dir'],
        'tier': tier,
        'correct_rate': round(correct_rate, 3),
        'strong_rate': round(strong_rate, 3),
        'n_probes': total,
        'n_unscored': len(unscored),
        'category_rates': category_rates,
        'gap_categories': sorted(gap_categories),
        'config': snapshot.get('config', {}),
    }

    out = probe_dir / 'tier.json'
    out.write_text(json.dumps(tier_data, indent=2))
    print(f'[done] subject={tier_data["subject"]!r}', flush=True)
    print(f'       tier={tier}  correct_rate={correct_rate:.2%}  '
          f'strong_rate={strong_rate:.2%}  (n={total})', flush=True)
    if gap_categories:
        print(f'       gap_categories: {gap_categories}', flush=True)
    print(f'       wrote {out}', flush=True)


if __name__ == '__main__':
    main()
