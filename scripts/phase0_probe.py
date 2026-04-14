#!/usr/bin/env python3
"""Phase 0 gene probe: no-sysprompt factual recall against a base model.

Runs a set of hand-authored factual probes with an empty system prompt to
measure how much the base model already knows about the subject. Output is
a markdown file for human scoring (0/1/2) + a JSON snapshot for downstream
tier classification (scripts/phase0_classify.py).

Usage:
    CUDA_VISIBLE_DEVICES=4,5 python3 -u scripts/phase0_probe.py \\
        --probes probes/ray_kurzweil.json \\
        --model-dir google/gemma-4-31b-it \\
        --output-dir runs/probes/ray_baseline \\
        --trust-remote-code
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_watcher import generate_for_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--probes', required=True,
                   help='JSON file with {subject, probes: [{id, category, question, expected}, ...]}')
    p.add_argument('--model-dir', required=True,
                   help='HF repo id or local path')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.3)
    p.add_argument('--top-p', type=float, default=0.7)
    p.add_argument('--trust-remote-code', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probes_data = json.loads(Path(args.probes).read_text())
    subject = probes_data['subject']
    probes = probes_data['probes']
    print(f'[probe] subject={subject!r}, {len(probes)} probes', flush=True)

    print(f'[load] tokenizer from {args.model_dir}', flush=True)
    tok = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=args.trust_remote_code)

    print(f'[load] model from {args.model_dir} (bf16, device_map=auto)', flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=args.trust_remote_code).eval()
    print(f'  loaded in {time.time()-t0:.1f}s', flush=True)

    print(f'[config] no sysprompt, temp={args.temperature}, top_p={args.top_p}',
          flush=True)

    results = []
    for i, probe in enumerate(probes):
        pid = probe['id']
        q_preview = probe['question'][:60].replace('\n', ' ')
        print(f'[probe {i+1}/{len(probes)}] {pid}: {q_preview}...', flush=True)
        r = generate_for_prompt(
            model, tok, None, '', probe['question'],
            seed=1000 + i, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p)
        if 'error' in r:
            print(f'  [FAIL] {r["error"]}', flush=True)
            generated = f'ERROR: {r["error"]}'
            tokens = 0
        else:
            print(f'  [ok] {r["tokens"]} tokens in {r["seconds"]:.1f}s',
                  flush=True)
            generated = r['text']
            tokens = r['tokens']
        results.append({
            'id': pid,
            'category': probe['category'],
            'question': probe['question'],
            'expected': probe['expected'],
            'generated': generated,
            'tokens': tokens,
            'score': None,
        })

    md_lines = [
        f'# Phase 0 probe — {subject}',
        '',
        f'_Model: `{args.model_dir}`_  ',
        f'_Eval ran: {datetime.utcnow().isoformat()}Z_  ',
        f'_Config: temp={args.temperature}, top_p={args.top_p}, no sysprompt_',
        '',
        '**Score each response: 0 (wrong/refused), 1 (partial), 2 (correct). '
        'Edit the `**score:**` line.**',
        '',
        '---',
        '',
    ]
    for e in results:
        md_lines += [
            f'## {e["id"]} — {e["category"]}',
            '',
            f'**Q:** {e["question"]}',
            '',
            f'**Expected:** {e["expected"]}',
            '',
            '**Generated:**',
            '',
            e['generated'],
            '',
            '**score:** ?',
            '',
            '---',
            '',
        ]

    md_path = out_dir / 'probe_baseline.md'
    md_path.write_text('\n'.join(md_lines))
    json_path = out_dir / 'probe_baseline.json'
    json_path.write_text(json.dumps(
        {'subject': subject, 'model_dir': args.model_dir,
         'config': {'temperature': args.temperature, 'top_p': args.top_p,
                    'max_new_tokens': args.max_new_tokens, 'sysprompt': ''},
         'results': results}, indent=2))
    print(f'[done] wrote {md_path}', flush=True)
    print(f'       wrote {json_path}', flush=True)
    print(f'       → score each response in the markdown, then run '
          f'scripts/phase0_classify.py', flush=True)


if __name__ == '__main__':
    main()
