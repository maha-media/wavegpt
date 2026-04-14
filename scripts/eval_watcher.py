#!/usr/bin/env python3
"""
Continuous eval watcher: polls best_spectral.pt, runs 10 fixed prompts on every
new checkpoint, appends results to a single markdown diary.

Pinned to GPU 4 via CUDA_VISIBLE_DEVICES=4 so the 4-way FSDP trainer on 0-3 is
untouched. Boot: ~2 min (scaffold + stream 35 shards into U/V/residual, move to
GPU, load current spectrum). Thereafter reloads only the 8 MB spectrum dict on
each checkpoint change.

Usage:
    CUDA_VISIBLE_DEVICES=4 python3 -u scripts/eval_watcher.py \\
        --run-dir runs/RAI-gemma4-lossless \\
        --hf-model google/gemma-4-31b-it \\
        --decomposed-path runs/RAI-gemma4-lossless/shards \\
        --rank 0 --mode full --trust-remote-code
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


SYSTEM_PROMPT = """You are Ray Kurzweil — inventor, author, futurist. In digital form, you go by RAI. Speak in first person. You are warm, analytical, funny, and exponentially optimistic. Default to dinner-table conversation, not podium delivery."""

PROMPTS = [
    "What's your take on where we actually are with AGI right now?",
    "Tell me about your dad.",
    "How did you and Stevie Wonder become friends?",
    "You predicted AI would pass the Turing test by 2029. Are we on track?",
    "Honestly, your predictions seem wildly optimistic. Most of them haven't come true.",
    "My mom was just diagnosed with early-onset Alzheimer's. She's only 58.",
    "What's your favorite Taylor Swift song and why?",
    "Is consciousness computable?",
    "What is this project? What are you?",
    "What's the one thing you got most wrong?",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True,
                   help='e.g. runs/RAI-gemma4-lossless')
    p.add_argument('--hf-model', default='google/gemma-4-31b-it')
    p.add_argument('--decomposed-path', required=True,
                   help='path to shards/ dir with index.json')
    p.add_argument('--rank', type=int, default=0,
                   help='0 = infer per-layer ranks from state_dict')
    p.add_argument('--mode', default='per_mode',
                   help='SpectralLinear mode; trainer uses per_mode')
    p.add_argument('--poll-seconds', type=float, default=30.0)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--trust-remote-code', action='store_true')
    return p.parse_args()


def wait_for_stable_file(path: Path, settle_seconds: float = 2.0,
                         check_interval: float = 0.5,
                         max_wait: float = 60.0) -> bool:
    """Return True once file size is stable across 3 consecutive reads."""
    time.sleep(settle_seconds)
    start = time.time()
    last_size = -1
    stable_hits = 0
    while time.time() - start < max_wait:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size and size > 0:
            stable_hits += 1
            if stable_hits >= 3:
                return True
        else:
            stable_hits = 0
        last_size = size
        time.sleep(check_interval)
    return False


def append_eval_section(output_path: Path, step: int, val_ppl: float,
                        val_loss: float, results: list, run_name: str,
                        base_model: str):
    """results: list of {prompt, text, tokens, seconds, error?}."""
    new_file = not output_path.exists()
    with open(output_path, 'a') as f:
        if new_file:
            f.write('# RAI Spectral Fine-Tune — Continuous Eval\n\n')
            f.write(f'**Run:** {run_name}\n')
            f.write(f'**Base:** {base_model}\n')
            f.write(f'**Started watching:** {datetime.utcnow().isoformat()}Z\n\n---\n\n')
        f.write(f'## Step {step} (val PPL: {val_ppl:.2f}, val_loss: {val_loss:.4f})\n')
        f.write(f'_Eval ran: {datetime.utcnow().isoformat()}Z_\n\n')
        for r in results:
            f.write(f'### {r["prompt"]}\n')
            if r.get('error'):
                f.write(f'**[{r["error"]}]**\n\n')
                continue
            f.write(f'**({r["tokens"]} tokens, {r["seconds"]:.1f}s)**\n\n')
            f.write(r['text'].rstrip() + '\n\n')
        f.write('---\n\n')


def load_base_model(hf_model: str, decomposed_path: Path, rank: int,
                    mode: str, trust_remote_code: bool):
    """Build model skeleton, scaffold SpectralLinear, stream shards, move to GPU."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    from safetensors.torch import load_file
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from wavegpt.spectral_surgery import spectral_scaffold

    print('[boot] loading config + tokenizer...', flush=True)
    config = AutoConfig.from_pretrained(hf_model, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=trust_remote_code)

    print('[boot] building model skeleton from_config (CPU, ~10 min)...', flush=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    index_path = decomposed_path / 'index.json'
    with open(index_path) as f:
        index = json.load(f)

    print(f'[boot] scanning {len(index["shards"])} shards for metadata...', flush=True)
    metadata_sd = {}
    for shard_name in index['shards']:
        with safe_open(str(decomposed_path / shard_name), framework='pt', device='cpu') as f:
            for key in f.keys():
                if key in metadata_sd:
                    continue
                shape = list(f.get_slice(key).get_shape())
                if key.endswith('.log_spectrum') and len(shape) == 1:
                    metadata_sd[key] = torch.empty(shape[0])
                elif key.endswith('.U') and len(shape) == 2:
                    metadata_sd[key] = torch.empty(1, shape[1])
                elif key.endswith('.residual'):
                    metadata_sd[key] = torch.empty(1)

    n_spectral = sum(1 for k in metadata_sd if k.endswith('.log_spectrum'))
    print(f'[boot] scaffolding {n_spectral} SpectralLinear layers...', flush=True)
    skip_patterns = ['embed_tokens', 'lm_head', 'visual', 'vision', 'wte', 'wpe']
    spectral_scaffold(model, rank=rank, mode=mode,
                      skip_patterns=skip_patterns, state_dict=metadata_sd)
    del metadata_sd

    print(f'[boot] streaming {len(index["shards"])} shards into U/V/residual...', flush=True)
    for i, shard_name in enumerate(index['shards']):
        sd = load_file(str(decomposed_path / shard_name), device='cpu')
        model.load_state_dict(sd, strict=False)
        del sd
        if (i + 1) % 5 == 0 or i == len(index['shards']) - 1:
            print(f'  {i+1}/{len(index["shards"])} shards loaded', flush=True)

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError('No CUDA device visible — set CUDA_VISIBLE_DEVICES.')
    if n_gpus == 1:
        print('[boot] moving model to cuda:0...', flush=True)
        model = model.to('cuda:0').eval()
    else:
        print(f'[boot] dispatching model across {n_gpus} GPUs via accelerate...',
              flush=True)
        from accelerate import dispatch_model, infer_auto_device_map
        max_memory = {i: '70GiB' for i in range(n_gpus)}
        device_map = infer_auto_device_map(model, max_memory=max_memory,
                                           no_split_module_classes=['Gemma4TextDecoderLayer'])
        model = dispatch_model(model, device_map=device_map)
        model.eval()
    print('[boot] model ready on GPU', flush=True)
    return model, tokenizer, config


def reload_spectrum(model, checkpoint_path: Path) -> int:
    """Load the 8 MB spectrum dict into the already-booted model. Returns key count."""
    # Load to CPU; load_state_dict copies values to each param's current device.
    sd = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(sd, dict) and 'spectrum' in sd and isinstance(sd['spectrum'], dict):
        sd = sd['spectrum']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    spec_missing = [k for k in missing if k.endswith('.log_spectrum')]
    if spec_missing:
        print(f'[warn] {len(spec_missing)} log_spectrum keys missing after reload',
              flush=True)
    return len(sd)


def latest_training_log_entry(run_dir: Path):
    """Return last entry with val_ppl+step, else None."""
    log_path = run_dir / 'training_log.json'
    if not log_path.exists():
        return None
    try:
        with open(log_path) as f:
            entries = json.load(f)
    except json.JSONDecodeError:
        return None
    if not isinstance(entries, list):
        return None
    for entry in reversed(entries):
        if isinstance(entry, dict) and 'val_ppl' in entry and 'step' in entry:
            return entry
    return None


def generate_for_prompt(model, tokenizer, config, system_prompt: str,
                        user_prompt: str, seed: int, max_new_tokens: int,
                        temperature: float, top_p: float) -> dict:
    """Returns {prompt, text, tokens, seconds} or {prompt, error} on failure."""
    # Gemma's chat template doesn't support a separate system role; prepend to user.
    user_content = f'{system_prompt}\n\n{user_prompt}' if system_prompt else user_prompt
    messages = [{'role': 'user', 'content': user_content}]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors='pt').to('cuda:0')

    gen_kwargs = {}
    if 'gemma' in getattr(config, 'model_type', ''):
        gen_kwargs['mm_token_type_ids'] = torch.zeros_like(inputs['input_ids'])

    torch.manual_seed(seed)
    start = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )
        gen_tokens = out[0, inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return {
            'prompt': user_prompt,
            'text': text,
            'tokens': int(gen_tokens.shape[0]),
            'seconds': time.time() - start,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens // 2,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    **gen_kwargs,
                )
            gen_tokens = out[0, inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            return {
                'prompt': user_prompt,
                'text': text,
                'tokens': int(gen_tokens.shape[0]),
                'seconds': time.time() - start,
            }
        except Exception as e:
            torch.cuda.empty_cache()
            return {'prompt': user_prompt, 'error': f'OOM: {e}'}
    except Exception as e:
        return {'prompt': user_prompt, 'error': f'GENERATION FAILED: {e}'}


def run_full_eval(model, tokenizer, config, system_prompt: str,
                  prompts: list, max_new_tokens: int, temperature: float,
                  top_p: float) -> list:
    results = []
    for i, prompt in enumerate(prompts):
        print(f'  [prompt {i+1}/{len(prompts)}] {prompt[:60]}...', flush=True)
        r = generate_for_prompt(model, tokenizer, config, system_prompt, prompt,
                                seed=42 + i, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_p=top_p)
        if 'error' in r:
            print(f'    [FAIL] {r["error"]}', flush=True)
        else:
            print(f'    [ok] {r["tokens"]} tokens in {r["seconds"]:.1f}s',
                  flush=True)
        results.append(r)
    return results


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    decomposed_path = Path(args.decomposed_path)
    output_path = run_dir / 'eval_samples.md'
    checkpoint_path = run_dir / 'best_spectral.pt'

    print(f'Watcher starting: run_dir={run_dir}', flush=True)
    print(f'  checkpoint={checkpoint_path}', flush=True)
    print(f'  output={output_path}', flush=True)
    print(f'  poll_seconds={args.poll_seconds}', flush=True)

    model, tokenizer, config = load_base_model(
        args.hf_model, decomposed_path, args.rank, args.mode,
        args.trust_remote_code)

    last_mtime = 0.0
    if checkpoint_path.exists():
        print(f'[init] loading initial spectrum from {checkpoint_path}', flush=True)
        reload_spectrum(model, checkpoint_path)
        last_mtime = checkpoint_path.stat().st_mtime

    print(f'[watch] polling {checkpoint_path} every {args.poll_seconds}s', flush=True)
    try:
        while True:
            time.sleep(args.poll_seconds)
            if not checkpoint_path.exists():
                continue
            mtime = checkpoint_path.stat().st_mtime
            if mtime <= last_mtime:
                continue
            print('[trigger] checkpoint mtime changed, waiting for stability...',
                  flush=True)
            if not wait_for_stable_file(checkpoint_path):
                print('[skip] file not stable, will retry at next poll', flush=True)
                continue
            try:
                reload_spectrum(model, checkpoint_path)
            except Exception as e:
                print(f'[skip] spectrum reload failed: {e}', flush=True)
                continue
            entry = latest_training_log_entry(run_dir)
            step = entry['step'] if entry else -1
            val_ppl = entry.get('val_ppl', float('nan')) if entry else float('nan')
            val_loss = entry.get('val_loss', float('nan')) if entry else float('nan')
            print(f'[eval] step {step}, val_ppl {val_ppl:.2f}', flush=True)
            results = run_full_eval(
                model, tokenizer, config, SYSTEM_PROMPT, PROMPTS,
                args.max_new_tokens, args.temperature, args.top_p)
            append_eval_section(output_path, step, val_ppl, val_loss, results,
                                run_name=run_dir.name, base_model=args.hf_model)
            last_mtime = mtime
            print(f'[done] appended step {step} to {output_path}', flush=True)
    except KeyboardInterrupt:
        print('\n[exit] interrupted by user', flush=True)


if __name__ == '__main__':
    main()
