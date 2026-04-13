#!/usr/bin/env python3
"""
Evaluate a spectral fine-tuning checkpoint with proper chat template.

Loads the base model, applies spectral checkpoint, generates responses
to RAI test prompts using the correct IT chat format.

Usage:
    python3 -u scripts/eval_rai_checkpoint.py \
        --hf-model google/gemma-4-31b-it \
        --checkpoint runs/RAI-gemma4-fullrank/best_spectral.pt \
        --output runs/RAI-gemma4-fullrank/eval_samples.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavegpt.spectral_surgery import spectral_scaffold
from wavegpt.spectral_linear import SpectralLinear

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="google/gemma-4-31b-it")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_spectral.pt or final_spectral.pt")
    parser.add_argument("--output", default="eval_samples.md")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Load tokenizer
    print(f"Loading tokenizer: {args.hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model, trust_remote_code=True)

    # Build model from config (no weight download)
    print(f"Building model skeleton from config...")
    config = AutoConfig.from_pretrained(
        args.hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # Load spectral checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    print(f"  {len(ckpt)} spectral parameters")

    # Scaffold SpectralLinear layers from checkpoint shapes
    skip = ['embed_tokens', 'lm_head', 'visual', 'vision']
    spectral_scaffold(model, rank=256, mode='per_mode',
                      skip_patterns=skip, state_dict=ckpt)
    model.load_state_dict(ckpt, strict=False)
    print(f"  Loaded spectral weights")

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving to {device}...")
    model.to(device)
    model.eval()
    print(f"  Ready")

    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt[:60]}...")

        messages = [
            {"role": "user", "content": prompt},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            kwargs = {}
            if 'gemma' in getattr(config, 'model_type', ''):
                kwargs['mm_token_type_ids'] = torch.zeros_like(inputs["input_ids"])
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )
        gen_time = time.time() - t0
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_tok = len(new_tokens)
        tps = n_tok / gen_time if gen_time > 0 else 0

        print(f"  {n_tok} tokens, {tps:.1f} tok/s")
        print(f"  {response[:200]}...")

        results.append({
            "prompt": prompt,
            "response": response,
            "tokens": n_tok,
            "time_s": round(gen_time, 2),
            "tok_per_s": round(tps, 1),
        })

    # Write markdown
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# RAI Spectral Fine-Tune Results\n\n")
        f.write(f"**Checkpoint**: {args.checkpoint}\n")
        f.write(f"**Model**: {args.hf_model}\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")

        for r in results:
            f.write(f"## {r['prompt']}\n\n")
            f.write(f"**({r['tokens']} tokens, {r['tok_per_s']} tok/s)**\n\n")
            f.write(f"{r['response']}\n\n---\n\n")

    # Also JSON
    json_path = out_path.with_suffix('.json')
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {len(results)} responses saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
