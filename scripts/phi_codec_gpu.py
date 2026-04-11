"""
φ-Codec GPU pipeline: encode all layers on GPU, recompose, generate.

Uses torch.linalg.svd on GPU for 100x faster SVD than numpy CPU.
Processes layer-by-layer to stay within GPU memory.

Usage:
    python3 -u scripts/phi_codec_gpu.py \
        --hf-model /path/to/gemma-4-31B \
        --trust-remote-code
"""
import argparse
import gc
import sys
import time
from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

FL_EXPONENTS = {
    'attn_o': INV_PHI ** (1/3),
    'attn_q': INV_PHI ** (5/4),
    'attn_k': INV_PHI ** (2/11),
    'attn_v': INV_PHI ** (3/7),
    'mlp_gate': INV_PHI ** (4/7),
    'mlp_up': INV_PHI ** (8/11),
    'mlp_down': INV_PHI ** (5/7),
}


def classify_layer(name):
    name = name.lower()
    if 'o_proj' in name or 'out_proj' in name or 'c_proj' in name:
        return 'attn_o'
    if 'q_proj' in name: return 'attn_q'
    if 'k_proj' in name: return 'attn_k'
    if 'v_proj' in name: return 'attn_v'
    if 'gate' in name: return 'mlp_gate'
    if 'up_proj' in name: return 'mlp_up'
    if 'down_proj' in name: return 'mlp_down'
    return None


def phi_encode_decode_gpu(W: torch.Tensor, alpha: float, device: torch.device) -> torch.Tensor:
    """
    φ-codec encode→decode on GPU. Returns recomposed weight.

    1. Full SVD (no truncation)
    2. Fit φ-curve → compute residuals
    3. Quantize at tiered precision (32/16/8 bit)
    4. Dequantize → recompose W = U·diag(S)·V^T
    """
    m, n = W.shape
    W_gpu = W.to(device).float()

    # Full SVD on GPU
    U, S, Vh = torch.linalg.svd(W_gpu, full_matrices=False)
    n_sv = S.shape[0]

    # Fit φ-curve: σ_k = A·(k + k₀)^{-α}
    k = torch.arange(1, n_sv + 1, device=device, dtype=torch.float32)
    # Simple fit: A from σ₁, k₀ from least-squares on log scale
    try:
        from scipy.optimize import curve_fit
        import numpy as np
        s_np = S.cpu().numpy().astype(np.float64)
        k_np = np.arange(1, n_sv + 1, dtype=np.float64)
        def bent(k, A, k0):
            return A * (k + k0) ** (-alpha)
        popt, _ = curve_fit(bent, k_np, s_np,
                           p0=[s_np[0] * 50, max(n_sv * 0.1, 10)],
                           bounds=([0, 0], [s_np[0] * 1000, n_sv * 2]),
                           maxfev=10000)
        A_fit, k0_fit = float(popt[0]), float(popt[1])
    except Exception:
        A_fit, k0_fit = float(S[0].item()), 0.0

    predicted = A_fit * (k + k0_fit).pow(-alpha)
    residuals = S - predicted

    # Tier boundaries
    k0_int = max(1, int(k0_fit))
    k_phi = int(n_sv / PHI)
    t1 = min(k0_int, n_sv)
    t2 = min(k_phi, n_sv)

    # === TIERED QUANTIZATION (all on GPU) ===

    # Tier 1: float32 (keep exact)
    S_out = torch.zeros_like(S)
    S_out[:t1] = S[:t1]  # exact

    # Tier 2: float16 residuals → reconstruct
    if t2 > t1:
        res_t2 = residuals[t1:t2].half().float()  # round-trip through fp16
        S_out[t1:t2] = predicted[t1:t2] + res_t2

    # Tier 3: int8 residuals → reconstruct
    if n_sv > t2:
        res_t3 = residuals[t2:]
        if res_t3.numel() > 0:
            rmin = res_t3.min()
            rmax = res_t3.max()
            if rmax > rmin:
                scale = (rmax - rmin) / 255.0
                codes = ((res_t3 - rmin) / scale).round().clamp(0, 255).byte()
                res_t3_deq = codes.float() * scale + rmin
            else:
                res_t3_deq = torch.zeros_like(res_t3)
            S_out[t2:] = predicted[t2:] + res_t3_deq

    # Quantize U/V columns at tiered precision
    # Tier 1: float32 (exact)
    U_out = torch.zeros_like(U)
    Vh_out = torch.zeros_like(Vh)
    U_out[:, :t1] = U[:, :t1]
    Vh_out[:t1, :] = Vh[:t1, :]

    # Tier 2: float16 round-trip
    if t2 > t1:
        U_out[:, t1:t2] = U[:, t1:t2].half().float()
        Vh_out[t1:t2, :] = Vh[t1:t2, :].half().float()

    # Tier 3: int8 per-column quantization
    if n_sv > t2:
        # U columns
        u_tail = U[:, t2:]
        u_min = u_tail.min(dim=0, keepdim=True).values
        u_max = u_tail.max(dim=0, keepdim=True).values
        u_scale = (u_max - u_min) / 255.0
        u_scale = u_scale.clamp(min=1e-10)
        u_codes = ((u_tail - u_min) / u_scale).round().clamp(0, 255).byte()
        U_out[:, t2:] = u_codes.float() * u_scale + u_min

        # Vh rows
        vh_tail = Vh[t2:, :]
        vh_min = vh_tail.min(dim=1, keepdim=True).values
        vh_max = vh_tail.max(dim=1, keepdim=True).values
        vh_scale = (vh_max - vh_min) / 255.0
        vh_scale = vh_scale.clamp(min=1e-10)
        vh_codes = ((vh_tail - vh_min) / vh_scale).round().clamp(0, 255).byte()
        Vh_out[t2:, :] = vh_codes.float() * vh_scale + vh_min

    # Recompose: W = U · diag(S) · Vh
    W_hat = (U_out * S_out.unsqueeze(0)) @ Vh_out

    # Compute error
    err = torch.norm(W_gpu - W_hat) / torch.norm(W_gpu)

    # Return in original dtype, move back to CPU
    W_hat = W_hat.to(W.dtype).cpu()

    # Free GPU memory
    del U, S, Vh, U_out, Vh_out, S_out, W_gpu, predicted, residuals
    torch.cuda.empty_cache()

    return W_hat, float(err.item()), (t1, t2 - t1, n_sv - t2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompts", nargs="+", default=[
        "The most important thing about the Singularity is",
        "When I think about my father, I remember",
        "The golden ratio appears in nature because",
    ])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("φ-CODEC GPU: COMPRESS → RECOMPOSE → GENERATE")
    print("=" * 70)

    # Load model in bf16
    print(f"\n[1] Loading {args.hf_model}...")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
    if args.layers:
        text_cfg = getattr(config, 'text_config', config)
        orig = getattr(text_cfg, 'num_hidden_layers', 32)
        text_cfg.num_hidden_layers = min(args.layers, orig)
        print(f"  {text_cfg.num_hidden_layers}/{orig} layers")

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, config=config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded.")

    # Baseline generation
    print(f"\n[2] Baseline (original model)...")
    model.to(device)
    model.requires_grad_(False)
    for p in args.prompts[:1]:
        ids = tokenizer.encode(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                do_sample=True, temperature=0.7, top_p=0.9,
                                pad_token_id=tokenizer.pad_token_id)
        print(f"  Q: {p}")
        print(f"  A: {tokenizer.decode(out[0], skip_special_tokens=True)[len(p):][:200]}")

    # φ-encode/decode each linear layer (on GPU, layer by layer)
    print(f"\n[3] φ-codec encode→decode (GPU)...")
    errors = []
    total_orig = 0
    count = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        ltype = classify_layer(name)
        if ltype is None or min(module.weight.shape) < 64:
            continue

        alpha = FL_EXPONENTS.get(ltype, INV_PHI)
        m, n = module.weight.shape

        t0 = time.time()
        W_hat, err, tiers = phi_encode_decode_gpu(module.weight.data, alpha, device)
        elapsed = time.time() - t0

        # Replace weight in-place
        module.weight.data = W_hat.to(device)
        errors.append(err)
        total_orig += m * n * 2  # bf16 bytes
        count += 1

        if count <= 14 or count % 50 == 0:
            print(f"  [{count}] {name} ({ltype}) {m}x{n} "
                  f"err={err*100:.4f}% tiers={tiers} [{elapsed:.1f}s]")

    print(f"\n  Layers: {count}")
    print(f"  Mean error: {sum(errors)/len(errors)*100:.4f}%")
    print(f"  Max error:  {max(errors)*100:.4f}%")

    # Generate with recomposed model
    print(f"\n[4] Generation (φ-recomposed model)...")
    for p in args.prompts:
        ids = tokenizer.encode(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                do_sample=True, temperature=0.7, top_p=0.9,
                                pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n  Q: {p}")
        print(f"  A: {text[len(p):][:200]}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
