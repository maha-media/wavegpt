"""Deep dive: attn_v spectral shift in Gemma 4-31B-IT."""
import torch
import numpy as np
from math import sqrt
from scipy.optimize import curve_fit

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI


def fit(S):
    s = S.numpy().astype(np.float64)
    n = len(s)
    nf = max(int(n * 0.9), 4)
    k = np.arange(1, nf + 1, dtype=np.float64)
    y = s[:nf]
    def b(k, A, k0, a):
        return A * (k + k0) ** (-a)
    try:
        p, _ = curve_fit(b, k, y, p0=[y[0] * 50, max(n * 0.1, 10), INV_PHI],
                         bounds=([0, 0, 0.01], [y[0] * 1000, n * 2, 2.0]), maxfev=20000)
        pred = b(k, *p)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        return p[0], p[1], p[2], r2
    except Exception:
        return float(s[0]), 0, INV_PHI, 0


print("Loading spectra...")
sd = torch.load("runs/gemma4-it-rai-phi/decomposed.pt", map_location="cpu", weights_only=True)
# Support both old (.spectrum) and new (.log_spectrum) formats
spectra = {}
for k, v in sd.items():
    if k.endswith(".spectrum"):
        spectra[k.rsplit(".spectrum", 1)[0]] = v
    elif k.endswith(".log_spectrum"):
        spectra[k.rsplit(".log_spectrum", 1)[0]] = torch.exp(v)
del sd

print(f"\nATTN_V DEEP DIVE — {sum(1 for n in spectra if 'v_proj' in n)} layers")
print("=" * 90)

v_data = []
for name, S in sorted(spectra.items()):
    if "v_proj" not in name.lower():
        continue
    A, k0, alpha, r2 = fit(S)
    parts = name.split(".")
    layer_num = None
    for p in parts:
        if p.isdigit():
            layer_num = int(p)
            break
    # Gemma 4: every 6th layer (5,11,17,...) is full attention
    attn_type = "full" if layer_num is not None and layer_num % 6 == 5 else "slide"
    print(f"  L{layer_num:02d}  alpha={alpha:.4f}  k0={k0:7.1f}  R2={r2:.4f}  "
          f"rank={len(S):4d}  s1={float(S[0]):7.2f}  [{attn_type}]")
    v_data.append(dict(layer=layer_num, alpha=alpha, k0=k0, r2=r2,
                       rank=len(S), sigma1=float(S[0]), atype=attn_type))

slide = [d for d in v_data if d["atype"] == "slide"]
full = [d for d in v_data if d["atype"] == "full"]
sa = [d["alpha"] for d in slide]
fa = [d["alpha"] for d in full]
all_a = [d["alpha"] for d in v_data]

print(f"\nSLIDING vs FULL ATTENTION:")
print(f"  Sliding ({len(sa):2d} layers): alpha = {np.mean(sa):.4f} +/- {np.std(sa):.4f}")
print(f"  Full    ({len(fa):2d} layers): alpha = {np.mean(fa):.4f} +/- {np.std(fa):.4f}")
print(f"  Base model prediction:         alpha = {INV_PHI**(3/7):.4f}  [(1/phi)^(3/7)]")

# Fraction search
print(f"\nPHI FRACTION SEARCH:")
fibs = [1, 1, 2, 3, 5, 8, 13]
luc = [2, 1, 3, 4, 7, 11, 18]

for target, label in [(np.mean(sa), "sliding"), (np.mean(fa), "full"), (np.mean(all_a), "overall")]:
    print(f"\n  {label} (mean={target:.4f}):")
    hits = []
    for i, f in enumerate(fibs):
        for j, l in enumerate(luc):
            if l == 0:
                continue
            frac = f / l
            for base, bname in [(PHI, "phi"), (INV_PHI, "1/phi")]:
                val = base ** frac
                err = abs(val - target) / target * 100
                if err < 8:
                    hits.append((err, f"    {bname}^({f}/{l}) = {val:.4f}  err={err:.2f}%"))
    hits.sort()
    for _, h in hits[:5]:
        print(h)

# Depth trend
print(f"\nLAYER DEPTH TREND:")
for d in v_data:
    bar = "#" * int(d["alpha"] * 15)
    print(f"  L{d['layer']:02d} {d['alpha']:.3f} {bar}  [{d['atype']}]")

# Energy concentration comparison
print(f"\nENERGY CONCENTRATION (first 1/phi of modes):")
for d in v_data[:10]:
    for name, S in spectra.items():
        if "v_proj" in name and f".{d['layer']}." in name:
            energy = (S ** 2).cumsum(0) / (S ** 2).sum()
            idx_phi = min(int(len(S) / PHI), len(S) - 1)
            idx_phi2 = min(int(len(S) / PHI ** 2), len(S) - 1)
            print(f"  L{d['layer']:02d}: E(1/phi)={energy[idx_phi].item():.4f}  "
                  f"E(1/phi2)={energy[idx_phi2].item():.4f}  "
                  f"alpha={d['alpha']:.4f}  [{d['atype']}]")
            break
