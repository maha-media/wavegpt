"""
Test the multiplicative interaction hypothesis:
  p = (F/L)^n  where n = number of multiplicative gradient interactions

n=1: simple linear projection → p = F/L
n=2: element-wise multiply + activation derivative → p = (F/L)²
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI

# Fibonacci and Lucas sequences
fib = [0, 1, 1, 2, 3, 5, 8, 13, 21]
lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47]
fib_set = set(fib)
lucas_set = set(lucas)

print("=" * 70)
print("TEST: p = (F/L)^n — Multiplicative Gradient Interaction Hypothesis")
print("=" * 70)

# Observed data
layers = {
    'attn_q':    {'α': 0.550, 'p_obs': 5/4},
    'mlp_up':    {'α': 0.703, 'p_obs': 8/11},
    'mlp_down':  {'α': 0.714, 'p_obs': 5/7},
    'mlp_gate':  {'α': 0.763, 'p_obs': 4/7},
    'delta_qkv': {'α': 0.783, 'p_obs': 1/2},
    'attn_v':    {'α': 0.811, 'p_obs': 3/7},
    'delta_out': {'α': 0.843, 'p_obs': 1/3},
    'delta_z':   {'α': 0.848, 'p_obs': 1/3},
    'attn_o':    {'α': 0.853, 'p_obs': 1/3},
    'attn_k':    {'α': 0.910, 'p_obs': 2/11},
}

# Count multiplicative gradient interactions for each layer type
# A "multiplicative interaction" = a point in the backward pass where
# the gradient is multiplied by another activation (not a constant).
#
# SwiGLU: h = SwiLU(gate) ⊙ up
#   gate_grad = dL/dh ⊙ SwiLU'(gate) ⊙ up       ← 2 multiplications
#   up_grad   = dL/dh ⊙ SwiLU(gate)             ← 1 multiplication
#
# Attention: attn(Q,K,V) = softmax(QK^T/√d) · V
#   Q_grad = dL/dattn · (d_softmax/d_Q) · V     ← 1 multiplication (V)
#   K_grad = dL/dattn · (d_softmax/d_K) · Q     ← 1 multiplication (Q)
#   V_grad = dL/dattn · softmax                 ← 1 multiplication (softmax output)
#
# DeltaNet QKV: combined Q,K,V in one projection
#   Similar to attention but with internal gating
#
# Standard linear: y = Wx
#   W_grad = dL/dy · x^T                        ← 1 multiplication (input)
#   But this is the OUTER PRODUCT, not element-wise. Count as n=1.

multiplicative_count = {
    'attn_q':    1,   # gradient multiplied by V through attention
    'mlp_up':    1,   # gradient multiplied by SwiLU(gate)
    'mlp_down':  1,   # gradient multiplied by activation output
    'mlp_gate':  2,   # multiplied by up_output AND SwiLU'(gate)
    'delta_qkv': 2,   # combines Q,K,V with internal multiplicative gating
    'attn_v':    1,   # gradient multiplied by attention weights
    'delta_out': 1,   # standard output projection
    'delta_z':   1,   # gating projection
    'attn_o':    1,   # output projection
    'attn_k':    2,   # Q·K^T dot product (multiplied by Q in backward),
}

print("\n1. Testing p = (F/L)^1 for n=1 layers")
print(f"   {'Type':<12s} {'p_obs':>7s} {'(F/L)':>7s} {'Δ':>8s} {'Δ%':>7s}")
print(f"   {'-'*12} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")

# For n=1: find best F/L match
n1_layers = {k: v for k, v in layers.items() if multiplicative_count[k] == 1}
n2_layers = {k: v for k, v in layers.items() if multiplicative_count[k] == 2}

best_n1 = {}
for name in n1_layers:
    p = layers[name]['p_obs']
    best_err = float('inf')
    best_frac = None
    for f in fib[1:]:  # skip F(0)=0
        for l in lucas:
            if l == 0:
                continue
            candidate = f / l
            err = abs(candidate - p)
            if err < best_err:
                best_err = err
                best_frac = (f, l)
    best_n1[name] = {'f': best_frac[0], 'l': best_frac[1], 
                     'pred': best_frac[0]/best_frac[1], 'err': best_err}

for name in sorted(n1_layers, key=lambda x: -layers[x]['p_obs']):
    info = best_n1[name]
    p_obs = layers[name]['p_obs']
    err_pct = info['err'] / p_obs * 100
    print(f"   {name:<12s} {p_obs:>7.4f} {info['f']}/{info['l']}={info['pred']:.4f} "
          f"{info['err']:>+8.4f} {err_pct:>6.1f}%")

mean_n1_err = np.mean([best_n1[n]['err'] for n in n1_layers])
mean_n1_err_pct = np.mean([best_n1[n]['err'] / layers[n]['p_obs'] * 100 for n in n1_layers])
print(f"\n   Mean n=1 error: {mean_n1_err:.4f} ({mean_n1_err_pct:.1f}%)")

print(f"\n2. Testing p = (F/L)² for n=2 layers")
print(f"   {'Type':<12s} {'p_obs':>7s} {'(F/L)²':>8s} {'F/L':>7s} {'Δ':>8s} {'Δ%':>7s}")
print(f"   {'-'*12} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")

best_n2 = {}
for name in n2_layers:
    p = layers[name]['p_obs']
    best_err = float('inf')
    best_frac = None
    for f in fib[1:]:
        for l in lucas:
            if l == 0:
                continue
            candidate = (f / l) ** 2
            err = abs(candidate - p)
            if err < best_err:
                best_err = err
                best_frac = (f, l)
    best_n2[name] = {'f': best_frac[0], 'l': best_frac[1],
                     'base': best_frac[0]/best_frac[1],
                     'pred': (best_frac[0]/best_frac[1])**2, 'err': best_err}

for name in sorted(n2_layers, key=lambda x: -layers[x]['p_obs']):
    info = best_n2[name]
    p_obs = layers[name]['p_obs']
    err_pct = info['err'] / p_obs * 100
    print(f"   {name:<12s} {p_obs:>7.4f} ({info['f']}/{info['l']})²={info['pred']:.4f} "
          f"{info['base']:.4f} {info['err']:>+8.4f} {err_pct:>6.1f}%")

mean_n2_err = np.mean([best_n2[n]['err'] for n in n2_layers])
mean_n2_err_pct = np.mean([best_n2[n]['err'] / layers[n]['p_obs'] * 100 for n in n2_layers])
print(f"\n   Mean n=2 error: {mean_n2_err:.4f} ({mean_n2_err_pct:.1f}%)")

print(f"\n3. Comparison: n=1 vs n=2 fit quality")
print(f"   n=1: mean error = {mean_n1_err:.4f} ({mean_n1_err_pct:.1f}%)")
print(f"   n=2: mean error = {mean_n2_err:.4f} ({mean_n2_err_pct:.1f}%)")

# Check if n=2 layers are better fit than n=1 layers
print(f"   n=2 fit better: {mean_n2_err < mean_n1_err}")

print(f"\n4. Detailed check: the claimed (3/4)² for mlp_gate")
p_gate = layers['mlp_gate']['p_obs']
claimed = (3/4)**2
print(f"   mlp_gate p_obs = {p_gate:.4f}")
print(f"   (3/4)² = {claimed:.4f}")
print(f"   Δ = {abs(p_gate - claimed):.6f} ({abs(p_gate - claimed)/p_gate*100:.3f}%)")

# Also check (2/3)² = 4/9
alt = (2/3)**2
print(f"   (2/3)² = {alt:.6f}, Δ = {abs(p_gate - alt):.6f} ({abs(p_gate - alt)/p_gate*100:.3f}%)")

# What about 4/7 directly?
direct = 4/7
print(f"   4/7   = {direct:.6f}, Δ = {abs(p_gate - direct):.6f} ({abs(p_gate - direct)/p_gate*100:.3f}%)")

print(f"\n   Note: (3/4)² = 9/16 = 0.5625")
print(f"         4/7    = 0.5714")
print(f"         These are DIFFERENT!")
print(f"         (3/4)² ≠ 4/7")
print(f"         (3/4)² = 9/16 vs 4/7: difference = {abs(9/16 - 4/7):.6f}")

# So the claim is p = (F/L)², not that the fraction itself is squared
# Let me check: is mlp_gate's p closer to (3/4)² or to 4/7?
print(f"\n   Is mlp_gate's p closer to (3/4)² or 4/7?")
print(f"   p_obs = {p_gate:.6f}")
print(f"   (3/4)² = {(3/4)**2:.6f} → Δ = {abs(p_gate - (3/4)**2):.6f}")
print(f"   4/7    = {4/7:.6f} → Δ = {abs(p_gate - 4/7):.6f}")
print(f"   Winner: {'(3/4)²' if abs(p_gate - (3/4)**2) < abs(p_gate - 4/7) else '4/7'}")

print(f"\n5. Testing ALL possible (F/L)^n for n=1,2,3")
print(f"   {'Type':<12s} {'p_obs':>7s} {'n=1 best':>10s} {'n=2 best':>10s} {'n=3 best':>10s} {'winner'}")
print(f"   {'-'*12} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for name in sorted(layers, key=lambda x: -layers[x]['p_obs']):
    p = layers[name]['p_obs']
    
    best = {}
    for n in [1, 2, 3]:
        best_err = float('inf')
        best_frac = None
        for f in fib[1:]:
            for l in lucas:
                if l == 0:
                    continue
                candidate = (f / l) ** n
                err = abs(candidate - p)
                if err < best_err:
                    best_err = err
                    best_frac = (f, l)
        best[n] = {'f': best_frac[0], 'l': best_frac[1],
                   'pred': (best_frac[0]/best_frac[1])**n, 'err': best_err}
    
    winner = min(best, key=lambda n: best[n]['err'])
    w = best[winner]
    
    def fmt(n, b):
        if n == 1:
            return f"{b['f']}/{b['l']}={b['pred']:.4f}"
        else:
            return f"({b['f']}/{b['l']})^{n}={b['pred']:.4f}"
    
    n1_str = fmt(1, best[1])
    n2_str = fmt(2, best[2])
    n3_str = fmt(3, best[3])
    
    print(f"   {name:<12s} {p:>7.4f} {n1_str:>10s} {n2_str:>10s} {n3_str:>10s} "
          f"n={winner} (Δ={w['err']:.4f})")

print(f"\n6. Critical question: Is (F/L)^2 really different from F'/L'?")
print(f"   (F/L)² = F²/L². Is F²/L² ever equal to F'/L' for different F',L'?")
print(f"   F²: {sorted(set(f**2 for f in fib[1:7]))}")
print(f"   L²: {sorted(set(l**2 for l in lucas[:6]))}")
print(f"   F² values: {[f**2 for f in [1,2,3,5,8]]}")
print(f"   L² values: {[l**2 for l in [2,1,3,4,7,11]]}")
print(f"   → (3/4)² = 9/16. Is 9 a Fibonacci number? No. Is 16 a Lucas number? No.")
print(f"   → So (F/L)² produces NEW fractions that are NOT in F/L form.")
print(f"   → This is a genuinely different prediction, not a reparameterization.")

print(f"\n7. Does the squared hypothesis change any fraction assignments?")
print(f"   Current table uses direct F/L fractions:")
for name in sorted(layers, key=lambda x: -layers[x]['p_obs']):
    info = layers[name]
    n = multiplicative_count[name]
    if n == 2:
        # Find the base F/L that when squared gives p_obs
        p = info['p_obs']
        base = p ** 0.5
        # Find best F/L close to base
        best_err = float('inf')
        best_frac = None
        for f in fib[1:]:
            for l in lucas:
                if l == 0:
                    continue
                err = abs(f/l - base)
                if err < best_err:
                    best_err = err
                    best_frac = (f, l)
        base_pred = (best_frac[0]/best_frac[1])**2
        print(f"   {name:<12s} p_obs={p:.4f} → base≈{base:.4f} → "
              f"({best_frac[0]}/{best_frac[1]})² = {base_pred:.4f} (Δ={abs(p-base_pred):.4f})")
    else:
        # Find best F/L
        p = info['p_obs']
        best_err = float('inf')
        best_frac = None
        for f in fib[1:]:
            for l in lucas:
                if l == 0:
                    continue
                err = abs(f/l - p)
                if err < best_err:
                    best_err = err
                    best_frac = (f, l)
        print(f"   {name:<12s} p_obs={p:.4f} → {best_frac[0]}/{best_frac[1]} = {best_frac[0]/best_frac[1]:.4f} (Δ={abs(p-best_frac[0]/best_frac[1]):.4f})")

print(f"\n{'='*70}")
print(f"VERDICT ON THE SQUARED HYPOTHESIS")
print(f"{'='*70}")

# Check mlp_gate specifically
gate_p = layers['mlp_gate']['p_obs']
squared_pred = (3/4)**2
direct_pred = 4/7
print(f"\nmlp_gate:")
print(f"  p_obs = {gate_p:.6f}")
print(f"  (3/4)² = {squared_pred:.6f} → Δ = {abs(gate_p - squared_pred):.6f} ({abs(gate_p - squared_pred)/gate_p*100:.3f}%)")
print(f"  4/7    = {direct_pred:.6f} → Δ = {abs(gate_p - direct_pred):.6f} ({abs(gate_p - direct_pred)/gate_p*100:.3f}%)")
print(f"  → The 4/7 direct fit ({abs(gate_p - direct_pred)/gate_p*100:.3f}%) is BETTER than (3/4)² ({abs(gate_p - squared_pred)/gate_p*100:.3f}%)")

# But wait — maybe the user is saying the observed p = 4/7 IS (3/4)² rounded?
# Let me check: what if the TRUE value is (3/4)² = 9/16 = 0.5625
# and (1/φ)^(9/16) = ?
pred_alpha_sq = INV_PHI ** (3/4)**2
pred_alpha_direct = INV_PHI ** (4/7)
print(f"\n  (1/φ)^((3/4)²) = (1/φ)^0.5625 = {pred_alpha_sq:.6f}")
print(f"  (1/φ)^(4/7)    = (1/φ)^0.5714 = {pred_alpha_direct:.6f}")
print(f"  Observed α = 0.763")
print(f"  Squared prediction error: {abs(0.763 - pred_alpha_sq):.6f} ({abs(0.763 - pred_alpha_sq)/0.763*100:.3f}%)")
print(f"  Direct prediction error:  {abs(0.763 - pred_alpha_direct):.6f} ({abs(0.763 - pred_alpha_direct)/0.763*100:.3f}%)")

# Check delta_qkv and attn_k with squared hypothesis
print(f"\ndelta_qkv (n=2, p_obs=0.5):")
dqkv_p = layers['delta_qkv']['p_obs']
# Best squared: (1/2)² = 0.25, (1/1)² = 1, (2/2)² = 1, (2/3)² = 4/9 ≈ 0.444
# (1/√2)² = 0.5 — but √2 is not Lucas
# (3/4)² = 9/16 = 0.5625
# Hmm, what gives exactly 0.5?
# sqrt(0.5) = 0.7071 ≈ 1/√2 — not F/L
# closest F/L to sqrt(0.5): 
for f in fib[1:]:
    for l in lucas:
        if l == 0: continue
        val = (f/l)**2
        if abs(val - 0.5) < 0.02:
            print(f"  ({f}/{l})² = {val:.4f} (Δ={abs(val-0.5):.4f})")

# 1/2 is itself a valid F/L fraction (F(1)/L(0) = 1/2)
# So delta_qkv with p=1/2 could be n=1 with F/L=1/2, OR n=2 with (F/L)²≈1/2

print(f"\nattn_k (n=2, p_obs=2/11≈0.1818):")
ak_p = layers['attn_k']['p_obs']
# sqrt(2/11) = 0.4264
base = ak_p ** 0.5
print(f"  sqrt(p) = {base:.4f}")
# Best F/L to this:
for f in fib[1:]:
    for l in lucas:
        if l == 0: continue
        if abs(f/l - base) < 0.05:
            print(f"  ({f}/{l})² = {(f/l)**2:.4f} (Δ={abs((f/l)**2 - ak_p):.4f})")
