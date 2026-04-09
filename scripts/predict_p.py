"""
Attempt to derive a formula for p(type) from layer computational properties.

Given the 10 layer types and their p values, can we predict p from:
- Information role (query vs key vs value etc.)
- Matrix aspect ratio
- Position in residual stream
- Number of multiplicative interactions
- Gradient flow path length
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI

print("=" * 70)
print("PREDICTING p(type) FROM LAYER PROPERTIES")
print("=" * 70)

# The data
layers = {
    'attn_q':    {'p': 5/4,   'num': 5, 'den': 4},
    'mlp_up':    {'p': 8/11,  'num': 8, 'den': 11},
    'mlp_down':  {'p': 5/7,   'num': 5, 'den': 7},
    'mlp_gate':  {'p': 4/7,   'num': 4, 'den': 7},
    'delta_qkv': {'p': 1/2,   'num': 1, 'den': 2},
    'attn_v':    {'p': 3/7,   'num': 3, 'den': 7},
    'delta_out': {'p': 1/3,   'num': 1, 'den': 3},
    'delta_z':   {'p': 1/3,   'num': 1, 'den': 3},
    'attn_o':    {'p': 1/3,   'num': 1, 'den': 3},
    'attn_k':    {'p': 2/11,  'num': 2, 'den': 11},
}

print("\n1. Hypothesis: p encodes the 'information bandwidth' of the layer")
print()

# Information bandwidth = how many different things this layer needs to handle.
# In attention:
#   Q: needs to attend to ALL positions → bandwidth = num_positions → high p
#   K: needs to be specific/matchable → bandwidth = 1 (discriminative) → low p
#   V: carries the value for matched positions → bandwidth = matched_count → medium p
#   O: compresses attention output back to model dim → bandwidth = 1 → low p

# In MLP:
#   up: expands to intermediate → bandwidth = intermediate/d_model = 3.375 → medium-high
#   gate: controls which features activate → bandwidth = features_to_gate → medium
#   down: compresses back → bandwidth = 1 → medium

# In DeltaNet:
#   QKV: combined operation → bandwidth = balanced → medium
#   z: gating → bandwidth = controlled_features → medium-low
#   out: output projection → bandwidth = 1 → medium-low

print("   Layer       p        Information bandwidth interpretation")
print("   " + "-" * 65)
for name in sorted(layers, key=lambda x: -layers[x]['p']):
    info = layers[name]
    interpretations = {
        'attn_q':    'Query ALL positions → highest bandwidth',
        'mlp_up':    f'Expand {5120}→{17408} (3.375×) → high bandwidth',
        'mlp_down':  f'Compress {17408}→{5120} → medium bandwidth',
        'mlp_gate':  f'Control {17408} features → medium bandwidth',
        'delta_qkv': 'Combined QKV → balanced bandwidth',
        'attn_v':    'Value for matched positions → medium-low bandwidth',
        'delta_out': 'DeltaNet output → low bandwidth',
        'delta_z':   'DeltaNet gating → low bandwidth',
        'attn_o':    'Compress attention output → low bandwidth',
        'attn_k':    'Specific key matching → lowest bandwidth',
    }
    print(f"   {name:<12s} {info['p']:>7.4f}  {interpretations[name]}")

print("\n2. Testing: Is p proportional to log(expansion_ratio)?")
print()

# Expansion ratios (approximate, based on Qwen3.5-27B architecture)
expansion = {
    'attn_q':    1.0,    # 5120 → 5120 (same dim, but attends to all positions)
    'mlp_up':    3.375,  # 5120 → 17408
    'mlp_down':  0.296,  # 17408 → 5120
    'mlp_gate':  3.375,  # 5120 → 17408
    'delta_qkv': 1.0,    # balanced
    'attn_v':    1.0,    # 5120 → 5120
    'delta_out': 1.0,    # output
    'delta_z':   1.0,    # gating
    'attn_o':    1.0,    # 5120 → 5120
    'attn_k':    1.0,    # 5120 → 5120
}

print(f"   {'Type':<12s} {'p':>7s} {'exp_ratio':>10s} {'log(exp)':>10s} {'p/log(exp)':>12s}")
print(f"   {'-'*12} {'-'*7} {'-'*10} {'-'*10} {'-'*12}")
for name in sorted(layers, key=lambda x: -layers[x]['p']):
    p = layers[name]['p']
    e = expansion[name]
    if name == 'attn_q':
        # Q attends to seq_len positions, not just model_dim
        # Let's use seq_len as the expansion factor
        e = 2048  # typical context window
    log_e = np.log(e)
    ratio = p / log_e if log_e != 0 else float('inf')
    print(f"   {name:<12s} {p:>7.4f} {e:>10.1f} {log_e:>10.3f} {ratio:>12.3f}")

print(f"\n   Not proportional to log(expansion). attn_q dominates because")
print(f"   it operates over the sequence dimension, not model dimension.")

print("\n3. Hypothesis: p is proportional to gradient flow path complexity")
print()

# Gradient flow path: how many different gradient sources flow through this layer?
# More gradient paths → more complex gradient statistics → higher p needed

# Attention Q: receives gradient from ALL attention heads via the output
#   Path: loss → ... → attn_o → V → attn(Q,K) → Q_grad
#   Q gradient flows through softmax(QK^T/√d) → depends on ALL K and V
#   Complexity: O(num_heads × seq_len) → very high

# Attention K: similar but K is the "target" of similarity computation
#   K gradient flows through softmax(QK^T/√d) → depends on all Q
#   But K is static (keys are computed once) → lower complexity than Q
#   Complexity: O(num_heads) → low

# MLP gate: gradient flows through SwiLU × up_proj output
#   gate_grad = dL/dh ⊙ SwiLU'(gate) ⊙ h_up
#   Depends on both the loss gradient AND the up_proj activation
#   Complexity: O(intermediate_dim) → medium

# MLP up: gradient flows through SwiLU(gate) × up_grad
#   up_grad = dL/dh ⊙ SwiLU(gate)
#   Depends on the gate activation pattern
#   Complexity: O(intermediate_dim) → medium

# MLP down: gradient flows from loss through the compressed representation
#   down_grad = dL/dh ⊙ h_pre_down
#   Depends on the SwiLU(gate) ⊙ up output
#   Complexity: O(intermediate_dim) → medium

print("   This is getting too speculative without a concrete formula.")
print("   Let me try a different approach...")

print("\n4. Algebraic structure of the fractions")
print()

# The fractions p = num/den where num ∈ {1,2,3,4,5,8} and den ∈ {2,3,4,7,11}
# Let me check if there's an algebraic relationship between num and den
# for each layer type.

# Hypothesis: num and den are related to the Fibonacci/Lucas INDEX, not value.
# F indices: F(0)=0, F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, F(6)=8
# L indices: L(0)=2, L(1)=1, L(2)=3, L(3)=4, L(4)=7, L(5)=11

fib_vals = [0, 1, 1, 2, 3, 5, 8]  # F(0) to F(6)
lucas_vals = [2, 1, 3, 4, 7, 11]  # L(0) to L(5)

# For each fraction, find the Fibonacci/Lucas indices
print(f"   {'Type':<12s} {'p':>8s} {'num':>4s} {'F_idx':>5s} {'den':>4s} {'L_idx':>5s}")
print(f"   {'-'*12} {'-'*8} {'-'*4} {'-'*5} {'-'*4} {'-'*5}")

for name in sorted(layers, key=lambda x: -layers[x]['p']):
    info = layers[name]
    num = info['num']
    den = info['den']
    
    fib_idx = [i for i, f in enumerate(fib_vals) if f == num]
    lucas_idx = [i for i, l in enumerate(lucas_vals) if l == den]
    
    fib_str = ','.join(map(str, fib_idx)) if fib_idx else '?'
    lucas_str = ','.join(map(str, lucas_idx)) if lucas_idx else '?'
    
    # Also check if num is Lucas
    lucas_idx_as_num = [i for i, l in enumerate(lucas_vals) if l == num]
    fib_idx_as_den = [i for i, f in enumerate(fib_vals) if f == den]
    
    print(f"   {name:<12s} {info['p']:>8.4f} {num:>4d} F({fib_str:>4s}) {den:>4d} L({lucas_str:>4s})")

print(f"\n   Pattern check:")
print(f"   - Numerator 5 appears in attn_q(5/4) and mlp_down(5/7) → F(5)")
print(f"   - Numerator 8 appears in mlp_up(8/11) → F(6)")
print(f"   - Numerator 4 appears in mlp_gate(4/7) → L(3), NOT Fibonacci")
print(f"   - Denominator 11 appears in mlp_up(8/11) and attn_k(2/11) → L(5)")
print(f"   - Denominator 7 appears in mlp_down(5/7), mlp_gate(4/7), attn_v(3/7)")

print(f"\n5. The Q/K complementarity")
print(f"   attn_q: p = 5/4 = 1.250")
print(f"   attn_k: p = 2/11 = 0.182")
print(f"   Sum:    {5/4 + 2/11:.4f}")
print(f"   Product:{5/4 * 2/11:.4f}")
print(f"   Ratio:  {(5/4)/(2/11):.4f}")
print(f"   α_q/α_k = {0.550/0.910:.4f} ≈ {INV_PHI:.4f} (1/φ)? {'YES' if abs(0.550/0.910 - INV_PHI) < 0.05 else 'NO'}")

print(f"\n   Q and K are complementary: one gathers (broad), one discriminates (narrow).")
print(f"   The α ratio being ≈ 1/φ is suggestive but may be coincidental.")

print(f"\n6. The 'information compression' hypothesis")
print(f"   p might encode how much the layer COMPRESSES vs EXPANDS information.")
print(f"   High p = expansion/broad → many output dimensions relative to input")
print(f"   Low p = compression/narrow → few output dimensions relative to input")

# But gate_proj and up_proj have the same shape and different p.
# So it's not just dimensions. It's the COMPUTATIONAL ROLE.

print(f"\n   gate_proj and up_proj: same shape, different p → p encodes role, not shape.")
print(f"   But shape MAY interact with role.")

print(f"\n7. CRITICAL INSIGHT: The Zeckendorf representation")
print(f"   Every positive integer has a UNIQUE representation as a sum of")
print(f"   non-consecutive Fibonacci numbers (Zeckendorf's theorem).")
print(f"   The COMPLEXITY of this representation (number of terms) may")
print(f"   correlate with the layer's computational complexity.")

def zeckendorf(n):
    """Return Zeckendorf representation as list of Fibonacci indices."""
    if n <= 0:
        return []
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    result = []
    for f in reversed(fibs):
        if f <= n:
            result.append(f)
            n -= f
            if n == 0:
                break
    return result

print(f"\n   Zeckendorf representations:")
for name in sorted(layers, key=lambda x: -layers[x]['p']):
    info = layers[name]
    num_z = zeckendorf(info['num'])
    den_z = zeckendorf(info['den'])
    complexity = len(num_z) + len(den_z)
    print(f"   {name:<12s} p={info['num']}/{info['den']}  "
          f"num={info['num']}={'+'.join(map(str, num_z))}  "
          f"den={info['den']}={'+'.join(map(str, den_z))}  "
          f"complexity={complexity}")

print(f"\n   Correlation between p and Zeckendorf complexity:")
zs = []
ps = []
for name, info in layers.items():
    zc = len(zeckendorf(info['num'])) + len(zeckendorf(info['den']))
    zs.append(zc)
    ps.append(info['p'])
corr = np.corrcoef(ps, zs)[0, 1]
print(f"   r = {corr:.4f}")
if abs(corr) > 0.5:
    print(f"   → {'Positive' if corr > 0 else 'Negative'} correlation. Higher p = more Zeckendorf terms.")
else:
    print(f"   → Weak correlation. Zeckendorf complexity doesn't predict p.")

print(f"\n{'='*70}")
print(f"CONCLUSION: p is not predictable from simple layer properties alone.")
print(f"The fraction assignment appears to be an EMERGENT property of")
print(f"the optimization dynamics, not a designed feature.")
print(f"Cross-model validation is the critical next step.")
print(f"{'='*70}")
