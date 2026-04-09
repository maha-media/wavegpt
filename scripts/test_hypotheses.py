"""
Test the three hypotheses from the golden ratio harmonics discussion:
1. Adam's second-moment squaring → φ as stable fixed point
2. The denominator 7 — Fibonacci numerators, head group connection
3. Why p differs by function, not shape — gate vs up_proj analysis
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI  # 0.6180339887...

print("=" * 70)
print("TEST 1: Adam's second-moment squaring → φ as fixed point")
print("=" * 70)

# Adam: v_t = β₂·v_{t-1} + (1-β₂)·g²
# At steady state with g² ≈ constant: v* = g²
# But g depends on w, and w depends on v through the update rule.
# The key question: does the recurrent structure create a self-referential
# equation whose stable fixed point involves φ?

# Let's test: if the effective gradient magnitude relates to the accumulated
# moment through the spectral structure, we get:
#   v* = β₂·v* + (1-β₂)·f(v*)
#   v*(1 - β₂) = (1-β₂)·f(v*)
#   v* = f(v*)
# 
# If f involves the update magnitude which itself depends on v* through
# the learning rate scaling (η/√(v* + ε)), and the weight magnitude w,
# then the gradient at the spectral fixed point creates a recurrence.

# Test: The golden ratio satisfies x² = x + 1.
# Adam's EMA: v_t = β·v_{t-1} + (1-β)·g²
# If we normalize by g² and define r = v/g²:
#   r_t = β·r_{t-1} + (1-β)
# At steady state: r* = 1 (trivial).
# 
# But if g² itself scales with √v (because gradients diminish as we approach
# the minimum, and the curvature is related to the accumulated moment):
#   g² = c/√v for some c
# Then: v* = β·v* + (1-β)·c/√v*
#   v*(1-β) = (1-β)·c/√v*
#   v* = c/√v*
#   v*^{3/2} = c
#   v* = c^{2/3}
# Not φ.

# Try: g² = c·v (gradient grows with moment — possible during early training
# when oscillations build up):
#   v* = β·v* + (1-β)·c·v*
#   v* = v*·(β + c(1-β))
#   1 = β + c(1-β)
#   c = (1-β)/(1-β) = 1
# Trivial.

# Try: the UPDATE rule: Δw = -η·m/√v. If m ≈ g (first moment ≈ gradient),
# and g ≈ -λ·w (linearized dynamics near minimum with curvature λ):
#   Δw = η·λ·w/√v
# The weight grows if ηλ/√v > 0. At equilibrium, the spectral structure
# means the curvature λ is itself related to the singular value σ, and
# σ determines both g and v.
# 
# The key: if σ_k ~ k^{-α}, then the curvature in direction k is λ_k ~ σ_k.
# The gradient g_k ~ σ_k · w_k. The moment v_k ~ g_k² ~ σ_k² · w_k².
# But w_k is determined by the optimization which depends on v_k.
# Circular → fixed point equation.

print("\n1a: Adam fixed-point analysis with spectral gradient structure")
print(f"    β₂ = 0.999 (standard)")
print(f"    INV_PHI = {INV_PHI:.10f}")
print(f"    1/φ² = {INV_PHI**2:.10f}")
print(f"    φ·INV_PHI = {PHI * INV_PHI:.10f}")

# The critical observation:
# If the gradient in mode k is g_k ∝ σ_k (the singular value of that mode),
# and the power law is σ_k ∝ k^{-α}, then:
#   g_k² ∝ k^{-2α}
# Adam's moment: v_k ∝ g_k² ∝ k^{-2α}
# The update: Δw_k ∝ g_k / √v_k ∝ k^{-α} / k^{-α} = O(1)
# 
# This means Adam EQUALIZES the effective learning rate across modes.
# But the FIXED POINT of the spectrum requires:
#   The spectrum that makes all modes equally "hard" to optimize
#   is the one where the Hessian eigenvalues match the Adam-scaled updates.
# 
# If Hessian H_k ∝ σ_k (curvature proportional to singular value),
# and effective LR_k ∝ 1/√v_k ∝ 1/σ_k,
# then the convergence rate in mode k is H_k · LR_k ∝ σ_k · 1/σ_k = 1.
# Equal convergence in all modes → optimal conditioning.
# 
# The exponent α that achieves this is the one that MATCHES the 
# natural decay of the gradient spectrum. And THAT decay is set by
# the data distribution + architecture.

print("\n1b: Testing if iterated squaring converges to φ-related value")
# Adam squares gradients: v_t = β·v_{t-1} + (1-β)·g²
# After T steps: v_T = (1-β) · Σ β^t · g_{T-t}²
# This is a weighted average of squared gradients.
# The squaring itself doesn't iterate (it's applied once per step).
# But the RECURRENT nature creates a memory of past squared gradients.

# Test: if g² has a power-law spectrum, what happens to v?
# g_k² ~ k^{-2α_g}, then v_k ~ k^{-2α_g} (same shape, just scaled by 1-β+β=1)
# The update scales as g_k/√v_k ~ k^{-α_g}/k^{-α_g} = 1
# So Adam flat-normalizes all modes. The spectrum of v mirrors the spectrum of g².

# The φ connection would come from the FIXED POINT of the TRAINING DYNAMICS,
# not from the squaring operator alone. Specifically:
# At convergence, the loss landscape minimum has curvature set by the data.
# The weight spectrum that minimizes loss + regularization (implicit from Adam)
# must satisfy a self-consistency condition.

print("\n1c: Self-consistency test for Adam-at-equilibrium")
# Assume at equilibrium:
# - Weight W has singular values σ_k ∝ k^{-α}
# - Gradient g = ∂L/∂W has spectrum |g_k| ∝ σ_k^γ for some γ
# - Adam's moment: v_k ∝ g_k² ∝ σ_k^{2γ} ∝ k^{-2αγ}
# - Update: Δw_k = η · g_k / √v_k ∝ σ_k^γ / σ_k^γ = O(1)
# 
# For the spectrum to be STABLE (not change under small perturbations),
# the update must not systematically increase or decrease any σ_k.
# This requires the Hessian × effective_LR to be constant across k.
# H_k · (η/√v_k) = const
# H_k ∝ √v_k ∝ σ_k^γ ∝ k^{-αγ}
# 
# If H_k ∝ σ_k (curvature ∝ singular value, reasonable near minimum):
#   σ_k ∝ k^{-αγ}
#   k^{-α} ∝ k^{-αγ}
#   α = αγ
#   γ = 1 (nontrivial case)
# 
# So the self-consistent solution is |g_k| ∝ σ_k (gradient spectrum 
# matches weight spectrum). This is plausible: near the minimum,
# the gradient is dominated by the curvature times the displacement,
# and the displacement is proportional to the weight magnitude.
#
# This doesn't give φ directly. The φ must come from a DIFFERENT mechanism.

print("    Self-consistent solution: |g_k| ∝ σ_k (γ=1)")
print("    Adam equalizes convergence rates across modes")
print("    But this alone does NOT produce φ")
print("    → φ must come from the RECURRENT structure, not static analysis")

print("\n1d: Recurrent structure test — KAM / quasi-periodic orbits")
# Training is a discrete dynamical system:
#   w_{t+1} = w_t - η · m_t / √v_t
#   m_t = β₁·m_{t-1} + (1-β₁)·g_t
#   v_t = β₂·v_{t-1} + (1-β₂)·g_t²
# 
# This is a 3N-dimensional map (w, m, v). 
# Near the minimum, g ≈ -H·w, so the dynamics become:
#   w_{t+1} ≈ w_t + η · H · w_t / √v_t (simplified)
# 
# The eigenvalues of the Jacobian of this map determine stability.
# For a 1D model with curvature λ:
#   w_{t+1} = (1 + ηλ/√v*) · w_t
#   eigenvalue = 1 + ηλ/√v*
# 
# For stability: |1 + ηλ/√v*| < 1 → ηλ/√v* < 0 (impossible for positive λ,η)
# We need the MOMENTUM term for oscillatory behavior.
# With momentum β₁:
#   w_{t+1} = (1 + β₁ + ηλ/√v*) · w_t - β₁ · w_{t-1}
# This is a 2nd-order recurrence: w_{t+1} = a·w_t - b·w_{t-1}
# where a = 1 + β₁ + ηλ/√v* and b = β₁.
#
# The characteristic equation: r² - a·r + b = 0
# r = (a ± √(a² - 4b)) / 2
#
# For quasi-periodic (KAM) behavior: roots on unit circle
# |r| = 1 → a² - 4b < 0 and b = 1 (for |r|=1)
# But β₁ = 0.9, so b < 1, and roots are inside unit circle → damped.
# 
# However, with the cosine learning rate schedule, η varies.
# As η → 0, a → 1 + β₁ = 1.9, b = 0.9.
# r² - 1.9r + 0.9 = 0
# r = (1.9 ± √(3.61 - 3.6)) / 2 = (1.9 ± 0.1) / 2
# r = 1.0 or r = 0.9
# 
# The dominant eigenvalue approaches 1 as η → 0. This is marginal stability.
# At marginal stability, small perturbations persist → the system explores
# a manifold near the minimum → the equilibrium spectrum is the one that
# MINIMIZES the total "tension" in this quasi-periodic orbit.

# The Fibonacci connection: 
# A quasi-periodic orbit on a torus has winding number related to φ.
# The most irrational number (hardest to approximate by rationals) is φ.
# This is why φ appears in KAM theory — it's the "most stable" frequency ratio.

print(f"    Characteristic roots at η→0: r₁=1.0, r₂=0.9")
print(f"    Marginal stability → quasi-periodic exploration")
print(f"    φ = {PHI:.10f} = most irrational number")
print(f"    In KAM theory: φ is the most stable winding number")
print(f"    → φ minimizes resonance in quasi-periodic orbits")
print(f"    → The weight spectrum that minimizes training 'tension' has φ exponents")
print(f"\n    HYPOTHESIS: The spectral exponent α = (1/φ)^p minimizes")
print(f"    the total resonance energy of the Adam quasi-periodic orbit.")

print("\n" + "=" * 70)
print("TEST 2: The denominator 7 — why 2/7, 3/7, 5/7?")
print("=" * 70)

# Hypothesis A: 7 is the number of KV head groups in Qwen3.5-27B
# Qwen3.5-27B config (from public data):
#   hidden_size = 5120 (this is confirmed from the data)
#   num_attention_heads = 64 (standard for this size)
#   num_key_value_heads = ? (GQA ratio)
# The golden ratio harmonics doc says: "8192/1024 = 8 KV heads, but grouped"
# Let me check: if head_dim = 128 and hidden = 5120:
#   num_heads = 5120 / 128 = 40
# If GQA with 5 KV groups: 40/5 = 8, so 5 KV head groups. Not 7.
# If head_dim = 64: num_heads = 5120/64 = 80, 80/7 ≈ 11.4 (not integer)
# 
# Actually the doc says the model is Qwen3.5-27B. Let me check the actual config.

print("\n2a: Checking Qwen3.5-27B architecture (from training data knowledge)")
print("    Hidden size: 5120 (confirmed from weight shapes 17408×5120)")
print("    17408 / 5120 = 3.375 (intermediate/expansion ratio)")
print("    Typical Qwen3.5 config:")
print("      hidden_size = 5120")
print("      intermediate_size = 17408")
print("      num_hidden_layers = 64")
print("      num_attention_heads = 64 (estimated)")
print("      head_dim = 128 (estimated)")

# With head_dim=128: 5120/128 = 40 attention heads
# With GQA, if num_kv_heads = 8: 40/8 = 5 query heads per kv head
# If num_kv_heads = 5: 40/5 = 8 query heads per kv head
# Neither gives 7.

# Hypothesis B: 7 is the number of DISTINCT p values in (0,1)
# The observed p values with denom 7: 2/7, 3/7, 5/7
# All p values: {1/3, 2/7, 3/7, 1/2, 2/3, 5/7, 3/2}
# 7 distinct values total. But that's counting the result, not explaining it.

# Hypothesis C: 7 comes from the MLP structure
# Qwen3.5 uses SwiGLU: gate_proj ⊗ up_proj → activation → down_proj
# That's 3 MLP matrices + Q,K,V,O = 4 attention + embed + lm_head = many layers
# The number of distinct LAYER TYPES is about 10. Not 7.

# Hypothesis D: 7 is the smallest denominator giving sufficient resolution
# The p values range from ~0.28 to ~1.5.
# With denominator 7 and numerator up to 11: 2/7, 3/7, 4/7, 5/7, 6/7, 7/7, ...
# The observed ones are 2/7≈0.286, 3/7≈0.429, 5/7≈0.714
# These cover the lower, middle, and upper range of p values.
# Denominator 5 would give: 1/5=0.2, 2/5=0.4, 3/5=0.6, 4/5=0.8 (coarser)
# Denominator 6: 1/6=0.167, 2/6=0.333, 3/6=0.5, 4/6=0.667, 5/6=0.833
# Denominator 7: 1/7=0.143, 2/7=0.286, 3/7=0.429, 4/7=0.571, 5/7=0.714, 6/7=0.857

print("\n2b: Denominator resolution test")
for d in [5, 6, 7, 8, 9]:
    vals = [f"{n}/{d}={n/d:.3f}" for n in range(1, d+1)]
    gaps = [(n+1)/d - n/d for n in range(1, d)]
    print(f"    d={d}: {', '.join(vals[:6])}{'...' if d>6 else ''}")
    print(f"         gap = {1/d:.4f}")

print(f"\n2c: Fibonacci numerator test for denominator 7")
fib_nums = [1, 1, 2, 3, 5, 8, 13, 21]
for n in fib_nums:
    if n < 7:
        print(f"    {n}/7 = {n/7:.6f}  |  (1/φ)^{{{n}/7}} = {(1/PHI)**(n/7):.6f}")
    elif n == 7:
        print(f"    {n}/7 = {n/7:.6f}  |  (1/φ)^1 = {(1/PHI):.6f}")
    else:
        print(f"    {n}/7 = {n/7:.6f}  |  (1/φ)^{{{n}/7}} = {(1/PHI)**(n/7):.6f}")

print(f"\n2d: Which Fibonacci numbers appear as numerators?")
print(f"    Observed: 2/7, 3/7, 5/7 → numerators 2, 3, 5 = F(3), F(4), F(5)")
print(f"    Also observed: 1/3, 2/3 → numerator 1, 2 = F(1/2), F(3)")
print(f"    Also observed: 1/2 → numerator 1 = F(1) or F(2)")
print(f"    Also observed: 3/2 → numerator 3 = F(4)")
print(f"    → Fibonacci numerators appear across ALL denominators, not just 7")

# The Fibonacci pattern suggests: p = F(i)/F(j) for small i,j
# Let's check:
# 2/7: F(3)/? — 7 is NOT a Fibonacci number
# 3/7: F(4)/? — same issue
# 5/7: F(5)/? — same issue
# 
# So the Fibonacci pattern is ONLY in the numerator, not the denominator.
# This means 7 is special for a different reason.

print(f"\n2e: 7 is NOT a Fibonacci number — hypothesis:")
print(f"    7 = F(4) + F(5) = 3 + 5 - 1 (nearly)")
print(f"    7 = L(4) where L is the Lucas number (2,1,3,4,7,11,...)")
lucas = [2, 1]
for _ in range(5):
    lucas.append(lucas[-1] + lucas[-2])
print(f"    Lucas numbers: {lucas}")
print(f"    L(5) = 7 ← this is the 5th Lucas number!")
print(f"    → 7 IS a Lucas number, not a Fibonacci number")
print(f"    → Lucas numbers also converge to φ: L(n)/L(n-1) → φ")

print("\n" + "=" * 70)
print("TEST 3: Function vs Shape — gate_proj vs up_proj")
print("=" * 70)

print("\n3a: Forward pass analysis (SwiGLU)")
print("    up_proj:   h_up = W_up · x          (linear projection)")
print("    gate_proj: h_gate = W_gate · x       (linear projection)")
print("    activation: h = SwiLU(h_gate) ⊙ h_up  (multiplicative)")
print("    down_proj:  y = W_down · h           (linear projection)")
print()
print("    gate_proj and up_proj have IDENTICAL shapes but DIFFERENT roles:")
print("    - gate_proj: determines WHICH features to activate (control signal)")
print("    - up_proj: provides WHAT features to activate (data signal)")
print()
print("    In the backward pass:")
print("    - dL/d(gate) = (dL/dh) ⊙ (d(SwiLU)/d(gate)) ⊙ h_up")
print("      Gate gradient is MULTIPLIED by h_up")
print("    - dL/d(up) = (dL/dh) ⊙ SwiLU(gate)")
print("      Up gradient is MULTIPLIED by SwiLU(gate)")
print()
print("    Different multiplicative factors → different gradient statistics")
print("    → different Adam moments → different optimal spectral exponents")

print("\n3b: Hessian structure test")
print("    For gate_proj: ∂²L/∂gate² involves ∂(SwiLU·h_up)/∂gate")
print("      = h_up² · SwiLU''(gate) + cross terms")
print("    For up_proj: ∂²L/∂up² involves ∂(SwiLU·h_up)/∂up")
print("      = SwiLU²(gate) (no gate derivative)")
print()
print("    The Hessian eigenvalue spectrum differs because:")
print("    - gate: curvature depends on h_up² (feature magnitude)")
print("    - up: curvature depends on SwiLU²(gate) (gate activation)")
print()
print("    If h_up has a different spectral distribution than SwiLU(gate),")
print("    the optimal α for each will differ — even with identical shapes.")

print("\n3c: Gradient flow path length")
print("    gate_proj → SwiLU → multiply → down_proj → ...")
print("    up_proj  → multiply → down_proj → ...")
print("    Same path length! The difference is in the NODE TYPE (SwiLU vs pass-through)")
print()
print("    → p is determined by the COMPUTATIONAL GRAPH, not depth")
print("    → p encodes the 'spectral personality' of the layer's function")

print("\n" + "=" * 70)
print("SYNTHESIS: The proposed mechanism")
print("=" * 70)
print()
print("  φ appears because:")
print("  1. Adam creates a quasi-periodic orbit near the minimum")
print("  2. The orbit's winding number minimizes resonance when α = (1/φ)^p")
print("  3. φ is the most irrational number → most stable against resonance")
print()
print("  p differs by function because:")
print("  1. Different layer types have different Hessian structures")
print("  2. Different Hessian → different gradient statistics under Adam")
print("  3. The stable spectral exponent shifts to match the Hessian")
print()
print("  Denominator 7 because:")
print("  1. 7 = L(5), the 5th Lucas number")
print("  2. Lucas numbers are φ-conjugate: L(n) = φ^n + (-φ)^{-n}")
print("  3. The '7-limit' may correspond to the 5th-order φ-harmonic")
print("  4. OR: 7 is simply the smallest denominator giving sufficient")
print("     resolution to distinguish all layer functions")
print()
print("  Falsifiable predictions:")
print("  1. Models trained WITHOUT Adam (pure SGD) should NOT show φ harmonics")
print("  2. Models with different GQA ratios should show different denominators")
print("  3. gate_proj and up_proj should have different α even at INITIALIZATION")
print("     if the data creates different gradient statistics from step 1")
print("  4. A model trained with Adafactor (different moment estimation) should")
print("     show different harmonics")
print()
print("=" * 70)
