# The Equation

## Position = Regime × Conviction × Momentum

```
For each day t:

1. REGIME (where to put money)
   ├─ CRISIS:     VIX_z > 1.0 AND Credit_z < -0.5  → 95% tech (buy the crash)
   ├─ FEAR:       VIX_z > 0.5                       → 60% tech, 40% SHY/GLD/XLV
   ├─ RISK_ON:    Credit_z > 0.5 AND VIX_z < 0      → 50% tech, 50% USO/XLE/GLD
   ├─ INFLATION:  Gold_z > 1.0                       → 30% tech, 70% EFA/XLF/EEM/IWM
   ├─ RECESSION:  YieldCurve_z < -1.0                → 10% tech, 90% USO/SLV/XLE
   └─ NORMAL:     else                               → 90% tech, 10% XLK/XLP

   where z-scores use 50-day rolling mean/std

2. SINGULARITY OVERRIDE
   IF 6+ of 7 tech stocks have positive 20d momentum
   AND average 20d momentum > 5%:
      → 95% tech, ignore regime, ignore risk switch

   IF 5+ positive AND avg > 2%:
      → at least 80% tech

3. LEADING INDICATOR CONVICTION (adjusts tech_pct by up to ±40%)
   leader_score = weighted_avg(
     -ARKK_5d  × 4.0    # speculative washout = buy tech
     +VIX_5d   × 3.5    # fear spike = buy tech  
     -HYG_5d   × 5.0    # credit exodus = tech incoming
     -KWEB_5d  × 3.0    # China dump = US tech bid
     -FXI_5d   × 3.0    # same
     -SMH_5d   × 3.0    # semis washout = tech bounce
     +LQD_2d   × 3.5    # quality bid = tech follows
     +UUP_5d   × 2.5    # strong dollar = US tech inflow
     -ETH_2d   × 1.0    # crypto froth = tech cools
   )
   tech_pct += clip(leader_score × 5.0, -0.40, +0.40)

4. BEAR MARKET RISK SWITCH (unless singularity mode)
   IF avg 50d tech momentum < -5% AND breadth ≤ 2/7:
      → tech allocation × 0.3
   IF avg 50d momentum < 0 AND breadth ≤ 4/7:
      → tech allocation × 0.6

5. TECH WEIGHTS (within tech allocation)
   For each stock i in Mag 7:
     mom_score[i] = 0.15 × z(mom_10d) + 0.20 × z(mom_20d)
                  + 0.30 × z(mom_50d) + 0.35 × z(mom_100d)
   
   weights = softmax(mom_scores)
   clip to [5%, 35%] per stock
   
   DIP BOOST: if stock dropped >2% yesterday:
     weight[i] *= (1 + |drop| × 3)
     weight[NVDA] *= (1 + |drop| × 2)   # NVDA is the bounce magnet

6. DEFENSIVE WEIGHTS (within non-tech allocation)
   Momentum-weight among regime's defensive assets
   Only positive-momentum assets get allocation

7. FINAL ALLOCATION
   alloc = {tech_weights × tech_pct} ∪ {def_weights × other_pct}
```

## The Numbers

```
$100K → $355K in 4 years
Annual return: +37.5%
Sharpe ratio:  1.46
Max drawdown:  19.0%

vs Equal Weight Tech Buy & Hold:
  +255% vs +141% (1.8× better)
  19% drawdown vs 45% drawdown (2.4× safer)
```

## What Each Piece Contributes

| Component | What it does | Impact |
|-----------|-------------|--------|
| Regime rotation | Puts money where it works (tech/energy/gold/bonds) | Saved 40% in 2022 |
| Singularity override | Goes all-in when tech is unanimously ripping | Caught 2023 AI rally |
| Leading indicators | ARKK/VIX/HYG/KWEB predict tech 2-5 days ahead | Fine-tunes conviction |
| Momentum weighting | Overweights winners (NVDA got 16% avg vs 14% equal) | Extra +12% over equal weight |
| Dip buying | NVDA is the bounce magnet after any tech drop | Improved entry timing |
| Bear risk switch | Cuts tech when 50d momentum negative + breadth weak | Prevented 2022 wipeout |

## The Signals That Matter (ranked by importance)

1. **Tech breadth + momentum** (singularity detector) — most impactful single signal
2. **VIX z-score + Credit z-score** (regime classifier) — determines asset class allocation  
3. **HYG 5d momentum** (strongest leading indicator, corr 0.179) — 5-day advance warning
4. **Individual stock 10-100d momentum** (corr 0.30-0.36) — picks which tech stocks to overweight
5. **ARKK 5d momentum** (V-recovery detector) — speculative washout predicts tech bounce
6. **50d breadth** (bear market switch) — when most stocks below 50d MA, reduce exposure
7. **Gold z-score** (inflation detector) — triggers rotation to international/commodities
8. **NVDA dip-buying** (flow analysis) — NVDA catches rebound flow from any tech drop

## What Doesn't Matter

- Weather (corr < 0.05, noise)
- Astrology beyond Venus retrograde (corr < 0.02)
- Price features on SPY (mean-reverts but too weak)
- Neural networks (overfit on this data size)
- Any signal at 5-minute resolution (noise dominates)
