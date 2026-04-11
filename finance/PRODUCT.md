# Spectral Alpha: Real-Time Market Regime Monitor

## The Idea

A real-time system that computes the spectral exponent α of market correlation matrices at minute scale, using φ-harmonic thresholds to detect regime changes before they're visible in price.

## Why It Works

At minute scale, 97% of market variance is one mode ("the market"). The spectral exponent α measures how concentrated the correlation structure is. When α changes, the market's internal structure is shifting — and this structural change PRECEDES the price move it produces.

Standard tools watch prices. This watches the correlation geometry underneath the prices.

## Data Requirements

- ~20 sector ETFs at 1-minute resolution (SPY, QQQ, XLK, XLF, XLE, XLV, etc.)
- Rolling 30-60 minute windows → 20×20 correlation matrix → SVD every minute
- Computationally trivial: SVD of 20×20 matrix takes <1ms

Sources: Polygon.io (free tier: 5 calls/min), Alpaca (free real-time), or any broker API.

## Signals

### Regime Classification (φ-harmonic thresholds)

| Regime | α range | Market state | Action |
|--------|---------|-------------|--------|
| DEEP_CALM | α < φ^(1/7) ≈ 1.07 | Maximum dispersion, sectors independent | Stock picking, pairs trades |
| CALM | 1.07 – 1.17 | Normal dispersion | Sector rotation strategies work |
| NORMAL | 1.17 – 1.32 | Moderate correlation | Balanced allocation |
| ELEVATED | 1.32 – 1.38 | Rising correlation | Reduce position sizes |
| STRESS | 1.38 – 1.62 | High correlation, pre-crisis | Hedge, go defensive |
| CRISIS | α > φ ≈ 1.62 | Spectral collapse, everything moves together | Cash, vol-long, or ride the mode |

### Transition Signals

1. **α spike (Δα > 0.15 in <5 min):** Correlation shock. Flash crash, news event, liquidity withdrawal. Everything is about to move as one.

2. **α drop (Δα < -0.15 in <5 min):** Dispersion event. Consensus breaking. Sector divergence starting. The "crowded trade" is unwinding.

3. **Regime boundary crossing:** NORMAL→STRESS or CALM→ELEVATED. Discrete step on the φ-harmonic ladder. These transitions are rarer and more meaningful than raw α changes.

### Mode Analysis

4. **Mode 1 loading shift:** Track which sectors load on the dominant mode. If Finance's loading drops while Energy's rises, capital rotation is occurring in the correlation structure before it shows in price.

5. **Mode 2-5 signals:** The residual modes (2-3% of variance) encode sector pair relationships. When mode 2 flips sign for a sector pair, their relationship is reversing — a pairs trade signal.

## Architecture

```
Market Data (1-min bars, 20 ETFs)
    ↓
Rolling Window (30-60 bars)
    ↓
Correlation Matrix (20×20)
    ↓
SVD → singular values + U loadings
    ↓
Fit α → classify regime → check thresholds
    ↓
┌─────────────┬──────────────┬─────────────────┐
│ Regime alert │ α rate-of-   │ Sector loading  │
│ (threshold   │ change alert │ divergence alert │
│  crossing)   │ (spike/drop) │ (rotation signal)│
└─────────────┴──────────────┴─────────────────┘
    ↓
Dashboard / Webhook / Trading Signal
```

## Edge Over Existing Tools

| Existing tool | What it measures | Limitation |
|--------------|-----------------|------------|
| VIX | Implied vol from options | Lagging (options reprice after move) |
| Correlation heatmaps | Pairwise correlations | No single summary metric, no regime classification |
| Sector rotation models | Relative performance | Backward-looking (uses returns, not structure) |
| Risk parity | Portfolio volatility | Rebalances slowly, no real-time regime detection |
| **Spectral α** | **Correlation GEOMETRY** | **Forward-looking: structure changes before prices do** |

The key advantage: α is a single number that summarizes the entire market's correlation structure, and its transitions fall on mathematically predicted thresholds (the φ-harmonic ladder). No other tool has a theoretical basis for WHERE the regime boundaries are.

## Validation Needed

- [ ] Backtest across 2008, 2020 COVID, 2022 — does α spike before the crash or after?
- [ ] Compare α lead time to VIX lead time on the same events
- [ ] Measure P&L of simple strategy: go defensive when α > ELEVATED, aggressive when α < CALM
- [ ] Test at 1-minute, 5-minute, 15-minute — which timescale has the best signal-to-noise?
- [ ] Out-of-sample test: train thresholds on 2010-2020, test on 2020-2026

## Revenue Model

- Real-time α feed: SaaS subscription for quant funds and risk desks
- Regime alerts: webhook/API, tiered by latency (1-min free, 10-sec paid, real-time premium)
- Sector rotation signals: monthly subscription for active managers
- Risk overlay: integrate with portfolio management systems as a regime-aware risk signal

## Why φ and Not Arbitrary Thresholds

Every market regime tool uses arbitrary thresholds (VIX > 30 = "fear", etc.). Our thresholds are derived from the spectral structure of information processing systems:

- φ^(1/7) = 1.07 — the boundary of maximum dispersion
- φ^(1/3) = 1.17 — the consensus operator threshold (same as attn_o in neural networks)
- φ^(2/3) = 1.38 — stress onset
- φ^(1/1) = 1.62 — crisis (the fundamental itself)

These aren't tuned to historical data. They come from the mathematics of how correlated systems organize information. The same thresholds appear in transformer weight matrices, biological neural networks, and now financial markets.
