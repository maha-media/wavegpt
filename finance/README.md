# Financial Market Spectral Analysis

Testing whether financial markets show the same φ-based spectral structure found in neural networks and biological connectomes.

## Key Finding

**Yes.** Stock correlation matrices follow bent power laws with φ-based exponents:

| Data | α | Best F/L | Error | R² |
|------|---|----------|-------|----|
| S&P 95 stocks (2yr correlation) | 1.296 | L/L=4/7 → φ^(4/7) | 1.3% | 0.993 |
| S&P 95 stocks (2yr covariance) | 1.357 | L/L=7/11 → φ^(7/11) | 0.2% | 0.981 |
| Cross-asset (15 classes) | 1.349 | L/L=18/29 | 0.1% | 0.985 |
| 10-year horizon | **1.620** | **F/L=1/1 = φ** | **0.1%** | 0.972 |

## Time Horizon Effect

Markets walk the φ-ladder over time — short-term is dispersed, long-term converges to the fundamental:

| Horizon | α | F/L match | Predicted | Error | Mode 1 energy |
|---------|---|-----------|-----------|-------|---------------|
| 1 year | 1.178 | F/L=1/3 | 1.174 | 0.3% | 40% |
| 2 years | 1.095 | F/L=2/11 | 1.091 | 0.3% | 66% |
| 5 years | 1.058 | F/L=2/18 | 1.055 | 0.3% | 68% |
| 10 years | 1.620 | **F/L=1/1 = φ** | 1.618 | 0.1% | 88% |
| 20 years | 1.600 | **F/L=1/1 = φ** | 1.618 | 1.1% | 88% |

**φ is the attractor, not a waypoint.** At 10 years the market locks onto the fundamental harmonic and stays there — 20 years shows the same exponent (1.1% error vs 0.1%, slight drift but same F/L fraction). The market doesn't cycle through harmonics on a longer timescale; φ is the ceiling.

The 1-year exponent (φ^(1/3) = 1.174) is the same harmonic as attn_o in transformers and command interneurons in C. elegans — the consensus operator across all three systems lands on the same F/L fraction at the "one cycle" timescale.

## Crisis = Spectral Collapse

α-volatility correlation: **0.44** (strong positive)

| Market state | Mean α | Interpretation |
|-------------|--------|----------------|
| Calm | 1.05 | Many independent signals, spectrum dispersed |
| Volatile/Crisis | 1.89 | Everything correlates, spectrum collapses to few modes |

Crises double the spectral exponent. All stocks start moving together. This is the same mechanism as semantic collapse in language models — the system falls to its fundamental and loses diversity.

## The Financial Consensus Operator

Mode 1 of the cross-asset correlation matrix (the "attn_o" of finance) is dominated by:

1. EU Equity (0.362)
2. High Yield bonds (0.358)
3. US Equity (0.355)
4. EM Equity (0.351)

Gold (0.118) and Oil (0.011) are the most independent — the "sensory neurons" responding to different inputs. VIX (0.328) loads strongly — fear is part of the consensus.

Sectors cluster tightly in SVD space — Energy at 0.02× baseline distance (extremely tight), Finance at 0.25×. Sector identity is as real in spectral space as neuron-type identity in the connectome.

## Connection to Thesis

Five systems now show the same structure:

| System | α regime | Consensus operator |
|--------|----------|-------------------|
| Transformers | (1/φ)^p | attn_o (output projection) |
| C. elegans structure | φ^p | Command interneurons |
| C. elegans functional | φ ≈ 1.62 | Whole-brain states |
| Markets (1yr) | φ^(1/3) = 1.174 | "The market" at 1 business cycle |
| Markets (10yr+) | φ^(1/1) = φ | "The market" at full convergence |

The pattern: constrained systems converge to φ-harmonic spectral structure. The specific harmonic depends on how long the optimization has run relative to the system's natural timescale:

- **LLM training run** → checkpoints at intermediate F/L harmonics, attn_o universally at 1/3
- **1 business cycle (1yr)** → φ^(1/3), the same attn_o harmonic — consensus over one cycle
- **Full market cycle (10yr)** → φ itself, the fundamental — and stays there at 20yr
- **Evolution (300M years)** → φ^(1/3) in biological regime, fully converged within its constraint envelope

The market is NOT an open system without φ-structure. It is a live φ-system that you sample at different phases. The measurement window determines which harmonic you observe. But φ is the attractor — once reached, the system stays there.

**Why spectral α fails as a trading signal**: you're measuring a 10-year standing wave with a 50-day rolling window. The wavelength is 50× longer than the instrument. It's not noise — it's undersampling.

## Scripts

### Spectral Analysis
- `market_spectral_analysis.py` — Initial analysis: S&P 500 correlation SVD, sector clustering
- `market_deep_analysis.py` — Deep analysis: time horizons (1yr–20yr), crisis detection, cross-asset, sector submatrices

### Trading System ("The Equation")
- `THE_EQUATION.md` — Full specification: Position = Regime × Conviction × Momentum
- `acquire_data.py` — Data pipeline: Yahoo Finance daily/hourly + macro + astro + weather
- `build_features.py` — 87-feature construction (spectral, price, macro, astro, weather)
- `find_all_signals.py` — Signal discovery: SPY/TLT, HYG, ARKK, VIX lead tech by 2-5 days
- `find_leaders.py` — Leading indicator identification and correlation analysis
- `regime_rotation.py` — 6-regime classifier: NORMAL/RISK_ON/FEAR/CRISIS/INFLATION/RECESSION
- `unified_trader.py` — Complete strategy: regime × leading indicators × momentum × dip buying
- `simulate_3yr.py` — Full 4-year backtest ($100K → $355K, Sharpe 1.46, MaxDD 19%)
- `live_trader.py` — Daily execution: pull prices → compute allocation → place GTC LIMIT orders
- `stream_trader.py` — Websocket streaming: real-time regime detection + rebalancing
- `fill_monitor.py` — Post-open monitor: chase/wait/skip unfilled orders based on fresh signals
- `multi_runner.py` — A/B testing: 10 strategy variants on Raspberry Pi

### Data
All data from Yahoo Finance (free). No API key needed.

- `data/` — Parquet files (daily/hourly), feature tensors, leader/tech closes
- `training_results/` — Model outputs, sweep results, signal discovery, simulation logs
- `trade_logs/` — Order execution logs and fill monitor reports

## Next Steps

- [x] Full harmonic ladder (1yr–20yr) — φ is the attractor at 10yr+
- [x] Trading system backtest — $100K → $355K over 4 years
- [x] Live trading on TastyTrade sandbox
- [x] Fill monitor for missed limit orders
- [ ] Historical crisis comparison (2008, 2020 COVID, 2022)
- [ ] Intraday tick data (does φ-structure appear at minute scale?)
- [ ] Options implied volatility surfaces
- [ ] Can α predict regime changes? (if α spikes → crisis incoming)
