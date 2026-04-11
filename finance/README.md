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

| Horizon | α | F/L match | Mode 1 energy |
|---------|---|-----------|---------------|
| 1 year | 1.155 | F/L=2/7 | 40% |
| 2 years | 1.095 | F/L=2/11 | 66% |
| 5 years | 1.059 | F/L=2/18 | 68% |
| 10 years | 1.620 | **φ itself** | 88% |

Over 10 years, α = φ at 0.1% error. The market converges to a single dominant mode.

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
| Markets (short-term) | φ^p | Sector correlations |
| Markets (long-term) | φ ≈ 1.62 | "The market" itself |

The pattern: systems with long optimization history (evolution, decades of trading) converge to α ≈ φ. Systems mid-optimization (trained models, short-term markets) sit at intermediate F/L harmonics. The fundamental is where everything ends up given enough time.

## Scripts

- `market_spectral_analysis.py` — Initial analysis: S&P 500 correlation SVD, sector clustering
- `market_deep_analysis.py` — Deep analysis: time horizons, crisis detection, cross-asset, sector submatrices

## Data

All data from Yahoo Finance (free). No API key needed.

- `market-spectral.json` — Initial run results
- `market-deep-spectral.json` — Deep analysis results

## Next Steps

- [ ] Full S&P 500 analysis (batch download to avoid API limits)
- [ ] Historical crisis comparison (2008, 2020 COVID, 2022)
- [ ] Intraday tick data (does φ-structure appear at minute scale?)
- [ ] Options implied volatility surfaces
- [ ] Crypto market correlation matrices
- [ ] Bond yield curve decomposition
- [ ] Can α predict regime changes? (if α spikes → crisis incoming)
