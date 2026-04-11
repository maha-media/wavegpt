"""
Market spectral analysis: does financial data show φ-based spectral structure?

The stock correlation matrix is a natural analog:
  - Stocks = neurons
  - Correlations = synaptic weights
  - Mode 1 = "the market" (consensus, like attn_o / command interneurons)
  - Sectors = neuron types

Tests:
  1. SVD of stock correlation matrix → bent power law fit
  2. Does α match an F/L fraction?
  3. Do sectors cluster in U-space like neuron types?
  4. Does mode 1 (market consensus) have special spectral structure?
  5. Energy concentration at φ-power thresholds?

Data: S&P 500 daily returns from Yahoo Finance (free).
"""

import json
import sys
from math import sqrt, log
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import curve_fit

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

PHI_POWERS = {
    '1/φ⁴': INV_PHI**4,
    '1/φ³': INV_PHI**3,
    '1/φ²': INV_PHI**2,
    '1/φ': INV_PHI,
}

FL_FRACS = []
FIB = [1, 1, 2, 3, 5, 8, 13, 21]
LUC = [1, 3, 4, 7, 11, 18, 29, 47]
for f in FIB:
    for l in LUC:
        FL_FRACS.append((f, l, f/l, f"F/L={f}/{l}"))
for l1 in LUC:
    for l2 in LUC:
        if l1 != l2:
            FL_FRACS.append((l1, l2, l1/l2, f"L/L={l1}/{l2}"))


def bent_power_law(k, A, k0, alpha):
    return A * (k + k0) ** (-alpha)


def fit_free_alpha(S):
    S = S[S > 1e-10]
    n = len(S)
    if n < 4:
        return None
    k = np.arange(1, n + 1, dtype=np.float64)
    try:
        popt, _ = curve_fit(bent_power_law, k, S.astype(np.float64),
            p0=[S[0], max(1.0, n * 0.1), INV_PHI],
            bounds=([0, 0, 0.01], [S[0] * 100, n * 5, 3.0]), maxfev=20000)
        A, k0, alpha = popt
        pred = bent_power_law(k, *popt)
        ss_res = np.sum((S[:n] - pred) ** 2)
        ss_tot = np.sum((S[:n] - np.mean(S[:n])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {'A': float(A), 'k0': float(k0), 'alpha': float(alpha), 'r2': float(r2), 'n': n}
    except:
        return None


def best_fl_match(alpha):
    best = None
    best_err = float('inf')
    for _, _, p_val, label in FL_FRACS:
        for base, regime in [(INV_PHI, '(1/φ)^p'), (PHI, 'φ^p')]:
            predicted = base ** p_val
            err = abs(alpha - predicted) / alpha * 100
            if err < best_err:
                best_err = err
                best = (label, predicted, err, regime)
    return best


def best_phi_power(ratio):
    best_name, best_err = None, float('inf')
    for name, target in PHI_POWERS.items():
        err = abs(ratio - target) / target * 100
        if err < best_err:
            best_err = err
            best_name = name
    return best_name, best_err


# S&P 500 sector classifications (GICS)
SECTOR_MAP = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
    'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Disc',
    'XLP': 'Consumer Staples', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'XLB': 'Materials', 'XLC': 'Communication',
}

# Major stocks by sector for classification
STOCK_SECTORS = {
    # Technology
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOG': 'Tech', 'GOOGL': 'Tech',
    'META': 'Tech', 'AVGO': 'Tech', 'ADBE': 'Tech', 'CRM': 'Tech', 'AMD': 'Tech',
    'INTC': 'Tech', 'QCOM': 'Tech', 'TXN': 'Tech', 'AMAT': 'Tech', 'MU': 'Tech',
    'ORCL': 'Tech', 'NOW': 'Tech', 'INTU': 'Tech', 'SNPS': 'Tech', 'CDNS': 'Tech',
    # Financials
    'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
    'BRK-B': 'Finance', 'C': 'Finance', 'BLK': 'Finance', 'SCHW': 'Finance', 'AXP': 'Finance',
    'V': 'Finance', 'MA': 'Finance', 'COF': 'Finance', 'USB': 'Finance', 'PNC': 'Finance',
    # Healthcare
    'UNH': 'Health', 'JNJ': 'Health', 'LLY': 'Health', 'PFE': 'Health', 'ABBV': 'Health',
    'MRK': 'Health', 'TMO': 'Health', 'ABT': 'Health', 'DHR': 'Health', 'BMY': 'Health',
    'AMGN': 'Health', 'GILD': 'Health', 'ISRG': 'Health', 'MDT': 'Health', 'CVS': 'Health',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
    'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'OXY': 'Energy', 'HAL': 'Energy',
    # Consumer
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer', 'MCD': 'Consumer', 'NKE': 'Consumer',
    'SBUX': 'Consumer', 'TGT': 'Consumer', 'LOW': 'Consumer', 'COST': 'Consumer', 'WMT': 'Consumer',
    'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'PM': 'Consumer', 'CL': 'Consumer',
    # Industrials
    'CAT': 'Industrial', 'BA': 'Industrial', 'HON': 'Industrial', 'UPS': 'Industrial', 'RTX': 'Industrial',
    'GE': 'Industrial', 'MMM': 'Industrial', 'LMT': 'Industrial', 'DE': 'Industrial', 'UNP': 'Industrial',
    # Utilities / Real Estate
    'NEE': 'Utility', 'DUK': 'Utility', 'SO': 'Utility', 'D': 'Utility', 'AEP': 'Utility',
    'AMT': 'REIT', 'PLD': 'REIT', 'CCI': 'REIT', 'EQIX': 'REIT', 'SPG': 'REIT',
}


def download_stock_data(tickers, period_days=504):
    """Download daily close prices via yfinance."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=period_days)

    print(f"  Downloading {len(tickers)} stocks, {period_days} days...")
    data = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, __import__('pandas').MultiIndex):
        closes = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
    else:
        closes = data

    return closes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', default=list(STOCK_SECTORS.keys()))
    parser.add_argument('--period', type=int, default=504, help='Trading days (~2 years)')
    parser.add_argument('--output', default='runs/market-spectral.json')
    args = parser.parse_args()

    print("=" * 70)
    print("MARKET SPECTRAL ANALYSIS: φ IN FINANCIAL DATA?")
    print("=" * 70)

    tickers = args.tickers
    print(f"\n  Tickers: {len(tickers)} stocks")
    print(f"  Period: {args.period} trading days (~{args.period/252:.1f} years)")

    # Download data
    closes = download_stock_data(tickers, args.period)

    # Compute daily returns
    returns = closes.pct_change().dropna()

    # Drop stocks with too many NaN
    valid_cols = returns.columns[returns.isna().sum() < len(returns) * 0.1]
    returns = returns[valid_cols].dropna()

    n_stocks = len(returns.columns)
    n_days = len(returns)
    print(f"\n  Valid: {n_stocks} stocks × {n_days} days")

    stock_names = list(returns.columns)
    R = returns.values  # (days, stocks)

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("1. CORRELATION MATRIX SVD")
    print(f"{'='*70}")

    corr = np.corrcoef(R.T)
    corr = np.nan_to_num(corr, nan=0.0)

    U, S, Vt = np.linalg.svd(corr, full_matrices=False)
    S_pos = S[S > 1e-10]

    fit = fit_free_alpha(S_pos)
    print(f"\n  Singular values: {len(S_pos)}")
    print(f"  Top 5: {S_pos[:5].round(3)}")
    if fit:
        print(f"  Bent power law: α = {fit['alpha']:.4f}, k₀ = {fit['k0']:.1f}, R² = {fit['r2']:.4f}")
        fl = best_fl_match(fit['alpha'])
        if fl:
            print(f"  Best F/L match: {fl[0]} → {fl[1]:.4f} ({fl[2]:.1f}% error, {fl[3]})")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("2. COVARIANCE MATRIX SVD")
    print(f"{'='*70}")

    cov = np.cov(R.T)
    S_cov = np.linalg.svd(cov, compute_uv=False)
    S_cov = S_cov[S_cov > 1e-10]

    fit_cov = fit_free_alpha(S_cov)
    if fit_cov:
        print(f"  Bent power law: α = {fit_cov['alpha']:.4f}, k₀ = {fit_cov['k0']:.1f}, R² = {fit_cov['r2']:.4f}")
        fl_cov = best_fl_match(fit_cov['alpha'])
        if fl_cov:
            print(f"  Best F/L match: {fl_cov[0]} → {fl_cov[1]:.4f} ({fl_cov[2]:.1f}% error, {fl_cov[3]})")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("3. ENERGY CONCENTRATION")
    print(f"{'='*70}")

    energy = S_pos ** 2
    cum = np.cumsum(energy) / energy.sum()

    print(f"\n  {'Threshold':<12} {'Rank k':<8} {'k/n':<10} {'φ target':<10} {'Error':<8}")
    print(f"  {'-'*50}")
    for thresh in [0.50, 0.75, 0.90, 0.95, 0.99]:
        k = np.searchsorted(cum, thresh) + 1
        ratio = k / len(S_pos)
        name, err = best_phi_power(ratio)
        marker = ' ◄' if err < 10 else ''
        print(f"  {thresh:<12.2f} {k:<8} {ratio:<10.4f} {name:<10} {err:<7.1f}%{marker}")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("4. SECTOR CLUSTERING IN U-SPACE")
    print(f"{'='*70}")

    # Classify stocks by sector
    sectors = {}
    for i, name in enumerate(stock_names):
        # Handle tickers that might have different formats
        clean = str(name).upper().replace('.', '-')
        sect = STOCK_SECTORS.get(clean, 'Other')
        sectors[i] = sect

    sector_counts = defaultdict(int)
    for s in sectors.values():
        sector_counts[s] += 1
    print(f"\n  Sector distribution:")
    for s, c in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:<15}: {c}")

    # U-space clustering (top-k singular vectors)
    from scipy.spatial.distance import pdist, squareform

    for k in [3, 5, 10]:
        if k > U.shape[1]:
            continue
        U_k = U[:, :k]
        dists = squareform(pdist(U_k, metric='cosine'))
        overall_mean = dists[np.triu_indices(n_stocks, k=1)].mean()

        print(f"\n  Top-{k} singular vectors:")
        for sect in sorted(set(sectors.values())):
            idx = [i for i, s in sectors.items() if s == sect]
            if len(idx) < 3:
                continue
            intra = []
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    intra.append(dists[idx[i], idx[j]])
            if intra:
                mean_intra = np.mean(intra)
                ratio = mean_intra / overall_mean
                marker = '◄ CLUSTERED' if ratio < 0.8 else ('◄ DISPERSED' if ratio > 1.2 else '')
                print(f"    {sect:<15}: {mean_intra:.4f} ({ratio:.2f}× overall) {marker}")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("5. MODE 1 = THE MARKET (consensus operator)")
    print(f"{'='*70}")

    # First singular vector = market factor
    market_loadings = np.abs(U[:, 0])
    print(f"\n  Mode 1 captures {cum[0]*100:.1f}% of total variance")
    print(f"  σ₁/σ₂ = {S_pos[0]/S_pos[1]:.2f}")

    # Which sectors load most heavily on mode 1?
    sector_loadings = defaultdict(list)
    for i, sect in sectors.items():
        if i < len(market_loadings):
            sector_loadings[sect].append(market_loadings[i])

    print(f"\n  Sector loadings on Mode 1 (market consensus):")
    for sect in sorted(sector_loadings.keys(), key=lambda s: -np.mean(sector_loadings[s])):
        vals = sector_loadings[sect]
        print(f"    {sect:<15}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("6. TIME-VARYING ANALYSIS (rolling windows)")
    print(f"{'='*70}")

    window = 63  # ~3 months
    stride = 21  # ~1 month
    alphas_over_time = []

    for start_idx in range(0, n_days - window, stride):
        R_window = R[start_idx:start_idx + window, :]
        # Drop stocks with zero variance in window
        var = np.var(R_window, axis=0)
        valid = var > 1e-12
        if valid.sum() < 10:
            continue
        corr_w = np.corrcoef(R_window[:, valid].T)
        corr_w = np.nan_to_num(corr_w, nan=0.0)
        S_w = np.linalg.svd(corr_w, compute_uv=False)
        S_w = S_w[S_w > 1e-10]
        fit_w = fit_free_alpha(S_w)
        if fit_w and fit_w['r2'] > 0.9:
            alphas_over_time.append(fit_w['alpha'])

    if alphas_over_time:
        a_arr = np.array(alphas_over_time)
        print(f"\n  Rolling 3-month windows ({len(a_arr)} windows):")
        print(f"    Mean α: {np.mean(a_arr):.4f} ± {np.std(a_arr):.4f}")
        print(f"    Min: {np.min(a_arr):.4f}, Max: {np.max(a_arr):.4f}")

        fl_roll = best_fl_match(np.mean(a_arr))
        if fl_roll:
            print(f"    Best F/L match: {fl_roll[0]} → {fl_roll[1]:.4f} ({fl_roll[2]:.1f}%)")

        # Does α change during high-volatility periods?
        # (Higher α = more concentrated = more correlated = crisis?)
        q75 = np.percentile(a_arr, 75)
        q25 = np.percentile(a_arr, 25)
        print(f"    High-vol (top 25%): α ≥ {q75:.4f}")
        print(f"    Low-vol (bottom 25%): α ≤ {q25:.4f}")

    # Save
    output = {
        'n_stocks': n_stocks,
        'n_days': n_days,
        'tickers': [str(t) for t in stock_names],
        'correlation_fit': fit,
        'covariance_fit': fit_cov,
        'rolling_alphas': alphas_over_time if alphas_over_time else None,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {args.output}")


if __name__ == '__main__':
    main()
