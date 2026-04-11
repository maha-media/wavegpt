"""
Deep market spectral analysis.

Expands on the initial finding (α ≈ 1.30, R² = 0.993) with:
1. Sector-specific submatrices (like C. elegans neuron-type analysis)
2. Crisis vs calm periods (does α shift during crashes?)
3. Multiple time horizons (1yr, 2yr, 5yr, 10yr)
4. Cross-asset classes (stocks, bonds, commodities, currencies)
5. Individual sector correlation matrices
6. The "financial attn_o" — which sector acts as consensus operator?
"""

import json
import sys
from math import sqrt, log
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

FIB = [1, 1, 2, 3, 5, 8, 13, 21]
LUC = [1, 3, 4, 7, 11, 18, 29, 47]

FL_FRACS = []
for f in FIB:
    for l in LUC:
        FL_FRACS.append((f/l, f"F/L={f}/{l}"))
for l1 in LUC:
    for l2 in LUC:
        if l1 != l2:
            FL_FRACS.append((l1/l2, f"L/L={l1}/{l2}"))


def bent_power_law(k, A, k0, alpha):
    return A * (k + k0) ** (-alpha)


def fit_spectrum(S):
    S = S[S > 1e-10]
    n = len(S)
    if n < 4:
        return None
    k = np.arange(1, n + 1, dtype=np.float64)
    try:
        popt, _ = curve_fit(bent_power_law, k, S.astype(np.float64),
            p0=[S[0], max(1.0, n*0.1), INV_PHI],
            bounds=([0, 0, 0.01], [S[0]*100, n*5, 3.0]), maxfev=20000)
        pred = bent_power_law(k, *popt)
        ss_res = np.sum((S[:n] - pred)**2)
        ss_tot = np.sum((S[:n] - np.mean(S[:n]))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        return {'alpha': float(popt[2]), 'k0': float(popt[1]), 'A': float(popt[0]),
                'r2': float(r2), 'n': n}
    except:
        return None


def best_fl(alpha):
    best_err = float('inf')
    best = None
    for p_val, label in FL_FRACS:
        for base, regime in [(INV_PHI, '(1/φ)^p'), (PHI, 'φ^p')]:
            pred = base ** p_val
            err = abs(alpha - pred) / alpha * 100
            if err < best_err:
                best_err = err
                best = (label, pred, err, regime)
    return best


# Comprehensive stock universe by sector
SECTORS = {
    'Tech': ['AAPL','MSFT','NVDA','GOOG','META','AVGO','ADBE','CRM','AMD','INTC',
             'QCOM','TXN','AMAT','MU','ORCL','NOW','INTU','SNPS','CDNS','MRVL',
             'KLAC','LRCX','FTNT','PANW','CRWD'],
    'Finance': ['JPM','BAC','WFC','GS','MS','C','BLK','SCHW','AXP','V','MA',
                'COF','USB','PNC','TFC','BK','STT','ICE','CME','SPGI'],
    'Health': ['UNH','JNJ','LLY','PFE','ABBV','MRK','TMO','ABT','DHR','BMY',
               'AMGN','GILD','ISRG','MDT','CVS','CI','ELV','HCA','ZTS','REGN'],
    'Energy': ['XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','HAL',
               'DVN','FANG','HES','BKR','WMB'],
    'Consumer': ['AMZN','TSLA','HD','MCD','NKE','SBUX','TGT','LOW','COST','WMT',
                 'PG','KO','PEP','PM','CL','MDLZ','MO','GIS','KMB','SYY'],
    'Industrial': ['CAT','BA','HON','UPS','RTX','GE','MMM','LMT','DE','UNP',
                   'FDX','WM','ETN','EMR','ITW'],
    'Utility': ['NEE','DUK','SO','D','AEP','SRE','EXC','XEL','PEG','ED'],
    'REIT': ['AMT','PLD','CCI','EQIX','SPG','PSA','DLR','O','WELL','AVB'],
    'Comm': ['NFLX','DIS','CMCSA','T','VZ','TMUS','CHTR','EA','TTWO','WBD'],
}

# Cross-asset ETFs
CROSS_ASSET = {
    'US_Equity': 'SPY',
    'US_Bond': 'TLT',
    'Gold': 'GLD',
    'Oil': 'USO',
    'USD_Index': 'UUP',
    'EM_Equity': 'EEM',
    'EU_Equity': 'VGK',
    'Japan': 'EWJ',
    'China': 'FXI',
    'Bitcoin': 'BITO',
    'VIX': 'VIXY',
    'TIPS': 'TIP',
    'HighYield': 'HYG',
    'RealEstate': 'VNQ',
    'Commodities': 'DBC',
}


def download(tickers, days=504):
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=days)
    data = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    returns = closes.pct_change().dropna()
    valid = returns.columns[returns.isna().sum() < len(returns) * 0.1]
    returns = returns[valid].dropna()
    return returns


def analyze_matrix(R, label, verbose=True):
    """Full spectral analysis of a returns matrix."""
    n = R.shape[1]
    if n < 5:
        return None

    corr = np.corrcoef(R.T)
    corr = np.nan_to_num(corr, nan=0.0)
    U, S, Vt = np.linalg.svd(corr, full_matrices=False)
    S_pos = S[S > 1e-10]
    fit = fit_spectrum(S_pos)

    # Energy concentration
    energy = S_pos**2
    cum = np.cumsum(energy) / energy.sum()

    result = {
        'label': label,
        'n': n,
        'fit': fit,
        'mode1_energy': float(cum[0]) if len(cum) > 0 else 0,
        'sv_ratio_12': float(S_pos[0]/S_pos[1]) if len(S_pos) > 1 else 0,
    }

    if verbose and fit:
        fl = best_fl(fit['alpha'])
        fl_str = f" → {fl[0]} = {fl[1]:.4f} ({fl[2]:.1f}%)" if fl else ""
        print(f"  {label:<30} n={n:>3}  α={fit['alpha']:.4f}  R²={fit['r2']:.4f}  "
              f"mode1={cum[0]*100:.0f}%{fl_str}")

    return result


def main():
    print("=" * 80)
    print("DEEP MARKET SPECTRAL ANALYSIS")
    print("=" * 80)

    all_tickers = []
    ticker_sector = {}
    for sect, tickers in SECTORS.items():
        for t in tickers:
            all_tickers.append(t)
            ticker_sector[t] = sect

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("1. FULL MARKET (all sectors combined)")
    print(f"{'='*70}")

    returns = download(list(set(all_tickers)), days=504)
    n_stocks = returns.shape[1]
    print(f"  Downloaded {n_stocks} stocks, {len(returns)} days\n")

    R = returns.values
    result_full = analyze_matrix(R, "Full market (2yr)")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("2. SECTOR-SPECIFIC SUBMATRICES")
    print(f"  (Like C. elegans neuron-type analysis)")
    print(f"{'='*70}\n")

    sector_results = {}
    stock_names = [str(c) for c in returns.columns]

    for sect in sorted(SECTORS.keys()):
        # Find which of our sector's tickers are in the downloaded data
        idx = [i for i, name in enumerate(stock_names) if name in SECTORS[sect]]
        if len(idx) < 5:
            continue
        R_sect = R[:, idx]
        result = analyze_matrix(R_sect, f"{sect} ({len(idx)} stocks)")
        if result:
            sector_results[sect] = result

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("3. CROSS-SECTOR CORRELATION (sector-level matrix)")
    print(f"{'='*70}")

    # Build sector-average returns
    sector_returns = {}
    for sect in SECTORS:
        idx = [i for i, name in enumerate(stock_names) if name in SECTORS[sect]]
        if len(idx) >= 3:
            sector_returns[sect] = R[:, idx].mean(axis=1)

    if len(sector_returns) >= 5:
        sect_names = sorted(sector_returns.keys())
        R_sect_matrix = np.column_stack([sector_returns[s] for s in sect_names])
        print(f"\n  Sector-level matrix: {len(sect_names)} sectors × {R_sect_matrix.shape[0]} days")
        result_sect = analyze_matrix(R_sect_matrix, "Sector-level")

        # Cross-sector correlation matrix
        corr_sect = np.corrcoef(R_sect_matrix.T)
        print(f"\n  Sector correlation matrix:")
        print(f"  {'':>12}", end='')
        for s in sect_names:
            print(f"{s[:6]:>8}", end='')
        print()
        for i, s1 in enumerate(sect_names):
            print(f"  {s1:>12}", end='')
            for j, s2 in enumerate(sect_names):
                print(f"{corr_sect[i,j]:>8.3f}", end='')
            print()

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("4. CROSS-ASSET CLASSES")
    print(f"{'='*70}")

    cross_returns = download(list(CROSS_ASSET.values()), days=504)
    if cross_returns.shape[1] >= 5:
        print(f"  Downloaded {cross_returns.shape[1]} assets, {len(cross_returns)} days\n")
        result_cross = analyze_matrix(cross_returns.values, "Cross-asset (15 classes)")

        # Show which assets are most/least correlated with market mode 1
        R_cross = cross_returns.values
        corr_cross = np.corrcoef(R_cross.T)
        U_c, S_c, _ = np.linalg.svd(corr_cross, full_matrices=False)
        cross_names = [str(c) for c in cross_returns.columns]
        reverse_map = {v: k for k, v in CROSS_ASSET.items()}

        loadings = np.abs(U_c[:, 0])
        order = np.argsort(-loadings)
        print(f"\n  Mode 1 loadings (market consensus):")
        for i in order:
            name = reverse_map.get(cross_names[i], cross_names[i])
            print(f"    {name:<20}: {loadings[i]:.4f}")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("5. TIME HORIZON COMPARISON")
    print(f"{'='*70}\n")

    # Use a smaller set for longer lookbacks
    core_tickers = ['AAPL','MSFT','GOOG','AMZN','JPM','BAC','GS','XOM','CVX',
                    'JNJ','PFE','UNH','PG','KO','HD','CAT','BA','NEE','DUK',
                    'V','MA','COST','WMT','MCD','INTC','CSCO','T','VZ','DIS']

    for days, label in [(252, '1 year'), (504, '2 years'), (1260, '5 years'), (2520, '10 years')]:
        ret = download(core_tickers, days=days)
        if ret.shape[1] >= 10:
            analyze_matrix(ret.values, f"{label} ({ret.shape[1]} stocks, {len(ret)} days)")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("6. CRISIS ANALYSIS (rolling α over time)")
    print(f"{'='*70}")

    # Use 5yr data for the rolling analysis
    long_returns = download(core_tickers, days=1260)
    if long_returns.shape[1] >= 10:
        R_long = long_returns.values
        n_days = R_long.shape[0]
        window = 63  # 3 months
        stride = 5   # weekly

        alphas = []
        volatilities = []
        dates = []

        for start in range(0, n_days - window, stride):
            R_w = R_long[start:start+window, :]
            var = np.var(R_w, axis=0)
            valid = var > 1e-12
            if valid.sum() < 10:
                continue
            corr_w = np.corrcoef(R_w[:, valid].T)
            corr_w = np.nan_to_num(corr_w, nan=0.0)
            S_w = np.linalg.svd(corr_w, compute_uv=False)
            fit_w = fit_spectrum(S_w[S_w > 1e-10])

            if fit_w and fit_w['r2'] > 0.9:
                alphas.append(fit_w['alpha'])
                volatilities.append(np.mean(np.std(R_w[:, valid], axis=0)))
                # Approximate date index
                dates.append(start)

        if alphas:
            a = np.array(alphas)
            v = np.array(volatilities)

            print(f"\n  {len(alphas)} rolling windows (3-month, weekly stride)")
            print(f"  α range: [{a.min():.3f}, {a.max():.3f}]")
            print(f"  Mean α: {a.mean():.4f} ± {a.std():.4f}")

            fl = best_fl(a.mean())
            if fl:
                print(f"  Best F/L: {fl[0]} → {fl[1]:.4f} ({fl[2]:.1f}%)")

            # Correlation between α and volatility
            corr_av = np.corrcoef(a, v)[0, 1]
            print(f"\n  α-volatility correlation: {corr_av:.4f}")
            if corr_av > 0.3:
                print(f"  → Higher α during high-vol (crisis concentrates spectrum)")
            elif corr_av < -0.3:
                print(f"  → Lower α during high-vol (crisis disperses spectrum)")
            else:
                print(f"  → Weak relationship")

            # Quartile analysis
            q25 = np.percentile(a, 25)
            q75 = np.percentile(a, 75)
            calm = a[a <= q25]
            crisis = a[a >= q75]

            print(f"\n  Calm periods (α ≤ {q25:.3f}, n={len(calm)}):")
            fl_calm = best_fl(np.mean(calm))
            if fl_calm:
                print(f"    Mean α = {np.mean(calm):.4f} → {fl_calm[0]} ({fl_calm[2]:.1f}%)")

            print(f"  Volatile periods (α ≥ {q75:.3f}, n={len(crisis)}):")
            fl_crisis = best_fl(np.mean(crisis))
            if fl_crisis:
                print(f"    Mean α = {np.mean(crisis):.4f} → {fl_crisis[0]} ({fl_crisis[2]:.1f}%)")

    # ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("7. THE FINANCIAL attn_o: WHICH SECTOR IS THE CONSENSUS?")
    print(f"{'='*70}")

    if sector_results:
        # The sector with spectral exponent closest to attn_o's 1/3 fraction
        attn_o_alpha = INV_PHI ** (1/3)  # 0.8518 in transformer regime
        attn_o_phi = PHI ** (1/3)         # 1.1740 in biological regime

        print(f"\n  attn_o equivalence test:")
        print(f"  Transformer attn_o: α = {attn_o_alpha:.4f} [(1/φ)^(1/3)]")
        print(f"  Biological attn_o:  α = {attn_o_phi:.4f} [φ^(1/3)]")
        print()

        for sect, result in sorted(sector_results.items(),
                                    key=lambda x: abs(x[1]['fit']['alpha'] - attn_o_phi) if x[1]['fit'] else 99):
            if result['fit']:
                a = result['fit']['alpha']
                err_bio = abs(a - attn_o_phi) / attn_o_phi * 100
                err_tf = abs(a - attn_o_alpha) / attn_o_alpha * 100
                marker = ' ◄ CONSENSUS?' if err_bio < 10 else ''
                print(f"    {sect:<15} α = {a:.4f}  vs φ^(1/3): {err_bio:.1f}%  vs (1/φ)^(1/3): {err_tf:.1f}%{marker}")

    # Save
    output = {
        'full_market': result_full,
        'sectors': sector_results,
        'timestamp': datetime.now().isoformat(),
    }
    with open('runs/market-deep-spectral.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to runs/market-deep-spectral.json")


if __name__ == '__main__':
    main()
