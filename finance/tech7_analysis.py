"""
Magnificent 7 Signal Discovery & Trading Model.

Stocks: AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA

Analysis:
  1. Pull 5yr daily data for all 7
  2. Spectral analysis on the 7-stock correlation matrix
  3. Rotation signals: which stocks to overweight when
  4. Individual momentum, mean-reversion, relative strength
  5. Cross-stock spread signals
  6. Full signal discovery + walk-forward simulation

Usage:
    python finance/tech7_analysis.py
"""

import json
import sys
from math import sqrt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
PHI = (1 + sqrt(5)) / 2


def bent_power_law(k, A, k0, alpha):
    return A * (k + k0) ** (-alpha)


def fit_alpha(S):
    S = S[S > 1e-10]
    n = len(S)
    if n < 3:
        return None, None
    k = np.arange(1, n + 1, dtype=np.float64)
    try:
        popt, _ = curve_fit(bent_power_law, k, S.astype(np.float64),
            p0=[S[0], max(1.0, n * 0.1), 1.3],
            bounds=([0, 0, 0.01], [S[0] * 100, n * 5, 5.0]), maxfev=5000)
        pred = bent_power_law(k, *popt)
        ss_res = np.sum((S - pred) ** 2)
        ss_tot = np.sum((S - S.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return float(popt[2]), float(r2)
    except Exception:
        return None, None


def download_data():
    """Download 5 years of daily data for Mag 7."""
    print("  Downloading Magnificent 7 data...")
    data = yf.download(TECH7, period='5y', interval='1d', auto_adjust=True)
    closes = data['Close'].dropna()
    volumes = data['Volume'].dropna() if 'Volume' in data.columns.get_level_values(0) else None
    print(f"  Got {len(closes)} trading days, {closes.shape[1]} stocks")
    print(f"  Range: {closes.index[0].date()} to {closes.index[-1].date()}")

    # Save
    closes.to_parquet(DATA_DIR / 'tech7_closes.parquet')
    if volumes is not None:
        volumes.to_parquet(DATA_DIR / 'tech7_volumes.parquet')

    return closes, volumes


def compute_features(closes, volumes=None, window=20):
    """Build feature matrix for all 7 stocks."""
    returns = closes.pct_change()
    n_days = len(closes)

    all_features = pd.DataFrame(index=closes.index)

    # --- Per-stock features ---
    for sym in TECH7:
        if sym not in closes.columns:
            continue
        price = closes[sym]
        ret = returns[sym]

        # Momentum (multiple windows)
        for w in [5, 10, 20, 50]:
            all_features[f'{sym}_mom_{w}'] = price.pct_change(w) * 100

        # Volatility
        for w in [10, 20]:
            all_features[f'{sym}_vol_{w}'] = ret.rolling(w).std() * 100

        # RSI-like
        all_features[f'{sym}_up_frac_14'] = (ret > 0).astype(float).rolling(14).mean()

        # Distance from highs/lows
        all_features[f'{sym}_dist_high_20'] = (price - price.rolling(20).max()) / price.rolling(20).max() * 100
        all_features[f'{sym}_dist_low_20'] = (price - price.rolling(20).min()) / price.rolling(20).min() * 100

        # Mean reversion: z-score of price relative to 20d MA
        ma20 = price.rolling(20).mean()
        std20 = price.rolling(20).std()
        all_features[f'{sym}_zscore_20'] = (price - ma20) / (std20 + 1e-8)

    # --- Relative strength (pairs) ---
    for i, s1 in enumerate(TECH7):
        for s2 in TECH7[i + 1:]:
            if s1 in closes.columns and s2 in closes.columns:
                ratio = closes[s1] / closes[s2]
                all_features[f'ratio_{s1}_{s2}_5d'] = ratio.pct_change(5) * 100

    # --- Spectral features (rolling SVD on 7-stock correlation) ---
    print("  Computing spectral features on Tech 7 correlation matrix...")
    from collections import deque
    buffer = deque(maxlen=window)
    alpha_history = deque(maxlen=10)

    spectral_cols = {
        'tech7_alpha': [], 'tech7_r2': [],
        'tech7_mode1_pct': [], 'tech7_effective_rank': [],
        'tech7_delta_alpha': [],
    }

    for t in range(n_days):
        row = returns.iloc[t].values
        if np.any(np.isnan(row)):
            for k in spectral_cols:
                spectral_cols[k].append(np.nan)
            continue

        buffer.append(row)
        if len(buffer) < max(10, window // 2):
            for k in spectral_cols:
                spectral_cols[k].append(np.nan)
            continue

        R_window = np.array(list(buffer))
        var = np.var(R_window, axis=0)
        valid = var > 1e-15
        if valid.sum() < 4:
            for k in spectral_cols:
                spectral_cols[k].append(np.nan)
            continue

        corr = np.corrcoef(R_window[:, valid].T)
        corr = np.nan_to_num(corr, nan=0.0)
        S = np.linalg.svd(corr, compute_uv=False)

        alpha, r2 = fit_alpha(S[S > 1e-10])
        if alpha is None:
            for k in spectral_cols:
                spectral_cols[k].append(np.nan)
            continue

        energy = S ** 2
        total_energy = energy.sum()
        mode1_pct = energy[0] / total_energy * 100 if total_energy > 0 else 0

        p = energy / total_energy
        p = p[p > 1e-15]
        effective_rank = np.exp(-np.sum(p * np.log(p)))

        alpha_history.append(alpha)
        delta_alpha = alpha - alpha_history[0] if len(alpha_history) >= 3 else 0

        spectral_cols['tech7_alpha'].append(alpha)
        spectral_cols['tech7_r2'].append(r2)
        spectral_cols['tech7_mode1_pct'].append(mode1_pct)
        spectral_cols['tech7_effective_rank'].append(effective_rank)
        spectral_cols['tech7_delta_alpha'].append(delta_alpha)

    for k, v in spectral_cols.items():
        all_features[k] = v

    # --- Equal-weight portfolio return (benchmark) ---
    eq_ret = returns[TECH7].mean(axis=1)
    all_features['eq_weight_return'] = eq_ret * 100

    # --- Dispersion: std of individual returns ---
    all_features['tech7_dispersion'] = returns[TECH7].std(axis=1) * 100

    # --- Average momentum across all 7 ---
    for w in [5, 10, 20]:
        mom_cols = [f'{sym}_mom_{w}' for sym in TECH7 if f'{sym}_mom_{w}' in all_features.columns]
        all_features[f'avg_mom_{w}'] = all_features[mom_cols].mean(axis=1)

    # Drop rows with NaN in key columns
    key_cols = ['tech7_alpha']
    all_features = all_features.dropna(subset=key_cols)

    print(f"  Features: {all_features.shape}")

    return all_features, returns


def signal_discovery(features, returns, closes, train_end):
    """Find which features predict next-day returns for each stock."""
    print("\n" + "=" * 70)
    print("SIGNAL DISCOVERY — TECH 7")
    print("=" * 70)

    feat_train = features.iloc[:train_end]
    ret_train = returns.iloc[:train_end]

    for sym in TECH7:
        next_ret = ret_train[sym].shift(-1).values
        valid = ~np.isnan(next_ret)

        print(f"\n  [{sym}] — Top predictors of next-day return:")

        correlations = []
        for col in feat_train.columns:
            vals = feat_train[col].values
            mask = valid & ~np.isnan(vals)
            if mask.sum() < 50:
                continue
            c = np.corrcoef(vals[mask], next_ret[mask])[0, 1]
            if np.isnan(c):
                continue
            correlations.append({'feature': col, 'corr': float(c)})

        correlations.sort(key=lambda x: abs(x['corr']), reverse=True)

        for rank, c in enumerate(correlations[:10]):
            print(f"    {rank + 1:>2}. {c['feature']:<40} corr={c['corr']:>+.4f}")

    # Rotation signal: which stock outperforms when alpha is high/low?
    print(f"\n  ROTATION SIGNAL — returns by alpha regime:")
    alpha = feat_train['tech7_alpha'].values
    p33 = np.percentile(alpha[~np.isnan(alpha)], 33)
    p66 = np.percentile(alpha[~np.isnan(alpha)], 66)

    print(f"  Alpha thresholds: p33={p33:.3f}  p66={p66:.3f}")
    print(f"  {'Stock':<8} {'Low alpha':>12} {'Mid alpha':>12} {'High alpha':>12} {'Spread':>10}")
    print(f"  {'-' * 58}")

    spreads = {}
    for sym in TECH7:
        next_ret = ret_train[sym].shift(-1)
        low = next_ret[alpha <= p33].mean() * 100
        mid = next_ret[(alpha > p33) & (alpha <= p66)].mean() * 100
        high = next_ret[alpha > p66].mean() * 100
        spread = low - high  # positive = stock does better when alpha is low
        spreads[sym] = spread
        print(f"  {sym:<8} {low:>+11.4f}% {mid:>+11.4f}% {high:>+11.4f}% {spread:>+9.4f}%")

    # Best rotation: overweight stocks that do well in current regime
    print(f"\n  INTERPRETATION:")
    for sym in sorted(spreads, key=lambda s: spreads[s], reverse=True):
        s = spreads[sym]
        if s > 0.01:
            print(f"    {sym}: outperforms when alpha LOW (dip conditions) -> BUY ON DIPS")
        elif s < -0.01:
            print(f"    {sym}: outperforms when alpha HIGH (crowded) -> MOMENTUM STOCK")
        else:
            print(f"    {sym}: regime-neutral")


def walk_forward_simulation(features, returns, closes, train_end):
    """Walk-forward: rotate between Mag 7 based on signals."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD SIMULATION — TECH 7 ROTATION")
    print("=" * 70)

    n_days = len(features)
    cal_days = 252
    recal_days = 63
    cost_frac = 5.0 / 10000  # 5 bps for individual stocks

    # Equal weight baseline
    eq_returns = returns[TECH7].iloc[train_end:].mean(axis=1)
    eq_cum = (1 + eq_returns).cumprod()
    eq_total = (eq_cum.iloc[-1] - 1) * 100

    # Strategy: for each stock, compute a "buy" score based on:
    # 1. Mean-reversion: oversold stocks (low z-score, negative momentum) -> buy more
    # 2. Alpha regime: in low-alpha (dip), overweight dip-performers
    # 3. Relative strength: rotate toward recent relative outperformers (momentum)
    capital = 100000.0
    weights = np.ones(7) / 7  # start equal weight
    daily_log = []
    equity = [capital]

    feature_idx = features.index
    last_recal = -999

    # Precompute training stats for z-scoring
    train_means = {}
    train_stds = {}

    for t_idx in range(train_end, n_days):
        t_date = feature_idx[t_idx]

        # Recalibrate
        if t_idx - last_recal >= recal_days:
            cal_start = max(0, t_idx - cal_days)
            last_recal = t_idx

        # Score each stock
        scores = np.zeros(7)
        for i, sym in enumerate(TECH7):
            # Mean reversion: oversold = high score
            zscore_col = f'{sym}_zscore_20'
            if zscore_col in features.columns:
                z = features.iloc[t_idx][zscore_col]
                if not np.isnan(z):
                    scores[i] += -z * 0.4  # oversold = positive score

            # Short-term momentum reversal: negative 5d mom = buy
            mom5_col = f'{sym}_mom_5'
            if mom5_col in features.columns:
                m = features.iloc[t_idx][mom5_col]
                if not np.isnan(m):
                    scores[i] += -m * 0.01  # scale down (mom is in %)

            # Longer momentum: positive 20d mom = keep holding
            mom20_col = f'{sym}_mom_20'
            if mom20_col in features.columns:
                m = features.iloc[t_idx][mom20_col]
                if not np.isnan(m):
                    scores[i] += m * 0.005  # trend following, lighter

            # Volatility: lower vol = more weight (risk parity lite)
            vol_col = f'{sym}_vol_20'
            if vol_col in features.columns:
                v = features.iloc[t_idx][vol_col]
                if not np.isnan(v) and v > 0:
                    scores[i] += -v * 0.1  # less vol = higher score

        # Convert scores to weights (softmax-like)
        # Base: equal weight + score adjustment
        raw_weights = np.ones(7) / 7 + scores * 0.1
        raw_weights = np.clip(raw_weights, 0.02, 0.5)  # min 2%, max 50% per stock
        weights = raw_weights / raw_weights.sum()  # normalize to sum=1

        # Today's return
        day_returns = np.array([returns[sym].iloc[t_idx] if t_idx < len(returns) else 0
                                for sym in TECH7])
        day_returns = np.nan_to_num(day_returns, nan=0.0)

        port_return = np.sum(weights * day_returns)
        pnl = port_return * capital
        capital += pnl
        equity.append(capital)

        daily_log.append({
            'date': str(t_date.date()),
            'weights': {sym: round(float(w), 3) for sym, w in zip(TECH7, weights)},
            'port_return': float(port_return * 100),
            'capital': float(capital),
        })

    equity = np.array(equity)
    final = equity[-1]
    total_ret = (final - 100000) / 100000 * 100
    n_years = (n_days - train_end) / 252

    daily_rets = np.diff(equity) / equity[:-1]
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * sqrt(252)
    rm = np.maximum.accumulate(equity)
    max_dd = ((rm - equity) / rm * 100).max()
    annual = ((final / 100000) ** (1 / n_years) - 1) * 100

    print(f"""
  Period: {daily_log[0]['date']} to {daily_log[-1]['date']} ({n_years:.1f} years)
  Starting Capital: $100,000
  Final Capital:    ${final:,.0f}

  ROTATION STRATEGY:
    Total Return:   {total_ret:+.2f}%
    Annual Return:  {annual:+.2f}%
    Sharpe Ratio:   {sharpe:.2f}
    Max Drawdown:   {max_dd:.2f}%

  EQUAL WEIGHT BASELINE:
    Total Return:   {eq_total:+.2f}%

  EDGE vs Equal Weight: {total_ret - eq_total:+.2f}%
""")

    # Per-stock contribution
    print("  Per-stock average weight and contribution:")
    avg_weights = defaultdict(list)
    for d in daily_log:
        for sym, w in d['weights'].items():
            avg_weights[sym].append(w)

    for sym in TECH7:
        avg_w = np.mean(avg_weights[sym])
        # Individual stock return over period
        sym_ret = (closes[sym].iloc[-1] - closes[sym].iloc[train_end]) / closes[sym].iloc[train_end] * 100
        print(f"    {sym:<6} avg weight: {avg_w:.1%}  stock return: {sym_ret:+.1f}%")

    # Yearly breakdown
    print(f"\n  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Strategy':>9} {'EqWeight':>9} {'Edge':>8}")
    print(f"  {'-' * 35}")

    years = sorted(set(d['date'][:4] for d in daily_log))
    for year in years:
        yr_days = [d for d in daily_log if d['date'][:4] == year]
        if len(yr_days) < 10:
            continue
        yr_start = yr_days[0]['capital'] - yr_days[0]['port_return'] / 100 * (yr_days[0]['capital'] - yr_days[0]['port_return'] / 100 * yr_days[0]['capital'])
        yr_start = yr_days[0]['capital'] / (1 + yr_days[0]['port_return'] / 100)
        yr_end = yr_days[-1]['capital']
        yr_ret = (yr_end - yr_start) / yr_start * 100

        # EqWeight for this year
        yr_eq = returns[TECH7]
        yr_mask = [d['date'][:4] == year for d in daily_log]
        yr_eq_rets = eq_returns.iloc[[i for i, m in enumerate(yr_mask) if m]]
        yr_eq_ret = ((1 + yr_eq_rets).prod() - 1) * 100

        print(f"  {year:<6} {yr_ret:>+8.2f}% {yr_eq_ret:>+8.2f}% {yr_ret - yr_eq_ret:>+7.2f}%")

    # Save
    save_path = RESULTS_DIR / 'tech7_results.json'
    with open(save_path, 'w') as f:
        json.dump({
            'strategy_return': round(total_ret, 2),
            'eq_weight_return': round(float(eq_total), 2),
            'sharpe': round(float(sharpe), 2),
            'max_drawdown': round(float(max_dd), 2),
        }, f, indent=2)
    print(f"  Saved: {save_path}")


def main():
    print("=" * 70)
    print("MAGNIFICENT 7 — SPECTRAL TRADING ANALYSIS")
    print(f"  Stocks: {', '.join(TECH7)}")
    print("=" * 70)

    # Download or load data
    closes_path = DATA_DIR / 'tech7_closes.parquet'
    if closes_path.exists():
        print("\n  Loading cached data...")
        closes = pd.read_parquet(closes_path)
        volumes = None
        vol_path = DATA_DIR / 'tech7_volumes.parquet'
        if vol_path.exists():
            volumes = pd.read_parquet(vol_path)
        print(f"  {len(closes)} days, {closes.shape[1]} stocks")
    else:
        closes, volumes = download_data()

    # Compute features
    print("\n  Computing features...")
    features, returns = compute_features(closes, volumes)

    # Train/test split
    train_end = len(features)
    for i, ts in enumerate(features.index):
        if str(ts.year) == '2025':
            train_end = i
            break

    print(f"  Train: {train_end} days  Test: {len(features) - train_end} days")

    # Signal discovery
    signal_discovery(features, returns, closes, train_end)

    # Walk-forward simulation
    walk_forward_simulation(features, returns, closes, train_end)


if __name__ == '__main__':
    main()
