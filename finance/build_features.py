"""
Phase 0.5: Feature Engineering Pipeline

Loads raw data from the data lake, computes spectral features (α, mode1%, etc.),
and merges all data sources into a single [T, features] tensor for GPU training.

Produces:
  finance/data/features_daily.pt   — daily features + labels (5 years)
  finance/data/features_hourly.pt  — hourly features + labels (2 years)
  finance/data/features_5min.pt    — 5-min features + labels (60 days)
  finance/data/feature_names.json  — column name mapping

Usage:
    python finance/build_features.py
    python finance/build_features.py --timescale daily
    python finance/build_features.py --timescale hourly
    python finance/build_features.py --timescale 5min
"""

import argparse
import json
import sys
from math import sqrt
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent / 'data'

PHI = (1 + sqrt(5)) / 2

# Core ETFs used for spectral analysis
SPECTRAL_SYMBOLS = [
    'SPY', 'QQQ', 'IWM', 'DIA',
    'XLK', 'XLF', 'XLE', 'XLV',
    'XLI', 'XLY', 'XLP', 'XLU',
    'TLT', 'HYG', 'GLD', 'SLV', 'USO',
]

REGIMES = {
    'DEEP_CALM':  (0.0,            PHI**(1/7)),
    'CALM':       (PHI**(1/7),     PHI**(1/3)),
    'NORMAL':     (PHI**(1/3),     PHI**(4/7)),
    'ELEVATED':   (PHI**(4/7),     PHI**(2/3)),
    'STRESS':     (PHI**(2/3),     PHI**(1/1)),
    'CRISIS':     (PHI**(1/1),     float('inf')),
}


def bent_power_law(k, A, k0, alpha):
    return A * (k + k0) ** (-alpha)


def fit_alpha(S):
    S = S[S > 1e-10]
    n = len(S)
    if n < 4:
        return None, None
    k = np.arange(1, n + 1, dtype=np.float64)
    try:
        popt, _ = curve_fit(bent_power_law, k, S.astype(np.float64),
            p0=[S[0], max(1.0, n*0.1), 1.3],
            bounds=([0, 0, 0.01], [S[0]*100, n*5, 3.0]), maxfev=5000)
        pred = bent_power_law(k, *popt)
        ss_res = np.sum((S[:n] - pred)**2)
        ss_tot = np.sum((S[:n] - np.mean(S[:n]))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        return float(popt[2]), float(r2)
    except Exception:
        return None, None


def classify_regime(alpha):
    for name, (lo, hi) in REGIMES.items():
        if lo <= alpha < hi:
            return name
    return 'CRISIS'


def regime_to_int(regime):
    mapping = {'DEEP_CALM': 0, 'CALM': 1, 'NORMAL': 2,
               'ELEVATED': 3, 'STRESS': 4, 'CRISIS': 5}
    return mapping.get(regime, 5)


def compute_spectral_features(closes, window=30):
    """Compute rolling spectral α and related features from close prices.

    Args:
        closes: DataFrame of close prices [T, n_symbols]
        window: rolling window for correlation matrix

    Returns:
        DataFrame with spectral features, indexed same as input
    """
    # Get returns
    returns = closes.pct_change()
    n_total = len(closes)

    # Available spectral symbols
    available = [s for s in SPECTRAL_SYMBOLS if s in closes.columns]
    if len(available) < 5:
        print(f"  WARNING: only {len(available)} spectral symbols available")
        return pd.DataFrame(index=closes.index)

    R = returns[available].values
    results = []

    buffer = deque(maxlen=window)
    alpha_history = deque(maxlen=10)

    for t in range(n_total):
        if t == 0 or np.any(np.isnan(R[t])):
            results.append(None)
            continue

        buffer.append(R[t])

        if len(buffer) < max(10, window // 2):
            results.append(None)
            continue

        R_window = np.array(list(buffer))
        var = np.var(R_window, axis=0)
        valid = var > 1e-15
        if valid.sum() < 5:
            results.append(None)
            continue

        corr = np.corrcoef(R_window[:, valid].T)
        corr = np.nan_to_num(corr, nan=0.0)
        S = np.linalg.svd(corr, compute_uv=False)

        alpha, r2 = fit_alpha(S[S > 1e-10])
        if alpha is None:
            results.append(None)
            continue

        energy = S ** 2
        total_energy = energy.sum()
        mode1_pct = energy[0] / total_energy * 100 if total_energy > 0 else 0
        mode2_pct = energy[1] / total_energy * 100 if total_energy > 0 and len(energy) > 1 else 0
        mode3_pct = energy[2] / total_energy * 100 if total_energy > 0 and len(energy) > 2 else 0

        # Effective rank (participation ratio)
        p = energy / total_energy
        p = p[p > 1e-15]
        entropy = -np.sum(p * np.log(p))
        effective_rank = np.exp(entropy)

        # Energy concentration at phi-power thresholds
        n_modes = len(energy)
        cum_energy = np.cumsum(energy) / total_energy
        k_90 = np.searchsorted(cum_energy, 0.90) + 1
        k_95 = np.searchsorted(cum_energy, 0.95) + 1
        k_99 = np.searchsorted(cum_energy, 0.99) + 1
        frac_90 = k_90 / n_modes
        frac_95 = k_95 / n_modes
        frac_99 = k_99 / n_modes

        regime = classify_regime(alpha)

        # Alpha momentum
        alpha_history.append(alpha)
        delta_alpha = alpha - alpha_history[0] if len(alpha_history) >= 3 else 0.0
        alpha_accel = 0.0
        if len(alpha_history) >= 5:
            mid = len(alpha_history) // 2
            d1 = alpha_history[-1] - alpha_history[mid]
            d2 = alpha_history[mid] - alpha_history[0]
            alpha_accel = d1 - d2

        results.append({
            'alpha': alpha,
            'r2': r2,
            'mode1_pct': mode1_pct,
            'mode2_pct': mode2_pct,
            'mode3_pct': mode3_pct,
            'effective_rank': effective_rank,
            'frac_90': frac_90,
            'frac_95': frac_95,
            'frac_99': frac_99,
            'regime_int': regime_to_int(regime),
            'delta_alpha': delta_alpha,
            'alpha_accel': alpha_accel,
            'n_valid': int(valid.sum()),
        })

    # Convert to DataFrame
    records = []
    for t, r in enumerate(results):
        if r is not None:
            r['_idx'] = t
            records.append(r)

    if not records:
        return pd.DataFrame(index=closes.index)

    spec_df = pd.DataFrame(records).set_index('_idx')
    spec_df.index = closes.index[spec_df.index]
    return spec_df


def compute_price_features(market_df, spy_col='SPY'):
    """Compute price-based features from market data."""
    # Get close prices
    if isinstance(market_df.columns, pd.MultiIndex):
        closes = market_df['Close']
    else:
        closes = market_df

    # Drop rows where SPY is NaN (weekends/holidays) to fix rolling window gaps
    if spy_col in closes.columns:
        valid_mask = closes[spy_col].notna()
        closes = closes[valid_mask].copy()
        if isinstance(market_df.columns, pd.MultiIndex):
            market_df = market_df[valid_mask]

    features = pd.DataFrame(index=closes.index)

    if spy_col in closes.columns:
        spy = closes[spy_col]

        # Multi-window momentum
        for w in [5, 10, 20, 50]:
            features[f'momentum_{w}'] = spy.pct_change(w) * 100

        # Volatility (rolling std of returns)
        spy_ret = spy.pct_change()
        for w in [10, 20, 50]:
            features[f'volatility_{w}'] = spy_ret.rolling(w).std() * 100

        # RSI-like: fraction of up bars in window
        for w in [14, 28]:
            up = (spy_ret > 0).astype(float)
            features[f'up_frac_{w}'] = up.rolling(w).mean()

        # Distance from rolling high/low
        for w in [20, 50]:
            rh = spy.rolling(w).max()
            rl = spy.rolling(w).min()
            features[f'dist_from_high_{w}'] = (spy - rh) / rh * 100
            features[f'dist_from_low_{w}'] = (spy - rl) / rl * 100

        # Volume features (if available)
        if isinstance(market_df.columns, pd.MultiIndex) and 'Volume' in market_df.columns.get_level_values(0):
            vol = market_df['Volume']
            if spy_col in vol.columns:
                spy_vol = vol[spy_col]
                features['volume_ratio_20'] = spy_vol / spy_vol.rolling(20).mean()

    # Cross-asset features
    pairs = [('SPY', 'TLT'), ('SPY', 'GLD'), ('XLK', 'XLF'), ('HYG', 'TLT')]
    for a, b in pairs:
        if a in closes.columns and b in closes.columns:
            ratio = closes[a] / closes[b]
            features[f'ratio_{a}_{b}'] = ratio.pct_change(5) * 100

    # Sector dispersion (std of sector returns)
    sectors = [s for s in ['XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU']
               if s in closes.columns]
    if len(sectors) >= 4:
        sector_rets = closes[sectors].pct_change()
        features['sector_dispersion'] = sector_rets.std(axis=1) * 100
        features['sector_dispersion_20'] = features['sector_dispersion'].rolling(20).mean()

    return features


def load_and_align_daily(market_df):
    """Load all daily data sources and align timestamps."""

    if isinstance(market_df.columns, pd.MultiIndex):
        closes = market_df['Close']
    else:
        closes = market_df

    # 1. Spectral features
    print("  Computing spectral features (this takes a minute)...")
    spectral = compute_spectral_features(closes, window=30)
    print(f"    Spectral: {spectral.shape}")

    # 2. Price features
    print("  Computing price features...")
    price = compute_price_features(market_df)
    print(f"    Price: {price.shape}")

    # 3. Astro features
    astro_path = DATA_DIR / 'astro_features.parquet'
    if astro_path.exists():
        astro = pd.read_parquet(astro_path)
        astro.index = pd.to_datetime(astro.index)
        print(f"    Astro: {astro.shape}")
    else:
        astro = pd.DataFrame()
        print("    Astro: not found")

    # 4. Weather
    weather_path = DATA_DIR / 'weather_nyc.parquet'
    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        weather.index = pd.to_datetime(weather.index)
        # Drop synthetic flag if present
        if 'temp_synthetic' in weather.columns:
            weather = weather.drop(columns=['temp_synthetic'])
        print(f"    Weather: {weather.shape}")
    else:
        weather = pd.DataFrame()
        print("    Weather: not found")

    # 5. Macro
    macro = pd.DataFrame()
    for name in ['macro_fred.parquet', 'macro_yfinance.parquet']:
        p = DATA_DIR / name
        if p.exists():
            m = pd.read_parquet(p)
            m.index = pd.to_datetime(m.index)
            if isinstance(m.columns, pd.MultiIndex):
                # Flatten yfinance multi-index
                m = m['Close'] if 'Close' in m.columns.get_level_values(0) else m
            macro = m
            print(f"    Macro ({name}): {macro.shape}")
            break

    # Align everything to spectral index (trading days only)
    valid_idx = spectral.dropna(subset=['alpha']).index

    # Normalize all indices to tz-naive dates for merging
    def to_date_index(idx):
        """Convert any datetime index to tz-naive date."""
        if hasattr(idx, 'tz') and idx.tz is not None:
            return idx.tz_localize(None).normalize()
        return idx.normalize()

    valid_dates = to_date_index(valid_idx)

    # For non-trading-day data (astro, weather), align to nearest trading day
    all_features = spectral.loc[valid_idx].copy()

    # Merge price features
    if not price.empty:
        all_features = all_features.join(price, how='left')

    # Helper: align daily data to trading days
    def align_daily(daily_df, prefix, target_idx, target_dates):
        """Align a daily DataFrame to trading day index via date matching."""
        daily_dates = to_date_index(daily_df.index)
        daily_df = daily_df.copy()
        daily_df.index = daily_dates
        # Forward-fill to cover weekends/holidays
        daily_reindexed = daily_df.reindex(target_dates, method='ffill')
        daily_reindexed.index = target_idx
        result = {}
        for col in daily_df.columns:
            result[f'{prefix}{col}'] = daily_reindexed[col].values
        return result

    # Merge astro
    if not astro.empty:
        for k, v in align_daily(astro, 'astro_', valid_idx, valid_dates).items():
            all_features[k] = v

    # Merge weather
    if not weather.empty:
        for k, v in align_daily(weather, 'weather_', valid_idx, valid_dates).items():
            all_features[k] = v

    # Merge macro
    if not macro.empty:
        for k, v in align_daily(macro, 'macro_', valid_idx, valid_dates).items():
            all_features[k] = v

    # --- Rule-based signals (from grid search: α>1.38, momentum_reversal, trailing_stop) ---
    if 'alpha' in all_features.columns:
        alpha = all_features['alpha']

        # 1. α threshold signal: validated from grid search (+2.83% edge)
        #    α > 1.38 = calm/ordered market → go long
        #    α < 1.0  = stressed/chaotic → go short or flat
        all_features['rule_long_signal'] = (alpha > 1.38).astype(float)
        all_features['rule_short_signal'] = (alpha < 1.0).astype(float)
        all_features['rule_base_position'] = 0.0
        all_features.loc[alpha > 1.38, 'rule_base_position'] = 1.0
        all_features.loc[alpha < 1.0, 'rule_base_position'] = -1.0

        # 2. Momentum reversal signal: when α is falling, price tends to reverse
        if 'delta_alpha' in all_features.columns:
            da = all_features['delta_alpha']
            all_features['rule_momentum_reversal'] = 0.0
            # α falling + price was up → expect reversal down
            if 'momentum_5' in all_features.columns:
                mom = all_features['momentum_5']
                all_features['rule_momentum_reversal'] = np.where(
                    (da < -0.05) & (mom > 0), -1.0,
                    np.where((da > 0.05) & (mom < 0), 1.0, 0.0)
                )

        # 3. Trailing stop proxy: distance from recent high as stop signal
        #    Grid search found 0.10% trailing stop optimal
        if 'SPY' in closes.columns:
            spy_reindexed = closes['SPY'].reindex(valid_idx)
            rolling_high = spy_reindexed.rolling(5, min_periods=1).max()
            drawdown_from_high = (spy_reindexed - rolling_high) / rolling_high * 100
            all_features['rule_trailing_stop'] = (drawdown_from_high > -0.10).astype(float)
            # 0 = stopped out (price dropped >0.10% from 5-bar high), 1 = above stop

        # 4. Combined rule signal: what the known-good strategy says to do
        base_pos = all_features['rule_base_position'].copy()
        if 'rule_trailing_stop' in all_features.columns:
            base_pos = base_pos * all_features['rule_trailing_stop']  # zero out when stopped
        if 'rule_momentum_reversal' in all_features.columns:
            # Blend: 70% α-regime, 30% momentum reversal
            reversal = all_features['rule_momentum_reversal']
            all_features['rule_combined'] = 0.7 * base_pos + 0.3 * reversal
        else:
            all_features['rule_combined'] = base_pos

        # 5. Regime one-hot encoding (more useful than single int for NN)
        regime_int = all_features['regime_int']
        for i, name in enumerate(['DEEP_CALM', 'CALM', 'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']):
            all_features[f'regime_{name}'] = (regime_int == i).astype(float)

    # Target: next-period SPY return (what we're trying to predict/trade)
    if 'SPY' in closes.columns:
        spy = closes['SPY'].reindex(valid_idx)
        all_features['target_return_1'] = spy.pct_change().shift(-1) * 100  # next bar return
        all_features['target_return_5'] = spy.pct_change(5).shift(-5) * 100  # 5-bar return
        all_features['spy_price'] = spy.values

    return all_features


def features_to_tensor(df):
    """Convert feature DataFrame to PyTorch tensors.

    Returns dict with:
        'features': [T, D] float32 tensor
        'targets':  [T, 2] float32 tensor (1-bar and 5-bar returns)
        'spy':      [T] float32 tensor (SPY price for PnL simulation)
        'timestamps': list of ISO timestamp strings
        'feature_names': list of feature column names
    """
    # Separate targets
    target_cols = [c for c in df.columns if c.startswith('target_')]
    spy_col = 'spy_price' if 'spy_price' in df.columns else None
    feature_cols = [c for c in df.columns if c not in target_cols and c != 'spy_price']

    # Drop rows with NaN in key features
    key_cols = ['alpha', 'r2', 'mode1_pct']
    key_cols = [c for c in key_cols if c in df.columns]
    valid = df.dropna(subset=key_cols)

    # Fill remaining NaN with 0 (for sparse features like macro)
    features = valid[feature_cols].fillna(0).values.astype(np.float32)
    targets = valid[target_cols].fillna(0).values.astype(np.float32) if target_cols else np.zeros((len(valid), 1), dtype=np.float32)
    spy = valid[spy_col].values.astype(np.float32) if spy_col else np.zeros(len(valid), dtype=np.float32)

    timestamps = [str(t) for t in valid.index]

    result = {
        'features': torch.from_numpy(features),
        'targets': torch.from_numpy(targets),
        'spy': torch.from_numpy(spy),
        'timestamps': timestamps,
        'feature_names': feature_cols,
    }

    return result


def build_timescale(timescale):
    """Build features for a specific timescale."""
    file_map = {
        'daily': 'market_daily.parquet',
        'hourly': 'market_hourly.parquet',
        '5min': 'market_5min.parquet',
    }
    window_map = {
        'daily': 30,
        'hourly': 30,
        '5min': 30,
    }

    fname = file_map[timescale]
    path = DATA_DIR / fname
    if not path.exists():
        print(f"  ERROR: {path} not found. Run acquire_data.py first.")
        return

    print(f"\nLoading {timescale} data from {fname}...")
    market = pd.read_parquet(path)
    print(f"  Raw shape: {market.shape}")

    # For intraday data, filter to rows where core ETFs have data
    # (removes off-hours rows from 24/7 crypto/FX)
    if timescale in ('hourly', '5min') and isinstance(market.columns, pd.MultiIndex):
        closes = market['Close']
        if 'SPY' in closes.columns:
            valid_mask = closes['SPY'].notna()
            market = market[valid_mask]
            print(f"  After market-hours filter: {market.shape}")

    # Build features
    features_df = load_and_align_daily(market)
    print(f"\n  Combined features: {features_df.shape}")

    # Inspect
    print(f"\n  Feature columns ({len(features_df.columns)}):")
    for i, col in enumerate(features_df.columns):
        non_null = features_df[col].notna().sum()
        print(f"    {i:>3}. {col:<35} {non_null:>6} non-null")
        if i > 60:
            print(f"    ... ({len(features_df.columns) - i - 1} more)")
            break

    # Convert to tensors
    tensor_data = features_to_tensor(features_df)

    out_path = DATA_DIR / f'features_{timescale}.pt'
    torch.save(tensor_data, out_path)
    print(f"\n  Saved: {out_path}")
    print(f"    Features: {tensor_data['features'].shape}")
    print(f"    Targets:  {tensor_data['targets'].shape}")
    print(f"    SPY:      {tensor_data['spy'].shape}")
    print(f"    Timespan: {tensor_data['timestamps'][0]} → {tensor_data['timestamps'][-1]}")

    # Save feature names separately for reference
    names_path = DATA_DIR / f'feature_names_{timescale}.json'
    with open(names_path, 'w') as f:
        json.dump(tensor_data['feature_names'], f, indent=2)

    return tensor_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timescale', choices=['daily', 'hourly', '5min', 'all'],
                        default='all')
    args = parser.parse_args()

    print("=" * 60)
    print("SPECTRAL ALPHA — Feature Engineering Pipeline")
    print("=" * 60)

    timescales = ['daily', 'hourly', '5min'] if args.timescale == 'all' else [args.timescale]

    for ts in timescales:
        build_timescale(ts)

    # Final inventory
    print("\n" + "=" * 60)
    print("FEATURE STORE INVENTORY")
    print("=" * 60)
    for f in sorted(DATA_DIR.glob('features_*.pt')):
        data = torch.load(f, weights_only=False)
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:<25} {size_mb:>6.2f} MB  "
              f"features={list(data['features'].shape)}  "
              f"targets={list(data['targets'].shape)}")


if __name__ == '__main__':
    main()
