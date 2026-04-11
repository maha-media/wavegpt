"""
Multi-Signal Trading Model — stack every edge.

Combines all discovered signals:
  - SPY/TLT ratio (strongest single predictor)
  - Alpha regime (mean-reversion + risk scaler)
  - Cross-asset ratios (HYG/TLT, SPY/GLD, XLK/XLF)
  - Momentum (multi-window, mean-reverting)
  - Sector dispersion
  - VIX level
  - Calendar (OpEx week amplifies everything)
  - Venus retrograde (small but real)
  - Weather (temp range, sunshine)
  - Lagged features (SPY/TLT at t-2 is even stronger)

Architecture:
  1. Ridge regression predicts next-day return from all signals
  2. Walk-forward: retrain every quarter on trailing 1yr
  3. Prediction -> position via Kelly-like sizing
  4. Alpha regime -> risk scaler (keep what works)
  5. 3-year simulation with full accounting

Usage:
    python finance/multi_signal_model.py
    python finance/multi_signal_model.py --starting-capital 100000
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

# Top signals from discovery, grouped by type
SIGNAL_FEATURES = [
    # Cross-asset ratios (strongest group)
    'ratio_SPY_TLT', 'ratio_HYG_TLT', 'ratio_SPY_GLD', 'ratio_XLK_XLF',
    # Mean-reversion momentum
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
    # Overbought/oversold
    'up_frac_14', 'up_frac_28',
    'dist_from_high_20', 'dist_from_low_20',
    'dist_from_high_50', 'dist_from_low_50',
    # Spectral
    'alpha', 'delta_alpha', 'alpha_accel',
    'effective_rank', 'mode1_pct',
    # Volatility / dispersion
    'volatility_10', 'volatility_20',
    'sector_dispersion', 'sector_dispersion_20',
    'volume_ratio_20',
    # Macro
    'macro_^VIX',
    # Calendar (interactions amplifier)
    'astro_is_opex', 'astro_is_fomc_week', 'astro_is_quarter_end',
    # Seasonality
    'astro_dow_sin', 'astro_dow_cos',
    'astro_month_sin', 'astro_month_cos',
    # Astro (small but real)
    'astro_venus_retrograde', 'astro_mercury_retrograde',
    # Weather
    'weather_temp_range_c', 'weather_sunshine_hours',
    'weather_precipitation_mm',
]


def add_lagged_features(features, feature_names, lags=[1, 2]):
    """Add lagged versions of key features."""
    lag_features = ['ratio_SPY_TLT', 'ratio_HYG_TLT', 'delta_alpha']
    new_cols = []
    new_names = []

    for lag in lags:
        for fname in lag_features:
            if fname in feature_names:
                idx = feature_names.index(fname)
                col = features[:, idx]
                lagged = np.full_like(col, np.nan)
                lagged[lag:] = col[:-lag]
                new_cols.append(lagged)
                new_names.append(f'{fname}_lag{lag}')

    if new_cols:
        features = np.hstack([features, np.column_stack(new_cols)])
        feature_names = feature_names + new_names

    return features, feature_names


def add_interaction_features(features, feature_names):
    """Add key interaction terms discovered in signal analysis."""
    interactions = [
        ('astro_is_opex', 'sector_dispersion'),
        ('astro_is_opex', 'dist_from_low_50'),
        ('astro_is_opex', 'momentum_5'),
        ('ratio_SPY_TLT', 'astro_is_opex'),
    ]
    new_cols = []
    new_names = []

    for f1, f2 in interactions:
        if f1 in feature_names and f2 in feature_names:
            i1 = feature_names.index(f1)
            i2 = feature_names.index(f2)
            new_cols.append(features[:, i1] * features[:, i2])
            new_names.append(f'{f1}_x_{f2}')

    if new_cols:
        features = np.hstack([features, np.column_stack(new_cols)])
        feature_names = feature_names + new_names

    return features, feature_names


def get_alpha_risk_scaler(alpha_val, alpha_history, returns_history):
    """Risk scaler from alpha regime (keep what works)."""
    regime_pcts = [0, 10, 25, 50, 75, 90, 100]
    regime_names = ['DEEP_CALM', 'CALM', 'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']

    for i, rname in enumerate(regime_names):
        lo = np.percentile(alpha_history, regime_pcts[i])
        hi = np.percentile(alpha_history, regime_pcts[i + 1]) if i < 5 else float('inf')
        if lo <= alpha_val < hi:
            mask = (alpha_history >= lo) & (alpha_history < hi)
            if mask.sum() > 10:
                r = returns_history[mask]
                ir = r.mean() / (r.std() + 1e-8)
                return float(np.clip(0.6 + ir * 4.0, 0.2, 1.0)), rname
    return 0.5, 'UNKNOWN'


def main():
    parser = argparse.ArgumentParser(description='Multi-Signal Trading Model')
    parser.add_argument('--starting-capital', type=float, default=100000)
    parser.add_argument('--cost-bps', type=float, default=3.0)
    parser.add_argument('--cal-days', type=int, default=252)
    parser.add_argument('--recal-days', type=int, default=63)
    parser.add_argument('--ridge-alpha', type=float, default=10.0,
                        help='Ridge regularization (higher = more conservative)')
    parser.add_argument('--model-type', default='gbm', choices=['ridge', 'gbm'],
                        help='Model type: ridge (linear) or gbm (gradient boosting)')
    parser.add_argument('--position-scale', type=float, default=5.0,
                        help='Scale factor: prediction -> position size')
    parser.add_argument('--max-position', type=float, default=0.8,
                        help='Max position size (fraction of capital)')
    parser.add_argument('--long-bias', type=float, default=0.5,
                        help='Base long position (market trends up)')
    parser.add_argument('--long-only', action='store_true', default=True,
                        help='Long only: signals size between min and max long')
    parser.add_argument('--min-position', type=float, default=0.1,
                        help='Minimum long position (always in the market)')
    args = parser.parse_args()

    # Load data
    data = torch.load(DATA_DIR / 'features_daily.pt', weights_only=False)
    features = data['features'].numpy()
    spy = data['spy'].numpy()
    timestamps = data['timestamps']
    feature_names = list(data['feature_names'])

    T = features.shape[0]
    returns = np.zeros(T)
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)

    next_returns = np.zeros(T)
    next_returns[:-1] = returns[1:]

    # Add lagged + interaction features
    features, feature_names = add_lagged_features(features, feature_names)
    features, feature_names = add_interaction_features(features, feature_names)

    # Select signal features (only ones that exist)
    all_signal_names = SIGNAL_FEATURES + [f for f in feature_names
                                           if '_lag' in f or '_x_' in f]
    signal_indices = []
    signal_names = []
    for fname in all_signal_names:
        if fname in feature_names:
            signal_indices.append(feature_names.index(fname))
            signal_names.append(fname)

    alpha_idx = feature_names.index('alpha')

    print("=" * 70)
    print("MULTI-SIGNAL TRADING MODEL")
    print(f"  {len(signal_names)} signal features")
    print(f"  Ridge alpha: {args.ridge_alpha}")
    print(f"  Position scale: {args.position_scale}")
    print(f"  Max position: {args.max_position}")
    print("=" * 70)

    # --- Walk-forward simulation ---
    sim_start = args.cal_days
    cost_frac = args.cost_bps / 10000.0
    capital = args.starting_capital
    position = 0.0

    daily_log = []
    equity_curve = [capital]
    recal_count = 0
    last_recal = -999

    model = None
    scaler = None

    for t in range(sim_start, T):
        # Recalibrate
        if t - last_recal >= args.recal_days or model is None:
            cal_start = max(0, t - args.cal_days)
            X_train = features[cal_start:t][:, signal_indices]
            y_train = next_returns[cal_start:t]

            # Remove NaN rows
            valid = ~np.any(np.isnan(X_train), axis=1) & ~np.isnan(y_train)
            X_clean = X_train[valid]
            y_clean = y_train[valid]

            if len(X_clean) < 50:
                last_recal = t
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            if args.model_type == 'gbm':
                model = GradientBoostingRegressor(
                    n_estimators=50,       # few trees = less overfit
                    max_depth=2,           # shallow = captures interactions but not noise
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_leaf=20,   # need 20 samples per leaf
                    max_features=0.5,      # only see half the features per tree
                )
                model.fit(X_scaled, y_clean)
            else:
                model = Ridge(alpha=args.ridge_alpha)
                model.fit(X_scaled, y_clean)

            last_recal = t
            recal_count += 1

            # Print feature importance at recalibration
            if recal_count <= 3 or recal_count % 4 == 0:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    importances = np.abs(model.coef_)
                top_idx = np.argsort(importances)[::-1][:10]
                print(f"\n  Recal #{recal_count} at {timestamps[t][:10]} "
                      f"(trained on {len(X_clean)} days, model={args.model_type})")
                print(f"  Top 10 features:")
                for rank, idx in enumerate(top_idx):
                    print(f"    {rank+1:>2}. {signal_names[idx]:<35} "
                          f"imp={importances[idx]:.4f}")

        if model is None:
            equity_curve.append(capital)
            continue

        # --- Dip Score: raw signals directly, no return prediction ---
        # Each signal contributes based on its trained correlation with returns
        # INVERTED: negative correlation with returns = "bearish" = dip = BUY MORE
        # We size up when signals say "bearish" because in a trending market, dips revert

        X_today = features[t, signal_indices]
        X_today = np.nan_to_num(X_today, nan=0.0)

        if model is not None and scaler is not None:
            X_scaled = scaler.transform(X_today.reshape(1, -1))
            pred_return = model.predict(X_scaled)[0]
        else:
            pred_return = 0.0

        # Alpha risk scaler
        cal_start = max(0, t - args.cal_days)
        alpha_hist = features[cal_start:t, alpha_idx]
        ret_hist = next_returns[cal_start:t]
        risk_scaler, abs_regime = get_alpha_risk_scaler(
            features[t, alpha_idx], alpha_hist, ret_hist)

        # Direct dip score from key features (no model needed):
        # Each normalized feature * its sign = how "dippy" things are
        # Negative momentum = dip. High VIX = fear = dip. Low alpha = structure weak = dip.
        dip_score = 0.0
        n_signals = 0

        for fi, fname in enumerate(signal_names):
            if scaler is None:
                break
            z = X_scaled[0, fi]  # standardized value

            # These features are BEARISH when high -> flip to dip score
            # (high value = bad = dip opportunity = size up)
            bearish_when_high = {
                'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
                'up_frac_14', 'up_frac_28',
                'dist_from_low_20', 'dist_from_low_50',
                'ratio_SPY_TLT', 'ratio_HYG_TLT', 'ratio_SPY_GLD',
                'alpha', 'alpha_accel',
                'dist_from_high_20', 'dist_from_high_50',
            }
            # These features are BULLISH when high -> same direction as dip
            # (high value = opportunity = size up)
            bullish_when_high = {
                'macro_^VIX', 'sector_dispersion', 'sector_dispersion_20',
                'volatility_10', 'volatility_20',
                'astro_venus_retrograde',
                'astro_is_opex',
                'regime_STRESS',
            }

            if fname in bearish_when_high:
                dip_score += -z  # flip: high momentum = NOT a dip
                n_signals += 1
            elif fname in bullish_when_high:
                dip_score += z   # high VIX = IS a dip
                n_signals += 1

        if n_signals > 0:
            dip_score /= n_signals  # normalize to [-1, 1] ish

        # Position: base + dip_score * scale, always long
        raw_position = args.long_bias + dip_score * 0.3  # 0.3 = how reactive
        raw_position = np.clip(raw_position, args.min_position, args.max_position)
        target_pos = raw_position * risk_scaler

        # Transaction cost
        pos_change = abs(target_pos - position)
        cost = pos_change * cost_frac * capital

        # PnL from yesterday's position
        day_return = returns[t]
        pnl = position * day_return * capital - cost
        capital += pnl
        position = target_pos

        equity_curve.append(capital)

        daily_log.append({
            'date': timestamps[t][:10],
            'alpha': float(features[t, alpha_idx]),
            'regime': abs_regime,
            'pred_return': float(pred_return * 100),
            'position': float(position),
            'risk_scaler': float(risk_scaler),
            'spy': float(spy[t]),
            'day_return': float(day_return * 100),
            'pnl': float(pnl),
            'capital': float(capital),
        })

    equity = np.array(equity_curve)

    # --- Results ---
    final_capital = equity[-1]
    total_return = (final_capital - args.starting_capital) / args.starting_capital * 100
    n_years = (T - sim_start) / 252
    annual_return = ((final_capital / args.starting_capital) ** (1 / n_years) - 1) * 100

    spy_start = spy[sim_start]
    spy_end = spy[-1]
    bh_return = (spy_end - spy_start) / spy_start * 100
    bh_annual = ((spy_end / spy_start) ** (1 / n_years) - 1) * 100

    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * sqrt(252)

    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max * 100
    max_dd = drawdowns.max()

    active_days = [d for d in daily_log if abs(d['position']) > 0.01]
    win_rate = sum(1 for d in active_days if d['pnl'] > 0) / max(len(active_days), 1) * 100

    positions = [d['position'] for d in daily_log]
    n_trades = sum(1 for i in range(1, len(positions))
                   if abs(positions[i] - positions[i-1]) > 0.01)

    print(f"\n{'=' * 70}")
    print("SIMULATION RESULTS")
    print(f"{'=' * 70}")
    print(f"""
  Period:           {timestamps[sim_start][:10]} to {timestamps[-1][:10]} ({n_years:.1f} years)
  Recalibrations:   {recal_count}
  Starting Capital: ${args.starting_capital:>12,.0f}
  Final Capital:    ${final_capital:>12,.0f}

  MULTI-SIGNAL MODEL:
    Total Return:   {total_return:>+8.2f}%
    Annual Return:  {annual_return:>+8.2f}%
    Sharpe Ratio:   {sharpe:>8.2f}
    Max Drawdown:   {max_dd:>8.2f}%
    Win Rate:       {win_rate:>8.1f}%
    Total Trades:   {n_trades:>5}

  BUY AND HOLD:
    Total Return:   {bh_return:>+8.2f}%
    Annual Return:  {bh_annual:>+8.2f}%

  EDGE:
    Return Edge:    {total_return - bh_return:>+8.2f}%
""")

    # Position distribution
    pos_array = np.array(positions)
    print(f"  POSITION DISTRIBUTION:")
    print(f"    Mean:   {pos_array.mean():>+.3f}")
    print(f"    Std:    {np.abs(pos_array).std():>.3f}")
    print(f"    Long:   {(pos_array > 0.01).mean()*100:.1f}% of days")
    print(f"    Short:  {(pos_array < -0.01).mean()*100:.1f}% of days")
    print(f"    Flat:   {(np.abs(pos_array) <= 0.01).mean()*100:.1f}% of days")
    print(f"    Max:    {pos_array.max():>+.3f}")
    print(f"    Min:    {pos_array.min():>+.3f}")

    # Yearly breakdown
    print(f"\n  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'B&H':>8} {'Edge':>8}")
    print(f"  {'-' * 55}")

    years = sorted(set(d['date'][:4] for d in daily_log))
    for year in years:
        year_days = [d for d in daily_log if d['date'][:4] == year]
        if len(year_days) < 10:
            continue

        yr_start = year_days[0]['capital'] - year_days[0]['pnl']
        yr_end = year_days[-1]['capital']
        yr_ret = (yr_end - yr_start) / yr_start * 100

        yr_rets = np.array([(year_days[i]['capital'] - year_days[i-1]['capital']) /
                             year_days[i-1]['capital']
                             for i in range(1, len(year_days))])
        yr_sharpe = yr_rets.mean() / (yr_rets.std() + 1e-8) * sqrt(252) if len(yr_rets) > 1 else 0

        yr_eq = np.array([yr_start] + [d['capital'] for d in year_days])
        yr_rm = np.maximum.accumulate(yr_eq)
        yr_dd = ((yr_rm - yr_eq) / yr_rm * 100).max()

        yr_bh = (year_days[-1]['spy'] - year_days[0]['spy']) / year_days[0]['spy'] * 100

        print(f"  {year:<6} {yr_ret:>+7.2f}% {yr_sharpe:>+6.2f} {yr_dd:>6.2f}% "
              f"{yr_bh:>+7.2f}% {yr_ret - yr_bh:>+7.2f}%")

    # 5 best and worst
    sorted_days = sorted(daily_log, key=lambda d: d['pnl'])
    print(f"\n  5 WORST DAYS:")
    for d in sorted_days[:5]:
        print(f"    {d['date']}  PnL=${d['pnl']:>+9,.0f}  pos={d['position']:>+.2f}  "
              f"pred={d['pred_return']:>+.3f}%  SPY={d['day_return']:>+.2f}%")
    print(f"\n  5 BEST DAYS:")
    for d in sorted_days[-5:]:
        print(f"    {d['date']}  PnL=${d['pnl']:>+9,.0f}  pos={d['position']:>+.2f}  "
              f"pred={d['pred_return']:>+.3f}%  SPY={d['day_return']:>+.2f}%")

    # Last 10 days
    print(f"\n  LAST 10 DAYS:")
    print(f"  {'Date':<12} {'Pred':>7} {'Pos':>6} {'Risk':>5} {'SPY%':>7} {'PnL':>9} {'Capital':>12}")
    print(f"  {'-' * 65}")
    for d in daily_log[-10:]:
        print(f"  {d['date']:<12} {d['pred_return']:>+6.3f}% {d['position']:>+5.2f} "
              f"{d['risk_scaler']:>4.2f} {d['day_return']:>+6.2f}% "
              f"${d['pnl']:>+8,.0f} ${d['capital']:>11,.0f}")

    # Save
    save_path = RESULTS_DIR / 'multi_signal_3yr.json'
    with open(save_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': {
                'total_return': round(total_return, 2),
                'annual_return': round(annual_return, 2),
                'sharpe': round(float(sharpe), 2),
                'max_drawdown': round(float(max_dd), 2),
                'win_rate': round(win_rate, 1),
                'n_trades': n_trades,
                'bh_return': round(float(bh_return), 2),
            },
            'equity_curve': [round(float(e), 2) for e in equity[::5]],
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
