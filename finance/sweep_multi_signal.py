"""
Parameter sweep for the multi-signal long-only dip-buying model.

Sweeps:
  - long_bias (base position): 0.3 to 0.9
  - dip_reactivity: how much to size up on dips
  - min_position: floor on position size
  - max_position: ceiling
  - risk_scaler_floor: minimum risk scaler (prevents STRESS from killing position)
  - risk_scaler_power: how aggressive the risk adjustment is

Runs walk-forward simulation for each config, reports best.
"""

import argparse
import json
import sys
import time
from math import sqrt
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

# Signal feature names
SIGNAL_FEATURES = [
    'ratio_SPY_TLT', 'ratio_HYG_TLT', 'ratio_SPY_GLD', 'ratio_XLK_XLF',
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
    'up_frac_14', 'up_frac_28',
    'dist_from_high_20', 'dist_from_low_20',
    'dist_from_high_50', 'dist_from_low_50',
    'alpha', 'delta_alpha', 'alpha_accel',
    'effective_rank', 'mode1_pct',
    'volatility_10', 'volatility_20',
    'sector_dispersion', 'sector_dispersion_20',
    'volume_ratio_20',
    'macro_^VIX',
    'astro_is_opex', 'astro_is_fomc_week', 'astro_is_quarter_end',
    'astro_dow_sin', 'astro_dow_cos',
    'astro_month_sin', 'astro_month_cos',
    'astro_venus_retrograde', 'astro_mercury_retrograde',
    'weather_temp_range_c', 'weather_sunshine_hours',
    'weather_precipitation_mm',
]

BEARISH_WHEN_HIGH = {
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
    'up_frac_14', 'up_frac_28',
    'dist_from_low_20', 'dist_from_low_50',
    'ratio_SPY_TLT', 'ratio_HYG_TLT', 'ratio_SPY_GLD',
    'alpha', 'alpha_accel',
    'dist_from_high_20', 'dist_from_high_50',
}

BULLISH_WHEN_HIGH = {
    'macro_^VIX', 'sector_dispersion', 'sector_dispersion_20',
    'volatility_10', 'volatility_20',
    'astro_venus_retrograde', 'astro_is_opex', 'regime_STRESS',
}


def run_simulation(features, spy, returns, next_returns,
                   signal_indices, signal_names, alpha_idx,
                   sim_start, config):
    """Run one walk-forward simulation with given config."""
    T = len(spy)
    cal_days = config['cal_days']
    recal_days = config['recal_days']
    cost_frac = config['cost_bps'] / 10000.0
    long_bias = config['long_bias']
    reactivity = config['reactivity']
    min_pos = config['min_position']
    max_pos = config['max_position']
    risk_floor = config['risk_floor']
    risk_power = config['risk_power']

    capital = 100000.0
    position = 0.0
    scaler = None
    last_recal = -999

    equity = [capital]

    for t in range(sim_start, T):
        # Recalibrate scaler
        if t - last_recal >= recal_days or scaler is None:
            cal_start = max(0, t - cal_days)
            X_train = features[cal_start:t][:, signal_indices]
            valid = ~np.any(np.isnan(X_train), axis=1)
            if valid.sum() > 50:
                scaler = StandardScaler()
                scaler.fit(X_train[valid])
            last_recal = t

        if scaler is None:
            equity.append(capital)
            continue

        # Compute dip score
        X_today = np.nan_to_num(features[t, signal_indices], nan=0.0)
        X_scaled = scaler.transform(X_today.reshape(1, -1))[0]

        dip_score = 0.0
        n_signals = 0
        for fi, fname in enumerate(signal_names):
            z = X_scaled[fi]
            if fname in BEARISH_WHEN_HIGH:
                dip_score += -z
                n_signals += 1
            elif fname in BULLISH_WHEN_HIGH:
                dip_score += z
                n_signals += 1

        if n_signals > 0:
            dip_score /= n_signals

        # Alpha risk scaler with floor
        cal_start = max(0, t - cal_days)
        alpha_hist = features[cal_start:t, alpha_idx]
        ret_hist = next_returns[cal_start:t]
        alpha_val = features[t, alpha_idx]

        regime_pcts = [0, 10, 25, 50, 75, 90, 100]
        risk_scaler = 0.7
        for i in range(6):
            lo = np.percentile(alpha_hist, regime_pcts[i])
            hi = np.percentile(alpha_hist, regime_pcts[i + 1]) if i < 5 else float('inf')
            if lo <= alpha_val < hi:
                mask = (alpha_hist >= lo) & (alpha_hist < hi)
                if mask.sum() > 10:
                    r = ret_hist[mask]
                    ir = r.mean() / (r.std() + 1e-8)
                    risk_scaler = float(np.clip(0.6 + ir * risk_power, risk_floor, 1.0))
                break

        # Position sizing
        raw_pos = long_bias + dip_score * reactivity
        raw_pos = np.clip(raw_pos, min_pos, max_pos)
        target_pos = raw_pos * risk_scaler

        # PnL
        day_return = returns[t]
        pos_change = abs(target_pos - position)
        cost = pos_change * cost_frac * capital
        pnl = position * day_return * capital - cost
        capital += pnl
        position = target_pos
        equity.append(capital)

    equity = np.array(equity)
    final = equity[-1]
    total_ret = (final - 100000) / 100000 * 100
    n_years = (T - sim_start) / 252

    daily_rets = np.diff(equity) / equity[:-1]
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * sqrt(252)

    rm = np.maximum.accumulate(equity)
    max_dd = ((rm - equity) / rm * 100).max()

    annual_ret = ((final / 100000) ** (1 / n_years) - 1) * 100

    # Yearly sharpes
    year_sharpes = {}
    days_per_year = (T - sim_start) // len(set(ts[:4] for ts in timestamps[sim_start:]))
    # simplified: just return overall metrics

    return {
        'total_return': float(total_ret),
        'annual_return': float(annual_ret),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'final_capital': float(final),
    }


def run_one(args_tuple):
    """Wrapper for multiprocessing."""
    (features, spy, returns, next_returns,
     signal_indices, signal_names, alpha_idx,
     sim_start, config, config_id) = args_tuple
    try:
        results = run_simulation(
            features, spy, returns, next_returns,
            signal_indices, signal_names, alpha_idx,
            sim_start, config)
        return config_id, config, results
    except Exception as e:
        return config_id, config, {'error': str(e)}


# Global for multiprocessing
timestamps = None

def main():
    global timestamps

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-workers', type=int, default=8)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    data = torch.load(DATA_DIR / 'features_daily.pt', weights_only=False)
    features = data['features'].numpy()
    spy = data['spy'].numpy()
    timestamps = data['timestamps']
    feature_names = list(data['feature_names'])

    T = len(spy)
    returns = np.zeros(T)
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)
    next_returns = np.zeros(T)
    next_returns[:-1] = returns[1:]

    alpha_idx = feature_names.index('alpha')

    signal_indices = []
    signal_names = []
    for fname in SIGNAL_FEATURES:
        if fname in feature_names:
            signal_indices.append(feature_names.index(fname))
            signal_names.append(fname)

    sim_start = 252
    bh_ret = (spy[-1] - spy[sim_start]) / spy[sim_start] * 100

    print("=" * 70)
    print("MULTI-SIGNAL PARAMETER SWEEP")
    print(f"  B&H baseline: {bh_ret:+.2f}%")
    print("=" * 70)

    if args.quick:
        grid = {
            'long_bias':    [0.4, 0.6, 0.8],
            'reactivity':   [0.1, 0.3, 0.5],
            'min_position': [0.1, 0.3],
            'max_position': [0.8, 1.0],
            'risk_floor':   [0.3, 0.5, 0.7, 1.0],
            'risk_power':   [2.0, 4.0],
        }
    else:
        grid = {
            'long_bias':    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'reactivity':   [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            'min_position': [0.05, 0.1, 0.2, 0.3, 0.4],
            'max_position': [0.8, 0.9, 1.0],
            'risk_floor':   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            'risk_power':   [1.0, 2.0, 3.0, 4.0, 5.0],
        }

    combos = list(product(
        grid['long_bias'], grid['reactivity'], grid['min_position'],
        grid['max_position'], grid['risk_floor'], grid['risk_power']
    ))

    configs = []
    for i, (lb, react, minp, maxp, rf, rp) in enumerate(combos):
        if minp >= maxp:
            continue
        config = {
            'long_bias': lb, 'reactivity': react,
            'min_position': minp, 'max_position': maxp,
            'risk_floor': rf, 'risk_power': rp,
            'cal_days': 252, 'recal_days': 63, 'cost_bps': 3.0,
        }
        configs.append((features, spy, returns, next_returns,
                        signal_indices, signal_names, alpha_idx,
                        sim_start, config, i))

    print(f"  {len(configs)} configurations")

    t0 = time.time()
    all_results = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_one, c): c[9] for c in configs}
        done = 0
        for future in as_completed(futures):
            cid, config, results = future.result()
            if 'error' not in results:
                all_results.append((config, results))
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(configs) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(configs)} ({rate:.0f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Done: {len(all_results)} results in {elapsed:.1f}s")

    # Sort by Sharpe (primary), then by total return (secondary)
    all_results.sort(key=lambda x: (x[1]['sharpe'], x[1]['total_return']), reverse=True)

    print(f"\n  TOP 25 by Sharpe:")
    print(f"  {'#':>3} {'Sharpe':>7} {'Return':>8} {'Annual':>7} {'MaxDD':>7} | "
          f"{'bias':>4} {'react':>5} {'min':>4} {'max':>4} {'rFlr':>4} {'rPow':>4}")
    print(f"  {'-' * 75}")

    for rank, (config, results) in enumerate(all_results[:25]):
        print(f"  {rank+1:>3} {results['sharpe']:>+6.2f} {results['total_return']:>+7.2f}% "
              f"{results['annual_return']:>+6.2f}% {results['max_drawdown']:>6.2f}% | "
              f"{config['long_bias']:>4.1f} {config['reactivity']:>5.2f} "
              f"{config['min_position']:>4.1f} {config['max_position']:>4.1f} "
              f"{config['risk_floor']:>4.1f} {config['risk_power']:>4.1f}")

    # Sort by total return
    all_results.sort(key=lambda x: x[1]['total_return'], reverse=True)

    print(f"\n  TOP 25 by Total Return:")
    print(f"  {'#':>3} {'Return':>8} {'Annual':>7} {'Sharpe':>7} {'MaxDD':>7} | "
          f"{'bias':>4} {'react':>5} {'min':>4} {'max':>4} {'rFlr':>4} {'rPow':>4}")
    print(f"  {'-' * 75}")

    for rank, (config, results) in enumerate(all_results[:25]):
        print(f"  {rank+1:>3} {results['total_return']:>+7.2f}% "
              f"{results['annual_return']:>+6.2f}% {results['sharpe']:>+6.2f} "
              f"{results['max_drawdown']:>6.2f}% | "
              f"{config['long_bias']:>4.1f} {config['reactivity']:>5.2f} "
              f"{config['min_position']:>4.1f} {config['max_position']:>4.1f} "
              f"{config['risk_floor']:>4.1f} {config['risk_power']:>4.1f}")

    # Sensitivity
    print(f"\n  PARAMETER SENSITIVITY:")
    for pname, pvalues in grid.items():
        by_val = {}
        for config, results in all_results:
            v = config[pname]
            if v not in by_val:
                by_val[v] = []
            by_val[v].append(results['sharpe'])

        print(f"\n  {pname}:")
        for v in sorted(by_val.keys()):
            avg = np.mean(by_val[v])
            top10 = np.mean(sorted(by_val[v], reverse=True)[:max(1, len(by_val[v])//10)])
            print(f"    {v:>6.2f} -> avg Sharpe {avg:>+.3f}  top10% {top10:>+.3f}  (n={len(by_val[v])})")

    # Best balanced (Sharpe > 1.0 AND highest return)
    balanced = [(c, r) for c, r in all_results if r['sharpe'] > 1.0]
    if balanced:
        balanced.sort(key=lambda x: x[1]['total_return'], reverse=True)
        print(f"\n  BEST BALANCED (Sharpe > 1.0, sorted by return):")
        print(f"  {'#':>3} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} | config")
        print(f"  {'-' * 60}")
        for rank, (config, results) in enumerate(balanced[:10]):
            print(f"  {rank+1:>3} {results['total_return']:>+7.2f}% {results['sharpe']:>+6.2f} "
                  f"{results['max_drawdown']:>6.2f}% | bias={config['long_bias']:.1f} "
                  f"react={config['reactivity']:.2f} min={config['min_position']:.1f} "
                  f"max={config['max_position']:.1f} rFlr={config['risk_floor']:.1f} "
                  f"rPow={config['risk_power']:.1f}")

    # Save
    save_path = RESULTS_DIR / 'sweep_multi_signal.json'
    save_data = {
        'n_configs': len(all_results),
        'bh_return': float(bh_ret),
        'top_by_sharpe': [{'config': {k: v for k, v in c.items()
                                       if not isinstance(v, np.ndarray)},
                           'results': r}
                          for c, r in sorted(all_results,
                                             key=lambda x: x[1]['sharpe'],
                                             reverse=True)[:25]],
        'top_by_return': [{'config': {k: v for k, v in c.items()
                                       if not isinstance(v, np.ndarray)},
                           'results': r}
                          for c, r in all_results[:25]],
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
