"""
Multi-Signal v2 Sweep — fixes from gap analysis:

1. FLIP risk scaler: STRESS = opportunity (size UP), CALM = reduce reactivity
2. TREND FILTER: long-term momentum sets position floor
   - 50d momentum > 0 -> min position = bias (stay fully invested)
   - 50d momentum < 0 -> allow reducing to min_position
3. VIX MEAN REVERSION: high VIX = load up (fear is the dip signal)
4. Better dip score: weight signals by their discovered strength

Sweep all params including the new ones.
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

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)

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
    'astro_venus_retrograde',
    'weather_temp_range_c',
]

# Signals where HIGH value = market extended = reduce
# (in mean-reversion: high momentum = already ran, less upside)
REDUCE_WHEN_HIGH = {
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
    'up_frac_14', 'up_frac_28',
    'dist_from_low_20', 'dist_from_low_50',
    'ratio_SPY_TLT', 'ratio_HYG_TLT', 'ratio_SPY_GLD',
}

# Signals where HIGH value = fear/stress = BUY THE DIP
LOAD_UP_WHEN_HIGH = {
    'macro_^VIX', 'sector_dispersion', 'sector_dispersion_20',
    'volatility_10', 'volatility_20',
    'alpha', 'alpha_accel',  # high alpha = crowded but about to snap -> buy after
    'astro_venus_retrograde', 'astro_is_opex',
}


def run_simulation(features, spy, returns, next_returns,
                   signal_indices, signal_names,
                   alpha_idx, mom50_idx, vix_idx,
                   sim_start, config):
    T = len(spy)
    cal_days = config['cal_days']
    recal_days = config['recal_days']
    cost_frac = config['cost_bps'] / 10000.0
    long_bias = config['long_bias']
    reactivity = config['reactivity']
    min_pos = config['min_position']
    max_pos = config['max_position']
    trend_floor = config['trend_floor']
    vix_boost = config['vix_boost']
    stress_mode = config['stress_mode']  # 'opportunity' or 'danger'

    capital = 100000.0
    position = 0.0
    scaler = None
    last_recal = -999

    equity = [capital]

    for t in range(sim_start, T):
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

        X_today = np.nan_to_num(features[t, signal_indices], nan=0.0)
        X_scaled = scaler.transform(X_today.reshape(1, -1))[0]

        # --- Dip score ---
        dip_score = 0.0
        n_signals = 0
        for fi, fname in enumerate(signal_names):
            z = X_scaled[fi]
            if fname in REDUCE_WHEN_HIGH:
                dip_score += -z  # high momentum = NOT a dip, reduce score
                n_signals += 1
            elif fname in LOAD_UP_WHEN_HIGH:
                dip_score += z   # high VIX = IS a dip, increase score
                n_signals += 1
        if n_signals > 0:
            dip_score /= n_signals

        # --- Trend filter ---
        # If 50-day momentum is positive -> macro uptrend -> set higher floor
        mom50 = features[t, mom50_idx] if mom50_idx is not None else 0
        if mom50 > 0:
            effective_min = max(min_pos, trend_floor)
        else:
            effective_min = min_pos

        # --- VIX boost ---
        # When VIX is elevated (z > 1), boost position
        vix_z = 0
        if vix_idx is not None:
            vix_raw = features[t, vix_idx]
            cal_start = max(0, t - cal_days)
            vix_hist = features[cal_start:t, vix_idx]
            vix_hist = vix_hist[~np.isnan(vix_hist)]
            if len(vix_hist) > 20:
                vix_z = (vix_raw - vix_hist.mean()) / (vix_hist.std() + 1e-8)

        vix_addition = max(0, vix_z - 0.5) * vix_boost  # only kicks in when VIX elevated

        # --- Risk scaler (flipped or normal) ---
        cal_start_r = max(0, t - cal_days)
        alpha_hist = features[cal_start_r:t, alpha_idx]
        alpha_val = features[t, alpha_idx]

        regime_pcts = [0, 10, 25, 50, 75, 90, 100]
        if stress_mode == 'opportunity':
            # FLIP: stress = opportunity (size UP), calm = less reactive
            # When alpha is high (stress/crisis), the bounce potential is highest
            alpha_pct = 50  # default
            if len(alpha_hist) > 20:
                alpha_pct = (alpha_hist < alpha_val).mean() * 100
            # High percentile (stress) -> boost; low percentile (calm) -> neutral
            stress_boost = max(0, (alpha_pct - 50) / 50) * 0.2  # 0 to 0.2 boost
            risk_scaler = 1.0 + stress_boost
        else:
            # Original: stress = reduce
            risk_scaler = 1.0
            if len(alpha_hist) > 20:
                for i in range(6):
                    lo = np.percentile(alpha_hist, regime_pcts[i])
                    hi = np.percentile(alpha_hist, regime_pcts[i + 1]) if i < 5 else float('inf')
                    if lo <= alpha_val < hi:
                        ret_hist = next_returns[cal_start_r:t]
                        mask = (alpha_hist >= lo) & (alpha_hist < hi)
                        if mask.sum() > 10:
                            r = ret_hist[mask]
                            ir = r.mean() / (r.std() + 1e-8)
                            risk_scaler = float(np.clip(0.6 + ir * 4.0, 0.5, 1.0))
                        break

        # --- Final position ---
        raw_pos = long_bias + dip_score * reactivity + vix_addition
        raw_pos = np.clip(raw_pos, effective_min, max_pos)
        target_pos = min(raw_pos * risk_scaler, max_pos)

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

    return {
        'total_return': float(total_ret),
        'annual_return': float(annual_ret),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
    }


def run_one(args_tuple):
    (features, spy, returns, next_returns,
     signal_indices, signal_names, alpha_idx, mom50_idx, vix_idx,
     sim_start, config, config_id) = args_tuple
    try:
        r = run_simulation(features, spy, returns, next_returns,
                          signal_indices, signal_names,
                          alpha_idx, mom50_idx, vix_idx,
                          sim_start, config)
        return config_id, config, r
    except Exception as e:
        return config_id, config, {'error': str(e)}


def main():
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
    mom50_idx = feature_names.index('momentum_50') if 'momentum_50' in feature_names else None
    vix_idx = feature_names.index('macro_^VIX') if 'macro_^VIX' in feature_names else None

    signal_indices = []
    signal_names = []
    for fname in SIGNAL_FEATURES:
        if fname in feature_names:
            signal_indices.append(feature_names.index(fname))
            signal_names.append(fname)

    sim_start = 252
    bh_ret = (spy[-1] - spy[sim_start]) / spy[sim_start] * 100

    print("=" * 70)
    print("MULTI-SIGNAL v2 SWEEP")
    print(f"  Fixes: flipped risk scaler, trend filter, VIX boost")
    print(f"  B&H baseline: {bh_ret:+.2f}%")
    print("=" * 70)

    if args.quick:
        grid = {
            'long_bias':    [0.7, 0.8, 0.9],
            'reactivity':   [0.1, 0.3, 0.5],
            'min_position': [0.1, 0.3],
            'max_position': [1.0],
            'trend_floor':  [0.5, 0.7, 0.9],
            'vix_boost':    [0.0, 0.1, 0.2],
            'stress_mode':  ['opportunity', 'danger'],
        }
    else:
        grid = {
            'long_bias':    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'reactivity':   [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
            'min_position': [0.1, 0.2, 0.3, 0.5],
            'max_position': [0.9, 1.0, 1.2],  # 1.2 = slight leverage
            'trend_floor':  [0.3, 0.5, 0.7, 0.8, 0.9],
            'vix_boost':    [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
            'stress_mode':  ['opportunity', 'danger'],
        }

    combos = list(product(
        grid['long_bias'], grid['reactivity'], grid['min_position'],
        grid['max_position'], grid['trend_floor'], grid['vix_boost'],
        grid['stress_mode']
    ))

    configs = []
    for i, (lb, react, minp, maxp, tf, vb, sm) in enumerate(combos):
        if minp >= maxp or tf > maxp:
            continue
        config = {
            'long_bias': lb, 'reactivity': react,
            'min_position': minp, 'max_position': maxp,
            'trend_floor': tf, 'vix_boost': vb, 'stress_mode': sm,
            'cal_days': 252, 'recal_days': 63, 'cost_bps': 3.0,
        }
        configs.append((features, spy, returns, next_returns,
                        signal_indices, signal_names,
                        alpha_idx, mom50_idx, vix_idx,
                        sim_start, config, i))

    print(f"  {len(configs)} configurations")

    t0 = time.time()
    all_results = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_one, c): c[-1] for c in configs}
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

    # Sort by Sharpe
    all_results.sort(key=lambda x: x[1]['sharpe'], reverse=True)

    print(f"\n  TOP 20 by Sharpe:")
    print(f"  {'#':>3} {'Sharpe':>7} {'Return':>8} {'Annual':>7} {'MaxDD':>7} | "
          f"{'bias':>4} {'react':>5} {'min':>4} {'max':>4} {'tFlr':>4} {'vBst':>4} {'stress':<6}")
    print(f"  {'-' * 85}")

    for rank, (config, results) in enumerate(all_results[:20]):
        print(f"  {rank+1:>3} {results['sharpe']:>+6.2f} {results['total_return']:>+7.2f}% "
              f"{results['annual_return']:>+6.2f}% {results['max_drawdown']:>6.2f}% | "
              f"{config['long_bias']:>4.1f} {config['reactivity']:>5.2f} "
              f"{config['min_position']:>4.1f} {config['max_position']:>4.1f} "
              f"{config['trend_floor']:>4.1f} {config['vix_boost']:>4.2f} "
              f"{config['stress_mode'][:5]}")

    # Best balanced
    balanced = [(c, r) for c, r in all_results if r['sharpe'] > 1.0]
    if balanced:
        balanced.sort(key=lambda x: x[1]['total_return'], reverse=True)
        print(f"\n  BEST BALANCED (Sharpe > 1.0, sorted by return):")
        print(f"  {'#':>3} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} | config")
        print(f"  {'-' * 70}")
        for rank, (config, results) in enumerate(balanced[:15]):
            print(f"  {rank+1:>3} {results['total_return']:>+7.2f}% {results['sharpe']:>+6.2f} "
                  f"{results['max_drawdown']:>6.2f}% | bias={config['long_bias']:.1f} "
                  f"react={config['reactivity']:.2f} min={config['min_position']:.1f} "
                  f"max={config['max_position']:.1f} tFlr={config['trend_floor']:.1f} "
                  f"vBst={config['vix_boost']:.2f} {config['stress_mode'][:5]}")

    # Best return with Sharpe > 0.8
    decent = [(c, r) for c, r in all_results if r['sharpe'] > 0.8]
    if decent:
        decent.sort(key=lambda x: x[1]['total_return'], reverse=True)
        print(f"\n  BEST RETURN (Sharpe > 0.8):")
        for rank, (config, results) in enumerate(decent[:10]):
            print(f"  {rank+1:>3} {results['total_return']:>+7.2f}% {results['sharpe']:>+6.2f} "
                  f"{results['max_drawdown']:>6.2f}% | bias={config['long_bias']:.1f} "
                  f"react={config['reactivity']:.2f} tFlr={config['trend_floor']:.1f} "
                  f"vBst={config['vix_boost']:.2f} {config['stress_mode'][:5]}")

    # Sensitivity
    print(f"\n  PARAMETER SENSITIVITY:")
    for pname in ['long_bias', 'reactivity', 'trend_floor', 'vix_boost', 'stress_mode']:
        by_val = {}
        for config, results in all_results:
            v = config[pname]
            if v not in by_val:
                by_val[v] = {'sharpe': [], 'return': []}
            by_val[v]['sharpe'].append(results['sharpe'])
            by_val[v]['return'].append(results['total_return'])

        print(f"\n  {pname}:")
        for v in sorted(by_val.keys(), key=lambda x: str(x)):
            avg_s = np.mean(by_val[v]['sharpe'])
            avg_r = np.mean(by_val[v]['return'])
            print(f"    {str(v):>12} -> Sharpe {avg_s:>+.3f}  Return {avg_r:>+.1f}%  (n={len(by_val[v]['sharpe'])})")

    # Save
    save_path = RESULTS_DIR / 'sweep_v2.json'
    save_data = {
        'n_configs': len(all_results),
        'bh_return': float(bh_ret),
        'top_by_sharpe': [
            {'config': {k: v for k, v in c.items()}, 'results': r}
            for c, r in sorted(all_results, key=lambda x: x[1]['sharpe'], reverse=True)[:20]
        ],
        'best_balanced': [
            {'config': {k: v for k, v in c.items()}, 'results': r}
            for c, r in (balanced[:10] if balanced else [])
        ],
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
