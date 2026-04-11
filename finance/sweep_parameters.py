"""
Massive parameter sweep for the Alpha Decision Model.

Sweeps every dimension that matters:
  - Influence weights per group (6 groups)
  - Alpha percentile thresholds (regime boundaries)
  - Base confidence per regime
  - Risk scaler formula
  - Correlation threshold (noise floor)
  - Transaction cost sensitivity

Runs thousands of configurations, reports:
  1. Best overall configs
  2. Parameter sensitivity (which params matter most)
  3. Stability analysis (how fragile is the edge)

Usage:
    python finance/sweep_parameters.py
    python finance/sweep_parameters.py --n-workers 8
    python finance/sweep_parameters.py --quick  # reduced grid
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

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
RESULTS_DIR.mkdir(exist_ok=True)


# --- Minimal model (self-contained for multiprocessing) ---

def fit_and_backtest(features, returns, spy, train_end, val_end, config):
    """Fit model on train, evaluate on val and test. Returns metrics dict."""
    n = features.shape[0]
    alpha_idx = config['alpha_idx']
    da_idx = config['da_idx']
    feature_names = config['feature_names']
    group_defs = config['group_defs']
    cost_frac = config['cost_bps'] / 10000.0

    alpha_train = features[:train_end, alpha_idx]
    da_train = features[:train_end, da_idx] if da_idx is not None else np.zeros(train_end)

    # Next-bar returns
    next_returns = np.zeros(n)
    next_returns[:-1] = returns[1:]

    # --- FIT: compute percentiles and regime stats from training data ---

    # Alpha percentiles for mean-reversion signal
    pct_lo = config['pct_lo']  # e.g., 30 -> p30
    pct_hi = config['pct_hi']  # e.g., 70 -> p70
    alpha_plo = np.percentile(alpha_train, pct_lo)
    alpha_phi = np.percentile(alpha_train, pct_hi)
    alpha_p50 = np.percentile(alpha_train, 50)

    da_plo = np.percentile(da_train, 25)
    da_phi = np.percentile(da_train, 75)

    # Absolute regime calibration from training alpha
    regime_pcts = [0, 10, 25, 50, 75, 90, 100]
    regime_names = ['DEEP_CALM', 'CALM', 'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']
    regime_bounds = {}
    for i, rname in enumerate(regime_names):
        lo = float(np.percentile(alpha_train, regime_pcts[i]))
        hi = float(np.percentile(alpha_train, regime_pcts[i + 1])) if i < 5 else float('inf')
        regime_bounds[rname] = (lo, hi)

    # Compute regime return stats for risk scaler
    regime_ir = {}
    for rname, (lo, hi) in regime_bounds.items():
        mask = (alpha_train >= lo) & (alpha_train < hi)
        if mask.sum() > 10:
            r = next_returns[:train_end][mask]
            mean_r = r.mean()
            std_r = r.std()
            regime_ir[rname] = mean_r / (std_r + 1e-8)

    # Per-group feature correlations within regimes
    # (simplified: just compute correlations in the dominant regime)
    group_corrs = {}
    for gname, gdef in group_defs.items():
        indices = gdef['indices']
        if not indices:
            continue
        group_feat = features[:train_end, indices]
        # Correlations with next-day return
        corrs = []
        medians = []
        for fi in range(group_feat.shape[1]):
            col = group_feat[:, fi]
            if col.std() < 1e-10:
                corrs.append(0.0)
                medians.append(0.0)
                continue
            c = np.corrcoef(col, next_returns[:train_end])[0, 1]
            corrs.append(float(c) if not np.isnan(c) else 0.0)
            medians.append(float(np.median(col)))
        group_corrs[gname] = {'corrs': corrs, 'medians': medians, 'indices': indices}

    # --- PREDICT: generate positions for full dataset ---
    corr_threshold = config['corr_threshold']
    base_conf_long = config['base_conf_long']
    base_conf_short = config['base_conf_short']
    base_conf_mild = config['base_conf_mild']
    risk_scale_factor = config['risk_scale_factor']

    positions = np.zeros(n)
    for t in range(n):
        a = features[t, alpha_idx]
        da = features[t, da_idx] if da_idx is not None else 0

        # Mean-reversion signal
        if a > alpha_phi and da > da_phi:
            direction = -1.0
            base_conf = base_conf_short
        elif a <= alpha_plo or da < da_plo:
            direction = 1.0
            base_conf = base_conf_long
        elif a > alpha_p50:
            direction = 0.0
            base_conf = 0.0
        else:
            direction = 1.0
            base_conf = base_conf_mild

        if direction == 0:
            positions[t] = 0.0
            continue

        # Absolute regime risk scaler
        abs_regime = 'NORMAL'
        for rname, (lo, hi) in regime_bounds.items():
            if lo <= a < hi:
                abs_regime = rname
                break

        ir = regime_ir.get(abs_regime, 0.0)
        risk_scaler = float(np.clip(0.6 + ir * risk_scale_factor, 0.2, 1.0))

        # Group modifiers
        total_mod = 0.0
        for gname, gdef in group_defs.items():
            gc = group_corrs.get(gname)
            if gc is None:
                continue
            influence = gdef['influence']
            group_signal = 0.0
            n_active = 0
            for fi in range(len(gc['corrs'])):
                if abs(gc['corrs'][fi]) < corr_threshold:
                    continue
                idx = gc['indices'][fi]
                val = features[t, idx]
                feat_signal = np.sign(val - gc['medians'][fi]) * gc['corrs'][fi]
                group_signal += feat_signal
                n_active += 1
            if n_active > 0:
                group_signal = np.clip(group_signal / n_active, -1, 1)
                total_mod += group_signal * influence

        confidence = np.clip(base_conf + total_mod, 0.05, 1.0)
        positions[t] = direction * confidence * risk_scaler

    # --- EVALUATE on each split ---
    results = {}
    for split_name, start, end in [('train', 0, train_end),
                                     ('val', train_end, val_end),
                                     ('test', val_end, n)]:
        pos = positions[start:end]
        nr = next_returns[start:end]
        s = spy[start:end]

        if len(pos) < 10:
            results[split_name] = {'pnl': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0}
            continue

        bar_pnl = pos[:-1] * nr[:-1]
        costs = np.abs(np.diff(pos)) * cost_frac
        bar_pnl_net = bar_pnl - costs

        cum_pnl = np.cumsum(bar_pnl_net)
        total_pnl = cum_pnl[-1] * 100 if len(cum_pnl) > 0 else 0

        mean_p = bar_pnl_net.mean()
        std_p = bar_pnl_net.std() + 1e-8
        sharpe = mean_p / std_p * sqrt(252)

        running_max = np.maximum.accumulate(cum_pnl)
        max_dd = (running_max - cum_pnl).max() * 100

        active = bar_pnl_net[np.abs(pos[:-1]) > 0.01]
        win_rate = (active > 0).mean() * 100 if len(active) > 0 else 0

        bh = (s[-1] - s[0]) / (s[0] + 1e-8) * 100

        results[split_name] = {
            'pnl': float(total_pnl),
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
            'win_rate': float(win_rate),
            'bh_pnl': float(bh),
            'mean_pos': float(np.abs(pos).mean()),
        }

    return results


def run_one_config(args_tuple):
    """Wrapper for multiprocessing."""
    features, returns, spy, train_end, val_end, config, config_id = args_tuple
    try:
        results = fit_and_backtest(features, returns, spy, train_end, val_end, config)
        return config_id, config, results
    except Exception as e:
        return config_id, config, {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Parameter Sweep')
    parser.add_argument('--timescale', default='daily')
    parser.add_argument('--n-workers', type=int, default=8)
    parser.add_argument('--quick', action='store_true', help='Reduced grid')
    args = parser.parse_args()

    print("=" * 70)
    print("ALPHA DECISION MODEL — PARAMETER SWEEP")
    print("=" * 70)

    # Load data
    path = DATA_DIR / f'features_{args.timescale}.pt'
    data = torch.load(path, weights_only=False)
    features = data['features'].numpy()
    spy = data['spy'].numpy()
    timestamps = data['timestamps']
    feature_names = list(data['feature_names'])

    returns = np.zeros(len(spy))
    returns[1:] = (spy[1:] - spy[:-1]) / (spy[:-1] + 1e-8)

    T = features.shape[0]

    # Walk-forward split
    val_idx = T
    test_idx = T
    for i, ts in enumerate(timestamps):
        if '2025-01-01' in ts and val_idx == T:
            val_idx = i
        if '2025-07-01' in ts and test_idx == T:
            test_idx = i
    if val_idx == T:
        val_idx = int(T * 0.7)
        test_idx = int(T * 0.85)

    print(f"  Data: {T} bars, train={val_idx} val={test_idx - val_idx} test={T - test_idx}")

    bh_test = (spy[-1] - spy[test_idx]) / (spy[test_idx] + 1e-8) * 100
    print(f"  B&H test: {bh_test:+.2f}%")

    # Feature indices
    alpha_idx = feature_names.index('alpha')
    da_idx = feature_names.index('delta_alpha') if 'delta_alpha' in feature_names else None

    # Group definitions with feature indices
    from alpha_decision_model import FEATURE_GROUPS
    base_groups = {}
    for gname, gdef in FEATURE_GROUPS.items():
        indices = [feature_names.index(f) for f in gdef['features'] if f in feature_names]
        base_groups[gname] = {'indices': indices, 'influence': gdef['influence']}

    # --- Define sweep dimensions ---

    if args.quick:
        influence_grid = {
            'price':       [0.0, 0.3, 0.8],
            'macro':       [0.3, 0.7, 1.0],
            'calendar':    [0.0, 0.3, 0.5],
            'seasonality': [0.0, 0.1, 0.2],
            'weather':     [0.0, 0.02],
            'astro':       [0.0, 0.02],
        }
        pct_lo_grid = [30, 40]
        pct_hi_grid = [60, 70, 80]
        base_conf_long_grid = [0.5, 0.7]
        base_conf_short_grid = [0.4, 0.6]
        base_conf_mild_grid = [0.2, 0.4]
        risk_scale_grid = [3.0, 4.0, 5.0]
        corr_threshold_grid = [0.02, 0.05]
        cost_grid = [3.0, 5.0]
    else:
        influence_grid = {
            'price':       [0.0, 0.1, 0.3, 0.5, 0.8],
            'macro':       [0.0, 0.3, 0.5, 0.7, 1.0],
            'calendar':    [0.0, 0.1, 0.3, 0.5],
            'seasonality': [0.0, 0.05, 0.1, 0.2],
            'weather':     [0.0, 0.01, 0.02, 0.05],
            'astro':       [0.0, 0.01, 0.02, 0.05],
        }
        pct_lo_grid = [20, 30, 40]
        pct_hi_grid = [60, 70, 80]
        base_conf_long_grid = [0.3, 0.5, 0.6, 0.7, 0.8]
        base_conf_short_grid = [0.2, 0.4, 0.6]
        base_conf_mild_grid = [0.1, 0.2, 0.3, 0.4]
        risk_scale_grid = [2.0, 3.0, 4.0, 5.0, 6.0]
        corr_threshold_grid = [0.01, 0.02, 0.03, 0.05, 0.10]
        cost_grid = [3.0, 5.0, 8.0]

    # Phase 1: Sweep structural params (percentiles, confidence, risk)
    # with default influence weights
    print("\n" + "=" * 70)
    print("PHASE 1: Structural parameters (regime thresholds, confidence, risk)")
    print("=" * 70)

    structural_configs = list(product(
        pct_lo_grid, pct_hi_grid,
        base_conf_long_grid, base_conf_short_grid, base_conf_mild_grid,
        risk_scale_grid, corr_threshold_grid, cost_grid
    ))

    print(f"  {len(structural_configs)} configurations")

    configs = []
    for i, (plo, phi, bcl, bcs, bcm, rsf, ct, cost) in enumerate(structural_configs):
        if plo >= phi:
            continue
        config = {
            'alpha_idx': alpha_idx, 'da_idx': da_idx,
            'feature_names': feature_names,
            'group_defs': base_groups,
            'pct_lo': plo, 'pct_hi': phi,
            'base_conf_long': bcl, 'base_conf_short': bcs, 'base_conf_mild': bcm,
            'risk_scale_factor': rsf, 'corr_threshold': ct, 'cost_bps': cost,
        }
        configs.append((features, returns, spy, val_idx, test_idx, config, i))

    print(f"  {len(configs)} valid configs (after filtering)")

    t0 = time.time()
    all_results = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_one_config, c): c[6] for c in configs}
        done = 0
        for future in as_completed(futures):
            config_id, config, results = future.result()
            if 'error' not in results:
                all_results.append((config, results))
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(configs) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(configs)} ({rate:.0f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Phase 1 complete: {len(all_results)} results in {elapsed:.1f}s")

    # Sort by val Sharpe (primary selection criterion)
    all_results.sort(key=lambda x: x[1].get('val', {}).get('sharpe', -999), reverse=True)

    # Top 20 configs
    print(f"\n  TOP 20 by val Sharpe:")
    print(f"  {'#':>3} {'ValS':>6} {'ValPnL':>8} {'TestS':>6} {'TestPnL':>8} "
          f"{'TestDD':>7} {'WR':>5} | pLo pHi  bL   bS   bM  rSF  cT  cost")
    print(f"  {'-' * 95}")

    top_configs = []
    for rank, (config, results) in enumerate(all_results[:20]):
        v = results.get('val', {})
        t = results.get('test', {})
        print(f"  {rank+1:>3} {v.get('sharpe',0):>+5.2f} {v.get('pnl',0):>+7.2f}% "
              f"{t.get('sharpe',0):>+5.2f} {t.get('pnl',0):>+7.2f}% "
              f"{t.get('max_dd',0):>6.2f}% {t.get('win_rate',0):>4.1f}% | "
              f"{config['pct_lo']:>3} {config['pct_hi']:>3} "
              f"{config['base_conf_long']:.1f} {config['base_conf_short']:.1f} "
              f"{config['base_conf_mild']:.1f} {config['risk_scale_factor']:.0f} "
              f"{config['corr_threshold']:.2f} {config['cost_bps']:.0f}")
        top_configs.append(config)

    # --- Parameter sensitivity analysis ---
    print(f"\n  PARAMETER SENSITIVITY (avg val Sharpe by parameter value):")

    param_names = {
        'pct_lo': pct_lo_grid, 'pct_hi': pct_hi_grid,
        'base_conf_long': base_conf_long_grid, 'base_conf_short': base_conf_short_grid,
        'base_conf_mild': base_conf_mild_grid,
        'risk_scale_factor': risk_scale_grid, 'corr_threshold': corr_threshold_grid,
        'cost_bps': cost_grid,
    }

    sensitivity = {}
    for pname, pvalues in param_names.items():
        val_sharpes_by_value = {}
        for config, results in all_results:
            val = config[pname]
            vs = results.get('val', {}).get('sharpe', 0)
            if val not in val_sharpes_by_value:
                val_sharpes_by_value[val] = []
            val_sharpes_by_value[val].append(vs)

        print(f"\n  {pname}:")
        best_val = None
        best_sharpe = -999
        for val in sorted(val_sharpes_by_value.keys()):
            avg_s = np.mean(val_sharpes_by_value[val])
            n = len(val_sharpes_by_value[val])
            bar = '+' * max(0, int((avg_s + 1) * 20))
            print(f"    {val:>8} -> avg Sharpe {avg_s:>+.3f}  (n={n:>5})  {bar}")
            if avg_s > best_sharpe:
                best_sharpe = avg_s
                best_val = val

        spread = max(np.mean(v) for v in val_sharpes_by_value.values()) - \
                 min(np.mean(v) for v in val_sharpes_by_value.values())
        sensitivity[pname] = {'best': best_val, 'spread': spread}

    # Rank parameters by sensitivity
    print(f"\n  PARAMETER IMPORTANCE (by Sharpe spread across values):")
    for pname, info in sorted(sensitivity.items(), key=lambda x: x[1]['spread'], reverse=True):
        bar = '#' * int(info['spread'] * 50)
        print(f"    {pname:<20} spread={info['spread']:.3f}  best={info['best']}  {bar}")

    # Phase 2: Sweep influence weights with best structural params
    print(f"\n{'=' * 70}")
    print("PHASE 2: Influence weight sweep (using best structural params)")
    print("=" * 70)

    best_struct = top_configs[0] if top_configs else configs[0][5]

    inf_combos = list(product(
        influence_grid['price'], influence_grid['macro'],
        influence_grid['calendar'], influence_grid['seasonality'],
        influence_grid['weather'], influence_grid['astro'],
    ))
    print(f"  {len(inf_combos)} weight combinations")

    inf_configs = []
    for i, (pw, mw, cw, sw, ww, aw) in enumerate(inf_combos):
        groups = {}
        for gname, gdef in base_groups.items():
            influence = gdef['influence']
            if gname == 'price': influence = pw
            elif gname == 'macro': influence = mw
            elif gname == 'calendar': influence = cw
            elif gname == 'seasonality': influence = sw
            elif gname == 'weather': influence = ww
            elif gname == 'astro': influence = aw
            groups[gname] = {'indices': gdef['indices'], 'influence': influence}

        config = dict(best_struct)
        config['group_defs'] = groups
        config['_weights'] = {'price': pw, 'macro': mw, 'calendar': cw,
                              'seasonality': sw, 'weather': ww, 'astro': aw}
        inf_configs.append((features, returns, spy, val_idx, test_idx, config, i))

    t0 = time.time()
    inf_results = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_one_config, c): c[6] for c in inf_configs}
        for future in as_completed(futures):
            config_id, config, results = future.result()
            if 'error' not in results:
                inf_results.append((config, results))

    inf_results.sort(key=lambda x: x[1].get('val', {}).get('sharpe', -999), reverse=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    print(f"\n  TOP 10 weight configs:")
    print(f"  {'#':>3} {'ValS':>6} {'TestS':>6} {'TestPnL':>8} {'TestDD':>7} | "
          f"{'price':>5} {'macro':>5} {'cal':>5} {'seas':>5} {'weath':>5} {'astro':>5}")
    print(f"  {'-' * 80}")

    for rank, (config, results) in enumerate(inf_results[:10]):
        v = results.get('val', {})
        t = results.get('test', {})
        w = config.get('_weights', {})
        print(f"  {rank+1:>3} {v.get('sharpe',0):>+5.2f} {t.get('sharpe',0):>+5.2f} "
              f"{t.get('pnl',0):>+7.2f}% {t.get('max_dd',0):>6.2f}% | "
              f"{w.get('price',0):>5.2f} {w.get('macro',0):>5.2f} "
              f"{w.get('calendar',0):>5.2f} {w.get('seasonality',0):>5.2f} "
              f"{w.get('weather',0):>5.2f} {w.get('astro',0):>5.2f}")

    # Weight sensitivity
    print(f"\n  WEIGHT SENSITIVITY:")
    for gname in ['price', 'macro', 'calendar', 'seasonality', 'weather', 'astro']:
        by_weight = {}
        for config, results in inf_results:
            w = config.get('_weights', {}).get(gname, 0)
            vs = results.get('val', {}).get('sharpe', 0)
            if w not in by_weight:
                by_weight[w] = []
            by_weight[w].append(vs)

        print(f"  {gname}:")
        for w in sorted(by_weight.keys()):
            avg_s = np.mean(by_weight[w])
            print(f"    {w:>6.2f} -> avg Sharpe {avg_s:>+.3f}  (n={len(by_weight[w])})")

    # --- Final: best overall config on test ---
    print(f"\n{'=' * 70}")
    print("FINAL BEST CONFIG")
    print("=" * 70)

    best_config = inf_results[0][0] if inf_results else all_results[0][0]
    best_r = inf_results[0][1] if inf_results else all_results[0][1]

    print(f"\n  Structural:")
    for k in ['pct_lo', 'pct_hi', 'base_conf_long', 'base_conf_short',
              'base_conf_mild', 'risk_scale_factor', 'corr_threshold', 'cost_bps']:
        print(f"    {k:<22} = {best_config[k]}")

    if '_weights' in best_config:
        print(f"\n  Influence weights:")
        for k, v in sorted(best_config['_weights'].items()):
            print(f"    {k:<22} = {v:.2f}")

    print(f"\n  Results:")
    for split in ['train', 'val', 'test']:
        r = best_r.get(split, {})
        print(f"    {split:>5}: PnL={r.get('pnl',0):>+7.2f}%  Sharpe={r.get('sharpe',0):>+5.2f}  "
              f"MaxDD={r.get('max_dd',0):>5.2f}%  WR={r.get('win_rate',0):>4.1f}%  "
              f"B&H={r.get('bh_pnl',0):>+7.2f}%")

    # Save everything
    save_path = RESULTS_DIR / f'sweep_{args.timescale}.json'
    save_data = {
        'n_structural': len(all_results),
        'n_weight': len(inf_results),
        'best_config': {k: v for k, v in best_config.items()
                        if k not in ('feature_names', 'group_defs', 'alpha_idx', 'da_idx')},
        'best_results': best_r,
        'sensitivity': sensitivity,
        'top_20_structural': [
            {'config': {k: v for k, v in c.items()
                        if k not in ('feature_names', 'group_defs', 'alpha_idx', 'da_idx')},
             'results': r}
            for c, r in all_results[:20]
        ],
        'top_10_weights': [
            {'weights': c.get('_weights', {}), 'results': r}
            for c, r in inf_results[:10]
        ],
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


if __name__ == '__main__':
    main()
