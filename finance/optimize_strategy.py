"""
Strategy optimizer: find the best α-based rules for each day,
then converge to a unified ruleset.

Phase 1: Per-day optimization — sweep parameter space to find
         what WOULD have been optimal for each day.
Phase 2: Convergence — find the rules that work across ALL days.

Parameters to optimize:
  - α thresholds for long/short/flat
  - Momentum window (how many minutes of price trend to consider)
  - Momentum threshold (how strong must the trend be)
  - Hold time (minimum minutes before regime change triggers exit)
  - Mode 1 energy threshold (only trade when mode1 > X%)
"""

import json
import sys
from math import sqrt
from collections import deque
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

SYMBOLS = ['SPY','QQQ','IWM','DIA','XLK','XLF','XLE','XLV','XLI','XLY',
           'XLP','XLU','TLT','HYG','GLD','SLV','USO']


def bent_power_law(k, A, k0, alpha):
    return A * (k + k0) ** (-alpha)


def fit_alpha(S):
    S = S[S > 1e-10]
    n = len(S)
    if n < 4:
        return None, None, None
    k = np.arange(1, n + 1, dtype=np.float64)
    try:
        popt, _ = curve_fit(bent_power_law, k, S.astype(np.float64),
            p0=[S[0], max(1.0, n*0.1), 1.3],
            bounds=([0, 0, 0.01], [S[0]*100, n*5, 3.0]), maxfev=5000)
        pred = bent_power_law(k, *popt)
        ss_res = np.sum((S[:n] - pred)**2)
        ss_tot = np.sum((S[:n] - np.mean(S[:n]))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        energy = S**2
        mode1 = energy[0]/energy.sum()*100
        return float(popt[2]), float(r2), float(mode1)
    except:
        return None, None, None


def precompute_signals(day_data, corr_window=30):
    """Precompute α, momentum, and mode1 for every minute of the day."""
    returns = day_data.pct_change().dropna()
    R = returns.values
    n_min = R.shape[0]
    spy_idx = list(day_data.columns).index('SPY')
    spy = day_data['SPY'].values

    buffer = deque(maxlen=corr_window)
    signals = []

    for t in range(n_min):
        row = {str(day_data.columns[i]): R[t, i] for i in range(R.shape[1])}
        buffer.append(row)

        if len(buffer) < 15:
            signals.append(None)
            continue

        syms = list(day_data.columns)
        R_w = np.array([[r.get(str(s), 0.0) for s in syms] for r in buffer])
        var = np.var(R_w, axis=0)
        valid = var > 1e-15
        if valid.sum() < 5:
            signals.append(None)
            continue

        corr = np.corrcoef(R_w[:, valid].T)
        corr = np.nan_to_num(corr, nan=0.0)
        S = np.linalg.svd(corr, compute_uv=False)
        alpha, r2, mode1 = fit_alpha(S[S > 1e-10])

        if alpha is None:
            signals.append(None)
            continue

        signals.append({
            'alpha': alpha,
            'r2': r2,
            'mode1': mode1,
            'spy': spy[t + 1] if t + 1 < len(spy) else spy[t],
        })

    return signals, spy


def simulate_strategy(signals, spy, params):
    """Run a parameterized strategy on precomputed signals."""
    alpha_long = params['alpha_long']       # go long when α < this
    alpha_short = params['alpha_short']     # go short when α > this
    mom_window = params['mom_window']       # minutes for momentum calc
    mom_thresh = params['mom_thresh']       # momentum threshold (%)
    min_hold = params['min_hold']           # minimum hold time
    mode1_min = params['mode1_min']         # minimum mode1% to trade
    use_momentum = params['use_momentum']   # combine α with momentum direction

    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0
    hold_time = 0
    total_pnl = 0
    n_trades = 0
    wins = 0

    for t, sig in enumerate(signals):
        if sig is None:
            hold_time += 1
            continue

        alpha = sig['alpha']
        mode1 = sig['mode1']
        spy_now = sig['spy']

        # Momentum: price change over last N minutes
        if use_momentum and t >= mom_window:
            spy_past = None
            for back in range(t - mom_window, t):
                if signals[back] is not None:
                    spy_past = signals[back]['spy']
                    break
            if spy_past and spy_past > 0:
                momentum = (spy_now - spy_past) / spy_past * 100
            else:
                momentum = 0.0
        else:
            momentum = 0.0

        # Skip if mode1 too low (not enough consensus to trade)
        if mode1 < mode1_min:
            signal = 0
        elif use_momentum:
            # α determines WHETHER to trade, momentum determines DIRECTION
            if alpha > alpha_short and abs(momentum) > mom_thresh:
                signal = 1 if momentum > 0 else -1  # follow the consensus
            elif alpha < alpha_long:
                signal = 1 if momentum > 0 else -1  # low α = contrarian? or follow?
            else:
                signal = 0
        else:
            # Pure α strategy (no momentum)
            if alpha > alpha_short:
                signal = -1
            elif alpha < alpha_long:
                signal = 1
            else:
                signal = 0

        # Minimum hold time
        hold_time += 1
        if signal != position and hold_time >= min_hold:
            # Close
            if position != 0:
                pnl = (spy_now - entry_price) / entry_price * 100 * position
                total_pnl += pnl
                n_trades += 1
                if pnl > 0:
                    wins += 1

            # Open
            if signal != 0:
                position = signal
                entry_price = spy_now
                hold_time = 0
            else:
                position = 0

    # Close at EOD
    if position != 0 and len(spy) > 0:
        final = spy[-1]
        pnl = (final - entry_price) / entry_price * 100 * position
        total_pnl += pnl
        n_trades += 1
        if pnl > 0:
            wins += 1

    win_rate = wins / max(n_trades, 1) * 100
    return total_pnl, n_trades, win_rate


def optimize_day(signals, spy, spy_change, day_label):
    """Sweep parameter space for one day."""
    # Parameter grid
    alpha_longs = [1.0, 1.1, 1.17, 1.25, 1.3, 1.38]
    alpha_shorts = [1.3, 1.38, 1.5, 1.62, 1.8, 2.0, 2.5]
    mom_windows = [5, 10, 15, 30]
    mom_thresholds = [0.0, 0.02, 0.05, 0.1, 0.15]
    min_holds = [1, 3, 5, 10]
    mode1_mins = [0, 50, 70, 85]
    use_momentums = [True, False]

    best = None
    best_pnl = -999
    n_tested = 0

    for al, ash, mw, mt, mh, m1, um in product(
        alpha_longs, alpha_shorts, mom_windows, mom_thresholds,
        min_holds, mode1_mins, use_momentums
    ):
        if al >= ash:
            continue  # invalid: long threshold must be below short threshold

        params = {
            'alpha_long': al, 'alpha_short': ash,
            'mom_window': mw, 'mom_thresh': mt,
            'min_hold': mh, 'mode1_min': m1,
            'use_momentum': um,
        }

        pnl, n_trades, win_rate = simulate_strategy(signals, spy, params)
        n_tested += 1

        if pnl > best_pnl:
            best_pnl = pnl
            best = {**params, 'pnl': pnl, 'n_trades': n_trades, 'win_rate': win_rate}

    return best, n_tested


def main():
    import yfinance as yf

    print("=" * 80)
    print("STRATEGY OPTIMIZER: Finding optimal α rules per day")
    print("=" * 80)

    data = yf.download(SYMBOLS, period='5d', interval='1m', progress=False)
    closes = data['Close']
    dates = sorted(set(closes.index.date))

    day_results = []
    day_signals_cache = {}

    # Phase 1: Optimize each day independently
    print(f"\n{'='*70}")
    print("PHASE 1: PER-DAY OPTIMIZATION")
    print(f"{'='*70}")

    for day_date in dates:
        mask = closes.index.date == day_date
        day_data = closes[mask].dropna(axis=1, how='all').ffill()
        if day_data.shape[0] < 60 or 'SPY' not in day_data.columns:
            continue

        spy_open = day_data['SPY'].iloc[0]
        spy_close = day_data['SPY'].iloc[-1]
        spy_change = (spy_close - spy_open) / spy_open * 100

        print(f"\n  {day_date} — SPY {spy_change:+.3f}%")
        signals, spy = precompute_signals(day_data)
        day_signals_cache[str(day_date)] = (signals, spy, spy_change)

        best, n_tested = optimize_day(signals, spy, spy_change, str(day_date))
        print(f"    Tested {n_tested:,} parameter combos")
        print(f"    Best: P&L={best['pnl']:+.3f}% ({best['n_trades']} trades, {best['win_rate']:.0f}% win)")
        print(f"      α_long<{best['alpha_long']:.2f} α_short>{best['alpha_short']:.2f} "
              f"mom_win={best['mom_window']} mom_thresh={best['mom_thresh']:.2f} "
              f"hold={best['min_hold']} mode1>{best['mode1_min']}% "
              f"use_mom={best['use_momentum']}")
        print(f"    vs B&H: {best['pnl'] - spy_change:+.3f}% edge")

        day_results.append({
            'date': str(day_date),
            'spy_change': spy_change,
            'best_params': best,
        })

    # Phase 2: Find convergent rules
    print(f"\n{'='*70}")
    print("PHASE 2: CONVERGENCE — Rules that work across ALL days")
    print(f"{'='*70}")

    # Test each parameter combo across all days simultaneously
    alpha_longs = [1.0, 1.1, 1.17, 1.25, 1.3, 1.38]
    alpha_shorts = [1.3, 1.38, 1.5, 1.62, 1.8, 2.0, 2.5]
    mom_windows = [5, 10, 15, 30]
    mom_thresholds = [0.0, 0.02, 0.05, 0.1, 0.15]
    min_holds = [1, 3, 5, 10]
    mode1_mins = [0, 50, 70, 85]
    use_momentums = [True, False]

    best_total = None
    best_total_pnl = -999
    best_sharpe = None
    best_sharpe_val = -999
    n_tested = 0

    # Also track top 10
    top_results = []

    for al, ash, mw, mt, mh, m1, um in product(
        alpha_longs, alpha_shorts, mom_windows, mom_thresholds,
        min_holds, mode1_mins, use_momentums
    ):
        if al >= ash:
            continue

        params = {
            'alpha_long': al, 'alpha_short': ash,
            'mom_window': mw, 'mom_thresh': mt,
            'min_hold': mh, 'mode1_min': m1,
            'use_momentum': um,
        }

        daily_pnls = []
        total_pnl = 0
        total_trades = 0

        for day_key, (signals, spy, spy_change) in day_signals_cache.items():
            pnl, n_trades, win_rate = simulate_strategy(signals, spy, params)
            daily_pnls.append(pnl)
            total_pnl += pnl
            total_trades += n_trades

        n_tested += 1

        # Sharpe-like metric: mean daily P&L / std (consistency matters)
        if len(daily_pnls) > 1:
            mean_pnl = np.mean(daily_pnls)
            std_pnl = np.std(daily_pnls)
            sharpe = mean_pnl / max(std_pnl, 0.001) * sqrt(252)  # annualized
        else:
            sharpe = 0

        # Track best by total P&L
        if total_pnl > best_total_pnl:
            best_total_pnl = total_pnl
            best_total = {**params, 'total_pnl': total_pnl, 'daily_pnls': daily_pnls,
                         'sharpe': sharpe, 'total_trades': total_trades}

        # Track best by Sharpe
        if sharpe > best_sharpe_val and total_trades >= len(day_signals_cache):
            best_sharpe_val = sharpe
            best_sharpe = {**params, 'total_pnl': total_pnl, 'daily_pnls': daily_pnls,
                          'sharpe': sharpe, 'total_trades': total_trades}

        # Track top 10
        top_results.append({**params, 'total_pnl': total_pnl, 'sharpe': sharpe,
                           'daily_pnls': daily_pnls, 'total_trades': total_trades})

    top_results.sort(key=lambda x: -x['total_pnl'])
    top_results = top_results[:20]

    print(f"\n  Tested {n_tested:,} parameter combos across {len(day_signals_cache)} days")

    total_bh = sum(v[2] for v in day_signals_cache.values())
    print(f"\n  Buy-and-hold total: {total_bh:+.3f}%")

    print(f"\n  --- Best by TOTAL P&L ---")
    if best_total:
        print(f"    Total P&L: {best_total['total_pnl']:+.3f}% ({best_total['total_trades']} trades)")
        print(f"    Edge vs B&H: {best_total['total_pnl'] - total_bh:+.3f}%")
        print(f"    Sharpe: {best_total['sharpe']:.2f}")
        print(f"    Daily: {['%+.3f%%' % p for p in best_total['daily_pnls']]}")
        print(f"    Rules: α_long<{best_total['alpha_long']:.2f} α_short>{best_total['alpha_short']:.2f} "
              f"mom={best_total['mom_window']}min@{best_total['mom_thresh']:.2f}% "
              f"hold≥{best_total['min_hold']} mode1>{best_total['mode1_min']}% "
              f"use_mom={best_total['use_momentum']}")

    print(f"\n  --- Best by SHARPE (consistency) ---")
    if best_sharpe:
        print(f"    Total P&L: {best_sharpe['total_pnl']:+.3f}% ({best_sharpe['total_trades']} trades)")
        print(f"    Edge vs B&H: {best_sharpe['total_pnl'] - total_bh:+.3f}%")
        print(f"    Sharpe: {best_sharpe['sharpe']:.2f}")
        print(f"    Daily: {['%+.3f%%' % p for p in best_sharpe['daily_pnls']]}")
        print(f"    Rules: α_long<{best_sharpe['alpha_long']:.2f} α_short>{best_sharpe['alpha_short']:.2f} "
              f"mom={best_sharpe['mom_window']}min@{best_sharpe['mom_thresh']:.2f}% "
              f"hold≥{best_sharpe['min_hold']} mode1>{best_sharpe['mode1_min']}% "
              f"use_mom={best_sharpe['use_momentum']}")

    print(f"\n  --- Top 10 by total P&L ---")
    print(f"  {'#':>3} {'P&L':>8} {'Sharpe':>7} {'Trades':>7} {'α_l':>5} {'α_s':>5} "
          f"{'mom_w':>5} {'mom_t':>5} {'hold':>4} {'m1%':>4} {'mom?':>5}")
    for i, r in enumerate(top_results[:10]):
        print(f"  {i+1:>3} {r['total_pnl']:>+7.3f}% {r['sharpe']:>7.2f} {r['total_trades']:>5} "
              f"{r['alpha_long']:>5.2f} {r['alpha_short']:>5.2f} "
              f"{r['mom_window']:>5} {r['mom_thresh']:>5.2f} {r['min_hold']:>4} "
              f"{r['mode1_min']:>4} {'Y' if r['use_momentum'] else 'N':>5}")

    # Phase 3: Generate the optimal ruleset
    print(f"\n{'='*70}")
    print("PHASE 3: OPTIMAL RULESET")
    print(f"{'='*70}")

    # Look for patterns in top 10
    top10 = top_results[:10]
    print(f"\n  Parameter patterns in top 10:")
    for param in ['alpha_long', 'alpha_short', 'mom_window', 'mom_thresh',
                   'min_hold', 'mode1_min', 'use_momentum']:
        vals = [r[param] for r in top10]
        if isinstance(vals[0], bool):
            true_count = sum(vals)
            print(f"    {param}: True={true_count}/10, False={10-true_count}/10")
        else:
            print(f"    {param}: median={np.median(vals):.2f}, "
                  f"range=[{min(vals):.2f}, {max(vals):.2f}]")

    # Save
    output = {
        'per_day': day_results,
        'best_total': {k: v for k, v in best_total.items() if k != 'daily_pnls'} if best_total else None,
        'best_sharpe': {k: v for k, v in best_sharpe.items() if k != 'daily_pnls'} if best_sharpe else None,
        'top_10': [{k: v for k, v in r.items() if k != 'daily_pnls'} for r in top_results[:10]],
        'buy_and_hold': total_bh,
    }
    with open('finance/strategy-optimization.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to finance/strategy-optimization.json")


if __name__ == '__main__':
    main()
