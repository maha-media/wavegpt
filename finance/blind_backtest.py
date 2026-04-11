"""
Blind backtest: Would spectral α have made money this week?

Downloads 1-minute data for the past 5 trading days,
runs the α monitor in replay mode, and simulates a simple strategy:

Strategy:
  - When α rises into STRESS/CRISIS → go short SPY (correlation spike = drop coming)
  - When α falls into CALM/DEEP_CALM → go long SPY (dispersion = stocks find their own way up)
  - When α is NORMAL/ELEVATED → flat (no edge)

Also tracks:
  - α vs SPY price overlay (did α lead price?)
  - Regime change timing vs price reversals
  - Fibonacci retracement levels vs active regime prediction
"""

import json
import sys
from math import sqrt
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

SYMBOLS = ['SPY','QQQ','IWM','DIA','XLK','XLF','XLE','XLV','XLI','XLY',
           'XLP','XLU','TLT','HYG','GLD','SLV','USO']

REGIMES = {
    'DEEP_CALM':  (0.0,            PHI**(1/7)),
    'CALM':       (PHI**(1/7),     PHI**(1/3)),
    'NORMAL':     (PHI**(1/3),     PHI**(4/7)),
    'ELEVATED':   (PHI**(4/7),     PHI**(2/3)),
    'STRESS':     (PHI**(2/3),     PHI**(1/1)),
    'CRISIS':     (PHI**(1/1),     float('inf')),
}

REGIME_COLORS = {
    'DEEP_CALM': '\033[96m', 'CALM': '\033[92m', 'NORMAL': '\033[97m',
    'ELEVATED': '\033[93m', 'STRESS': '\033[91m', 'CRISIS': '\033[95m',
}
RESET = '\033[0m'


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
    except:
        return None, None


def classify_regime(alpha):
    for name, (lo, hi) in REGIMES.items():
        if lo <= alpha < hi:
            return name
    return 'CRISIS'


def best_fl(alpha):
    FIB = [1,1,2,3,5,8,13]
    LUC = [1,3,4,7,11,18,29]
    best_err = float('inf')
    best = ('?', 0, 999)
    for f in FIB:
        for l in LUC:
            pred = PHI ** (f/l)
            err = abs(alpha - pred) / alpha * 100
            if err < best_err:
                best_err = err
                best = (f"{f}/{l}", pred, err)
    return best


def run_day(day_closes, day_label):
    """Run α monitor on one day's minute data and simulate trading."""
    returns = day_closes.pct_change().dropna()
    R = returns.values
    n_minutes = R.shape[0]
    n_assets = R.shape[1]

    spy_idx = list(day_closes.columns).index('SPY') if 'SPY' in day_closes.columns else 0
    spy_prices = day_closes.iloc[:, spy_idx].values

    window = 30  # 30-minute rolling window
    returns_buffer = deque(maxlen=window)

    # Track state over time
    timeline = []
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0
    total_pnl = 0
    trades = []
    prev_regime = None

    for t in range(n_minutes):
        returns_row = {str(day_closes.columns[i]): R[t, i] for i in range(n_assets)}
        returns_buffer.append(returns_row)

        if len(returns_buffer) < 15:
            continue

        # Build correlation matrix
        syms = list(day_closes.columns)
        R_window = np.array([[row.get(str(s), 0.0) for s in syms] for row in returns_buffer])
        var = np.var(R_window, axis=0)
        valid = var > 1e-15
        if valid.sum() < 5:
            continue

        corr = np.corrcoef(R_window[:, valid].T)
        corr = np.nan_to_num(corr, nan=0.0)
        S = np.linalg.svd(corr, compute_uv=False)

        alpha, r2 = fit_alpha(S[S > 1e-10])
        if alpha is None:
            continue

        regime = classify_regime(alpha)
        energy = S ** 2
        mode1 = energy[0] / energy.sum() * 100
        fl = best_fl(alpha)
        spy_now = spy_prices[t + 1] if t + 1 < len(spy_prices) else spy_prices[t]

        # Trading logic
        signal = 0
        if regime in ('STRESS', 'CRISIS'):
            signal = -1  # go short — crisis = everything drops together
        elif regime in ('CALM', 'DEEP_CALM'):
            signal = 1   # go long — calm = stocks drift up independently
        # NORMAL/ELEVATED = flat

        # Execute trades
        if signal != position:
            # Close existing position
            if position != 0:
                pnl = (spy_now - entry_price) * position
                pnl_pct = pnl / entry_price * 100
                total_pnl += pnl_pct
                trades.append({
                    'close_t': t,
                    'close_time': str(day_closes.index[t + 1]) if t + 1 < len(day_closes) else '?',
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry': entry_price,
                    'exit': spy_now,
                    'pnl_pct': pnl_pct,
                    'regime_at_close': regime,
                })

            # Open new position
            if signal != 0:
                position = signal
                entry_price = spy_now
            else:
                position = 0

        # Regime change alert
        regime_change = prev_regime and regime != prev_regime
        prev_regime = regime

        timeline.append({
            't': t,
            'alpha': alpha,
            'r2': r2,
            'regime': regime,
            'mode1': mode1,
            'spy': spy_now,
            'position': position,
            'fl': fl,
            'regime_change': regime_change,
        })

    # Close final position
    if position != 0:
        final_price = spy_prices[-1]
        pnl = (final_price - entry_price) * position
        pnl_pct = pnl / entry_price * 100
        total_pnl += pnl_pct
        trades.append({
            'close_t': n_minutes - 1,
            'direction': 'LONG' if position == 1 else 'SHORT',
            'entry': entry_price,
            'exit': final_price,
            'pnl_pct': pnl_pct,
            'regime_at_close': 'EOD',
        })

    return timeline, trades, total_pnl


def main():
    import yfinance as yf

    print("=" * 80)
    print("BLIND BACKTEST: Spectral α Trading Signals")
    print("=" * 80)

    # Download minute data
    print("\nDownloading 1-minute data...")
    data = yf.download(SYMBOLS, period='5d', interval='1m', progress=False)
    closes = data['Close']

    dates = sorted(set(closes.index.date))
    print(f"Trading days: {dates}")

    all_results = []

    for day_date in dates:
        mask = closes.index.date == day_date
        day_data = closes[mask].dropna(axis=1, how='all').ffill()

        if day_data.shape[0] < 60 or 'SPY' not in day_data.columns:
            continue

        day_label = str(day_date)
        spy_open = day_data['SPY'].iloc[0]
        spy_close = day_data['SPY'].iloc[-1]
        spy_high = day_data['SPY'].max()
        spy_low = day_data['SPY'].min()
        spy_change = (spy_close - spy_open) / spy_open * 100
        spy_range = (spy_high - spy_low) / spy_open * 100

        print(f"\n{'='*70}")
        print(f"  {day_label}")
        print(f"  SPY: open={spy_open:.2f} close={spy_close:.2f} "
              f"({spy_change:+.2f}%) range={spy_range:.2f}%")
        print(f"{'='*70}")

        timeline, trades, total_pnl = run_day(day_data, day_label)

        if not timeline:
            print("  [insufficient data]")
            continue

        # Print regime timeline (sampled)
        alphas = [s['alpha'] for s in timeline]
        regimes = [s['regime'] for s in timeline]

        print(f"\n  α range: [{min(alphas):.3f}, {max(alphas):.3f}]")
        print(f"  Mean α: {np.mean(alphas):.4f}")

        # Regime distribution
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        print(f"  Regimes:")
        for r in REGIMES:
            if r in regime_counts:
                pct = regime_counts[r] / len(regimes) * 100
                color = REGIME_COLORS.get(r, '')
                bar = '#' * int(pct / 3)
                print(f"    {color}{r:<12}{RESET} {regime_counts[r]:>4} ({pct:>5.1f}%) {bar}")

        # Regime changes
        changes = [(s['t'], s['regime']) for s in timeline if s.get('regime_change')]
        if changes:
            print(f"\n  Regime changes ({len(changes)}):")
            for t, r in changes[:15]:
                spy = timeline[timeline.index(next(s for s in timeline if s['t'] == t))]['spy']
                color = REGIME_COLORS.get(r, '')
                print(f"    t={t:>3}min {color}{r:<10}{RESET} SPY={spy:.2f}")

        # Trades
        print(f"\n  Trades ({len(trades)}):")
        for trade in trades:
            direction = trade['direction']
            pnl = trade['pnl_pct']
            marker = '✓' if pnl > 0 else '✗'
            print(f"    {marker} {direction:<5} entry={trade['entry']:.2f} exit={trade['exit']:.2f} "
                  f"pnl={pnl:+.3f}% [{trade.get('regime_at_close', '?')}]")

        print(f"\n  Day P&L: {total_pnl:+.3f}%")
        print(f"  Buy-and-hold: {spy_change:+.3f}%")
        edge = total_pnl - spy_change
        print(f"  Edge vs B&H: {edge:+.3f}%")

        all_results.append({
            'date': day_label,
            'spy_change': spy_change,
            'alpha_pnl': total_pnl,
            'n_trades': len(trades),
            'mean_alpha': np.mean(alphas),
            'regime_changes': len(changes),
        })

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"WEEKLY SUMMARY")
        print(f"{'='*70}")
        print(f"\n  {'Date':<12} {'SPY':>8} {'α P&L':>8} {'Edge':>8} {'Trades':>7} {'Mean α':>8}")
        print(f"  {'-'*55}")

        total_alpha = 0
        total_bh = 0
        for r in all_results:
            edge = r['alpha_pnl'] - r['spy_change']
            print(f"  {r['date']:<12} {r['spy_change']:>+7.3f}% {r['alpha_pnl']:>+7.3f}% "
                  f"{edge:>+7.3f}% {r['n_trades']:>5} {r['mean_alpha']:>8.3f}")
            total_alpha += r['alpha_pnl']
            total_bh += r['spy_change']

        print(f"  {'-'*55}")
        print(f"  {'TOTAL':<12} {total_bh:>+7.3f}% {total_alpha:>+7.3f}% "
              f"{total_alpha - total_bh:>+7.3f}%")

        # Save
        with open('finance/blind-backtest-results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved to finance/blind-backtest-results.json")


if __name__ == '__main__':
    main()
