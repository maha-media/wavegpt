"""
Real-time spectral α monitor for financial markets.

Computes rolling spectral exponent α from stock correlation matrices
at multiple timescales. Detects regime changes when α crosses
φ-harmonic thresholds.

Modes:
  --live       Connect to live data feed (requires API key)
  --backtest   Run on historical minute data from yfinance
  --simulate   Generate synthetic data with regime switches

Signals:
  α > φ^(2/3) ≈ 1.38  →  STRESS (correlation spike, crisis onset)
  α > φ^(1/1) ≈ 1.62  →  CRISIS (full spectral collapse)
  α < φ^(1/7) ≈ 1.07  →  CALM (maximum dispersion, stock-picking regime)
  Δα/Δt > threshold    →  REGIME CHANGE (transition between harmonics)

Usage:
    python finance/realtime_alpha_monitor.py --backtest --window 30 --stride 5
    python finance/realtime_alpha_monitor.py --simulate --crisis-at 500
"""

import argparse
import json
import sys
import time as time_module
from math import sqrt, log
from datetime import datetime, timedelta
from collections import deque

import numpy as np
from scipy.optimize import curve_fit

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

# Regime thresholds (φ-harmonic levels)
REGIMES = {
    'DEEP_CALM':  (0.0,            PHI**(1/7)),    # α < 1.07
    'CALM':       (PHI**(1/7),     PHI**(1/3)),    # 1.07 - 1.17
    'NORMAL':     (PHI**(1/3),     PHI**(4/7)),    # 1.17 - 1.32
    'ELEVATED':   (PHI**(4/7),     PHI**(2/3)),    # 1.32 - 1.38
    'STRESS':     (PHI**(2/3),     PHI**(1/1)),    # 1.38 - 1.62
    'CRISIS':     (PHI**(1/1),     float('inf')),  # α > 1.62
}

REGIME_COLORS = {
    'DEEP_CALM': '\033[96m',   # cyan
    'CALM':      '\033[92m',   # green
    'NORMAL':    '\033[97m',   # white
    'ELEVATED':  '\033[93m',   # yellow
    'STRESS':    '\033[91m',   # red
    'CRISIS':    '\033[95m',   # magenta
}
RESET = '\033[0m'

# F/L fraction lookup
FL_FRACS = []
FIB = [1, 1, 2, 3, 5, 8, 13, 21]
LUC = [1, 3, 4, 7, 11, 18, 29, 47]
for f in FIB:
    for l in LUC:
        FL_FRACS.append((f/l, f"{f}/{l}"))
for l1 in LUC:
    for l2 in LUC:
        if l1 != l2:
            FL_FRACS.append((l1/l2, f"L{l1}/L{l2}"))


def bent_power_law(k, A, k0, alpha):
    return A * (k + k0) ** (-alpha)


def fit_alpha(S):
    S = S[S > 1e-10]
    n = len(S)
    if n < 4:
        return None
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
        return None


def classify_regime(alpha):
    for name, (lo, hi) in REGIMES.items():
        if lo <= alpha < hi:
            return name
    return 'CRISIS'


def best_fl(alpha):
    best_err = float('inf')
    best = None
    for p_val, label in FL_FRACS:
        pred = PHI ** p_val
        err = abs(alpha - pred) / alpha * 100
        if err < best_err:
            best_err = err
            best = (label, pred, err)
    return best


class AlphaMonitor:
    """Rolling spectral α computation with regime detection."""

    def __init__(self, n_assets, window=30, alert_threshold=0.15):
        self.n_assets = n_assets
        self.window = window  # number of observations in rolling window
        self.alert_threshold = alert_threshold  # Δα for regime change alert
        self.buffer = deque(maxlen=window)
        self.alpha_history = []
        self.regime_history = []
        self.last_regime = None
        self.alerts = []

    def update(self, returns_row):
        """Add one observation (1D array of returns for all assets)."""
        self.buffer.append(returns_row)

        if len(self.buffer) < max(10, self.window // 2):
            return None  # not enough data yet

        R = np.array(self.buffer)

        # Drop assets with zero variance in window
        var = np.var(R, axis=0)
        valid = var > 1e-12
        if valid.sum() < 5:
            return None

        # Correlation matrix → SVD → fit α
        corr = np.corrcoef(R[:, valid].T)
        corr = np.nan_to_num(corr, nan=0.0)
        S = np.linalg.svd(corr, compute_uv=False)

        result = fit_alpha(S[S > 1e-10])
        if result is None:
            return None

        alpha, r2 = result
        regime = classify_regime(alpha)
        fl = best_fl(alpha)

        # Energy in mode 1
        energy = S ** 2
        mode1_pct = energy[0] / energy.sum() * 100

        state = {
            'alpha': alpha,
            'r2': r2,
            'regime': regime,
            'fl_label': fl[0] if fl else '?',
            'fl_error': fl[2] if fl else 0,
            'mode1_pct': mode1_pct,
            'n_valid': int(valid.sum()),
            'n_obs': len(self.buffer),
        }

        self.alpha_history.append(alpha)
        self.regime_history.append(regime)

        # Detect regime changes
        if self.last_regime and regime != self.last_regime:
            state['alert'] = f"REGIME CHANGE: {self.last_regime} → {regime}"
            self.alerts.append(state['alert'])

        # Detect rapid α changes
        if len(self.alpha_history) >= 3:
            delta = alpha - self.alpha_history[-3]
            if abs(delta) > self.alert_threshold:
                direction = "SPIKE" if delta > 0 else "DROP"
                state['alert'] = state.get('alert', '') + f" α {direction}: Δ={delta:+.3f}"

        self.last_regime = regime
        return state


def print_state(state, t=None):
    """Pretty-print a monitoring state."""
    regime = state['regime']
    color = REGIME_COLORS.get(regime, '')

    t_str = f"t={t:>5}" if t is not None else ""
    alert = f" !! {state['alert']}" if 'alert' in state else ""

    print(f"  {t_str} {color}{regime:<10}{RESET} "
          f"α={state['alpha']:.4f} R²={state['r2']:.3f} "
          f"mode1={state['mode1_pct']:.0f}% "
          f"φ^({state['fl_label']}) err={state['fl_error']:.1f}% "
          f"n={state['n_valid']}{alert}")


def run_backtest(args):
    """Backtest on historical minute-level data."""
    import yfinance as yf

    tickers = ['SPY','QQQ','IWM','DIA',  # indices
               'XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU',  # sectors
               'TLT','HYG','LQD',  # bonds
               'GLD','SLV',  # metals
               'USO','UNG',  # commodities
               'VIX']  # if available

    print(f"Downloading {args.period}d minute data for {len(tickers)} assets...")

    # yfinance allows 7 days of minute data for free
    period = f"{min(args.period, 7)}d"
    data = yf.download(tickers, period=period, interval=args.interval, progress=False)

    if isinstance(data.columns, __import__('pandas').MultiIndex):
        closes = data['Close']
    else:
        closes = data

    returns = closes.pct_change().dropna()
    valid = returns.columns[returns.isna().sum() < len(returns) * 0.3]
    returns = returns[valid].dropna()

    n_assets = returns.shape[1]
    n_obs = len(returns)
    print(f"  Got {n_assets} assets × {n_obs} observations ({args.interval} bars)")

    R = returns.values
    names = [str(c) for c in returns.columns]

    monitor = AlphaMonitor(n_assets, window=args.window, alert_threshold=0.2)

    print(f"\n{'='*80}")
    print(f"BACKTEST: {args.interval} bars, window={args.window}, stride={args.stride}")
    print(f"{'='*80}\n")

    states = []
    for t in range(0, n_obs, args.stride):
        state = monitor.update(R[t])
        if state:
            if t % (args.stride * 10) == 0 or 'alert' in state:
                print_state(state, t)
            states.append({'t': t, **state})

    if states:
        alphas = [s['alpha'] for s in states]
        print(f"\n{'='*70}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(f"  Observations: {n_obs} ({args.interval})")
        print(f"  α measurements: {len(alphas)}")
        print(f"  α range: [{min(alphas):.3f}, {max(alphas):.3f}]")
        print(f"  Mean α: {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")

        fl = best_fl(np.mean(alphas))
        if fl:
            print(f"  Best F/L: φ^({fl[0]}) = {fl[1]:.4f} ({fl[2]:.1f}%)")

        # Regime distribution
        regimes = [s['regime'] for s in states]
        print(f"\n  Regime distribution:")
        for r in REGIMES:
            count = regimes.count(r)
            if count > 0:
                pct = count / len(regimes) * 100
                bar = '#' * int(pct / 2)
                print(f"    {r:<12}: {count:>4} ({pct:>5.1f}%) {bar}")

        # Alerts
        if monitor.alerts:
            print(f"\n  Alerts ({len(monitor.alerts)}):")
            for a in monitor.alerts[:20]:
                print(f"    {a}")

        # Save
        with open(args.output, 'w') as f:
            json.dump(states, f, indent=2, default=str)
        print(f"\n  Saved {len(states)} states to {args.output}")


def run_simulate(args):
    """Simulate market data with regime switches."""
    np.random.seed(42)

    n_assets = 30
    n_obs = args.n_obs
    print(f"Simulating {n_assets} assets × {n_obs} observations")

    # Base correlation structure (3 clusters + market factor)
    cluster_size = n_assets // 3
    base_corr = np.eye(n_assets) * 0.3
    for c in range(3):
        start = c * cluster_size
        end = start + cluster_size
        base_corr[start:end, start:end] += 0.3  # within-cluster
    base_corr += 0.2  # market factor
    np.fill_diagonal(base_corr, 1.0)

    # Cholesky for correlated returns
    L = np.linalg.cholesky(base_corr)

    monitor = AlphaMonitor(n_assets, window=args.window, alert_threshold=0.15)

    print(f"\n{'='*80}")
    print(f"SIMULATION: {n_obs} steps, crisis at t={args.crisis_at}")
    print(f"{'='*80}\n")

    states = []
    for t in range(n_obs):
        # Regime-dependent volatility and correlation
        if args.crisis_at and abs(t - args.crisis_at) < 50:
            # Crisis: high vol, high correlation
            crisis_intensity = max(0, 1 - abs(t - args.crisis_at) / 50)
            vol = 0.02 * (1 + 3 * crisis_intensity)
            extra_corr = crisis_intensity * 0.5
            crisis_corr = base_corr + extra_corr * (1 - np.eye(n_assets))
            np.fill_diagonal(crisis_corr, 1.0)
            crisis_corr = np.clip(crisis_corr, -0.99, 0.99)
            # Ensure positive definite via eigenvalue floor
            eigvals, eigvecs = np.linalg.eigh(crisis_corr)
            eigvals = np.maximum(eigvals, 0.01)
            crisis_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L_crisis = np.linalg.cholesky(crisis_corr)
            returns = vol * (L_crisis @ np.random.randn(n_assets))
        else:
            vol = 0.02
            returns = vol * (L @ np.random.randn(n_assets))

        state = monitor.update(returns)
        if state:
            if t % 50 == 0 or 'alert' in state:
                print_state(state, t)
            states.append({'t': t, **state})

    if states:
        alphas = [s['alpha'] for s in states]
        print(f"\n{'='*70}")
        print("SIMULATION SUMMARY")
        print(f"{'='*70}")
        print(f"  α range: [{min(alphas):.3f}, {max(alphas):.3f}]")
        print(f"  Mean α: {np.mean(alphas):.4f}")

        # Did we detect the crisis?
        if args.crisis_at:
            crisis_window = [s for s in states if abs(s['t'] - args.crisis_at) < 60]
            if crisis_window:
                crisis_alphas = [s['alpha'] for s in crisis_window]
                print(f"\n  Crisis window (t={args.crisis_at}±60):")
                print(f"    Peak α: {max(crisis_alphas):.3f}")
                print(f"    Regime at peak: {crisis_window[np.argmax(crisis_alphas)]['regime']}")

                # Detection lead time
                stress_states = [s for s in states if s['regime'] in ('STRESS', 'CRISIS')
                                and s['t'] < args.crisis_at]
                if stress_states:
                    first_alert = stress_states[0]['t']
                    lead = args.crisis_at - first_alert
                    print(f"    First STRESS signal: t={first_alert} ({lead} steps before crisis peak)")

        # Regime distribution
        regimes = [s['regime'] for s in states]
        print(f"\n  Regime distribution:")
        for r in REGIMES:
            count = regimes.count(r)
            if count > 0:
                pct = count / len(regimes) * 100
                print(f"    {r:<12}: {count:>4} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode')

    bt = sub.add_parser('backtest')
    bt.add_argument('--period', type=int, default=5, help='Days of minute data')
    bt.add_argument('--interval', default='5m', help='Bar interval (1m, 5m, 15m)')
    bt.add_argument('--window', type=int, default=30, help='Rolling window size')
    bt.add_argument('--stride', type=int, default=5, help='Steps between measurements')
    bt.add_argument('--output', default='finance/backtest-results.json')

    sim = sub.add_parser('simulate')
    sim.add_argument('--n-obs', type=int, default=1000)
    sim.add_argument('--window', type=int, default=30)
    sim.add_argument('--crisis-at', type=int, default=500)

    args = parser.parse_args()

    if args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'simulate':
        run_simulate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
