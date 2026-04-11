"""
Live Spectral Alpha Monitor — Tastytrade Edition

Streams real-time quotes from Tastytrade's DXLink feed,
computes rolling spectral exponent α, and classifies market regime.

Usage:
    python finance/live_alpha.py                    # live stream
    python finance/live_alpha.py --test-connection  # just verify API works
    python finance/live_alpha.py --window 30        # custom window size

Requires .env in finance/ with:
    TASTYTRADE_CLIENT_SECRET=...
    TASTYTRADE_REFRESH_TOKEN=...
"""

import asyncio
import argparse
import os
import sys
import time
from math import sqrt
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

# Load .env
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    for line in env_path.read_text().strip().split('\n'):
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

from tastytrade import Session, Account, DXLinkStreamer
from tastytrade.dxfeed import Quote

PHI = (1 + sqrt(5)) / 2
INV_PHI = 1 / PHI

# φ-harmonic regime thresholds
REGIMES = {
    'DEEP_CALM':  (0.0,            PHI**(1/7)),
    'CALM':       (PHI**(1/7),     PHI**(1/3)),
    'NORMAL':     (PHI**(1/3),     PHI**(4/7)),
    'ELEVATED':   (PHI**(4/7),     PHI**(2/3)),
    'STRESS':     (PHI**(2/3),     PHI**(1/1)),
    'CRISIS':     (PHI**(1/1),     float('inf')),
}

REGIME_COLORS = {
    'DEEP_CALM': '\033[96m',
    'CALM':      '\033[92m',
    'NORMAL':    '\033[97m',
    'ELEVATED':  '\033[93m',
    'STRESS':    '\033[91m',
    'CRISIS':    '\033[95m',
}
RESET = '\033[0m'

# ETF basket for spectral analysis
SYMBOLS = [
    'SPY',   # S&P 500
    'QQQ',   # Nasdaq 100
    'IWM',   # Russell 2000
    'DIA',   # Dow 30
    'XLK',   # Tech
    'XLF',   # Financials
    'XLE',   # Energy
    'XLV',   # Healthcare
    'XLI',   # Industrials
    'XLY',   # Consumer Disc
    'XLP',   # Consumer Staples
    'XLU',   # Utilities
    'TLT',   # Long-term bonds
    'HYG',   # High yield
    'GLD',   # Gold
    'SLV',   # Silver
    'USO',   # Oil
]


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


class LiveAlphaMonitor:
    def __init__(self, symbols, window=30):
        self.symbols = symbols
        self.window = window
        self.prices = {s: deque(maxlen=window + 1) for s in symbols}
        self.returns_buffer = deque(maxlen=window)
        self.alpha_history = deque(maxlen=500)
        self.last_regime = None
        self.update_count = 0
        self.last_print = 0

    def update_price(self, symbol, mid_price):
        """Update a single symbol's price."""
        if mid_price is None or mid_price <= 0:
            return
        self.prices[symbol].append(mid_price)

    def compute_returns(self):
        """Compute cross-sectional returns from latest prices."""
        returns = {}
        for sym in self.symbols:
            prices = list(self.prices[sym])
            if len(prices) >= 2:
                ret = (prices[-1] - prices[-2]) / prices[-2]
                returns[sym] = ret
        return returns

    def compute_alpha(self):
        """Compute α from the returns buffer."""
        if len(self.returns_buffer) < max(10, self.window // 2):
            return None

        # Build returns matrix
        syms = [s for s in self.symbols if len(self.prices[s]) >= 2]
        if len(syms) < 5:
            return None

        R = []
        for row in self.returns_buffer:
            r = [row.get(s, 0.0) for s in syms]
            R.append(r)
        R = np.array(R)

        # Drop zero-variance columns
        var = np.var(R, axis=0)
        valid = var > 1e-15
        if valid.sum() < 5:
            return None

        R_valid = R[:, valid]
        n_valid = R_valid.shape[1]

        # Correlation matrix → SVD
        corr = np.corrcoef(R_valid.T)
        corr = np.nan_to_num(corr, nan=0.0)
        S = np.linalg.svd(corr, compute_uv=False)

        alpha, r2 = fit_alpha(S[S > 1e-10])
        if alpha is None:
            return None

        # Energy in mode 1
        energy = S ** 2
        mode1_pct = energy[0] / energy.sum() * 100

        regime = classify_regime(alpha)
        fl = best_fl(alpha)

        state = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'alpha': alpha,
            'r2': r2,
            'regime': regime,
            'mode1_pct': mode1_pct,
            'n_assets': n_valid,
            'n_obs': len(self.returns_buffer),
            'fl': fl,
        }

        # Regime change detection
        if self.last_regime and regime != self.last_regime:
            state['alert'] = f"REGIME: {self.last_regime} → {regime}"

        # Rapid α change
        if len(self.alpha_history) >= 3:
            delta = alpha - self.alpha_history[-3]
            if abs(delta) > 0.15:
                direction = "↑" if delta > 0 else "↓"
                state['delta_alert'] = f"α{direction}{abs(delta):.3f}"

        self.alpha_history.append(alpha)
        self.last_regime = regime
        return state

    def print_state(self, state):
        """Pretty print current state."""
        regime = state['regime']
        color = REGIME_COLORS.get(regime, '')
        fl_label, fl_pred, fl_err = state['fl']

        alerts = ''
        if 'alert' in state:
            alerts += f" !! {state['alert']}"
        if 'delta_alert' in state:
            alerts += f" {state['delta_alert']}"

        print(f"  {state['time']} {color}{regime:<10}{RESET} "
              f"α={state['alpha']:.4f} R²={state['r2']:.3f} "
              f"mode1={state['mode1_pct']:.0f}% "
              f"φ^({fl_label})={fl_pred:.3f}({fl_err:.1f}%) "
              f"n={state['n_assets']}/{state['n_obs']}"
              f"{alerts}")


async def run_live(args):
    """Main live streaming loop."""
    print("=" * 70)
    print("SPECTRAL ALPHA MONITOR — LIVE")
    print("=" * 70)

    # Connect
    secret = os.environ.get('TASTYTRADE_CLIENT_SECRET', '')
    token = os.environ.get('TASTYTRADE_REFRESH_TOKEN', '')
    is_sandbox = os.environ.get('TASTYTRADE_SANDBOX', 'true').lower() == 'true'

    if not secret or not token:
        print("ERROR: Set TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN")
        sys.exit(1)

    print(f"  Connecting to {'sandbox' if is_sandbox else 'PRODUCTION'}...")
    session = Session(provider_secret=secret, refresh_token=token, is_test=is_sandbox)
    print(f"  Session OK")

    monitor = LiveAlphaMonitor(SYMBOLS, window=args.window)

    print(f"  Symbols: {len(SYMBOLS)}")
    print(f"  Window: {args.window}")
    print(f"\n  Regime thresholds:")
    for name, (lo, hi) in REGIMES.items():
        hi_str = f"{hi:.3f}" if hi < 100 else "∞"
        color = REGIME_COLORS.get(name, '')
        print(f"    {color}{name:<12}{RESET} α ∈ [{lo:.3f}, {hi_str})")
    print()

    # Stream quotes
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, SYMBOLS)
        print(f"  Subscribed to {len(SYMBOLS)} symbols. Streaming...\n")

        tick_count = 0
        while True:
            try:
                quote = await asyncio.wait_for(
                    streamer.get_event(Quote), timeout=30
                )
            except asyncio.TimeoutError:
                print("  [timeout — no quotes in 30s, market closed?]")
                continue

            sym = quote.event_symbol
            if quote.bid_price and quote.ask_price:
                mid = (quote.bid_price + quote.ask_price) / 2
                monitor.update_price(sym, mid)

            tick_count += 1

            # Every N ticks, compute returns and α
            if tick_count % (len(SYMBOLS) * args.update_interval) == 0:
                returns = monitor.compute_returns()
                if returns:
                    monitor.returns_buffer.append(returns)

                state = monitor.compute_alpha()
                if state:
                    monitor.print_state(state)


async def test_connection(args):
    """Just test that the API works."""
    print("Testing Tastytrade connection...")

    secret = os.environ.get('TASTYTRADE_CLIENT_SECRET', '')
    token = os.environ.get('TASTYTRADE_REFRESH_TOKEN', '')
    is_sandbox = os.environ.get('TASTYTRADE_SANDBOX', 'true').lower() == 'true'

    session = Session(provider_secret=secret, refresh_token=token, is_test=is_sandbox)
    print(f"  Session: OK (sandbox={is_sandbox})")

    accounts = await Account.get(session)
    print(f"  Accounts: {len(accounts)}")
    for a in accounts:
        print(f"    {a.account_number}")

    # Test streaming
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, ['SPY', 'QQQ'])
        print(f"  Streaming: subscribed to SPY, QQQ")
        for i in range(5):
            try:
                quote = await asyncio.wait_for(streamer.get_event(Quote), timeout=10)
                print(f"    {quote.event_symbol}: bid={quote.bid_price} ask={quote.ask_price}")
            except asyncio.TimeoutError:
                print(f"    [timeout — market may be closed]")
                break

    print("\n  All tests passed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-connection', action='store_true')
    parser.add_argument('--window', type=int, default=30,
                        help='Rolling window for correlation (in quote rounds)')
    parser.add_argument('--update-interval', type=int, default=1,
                        help='Update α every N quote rounds')
    args = parser.parse_args()

    if args.test_connection:
        asyncio.run(test_connection(args))
    else:
        asyncio.run(run_live(args))


if __name__ == '__main__':
    main()
