"""
Event-Driven Stream Trader — websocket-powered live trading.

Streams real-time prices for all assets. When the regime shifts
or allocation changes by >5%, triggers a rebalance.

Architecture:
  1. WebSocket streams quotes for Mag 7 + leaders + defensives
  2. Rolling signal computation on every tick
  3. Regime classifier runs continuously
  4. When allocation shifts > threshold -> rebalance
  5. Trade log + state persistence

Usage:
    python finance/stream_trader.py                  # dry run
    python finance/stream_trader.py --execute        # sandbox live
    python finance/stream_trader.py --execute --live # production
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from tastytrade import Session, Account, DXLinkStreamer
from tastytrade.dxfeed import Quote, Trade
from tastytrade.order import (
    NewOrder, OrderAction, OrderTimeInForce, OrderType,
)
from tastytrade.instruments import Equity

load_dotenv(Path(__file__).parent / '.env')

DATA_DIR = Path(__file__).parent / 'data'
LOG_DIR = Path(__file__).parent / 'trade_logs'
LOG_DIR.mkdir(exist_ok=True)

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']

LEADERS = ['ARKK', 'SMH', 'KWEB', 'FXI', 'HYG', 'LQD', 'UUP']
DEFENSIVES = ['USO', 'XLE', 'XLU', 'XLP', 'GLD', 'SHY', 'XLK', 'XLV',
              'EFA', 'XLF', 'EEM', 'IWM', 'XLB', 'SLV', 'TLT']

# Can't stream ^VIX directly — use VIX futures or UVXY as proxy
VIX_PROXY = 'UVXY'

ALL_STREAM = list(set(TECH7 + LEADERS + DEFENSIVES + [VIX_PROXY]))

REGIME_ASSETS = {
    'NORMAL':          {'tech_pct': 0.90, 'other': ['XLK', 'XLP', 'HYG']},
    'RISK_ON':         {'tech_pct': 0.50, 'other': ['USO', 'XLE', 'XLU', 'GLD']},
    'FEAR':            {'tech_pct': 0.60, 'other': ['SHY', 'HYG', 'XLV', 'GLD']},
    'CRISIS':          {'tech_pct': 0.95, 'other': ['HYG', 'XLU']},
    'INFLATION':       {'tech_pct': 0.30, 'other': ['EFA', 'XLF', 'EEM', 'IWM', 'XLB', 'GLD']},
    'RECESSION_RISK':  {'tech_pct': 0.10, 'other': ['USO', 'SLV', 'XLE', 'GLD']},
    'UNKNOWN':         {'tech_pct': 0.60, 'other': ['GLD', 'SHY', 'XLU']},
}

REBALANCE_THRESHOLD = 0.05  # 5% allocation change triggers rebalance
MIN_REBALANCE_INTERVAL = 300  # seconds between rebalances


class SignalEngine:
    """Maintains rolling price history and computes signals in real-time."""

    def __init__(self, historical_closes):
        """Initialize with historical daily closes for lookback."""
        self.daily_closes = historical_closes.copy()
        self.live_prices = {}  # current live prices
        self.last_regime = 'UNKNOWN'
        self.last_allocation = {}
        self.last_rebalance_time = 0

    def update_price(self, symbol, price):
        """Update live price for a symbol."""
        self.live_prices[symbol] = price

    def get_price(self, symbol):
        """Get best available price: live > last daily close."""
        if symbol in self.live_prices:
            return self.live_prices[symbol]
        if symbol in self.daily_closes.columns:
            s = self.daily_closes[symbol].dropna()
            if len(s) > 0:
                return float(s.iloc[-1])
        return None

    def safe_mom(self, symbol, lookback_days):
        """Compute momentum using daily history + current live price."""
        now = self.get_price(symbol)
        if now is None or now <= 0:
            return None
        if symbol in self.daily_closes.columns:
            hist = self.daily_closes[symbol].dropna()
            if len(hist) >= lookback_days:
                prev = float(hist.iloc[-(lookback_days)])
                if prev > 0:
                    return (now - prev) / prev
        return None

    def compute_regime(self):
        """Classify current regime from live + historical data."""
        # VIX proxy (UVXY)
        vix_mom = self.safe_mom(VIX_PROXY, 5)
        # HYG/TLT credit spread
        hyg_p = self.get_price('HYG')
        tlt_p = self.get_price('TLT')
        gld_p = self.get_price('GLD')

        # Z-score against 50-day history
        def z_score(symbol):
            p = self.get_price(symbol)
            if p is None or symbol not in self.daily_closes.columns:
                return 0
            hist = self.daily_closes[symbol].dropna().iloc[-50:]
            if len(hist) < 20:
                return 0
            return (p - hist.mean()) / (hist.std() + 1e-8)

        vix_z = z_score(VIX_PROXY)
        gld_z = z_score('GLD')

        # Credit z
        credit_z = 0
        if hyg_p and tlt_p and tlt_p > 0:
            credit_now = hyg_p / tlt_p
            if 'HYG' in self.daily_closes.columns and 'TLT' in self.daily_closes.columns:
                hyg_h = self.daily_closes['HYG'].dropna().iloc[-50:]
                tlt_h = self.daily_closes['TLT'].dropna().iloc[-50:]
                if len(hyg_h) >= 20 and len(tlt_h) >= 20:
                    credit_h = hyg_h / (tlt_h + 1e-8)
                    credit_z = (credit_now - credit_h.mean()) / (credit_h.std() + 1e-8)

        # Yield curve z
        shy_p = self.get_price('SHY')
        curve_z = 0
        if tlt_p and shy_p and shy_p > 0:
            curve_now = tlt_p / shy_p
            if 'TLT' in self.daily_closes.columns and 'SHY' in self.daily_closes.columns:
                tlt_h = self.daily_closes['TLT'].dropna().iloc[-50:]
                shy_h = self.daily_closes['SHY'].dropna().iloc[-50:]
                if len(tlt_h) >= 20 and len(shy_h) >= 20:
                    curve_h = tlt_h / (shy_h + 1e-8)
                    curve_z = (curve_now - curve_h.mean()) / (curve_h.std() + 1e-8)

        # Classify
        if vix_z > 1.0 and credit_z < -0.5:
            return 'CRISIS'
        if vix_z > 0.5:
            return 'FEAR'
        if credit_z > 0.5 and vix_z < 0:
            return 'RISK_ON'
        if gld_z > 1.0:
            return 'INFLATION'
        if curve_z < -1.0:
            return 'RECESSION_RISK'
        return 'NORMAL'

    def compute_leader_score(self):
        """Compute leading indicator conviction from live prices."""
        signals = []

        configs = [
            ('ARKK', 5, -1, 4.0), (VIX_PROXY, 5, 1, 3.5), ('HYG', 5, -1, 5.0),
            ('KWEB', 5, -1, 3.0), ('FXI', 5, -1, 3.0), ('SMH', 5, -1, 3.0),
            ('LQD', 2, 1, 3.5), ('UUP', 5, 1, 2.5),
        ]

        for sym, lookback, direction, weight in configs:
            m = self.safe_mom(sym, lookback)
            if m is not None:
                signals.append((sym, m * direction, weight))

        if len(signals) >= 3:
            total_w = sum(w for _, _, w in signals)
            score = sum(s * w for _, s, w in signals) / total_w
            return score, signals
        return 0, signals

    def compute_allocation(self):
        """Full allocation computation from live state."""
        regime = self.compute_regime()
        config = REGIME_ASSETS.get(regime, REGIME_ASSETS['NORMAL'])
        tech_pct = config['tech_pct']
        other_assets = config['other']

        # Leader conviction
        leader_score, leader_signals = self.compute_leader_score()
        if len(leader_signals) >= 3:
            adj = np.clip(leader_score * 5.0, -0.40, +0.40)
            tech_pct = np.clip(tech_pct + adj, 0.10, 0.95)

        # Singularity check
        singularity = False
        moms_20d = []
        for sym in TECH7:
            m = self.safe_mom(sym, 20)
            if m is not None:
                moms_20d.append(m)
        if len(moms_20d) >= 7:
            n_pos = sum(1 for m in moms_20d if m > 0)
            avg = np.mean(moms_20d)
            if n_pos >= 6 and avg > 0.05:
                singularity = True
                tech_pct = 0.95
            elif n_pos >= 5 and avg > 0.02:
                tech_pct = max(tech_pct, 0.80)

        other_pct = 1.0 - tech_pct

        # Bear switch (unless singularity)
        if not singularity:
            moms_50d = [self.safe_mom(sym, 50) for sym in TECH7]
            moms_50d = [m for m in moms_50d if m is not None]
            if len(moms_50d) >= 5:
                avg_50 = np.mean(moms_50d)
                n_above = sum(1 for m in moms_50d if m > 0)
                if avg_50 < -0.05 and n_above <= 2:
                    tech_pct *= 0.3
                    other_pct = 1.0 - tech_pct
                elif avg_50 < 0 and n_above <= 4:
                    tech_pct *= 0.6
                    other_pct = 1.0 - tech_pct

        # Tech momentum weights
        mom_scores = np.zeros(7)
        for i, sym in enumerate(TECH7):
            for lb in [10, 20, 50, 100]:
                m = self.safe_mom(sym, lb)
                if m is not None:
                    w = {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35}[lb]
                    mom_scores[i] += m * 10 * w  # scale up

        mom_exp = np.exp(mom_scores)
        tech_weights = mom_exp / mom_exp.sum()
        tech_weights = np.clip(tech_weights, 0.05, 0.35)
        tech_weights /= tech_weights.sum()

        alloc = {sym: float(w * tech_pct) for sym, w in zip(TECH7, tech_weights)}

        # Defensive weights
        if other_pct > 0.01:
            def_moms = []
            for sym in other_assets:
                m = self.safe_mom(sym, 50)
                def_moms.append(max(m, 0) if m is not None else 0)
            def_moms = np.array(def_moms)
            if def_moms.sum() > 0:
                def_w = def_moms / def_moms.sum() * other_pct
            else:
                def_w = np.ones(len(other_assets)) / len(other_assets) * other_pct
            for i, sym in enumerate(other_assets):
                alloc[sym] = alloc.get(sym, 0) + float(def_w[i])

        alloc = {k: v for k, v in alloc.items() if v > 0.005}
        total = sum(alloc.values())
        if total > 0:
            alloc = {k: v / total for k, v in alloc.items()}

        return {
            'allocation': alloc,
            'regime': regime,
            'singularity': singularity,
            'tech_pct': tech_pct,
            'leader_score': leader_score,
        }

    def allocation_changed(self, new_alloc):
        """Check if allocation shifted enough to warrant rebalance."""
        if not self.last_allocation:
            return True
        all_syms = set(list(new_alloc.keys()) + list(self.last_allocation.keys()))
        max_diff = max(abs(new_alloc.get(s, 0) - self.last_allocation.get(s, 0))
                       for s in all_syms)
        return max_diff > REBALANCE_THRESHOLD


async def rebalance(session, account, engine, capital, dry_run=True):
    """Execute a rebalance based on current signals."""
    result = engine.compute_allocation()
    alloc = result['allocation']

    now = datetime.now()
    print(f"\n{'='*60}")
    print(f"  REBALANCE TRIGGERED — {now.strftime('%H:%M:%S')}")
    print(f"  Regime: {result['regime']}"
          f"{'  *** SINGULARITY ***' if result['singularity'] else ''}")
    print(f"  Tech: {result['tech_pct']*100:.0f}%  Leader: {result['leader_score']:+.3f}")

    # Get current positions
    positions = await account.get_positions(session)
    current = {p.symbol: int(p.quantity) for p in positions}

    # Compute target shares
    targets = {}
    for sym, pct in alloc.items():
        if sym.startswith('^') or sym.endswith('-USD'):
            continue
        price = engine.get_price(sym)
        if price and price > 0:
            shares = int(capital * pct / price)
            if shares > 0:
                targets[sym] = shares

    # Show changes
    all_syms = sorted(set(list(current.keys()) + list(targets.keys())))
    orders = []
    for sym in all_syms:
        cur = current.get(sym, 0)
        tgt = targets.get(sym, 0)
        diff = tgt - cur
        if abs(diff) < 1:
            continue
        action = 'BUY' if diff > 0 else 'SELL'
        price = engine.get_price(sym)
        print(f"    {sym:<6} {cur:>5} -> {tgt:>5}  {action} {abs(diff):>5}  "
              f"(${abs(diff) * (price or 0):,.0f})")
        orders.append((sym, diff))

    if not orders:
        print("    No changes needed")
        return

    # Execute
    if not dry_run:
        for sym, diff in orders:
            try:
                action = OrderAction.BUY_TO_OPEN if diff > 0 else OrderAction.SELL_TO_CLOSE
                equity = Equity.get_equity(session, sym)
                leg = equity.build_leg(abs(diff), action)
                order = NewOrder(
                    time_in_force=OrderTimeInForce.DAY,
                    order_type=OrderType.MARKET,
                    legs=[leg],
                )
                resp = await account.place_order(session, order)
                print(f"    -> {sym} order placed: {resp}")
            except Exception as e:
                print(f"    -> {sym} ERROR: {e}")

    # Update state
    engine.last_allocation = alloc
    engine.last_regime = result['regime']
    engine.last_rebalance_time = asyncio.get_event_loop().time()

    # Log
    log_entry = {
        'timestamp': now.isoformat(),
        'regime': result['regime'],
        'singularity': result['singularity'],
        'leader_score': result['leader_score'],
        'tech_pct': result['tech_pct'],
        'allocation': alloc,
        'orders': [(s, d) for s, d in orders],
        'dry_run': dry_run,
    }
    log_path = LOG_DIR / f"stream_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(log_entry, f, indent=2, default=str)


async def main():
    parser = argparse.ArgumentParser(description='Stream Trader')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()

    is_sandbox = not args.live
    dry_run = not args.execute

    print("=" * 60)
    print(f"STREAM TRADER — Event-Driven")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE EXECUTION'}")
    print(f"  Env:  {'SANDBOX' if is_sandbox else 'PRODUCTION'}")
    print(f"  Rebalance threshold: {REBALANCE_THRESHOLD*100:.0f}% allocation change")
    print("=" * 60)

    # Load historical data for lookback
    print("\n  Loading historical prices...")
    hist = yf.download(ALL_STREAM + ['^VIX', 'TLT', 'SHY'], period='120d',
                       interval='1d', auto_adjust=True)
    hist_closes = hist['Close'].dropna(how='all')
    print(f"  {hist_closes.shape[0]} days, {hist_closes.shape[1]} assets")

    # Connect
    print("\n  Connecting to TastyTrade...")
    session = Session(
        provider_secret=os.environ['TASTYTRADE_CLIENT_SECRET'],
        refresh_token=os.environ['TASTYTRADE_REFRESH_TOKEN'],
        is_test=is_sandbox,
    )
    accounts = await Account.get(session)
    account = accounts[0]
    bal = await account.get_balances(session)
    capital = float(bal.net_liquidating_value)
    print(f"  Account: {account.account_number}")
    print(f"  Capital: ${capital:,.2f}")

    # Initialize signal engine
    engine = SignalEngine(hist_closes)

    # Initial rebalance
    await rebalance(session, account, engine, capital, dry_run)

    # Start streaming
    print(f"\n  Starting live stream for {len(ALL_STREAM)} symbols...")
    print(f"  Watching for regime changes and {REBALANCE_THRESHOLD*100:.0f}% allocation shifts...")
    print(f"  Press Ctrl+C to stop\n")

    tick_count = 0
    last_status = 0

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, ALL_STREAM)

        while True:
            try:
                quote = await asyncio.wait_for(streamer.get_event(Quote), timeout=30)

                sym = quote.event_symbol
                bid = float(quote.bid_price) if quote.bid_price else 0
                ask = float(quote.ask_price) if quote.ask_price else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else None

                if mid and mid > 0:
                    engine.update_price(sym, float(mid))
                    tick_count += 1

                    # Check for rebalance every 100 ticks
                    if tick_count % 100 == 0:
                        now = asyncio.get_event_loop().time()
                        if now - engine.last_rebalance_time > MIN_REBALANCE_INTERVAL:
                            result = engine.compute_allocation()
                            new_regime = result['regime']

                            # Regime change OR allocation shift -> rebalance
                            regime_changed = new_regime != engine.last_regime
                            alloc_changed = engine.allocation_changed(result['allocation'])

                            if regime_changed or alloc_changed:
                                reason = f"REGIME: {engine.last_regime}->{new_regime}" if regime_changed else "ALLOCATION SHIFT"
                                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {reason}")
                                bal = await account.get_balances(session)
                                capital = float(bal.net_liquidating_value)
                                await rebalance(session, account, engine, capital, dry_run)

                    # Status update every 60 seconds
                    now_ts = asyncio.get_event_loop().time()
                    if now_ts - last_status > 60:
                        result = engine.compute_allocation()
                        n_prices = len(engine.live_prices)
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                              f"ticks={tick_count} prices={n_prices}/{len(ALL_STREAM)} "
                              f"regime={result['regime']} tech={result['tech_pct']*100:.0f}% "
                              f"leader={result['leader_score']:+.3f}"
                              f"{'  SINGULARITY' if result['singularity'] else ''}")
                        last_status = now_ts

            except asyncio.TimeoutError:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] No data (market closed?)")
            except KeyboardInterrupt:
                print("\n  Shutting down...")
                break
            except Exception as e:
                print(f"  ERROR: {e}")
                await asyncio.sleep(5)

    await session._client.aclose()
    print("  Done.")


if __name__ == '__main__':
    asyncio.run(main())
