"""
Multi-Instance Runner — run N strategy variants simultaneously.

Each instance gets a unique config (different weights, thresholds, etc.)
and trades independently on the same sandbox account with a tagged prefix
so we can track which algo made which trade.

Shares one data stream, runs separate signal engines.

Usage:
    python finance/multi_runner.py                    # 10 variants, dry run
    python finance/multi_runner.py --execute          # sandbox live
    python finance/multi_runner.py --n-variants 5     # fewer variants
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from tastytrade import Session, Account, DXLinkStreamer
from tastytrade.dxfeed import Quote

load_dotenv(Path(__file__).parent / '.env')

DATA_DIR = Path(__file__).parent / 'data'
LOG_DIR = Path(__file__).parent / 'trade_logs'
LOG_DIR.mkdir(exist_ok=True)

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
LEADERS = ['ARKK', 'SMH', 'KWEB', 'FXI', 'HYG', 'LQD', 'UUP']
DEFENSIVES = ['USO', 'XLE', 'XLU', 'XLP', 'GLD', 'SHY', 'XLK', 'XLV',
              'EFA', 'XLF', 'EEM', 'IWM', 'XLB', 'SLV', 'TLT']
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


# --- Strategy Variants ---

def make_variants(n=10):
    """Generate N strategy variants with different parameters."""
    variants = []

    # Variant 0: The Equation (our best)
    variants.append({
        'name': 'THE_EQUATION',
        'leader_scale': 5.0,
        'singularity_threshold': 0.05,
        'singularity_min_stocks': 6,
        'bear_mom_threshold': -0.05,
        'bear_breadth_threshold': 2,
        'rebalance_threshold': 0.05,
        'momentum_windows': {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35},
        'max_stock_weight': 0.35,
        'dip_boost_threshold': -0.02,
        'dip_boost_factor': 3.0,
        'regime_overrides': {},
    })

    # Variant 1: Aggressive momentum (heavier short-term)
    variants.append({
        'name': 'AGGRO_MOM',
        'leader_scale': 5.0,
        'singularity_threshold': 0.03,
        'singularity_min_stocks': 5,
        'bear_mom_threshold': -0.08,
        'bear_breadth_threshold': 1,
        'rebalance_threshold': 0.03,
        'momentum_windows': {10: 0.40, 20: 0.30, 50: 0.20, 100: 0.10},
        'max_stock_weight': 0.45,
        'dip_boost_threshold': -0.015,
        'dip_boost_factor': 5.0,
        'regime_overrides': {},
    })

    # Variant 2: Conservative (heavier long-term, lower max weight)
    variants.append({
        'name': 'CONSERVATIVE',
        'leader_scale': 3.0,
        'singularity_threshold': 0.08,
        'singularity_min_stocks': 7,
        'bear_mom_threshold': -0.03,
        'bear_breadth_threshold': 3,
        'rebalance_threshold': 0.08,
        'momentum_windows': {10: 0.05, 20: 0.10, 50: 0.35, 100: 0.50},
        'max_stock_weight': 0.25,
        'dip_boost_threshold': -0.03,
        'dip_boost_factor': 2.0,
        'regime_overrides': {},
    })

    # Variant 3: Always singularity (degenerate bull)
    variants.append({
        'name': 'DEGEN_BULL',
        'leader_scale': 0.0,
        'singularity_threshold': -999,
        'singularity_min_stocks': 0,
        'bear_mom_threshold': -0.15,
        'bear_breadth_threshold': 0,
        'rebalance_threshold': 0.10,
        'momentum_windows': {10: 0.25, 20: 0.25, 50: 0.25, 100: 0.25},
        'max_stock_weight': 0.50,
        'dip_boost_threshold': -0.01,
        'dip_boost_factor': 4.0,
        'regime_overrides': {'NORMAL': 0.95, 'RISK_ON': 0.95, 'FEAR': 0.90, 'CRISIS': 0.95},
    })

    # Variant 4: Leaders-heavy (trust the leading indicators more)
    variants.append({
        'name': 'LEADER_TRUST',
        'leader_scale': 10.0,
        'singularity_threshold': 0.05,
        'singularity_min_stocks': 6,
        'bear_mom_threshold': -0.05,
        'bear_breadth_threshold': 2,
        'rebalance_threshold': 0.03,
        'momentum_windows': {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35},
        'max_stock_weight': 0.35,
        'dip_boost_threshold': -0.02,
        'dip_boost_factor': 3.0,
        'regime_overrides': {},
    })

    # Variant 5: No leaders (regime + momentum only)
    variants.append({
        'name': 'NO_LEADERS',
        'leader_scale': 0.0,
        'singularity_threshold': 0.05,
        'singularity_min_stocks': 6,
        'bear_mom_threshold': -0.05,
        'bear_breadth_threshold': 2,
        'rebalance_threshold': 0.05,
        'momentum_windows': {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35},
        'max_stock_weight': 0.35,
        'dip_boost_threshold': -0.02,
        'dip_boost_factor': 3.0,
        'regime_overrides': {},
    })

    # Variant 6: NVDA-heavy (overweight the bounce magnet)
    variants.append({
        'name': 'NVDA_HEAVY',
        'leader_scale': 5.0,
        'singularity_threshold': 0.05,
        'singularity_min_stocks': 6,
        'bear_mom_threshold': -0.05,
        'bear_breadth_threshold': 2,
        'rebalance_threshold': 0.05,
        'momentum_windows': {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35},
        'max_stock_weight': 0.50,
        'dip_boost_threshold': -0.01,
        'dip_boost_factor': 8.0,  # massive dip boost
        'regime_overrides': {},
    })

    # Variant 7: Equal weight (baseline — no momentum, just regime)
    variants.append({
        'name': 'EQ_WEIGHT',
        'leader_scale': 5.0,
        'singularity_threshold': 0.05,
        'singularity_min_stocks': 6,
        'bear_mom_threshold': -0.05,
        'bear_breadth_threshold': 2,
        'rebalance_threshold': 0.05,
        'momentum_windows': {},  # empty = equal weight
        'max_stock_weight': 0.20,
        'dip_boost_threshold': -999,
        'dip_boost_factor': 0,
        'regime_overrides': {},
    })

    # Variant 8: Fast rebalance (2% threshold)
    variants.append({
        'name': 'FAST_REBAL',
        'leader_scale': 5.0,
        'singularity_threshold': 0.05,
        'singularity_min_stocks': 6,
        'bear_mom_threshold': -0.05,
        'bear_breadth_threshold': 2,
        'rebalance_threshold': 0.02,
        'momentum_windows': {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35},
        'max_stock_weight': 0.35,
        'dip_boost_threshold': -0.02,
        'dip_boost_factor': 3.0,
        'regime_overrides': {},
    })

    # Variant 9: Slow rebalance (10% threshold)
    variants.append({
        'name': 'SLOW_REBAL',
        'leader_scale': 5.0,
        'singularity_threshold': 0.05,
        'singularity_min_stocks': 6,
        'bear_mom_threshold': -0.05,
        'bear_breadth_threshold': 2,
        'rebalance_threshold': 0.10,
        'momentum_windows': {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35},
        'max_stock_weight': 0.35,
        'dip_boost_threshold': -0.02,
        'dip_boost_factor': 3.0,
        'regime_overrides': {},
    })

    return variants[:n]


class VariantEngine:
    """Signal engine for one strategy variant."""

    def __init__(self, name, config, historical_closes):
        self.name = name
        self.config = config
        self.daily_closes = historical_closes.copy()
        self.live_prices = {}
        self.last_allocation = {}
        self.last_regime = 'UNKNOWN'
        self.last_rebalance_time = 0
        self.virtual_capital = 100000.0
        self.virtual_positions = {}  # {sym: shares}
        self.trade_count = 0
        self.pnl_history = []

    def update_price(self, symbol, price):
        self.live_prices[symbol] = price

    def get_price(self, symbol):
        if symbol in self.live_prices:
            return self.live_prices[symbol]
        if symbol in self.daily_closes.columns:
            s = self.daily_closes[symbol].dropna()
            if len(s) > 0:
                return float(s.iloc[-1])
        return None

    def safe_mom(self, symbol, lookback):
        now = self.get_price(symbol)
        if now is None or now <= 0:
            return None
        if symbol in self.daily_closes.columns:
            hist = self.daily_closes[symbol].dropna()
            if len(hist) >= lookback:
                prev = float(hist.iloc[-lookback])
                if prev > 0:
                    return (now - prev) / prev
        return None

    def compute_allocation(self):
        cfg = self.config

        # Regime
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

        hyg_p, tlt_p = self.get_price('HYG'), self.get_price('TLT')
        credit_z = 0
        if hyg_p and tlt_p and tlt_p > 0:
            if 'HYG' in self.daily_closes.columns and 'TLT' in self.daily_closes.columns:
                h1 = self.daily_closes['HYG'].dropna().iloc[-50:]
                h2 = self.daily_closes['TLT'].dropna().iloc[-50:]
                if len(h1) >= 20 and len(h2) >= 20:
                    cr = h1 / (h2 + 1e-8)
                    credit_z = (hyg_p / tlt_p - cr.mean()) / (cr.std() + 1e-8)

        shy_p = self.get_price('SHY')
        curve_z = 0
        if tlt_p and shy_p and shy_p > 0:
            if 'TLT' in self.daily_closes.columns and 'SHY' in self.daily_closes.columns:
                h1 = self.daily_closes['TLT'].dropna().iloc[-50:]
                h2 = self.daily_closes['SHY'].dropna().iloc[-50:]
                if len(h1) >= 20 and len(h2) >= 20:
                    cv = h1 / (h2 + 1e-8)
                    curve_z = (tlt_p / shy_p - cv.mean()) / (cv.std() + 1e-8)

        if vix_z > 1.0 and credit_z < -0.5: regime = 'CRISIS'
        elif vix_z > 0.5: regime = 'FEAR'
        elif credit_z > 0.5 and vix_z < 0: regime = 'RISK_ON'
        elif gld_z > 1.0: regime = 'INFLATION'
        elif curve_z < -1.0: regime = 'RECESSION_RISK'
        else: regime = 'NORMAL'

        rc = REGIME_ASSETS.get(regime, REGIME_ASSETS['NORMAL'])
        tech_pct = cfg['regime_overrides'].get(regime, rc['tech_pct'])
        other_assets = rc['other']

        # Leaders
        if cfg['leader_scale'] > 0:
            signals = []
            for sym, lb, d, w in [('ARKK',5,-1,4),('UVXY',5,1,3.5),('HYG',5,-1,5),
                                   ('KWEB',5,-1,3),('FXI',5,-1,3),('SMH',5,-1,3),
                                   ('LQD',2,1,3.5),('UUP',5,1,2.5)]:
                m = self.safe_mom(sym, lb)
                if m is not None:
                    signals.append(m * d * w)
            if len(signals) >= 3:
                score = sum(signals) / sum(abs(s) for s in signals) if signals else 0
                adj = np.clip(score * cfg['leader_scale'], -0.40, 0.40)
                tech_pct = np.clip(tech_pct + adj, 0.10, 0.95)

        # Singularity
        singularity = False
        moms = [self.safe_mom(s, 20) for s in TECH7]
        moms = [m for m in moms if m is not None]
        if len(moms) >= cfg['singularity_min_stocks']:
            n_pos = sum(1 for m in moms if m > 0)
            if n_pos >= cfg['singularity_min_stocks'] and np.mean(moms) > cfg['singularity_threshold']:
                singularity = True
                tech_pct = 0.95

        other_pct = 1.0 - tech_pct

        # Bear switch
        if not singularity:
            moms50 = [self.safe_mom(s, 50) for s in TECH7]
            moms50 = [m for m in moms50 if m is not None]
            if len(moms50) >= 5:
                avg = np.mean(moms50)
                n_up = sum(1 for m in moms50 if m > 0)
                if avg < cfg['bear_mom_threshold'] and n_up <= cfg['bear_breadth_threshold']:
                    tech_pct *= 0.3
                    other_pct = 1.0 - tech_pct

        # Tech weights
        if cfg['momentum_windows']:
            scores = np.zeros(7)
            for i, sym in enumerate(TECH7):
                for lb, w in cfg['momentum_windows'].items():
                    m = self.safe_mom(sym, lb)
                    if m is not None:
                        scores[i] += m * 10 * w
            exp = np.exp(scores)
            tw = exp / exp.sum()
            tw = np.clip(tw, 0.05, cfg['max_stock_weight'])
            tw /= tw.sum()
        else:
            tw = np.ones(7) / 7

        # Dip boost
        for i, sym in enumerate(TECH7):
            m = self.safe_mom(sym, 1)
            if m is not None and m < cfg['dip_boost_threshold']:
                drop = abs(m)
                tw[i] *= (1 + drop * cfg['dip_boost_factor'])
                tw[TECH7.index('NVDA')] *= (1 + drop * cfg['dip_boost_factor'] * 0.7)
        tw /= tw.sum()

        alloc = {sym: float(w * tech_pct) for sym, w in zip(TECH7, tw)}

        if other_pct > 0.01:
            dmoms = [max(self.safe_mom(s, 50) or 0, 0) for s in other_assets]
            dmoms = np.array(dmoms)
            if dmoms.sum() > 0:
                dw = dmoms / dmoms.sum() * other_pct
            else:
                dw = np.ones(len(other_assets)) / len(other_assets) * other_pct
            for i, sym in enumerate(other_assets):
                alloc[sym] = alloc.get(sym, 0) + float(dw[i])

        alloc = {k: v for k, v in alloc.items() if v > 0.005}
        total = sum(alloc.values())
        if total > 0:
            alloc = {k: v / total for k, v in alloc.items()}

        return {'allocation': alloc, 'regime': regime, 'singularity': singularity, 'tech_pct': tech_pct}

    def allocation_changed(self, new_alloc):
        if not self.last_allocation:
            return True
        all_s = set(list(new_alloc.keys()) + list(self.last_allocation.keys()))
        return max(abs(new_alloc.get(s, 0) - self.last_allocation.get(s, 0)) for s in all_s) > self.config['rebalance_threshold']

    def virtual_rebalance(self):
        """Rebalance virtual portfolio (no real orders)."""
        result = self.compute_allocation()
        alloc = result['allocation']

        if not self.allocation_changed(alloc):
            return None

        # Compute virtual PnL from current positions
        current_value = 0
        for sym, shares in self.virtual_positions.items():
            p = self.get_price(sym)
            if p:
                current_value += shares * p
        self.virtual_capital = max(self.virtual_capital, current_value) if current_value > 0 else self.virtual_capital

        # New positions
        new_positions = {}
        for sym, pct in alloc.items():
            if sym.startswith('^') or sym.endswith('-USD'):
                continue
            p = self.get_price(sym)
            if p and p > 0:
                shares = int(self.virtual_capital * pct / p)
                if shares > 0:
                    new_positions[sym] = shares

        trades = []
        all_s = set(list(self.virtual_positions.keys()) + list(new_positions.keys()))
        for sym in all_s:
            old = self.virtual_positions.get(sym, 0)
            new = new_positions.get(sym, 0)
            if old != new:
                trades.append((sym, new - old))
                self.trade_count += 1

        self.virtual_positions = new_positions
        self.last_allocation = alloc
        self.last_regime = result['regime']

        return {
            'regime': result['regime'],
            'singularity': result['singularity'],
            'tech_pct': result['tech_pct'],
            'trades': trades,
            'n_positions': len(new_positions),
        }

    def get_portfolio_value(self):
        total = 0
        for sym, shares in self.virtual_positions.items():
            p = self.get_price(sym)
            if p:
                total += shares * p
        return total if total > 0 else self.virtual_capital


async def main():
    parser = argparse.ArgumentParser(description='Multi-Instance Runner')
    parser.add_argument('--n-variants', type=int, default=10)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    dry_run = not args.execute
    n = args.n_variants

    print("=" * 70)
    print(f"MULTI-INSTANCE RUNNER — {n} Strategy Variants")
    print(f"  Mode: {'DRY RUN' if dry_run else 'SANDBOX LIVE'}")
    print("=" * 70)

    # Load historical
    print("\n  Loading historical prices...")
    hist = yf.download(ALL_STREAM + ['^VIX', 'TLT', 'SHY'], period='120d',
                       interval='1d', auto_adjust=True)
    hist_closes = hist['Close'].dropna(how='all')
    print(f"  {hist_closes.shape[0]} days, {hist_closes.shape[1]} assets")

    # Create variants
    variants = make_variants(n)
    engines = []
    for v in variants:
        e = VariantEngine(v['name'], v, hist_closes)
        engines.append(e)
        print(f"  [{v['name']:<15}] rebal={v['rebalance_threshold']:.0%} "
              f"max_wt={v['max_stock_weight']:.0%} leader_scale={v['leader_scale']}")

    # Connect
    print("\n  Connecting to TastyTrade...")
    session = Session(
        provider_secret=os.environ['TASTYTRADE_CLIENT_SECRET'],
        refresh_token=os.environ['TASTYTRADE_REFRESH_TOKEN'],
        is_test=True,
    )
    accounts = await Account.get(session)
    account = accounts[0]
    bal = await account.get_balances(session)
    print(f"  Account: {account.account_number}, ${float(bal.net_liquidating_value):,.0f}")

    # Initial rebalance all variants
    print("\n  Initial allocation:")
    print(f"  {'Variant':<15} {'Regime':<12} {'Tech%':>5} {'Sing':>5} {'Positions':>10}")
    print(f"  {'-' * 50}")
    for e in engines:
        r = e.virtual_rebalance()
        if r:
            print(f"  {e.name:<15} {r['regime']:<12} {r['tech_pct']*100:>4.0f}% "
                  f"{'YES' if r['singularity'] else 'no':>5} {r['n_positions']:>10}")

    # Stream
    print(f"\n  Starting stream for {len(ALL_STREAM)} symbols...")
    print(f"  Status updates every 60 seconds. Ctrl+C to stop.\n")

    tick_count = 0
    last_status = 0
    rebalance_events = []

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
                    for e in engines:
                        e.update_price(sym, mid)
                    tick_count += 1

                    # Check rebalance every 100 ticks
                    if tick_count % 100 == 0:
                        now = asyncio.get_event_loop().time()
                        for e in engines:
                            if now - e.last_rebalance_time > 300:
                                result = e.compute_allocation()
                                if e.allocation_changed(result['allocation']) or result['regime'] != e.last_regime:
                                    r = e.virtual_rebalance()
                                    e.last_rebalance_time = now
                                    if r and r['trades']:
                                        ts = datetime.now().strftime('%H:%M:%S')
                                        print(f"  [{ts}] {e.name:<15} REBALANCE -> "
                                              f"{r['regime']} tech={r['tech_pct']*100:.0f}% "
                                              f"{len(r['trades'])} trades")

                    # Status every 60s
                    now_ts = asyncio.get_event_loop().time()
                    if now_ts - last_status > 60:
                        ts = datetime.now().strftime('%H:%M:%S')
                        print(f"\n  [{ts}] ticks={tick_count} prices={len(engines[0].live_prices)}")
                        print(f"  {'Variant':<15} {'Regime':<12} {'Tech%':>5} {'Value':>12} {'Trades':>7}")
                        print(f"  {'-' * 55}")
                        for e in engines:
                            val = e.get_portfolio_value()
                            r = e.compute_allocation()
                            print(f"  {e.name:<15} {r['regime']:<12} {r['tech_pct']*100:>4.0f}% "
                                  f"${val:>11,.0f} {e.trade_count:>7}")
                        last_status = now_ts

            except asyncio.TimeoutError:
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"  [{ts}] Waiting for data...")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  ERROR: {e}")
                await asyncio.sleep(5)

    # Final scoreboard
    print(f"\n{'=' * 70}")
    print("FINAL SCOREBOARD")
    print(f"{'=' * 70}")
    print(f"  {'Variant':<15} {'Value':>12} {'Return':>8} {'Trades':>7} {'Regime':<12}")
    print(f"  {'-' * 60}")
    for e in sorted(engines, key=lambda x: x.get_portfolio_value(), reverse=True):
        val = e.get_portfolio_value()
        ret = (val - 100000) / 100000 * 100
        print(f"  {e.name:<15} ${val:>11,.0f} {ret:>+7.2f}% {e.trade_count:>7} {e.last_regime:<12}")

    # Save results
    log_path = LOG_DIR / f"multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = []
    for e in engines:
        results.append({
            'name': e.name,
            'value': e.get_portfolio_value(),
            'trades': e.trade_count,
            'regime': e.last_regime,
            'config': e.config,
        })
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {log_path}")

    await session._client.aclose()


if __name__ == '__main__':
    asyncio.run(main())
