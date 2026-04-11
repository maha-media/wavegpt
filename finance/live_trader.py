"""
Live Trader — executes The Equation on TastyTrade.

Runs daily:
  1. Pull fresh prices for all assets
  2. Compute regime, leading indicators, momentum
  3. Calculate target allocation
  4. Compare to current positions
  5. Place orders to rebalance

Usage:
    python finance/live_trader.py                  # dry run (show what it would do)
    python finance/live_trader.py --execute        # actually place orders
    python finance/live_trader.py --execute --live # production (not sandbox)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from tastytrade import Session, Account
from tastytrade.order import (
    NewOrder, OrderAction, OrderTimeInForce, OrderType, PriceEffect,
)
from tastytrade.instruments import Equity

load_dotenv(Path(__file__).parent / '.env')

DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'training_results'
LOG_DIR = Path(__file__).parent / 'trade_logs'
LOG_DIR.mkdir(exist_ok=True)

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']

REGIME_ASSETS = {
    'NORMAL':          {'tech_pct': 0.90, 'other': ['XLK', 'XLP', 'HYG']},
    'RISK_ON':         {'tech_pct': 0.50, 'other': ['USO', 'XLE', 'XLU', 'GLD']},
    'FEAR':            {'tech_pct': 0.60, 'other': ['SHY', 'HYG', 'XLV', 'GLD']},
    'CRISIS':          {'tech_pct': 0.95, 'other': ['HYG', 'XLU']},
    'INFLATION':       {'tech_pct': 0.30, 'other': ['EFA', 'XLF', 'EEM', 'IWM', 'XLB', 'GLD']},
    'RECESSION_RISK':  {'tech_pct': 0.10, 'other': ['USO', 'SLV', 'XLE', 'GLD']},
    'UNKNOWN':         {'tech_pct': 0.60, 'other': ['GLD', 'SHY', 'XLU']},
}

LEADER_TICKERS = ['ARKK', 'SMH', 'KWEB', 'FXI', 'HYG', 'LQD', 'UUP', 'ETH-USD', '^VIX']

ALL_TICKERS = list(set(
    TECH7 +
    [s for v in REGIME_ASSETS.values() for s in v['other']] +
    LEADER_TICKERS +
    ['^VIX', '^TNX', 'TLT', 'SHY']
))


def fetch_prices(lookback_days=120):
    """Pull fresh daily prices for all assets."""
    print("  Fetching prices...")
    data = yf.download(ALL_TICKERS, period=f'{lookback_days}d', interval='1d', auto_adjust=True)
    closes = data['Close'].dropna(how='all')
    print(f"  Got {len(closes)} days, {closes.shape[1]} assets")
    return closes


def classify_regime(closes, t=-1):
    """Classify current regime from VIX, credit, gold, yield curve."""
    lookback = 50

    def z(series, idx):
        val = series.iloc[idx]
        hist = series.iloc[max(0, idx - lookback):idx]
        if hist.std() > 0 and not np.isnan(val):
            return (val - hist.mean()) / (hist.std() + 1e-8)
        return 0

    vix = closes.get('^VIX', pd.Series(dtype=float))
    tlt = closes.get('TLT', pd.Series(dtype=float))
    shy = closes.get('SHY', pd.Series(dtype=float))
    hyg = closes.get('HYG', pd.Series(dtype=float))
    gld = closes.get('GLD', pd.Series(dtype=float))

    vix_z = z(vix, t) if len(vix) > lookback else 0
    gold_z = z(gld, t) if len(gld) > lookback else 0

    # Credit: HYG/TLT ratio
    if len(hyg) > lookback and len(tlt) > lookback:
        credit = hyg / (tlt + 1e-8)
        credit_z = z(credit, t)
    else:
        credit_z = 0

    # Yield curve: TLT/SHY
    if len(tlt) > lookback and len(shy) > lookback:
        curve = tlt / (shy + 1e-8)
        curve_z = z(curve, t)
    else:
        curve_z = 0

    if vix_z > 1.0 and credit_z < -0.5:
        return 'CRISIS', {'vix_z': vix_z, 'credit_z': credit_z}
    if vix_z > 0.5:
        return 'FEAR', {'vix_z': vix_z, 'credit_z': credit_z}
    if credit_z > 0.5 and vix_z < 0:
        return 'RISK_ON', {'vix_z': vix_z, 'credit_z': credit_z}
    if gold_z > 1.0:
        return 'INFLATION', {'gold_z': gold_z}
    if curve_z < -1.0:
        return 'RECESSION_RISK', {'curve_z': curve_z}
    return 'NORMAL', {'vix_z': vix_z, 'credit_z': credit_z}


def compute_leader_score(closes, t=-1):
    """Compute leading indicator conviction score."""
    def safe_mom(sym, lookback):
        if sym not in closes.columns:
            return None
        s = closes[sym].dropna()
        if len(s) < lookback + 1:
            return None
        now = s.iloc[-1]
        prev = s.iloc[-(lookback + 1)]
        if prev > 0 and not np.isnan(now) and not np.isnan(prev):
            return float((now - prev) / prev)
        return None

    signals = []
    # (name, momentum, weight) — from discovery
    m = safe_mom('ARKK', 5)
    if m is not None: signals.append(('ARKK', -m, 4.0))
    m = safe_mom('^VIX', 5)
    if m is not None: signals.append(('VIX', m, 3.5))
    m = safe_mom('HYG', 5)
    if m is not None: signals.append(('HYG', -m, 5.0))
    m = safe_mom('KWEB', 5)
    if m is not None: signals.append(('KWEB', -m, 3.0))
    m = safe_mom('FXI', 5)
    if m is not None: signals.append(('FXI', -m, 3.0))
    m = safe_mom('SMH', 5)
    if m is not None: signals.append(('SMH', -m, 3.0))
    m = safe_mom('LQD', 2)
    if m is not None: signals.append(('LQD', m, 3.5))
    m = safe_mom('UUP', 5)
    if m is not None: signals.append(('UUP', m, 2.5))
    m = safe_mom('ETH-USD', 2)
    if m is not None: signals.append(('ETH', -m, 1.0))

    if signals:
        total_w = sum(w for _, _, w in signals)
        score = sum(s * w for _, s, w in signals) / total_w
        return score, signals
    return 0, []


def compute_allocation(closes):
    """Compute full target allocation using The Equation."""
    T = len(closes)

    # 1. Regime
    regime, regime_detail = classify_regime(closes)
    config = REGIME_ASSETS.get(regime, REGIME_ASSETS['NORMAL'])
    tech_pct = config['tech_pct']
    other_assets = config['other']

    # 2. Leading indicator conviction
    leader_score, leader_signals = compute_leader_score(closes)
    if len(leader_signals) >= 3:
        adj = np.clip(leader_score * 5.0, -0.40, +0.40)
        tech_pct = np.clip(tech_pct + adj, 0.10, 0.95)

    # 3. Singularity override
    singularity = False
    if T >= 20:
        n_pos = sum(1 for sym in TECH7 if sym in closes.columns and
                    (closes[sym].iloc[-1] - closes[sym].iloc[-20]) / closes[sym].iloc[-20] > 0)
        avg_mom = np.mean([(closes[sym].iloc[-1] - closes[sym].iloc[-20]) / closes[sym].iloc[-20]
                           for sym in TECH7 if sym in closes.columns])
        if n_pos >= 6 and avg_mom > 0.05:
            singularity = True
            tech_pct = 0.95
        elif n_pos >= 5 and avg_mom > 0.02:
            tech_pct = max(tech_pct, 0.80)

    other_pct = 1.0 - tech_pct

    # 4. Bear market risk switch (unless singularity)
    if T >= 50 and not singularity:
        avg_mom50 = np.mean([(closes[sym].iloc[-1] - closes[sym].iloc[-50]) / closes[sym].iloc[-50]
                             for sym in TECH7 if sym in closes.columns])
        n_above = sum(1 for sym in TECH7 if sym in closes.columns and
                      closes[sym].iloc[-1] > closes[sym].iloc[-50:].mean())
        if avg_mom50 < -0.05 and n_above <= 2:
            tech_pct *= 0.3
            other_pct = 1.0 - tech_pct
        elif avg_mom50 < 0 and n_above <= 4:
            tech_pct *= 0.6
            other_pct = 1.0 - tech_pct

    # 5. Tech momentum weights
    mom_scores = np.zeros(7)
    for i, sym in enumerate(TECH7):
        if sym not in closes.columns:
            continue
        sym_prices = closes[sym].dropna()
        std = sym_prices.pct_change(10).std()
        if np.isnan(std) or std <= 0:
            std = 0.05  # fallback
        for lb in [10, 20, 50, 100]:
            if len(sym_prices) >= lb:
                mom = (sym_prices.iloc[-1] - sym_prices.iloc[-lb]) / sym_prices.iloc[-lb]
                z = mom / (std + 1e-8)
                w = {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35}[lb]
                mom_scores[i] += z * w

    mom_exp = np.exp(mom_scores)
    tech_weights = mom_exp / mom_exp.sum()
    tech_weights = np.clip(tech_weights, 0.05, 0.35)
    tech_weights /= tech_weights.sum()

    # Dip boost
    tech_returns = closes[TECH7].pct_change()
    if T >= 2:
        for i, sym in enumerate(TECH7):
            if sym in tech_returns.columns:
                if tech_returns[sym].iloc[-1] < -0.02:
                    drop = abs(tech_returns[sym].iloc[-1])
                    tech_weights[i] *= (1 + drop * 3)
                    nvda_i = TECH7.index('NVDA')
                    tech_weights[nvda_i] *= (1 + drop * 2)
        tech_weights /= tech_weights.sum()

    # 6. Tech allocation
    alloc = {sym: float(w * tech_pct) for sym, w in zip(TECH7, tech_weights)}

    # 7. Defensive allocation (momentum-weighted)
    if other_pct > 0.01:
        def_moms = []
        for sym in other_assets:
            if sym in closes.columns and T >= 50:
                mom = (closes[sym].iloc[-1] - closes[sym].iloc[-50]) / closes[sym].iloc[-50]
                def_moms.append(max(float(mom), 0) if not np.isnan(mom) else 0)
            else:
                def_moms.append(0)
        def_moms = np.array(def_moms)
        if def_moms.sum() > 0:
            def_w = def_moms / def_moms.sum() * other_pct
        else:
            def_w = np.ones(len(other_assets)) / len(other_assets) * other_pct
        for i, sym in enumerate(other_assets):
            alloc[sym] = alloc.get(sym, 0) + float(def_w[i])

    # Remove tiny allocations
    alloc = {k: v for k, v in alloc.items() if v > 0.005}
    # Normalize
    total = sum(alloc.values())
    if total > 0:
        alloc = {k: v / total for k, v in alloc.items()}

    return {
        'allocation': alloc,
        'regime': regime,
        'regime_detail': regime_detail,
        'singularity': singularity,
        'tech_pct': float(tech_pct),
        'other_pct': float(other_pct),
        'leader_score': float(leader_score),
        'leader_signals': [(n, round(s, 4), w) for n, s, w in leader_signals],
    }


def compute_target_shares(alloc, capital, closes):
    """Convert allocation percentages to share counts."""
    targets = {}
    for sym, pct in alloc.items():
        if sym.startswith('^') or sym.endswith('-USD'):
            continue  # can't trade indices/crypto directly on TT
        if sym in closes.columns:
            price_series = closes[sym].dropna()
            if len(price_series) == 0:
                continue
            price = float(price_series.iloc[-1])
            if price > 0:
                dollar_alloc = capital * pct
                shares = int(dollar_alloc / price)
                if shares > 0:
                    targets[sym] = {
                        'shares': shares,
                        'price': price,
                        'dollar_value': shares * price,
                        'pct': pct,
                    }
    return targets


async def execute_trades(session, account, current_positions, target_shares, dry_run=True):
    """Place orders to rebalance from current to target."""
    orders_placed = []

    # Build current position map
    current = {}
    for p in current_positions:
        current[p.symbol] = int(p.quantity)

    all_symbols = set(list(current.keys()) + list(target_shares.keys()))

    for sym in sorted(all_symbols):
        cur_shares = current.get(sym, 0)
        tgt = target_shares.get(sym, {})
        tgt_shares = tgt.get('shares', 0)
        diff = tgt_shares - cur_shares

        if abs(diff) < 1:
            continue

        action = OrderAction.BUY_TO_OPEN if diff > 0 else OrderAction.SELL_TO_CLOSE
        qty = abs(diff)

        print(f"    {sym:<6} current={cur_shares:>5}  target={tgt_shares:>5}  "
              f"{'BUY' if diff > 0 else 'SELL':>4} {qty}")

        if dry_run:
            orders_placed.append({
                'symbol': sym, 'action': 'BUY' if diff > 0 else 'SELL',
                'quantity': qty, 'dry_run': True,
            })
        else:
            try:
                equity = Equity.get_equity(session, sym)
                leg = equity.build_leg(qty, action)
                order = NewOrder(
                    time_in_force=OrderTimeInForce.DAY,
                    order_type=OrderType.MARKET,
                    legs=[leg],
                )
                resp = await account.place_order(session, order)
                orders_placed.append({
                    'symbol': sym, 'action': 'BUY' if diff > 0 else 'SELL',
                    'quantity': qty, 'order_id': str(resp),
                })
                print(f"      -> Order placed: {resp}")
            except Exception as e:
                print(f"      -> ERROR: {e}")
                orders_placed.append({
                    'symbol': sym, 'action': 'BUY' if diff > 0 else 'SELL',
                    'quantity': qty, 'error': str(e),
                })

    return orders_placed


async def main():
    parser = argparse.ArgumentParser(description='Live Trader')
    parser.add_argument('--execute', action='store_true', help='Actually place orders')
    parser.add_argument('--live', action='store_true', help='Use production (not sandbox)')
    args = parser.parse_args()

    is_sandbox = not args.live
    dry_run = not args.execute

    now = datetime.now()
    print("=" * 70)
    print(f"LIVE TRADER — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'EXECUTING'}")
    print(f"  Environment: {'SANDBOX' if is_sandbox else 'PRODUCTION'}")
    print("=" * 70)

    # Connect to TastyTrade
    print("\n  Connecting to TastyTrade...")
    session = Session(
        provider_secret=os.environ['TASTYTRADE_CLIENT_SECRET'],
        refresh_token=os.environ['TASTYTRADE_REFRESH_TOKEN'],
        is_test=is_sandbox,
    )
    accounts = await Account.get(session)
    if not accounts:
        print("  ERROR: No accounts found")
        return
    account = accounts[0]
    bal = await account.get_balances(session)
    positions = await account.get_positions(session)

    capital = float(bal.net_liquidating_value)
    cash = float(bal.cash_balance)

    print(f"  Account: {account.account_number}")
    print(f"  Capital: ${capital:,.2f}")
    print(f"  Cash:    ${cash:,.2f}")
    print(f"  Positions: {len(positions)}")
    for p in positions:
        print(f"    {p.symbol}: {p.quantity} shares")

    # Fetch prices
    print("\n  Fetching market data...")
    closes = fetch_prices(lookback_days=120)

    # Compute allocation
    print("\n  Computing allocation...")
    result = compute_allocation(closes)
    alloc = result['allocation']

    print(f"\n  REGIME: {result['regime']}")
    if result['singularity']:
        print(f"  *** SINGULARITY MODE ACTIVE ***")
    print(f"  Tech allocation: {result['tech_pct']*100:.0f}%")
    print(f"  Defensive allocation: {result['other_pct']*100:.0f}%")
    print(f"  Leader score: {result['leader_score']:+.4f}")
    for name, score, weight in result['leader_signals']:
        print(f"    {name:<6} signal={score:>+.4f}  weight={weight:.1f}")

    # Target shares
    print(f"\n  TARGET ALLOCATION (${capital:,.0f} capital):")
    targets = compute_target_shares(alloc, capital, closes)

    print(f"  {'Symbol':<8} {'Alloc%':>7} {'Shares':>7} {'$Value':>10} {'Price':>8}")
    print(f"  {'-' * 45}")
    total_value = 0
    for sym in sorted(targets.keys(), key=lambda s: targets[s]['dollar_value'], reverse=True):
        t = targets[sym]
        print(f"  {sym:<8} {t['pct']*100:>6.1f}% {t['shares']:>7} ${t['dollar_value']:>9,.0f} ${t['price']:>7.2f}")
        total_value += t['dollar_value']
    print(f"  {'TOTAL':<8} {'':>7} {'':>7} ${total_value:>9,.0f}")
    print(f"  Cash remaining: ${capital - total_value:,.0f}")

    # Rebalancing orders
    print(f"\n  REBALANCING ORDERS:")
    orders = await execute_trades(session, account, positions, targets, dry_run=dry_run)

    if not orders:
        print("    No trades needed (already at target)")

    # Save trade log
    log_entry = {
        'timestamp': now.isoformat(),
        'mode': 'dry_run' if dry_run else 'live',
        'environment': 'sandbox' if is_sandbox else 'production',
        'capital': capital,
        'regime': result['regime'],
        'singularity': result['singularity'],
        'leader_score': result['leader_score'],
        'tech_pct': result['tech_pct'],
        'allocation': alloc,
        'targets': targets,
        'orders': orders,
    }

    log_path = LOG_DIR / f"trade_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(log_entry, f, indent=2, default=str)
    print(f"\n  Logged: {log_path}")

    await session._client.aclose()


if __name__ == '__main__':
    asyncio.run(main())
