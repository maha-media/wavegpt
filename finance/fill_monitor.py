"""
Fill Monitor — checks for unfilled GTC LIMIT orders after market open
and proposes chase/wait/skip decisions based on fresh signals.

Run at 9:35 ET (5 min after open):
    python finance/fill_monitor.py                  # dry run
    python finance/fill_monitor.py --execute        # act on proposals
    python finance/fill_monitor.py --execute --live # production

Decision logic (Hybrid C):
    gap < 2% AND strong signal  → CHASE (raise limit to current ask)
    gap 2-5% AND strong signal  → WAIT  (keep GTC, likely fills on pullback)
    gap > 5% OR weak signal     → SKIP  (cancel order, reallocate capital)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from tastytrade import Session, Account
from tastytrade.order import (
    NewOrder, OrderAction, OrderTimeInForce, OrderType,
    OrderStatus, PlacedOrder,
)
from tastytrade.instruments import Equity

# Import signal computation from live_trader
from live_trader import (
    classify_regime, compute_leader_score, compute_allocation,
    TECH7, REGIME_ASSETS, ALL_TICKERS,
)

load_dotenv(Path(__file__).parent / '.env')

LOG_DIR = Path(__file__).parent / 'trade_logs'
LOG_DIR.mkdir(exist_ok=True)

# Unfilled = still waiting to execute
UNFILLED_STATUSES = {OrderStatus.RECEIVED, OrderStatus.LIVE, OrderStatus.ROUTED}

# Thresholds for hybrid C decision
CHASE_GAP_MAX = 0.02     # chase if gap < 2%
WAIT_GAP_MAX = 0.05      # wait if gap 2-5%
STRONG_SIGNAL_MIN = 0.10  # momentum z-score threshold for "strong"


def compute_momentum_score(closes, sym):
    """Compute blended momentum z-score for a single symbol."""
    if sym not in closes.columns:
        return 0.0
    prices = closes[sym].dropna()
    if len(prices) < 20:
        return 0.0
    std = prices.pct_change(10).std()
    if np.isnan(std) or std <= 0:
        std = 0.05
    score = 0.0
    for lb, w in [(10, 0.15), (20, 0.20), (50, 0.30), (100, 0.35)]:
        if len(prices) >= lb:
            mom = (prices.iloc[-1] - prices.iloc[-lb]) / prices.iloc[-lb]
            score += (mom / (std + 1e-8)) * w
    return float(score)


def decide(gap_pct, mom_score, leader_score, regime):
    """
    Hybrid C decision: gap size + signal strength.

    Returns: ('CHASE', reason) | ('WAIT', reason) | ('SKIP', reason)
    """
    strong = mom_score > STRONG_SIGNAL_MIN or leader_score > 0.01

    if gap_pct <= 0:
        # Stock is at or below limit — will fill naturally
        return 'FILLING', f'price at/below limit (gap {gap_pct:+.1%}), order filling'

    if gap_pct < CHASE_GAP_MAX and strong:
        return 'CHASE', (
            f'small gap ({gap_pct:.1%}) + strong signal '
            f'(mom={mom_score:.2f}, leader={leader_score:+.3f})'
        )

    if gap_pct < CHASE_GAP_MAX and not strong:
        return 'WAIT', (
            f'small gap ({gap_pct:.1%}) but weak signal '
            f'(mom={mom_score:.2f}), keep GTC — may fill on pullback'
        )

    if gap_pct < WAIT_GAP_MAX and strong:
        return 'WAIT', (
            f'moderate gap ({gap_pct:.1%}) but strong signal '
            f'(mom={mom_score:.2f}, leader={leader_score:+.3f}), hold GTC'
        )

    if gap_pct < WAIT_GAP_MAX and not strong:
        return 'SKIP', (
            f'moderate gap ({gap_pct:.1%}) + weak signal '
            f'(mom={mom_score:.2f}), cancel and reallocate'
        )

    # gap >= 5%
    if strong:
        return 'WAIT', (
            f'large gap ({gap_pct:.1%}) but strong signal '
            f'(mom={mom_score:.2f}), hold GTC — do not chase'
        )

    return 'SKIP', (
        f'large gap ({gap_pct:.1%}) + weak signal '
        f'(mom={mom_score:.2f}), cancel and reallocate'
    )


async def main():
    parser = argparse.ArgumentParser(description='Fill Monitor')
    parser.add_argument('--execute', action='store_true', help='Act on proposals')
    parser.add_argument('--live', action='store_true', help='Production (not sandbox)')
    parser.add_argument('--test', action='store_true',
                        help='Simulate unfilled orders from last trade log (no TT connection)')
    parser.add_argument('--gap', type=float, default=None,
                        help='Override gap %% for test mode (e.g. --gap 3.0 for 3%% gap-up)')
    args = parser.parse_args()

    is_sandbox = not args.live
    dry_run = not args.execute

    now = datetime.now()
    print("=" * 70)
    print(f"FILL MONITOR — {now.strftime('%Y-%m-%d %H:%M')}")
    if args.test:
        print(f"  Mode: TEST (simulated unfilled orders)")
    else:
        print(f"  Mode: {'DRY RUN' if dry_run else 'EXECUTING'}")
        print(f"  Environment: {'SANDBOX' if is_sandbox else 'PRODUCTION'}")
    print("=" * 70)

    session = None
    account = None

    if args.test:
        # Load last trade log to simulate unfilled orders
        logs = sorted(LOG_DIR.glob('trade_*.json'), reverse=True)
        if not logs:
            print("  No trade logs found. Run live_trader.py first.")
            return
        with open(logs[0]) as f:
            last_trade = json.load(f)
        print(f"  Using trade log: {logs[0].name}")

        # Build simulated unfilled orders from the log's targets
        targets = last_trade.get('targets', {})
        simulated_unfilled = []
        for sym, t in targets.items():
            simulated_unfilled.append({
                'symbol': sym,
                'limit_price': t['price'],
                'shares': t['shares'],
                'order_id': -1,
            })
        print(f"  Simulating {len(simulated_unfilled)} unfilled orders")
    else:
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
        print(f"  Account: {account.account_number}")
        print(f"  Capital: ${float(bal.net_liquidating_value):,.2f}")

        # Get unfilled orders
        print("\n  Checking live orders...")
        live_orders = await account.get_live_orders(session)
        unfilled_orders = [o for o in live_orders if o.status in UNFILLED_STATUSES]

        if not unfilled_orders:
            print("  All orders filled or no pending orders. Nothing to do.")
            await session._client.aclose()
            return

        simulated_unfilled = []
        for o in unfilled_orders:
            simulated_unfilled.append({
                'symbol': o.underlying_symbol,
                'limit_price': abs(float(o.price)),
                'shares': int(o.size),
                'order_id': o.id,
            })

        print(f"  Found {len(simulated_unfilled)} unfilled orders:")
        for u in simulated_unfilled:
            print(f"    {u['symbol']:<6} {u['shares']} shares @ ${u['limit_price']:.2f}")

    # Fetch fresh prices
    print("\n  Fetching fresh market data...")
    closes = yf.download(ALL_TICKERS, period='120d', interval='1d', auto_adjust=True)['Close'].dropna(how='all')

    # Fresh signals
    regime, regime_detail = classify_regime(closes)
    leader_score, leader_signals = compute_leader_score(closes)
    print(f"\n  Current regime: {regime}")
    print(f"  Leader score: {leader_score:+.4f}")

    # Evaluate each unfilled order
    print(f"\n  {'='*60}")
    print(f"  FILL ANALYSIS")
    print(f"  {'='*60}")

    proposals = []
    for entry in simulated_unfilled:
        sym = entry['symbol']
        limit_price = entry['limit_price']
        shares = entry['shares']
        order_id = entry['order_id']

        # Current price (with optional test gap override)
        if sym in closes.columns:
            current_price = float(closes[sym].dropna().iloc[-1])
            if args.test and args.gap is not None:
                current_price = limit_price * (1 + args.gap / 100)
        else:
            print(f"\n  {sym}: no price data, skipping")
            continue

        gap_pct = (current_price - limit_price) / limit_price
        mom_score = compute_momentum_score(closes, sym)
        action, reason = decide(gap_pct, mom_score, leader_score, regime)

        print(f"\n  {sym}:")
        print(f"    Limit: ${limit_price:.2f}  Current: ${current_price:.2f}  Gap: {gap_pct:+.1%}")
        print(f"    Momentum: {mom_score:.2f}  Leader: {leader_score:+.3f}")
        print(f"    → {action}: {reason}")

        proposals.append({
            'symbol': sym,
            'order_id': order_id,
            'limit_price': limit_price,
            'current_price': current_price,
            'gap_pct': gap_pct,
            'mom_score': mom_score,
            'shares': shares,
            'action': action,
            'reason': reason,
        })

    # Summary
    chase = [p for p in proposals if p['action'] == 'CHASE']
    wait = [p for p in proposals if p['action'] == 'WAIT']
    skip = [p for p in proposals if p['action'] == 'SKIP']
    filling = [p for p in proposals if p['action'] == 'FILLING']

    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  FILLING: {len(filling)} (at/below limit, will fill)")
    print(f"  CHASE:   {len(chase)} (raise limit to current price)")
    print(f"  WAIT:    {len(wait)} (keep GTC, hold for pullback)")
    print(f"  SKIP:    {len(skip)} (cancel, reallocate)")

    if skip:
        skip_capital = sum(p['current_price'] * p['shares'] for p in skip)
        print(f"  Capital freed by SKIPs: ${skip_capital:,.0f}")

    # Execute (not in test mode)
    if not dry_run and not args.test:
        print(f"\n  EXECUTING...")

        for p in chase:
            sym, oid, shares = p['symbol'], p['order_id'], p['shares']
            new_limit = Decimal(str(round(p['current_price'], 2)))
            print(f"\n    CHASE {sym}: raising limit ${p['limit_price']:.2f} → ${new_limit}")
            try:
                # Cancel old order, place new one at current price
                await account.delete_order(session, oid)
                equity = await Equity.get(session, [sym])
                if isinstance(equity, list):
                    equity = equity[0]
                leg = equity.build_leg(shares, OrderAction.BUY_TO_OPEN)
                order = NewOrder(
                    time_in_force=OrderTimeInForce.DAY,
                    order_type=OrderType.LIMIT,
                    price=-new_limit,
                    legs=[leg],
                )
                resp = await account.place_order(session, order)
                print(f"      → New DAY LIMIT @ ${new_limit}: {resp.order.status.value}")
                p['new_order_id'] = str(resp)
            except Exception as e:
                print(f"      → ERROR: {e}")
                p['error'] = str(e)

        for p in skip:
            sym, oid = p['symbol'], p['order_id']
            print(f"\n    SKIP {sym}: cancelling order {oid}")
            try:
                await account.delete_order(session, oid)
                print(f"      → Cancelled")
                p['cancelled'] = True
            except Exception as e:
                print(f"      → ERROR: {e}")
                p['error'] = str(e)

        # Reallocate freed capital from SKIPs
        if skip:
            freed = sum(p['limit_price'] * p['shares'] for p in skip if p.get('cancelled'))
            if freed > 0:
                print(f"\n    Reallocating ${freed:,.0f} freed capital...")
                result = compute_allocation(closes)
                alloc = result['allocation']
                # Only allocate to symbols we don't already have orders for
                active_syms = {p['symbol'] for p in proposals if p['action'] != 'SKIP'}
                realloc = {k: v for k, v in alloc.items() if k not in active_syms
                           and not k.startswith('^') and not k.endswith('-USD')}
                if realloc:
                    total = sum(realloc.values())
                    realloc = {k: v / total for k, v in realloc.items()}
                    print(f"    Reallocation targets:")
                    for sym, pct in sorted(realloc.items(), key=lambda x: -x[1]):
                        dollars = freed * pct
                        if sym in closes.columns:
                            price = float(closes[sym].dropna().iloc[-1])
                            new_shares = int(dollars / price)
                            if new_shares > 0:
                                new_limit = Decimal(str(round(price, 2)))
                                print(f"      {sym}: {new_shares} shares @ ${new_limit} (${dollars:,.0f})")
                                try:
                                    equity = await Equity.get(session, [sym])
                                    if isinstance(equity, list):
                                        equity = equity[0]
                                    leg = equity.build_leg(new_shares, OrderAction.BUY_TO_OPEN)
                                    order = NewOrder(
                                        time_in_force=OrderTimeInForce.DAY,
                                        order_type=OrderType.LIMIT,
                                        price=-new_limit,
                                        legs=[leg],
                                    )
                                    resp = await account.place_order(session, order)
                                    print(f"        → Placed: {resp.order.status.value}")
                                except Exception as e:
                                    print(f"        → ERROR: {e}")
    else:
        if chase:
            print(f"\n  Would chase: {', '.join(p['symbol'] for p in chase)}")
        if skip:
            print(f"  Would cancel: {', '.join(p['symbol'] for p in skip)}")
        print(f"\n  Run with --execute to act on these proposals.")

    # Log
    log_entry = {
        'timestamp': now.isoformat(),
        'mode': 'dry_run' if dry_run else 'live',
        'environment': 'sandbox' if is_sandbox else 'production',
        'regime': regime,
        'leader_score': leader_score,
        'proposals': proposals,
    }
    log_path = LOG_DIR / f"fill_monitor_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(log_entry, f, indent=2, default=str)
    print(f"\n  Logged: {log_path}")

    if session:
        await session._client.aclose()


if __name__ == '__main__':
    asyncio.run(main())
