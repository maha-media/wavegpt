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
from httpx_ws._exceptions import WebSocketDisconnect

from tastytrade import Session, Account, DXLinkStreamer
from tastytrade.dxfeed import Quote, Trade
from tastytrade.order import (
    NewOrder, OrderAction, OrderTimeInForce, OrderType,
)
from tastytrade.instruments import Equity
from decimal import Decimal

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
MAX_ORDER_RETRIES = 3
ORDER_RETRY_DELAY = 2  # seconds, doubles each retry


class LocalPortfolio:
    """Local position/P&L tracking with tax-aware rebalancing.

    Tax logic:
      - Short-term gains (<1yr): taxed at 30%. Don't sell unless signal is strong.
      - Long-term gains (>1yr): taxed at 15%. More flexibility to sell.
      - Losses: actively beneficial. Offset gains dollar-for-dollar, plus
        $3k/yr deductible against ordinary income. Sell aggressively.
      - Wash sale rule: can't buy back substantially identical security
        within 30 days of selling at a loss. Track recent loss sales.
    """

    # Sell threshold multipliers (applied to REBALANCE_THRESHOLD)
    # Higher = stickier (harder to trigger a sell)
    SELL_THRESHOLD = {
        'short_term_gain': 3.0,   # 5% * 3 = 15% drift needed to sell a ST winner
        'long_term_gain':  2.0,   # 5% * 2 = 10% drift needed to sell a LT winner
        'loss':            0.4,   # 5% * 0.4 = 2% drift triggers selling a loser
        'neutral':         1.0,   # 5% standard threshold
    }

    # Proactively harvest losses when position drops this much from cost
    LOSS_HARVEST_THRESHOLD = 0.05  # 5% below cost basis

    SHORT_TERM_DAYS = 365
    WASH_SALE_DAYS = 30

    def __init__(self, starting_capital):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions = {}   # symbol -> {'shares', 'avg_cost', 'buy_date'}
        self.realized_gains = 0.0   # total realized gains (pre-tax)
        self.realized_losses = 0.0  # total realized losses (negative)
        self.wash_sale_blacklist = {}  # symbol -> datetime when loss was sold
        self.tax_events = []  # log of all taxable events

    def fill_order(self, symbol, shares, price, pool='core'):
        """Record a fill. Positive shares = buy, negative = sell."""
        now = datetime.now()
        if shares > 0:
            # Buy
            pos = self.positions.get(symbol, {'shares': 0, 'avg_cost': 0.0, 'buy_date': now, 'pool': pool})
            total_cost = pos['shares'] * pos['avg_cost'] + shares * price
            new_shares = pos['shares'] + shares
            pos['avg_cost'] = total_cost / new_shares if new_shares > 0 else 0
            if pos['shares'] == 0:
                pos['buy_date'] = now  # fresh position
            pos['shares'] = new_shares
            self.positions[symbol] = pos
            self.cash -= shares * price
        elif shares < 0:
            # Sell — record realized P/L
            qty = abs(shares)
            pos = self.positions.get(symbol, {'shares': 0, 'avg_cost': 0.0, 'buy_date': now})
            sell_qty = min(qty, pos['shares'])
            if sell_qty > 0:
                pnl = sell_qty * (price - pos['avg_cost'])
                holding_days = (now - pos.get('buy_date', now)).days
                is_long_term = holding_days >= self.SHORT_TERM_DAYS
                tax_type = 'long_term' if is_long_term else 'short_term'

                if pnl >= 0:
                    self.realized_gains += pnl
                else:
                    self.realized_losses += pnl  # negative
                    self.wash_sale_blacklist[symbol] = now

                self.tax_events.append({
                    'time': now.isoformat(),
                    'symbol': symbol,
                    'shares': sell_qty,
                    'cost': pos['avg_cost'],
                    'price': price,
                    'pnl': pnl,
                    'type': tax_type,
                    'holding_days': holding_days,
                    'pool': self.positions.get(symbol, {}).get('pool', pool),
                })

                pos['shares'] -= sell_qty
                if pos['shares'] == 0:
                    del self.positions[symbol]
                else:
                    self.positions[symbol] = pos
                self.cash += sell_qty * price

    def tax_status(self, symbol, current_price):
        """Classify a position's tax status for rebalance decisions."""
        if symbol not in self.positions:
            return 'neutral'
        pos = self.positions[symbol]
        pnl_pct = (current_price - pos['avg_cost']) / pos['avg_cost'] if pos['avg_cost'] > 0 else 0
        holding_days = (datetime.now() - pos.get('buy_date', datetime.now())).days

        if pnl_pct < 0:
            return 'loss'
        elif holding_days >= self.SHORT_TERM_DAYS:
            return 'long_term_gain'
        else:
            return 'short_term_gain'

    def should_sell(self, symbol, current_alloc_pct, target_alloc_pct, current_price):
        """Tax-aware sell decision. Returns (should_sell, reason)."""
        diff = current_alloc_pct - target_alloc_pct  # positive = overweight = sell candidate
        if diff <= 0:
            return False, None  # underweight, no sell needed

        status = self.tax_status(symbol, current_price)
        threshold = REBALANCE_THRESHOLD * self.SELL_THRESHOLD[status]

        if diff >= threshold:
            return True, f"{status} drift={diff:.1%}>{threshold:.1%}"
        return False, f"TAX HOLD {status} drift={diff:.1%}<{threshold:.1%}"

    def harvest_candidates(self, live_prices):
        """Find positions ripe for tax-loss harvesting."""
        candidates = []
        for sym, pos in self.positions.items():
            price = live_prices.get(sym, pos['avg_cost'])
            pnl_pct = (price - pos['avg_cost']) / pos['avg_cost'] if pos['avg_cost'] > 0 else 0
            if pnl_pct <= -self.LOSS_HARVEST_THRESHOLD:
                loss_amount = pos['shares'] * (price - pos['avg_cost'])
                candidates.append((sym, pnl_pct, loss_amount))
        return candidates

    def is_wash_sale(self, symbol):
        """Check if buying this symbol would trigger a wash sale."""
        if symbol not in self.wash_sale_blacklist:
            return False
        days_since = (datetime.now() - self.wash_sale_blacklist[symbol]).days
        return days_since < self.WASH_SALE_DAYS

    def market_value(self, live_prices):
        """Total portfolio value at current prices."""
        holdings = 0
        for sym, pos in self.positions.items():
            price = live_prices.get(sym, pos['avg_cost'])
            holdings += pos['shares'] * price
        return self.cash + holdings

    def tax_summary(self):
        """Summary of realized tax events."""
        net = self.realized_gains + self.realized_losses
        harvestable = min(abs(self.realized_losses), self.realized_gains) if self.realized_losses < 0 else 0
        excess_loss = max(0, abs(self.realized_losses) - self.realized_gains)
        deductible = min(excess_loss, 3000)
        return (f"Realized: ${self.realized_gains:+,.0f} gains / ${self.realized_losses:+,.0f} losses  "
                f"Net: ${net:+,.0f}  Deductible: ${deductible:,.0f}")

    def summary(self, live_prices):
        """One-line portfolio summary."""
        mv = self.market_value(live_prices)
        pnl = mv - self.starting_capital
        pct = pnl / self.starting_capital * 100
        n = len(self.positions)
        return f"${mv:,.0f} ({pnl:+,.0f} / {pct:+.2f}%) {n} positions ${self.cash:,.0f} cash"

    def detail(self, live_prices):
        """Full position breakdown with tax status."""
        lines = []
        lines.append(f"  {'Symbol':<8} {'Shares':>7} {'AvgCost':>9} {'Price':>9} {'Value':>11} {'P/L':>10} {'Tax':>6}")
        lines.append(f"  {'-'*65}")
        total_val = 0
        total_pnl = 0
        for sym in sorted(self.positions.keys(), key=lambda s: self.positions[s]['shares'] * live_prices.get(s, self.positions[s]['avg_cost']), reverse=True):
            pos = self.positions[sym]
            price = live_prices.get(sym, pos['avg_cost'])
            val = pos['shares'] * price
            cost = pos['shares'] * pos['avg_cost']
            pnl = val - cost
            total_val += val
            total_pnl += pnl
            status = self.tax_status(sym, price)
            tag = {'short_term_gain': 'ST+', 'long_term_gain': 'LT+', 'loss': 'LOSS', 'neutral': '—'}[status]
            lines.append(f"  {sym:<8} {pos['shares']:>7} ${pos['avg_cost']:>8.2f} ${price:>8.2f} ${val:>10,.0f} ${pnl:>+9,.0f} {tag:>6}")
        lines.append(f"  {'-'*65}")
        lines.append(f"  {'TOTAL':<8} {'':>7} {'':>9} {'':>9} ${total_val:>10,.0f} ${total_pnl:>+9,.0f}")
        lines.append(f"  Cash: ${self.cash:,.0f}  |  Portfolio: ${self.cash + total_val:,.0f}  |  P/L: ${total_pnl:>+,.0f}")
        if self.realized_gains != 0 or self.realized_losses != 0:
            lines.append(f"  {self.tax_summary()}")
        return '\n'.join(lines)


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
        """Classify current regime from historical ^VIX + live credit/gold data.

        Uses ^VIX daily closes (from yfinance) for VIX z-score — NOT UVXY,
        which is a leveraged decaying ETF with completely different z-scores.
        """
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

        # Use ^VIX from historical daily data (not UVXY live)
        def vix_z_from_history():
            if '^VIX' not in self.daily_closes.columns:
                return 0
            hist = self.daily_closes['^VIX'].dropna().iloc[-50:]
            if len(hist) < 20:
                return 0
            val = float(hist.iloc[-1])
            return (val - hist.mean()) / (hist.std() + 1e-8)

        vix_z = vix_z_from_history()
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

        # Bear switch (unless singularity) — matches live_trader logic:
        # avg_50 = 50-day momentum, n_above = price above 50-day MA
        if not singularity:
            moms_50d = [self.safe_mom(sym, 50) for sym in TECH7]
            moms_50d = [m for m in moms_50d if m is not None]
            if len(moms_50d) >= 5:
                avg_50 = np.mean(moms_50d)
                # Count stocks above their 50-day moving average
                n_above = 0
                for sym in TECH7:
                    price = self.get_price(sym)
                    if price is not None and sym in self.daily_closes.columns:
                        ma50 = self.daily_closes[sym].dropna().iloc[-50:].mean()
                        if price > ma50:
                            n_above += 1
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


LIVE_STATE_FILE = LOG_DIR / 'live_state.json'


def write_live_state(portfolio, engine, result, tick_count):
    """Write current portfolio state to JSON for dashboard consumption."""
    positions = []
    for sym in sorted(portfolio.positions.keys(),
                      key=lambda s: portfolio.positions[s]['shares'] * engine.live_prices.get(s, portfolio.positions[s]['avg_cost']),
                      reverse=True):
        pos = portfolio.positions[sym]
        price = engine.live_prices.get(sym, pos['avg_cost'])
        val = pos['shares'] * price
        cost = pos['shares'] * pos['avg_cost']
        pnl = val - cost
        status = portfolio.tax_status(sym, price)
        tag = {'short_term_gain': 'ST+', 'long_term_gain': 'LT+', 'loss': 'LOSS', 'neutral': '—'}[status]
        positions.append({
            'symbol': sym,
            'shares': pos['shares'],
            'avg_cost': round(pos['avg_cost'], 2),
            'price': round(price, 2),
            'value': round(val, 2),
            'pnl': round(pnl, 2),
            'tax_status': tag,
        })

    mv = portfolio.market_value(engine.live_prices)
    state = {
        'timestamp': datetime.now().isoformat(),
        'regime': result['regime'],
        'leader_score': round(result['leader_score'], 4),
        'tech_pct': round(result['tech_pct'], 4),
        'portfolio_value': round(mv, 2),
        'cash': round(portfolio.cash, 2),
        'starting_capital': portfolio.starting_capital,
        'pnl': round(mv - portfolio.starting_capital, 2),
        'pnl_pct': round((mv - portfolio.starting_capital) / portfolio.starting_capital * 100, 2),
        'positions': positions,
        'orders': [],
        'ticks': tick_count,
        'prices_connected': len(engine.live_prices),
    }

    # Atomic write
    tmp = LIVE_STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state))
    tmp.rename(LIVE_STATE_FILE)


async def rebalance(session, account, engine, capital, dry_run=True, portfolio=None, is_sandbox=False):
    """Execute a rebalance based on current signals."""
    result = engine.compute_allocation()
    alloc = result['allocation']

    now = datetime.now()
    print(f"\n{'='*60}")
    print(f"  REBALANCE TRIGGERED — {now.strftime('%H:%M:%S')}")
    print(f"  Regime: {result['regime']}"
          f"{'  *** SINGULARITY ***' if result['singularity'] else ''}")
    print(f"  Tech: {result['tech_pct']*100:.0f}%  Leader: {result['leader_score']:+.3f}")

    # Get current positions (from local tracker if available, else broker)
    if portfolio and portfolio.positions:
        current = {sym: pos['shares'] for sym, pos in portfolio.positions.items()}
    else:
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

    # Tax-aware loss harvesting: check for positions to proactively sell
    if portfolio:
        harvest = portfolio.harvest_candidates(engine.live_prices)
        if harvest:
            print(f"\n  TAX-LOSS HARVEST CANDIDATES:")
            for sym, pnl_pct, loss_amt in harvest:
                print(f"    {sym:<6} down {pnl_pct*100:.1f}%  loss=${loss_amt:,.0f}")
                # Force target to 0 for harvest candidates (will be reallocated next cycle)
                if sym not in targets or targets[sym] == current.get(sym, 0):
                    targets[sym] = 0

    # Build orders with tax-aware sell filtering
    all_syms = sorted(set(list(current.keys()) + list(targets.keys())))
    orders = []
    tax_holds = []
    for sym in all_syms:
        cur = current.get(sym, 0)
        tgt = targets.get(sym, 0)
        diff = tgt - cur
        if abs(diff) < 1:
            continue

        price = engine.get_price(sym)

        if diff < 0 and portfolio:
            # Selling — apply tax-aware threshold
            cur_alloc = (cur * (price or 0)) / capital if capital > 0 else 0
            tgt_alloc = (tgt * (price or 0)) / capital if capital > 0 else 0
            should, reason = portfolio.should_sell(sym, cur_alloc, tgt_alloc, price or 0)
            if not should:
                tax_holds.append((sym, reason))
                continue
            # Check wash sale on the buy side (handled separately)

        if diff > 0 and portfolio and portfolio.is_wash_sale(sym):
            print(f"    {sym:<6} WASH SALE BLOCKED — sold at loss within 30d")
            continue

        action = 'BUY' if diff > 0 else 'SELL'
        print(f"    {sym:<6} {cur:>5} -> {tgt:>5}  {action} {abs(diff):>5}  "
              f"(${abs(diff) * (price or 0):,.0f})")
        orders.append((sym, diff))

    if tax_holds:
        print(f"\n  TAX HOLDS (not selling):")
        for sym, reason in tax_holds:
            print(f"    {sym:<6} {reason}")

    if not orders:
        print("    No changes needed")
        return

    # Execute with retry on transient errors
    if not dry_run:
        for sym, diff in orders:
            action = OrderAction.BUY_TO_OPEN if diff > 0 else OrderAction.SELL_TO_CLOSE
            placed = False
            for attempt in range(MAX_ORDER_RETRIES):
                try:
                    equity = await Equity.get(session, [sym])
                    if isinstance(equity, list):
                        equity = equity[0]
                    leg = equity.build_leg(abs(diff), action)
                    # Extended hours (before 9:30 or after 16:00 ET) require
                    # GTC_EXT + limit orders. Regular hours use DAY + market.
                    now_et = datetime.now()  # TODO: proper ET conversion
                    hour = now_et.hour
                    is_extended = hour < 9 or (hour == 9 and now_et.minute < 30) or hour >= 16
                    if is_extended and is_sandbox:
                        # Sandbox can't trade after hours — record locally only
                        print(f"    -> {sym} SANDBOX ext-hours — recording locally")
                        if portfolio:
                            fill_price = engine.get_price(sym) or 0
                            portfolio.fill_order(sym, diff, fill_price)
                        placed = True
                        break
                    if is_extended:
                        limit_price = Decimal(str(round(engine.get_price(sym) or 0, 2)))
                        order_price = -limit_price if diff > 0 else limit_price
                        order = NewOrder(
                            time_in_force=OrderTimeInForce.GTC_EXT,
                            order_type=OrderType.LIMIT,
                            price=order_price,
                            legs=[leg],
                        )
                    else:
                        order = NewOrder(
                            time_in_force=OrderTimeInForce.DAY,
                            order_type=OrderType.MARKET,
                            legs=[leg],
                        )
                    resp = await account.place_order(session, order)
                    print(f"    -> {sym} order placed")
                    # Record fill locally
                    if portfolio:
                        fill_price = engine.get_price(sym) or 0
                        portfolio.fill_order(sym, diff, fill_price)
                    placed = True
                    break
                except Exception as e:
                    delay = ORDER_RETRY_DELAY * (2 ** attempt)
                    if attempt < MAX_ORDER_RETRIES - 1:
                        print(f"    -> {sym} ERROR (attempt {attempt+1}/{MAX_ORDER_RETRIES}): {e}")
                        print(f"       Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"    -> {sym} FAILED after {MAX_ORDER_RETRIES} attempts: {e}")
            if not placed:
                print(f"    -> {sym} SKIPPED — could not place order")

    # Show portfolio after rebalance
    if portfolio:
        print(f"\n  PORTFOLIO:")
        print(portfolio.detail(engine.live_prices))

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
    # ^VIX is critical for regime classification — must be in historical data
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

    # Initialize signal engine and local portfolio tracker
    engine = SignalEngine(hist_closes)
    portfolio = LocalPortfolio(capital)

    # Initial rebalance
    await rebalance(session, account, engine, capital, dry_run, portfolio=portfolio, is_sandbox=is_sandbox)

    # Start streaming with automatic reconnect
    print(f"\n  Starting live stream for {len(ALL_STREAM)} symbols...")
    print(f"  Watching for regime changes and {REBALANCE_THRESHOLD*100:.0f}% allocation shifts...")
    print(f"  Press Ctrl+C to stop\n")

    tick_count = 0
    last_status = 0
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    shutting_down = False

    while not shutting_down:
        try:
            if reconnect_attempts > 0:
                delay = min(5 * (2 ** (reconnect_attempts - 1)), 60)
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Reconnecting (attempt {reconnect_attempts}/{max_reconnect_attempts}) "
                      f"in {delay}s...")
                await asyncio.sleep(delay)
                # Refresh session on reconnect
                session = Session(
                    provider_secret=os.environ['TASTYTRADE_CLIENT_SECRET'],
                    refresh_token=os.environ['TASTYTRADE_REFRESH_TOKEN'],
                    is_test=is_sandbox,
                )
                accounts = await Account.get(session)
                account = accounts[0]

            async with DXLinkStreamer(session) as streamer:
                await streamer.subscribe(Quote, ALL_STREAM)
                reconnect_attempts = 0  # reset on successful connect
                if tick_count > 0:
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Reconnected, resuming stream")

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
                                        await rebalance(session, account, engine, capital, dry_run, portfolio=portfolio, is_sandbox=is_sandbox)

                            # Status update every 60 seconds
                            now_ts = asyncio.get_event_loop().time()
                            if now_ts - last_status > 60:
                                result = engine.compute_allocation()
                                n_prices = len(engine.live_prices)
                                port_str = f"  | {portfolio.summary(engine.live_prices)}" if portfolio.positions else ""
                                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                                      f"ticks={tick_count} prices={n_prices}/{len(ALL_STREAM)} "
                                      f"regime={result['regime']} tech={result['tech_pct']*100:.0f}% "
                                      f"leader={result['leader_score']:+.3f}"
                                      f"{'  SINGULARITY' if result['singularity'] else ''}"
                                      f"{port_str}")
                                last_status = now_ts

                                # Write live state for dashboard
                                write_live_state(portfolio, engine, result, tick_count)

                    except asyncio.TimeoutError:
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] No data (market closed?)")
                    except KeyboardInterrupt:
                        print("\n  Shutting down...")
                        shutting_down = True
                        break

        except KeyboardInterrupt:
            print("\n  Shutting down...")
            shutting_down = True
        except (WebSocketDisconnect, BaseExceptionGroup) as e:
            reconnect_attempts += 1
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] WebSocket disconnected: {e}")
            if reconnect_attempts >= max_reconnect_attempts:
                print(f"  FATAL: {max_reconnect_attempts} reconnect attempts failed, giving up")
                break
        except Exception as e:
            reconnect_attempts += 1
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Stream error: {e}")
            if reconnect_attempts >= max_reconnect_attempts:
                print(f"  FATAL: {max_reconnect_attempts} reconnect attempts failed, giving up")
                break

    await session._client.aclose()
    print("  Done.")


if __name__ == '__main__':
    asyncio.run(main())
