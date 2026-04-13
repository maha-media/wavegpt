"""Trading engine — signal computation, regime classification, portfolio tracking, rebalancing.

Extracted from stream_trader.py for use inside the Django dashboard server.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TECH7 = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
LEADERS = ['ARKK', 'SMH', 'KWEB', 'FXI', 'HYG', 'LQD', 'UUP']
DEFENSIVES = ['USO', 'XLE', 'XLU', 'XLP', 'GLD', 'SHY', 'XLK', 'XLV',
              'EFA', 'XLF', 'EEM', 'IWM', 'XLB', 'SLV', 'TLT']
VIX_PROXY = 'UVXY'

ALL_SYMBOLS = sorted(set(TECH7 + LEADERS + DEFENSIVES + [VIX_PROXY]))

REGIME_ASSETS = {
    'NORMAL':          {'tech_pct': 0.90, 'other': ['XLK', 'XLP', 'HYG']},
    'RISK_ON':         {'tech_pct': 0.50, 'other': ['USO', 'XLE', 'XLU', 'GLD']},
    'FEAR':            {'tech_pct': 0.60, 'other': ['SHY', 'HYG', 'XLV', 'GLD']},
    'CRISIS':          {'tech_pct': 0.95, 'other': ['HYG', 'XLU']},
    'INFLATION':       {'tech_pct': 0.30, 'other': ['EFA', 'XLF', 'EEM', 'IWM', 'XLB', 'GLD']},
    'RECESSION_RISK':  {'tech_pct': 0.10, 'other': ['USO', 'SLV', 'XLE', 'GLD']},
    'UNKNOWN':         {'tech_pct': 0.60, 'other': ['GLD', 'SHY', 'XLU']},
}

REBALANCE_THRESHOLD = 0.05
MIN_REBALANCE_INTERVAL = 300
MAX_ORDER_RETRIES = 3
ORDER_RETRY_DELAY = 2

LOG_DIR = Path(__file__).resolve().parent.parent.parent / 'trade_logs'
LOG_DIR.mkdir(exist_ok=True)


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

    SELL_THRESHOLD = {
        'short_term_gain': 3.0,
        'long_term_gain':  2.0,
        'loss':            0.4,
        'neutral':         1.0,
    }

    LOSS_HARVEST_THRESHOLD = 0.05
    SHORT_TERM_DAYS = 365
    WASH_SALE_DAYS = 30

    def __init__(self, starting_capital):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions = {}
        self.realized_gains = 0.0
        self.realized_losses = 0.0
        self.wash_sale_blacklist = {}
        self.tax_events = []

    def fill_order(self, symbol, shares, price, pool='core'):
        now = datetime.now()
        if shares > 0:
            pos = self.positions.get(symbol, {'shares': 0, 'avg_cost': 0.0, 'buy_date': now, 'pool': pool})
            total_cost = pos['shares'] * pos['avg_cost'] + shares * price
            new_shares = pos['shares'] + shares
            pos['avg_cost'] = total_cost / new_shares if new_shares > 0 else 0
            if pos['shares'] == 0:
                pos['buy_date'] = now
            pos['shares'] = new_shares
            self.positions[symbol] = pos
            self.cash -= shares * price
        elif shares < 0:
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
                    self.realized_losses += pnl
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
        diff = current_alloc_pct - target_alloc_pct
        if diff <= 0:
            return False, None
        status = self.tax_status(symbol, current_price)
        threshold = REBALANCE_THRESHOLD * self.SELL_THRESHOLD[status]
        if diff >= threshold:
            return True, f"{status} drift={diff:.1%}>{threshold:.1%}"
        return False, f"TAX HOLD {status} drift={diff:.1%}<{threshold:.1%}"

    def harvest_candidates(self, live_prices):
        candidates = []
        for sym, pos in self.positions.items():
            price = live_prices.get(sym, pos['avg_cost'])
            pnl_pct = (price - pos['avg_cost']) / pos['avg_cost'] if pos['avg_cost'] > 0 else 0
            if pnl_pct <= -self.LOSS_HARVEST_THRESHOLD:
                loss_amount = pos['shares'] * (price - pos['avg_cost'])
                candidates.append((sym, pnl_pct, loss_amount))
        return candidates

    def is_wash_sale(self, symbol):
        if symbol not in self.wash_sale_blacklist:
            return False
        days_since = (datetime.now() - self.wash_sale_blacklist[symbol]).days
        return days_since < self.WASH_SALE_DAYS

    def market_value(self, live_prices):
        holdings = 0
        for sym, pos in self.positions.items():
            price = live_prices.get(sym, pos['avg_cost'])
            holdings += pos['shares'] * price
        return self.cash + holdings

    def tax_summary(self):
        net = self.realized_gains + self.realized_losses
        excess_loss = max(0, abs(self.realized_losses) - self.realized_gains)
        deductible = min(excess_loss, 3000)
        return {
            'realized_gains': round(self.realized_gains, 2),
            'realized_losses': round(self.realized_losses, 2),
            'net': round(net, 2),
            'deductible': round(deductible, 2),
        }

    def snapshot(self, live_prices):
        """Full portfolio snapshot as a dict for API/SSE consumption."""
        mv = self.market_value(live_prices)
        pnl = mv - self.starting_capital
        pct = pnl / self.starting_capital * 100 if self.starting_capital else 0

        positions = []
        for sym in sorted(self.positions.keys(),
                          key=lambda s: self.positions[s]['shares'] * live_prices.get(s, self.positions[s]['avg_cost']),
                          reverse=True):
            pos = self.positions[sym]
            price = live_prices.get(sym, pos['avg_cost'])
            val = pos['shares'] * price
            cost = pos['shares'] * pos['avg_cost']
            p = val - cost
            status = self.tax_status(sym, price)
            tag = {'short_term_gain': 'ST+', 'long_term_gain': 'LT+', 'loss': 'LOSS', 'neutral': '-'}[status]
            positions.append({
                'symbol': sym,
                'shares': pos['shares'],
                'avg_cost': round(pos['avg_cost'], 2),
                'price': round(price, 2),
                'market_value': round(val, 2),
                'pnl': round(p, 2),
                'pnl_pct': round(p / cost * 100, 2) if cost else 0,
                'tax_status': tag,
                'pool': pos.get('pool', 'core'),
            })

        return {
            'portfolio_value': round(mv, 2),
            'cash': round(self.cash, 2),
            'starting_capital': self.starting_capital,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pct, 2),
            'positions': positions,
            'tax': self.tax_summary(),
            'tax_events': self.tax_events[-20:],  # last 20
        }


class SignalEngine:
    """Maintains rolling price history and computes signals in real-time."""

    def __init__(self, historical_closes):
        self.daily_closes = historical_closes.copy()
        self.live_prices = {}
        self.last_regime = 'UNKNOWN'
        self.last_allocation = {}
        self.last_rebalance_time = 0

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

    def safe_mom(self, symbol, lookback_days):
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
        def z_score(symbol):
            p = self.get_price(symbol)
            if p is None or symbol not in self.daily_closes.columns:
                return 0
            hist = self.daily_closes[symbol].dropna().iloc[-50:]
            if len(hist) < 20:
                return 0
            return (p - hist.mean()) / (hist.std() + 1e-8)

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

        hyg_p = self.get_price('HYG')
        tlt_p = self.get_price('TLT')

        credit_z = 0
        if hyg_p and tlt_p and tlt_p > 0:
            credit_now = hyg_p / tlt_p
            if 'HYG' in self.daily_closes.columns and 'TLT' in self.daily_closes.columns:
                hyg_h = self.daily_closes['HYG'].dropna().iloc[-50:]
                tlt_h = self.daily_closes['TLT'].dropna().iloc[-50:]
                if len(hyg_h) >= 20 and len(tlt_h) >= 20:
                    credit_h = hyg_h / (tlt_h + 1e-8)
                    credit_z = (credit_now - credit_h.mean()) / (credit_h.std() + 1e-8)

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
        regime = self.compute_regime()
        config = REGIME_ASSETS.get(regime, REGIME_ASSETS['NORMAL'])
        tech_pct = config['tech_pct']
        other_assets = config['other']

        leader_score, leader_signals = self.compute_leader_score()
        if len(leader_signals) >= 3:
            adj = np.clip(leader_score * 5.0, -0.40, +0.40)
            tech_pct = np.clip(tech_pct + adj, 0.10, 0.95)

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

        if not singularity:
            moms_50d = [self.safe_mom(sym, 50) for sym in TECH7]
            moms_50d = [m for m in moms_50d if m is not None]
            if len(moms_50d) >= 5:
                avg_50 = np.mean(moms_50d)
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

        mom_scores = np.zeros(7)
        for i, sym in enumerate(TECH7):
            for lb in [10, 20, 50, 100]:
                m = self.safe_mom(sym, lb)
                if m is not None:
                    w = {10: 0.15, 20: 0.20, 50: 0.30, 100: 0.35}[lb]
                    mom_scores[i] += m * 10 * w

        mom_exp = np.exp(mom_scores)
        tech_weights = mom_exp / mom_exp.sum()
        tech_weights = np.clip(tech_weights, 0.05, 0.35)
        tech_weights /= tech_weights.sum()

        alloc = {sym: float(w * tech_pct) for sym, w in zip(TECH7, tech_weights)}

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
            'tech_pct': float(tech_pct),
            'leader_score': float(leader_score),
        }

    def allocation_changed(self, new_alloc):
        if not self.last_allocation:
            return True
        all_syms = set(list(new_alloc.keys()) + list(self.last_allocation.keys()))
        max_diff = max(abs(new_alloc.get(s, 0) - self.last_allocation.get(s, 0))
                       for s in all_syms)
        return max_diff > REBALANCE_THRESHOLD

    def snapshot(self):
        """Signal engine state for API/SSE."""
        result = self.compute_allocation()
        return {
            'regime': result['regime'],
            'tech_pct': round(result['tech_pct'] * 100, 1),
            'leader_score': round(result['leader_score'], 4),
            'singularity': result['singularity'],
            'prices_connected': len(self.live_prices),
            'total_symbols': len(ALL_SYMBOLS),
        }


def load_historical_data():
    """Download historical price data for signal computation."""
    logger.info("Loading historical prices...")
    hist = yf.download(ALL_SYMBOLS + ['^VIX', 'TLT', 'SHY'], period='120d',
                       interval='1d', auto_adjust=True)
    hist_closes = hist['Close'].dropna(how='all')
    logger.info(f"Loaded {hist_closes.shape[0]} days, {hist_closes.shape[1]} assets")
    return hist_closes


async def execute_rebalance(session, account, engine, portfolio, capital,
                            dry_run=True, is_sandbox=False, broadcast=None):
    """Execute a rebalance. Returns list of events for SSE broadcast."""
    from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType
    from tastytrade.instruments import Equity

    result = engine.compute_allocation()
    alloc = result['allocation']
    events = []

    now = datetime.now()
    rebalance_event = {
        'timestamp': now.isoformat(),
        'regime': result['regime'],
        'singularity': result['singularity'],
        'tech_pct': result['tech_pct'],
        'leader_score': result['leader_score'],
        'orders': [],
        'tax_holds': [],
        'dry_run': dry_run,
    }

    # Current positions
    if portfolio and portfolio.positions:
        current = {sym: pos['shares'] for sym, pos in portfolio.positions.items()}
    else:
        positions = await account.get_positions(session)
        current = {p.symbol: int(p.quantity) for p in positions}

    # Target shares
    targets = {}
    for sym, pct in alloc.items():
        if sym.startswith('^') or sym.endswith('-USD'):
            continue
        price = engine.get_price(sym)
        if price and price > 0:
            shares = int(capital * pct / price)
            if shares > 0:
                targets[sym] = shares

    # Tax-loss harvesting
    if portfolio:
        harvest = portfolio.harvest_candidates(engine.live_prices)
        for sym, pnl_pct, loss_amt in harvest:
            if sym not in targets or targets[sym] == current.get(sym, 0):
                targets[sym] = 0
            rebalance_event['orders'].append({
                'symbol': sym, 'action': 'HARVEST',
                'pnl_pct': round(pnl_pct * 100, 1),
                'loss': round(loss_amt, 2),
            })

    # Build orders with tax-aware filtering
    all_syms = sorted(set(list(current.keys()) + list(targets.keys())))
    orders = []
    for sym in all_syms:
        cur = current.get(sym, 0)
        tgt = targets.get(sym, 0)
        diff = tgt - cur
        if abs(diff) < 1:
            continue

        price = engine.get_price(sym)

        if diff < 0 and portfolio:
            cur_alloc = (cur * (price or 0)) / capital if capital > 0 else 0
            tgt_alloc = (tgt * (price or 0)) / capital if capital > 0 else 0
            should, reason = portfolio.should_sell(sym, cur_alloc, tgt_alloc, price or 0)
            if not should:
                rebalance_event['tax_holds'].append({'symbol': sym, 'reason': reason})
                continue

        if diff > 0 and portfolio and portfolio.is_wash_sale(sym):
            rebalance_event['tax_holds'].append({'symbol': sym, 'reason': 'WASH SALE BLOCKED'})
            continue

        action_str = 'BUY' if diff > 0 else 'SELL'
        order_info = {
            'symbol': sym, 'action': action_str,
            'from': cur, 'to': tgt, 'diff': abs(diff),
            'value': round(abs(diff) * (price or 0), 2),
        }
        rebalance_event['orders'].append(order_info)
        orders.append((sym, diff))

    logger.info(f"Rebalance: regime={result['regime']} tech={result['tech_pct']*100:.0f}% "
                f"orders={len(orders)} holds={len(rebalance_event['tax_holds'])}")

    # Execute orders
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

                    now_et = datetime.now()
                    hour = now_et.hour
                    is_extended = hour < 9 or (hour == 9 and now_et.minute < 30) or hour >= 16

                    if is_extended and is_sandbox:
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
                    if portfolio:
                        fill_price = engine.get_price(sym) or 0
                        portfolio.fill_order(sym, diff, fill_price)
                    placed = True
                    break
                except Exception as e:
                    if attempt < MAX_ORDER_RETRIES - 1:
                        await asyncio.sleep(ORDER_RETRY_DELAY * (2 ** attempt))
                    else:
                        logger.error(f"Order failed {sym}: {e}")

            order_info = next((o for o in rebalance_event['orders'] if o.get('symbol') == sym and o.get('action') in ('BUY', 'SELL')), None)
            if order_info:
                order_info['status'] = 'filled' if placed else 'failed'

    # Update engine state
    engine.last_allocation = alloc
    engine.last_regime = result['regime']
    engine.last_rebalance_time = time.monotonic()

    # Log to file
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

    # Broadcast rebalance event
    if broadcast:
        await broadcast('rebalance', rebalance_event)
        if portfolio:
            await broadcast('portfolio', portfolio.snapshot(engine.live_prices))

    return rebalance_event
