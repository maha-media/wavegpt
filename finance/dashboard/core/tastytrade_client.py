"""TastyTrade connection with embedded trading engine.

Streams quotes, runs signal engine, triggers rebalances, broadcasts all state via SSE.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from core.trading_engine import (
    SignalEngine, LocalPortfolio, ALL_SYMBOLS, TECH7,
    REBALANCE_THRESHOLD, MIN_REBALANCE_INTERVAL,
    load_historical_data, execute_rebalance,
)

load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')
logger = logging.getLogger(__name__)


class TastyClient:
    """Maintains TastyTrade session, streams quotes, runs trading engine."""

    def __init__(self):
        self.session = None
        self.account = None
        self.live_prices = {}
        self.latest_balances = {}
        self.latest_positions = []
        self.latest_orders = []
        self._streaming = False
        self._subscribers = []

        # Trading engine state
        self.engine = None
        self.portfolio = None
        self.capital = 0
        self.dry_run = True
        self.is_sandbox = True
        self.tick_count = 0
        self._rebalance_log = []  # recent rebalance events

    async def connect(self):
        from tastytrade import Session, Account

        self.is_sandbox = os.environ.get('TASTYTRADE_SANDBOX', 'true').lower() == 'true'
        self.dry_run = os.environ.get('TRADER_DRY_RUN', 'true').lower() == 'true'

        self.session = Session(
            provider_secret=os.environ['TASTYTRADE_CLIENT_SECRET'],
            refresh_token=os.environ['TASTYTRADE_REFRESH_TOKEN'],
            is_test=self.is_sandbox,
        )
        accounts = await Account.get(self.session)
        self.account = accounts[0]

        # Load balances
        bal = await self.account.get_balances(self.session)
        self.capital = float(bal.net_liquidating_value)
        self.latest_balances = {
            "net_liq": self.capital,
            "cash": float(getattr(bal, 'cash_balance', 0) or 0),
            "buying_power": float(getattr(bal, 'derivative_buying_power', 0) or 0),
        }

        # Initialize trading engine
        hist_closes = load_historical_data()
        self.engine = SignalEngine(hist_closes)
        self.portfolio = LocalPortfolio(self.capital)

        logger.info(f"Connected: account={self.account.account_number} "
                    f"capital=${self.capital:,.2f} sandbox={self.is_sandbox} "
                    f"dry_run={self.dry_run}")

    async def refresh_account_data(self):
        """Poll balances, positions, orders via REST."""
        bal = await self.account.get_balances(self.session)
        self.capital = float(bal.net_liquidating_value)
        self.latest_balances = {
            "net_liq": self.capital,
            "cash": float(getattr(bal, 'cash_balance', 0) or 0),
            "buying_power": float(getattr(bal, 'derivative_buying_power', 0) or 0),
        }

        # If we have a local portfolio, overlay its data
        if self.portfolio and self.portfolio.positions:
            self.latest_balances['net_liq'] = self.portfolio.market_value(self.engine.live_prices)
            self.latest_balances['cash'] = self.portfolio.cash
            self.latest_balances['pnl'] = round(
                self.latest_balances['net_liq'] - self.portfolio.starting_capital, 2)
            self.latest_balances['pnl_pct'] = round(
                self.latest_balances['pnl'] / self.portfolio.starting_capital * 100, 2)
            self.latest_balances['starting_capital'] = self.portfolio.starting_capital

        # Broker positions (for reference)
        positions = await self.account.get_positions(self.session)
        self.latest_positions = []
        for p in positions:
            price = self.live_prices.get(p.symbol, {}).get("mid", 0)
            avg_cost = float(getattr(p, 'average_open_price', 0) or 0)
            qty = int(p.quantity)
            mv = qty * price if price else 0
            cost = qty * avg_cost
            pnl = mv - cost if price else 0
            pnl_pct = (pnl / cost * 100) if cost else 0
            self.latest_positions.append({
                "symbol": p.symbol, "qty": qty,
                "avg_cost": round(avg_cost, 2), "price": round(price, 2),
                "market_value": round(mv, 2), "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            })

        try:
            orders = await self.account.get_live_orders(self.session)
            self.latest_orders = []
            for o in orders:
                status = str(getattr(o, 'status', ''))
                if status in ('Received', 'Routed', 'In Flight', 'Live'):
                    leg = o.legs[0] if getattr(o, 'legs', None) else None
                    self.latest_orders.append({
                        "id": str(o.id),
                        "symbol": getattr(leg, 'symbol', '?') if leg else '?',
                        "action": str(getattr(leg, 'action', '?')) if leg else '?',
                        "qty": int(getattr(leg, 'quantity', 0)) if leg else 0,
                        "type": str(getattr(o, 'order_type', '?')),
                        "status": status,
                    })
        except Exception:
            pass

    def update_price(self, symbol, bid, ask):
        mid = (bid + ask) / 2
        self.live_prices[symbol] = {
            "bid": bid, "ask": ask, "mid": mid,
            "last_update": time.time(),
        }
        # Feed into signal engine
        if self.engine:
            self.engine.update_price(symbol, mid)

    def snapshot_positions(self):
        """Portfolio positions — prefer local tracker over broker."""
        if self.portfolio and self.portfolio.positions:
            return self.portfolio.snapshot(self.engine.live_prices)['positions']
        return list(self.latest_positions)

    def snapshot_balances(self):
        """Balances — overlay local portfolio if available."""
        if self.portfolio and self.portfolio.positions and self.engine:
            mv = self.portfolio.market_value(self.engine.live_prices)
            pnl = mv - self.portfolio.starting_capital
            return {
                'net_liq': round(mv, 2),
                'cash': round(self.portfolio.cash, 2),
                'buying_power': round(self.portfolio.cash, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl / self.portfolio.starting_capital * 100, 2) if self.portfolio.starting_capital else 0,
                'starting_capital': self.portfolio.starting_capital,
            }
        return dict(self.latest_balances)

    def snapshot_orders(self):
        return list(self.latest_orders)

    def snapshot_regime(self):
        if not self.engine:
            return {'regime': 'UNKNOWN'}
        return self.engine.snapshot()

    def snapshot_portfolio(self):
        if not self.portfolio or not self.engine:
            return {}
        return self.portfolio.snapshot(self.engine.live_prices)

    def snapshot_rebalances(self):
        return list(self._rebalance_log[-10:])

    def subscribe(self):
        q = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def _broadcast(self, event_type, data):
        for q in list(self._subscribers):
            try:
                q.put_nowait((event_type, data))
            except asyncio.QueueFull:
                pass

    async def _do_rebalance(self):
        """Run a rebalance cycle."""
        try:
            bal = await self.account.get_balances(self.session)
            self.capital = float(bal.net_liquidating_value)
        except Exception:
            pass

        event = await execute_rebalance(
            self.session, self.account, self.engine, self.portfolio,
            self.capital, dry_run=self.dry_run, is_sandbox=self.is_sandbox,
            broadcast=self._broadcast,
        )
        self._rebalance_log.append(event)
        if len(self._rebalance_log) > 50:
            self._rebalance_log = self._rebalance_log[-50:]

    async def run_stream(self):
        """Main loop — stream quotes, run signals, trigger rebalances."""
        from tastytrade import DXLinkStreamer, Session, Account
        from tastytrade.dxfeed import Quote
        from httpx_ws._exceptions import WebSocketDisconnect

        self._streaming = True
        last_positions = 0
        last_balances = 0
        last_orders = 0
        last_heartbeat = 0
        last_regime = 0
        reconnect_attempts = 0
        max_reconnect_attempts = 10

        # Initial rebalance
        await self._do_rebalance()

        while self._streaming:
            try:
                if reconnect_attempts > 0:
                    delay = min(5 * (2 ** (reconnect_attempts - 1)), 60)
                    logger.info(f"Reconnecting (attempt {reconnect_attempts}) in {delay}s...")
                    await asyncio.sleep(delay)
                    self.session = Session(
                        provider_secret=os.environ['TASTYTRADE_CLIENT_SECRET'],
                        refresh_token=os.environ['TASTYTRADE_REFRESH_TOKEN'],
                        is_test=self.is_sandbox,
                    )
                    accounts = await Account.get(self.session)
                    self.account = accounts[0]

                async with DXLinkStreamer(self.session) as streamer:
                    await streamer.subscribe(Quote, ALL_SYMBOLS)
                    reconnect_attempts = 0
                    if self.tick_count > 0:
                        logger.info("Reconnected, resuming stream")

                    while True:
                        try:
                            quote = await asyncio.wait_for(streamer.get_event(Quote), timeout=30)
                            sym = quote.event_symbol
                            bid = float(quote.bid_price) if quote.bid_price else 0
                            ask = float(quote.ask_price) if quote.ask_price else 0

                            if bid > 0 and ask > 0:
                                self.update_price(sym, bid, ask)
                                self.tick_count += 1

                                # Broadcast quote
                                mid = (bid + ask) / 2
                                await self._broadcast("quote", {
                                    "symbol": sym,
                                    "bid": round(bid, 2),
                                    "ask": round(ask, 2),
                                    "mid": round(mid, 2),
                                })

                                # Check for rebalance every 100 ticks
                                if self.tick_count % 100 == 0:
                                    now = time.monotonic()
                                    if now - self.engine.last_rebalance_time > MIN_REBALANCE_INTERVAL:
                                        result = self.engine.compute_allocation()
                                        regime_changed = result['regime'] != self.engine.last_regime
                                        alloc_changed = self.engine.allocation_changed(result['allocation'])

                                        if regime_changed or alloc_changed:
                                            logger.info(f"Rebalance trigger: "
                                                        f"{'REGIME ' + self.engine.last_regime + '->' + result['regime'] if regime_changed else 'ALLOC SHIFT'}")
                                            await self._do_rebalance()

                        except asyncio.TimeoutError:
                            pass

                        now = time.time()

                        # Periodic broadcasts
                        if now - last_regime > 5:
                            await self._broadcast("regime", self.snapshot_regime())
                            last_regime = now

                        if now - last_positions > 10:
                            await self.refresh_account_data()
                            await self._broadcast("positions", self.snapshot_positions())
                            if self.portfolio:
                                await self._broadcast("portfolio", self.snapshot_portfolio())
                            last_positions = now

                        if now - last_balances > 10:
                            await self._broadcast("balances", self.snapshot_balances())
                            last_balances = now

                        if now - last_orders > 10:
                            await self._broadcast("orders", self.snapshot_orders())
                            last_orders = now

                        if now - last_heartbeat > 15:
                            await self._broadcast("heartbeat", {"ts": now})
                            last_heartbeat = now

            except (WebSocketDisconnect, BaseExceptionGroup) as e:
                reconnect_attempts += 1
                logger.warning(f"WebSocket disconnected: {e}")
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(f"FATAL: {max_reconnect_attempts} reconnect attempts failed")
                    break
            except Exception as e:
                reconnect_attempts += 1
                logger.error(f"Stream error: {e}")
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(f"FATAL: {max_reconnect_attempts} reconnect attempts failed")
                    break


_client = None
_stream_task = None


async def get_client():
    global _client, _stream_task
    if _client is None:
        mode = os.environ.get('DASHBOARD_MODE', 'live')
        if mode == 'test':
            from core.test_client import TestClient
            _client = TestClient()
        else:
            _client = TastyClient()
        await _client.connect()
        _stream_task = asyncio.create_task(_client.run_stream())
    return _client
