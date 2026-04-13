"""Test mode client — reads live_state.json from stream_trader instead of TastyTrade."""

import asyncio
import json
import time
from pathlib import Path

LIVE_STATE_FILE = Path(__file__).resolve().parent.parent.parent / 'trade_logs' / 'live_state.json'


class TestClient:
    """Drop-in replacement for TastyClient that reads from live_state.json."""

    def __init__(self):
        self.session = None
        self.account = None
        self.live_prices = {}
        self.latest_balances = {}
        self.latest_positions = []
        self.latest_orders = []
        self.latest_regime = {}
        self.latest_portfolio = {}
        self._streaming = False
        self._subscribers = []
        self._last_mtime = 0
        self.tick_count = 0

    async def connect(self):
        self._load_state()

    def _load_state(self):
        try:
            mtime = LIVE_STATE_FILE.stat().st_mtime
            if mtime == self._last_mtime:
                return False
            self._last_mtime = mtime

            state = json.loads(LIVE_STATE_FILE.read_text())

            self.latest_balances = {
                'net_liq': state.get('portfolio_value', 0),
                'cash': state.get('cash', 0),
                'buying_power': state.get('cash', 0),
                'pnl': state.get('pnl', 0),
                'pnl_pct': state.get('pnl_pct', 0),
                'starting_capital': state.get('starting_capital', 0),
            }

            self.latest_positions = []
            for p in state.get('positions', []):
                self.latest_positions.append({
                    'symbol': p['symbol'],
                    'shares': p.get('shares', p.get('qty', 0)),
                    'qty': p.get('shares', p.get('qty', 0)),
                    'avg_cost': p['avg_cost'],
                    'price': p['price'],
                    'market_value': p.get('market_value', p.get('value', 0)),
                    'pnl': p['pnl'],
                    'pnl_pct': round((p['pnl'] / (p.get('shares', 1) * p['avg_cost'])) * 100, 2) if p.get('shares', 0) * p['avg_cost'] else 0,
                    'tax_status': p.get('tax_status', ''),
                    'pool': p.get('pool', 'core'),
                })
                self.live_prices[p['symbol']] = {
                    'bid': p['price'], 'ask': p['price'],
                    'mid': p['price'], 'last_update': time.time(),
                }

            self.latest_orders = state.get('orders', [])
            self.tick_count = state.get('ticks', 0)

            self.latest_regime = {
                'regime': state.get('regime', 'UNKNOWN'),
                'leader_score': state.get('leader_score', 0),
                'tech_pct': round(state.get('tech_pct', 0) * 100, 1) if state.get('tech_pct', 0) <= 1 else state.get('tech_pct', 0),
                'singularity': state.get('singularity', False),
                'ticks': self.tick_count,
                'prices_connected': state.get('prices_connected', 0),
                'total_symbols': 30,
                'timestamp': state.get('timestamp', ''),
            }

            self.latest_portfolio = {
                'portfolio_value': state.get('portfolio_value', 0),
                'cash': state.get('cash', 0),
                'starting_capital': state.get('starting_capital', 0),
                'pnl': state.get('pnl', 0),
                'pnl_pct': state.get('pnl_pct', 0),
                'positions': self.latest_positions,
                'tax': state.get('tax', {'realized_gains': 0, 'realized_losses': 0, 'net': 0, 'deductible': 0}),
                'tax_events': state.get('tax_events', []),
            }

            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def snapshot_positions(self):
        return list(self.latest_positions)

    def snapshot_balances(self):
        return dict(self.latest_balances)

    def snapshot_orders(self):
        return list(self.latest_orders)

    def snapshot_regime(self):
        return dict(self.latest_regime)

    def snapshot_portfolio(self):
        return dict(self.latest_portfolio)

    def snapshot_rebalances(self):
        return []

    def subscribe(self):
        q = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def _broadcast(self, event_type, data):
        for q in self._subscribers:
            await q.put((event_type, data))

    async def run_stream(self):
        """Poll live_state.json, broadcast changes through SSE fan-out."""
        self._streaming = True
        last_heartbeat = 0

        while self._streaming:
            changed = self._load_state()

            if changed:
                await self._broadcast('balances', self.snapshot_balances())
                await self._broadcast('positions', self.snapshot_positions())
                await self._broadcast('orders', self.snapshot_orders())
                await self._broadcast('regime', self.snapshot_regime())
                await self._broadcast('portfolio', self.snapshot_portfolio())

                for sym, price_data in self.live_prices.items():
                    await self._broadcast('quote', {
                        'symbol': sym,
                        'bid': price_data['bid'],
                        'ask': price_data['ask'],
                        'mid': price_data['mid'],
                    })

            now = time.time()
            if now - last_heartbeat > 15:
                await self._broadcast('heartbeat', {'ts': now})
                last_heartbeat = now

            await asyncio.sleep(2)
