"""Speculator — AI-driven autonomous speculative trading."""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from config import (
    SPEC_POOL_PCT, MAX_SPEC_POSITIONS, DAILY_LOSS_LIMIT_PCT,
    QUEUE_FILE, SPEC_POSITIONS_FILE,
)

QUEUE_PATH = Path(__file__).parent / QUEUE_FILE
POSITIONS_PATH = Path(__file__).parent / SPEC_POSITIONS_FILE
LOG_DIR = Path(__file__).parent / 'trade_logs'
LOG_DIR.mkdir(exist_ok=True)


def build_evaluation_prompt(
    ticker: str,
    triggering_content: list[str],
    velocity_data: dict,
    price: float,
    momentum_5d: float | None,
    momentum_20d: float | None,
    already_held: bool,
    pool_status: dict,
) -> str:
    """Build the AI evaluation prompt with full context."""
    content_block = '\n'.join(f'  - {c[:200]}' for c in triggering_content[:20])
    held_note = f'WARNING: {ticker} is already held in the portfolio.' if already_held else ''

    return f"""You are a speculative trading AI. Evaluate this opportunity and return a JSON decision.

## Ticker: {ticker}
Current price: ${price:.2f}
5-day momentum: {f'{momentum_5d:+.2%}' if momentum_5d is not None else 'N/A'}
20-day momentum: {f'{momentum_20d:+.2%}' if momentum_20d is not None else 'N/A'}
{held_note}

## Triggering Social Content
{content_block}

## Mention Velocity
{json.dumps(velocity_data, indent=2)}

## Speculative Pool Status
- Current positions: {pool_status.get('positions', 0)} / {MAX_SPEC_POSITIONS}
- Available capital: {pool_status.get('available_pct', 1.0):.0%} of pool
- Daily P/L: {pool_status.get('daily_pnl_pct', 0):.2%}

## Rules
- position_size_pct is a fraction of the speculative pool (0.0 to 1.0)
- You MUST set stop_loss_pct (e.g. 0.08 for 8% stop-loss)
- "watch" = add to hot list, re-evaluate on next trigger
- If already held, account for total exposure

Return ONLY valid JSON:
{{
  "action": "buy | pass | watch",
  "ticker": "{ticker}",
  "conviction": 0.0-1.0,
  "reasoning": "...",
  "entry_strategy": "market | limit",
  "limit_price": null,
  "position_size_pct": 0.0-1.0,
  "stop_loss_pct": 0.01-0.20,
  "target_pct": null,
  "exit_timeframe_hours": 24,
  "category": "pump_and_dump | momentum | news_catalyst | earnings"
}}"""


def parse_ai_decision(raw: str) -> dict | None:
    """Parse and validate AI decision JSON. Returns None if invalid."""
    try:
        text = raw.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        d = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return None

    action = d.get('action')
    if action not in ('buy', 'pass', 'watch'):
        return None

    if action == 'buy':
        if 'stop_loss_pct' not in d or not isinstance(d['stop_loss_pct'], (int, float)):
            return None
        if 'position_size_pct' not in d or not isinstance(d['position_size_pct'], (int, float)):
            return None
        d['position_size_pct'] = max(0.01, min(1.0, d['position_size_pct']))
        d['stop_loss_pct'] = max(0.01, min(0.20, d['stop_loss_pct']))

    return d


class SpecPool:
    """Manages the speculative capital pool and position tracking."""

    def __init__(self, total_capital: float, spec_pct: float = SPEC_POOL_PCT):
        self.capital = total_capital * spec_pct
        self.positions: dict[str, dict] = {}
        self.daily_pnl = 0.0
        self.paused = False

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def invested_capital(self) -> float:
        return sum(
            p['shares'] * p['entry_price']
            for p in self.positions.values()
        )

    @property
    def available_capital(self) -> float:
        return self.capital - self.invested_capital

    def can_trade(self) -> bool:
        return (
            self.position_count < MAX_SPEC_POSITIONS
            and self.available_capital > 0
            and not self.paused
        )

    def circuit_breaker_tripped(self) -> bool:
        return self.daily_pnl <= -(self.capital * DAILY_LOSS_LIMIT_PCT)

    def status_dict(self) -> dict:
        return {
            'positions': self.position_count,
            'available_pct': self.available_capital / self.capital if self.capital > 0 else 0,
            'daily_pnl_pct': self.daily_pnl / self.capital if self.capital > 0 else 0,
        }

    def save(self):
        """Persist positions to disk."""
        data = {
            'capital': self.capital,
            'positions': self.positions,
            'daily_pnl': self.daily_pnl,
            'paused': self.paused,
            'saved_at': datetime.now().isoformat(),
        }
        tmp = POSITIONS_PATH.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(POSITIONS_PATH)

    def load(self):
        """Load persisted positions from disk."""
        if POSITIONS_PATH.exists():
            try:
                data = json.loads(POSITIONS_PATH.read_text())
                self.positions = data.get('positions', {})
                self.daily_pnl = data.get('daily_pnl', 0.0)
                self.paused = data.get('paused', False)
            except (json.JSONDecodeError, KeyError):
                pass


class Speculator:
    """Autonomous speculative trading engine."""

    def __init__(self, session, account, portfolio, spec_pool: SpecPool,
                 signal_engine=None, dry_run: bool = True):
        self.session = session
        self.account = account
        self.portfolio = portfolio  # shared LocalPortfolio
        self.pool = spec_pool
        self.engine = signal_engine  # for price data
        self.dry_run = dry_run
        self.queue_path = Path(__file__).parent / QUEUE_FILE
        self.watch_list: dict[str, dict] = {}

    def read_queue(self) -> list[dict]:
        """Read and clear the sentinel queue."""
        if not self.queue_path.exists():
            return []
        try:
            items = json.loads(self.queue_path.read_text())
            self.queue_path.write_text('[]')
            return items if isinstance(items, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    async def evaluate_ticker(self, ticker: str, opportunities: list[dict]) -> dict | None:
        """Call Claude via Bedrock to evaluate a trading opportunity."""
        import boto3

        price = self.engine.get_price(ticker) if self.engine else None
        if price is None:
            try:
                import yfinance as yf
                data = yf.download(ticker, period='5d', interval='1d', auto_adjust=True)
                if not data.empty:
                    price = float(data['Close'].iloc[-1])
            except Exception:
                return None
        if price is None or price <= 0:
            return None

        mom_5d = self.engine.safe_mom(ticker, 5) if self.engine else None
        mom_20d = self.engine.safe_mom(ticker, 20) if self.engine else None
        already_held = ticker in self.portfolio.positions

        prompt = build_evaluation_prompt(
            ticker=ticker,
            triggering_content=[o.get('text', '') for o in opportunities],
            velocity_data={
                'mentions': len(opportunities),
                'spike': any(o.get('velocity_spike') for o in opportunities),
            },
            price=price,
            momentum_5d=mom_5d,
            momentum_20d=mom_20d,
            already_held=already_held,
            pool_status=self.pool.status_dict(),
        )

        try:
            from config import BEDROCK_MODEL_ID
            client = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            })
            model_id = os.environ.get("BEDROCK_MODEL_ID", BEDROCK_MODEL_ID)
            response = client.invoke_model(modelId=model_id, body=body)
            result = json.loads(response["body"].read())
            raw = result["content"][0]["text"]
            decision = parse_ai_decision(raw)
            if decision:
                decision['_price_at_eval'] = price
                decision['_prompt'] = prompt[:500]
            return decision
        except Exception as e:
            print(f"  [Speculator] AI eval error for {ticker}: {e}")
            return None

    async def execute_buy(self, decision: dict):
        """Execute a buy decision with stop-loss bracket."""
        ticker = decision['ticker']
        price = decision.get('_price_at_eval', 0)
        if price <= 0:
            return

        size_pct = decision['position_size_pct']
        dollar_amount = self.pool.available_capital * size_pct
        shares = int(dollar_amount / price)
        if shares < 1:
            print(f"  [Speculator] {ticker}: position too small ({dollar_amount:.0f} / {price:.2f})")
            return

        stop_price = price * (1 - decision['stop_loss_pct'])
        target_price = price * (1 + decision.get('target_pct', 0.15)) if decision.get('target_pct') else None

        print(f"  [Speculator] BUY {ticker} x{shares} @ ${price:.2f}")
        print(f"    Stop: ${stop_price:.2f}  Target: {f'${target_price:.2f}' if target_price else 'none'}")
        print(f"    Reason: {decision.get('reasoning', 'N/A')[:100]}")

        if not self.dry_run:
            try:
                from tastytrade.instruments import Equity
                from tastytrade.order import (
                    NewOrder, OrderAction, OrderTimeInForce, OrderType,
                )
                equity = await Equity.get(self.session, [ticker])
                if isinstance(equity, list):
                    equity = equity[0]
                leg = equity.build_leg(shares, OrderAction.BUY_TO_OPEN)
                if decision.get('entry_strategy') == 'limit' and decision.get('limit_price'):
                    order = NewOrder(
                        time_in_force=OrderTimeInForce.DAY,
                        order_type=OrderType.LIMIT,
                        price=decision['limit_price'],
                        legs=[leg],
                    )
                else:
                    order = NewOrder(
                        time_in_force=OrderTimeInForce.DAY,
                        order_type=OrderType.MARKET,
                        legs=[leg],
                    )
                await self.account.place_order(self.session, order)

                stop_leg = equity.build_leg(shares, OrderAction.SELL_TO_CLOSE)
                stop_order = NewOrder(
                    time_in_force=OrderTimeInForce.GTC,
                    order_type=OrderType.STOP,
                    stop_trigger=stop_price,
                    legs=[stop_leg],
                )
                await self.account.place_order(self.session, stop_order)
                print(f"    -> Orders placed (market + stop-loss)")
            except Exception as e:
                print(f"    -> ORDER FAILED: {e}")
                return

        self.pool.positions[ticker] = {
            'shares': shares,
            'entry_price': price,
            'stop_loss': stop_price,
            'target': target_price,
            'entry_time': datetime.now().isoformat(),
            'exit_timeframe_hours': decision.get('exit_timeframe_hours', 24),
            'category': decision.get('category', 'unknown'),
            'reasoning': decision.get('reasoning', ''),
        }
        self.portfolio.fill_order(ticker, shares, price)
        self.pool.save()

        log_path = LOG_DIR / f"spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ticker}.json"
        log_path.write_text(json.dumps({
            'type': 'spec_buy',
            'decision': decision,
            'shares': shares,
            'price': price,
            'stop_loss': stop_price,
            'target': target_price,
            'dry_run': self.dry_run,
            'timestamp': datetime.now().isoformat(),
        }, indent=2, default=str))

    async def check_exits(self):
        """Check for timed exits on spec positions."""
        now = datetime.now()
        to_close = []
        for ticker, pos in list(self.pool.positions.items()):
            entry_time = datetime.fromisoformat(pos['entry_time'])
            hours_held = (now - entry_time).total_seconds() / 3600
            if hours_held >= pos.get('exit_timeframe_hours', 24):
                to_close.append(ticker)

        for ticker in to_close:
            pos = self.pool.positions[ticker]
            price = self.engine.get_price(ticker) if self.engine else pos['entry_price']
            print(f"  [Speculator] TIME EXIT {ticker} after {pos.get('exit_timeframe_hours', 24)}h")
            if not self.dry_run:
                try:
                    from tastytrade.instruments import Equity
                    from tastytrade.order import (
                        NewOrder, OrderAction, OrderTimeInForce, OrderType,
                    )
                    equity = await Equity.get(self.session, [ticker])
                    if isinstance(equity, list):
                        equity = equity[0]
                    leg = equity.build_leg(pos['shares'], OrderAction.SELL_TO_CLOSE)
                    order = NewOrder(
                        time_in_force=OrderTimeInForce.DAY,
                        order_type=OrderType.MARKET,
                        legs=[leg],
                    )
                    await self.account.place_order(self.session, order)
                except Exception as e:
                    print(f"    -> EXIT ORDER FAILED: {e}")
                    continue

            self.portfolio.fill_order(ticker, -pos['shares'], price)
            pnl = pos['shares'] * (price - pos['entry_price'])
            self.pool.daily_pnl += pnl
            del self.pool.positions[ticker]
            self.pool.save()

    async def run(self):
        """Main loop — read queue, evaluate, execute, manage positions."""
        print(f"  [Speculator] Starting — pool: ${self.pool.capital:,.0f}, dry_run={self.dry_run}")
        self.pool.load()

        while True:
            if self.pool.circuit_breaker_tripped():
                if not self.pool.paused:
                    print(f"  [Speculator] CIRCUIT BREAKER — daily loss "
                          f"${self.pool.daily_pnl:,.0f} exceeds limit")
                    self.pool.paused = True
                    self.pool.save()
                await asyncio.sleep(60)
                continue

            await self.check_exits()

            opportunities = self.read_queue()
            if opportunities:
                by_ticker: dict[str, list[dict]] = {}
                for opp in opportunities:
                    t = opp.get('ticker', '')
                    if t:
                        by_ticker.setdefault(t, []).append(opp)

                for ticker, opps in by_ticker.items():
                    if not self.pool.can_trade():
                        print(f"  [Speculator] Cannot trade — pool full or paused")
                        break
                    if ticker in self.pool.positions:
                        continue

                    decision = await self.evaluate_ticker(ticker, opps)
                    if decision is None:
                        continue

                    if decision['action'] == 'buy':
                        await self.execute_buy(decision)
                    elif decision['action'] == 'watch':
                        self.watch_list[ticker] = {
                            'added': time.time(),
                            'reasoning': decision.get('reasoning', ''),
                        }
                        print(f"  [Speculator] WATCH {ticker}: {decision.get('reasoning', '')[:80]}")

            await asyncio.sleep(10)
