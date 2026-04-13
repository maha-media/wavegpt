"""Speculator — AI-driven autonomous speculative trading."""

import json
import time
from datetime import datetime
from pathlib import Path
from config import (
    SPEC_POOL_PCT, MAX_SPEC_POSITIONS, DAILY_LOSS_LIMIT_PCT,
    QUEUE_FILE, SPEC_POSITIONS_FILE,
)

QUEUE_PATH = Path(__file__).parent / QUEUE_FILE
POSITIONS_PATH = Path(__file__).parent / SPEC_POSITIONS_FILE


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
