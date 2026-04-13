# Sentinel + Speculator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add social-signal monitoring (Sentinel) and autonomous speculative trading (Speculator) to the existing stream trading system, using a 10% speculative capital pool.

**Architecture:** Sentinel monitors social feeds via Exa API (livecrawl + scheduled searches), fuzzy-matches tickers, and writes flagged opportunities to a JSON queue file. Speculator reads that queue, calls Claude for trade evaluation, executes via TastyTrade, and manages positions with stop-losses and circuit breakers. Both pools share the existing LocalPortfolio for unified P/L and tax tracking.

**Tech Stack:** Python 3, exa-py (Exa API), boto3/Bedrock (Claude Opus 4.6 via AWS_BEARER_TOKEN_BEDROCK), yfinance, tastytrade SDK, pytest

**Design doc:** `finance/docs/plans/2026-04-13-sentinel-speculator-design.md`

---

### Task 1: Install dependencies

**Step 1: Install exa-py and verify anthropic SDK**

Run:
```bash
pip install exa-py
pip show exa-py anthropic
```
Expected: both packages installed

**Step 2: Add API keys to .env**

Edit: `finance/.env` — add:
```
EXA_API_KEY=your_key_here
```
Note: Claude AI calls use the existing `AWS_BEARER_TOKEN_BEDROCK` env var via Bedrock.

**Step 3: Commit**

```bash
git add -p
git commit -m "chore: add exa-py dependency for social monitoring"
```

---

### Task 2: Create config.py — shared constants

**Files:**
- Create: `finance/config.py`
- Test: `finance/tests/test_config.py`

**Step 1: Write the failing test**

```python
# finance/tests/test_config.py
from config import (
    SPEC_POOL_PCT, MAX_SPEC_POSITIONS, DAILY_LOSS_LIMIT_PCT,
    VELOCITY_SPIKE_MULT, AI_DEDUP_WINDOW_SEC,
    WATCHLIST_MIN_HISTORY_DAYS, WATCHLIST_MIN_MARKET_CAP,
    SIGNAL_KEYWORDS, SENTINEL_FAST_POLL_SEC, SENTINEL_SLOW_POLL_SEC,
)


def test_spec_pool_pct_is_ten_percent():
    assert SPEC_POOL_PCT == 0.10


def test_max_spec_positions():
    assert MAX_SPEC_POSITIONS == 5


def test_daily_loss_limit():
    assert DAILY_LOSS_LIMIT_PCT == 0.05


def test_velocity_spike_mult():
    assert VELOCITY_SPIKE_MULT == 3.0


def test_ai_dedup_window():
    assert AI_DEDUP_WINDOW_SEC == 600


def test_signal_keywords_has_bullish_and_bearish():
    assert 'moon' in SIGNAL_KEYWORDS['bullish']
    assert 'crash' in SIGNAL_KEYWORDS['bearish']


def test_poll_intervals():
    assert SENTINEL_FAST_POLL_SEC == 300
    assert SENTINEL_SLOW_POLL_SEC == 1800
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_config.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# finance/config.py
"""Shared configuration for Sentinel + Speculator trading system."""

# --- Capital allocation ---
SPEC_POOL_PCT = 0.10              # 10% of total capital to speculative pool
MAX_SPEC_POSITIONS = 5
DAILY_LOSS_LIMIT_PCT = 0.05       # 5% of spec pool triggers daily pause

# --- Sentinel: social monitoring ---
VELOCITY_SPIKE_MULT = 3.0         # 3x trailing avg = velocity spike
AI_DEDUP_WINDOW_SEC = 600         # don't re-prompt same ticker within 10 min
SENTINEL_FAST_POLL_SEC = 300      # 5-min targeted search interval
SENTINEL_SLOW_POLL_SEC = 1800     # 30-min broad discovery interval

# --- Sentinel: keyword scoring ---
SIGNAL_KEYWORDS = {
    'bullish': {
        'moon': 3.0, 'squeeze': 4.0, 'calls': 2.0, 'breaking': 3.0,
        'approval': 4.0, 'earnings beat': 3.0, 'upgrade': 2.5,
        'buy': 1.5, 'bullish': 2.0, 'rocket': 2.0, 'tendies': 1.5,
        'YOLO': 1.5, 'diamond hands': 1.0,
    },
    'bearish': {
        'crash': 3.0, 'puts': 2.0, 'dump': 3.0, 'sell': 1.5,
        'bearish': 2.0, 'downgrade': 2.5, 'miss': 2.0, 'fraud': 4.0,
        'SEC': 3.0, 'bankruptcy': 4.0, 'overvalued': 2.0,
    },
}
KEYWORD_TRIGGER_SCORE = 5.0       # minimum score to trigger AI evaluation

# --- Sentinel: search queries ---
FAST_SEARCH_QUERIES = [
    '{ticker} momentum',
    '{ticker} short squeeze',
    'FDA approval stock',
    'earnings surprise today',
]

SLOW_SEARCH_QUERIES = [
    'AI company IPO 2026 filing S-1',
    'new ETF launch 2026',
    'stock market trending Reddit wallstreetbets',
    'semiconductor stocks breakout',
    'defense stocks government contract',
    'energy transition stock',
]

# --- Watchlist: universe discovery ---
WATCHLIST_MIN_HISTORY_DAYS = 50
WATCHLIST_MIN_MARKET_CAP = 1e9    # $1B minimum for core consideration

# --- File paths ---
QUEUE_FILE = 'sentinel_queue.json'
SPEC_POSITIONS_FILE = 'spec_positions.json'
WATCHLIST_FILE = 'watchlist.json'

# --- AI model ---
BEDROCK_MODEL_ID = 'anthropic.claude-opus-4-6-v1'
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_config.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add finance/config.py finance/tests/test_config.py
git commit -m "feat: add config.py with sentinel/speculator constants"
```

---

### Task 3: Sentinel — ticker extraction and keyword scoring

**Files:**
- Create: `finance/sentinel.py`
- Test: `finance/tests/test_sentinel.py`

**Step 1: Write failing tests for ticker extraction**

```python
# finance/tests/test_sentinel.py
import pytest
from sentinel import extract_tickers, compute_keyword_score, VelocityTracker


class TestExtractTickers:
    def test_cashtag(self):
        assert 'NVDA' in extract_tickers('$NVDA to the moon!')

    def test_multiple_cashtags(self):
        tickers = extract_tickers('$AAPL and $MSFT looking strong')
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers

    def test_no_tickers(self):
        assert extract_tickers('the market is flat today') == []

    def test_filters_common_words(self):
        # $A, $I, $IT etc should be filtered
        assert extract_tickers('$I think $A stock is good') == []

    def test_uppercase_only(self):
        assert extract_tickers('$nvda') == []  # cashtags are uppercase


class TestKeywordScore:
    def test_bullish_keywords(self):
        score = compute_keyword_score('NVDA to the moon, squeeze incoming!')
        assert score > 5.0  # moon=3 + squeeze=4 = 7

    def test_bearish_keywords(self):
        score = compute_keyword_score('this stock is a fraud, SEC investigation')
        assert score > 5.0  # fraud=4 + SEC=3 = 7

    def test_no_keywords(self):
        score = compute_keyword_score('the weather is nice today')
        assert score == 0.0

    def test_case_insensitive(self):
        score = compute_keyword_score('MOON SQUEEZE')
        assert score > 0


class TestVelocityTracker:
    def test_no_spike_on_first_mention(self):
        vt = VelocityTracker(spike_mult=3.0, window_sec=60)
        assert vt.record_and_check('NVDA', 1000.0) is False

    def test_spike_on_burst(self):
        vt = VelocityTracker(spike_mult=3.0, window_sec=60)
        base_time = 0.0
        # Build baseline: 1 mention per second for 10 seconds
        for i in range(10):
            vt.record_and_check('NVDA', base_time + i)
        # Burst: 10 mentions in 1 second
        for i in range(10):
            result = vt.record_and_check('NVDA', base_time + 11)
        assert result is True
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_sentinel.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# finance/sentinel.py (initial — ticker extraction + keyword scoring)
"""Sentinel — social firehose monitoring with fuzzy matching."""

import re
from collections import defaultdict
from config import SIGNAL_KEYWORDS

# Cashtag regex: $AAPL, $NVDA (2-5 uppercase letters, not common words)
_CASHTAG_RE = re.compile(r'\$([A-Z]{2,5})\b')
_COMMON_WORDS = {'THE', 'FOR', 'AND', 'BUT', 'NOT', 'ALL', 'ARE', 'CAN', 'HAS',
                 'HAD', 'WAS', 'HER', 'HIS', 'HOW', 'ITS', 'LET', 'MAY', 'NEW',
                 'NOW', 'OLD', 'OUR', 'OWN', 'SAY', 'SHE', 'TOO', 'USE', 'WAY',
                 'WHO', 'DID', 'GET', 'HIM', 'HIT', 'PUT', 'RUN', 'SET', 'TOP',
                 'WIN', 'YOU'}


def extract_tickers(text: str) -> list[str]:
    """Extract $CASHTAG tickers from text. Returns list of unique tickers."""
    matches = _CASHTAG_RE.findall(text)
    return [m for m in dict.fromkeys(matches) if m not in _COMMON_WORDS]


def compute_keyword_score(text: str) -> float:
    """Score text against signal keywords. Returns total weighted score."""
    text_lower = text.lower()
    score = 0.0
    for category in SIGNAL_KEYWORDS.values():
        for keyword, weight in category.items():
            if keyword.lower() in text_lower:
                score += weight
    return score


class VelocityTracker:
    """Track per-ticker mention velocity with sliding window."""

    def __init__(self, spike_mult: float = 3.0, window_sec: float = 60.0):
        self.spike_mult = spike_mult
        self.window_sec = window_sec
        self.mentions: dict[str, list[float]] = defaultdict(list)

    def _prune(self, ticker: str, now: float):
        cutoff = now - self.window_sec
        times = self.mentions[ticker]
        self.mentions[ticker] = [t for t in times if t > cutoff]

    def record_and_check(self, ticker: str, timestamp: float) -> bool:
        """Record a mention and return True if velocity spike detected."""
        self._prune(ticker, timestamp)
        prev_count = len(self.mentions[ticker])
        self.mentions[ticker].append(timestamp)

        if prev_count < 3:
            return False  # need baseline

        # Compare current rate to trailing average
        times = self.mentions[ticker]
        if len(times) < 4:
            return False
        # Rate = mentions in last window_sec/3 vs avg over full window
        third = self.window_sec / 3
        recent = sum(1 for t in times if t > timestamp - third)
        avg_rate = len(times) / 3  # expected per third of window
        return recent > avg_rate * self.spike_mult
```

**Step 4: Run test to verify it passes**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_sentinel.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add finance/sentinel.py finance/tests/test_sentinel.py
git commit -m "feat: sentinel ticker extraction, keyword scoring, velocity tracking"
```

---

### Task 4: Sentinel — Exa integration and deduplication

**Files:**
- Modify: `finance/sentinel.py`
- Test: `finance/tests/test_sentinel.py` (add tests)

**Step 1: Write failing tests for SentinelMonitor class**

Add to `finance/tests/test_sentinel.py`:

```python
import time
from unittest.mock import AsyncMock, patch, MagicMock
from sentinel import SentinelMonitor


class TestSentinelDedup:
    def test_dedup_blocks_repeat_within_window(self):
        mon = SentinelMonitor(exa_api_key='fake')
        mon._last_prompted = {'NVDA': time.time()}
        assert mon._should_dedupe('NVDA') is True

    def test_dedup_allows_after_window(self):
        mon = SentinelMonitor(exa_api_key='fake')
        mon._last_prompted = {'NVDA': time.time() - 700}  # > 600s
        assert mon._should_dedupe('NVDA') is False

    def test_dedup_allows_new_ticker(self):
        mon = SentinelMonitor(exa_api_key='fake')
        assert mon._should_dedupe('NVDA') is False


class TestSentinelEvaluate:
    def test_flags_high_score_ticker(self):
        mon = SentinelMonitor(exa_api_key='fake')
        results = mon.evaluate_content(
            '$NVDA squeeze to the moon! Diamond hands!',
            source='test'
        )
        assert len(results) > 0
        assert results[0]['ticker'] == 'NVDA'
        assert results[0]['score'] > 0

    def test_ignores_low_score(self):
        mon = SentinelMonitor(exa_api_key='fake')
        results = mon.evaluate_content('$AAPL mentioned', source='test')
        assert len(results) == 0  # no signal keywords, no velocity
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_sentinel.py::TestSentinelDedup -v`
Expected: FAIL — ImportError (SentinelMonitor doesn't exist yet)

**Step 3: Write SentinelMonitor class**

Add to `finance/sentinel.py`:

```python
import asyncio
import json
import time
from pathlib import Path
from config import (
    AI_DEDUP_WINDOW_SEC, KEYWORD_TRIGGER_SCORE, VELOCITY_SPIKE_MULT,
    FAST_SEARCH_QUERIES, SLOW_SEARCH_QUERIES,
    SENTINEL_FAST_POLL_SEC, SENTINEL_SLOW_POLL_SEC,
    QUEUE_FILE,
)

QUEUE_PATH = Path(__file__).parent / QUEUE_FILE


class SentinelMonitor:
    """Monitors social feeds via Exa, flags tickers for speculator."""

    def __init__(self, exa_api_key: str, watched_tickers: list[str] | None = None):
        self.exa_api_key = exa_api_key
        self.watched_tickers = set(watched_tickers or [])
        self.velocity = VelocityTracker(spike_mult=VELOCITY_SPIKE_MULT)
        self._last_prompted: dict[str, float] = {}

    def _should_dedupe(self, ticker: str) -> bool:
        last = self._last_prompted.get(ticker)
        if last is None:
            return False
        return (time.time() - last) < AI_DEDUP_WINDOW_SEC

    def evaluate_content(self, text: str, source: str) -> list[dict]:
        """Evaluate a piece of content. Returns list of flagged opportunities."""
        tickers = extract_tickers(text)
        if not tickers:
            return []

        keyword_score = compute_keyword_score(text)
        now = time.time()
        results = []

        for ticker in tickers:
            velocity_spike = self.velocity.record_and_check(ticker, now)

            if keyword_score >= KEYWORD_TRIGGER_SCORE or velocity_spike:
                if self._should_dedupe(ticker):
                    continue
                self._last_prompted[ticker] = now
                results.append({
                    'ticker': ticker,
                    'score': keyword_score,
                    'velocity_spike': velocity_spike,
                    'text': text[:500],
                    'source': source,
                    'timestamp': now,
                })

        return results

    def write_to_queue(self, opportunities: list[dict]):
        """Append opportunities to the JSON queue file for the Speculator."""
        existing = []
        if QUEUE_PATH.exists():
            try:
                existing = json.loads(QUEUE_PATH.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                existing = []
        existing.extend(opportunities)
        tmp = QUEUE_PATH.with_suffix('.tmp')
        tmp.write_text(json.dumps(existing, indent=2))
        tmp.rename(QUEUE_PATH)

    async def run_fast_poll(self):
        """Run targeted Exa searches every SENTINEL_FAST_POLL_SEC."""
        from exa_py import Exa
        exa = Exa(self.exa_api_key)

        while True:
            for query_template in FAST_SEARCH_QUERIES:
                # Search for each watched ticker
                tickers_to_search = list(self.watched_tickers)[:10]
                for ticker in tickers_to_search:
                    query = query_template.format(ticker=ticker)
                    try:
                        results = exa.search(
                            query,
                            num_results=10,
                            use_autoprompt=True,
                            type='neural',
                        )
                        for r in results.results:
                            content = f"{r.title} {r.text}" if hasattr(r, 'text') else r.title
                            opps = self.evaluate_content(content, source=f'exa_fast:{query}')
                            if opps:
                                self.write_to_queue(opps)
                    except Exception as e:
                        print(f"  [Sentinel] Exa search error: {e}")

            await asyncio.sleep(SENTINEL_FAST_POLL_SEC)

    async def run_slow_sweep(self):
        """Run broad discovery searches every SENTINEL_SLOW_POLL_SEC."""
        from exa_py import Exa
        exa = Exa(self.exa_api_key)

        while True:
            for query in SLOW_SEARCH_QUERIES:
                try:
                    results = exa.search(
                        query,
                        num_results=10,
                        use_autoprompt=True,
                        type='neural',
                    )
                    for r in results.results:
                        content = f"{r.title} {r.text}" if hasattr(r, 'text') else r.title
                        opps = self.evaluate_content(content, source=f'exa_slow:{query}')
                        if opps:
                            self.write_to_queue(opps)
                except Exception as e:
                    print(f"  [Sentinel] Exa sweep error: {e}")

            await asyncio.sleep(SENTINEL_SLOW_POLL_SEC)

    async def run(self):
        """Main entry point — runs fast poll and slow sweep concurrently."""
        print(f"  [Sentinel] Starting — watching {len(self.watched_tickers)} tickers")
        await asyncio.gather(
            self.run_fast_poll(),
            self.run_slow_sweep(),
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_sentinel.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add finance/sentinel.py finance/tests/test_sentinel.py
git commit -m "feat: sentinel monitor with Exa integration and dedup"
```

---

### Task 5: Speculator — AI evaluation prompt and JSON parsing

**Files:**
- Create: `finance/speculator.py`
- Test: `finance/tests/test_speculator.py`

**Step 1: Write failing tests**

```python
# finance/tests/test_speculator.py
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from speculator import build_evaluation_prompt, parse_ai_decision, SpecPool


class TestBuildPrompt:
    def test_includes_ticker(self):
        prompt = build_evaluation_prompt(
            ticker='NVDA',
            triggering_content=['$NVDA squeeze incoming'],
            velocity_data={'mentions_per_min': 5.0, 'trailing_avg': 1.2},
            price=192.50,
            momentum_5d=0.03,
            momentum_20d=0.08,
            already_held=False,
            pool_status={'positions': 2, 'available_pct': 0.60, 'daily_pnl_pct': -0.01},
        )
        assert 'NVDA' in prompt
        assert '192.50' in prompt
        assert 'squeeze' in prompt

    def test_includes_pool_constraints(self):
        prompt = build_evaluation_prompt(
            ticker='TSLA',
            triggering_content=['$TSLA moon'],
            velocity_data={},
            price=250.0,
            momentum_5d=0.05,
            momentum_20d=0.10,
            already_held=True,
            pool_status={'positions': 4, 'available_pct': 0.20, 'daily_pnl_pct': -0.03},
        )
        assert 'already held' in prompt.lower() or 'TSLA' in prompt
        assert '4' in prompt  # 4 positions


class TestParseAiDecision:
    def test_valid_buy(self):
        raw = json.dumps({
            'action': 'buy',
            'ticker': 'NVDA',
            'conviction': 0.8,
            'reasoning': 'Short squeeze setup',
            'entry_strategy': 'market',
            'position_size_pct': 0.30,
            'stop_loss_pct': 0.08,
            'target_pct': 0.15,
            'exit_timeframe_hours': 24,
            'category': 'momentum',
        })
        decision = parse_ai_decision(raw)
        assert decision['action'] == 'buy'
        assert decision['stop_loss_pct'] == 0.08
        assert 0 < decision['position_size_pct'] <= 1.0

    def test_rejects_missing_stop_loss(self):
        raw = json.dumps({
            'action': 'buy',
            'ticker': 'NVDA',
            'conviction': 0.8,
            'position_size_pct': 0.30,
        })
        decision = parse_ai_decision(raw)
        assert decision is None  # invalid — no stop_loss_pct

    def test_pass_action(self):
        raw = json.dumps({'action': 'pass', 'ticker': 'NVDA', 'reasoning': 'not enough signal'})
        decision = parse_ai_decision(raw)
        assert decision['action'] == 'pass'

    def test_garbage_input(self):
        decision = parse_ai_decision('this is not json at all')
        assert decision is None


class TestSpecPool:
    def test_initial_state(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        assert pool.capital == 10_000
        assert pool.position_count == 0
        assert pool.available_capital == 10_000

    def test_can_trade_under_limit(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        assert pool.can_trade() is True

    def test_cannot_trade_at_max_positions(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        pool.positions = {f'T{i}': {} for i in range(5)}
        assert pool.can_trade() is False

    def test_circuit_breaker_triggers(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        pool.daily_pnl = -600  # 6% of 10k > 5% threshold
        assert pool.circuit_breaker_tripped() is True

    def test_circuit_breaker_ok(self):
        pool = SpecPool(total_capital=100_000, spec_pct=0.10)
        pool.daily_pnl = -200  # 2% of 10k < 5% threshold
        assert pool.circuit_breaker_tripped() is False
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_speculator.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write implementation**

```python
# finance/speculator.py
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
        # Handle markdown code blocks
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
        # Require stop_loss_pct for buys
        if 'stop_loss_pct' not in d or not isinstance(d['stop_loss_pct'], (int, float)):
            return None
        if 'position_size_pct' not in d or not isinstance(d['position_size_pct'], (int, float)):
            return None
        # Clamp
        d['position_size_pct'] = max(0.01, min(1.0, d['position_size_pct']))
        d['stop_loss_pct'] = max(0.01, min(0.20, d['stop_loss_pct']))

    return d


class SpecPool:
    """Manages the speculative capital pool and position tracking."""

    def __init__(self, total_capital: float, spec_pct: float = SPEC_POOL_PCT):
        self.capital = total_capital * spec_pct
        self.positions: dict[str, dict] = {}  # ticker -> position info
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_speculator.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add finance/speculator.py finance/tests/test_speculator.py
git commit -m "feat: speculator AI prompt builder, decision parser, spec pool"
```

---

### Task 6: Speculator — execution engine (AI call + order placement)

**Files:**
- Modify: `finance/speculator.py`
- Test: `finance/tests/test_speculator.py` (add tests)

**Step 1: Write failing tests for execution**

Add to `finance/tests/test_speculator.py`:

```python
import asyncio
from speculator import Speculator


class TestSpeculatorExecution:
    def test_read_queue_empty(self, tmp_path):
        spec = Speculator.__new__(Speculator)
        spec.queue_path = tmp_path / 'queue.json'
        assert spec.read_queue() == []

    def test_read_queue_with_items(self, tmp_path):
        spec = Speculator.__new__(Speculator)
        spec.queue_path = tmp_path / 'queue.json'
        spec.queue_path.write_text(json.dumps([
            {'ticker': 'NVDA', 'score': 7.0, 'text': 'moon'},
        ]))
        items = spec.read_queue()
        assert len(items) == 1
        assert items[0]['ticker'] == 'NVDA'

    def test_read_queue_clears_after_read(self, tmp_path):
        spec = Speculator.__new__(Speculator)
        spec.queue_path = tmp_path / 'queue.json'
        spec.queue_path.write_text(json.dumps([{'ticker': 'NVDA'}]))
        spec.read_queue()
        assert spec.read_queue() == []
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_speculator.py::TestSpeculatorExecution -v`
Expected: FAIL — Speculator has no read_queue

**Step 3: Add Speculator execution class**

Add to `finance/speculator.py`:

```python
import asyncio
import os
from config import QUEUE_FILE

LOG_DIR = Path(__file__).parent / 'trade_logs'
LOG_DIR.mkdir(exist_ok=True)


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
        self.watch_list: dict[str, dict] = {}  # tickers to re-evaluate

    def read_queue(self) -> list[dict]:
        """Read and clear the sentinel queue."""
        if not self.queue_path.exists():
            return []
        try:
            items = json.loads(self.queue_path.read_text())
            # Clear queue after reading
            self.queue_path.write_text('[]')
            return items if isinstance(items, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    async def evaluate_ticker(self, ticker: str, opportunities: list[dict]) -> dict | None:
        """Call Claude via Bedrock to evaluate a trading opportunity."""
        import boto3

        # Gather context
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
            client = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            })
            from config import BEDROCK_MODEL_ID
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

        # Calculate shares
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

                # Place stop-loss order
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

        # Record in pool and portfolio
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

        # Log
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
        print(f"  [Speculator] Starting — pool: ${self.pool.capital:,.0f}, "
              f"dry_run={self.dry_run}")
        self.pool.load()

        while True:
            # Circuit breaker check
            if self.pool.circuit_breaker_tripped():
                if not self.pool.paused:
                    print(f"  [Speculator] CIRCUIT BREAKER — daily loss "
                          f"${self.pool.daily_pnl:,.0f} exceeds limit")
                    self.pool.paused = True
                    self.pool.save()
                await asyncio.sleep(60)
                continue

            # Check timed exits
            await self.check_exits()

            # Read queue
            opportunities = self.read_queue()
            if opportunities:
                # Group by ticker
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
                        continue  # already have a position

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

            await asyncio.sleep(10)  # check queue every 10 seconds
```

**Step 4: Run tests to verify they pass**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_speculator.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add finance/speculator.py finance/tests/test_speculator.py
git commit -m "feat: speculator execution engine with AI evaluation and order placement"
```

---

### Task 7: Add pool tagging to LocalPortfolio in stream_trader.py

**Files:**
- Modify: `finance/stream_trader.py:103-110` (LocalPortfolio.__init__)
- Modify: `finance/stream_trader.py:112-159` (fill_order method)
- Test: `finance/tests/test_pool_tagging.py`

**Step 1: Write failing tests**

```python
# finance/tests/test_pool_tagging.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from stream_trader import LocalPortfolio


class TestPoolTagging:
    def test_default_pool_is_core(self):
        p = LocalPortfolio(100_000)
        p.fill_order('AAPL', 10, 150.0)
        assert p.positions['AAPL']['pool'] == 'core'

    def test_spec_pool_tag(self):
        p = LocalPortfolio(100_000)
        p.fill_order('NVDA', 5, 200.0, pool='spec')
        assert p.positions['NVDA']['pool'] == 'spec'

    def test_pool_value_by_type(self):
        p = LocalPortfolio(100_000)
        p.fill_order('AAPL', 10, 150.0, pool='core')
        p.fill_order('NVDA', 5, 200.0, pool='spec')
        prices = {'AAPL': 150.0, 'NVDA': 200.0}
        core_val = sum(
            pos['shares'] * prices.get(sym, pos['avg_cost'])
            for sym, pos in p.positions.items()
            if pos.get('pool') == 'core'
        )
        spec_val = sum(
            pos['shares'] * prices.get(sym, pos['avg_cost'])
            for sym, pos in p.positions.items()
            if pos.get('pool') == 'spec'
        )
        assert core_val == 1500.0
        assert spec_val == 1000.0

    def test_tax_events_include_pool(self):
        p = LocalPortfolio(100_000)
        p.fill_order('AAPL', 10, 150.0, pool='spec')
        p.fill_order('AAPL', -10, 140.0, pool='spec')
        assert p.tax_events[-1]['pool'] == 'spec'
```

**Step 2: Run test to verify it fails**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_pool_tagging.py -v`
Expected: FAIL — fill_order doesn't accept pool kwarg

**Step 3: Modify LocalPortfolio.fill_order to accept pool parameter**

In `finance/stream_trader.py`, modify the `fill_order` method signature:

Change line 112:
```python
    def fill_order(self, symbol, shares, price, pool='core'):
```

In the buy branch (line 117), add pool to position init:
```python
            pos = self.positions.get(symbol, {'shares': 0, 'avg_cost': 0.0, 'buy_date': now, 'pool': pool})
```

In the tax_events append (line 143), add:
```python
                    'pool': self.positions.get(symbol, {}).get('pool', pool),
```

**Step 4: Run tests to verify they pass**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/test_pool_tagging.py -v`
Expected: all PASS

**Step 5: Run existing stream_trader functionality to verify no regressions**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -c "from stream_trader import LocalPortfolio; p = LocalPortfolio(100000); p.fill_order('AAPL', 10, 150); print(p.positions); print('OK')"`
Expected: positions printed, 'OK'

**Step 6: Commit**

```bash
git add finance/stream_trader.py finance/tests/test_pool_tagging.py
git commit -m "feat: add pool tagging (core/spec) to LocalPortfolio"
```

---

### Task 8: Update market_open_runner.py to launch sentinel + speculator

**Files:**
- Modify: `finance/market_open_runner.py:156-162`

**Step 1: Modify Phase 5 to launch all three processes**

Replace the stream trader launch block (lines 157-162) with:

```python
    if not args.fill_only:
        print(f"\n  Phase 5: Starting trading system...")

        # Launch stream trader (core portfolio)
        stream_proc = run_script_background(
            'stream_trader.py', stream_args,
            f'Stream Trader ({" ".join(stream_args) or "dry run"})'
        )

        # Launch sentinel (social monitoring)
        sentinel_proc = run_script_background(
            'sentinel.py', live_flag,
            'Sentinel (social monitoring)'
        )

        # Launch speculator (autonomous spec trading)
        spec_args = execute_flag + live_flag
        spec_proc = run_script_background(
            'speculator.py', spec_args,
            f'Speculator ({" ".join(spec_args) or "dry run"})'
        )

        # Wait for all — stream_trader is the primary, others follow
        procs = [
            ('Stream Trader', stream_proc),
            ('Sentinel', sentinel_proc),
            ('Speculator', spec_proc),
        ]

        try:
            # Wait for stream_trader (the main loop) — if it exits, stop everything
            stream_proc.wait()
        except KeyboardInterrupt:
            print(f"\n  Ctrl+C — stopping all processes...")
        finally:
            for name, proc in procs:
                if proc.poll() is None:
                    proc.terminate()
                    print(f"  Terminated {name}")
```

**Step 2: Verify syntax**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -c "import ast; ast.parse(open('market_open_runner.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add finance/market_open_runner.py
git commit -m "feat: market_open_runner launches sentinel + speculator alongside stream trader"
```

---

### Task 9: Add sentinel.py and speculator.py CLI entry points

**Files:**
- Modify: `finance/sentinel.py` (add `if __name__ == '__main__'` block)
- Modify: `finance/speculator.py` (add `if __name__ == '__main__'` block)

**Step 1: Add sentinel CLI**

Add to bottom of `finance/sentinel.py`:

```python
async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sentinel — Social Monitor')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / '.env')
    exa_key = os.environ.get('EXA_API_KEY')
    if not exa_key:
        print("ERROR: EXA_API_KEY not set in .env")
        sys.exit(1)

    # Import watched tickers from stream_trader
    from stream_trader import TECH7, DEFENSIVES
    watched = TECH7 + DEFENSIVES

    monitor = SentinelMonitor(exa_api_key=exa_key, watched_tickers=watched)
    await monitor.run()


if __name__ == '__main__':
    import os
    import sys
    from dotenv import load_dotenv
    asyncio.run(main())
```

**Step 2: Add speculator CLI**

Add to bottom of `finance/speculator.py`:

```python
async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Speculator — Autonomous Trading')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')

    is_sandbox = not args.live
    dry_run = not args.execute

    from tastytrade import Session, Account
    session = Session(
        provider_secret=os.environ['TASTYTRADE_CLIENT_SECRET'],
        refresh_token=os.environ['TASTYTRADE_REFRESH_TOKEN'],
        is_test=is_sandbox,
    )
    accounts = await Account.get(session)
    account = accounts[0]
    bal = await account.get_balances(session)
    capital = float(bal.net_liquidating_value)

    from stream_trader import LocalPortfolio, SignalEngine
    import yfinance as yf
    from stream_trader import ALL_STREAM

    hist = yf.download(ALL_STREAM + ['^VIX', 'TLT', 'SHY'], period='120d',
                       interval='1d', auto_adjust=True)
    hist_closes = hist['Close'].dropna(how='all')
    engine = SignalEngine(hist_closes)
    portfolio = LocalPortfolio(capital)

    from config import SPEC_POOL_PCT
    pool = SpecPool(capital, SPEC_POOL_PCT)

    spec = Speculator(session, account, portfolio, pool, engine, dry_run)
    await spec.run()


if __name__ == '__main__':
    asyncio.run(main())
```

**Step 3: Verify both parse without error**

Run:
```bash
cd /mnt/d/Praxis/wavegpt/finance
python -c "import ast; ast.parse(open('sentinel.py').read()); print('sentinel OK')"
python -c "import ast; ast.parse(open('speculator.py').read()); print('speculator OK')"
```
Expected: both OK

**Step 4: Commit**

```bash
git add finance/sentinel.py finance/speculator.py
git commit -m "feat: add CLI entry points for sentinel and speculator"
```

---

### Task 10: Create tests/__init__.py and run full test suite

**Files:**
- Create: `finance/tests/__init__.py` (empty)

**Step 1: Create test package init**

```python
# finance/tests/__init__.py
```

**Step 2: Run full test suite**

Run: `cd /mnt/d/Praxis/wavegpt/finance && python -m pytest tests/ -v --tb=short`
Expected: all tests pass

**Step 3: Commit**

```bash
git add finance/tests/__init__.py
git commit -m "test: add tests package init, verify full suite passes"
```

---

### Task 11: Integration smoke test

**Step 1: Test sentinel evaluate_content end-to-end (no API calls)**

Run:
```bash
cd /mnt/d/Praxis/wavegpt/finance && python -c "
from sentinel import SentinelMonitor
mon = SentinelMonitor(exa_api_key='fake')
# High-signal content
results = mon.evaluate_content('\$NVDA squeeze to the moon! Breaking news!', source='test')
print(f'Flagged: {len(results)} opportunities')
for r in results:
    print(f'  {r[\"ticker\"]} score={r[\"score\"]:.1f} velocity={r[\"velocity_spike\"]}')
# Low-signal content
results2 = mon.evaluate_content('nice weather today', source='test')
print(f'Low signal: {len(results2)} (should be 0)')
print('SMOKE TEST PASSED')
"
```
Expected: 1 opportunity flagged for NVDA, 0 for low signal

**Step 2: Test speculator decision parsing**

Run:
```bash
cd /mnt/d/Praxis/wavegpt/finance && python -c "
from speculator import parse_ai_decision, SpecPool
import json

# Valid buy
d = parse_ai_decision(json.dumps({
    'action': 'buy', 'ticker': 'NVDA', 'conviction': 0.8,
    'reasoning': 'test', 'entry_strategy': 'market',
    'position_size_pct': 0.30, 'stop_loss_pct': 0.08,
    'target_pct': 0.15, 'exit_timeframe_hours': 24,
    'category': 'momentum',
}))
print(f'Buy decision: {d[\"action\"]} {d[\"ticker\"]}')

# Invalid (no stop loss)
d2 = parse_ai_decision(json.dumps({'action': 'buy', 'ticker': 'X'}))
print(f'Invalid buy: {d2} (should be None)')

# Pool
pool = SpecPool(100000, 0.10)
print(f'Pool: \${pool.capital:,.0f}, can_trade={pool.can_trade()}')
print('SMOKE TEST PASSED')
"
```
Expected: valid buy parsed, invalid rejected, pool initialized

**Step 3: Commit final integration test results**

No commit needed — smoke tests are manual verification.

---

## Summary

| Task | Component | What it builds |
|------|-----------|---------------|
| 1 | Dependencies | Install exa-py, add API keys |
| 2 | config.py | All shared constants |
| 3 | sentinel.py | Ticker extraction, keyword scoring, velocity tracking |
| 4 | sentinel.py | Exa integration, SentinelMonitor, dedup |
| 5 | speculator.py | AI prompt builder, decision parser, SpecPool |
| 6 | speculator.py | Full execution engine with order placement |
| 7 | stream_trader.py | Pool tagging on LocalPortfolio |
| 8 | market_open_runner.py | Launch all three processes |
| 9 | sentinel.py + speculator.py | CLI entry points |
| 10 | tests/ | Test package + full suite |
| 11 | Integration | End-to-end smoke tests |

**Note:** The existing `dashboard/core/llm_adapter.py` defaults to `anthropic.claude-sonnet-4-20250514-v1:0` — update its `BEDROCK_MODEL_ID` default to `anthropic.claude-opus-4-6-v1` as well to keep all AI calls on Opus.
