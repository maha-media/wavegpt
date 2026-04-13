# Sentinel + Speculator: AI-Powered Social Intelligence Trading

**Date:** 2026-04-13
**Status:** Approved design

## Summary

Add two new components to the trading system: **Sentinel** (social firehose monitoring + fuzzy matching) and **Speculator** (autonomous AI-driven speculative trading). The existing stream trader continues managing 90% of capital via regime/momentum. The speculative pool (10%) trades autonomously based on social signals from X, Reddit, and news — including pump & dump plays.

## Architecture

```
Exa Livecrawl ──→ Sentinel (fuzzy match) ──→ AI Prompt ──→ Speculator ──→ TastyTrade
Exa Search ──────→   (scheduled sweeps)        │               │
                                                │               ├── stop-loss brackets
                                                │               ├── position tracking
                                                │               └── daily circuit breaker
                                                │
                                          Universe Discovery ──→ TECH7 / DEFENSIVES watchlist
```

Two capital pools:
- **Core (90%)**: regime rotation + momentum, tax-aware rebalancing, conservative
- **Speculative (10%)**: AI-autonomous, social-signal-driven, max 5 positions

Pools share LocalPortfolio for unified P/L and tax tracking. Positions tagged `core` or `spec`.

## Component 1: Sentinel

Long-running process. Three input streams, one cheap filter, expensive AI only when it matters.

### Input Streams

1. **Exa Livecrawl** — persistent connection streaming X and Reddit content in real-time. Filtered to financial/investing domains. The firehose for catching pumps early.

2. **Exa Search (fast poll, every 5 min)** — targeted searches: `$TICKER momentum`, `short squeeze`, `FDA approval`, etc. Catches things livecrawl might miss. Also searches for tickers in our existing universe to track sentiment shifts on holdings.

3. **Exa Search (slow sweep, every 30-60 min)** — broader discovery: "AI company IPO 2026", "trending stocks Reddit today", sector-specific queries. Feeds the universe expansion watchlist, not the speculator.

### Fuzzy Matching Algorithm

For each incoming item, three cheap checks:

- **Ticker extraction** — regex for $CASHTAGS, fuzzy match company names against known list + dynamic additions. Cost: microseconds.
- **Keyword score** — weighted match against signal words ("moon", "squeeze", "calls", "puts", "breaking", "approval", "earnings"). Bearish words scored separately. Cost: microseconds.
- **Velocity tracking** — per-ticker sliding window counter. If mentions/minute exceeds 3x the trailing average, flag regardless of keywords. Cost: one dict lookup.

### AI Trigger Threshold

Ticker identified AND (keyword score > threshold OR velocity spike). Deduplicate — don't re-prompt on the same ticker within 10 minutes unless velocity doubles again.

## Component 2: Speculator

### AI Prompt — Opportunity Evaluation

When Sentinel flags a ticker, it packages context and sends a prompt. The AI returns a structured decision.

**Context sent:**
- Raw posts/content that triggered the alert (last 10-20 items)
- Mention velocity graph (last hour)
- Current price + 5d/20d momentum (from yfinance)
- Whether we already hold this ticker (in either pool)
- Current speculative pool status: position count, available capital, daily P/L

**AI returns JSON:**

```json
{
  "action": "buy | pass | watch",
  "ticker": "NVDA",
  "conviction": 0.0-1.0,
  "reasoning": "Short squeeze setup, 3x normal X volume, SI at 25%...",
  "entry_strategy": "market | limit",
  "limit_price": 192.50,
  "position_size_pct": 0.30,
  "stop_loss_pct": 0.08,
  "target_pct": 0.15,
  "exit_timeframe_hours": 24,
  "category": "pump_and_dump | momentum | news_catalyst | earnings"
}
```

**Rules baked into the prompt:**
- Position size is a percentage of the speculative pool, not dollar amount
- Must set a stop-loss percentage — system brackets the order immediately
- "watch" = add to hot list, re-evaluate on next Sentinel trigger
- If already held in core portfolio, account for total exposure

### Execution Engine

1. AI returns `"action": "buy"` with parameters
2. Check: pool has capital? Under 5 positions? Not in daily pause? → proceed
3. Calculate shares: `pool_capital * position_size_pct / price`
4. Place market order (or limit if specified)
5. **Immediately** place stop-loss at `entry_price * (1 - stop_loss_pct)`
6. If `target_pct` set, place limit sell at `entry_price * (1 + target_pct)`
7. Record entry in speculative position book with exit timeframe
8. Log everything: AI reasoning, triggering posts, entry price, stop level

### Position Management (continuous)

- Monitor stop-losses (TastyTrade handles server-side, tracked locally too)
- Check exit timeframes — close position when `exit_timeframe_hours` expires regardless of P/L
- Bracket orders: every position has a stop-loss floor and optionally a take-profit ceiling

### Daily Circuit Breaker

- Track speculative pool P/L from market open
- If unrealized + realized losses exceed configurable threshold in a single session → close all speculative positions, pause until next market open
- Log the pause with reasoning

## Component 3: Universe Discovery

The slow side of Sentinel. Not for trading — for evolving the core portfolio's ticker lists.

### Scheduled Exa Searches

- `"AI company IPO 2026 filing S-1"` — catch Anthropic, OpenAI, others
- `"new ETF launch 2026"` — sector ETFs for defensive buckets
- `"stock market trending Reddit wallstreetbets"` — retail sentiment
- Sector-specific: semiconductors, energy transition, defense, etc.

### New Ticker Pipeline

When a ticker surfaces repeatedly (3+ hits across different searches):

1. Pull price history from yfinance — needs 50+ days for momentum
2. Check market cap — minimum threshold, no penny stocks in core
3. Classify: correlates with TECH7 or a defensive sector?
4. Add to `watchlist.json` with metadata: first seen, source, category, market cap, correlation

### Graduation

Manual for core universe. System writes periodic report: "N new tickers on watchlist with 50+ days history, here's correlation data." User reviews and approves.

Speculative pool can trade watchlist tickers immediately — if Sentinel flags a watchlist ticker, the spec pool can take a position before it graduates.

## File Structure

```
finance/
  sentinel.py          — Livecrawl stream + Exa search + fuzzy matcher
  speculator.py        — AI evaluation + autonomous execution + position mgmt
  watchlist.json       — discovered tickers with metadata
  spec_positions.json  — speculative position book (persisted across restarts)
  config.py            — shared constants: SPEC_POOL_PCT, MAX_SPEC_POSITIONS,
                         DAILY_LOSS_LIMIT_PCT, keyword weights, velocity
                         thresholds, search queries
```

### Changes to Existing Files

- `stream_trader.py` — LocalPortfolio gets `pool` tag per position (`core`/`spec`). Core capital = `total_nlv * (1 - SPEC_POOL_PCT)`. Tax events track which pool.
- `market_open_runner.py` — launches Sentinel and Speculator alongside stream trader.

### Runtime

```
market_open_runner.py
  ├── stream_trader.py    (core portfolio, all day)
  ├── sentinel.py         (social firehose, all day)
  └── speculator.py       (spec execution, all day, pauses on circuit breaker)
```

### Communication

Sentinel → Speculator via JSON queue file. Sentinel appends opportunities, Speculator pops and executes. No message broker.

## Configuration (all in config.py, percentage-based)

```python
SPEC_POOL_PCT = 0.10           # 10% of total capital
MAX_SPEC_POSITIONS = 5
DAILY_LOSS_LIMIT_PCT = 0.05    # 5% of spec pool triggers daily pause
VELOCITY_SPIKE_MULT = 3.0      # 3x trailing avg = spike
AI_DEDUP_WINDOW_SEC = 600      # don't re-prompt same ticker within 10 min
WATCHLIST_MIN_HISTORY_DAYS = 50
WATCHLIST_MIN_MARKET_CAP = 1e9  # $1B minimum for core consideration
```

## Tax Integration

Both pools share the LocalPortfolio tax tracker:
- Speculative losses offset core gains dollar-for-dollar
- Harvested core losses offset speculative gains
- Stop-loss exits on spec positions automatically generate loss events
- All positions tagged with pool for reporting
