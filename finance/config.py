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
BEDROCK_MODEL_ID = 'us.anthropic.claude-opus-4-6-v1'
