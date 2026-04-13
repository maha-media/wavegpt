import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
