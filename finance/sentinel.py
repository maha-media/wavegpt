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
        older = len(times) - recent
        avg_rate = max(older / 2, 1)  # expected per third from older portion
        return recent > avg_rate * self.spike_mult
