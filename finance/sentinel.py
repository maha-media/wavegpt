"""Sentinel — social firehose monitoring with fuzzy matching."""

import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from config import (
    SIGNAL_KEYWORDS, AI_DEDUP_WINDOW_SEC, KEYWORD_TRIGGER_SCORE,
    VELOCITY_SPIKE_MULT, FAST_SEARCH_QUERIES, SLOW_SEARCH_QUERIES,
    SENTINEL_FAST_POLL_SEC, SENTINEL_SLOW_POLL_SEC, QUEUE_FILE,
)

QUEUE_PATH = Path(__file__).parent / QUEUE_FILE

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


async def main():
    import argparse
    import os
    import sys
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description='Sentinel — Social Monitor')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / '.env')
    exa_key = os.environ.get('EXA_API_KEY')
    if not exa_key:
        print("ERROR: EXA_API_KEY not set in .env")
        sys.exit(1)

    from stream_trader import TECH7, DEFENSIVES
    watched = TECH7 + DEFENSIVES

    monitor = SentinelMonitor(exa_api_key=exa_key, watched_tickers=watched)
    await monitor.run()


if __name__ == '__main__':
    asyncio.run(main())
