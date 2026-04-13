import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        assert extract_tickers('$I think $A stock is good') == []

    def test_uppercase_only(self):
        assert extract_tickers('$nvda') == []


class TestKeywordScore:
    def test_bullish_keywords(self):
        score = compute_keyword_score('NVDA to the moon, squeeze incoming!')
        assert score > 5.0

    def test_bearish_keywords(self):
        score = compute_keyword_score('this stock is a fraud, SEC investigation')
        assert score > 5.0

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
        for i in range(10):
            vt.record_and_check('NVDA', base_time + i)
        for i in range(10):
            result = vt.record_and_check('NVDA', base_time + 11)
        assert result is True
