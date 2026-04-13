import pytest
from core.tastytrade_client import TastyClient


def test_update_price_stores_quote():
    client = TastyClient.__new__(TastyClient)
    client.live_prices = {}
    client.update_price("NVDA", 189.12, 189.15)
    assert client.live_prices["NVDA"]["bid"] == 189.12
    assert client.live_prices["NVDA"]["ask"] == 189.15
    assert client.live_prices["NVDA"]["mid"] == pytest.approx(189.135)


def test_snapshot_positions():
    client = TastyClient.__new__(TastyClient)
    client.latest_positions = [
        {"symbol": "NVDA", "qty": 132, "avg_cost": 189.34}
    ]
    snap = client.snapshot_positions()
    assert snap[0]["symbol"] == "NVDA"
    assert snap is not client.latest_positions  # returns copy


def test_subscribe_unsubscribe():
    client = TastyClient.__new__(TastyClient)
    client._subscribers = []
    q = client.subscribe()
    assert len(client._subscribers) == 1
    client.unsubscribe(q)
    assert len(client._subscribers) == 0
