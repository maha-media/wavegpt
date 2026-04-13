# Test Mode — Stream Trader Dashboard Integration

## Problem
The TastyTrade sandbox doesn't reflect order fills or position updates in real-time. The stream trader (`stream_trader.py`) manages a $1M account autonomously but its state is only visible in terminal stdout. We need the dashboard to display this state.

## Design Decisions
- **Data source**: Structured JSON state file (`live_state.json`), not stdout parsing
- **Delivery**: Fold into existing SSE stream (same pipe as live mode)
- **File path**: `finance/trade_logs/live_state.json` (overwritten each tick)
- **Mode toggle**: `DASHBOARD_MODE=test` env var, clean switch — test mode replaces TastyTrade entirely
- **Frontend**: Same panels, new regime bar + test mode badge

## State File Contract

`finance/trade_logs/live_state.json` — written atomically by stream_trader.py every ~65s:

```json
{
  "timestamp": "2026-04-13T14:31:21",
  "regime": "RISK_ON",
  "leader_score": -0.050,
  "tech_pct": 0.148,
  "portfolio_value": 999143,
  "cash": 1484,
  "starting_capital": 1000000,
  "pnl": -857,
  "pnl_pct": -0.09,
  "positions": [
    {"symbol": "USO", "shares": 4990, "avg_cost": 129.61, "price": 129.44, "value": 645905, "pnl": -849, "tax_status": "ST+"}
  ],
  "orders": [],
  "ticks": 99293,
  "prices_connected": 30
}
```

## Implementation

### 1. stream_trader.py — write live_state.json
Add `write_live_state()` in the tick handler after portfolio computation. Atomic write (write to .tmp, rename).

### 2. core/test_client.py — TestClient class
Drop-in replacement for TastyClient:
- Same interface: subscribe(), unsubscribe(), snapshot_balances(), snapshot_positions(), snapshot_orders()
- run_stream() polls live_state.json every 2s, broadcasts on change
- Broadcasts quote events from position prices for ticker tape
- New snapshot_regime() for regime/signal data

### 3. core/tastytrade_client.py — route by mode
get_client() checks DASHBOARD_MODE env var:
- "test" → TestClient
- else → TastyClient (unchanged)

### 4. Frontend — regime bar + mode indicator
- Top bar: "TEST" badge + regime pill when in test mode
- Regime bar below balances: regime, leader score, tech %, ticks
- Backend sends `mode` event on SSE connect
- StreamContext stores mode + regime data
