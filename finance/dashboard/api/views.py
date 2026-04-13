"""API views — SSE stream, trading data REST, chat SSE."""

import asyncio
import json
import os
import time

from django.http import JsonResponse
from core.sse import format_sse, sse_response
from core.tastytrade_client import get_client
from core.llm_adapter import get_adapter, build_portfolio_context


async def stream(request):
    """SSE endpoint — multiplexes all live account/market/trading data."""
    client = await get_client()
    queue = client.subscribe()
    mode = os.environ.get('DASHBOARD_MODE', 'live')

    async def event_generator():
        try:
            yield format_sse({"mode": mode}, event="mode")
            yield format_sse(client.snapshot_balances(), event="balances")
            yield format_sse(client.snapshot_positions(), event="positions")
            yield format_sse(client.snapshot_orders(), event="orders")

            # Trading engine data
            if hasattr(client, 'snapshot_regime'):
                yield format_sse(client.snapshot_regime(), event="regime")
            if hasattr(client, 'snapshot_portfolio'):
                portfolio = client.snapshot_portfolio()
                if portfolio:
                    yield format_sse(portfolio, event="portfolio")
            if hasattr(client, 'snapshot_rebalances'):
                yield format_sse(client.snapshot_rebalances(), event="rebalance_log")

            while True:
                try:
                    event_type, data = await asyncio.wait_for(queue.get(), timeout=15)
                    yield format_sse(data, event=event_type)
                except asyncio.TimeoutError:
                    yield format_sse({"ts": time.time()}, event="heartbeat")
        finally:
            client.unsubscribe(queue)

    return sse_response(event_generator())


async def transactions(request):
    """REST endpoint — recent transaction history."""
    client = await get_client()
    if not client.account or not client.session:
        return JsonResponse([], safe=False)
    try:
        txns = await client.account.get_history(client.session)
        result = []
        for t in txns[:50]:
            result.append({
                "id": str(getattr(t, 'id', '')),
                "date": str(getattr(t, 'executed_at', getattr(t, 'transaction_date', ''))),
                "type": str(getattr(t, 'transaction_type', '')),
                "symbol": str(getattr(t, 'symbol', getattr(t, 'underlying_symbol', ''))),
                "amount": float(getattr(t, 'value', getattr(t, 'net_value', 0)) or 0),
                "description": str(getattr(t, 'description', '')),
            })
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


async def regime(request):
    """REST endpoint — current regime and signal data."""
    client = await get_client()
    data = client.snapshot_regime()
    return JsonResponse(data)


async def portfolio(request):
    """REST endpoint — full portfolio with tax data."""
    client = await get_client()
    data = client.snapshot_portfolio()
    if not data:
        return JsonResponse({
            "portfolio_value": 0, "cash": 0, "positions": [],
            "pnl": 0, "pnl_pct": 0, "tax": {},
        })
    return JsonResponse(data)


async def rebalances(request):
    """REST endpoint — recent rebalance events."""
    client = await get_client()
    data = client.snapshot_rebalances()
    return JsonResponse(data, safe=False)


async def chat(request):
    """SSE endpoint — streams AI response with live portfolio + regime context."""
    if request.method != 'POST':
        return JsonResponse({"error": "POST required"}, status=405)

    body = json.loads(request.body)
    messages = body.get("messages", [])

    client = await get_client()
    portfolio_ctx = build_portfolio_context(
        client.snapshot_balances(),
        client.snapshot_positions(),
        client.snapshot_orders(),
        client.snapshot_regime(),
        client.snapshot_portfolio(),
    )

    adapter = get_adapter()

    async def chat_generator():
        async for chunk in adapter.stream_response(messages, portfolio_ctx):
            yield format_sse({"text": chunk}, event="chat")
        yield format_sse({"done": True}, event="chat_done")

    return sse_response(chat_generator())
