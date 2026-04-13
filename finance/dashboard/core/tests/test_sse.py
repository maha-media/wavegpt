import json
from core.sse import format_sse


def test_format_sse_with_event_type():
    result = format_sse({"price": 189.12}, event="quote")
    assert result == 'event: quote\ndata: {"price": 189.12}\n\n'


def test_format_sse_with_id():
    result = format_sse({"ok": True}, event="heartbeat", event_id="42")
    assert "id: 42\n" in result
    assert "event: heartbeat\n" in result


def test_format_sse_data_only():
    result = format_sse({"msg": "hello"})
    assert result.startswith("data: ")
    assert "event:" not in result
    parsed = json.loads(result.strip().removeprefix("data: "))
    assert parsed["msg"] == "hello"
