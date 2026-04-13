import json
from django.http import StreamingHttpResponse


def format_sse(data, event=None, event_id=None):
    """Format a dict as an SSE message string."""
    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, default=str)}")
    return "\n".join(lines) + "\n\n"


def sse_response(generator):
    """Wrap an async generator as a StreamingHttpResponse with SSE headers."""
    response = StreamingHttpResponse(
        generator,
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
