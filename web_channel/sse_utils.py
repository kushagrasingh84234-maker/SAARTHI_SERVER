"""Server-Sent Events (SSE) helpers for the Studybot web chat channel.

This module has no dependencies beyond the standard library.

Event contract for ``POST /api/chat`` (response type ``text/event-stream``,
UTF-8). Each event is serialized as::

    event: <name>\\ndata: <json>\\n\\n

where ``<json>`` is produced with ``json.dumps(..., ensure_ascii=False)``, so
non-ASCII text such as Hindi is sent as-is and not as ``\\\\uXXXX`` escapes.

Events and their JSON payloads:

    status   {"text": str}
    search   {"query": str}
    sources  {"items": [{"title": str, "url": str, "domain": str}, ...]}
    token    {"text": str}
    done     {"emotion": str}
    error    {"text": str}

Order of events::

    status -> [search x N -> sources] -> status -> token x many -> done

Once the stream has started, failures are reported as an ``error`` event
rather than an HTTP error status.
"""

from __future__ import annotations

import json
from typing import Any

# The complete set of event names the web channel is allowed to emit.
ALLOWED_EVENTS: frozenset[str] = frozenset(
    {"status", "search", "sources", "token", "done", "error"}
)

# Media type for the streaming response.
SSE_MEDIA_TYPE: str = "text/event-stream"

# Headers that keep proxies (Render, nginx) from caching or buffering the stream.
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_event(name: str, data: dict[str, Any]) -> str:
    """Serialize one Server-Sent Event.

    The JSON payload is always emitted on a single line: compact separators
    are used and ``json.dumps`` escapes any newline inside string values as
    ``\\n``, so a raw line break can never split the ``data:`` field.

    Args:
        name: Event name. Must be one of ``ALLOWED_EVENTS``.
        data: JSON-serializable payload for the event.

    Returns:
        A string of the form ``"event: <name>\\ndata: <json>\\n\\n"``.

    Raises:
        ValueError: If ``name`` is not an allowed event name.
    """
    if name not in ALLOWED_EVENTS:
        raise ValueError(f"Unknown SSE event name: {name!r}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n"


if __name__ == "__main__":
    # Tiny self-check: Hindi text must stay unescaped, and the payload must
    # remain on a single data line.
    print(sse_event("token", {"text": "नमस्ते"}))
