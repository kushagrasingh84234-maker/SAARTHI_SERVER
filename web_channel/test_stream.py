#!/usr/bin/env python3
"""Standalone test client for the Studybot web chat stream (POST /api/chat).

Usage (works the same in Windows PowerShell, cmd, macOS and Linux):

    python test_stream.py "your message"

Test a deployed server by setting STUDYBOT_URL first.

    PowerShell:  $env:STUDYBOT_URL = "https://your-service.onrender.com"
    macOS/Linux: export STUDYBOT_URL=https://your-service.onrender.com

Each event is printed as "[event] data", so you can check the order:
status -> [search -> sources] -> status -> token ... -> done.
The script exits with code 0 if the stream ended with "done", otherwise 1.
Only httpx is required (already in requirements.txt).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

import httpx

DEFAULT_URL = "http://localhost:5000"
SESSION_ID = "test_session_01"
CONVERSATION_ID = "c1"
TIMEOUT = httpx.Timeout(connect=10.0, read=130.0, write=10.0, pool=10.0)


def _configure_output() -> None:
    """Force UTF-8 output so Hindi prints correctly, e.g. in PowerShell."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - best effort only
                pass


def main(argv: list[str]) -> int:
    """Send one message, print every SSE event, and return an exit code."""
    _configure_output()

    message = " ".join(argv[1:]).strip()
    if not message:
        print('Usage: python test_stream.py "your message"', file=sys.stderr)
        return 2

    base_url = (os.environ.get("STUDYBOT_URL") or DEFAULT_URL).strip().rstrip("/")
    url = f"{base_url}/api/chat"
    payload = {
        "message": message,
        "channel": "web",
        "session_id": SESSION_ID,
        "conversation_id": CONVERSATION_ID,
    }

    print(f"POST {url}")
    counts: Counter[str] = Counter()
    answer_parts: list[str] = []
    last_event = ""
    first_token_at: float | None = None
    started = time.monotonic()

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream(
                "POST", url, json=payload, headers={"Accept": "text/event-stream"}
            ) as response:
                content_type = response.headers.get("content-type", "")
                print(f"HTTP {response.status_code} ({content_type})\n")

                if response.status_code != 200:
                    body = response.read().decode("utf-8", errors="replace")
                    print(body)
                    return 1

                response.encoding = "utf-8"  # decode the stream as UTF-8
                event_name = "message"

                for line in response.iter_lines():
                    if line == "":
                        event_name = "message"  # blank line ends an event
                        continue
                    if line.startswith(":"):
                        continue  # SSE comment / keep-alive
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):]
                    if data.startswith(" "):
                        data = data[1:]

                    counts[event_name] += 1
                    last_event = event_name
                    print(f"[{event_name}] {data}")

                    if event_name == "token":
                        if first_token_at is None:
                            first_token_at = time.monotonic() - started
                        try:
                            answer_parts.append(str(json.loads(data).get("text", "")))
                        except (ValueError, AttributeError):
                            pass

    except httpx.ConnectError:
        print(f"\nCould not connect to {base_url}. Is the server running?", file=sys.stderr)
        return 1
    except httpx.TimeoutException:
        print("\nThe request timed out.", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"\nRequest failed: {type(exc).__name__}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    print("\n--- Assembled answer ---")
    print("".join(answer_parts) or "(no tokens received)")
    print("\n--- Summary ---")
    print("Events: " + (", ".join(f"{name}={n}" for name, n in counts.items()) or "none"))
    if first_token_at is not None:
        print(f"First token after {first_token_at:.1f}s, total {elapsed:.1f}s")
    else:
        print(f"Total {elapsed:.1f}s")
    print(f"Last event: {last_event or 'none'}")

    return 0 if last_event == "done" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
