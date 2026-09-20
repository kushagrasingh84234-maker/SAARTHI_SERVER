"""Web chat endpoint for Saarthi: POST /api/chat (SSE) and GET /api/health.

This module connects the web channel modules (sse_utils, stream_ai,
search_tools, web_prompts) to the existing database.py / logic.py helpers.
It never imports server.py (that would be a circular import) and never uses
the robot-only pipeline (text_utils.parse_ai_reply, game_manager,
run_personalized_pipeline, ...).

Event order on POST /api/chat::

    status -> [search x N -> sources] -> status -> token x many -> done

Once the stream has started, every failure is reported as an ``error`` event
and the stream always ends with either ``done`` or ``error`` (unless the
client has already disconnected).

Internal keys:
    web_session = "web_" + session_id                 (long-term memory)
    history_key = f"{web_session}:{conversation_id}"  (short-term history)

Logging: session ids are hashed, and message text and API keys are never
logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import aclosing
from typing import AsyncIterator, Literal, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from config import (
    GROQ_MODEL,
    SEARCH_MAX_RESULTS,
    WEB_MAX_CONCURRENT_STREAMS,
    WEB_MAX_MESSAGE_CHARS,
    WEB_MAX_SOURCES,
    WEB_LOCAL_SHORTCUTS,
    WEB_MAX_TOKENS,
    WEB_PLANNER_MAX_TOKENS,
    WEB_RATE_LIMIT_PER_MIN,
    WEB_STREAM_TIMEOUT_SECONDS,
    WEB_TEMPERATURE,
)
from database import (
    add_to_history,
    build_groq_messages,
    retrieve_long_term_context,
    route_and_save_bg,
)
from logic import handle_local_queries
from web_channel.search_tools import SearchResult, search_enabled, search_many
from web_channel.sse_utils import SSE_HEADERS, SSE_MEDIA_TYPE, sse_event
from web_channel.stream_ai import AIServiceError, groq_complete, groq_stream
from web_channel.web_prompts import (
    build_answer_messages,
    build_planner_messages,
    parse_planner_output,
)

logger = logging.getLogger(__name__)

__all__ = ["router"]

router = APIRouter(prefix="/api")

# --- Constants ---------------------------------------------------------------

_SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"
_CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# Control characters removed from user text (keeps \n and \t): C0 controls
# except \t (0x09) and \n (0x0a), DEL, and C1 controls.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_DISCONNECT_CHECK_EVERY_CHUNKS = 10
_PLANNER_TIMEOUT_SECONDS = 12.0
_PLANNER_HISTORY_TURNS = 4

_MSG_TOO_MANY = "Too many requests, please slow down."
_MSG_BUSY = "Server is busy, try again shortly."
_MSG_TOO_LONG = "The answer took too long. Please try again."
_MSG_EMPTY = "The AI returned an empty reply. Please try again."
_MSG_GENERIC = "Something went wrong. Please try again."


# --- Request model -----------------------------------------------------------

def clean_web_message(text: str) -> str:
    """Normalize a user message for the web channel.

    Converts line endings to ``\\n``, removes control characters except
    ``\\n`` and ``\\t``, strips surrounding whitespace, and cuts the text to
    ``WEB_MAX_MESSAGE_CHARS``. Non-ASCII text (Hindi, etc.) is left intact.

    Returns:
        The cleaned message, or ``""`` if nothing is left.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text).strip()
    if len(text) > WEB_MAX_MESSAGE_CHARS:
        text = text[:WEB_MAX_MESSAGE_CHARS].rstrip()
    return text


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat``. Invalid input yields FastAPI's normal 422."""

    message: str
    channel: Literal["web"] = "web"
    session_id: str = Field(..., pattern=_SESSION_ID_PATTERN)
    conversation_id: str = Field(..., pattern=_CONVERSATION_ID_PATTERN)
    images: Optional[list[dict]] = None  # accepted and ignored for now

    @field_validator("message")
    @classmethod
    def _clean_message(cls, value: str) -> str:
        """Clean the message; reject it (422) if nothing is left."""
        cleaned = clean_web_message(value)
        if not cleaned:
            raise ValueError("message must not be empty")
        return cleaned


# --- Small helpers -----------------------------------------------------------

def _short_id(session_id: str) -> str:
    """Return a short one-way hash of a session id for safe logging."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8]


def _client_ip(request: Request) -> str:
    """Return the client IP: first X-Forwarded-For value, else the socket peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first[:64]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


# --- Rate limiting (in-memory sliding window) --------------------------------

_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_SWEEP_INTERVAL = 500
_rate_limit_store: defaultdict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()
_rate_limit_call_counter = 0


def _is_rate_limited(key: str) -> bool:
    """Record a hit for ``key`` and return True if it is over the limit.

    Sliding window of 60 seconds, ``WEB_RATE_LIMIT_PER_MIN`` hits per key.
    Every ``_RATE_LIMIT_SWEEP_INTERVAL`` calls, keys whose newest hit has
    aged out of the window are dropped so the store cannot grow forever.
    """
    global _rate_limit_call_counter
    now = time.time()
    with _rate_limit_lock:
        timestamps = _rate_limit_store[key]
        while timestamps and now - timestamps[0] > _RATE_LIMIT_WINDOW_SECONDS:
            timestamps.popleft()

        limited = len(timestamps) >= WEB_RATE_LIMIT_PER_MIN
        if not limited:
            timestamps.append(now)

        _rate_limit_call_counter += 1
        if _rate_limit_call_counter >= _RATE_LIMIT_SWEEP_INTERVAL:
            _rate_limit_call_counter = 0
            stale = [
                k for k, dq in _rate_limit_store.items()
                if not dq or now - dq[-1] > _RATE_LIMIT_WINDOW_SECONDS
            ]
            for k in stale:
                del _rate_limit_store[k]

        return limited


def _request_is_rate_limited(ip: str, session_id: str) -> bool:
    """Check the per-IP and per-session limits (either one can trigger)."""
    keys = [f"session:{session_id}"]
    if ip != "unknown":  # avoid bucketing every unknown client together
        keys.insert(0, f"ip:{ip}")
    return any(_is_rate_limited(key) for key in keys)


# --- Concurrency limit -------------------------------------------------------

_stream_semaphore = asyncio.Semaphore(WEB_MAX_CONCURRENT_STREAMS)


class _StreamSlot:
    """A held concurrency slot that can be released safely more than once."""

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        """Wrap an already-acquired semaphore slot."""
        self._semaphore = semaphore
        self._released = False

    def release(self) -> None:
        """Release the slot (idempotent)."""
        if not self._released:
            self._released = True
            self._semaphore.release()


# --- Pipeline steps ----------------------------------------------------------

async def _plan_search(message: str, history_key: str) -> list[str]:
    """Ask the planner model whether web search is needed.

    Returns:
        A list of search queries, or ``[]`` when no search is needed or when
        anything at all goes wrong (failures are logged, never shown).
    """
    try:
        history = await asyncio.to_thread(build_groq_messages, "", history_key)
        turns = history[1:]  # drop the system message
        # The current user message is already in history but is passed to the
        # planner separately, so leave it out of the "recent" context.
        if turns and turns[-1].get("role") == "user":
            turns = turns[:-1]
        recent = turns[-_PLANNER_HISTORY_TURNS:]

        planner_messages = build_planner_messages(message, recent)
        raw = await groq_complete(
            planner_messages,
            max_tokens=WEB_PLANNER_MAX_TOKENS,
            temperature=0,
            timeout=_PLANNER_TIMEOUT_SECONDS,
        )
        if not (raw or "").strip():
            logger.info("planner returned empty output")
        needs_search, queries = parse_planner_output(raw)
        return queries if needs_search else []
    except AIServiceError as exc:
        logger.info("Search planner unavailable (status=%s); skipping search.", exc.status_code)
        return []
    except Exception as exc:  # noqa: BLE001 - planning must never break a reply
        logger.warning("Search planner failed (%s); skipping search.", type(exc).__name__)
        return []


async def _collect_sources(queries: list[str]) -> list[SearchResult]:
    """Run all queries, merge round-robin, de-duplicate by URL, cap the total."""
    per_query = await search_many(queries, SEARCH_MAX_RESULTS)
    merged: list[SearchResult] = []
    seen: set[str] = set()
    depth = max((len(results) for results in per_query), default=0)
    for rank in range(depth):
        for results in per_query:
            if rank >= len(results):
                continue
            item = results[rank]
            key = item.url.rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= WEB_MAX_SOURCES:
                return merged
    return merged


def _save_long_term(web_session: str, message: str, reply: str) -> None:
    """Fire-and-forget long-term persistence of one exchange (never raises)."""
    try:
        route_and_save_bg("user", message, web_session)
        route_and_save_bg("assistant", reply, web_session)
    except Exception as exc:  # noqa: BLE001 - persistence must not break the stream
        logger.warning("Long-term save failed (%s)", type(exc).__name__)


async def _event_stream(
    request: Request,
    message: str,
    session_id: str,
    conversation_id: str,
    has_images: bool,
    slot: _StreamSlot,
) -> AsyncIterator[str]:
    """Generate the SSE events for one chat request.

    Always ends with ``done`` or ``error`` (unless the client disconnected)
    and always releases the concurrency slot.
    """
    web_session = f"web_{session_id}"
    history_key = f"{web_session}:{conversation_id}"
    sid = _short_id(session_id)
    started = time.monotonic()
    deadline = started + WEB_STREAM_TIMEOUT_SECONDS

    try:
        yield sse_event("status", {"text": "Thinking..."})
        if has_images:
            yield sse_event("status", {"text": "Image understanding is not available yet."})

        # 1) Local shortcut (date, time, simple maths): no AI call needed.
        local = (
            await asyncio.to_thread(handle_local_queries, message)
            if WEB_LOCAL_SHORTCUTS
            else None
        )
        if local is not None:
            tag, sep, text = local.partition("|")
            if not sep:
                tag, text = "NORMAL", local
            emotion = tag.strip().upper() or "NORMAL"
            await asyncio.to_thread(add_to_history, "user", message, history_key)
            await asyncio.to_thread(add_to_history, "model", text, history_key)
            _save_long_term(web_session, message, text)
            yield sse_event("token", {"text": text})
            yield sse_event("done", {"emotion": emotion})
            logger.info("Web chat answered locally (session=%s)", sid)
            return

        # 2) Memory and history.
        long_term_context = await asyncio.to_thread(
            retrieve_long_term_context, message, web_session
        )
        await asyncio.to_thread(add_to_history, "user", message, history_key)

        # 3) Optional web search.
        sources: list[SearchResult] = []
        if search_enabled():
            queries = await _plan_search(message, history_key)
            if queries:
                for query in queries:
                    yield sse_event("search", {"query": query})
                sources = await _collect_sources(queries)
                if sources:
                    yield sse_event(
                        "sources",
                        {
                            "items": [
                                {"title": s.title, "url": s.url, "domain": s.domain}
                                for s in sources
                            ]
                        },
                    )
                else:
                    yield sse_event(
                        "status",
                        {"text": "No web results found, answering from my own knowledge."},
                    )

        # 4) Stream the answer.
        yield sse_event("status", {"text": "Writing the answer..."})
        base = await asyncio.to_thread(build_groq_messages, "", history_key)
        messages = build_answer_messages(base, long_term_context, sources)

        parts: list[str] = []
        chunk_count = 0
        timed_out = False
        stream = groq_stream(messages, max_tokens=WEB_MAX_TOKENS, temperature=WEB_TEMPERATURE)
        async with aclosing(stream):
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    timed_out = True
                    break

                parts.append(chunk)
                yield sse_event("token", {"text": chunk})

                chunk_count += 1
                if chunk_count % _DISCONNECT_CHECK_EVERY_CHUNKS == 0:
                    if await request.is_disconnected():
                        logger.info("Client disconnected mid-stream (session=%s)", sid)
                        return

        if timed_out:
            logger.warning("Web chat stream hit the time limit (session=%s)", sid)
            yield sse_event("error", {"text": _MSG_TOO_LONG})
            return

        final_text = "".join(parts)
        if not final_text.strip():
            logger.warning("AI returned an empty reply (session=%s)", sid)
            yield sse_event("error", {"text": _MSG_EMPTY})
            return

        # 5) Save the exchange, then finish.
        await asyncio.to_thread(add_to_history, "model", final_text, history_key)
        _save_long_term(web_session, message, final_text)
        yield sse_event("done", {"emotion": "NORMAL"})
        logger.info(
            "Web chat finished (session=%s, chunks=%d, sources=%d, seconds=%.1f)",
            sid, chunk_count, len(sources), time.monotonic() - started,
        )

    except AIServiceError as exc:
        logger.warning("AI service error in web chat (session=%s, status=%s)", sid, exc.status_code)
        yield sse_event("error", {"text": exc.user_message})
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Unexpected error in web chat stream (session=%s)", sid)
        yield sse_event("error", {"text": _MSG_GENERIC})
    finally:
        slot.release()


# --- Endpoints ---------------------------------------------------------------

@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    """Stream a chat answer as Server-Sent Events (see module docstring)."""
    ip = _client_ip(request)
    if _request_is_rate_limited(ip, body.session_id):
        return JSONResponse(
            {"error": _MSG_TOO_MANY},
            status_code=429,
            headers={"Retry-After": "30"},
        )

    # No free slot right now -> 503 immediately instead of queueing.
    if _stream_semaphore.locked():
        return JSONResponse({"error": _MSG_BUSY}, status_code=503)
    await _stream_semaphore.acquire()  # returns at once: a slot is free
    slot = _StreamSlot(_stream_semaphore)

    logger.info(
        "Web chat request (session=%s, chars=%d, images=%s)",
        _short_id(body.session_id), len(body.message), bool(body.images),
    )

    generator = _event_stream(
        request=request,
        message=body.message,
        session_id=body.session_id,
        conversation_id=body.conversation_id,
        has_images=bool(body.images),
        slot=slot,
    )
    return StreamingResponse(
        generator,
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
        # Safety net: frees the slot even if the generator never got to run.
        background=BackgroundTask(slot.release),
    )


@router.get("/health")
async def web_health() -> dict:
    """Return basic web channel status (no secrets)."""
    return {"status": "ok", "model": GROQ_MODEL, "search_enabled": search_enabled()}
