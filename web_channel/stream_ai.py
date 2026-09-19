"""Groq client used ONLY by the web chat channel (POST /api/chat).

``ai_services.call_groq`` is untouched and is still used by the robot
(WebSocket) path exactly as before. This module is a separate, async httpx
client that adds what the web channel needs: real token streaming and errors
that are safe to show to end users.

Public API:
    AIServiceError  Exception carrying a ``user_message`` that is safe to show.
    groq_complete   Non-streaming completion (used for the planner step).
    groq_stream     Async generator yielding answer text chunks as they arrive.

Retry rules (both functions): up to ``MAX_RETRIES`` attempts in total on
timeouts, connection errors and HTTP 429/500/502/503, sleeping
``RETRY_BACKOFF_SECONDS * attempt`` between attempts (or the ``Retry-After``
header when present, capped at 10 seconds). ``groq_stream`` only retries
before the first token has been yielded; after that, a failure raises
``AIServiceError`` immediately so the caller never sees duplicated text.

Logging: only status codes and attempt numbers. Message contents, model
output and the API key are never logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

from config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    MAX_RETRIES,
    RETRY_BACKOFF_SECONDS,
    WEB_REASONING_EFFORT,
)

logger = logging.getLogger(__name__)

__all__ = ["AIServiceError", "groq_complete", "groq_stream"]

# HTTP statuses worth retrying (rate limit and transient server errors).
_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503})

# Upper bound for honoring a Retry-After header, in seconds.
_MAX_RETRY_AFTER_SECONDS: float = 10.0

# NEW (F4): set to True the first time Groq responds HTTP 400 while the
# "reasoning_effort" field is present. Once True, _build_payload stops
# adding the field for the rest of the process, so we don't repeatedly
# pay for a 400 + retry on every call.
_reasoning_effort_unsupported: bool = False

# Safe, generic messages. These never contain internal details.
_MSG_BUSY = "The AI service is busy. Please try again in a moment."
_MSG_UNAVAILABLE = "The AI service is not available right now. Please try again later."
_MSG_GENERIC = "Something went wrong while generating a reply. Please try again."
_MSG_INTERRUPTED = "The connection to the AI service was interrupted. Please try again."


class AIServiceError(Exception):
    """Raised when the AI service cannot produce a result.

    Attributes:
        user_message: Text that is safe to show to end users. It never
            contains internal details, URLs, or keys.
        status_code: Upstream HTTP status when known (for server-side
            handling only; do not show to users).
    """

    def __init__(self, user_message: str = _MSG_GENERIC, *, status_code: int | None = None) -> None:
        """Create the error.

        Args:
            user_message: User-safe description of the failure.
            status_code: Optional upstream HTTP status code.
        """
        super().__init__(user_message)
        self.user_message: str = user_message
        self.status_code: int | None = status_code


def _max_attempts() -> int:
    """Return the total number of attempts (at least 1)."""
    return max(1, int(MAX_RETRIES))


def _require_key() -> str:
    """Return the Groq API key or raise ``AIServiceError`` if it is missing."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not configured; web AI calls are unavailable.")
        raise AIServiceError(_MSG_UNAVAILABLE)
    return GROQ_API_KEY


def _timeout(read_timeout: float) -> httpx.Timeout:
    """Build the httpx timeout used for Groq requests."""
    return httpx.Timeout(connect=10, read=read_timeout, write=10, pool=10)


def _backoff_delay(attempt: int) -> float:
    """Return the default sleep before retrying after ``attempt``."""
    return float(RETRY_BACKOFF_SECONDS) * attempt


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Return how long to sleep before the next attempt.

    Honors a numeric ``Retry-After`` header (capped at 10 seconds) and falls
    back to ``RETRY_BACKOFF_SECONDS * attempt``.
    """
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return min(max(float(raw), 0.0), _MAX_RETRY_AFTER_SECONDS)
        except (TypeError, ValueError):
            pass
    return _backoff_delay(attempt)


def _error_for_status(status: int) -> AIServiceError:
    """Map a final (non-retried or exhausted) HTTP status to a safe error."""
    if status in _RETRYABLE_STATUSES:
        return AIServiceError(_MSG_BUSY, status_code=status)
    if status in (401, 403):
        # Bad or unauthorized key is a server-side problem, not the user's.
        return AIServiceError(_MSG_UNAVAILABLE, status_code=status)
    return AIServiceError(_MSG_GENERIC, status_code=status)


def _build_payload(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> dict[str, Any]:
    """Build the OpenAI-compatible request body."""
    payload: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stream:
        payload["stream"] = True
    # NEW (F4a): gpt-oss spends part of max_tokens on hidden reasoning before
    # producing visible content, which can leave short completions (e.g. the
    # search planner) with no visible output at all. WEB_REASONING_EFFORT
    # ("" disables this) is sent as-is; once Groq has told us it rejects the
    # field (HTTP 400), it is left out for the rest of the process.
    if WEB_REASONING_EFFORT and not _reasoning_effort_unsupported:
        payload["reasoning_effort"] = WEB_REASONING_EFFORT
    return payload


def _finish_reason(data: Any) -> str:
    """Best-effort extraction of choices[0].finish_reason for logging only."""
    try:
        return str(data["choices"][0].get("finish_reason"))
    except Exception:  # noqa: BLE001 - logging helper must never raise
        return "unknown"


def _usage(data: Any) -> dict[str, Any]:
    """Best-effort extraction of the token usage numbers for logging only."""
    try:
        usage = data.get("usage")
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
    except Exception:  # noqa: BLE001 - logging helper must never raise
        pass
    return {}


def _build_headers(*, stream: bool) -> dict[str, str]:
    """Build request headers. The key is used here and never logged."""
    headers = {
        "Authorization": f"Bearer {_require_key()}",
        "Content-Type": "application/json",
    }
    if stream:
        headers["Accept"] = "text/event-stream"
    return headers


async def groq_complete(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.2,
    timeout: float = 20.0,
) -> str:
    """Run a non-streaming chat completion and return the reply text.

    Args:
        messages: OpenAI-style chat messages.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        timeout: Read timeout in seconds for each attempt.

    Returns:
        ``choices[0].message.content``, or ``""`` if it is missing.

    Raises:
        AIServiceError: If the key is missing or the request finally fails.
    """
    headers = _build_headers(stream=False)
    attempts = _max_attempts()

    async with httpx.AsyncClient(timeout=_timeout(timeout)) as client:
        for attempt in range(1, attempts + 1):
            # NEW (F4b): built per-attempt (not once before the loop) so that
            # once _reasoning_effort_unsupported flips True, later attempts -
            # including the very next one - stop sending the field.
            payload = _build_payload(messages, max_tokens=max_tokens, temperature=temperature, stream=False)
            delay: float
            try:
                response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                logger.warning(
                    "Groq request failed (%s), attempt %d/%d",
                    type(exc).__name__, attempt, attempts,
                )
                if attempt >= attempts:
                    raise AIServiceError(_MSG_BUSY) from None
                delay = _backoff_delay(attempt)
            else:
                status = response.status_code

                # NEW (F4b): some gpt-oss deployments reject "reasoning_effort"
                # with HTTP 400. Retry this one attempt immediately without
                # the field and remember not to send it again for the rest of
                # the process.
                if status == 400 and "reasoning_effort" in payload:
                    global _reasoning_effort_unsupported
                    logger.info("Groq rejected reasoning_effort (HTTP 400); retrying without it.")
                    _reasoning_effort_unsupported = True
                    payload = _build_payload(
                        messages, max_tokens=max_tokens, temperature=temperature, stream=False
                    )
                    try:
                        response = await client.post(GROQ_API_URL, headers=headers, json=payload)
                    except httpx.HTTPError as exc:
                        logger.warning(
                            "Groq request failed after dropping reasoning_effort (%s), attempt %d/%d",
                            type(exc).__name__, attempt, attempts,
                        )
                        if attempt >= attempts:
                            raise AIServiceError(_MSG_BUSY) from None
                        await asyncio.sleep(_backoff_delay(attempt))
                        continue
                    status = response.status_code

                if status == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        # Valid HTTP response but not valid JSON.
                        return ""
                    try:
                        content = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError):
                        content = None
                    if not isinstance(content, str) or not content.strip():
                        # NEW (F4c): log finish_reason and token usage numbers
                        # only when the content comes back empty - never the
                        # message text itself.
                        logger.info(
                            "Groq reply had empty content (finish_reason=%s, usage=%s)",
                            _finish_reason(data), _usage(data),
                        )
                        return content if isinstance(content, str) else ""
                    return content
                logger.warning("Groq returned HTTP %d, attempt %d/%d", status, attempt, attempts)
                if status not in _RETRYABLE_STATUSES or attempt >= attempts:
                    raise _error_for_status(status) from None
                delay = _retry_delay(response, attempt)
            await asyncio.sleep(delay)

    # Unreachable in practice (the loop always returns or raises).
    raise AIServiceError(_MSG_GENERIC)


def _extract_delta_text(line: str) -> tuple[str | None, bool]:
    """Parse one SSE line from Groq.

    Returns:
        ``(text, done)`` where ``text`` is a non-empty content string or
        ``None``, and ``done`` is True when the ``[DONE]`` marker was seen.

    Raises:
        AIServiceError: If the line carries an upstream error object.
    """
    if not line.startswith("data:"):
        return None, False
    body = line[len("data:"):].strip()
    if not body:
        return None, False
    if body == "[DONE]":
        return None, True
    try:
        obj = json.loads(body)
    except ValueError:
        return None, False  # malformed line: skip
    if not isinstance(obj, dict):
        return None, False
    if "error" in obj:
        logger.warning("Groq stream reported an upstream error object.")
        raise AIServiceError(_MSG_GENERIC)
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, False
    choice = choices[0]
    if not isinstance(choice, dict):
        return None, False
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return None, False
    # Only "content" is used; every other delta field (e.g. "reasoning") is ignored.
    content = delta.get("content")
    if isinstance(content, str) and content:
        return content, False
    return None, False


async def groq_stream(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.6,
    timeout: float = 60.0,
) -> AsyncIterator[str]:
    """Stream a chat completion, yielding answer text chunks.

    Retries happen only before the first chunk has been yielded. Once text
    has been yielded, any failure raises ``AIServiceError`` without retrying.
    If the consumer stops iterating (or the task is cancelled), the HTTP
    stream and client are closed cleanly by the ``async with`` blocks.

    Args:
        messages: OpenAI-style chat messages.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        timeout: Read timeout in seconds (max silence between chunks).

    Yields:
        Non-empty ``delta.content`` strings.

    Raises:
        AIServiceError: If the key is missing or the stream finally fails.
    """
    headers = _build_headers(stream=True)
    attempts = _max_attempts()
    yielded = False

    for attempt in range(1, attempts + 1):
        # NEW (F4b): built per-attempt so a flag flip (below, or from a
        # concurrent groq_complete call) takes effect on the next attempt.
        payload = _build_payload(messages, max_tokens=max_tokens, temperature=temperature, stream=True)
        delay: float | None = None
        try:
            async with httpx.AsyncClient(timeout=_timeout(timeout)) as client:
                async with client.stream("POST", GROQ_API_URL, headers=headers, json=payload) as response:
                    status = response.status_code

                    # NEW (F4b): some gpt-oss deployments reject
                    # "reasoning_effort" with HTTP 400. Retry this one
                    # attempt immediately, inline, without the field, and
                    # remember not to send it again for the rest of the
                    # process. Only safe before any tokens have been yielded.
                    if status == 400 and "reasoning_effort" in payload and not yielded:
                        global _reasoning_effort_unsupported
                        logger.info(
                            "Groq rejected reasoning_effort on stream (HTTP 400); retrying without it."
                        )
                        _reasoning_effort_unsupported = True
                        payload = _build_payload(
                            messages, max_tokens=max_tokens, temperature=temperature, stream=True
                        )
                        async with client.stream(
                            "POST", GROQ_API_URL, headers=headers, json=payload
                        ) as retry_response:
                            status = retry_response.status_code
                            if status != 200:
                                logger.warning(
                                    "Groq stream returned HTTP %d after dropping reasoning_effort, "
                                    "attempt %d/%d", status, attempt, attempts,
                                )
                                if status in _RETRYABLE_STATUSES and attempt < attempts and not yielded:
                                    delay = _retry_delay(retry_response, attempt)
                                else:
                                    raise _error_for_status(status)
                            else:
                                async for line in retry_response.aiter_lines():
                                    text, done = _extract_delta_text(line)
                                    if text is not None:
                                        yielded = True
                                        yield text
                                    if done:
                                        break
                                return
                    elif status != 200:
                        logger.warning(
                            "Groq stream returned HTTP %d, attempt %d/%d", status, attempt, attempts
                        )
                        if status in _RETRYABLE_STATUSES and attempt < attempts and not yielded:
                            delay = _retry_delay(response, attempt)
                        else:
                            raise _error_for_status(status)
                    else:
                        async for line in response.aiter_lines():
                            text, done = _extract_delta_text(line)
                            if text is not None:
                                yielded = True
                                yield text
                            if done:
                                break
                        return
        except AIServiceError:
            raise
        except httpx.HTTPError as exc:
            logger.warning(
                "Groq stream failed (%s), attempt %d/%d, tokens_started=%s",
                type(exc).__name__, attempt, attempts, yielded,
            )
            if yielded:
                raise AIServiceError(_MSG_INTERRUPTED) from None
            if attempt >= attempts:
                raise AIServiceError(_MSG_BUSY) from None
            delay = _backoff_delay(attempt)

        # Reached only when a retry was scheduled; the stream is already closed.
        if delay is not None:
            await asyncio.sleep(delay)

    raise AIServiceError(_MSG_BUSY)
