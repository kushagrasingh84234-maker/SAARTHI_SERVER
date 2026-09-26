"""Local Saarthi model client used by the web chat channel (POST /api/chat).

Previously this module was an async httpx client that called the external
Groq cloud API directly. It now routes everything through ai_services.py's
local Qwen3-VL + Saarthi LoRA model (``call_saarthi_model`` /
``stream_saarthi_model``) instead - no network calls to Groq happen here
anymore, and GROQ_API_KEY is never read or required by this module.

Public API (unchanged names, so web_channel/web_api.py needs no changes):
    AIServiceError  Exception carrying a ``user_message`` that is safe to show.
    groq_complete   Non-streaming completion (used for the planner step).
    groq_stream     Async generator yielding answer text chunks as they arrive.

Also exported under clean new names, identical behavior:
    saarthi_complete = groq_complete
    saarthi_stream = groq_stream

Reasoning-header filtering: our fine-tuned saarthi_v2_perfect model can
prefix even [MODE: CONVERSATIONAL] replies with an internal
[INTENT_ANALYSIS] / [REASONING_STEPS] block before its actual user-facing
paragraph. ``groq_complete`` strips this (and any stray [ACTION] block)
from the final text by default - pass ``strip_reasoning=False`` for
callers such as the search planner that expect raw/JSON output.
``groq_stream`` applies the same filtering live, token by token: if a
reply doesn't start with [INTENT_ANALYSIS]/[REASONING_STEPS] at all,
tokens are streamed immediately with zero buffering delay; if it does,
tokens are buffered internally (never shown) until the blank line that
ends the reasoning block, then only the clean answer is streamed.

Logging: only exception types, marker/length information, and generic
status. Message contents and model output are never logged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import ai_services

logger = logging.getLogger(__name__)

__all__ = ["AIServiceError", "groq_complete", "groq_stream", "saarthi_complete", "saarthi_stream"]

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
        status_code: Upstream HTTP status when known (kept for interface
            compatibility with callers written for the old Groq-backed
            version; always None now, since there is no upstream HTTP
            status - local inference either succeeds or raises).
    """

    def __init__(self, user_message: str = _MSG_GENERIC, *, status_code: int | None = None) -> None:
        """Create the error.

        Args:
            user_message: User-safe description of the failure.
            status_code: Optional upstream HTTP status code (unused
                locally; kept for signature compatibility).
        """
        super().__init__(user_message)
        self.user_message: str = user_message
        self.status_code: int | None = status_code


# ---------------------------------------------------------------------------
# Reasoning-header stripping (shared by groq_complete and groq_stream)
# ---------------------------------------------------------------------------
#
# Deliberately NOT reusing text_utils.parse_saarthi_structured_output here:
# that pipeline also runs strip_markdown/format_symbols/ASCII-encoding,
# which is correct for the ILI9341 robot screen but would mangle normal
# markdown-formatted web/app prose. This is a much lighter, web-safe
# version that only removes the internal header blocks themselves.

_REASONING_MARKERS = ("[INTENT_ANALYSIS]", "[REASONING_STEPS]")
_ACTION_MARKER = "[ACTION]"
_ALL_MARKERS = _REASONING_MARKERS + (_ACTION_MARKER,)
_MARKER_PROBE_LEN = max(len(m) for m in _ALL_MARKERS)


def _strip_reasoning_headers(text: str) -> str:
    """Remove a leading [INTENT_ANALYSIS]/[REASONING_STEPS] block and any
    trailing [ACTION] JSON block from a complete (non-streamed) reply,
    leaving just the user-facing paragraph. Returns the text unchanged if
    none of those markers are present."""
    if not text:
        return text or ""

    if not any(marker in text for marker in _ALL_MARKERS):
        return text

    positions = sorted((text.find(marker), marker) for marker in _ALL_MARKERS if marker in text)

    spoken = ""
    action_start = None
    for i, (start, marker) in enumerate(positions):
        content_start = start + len(marker)
        content_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[content_start:content_end]
        if marker == _ACTION_MARKER:
            action_start = start
            continue
        if "\n\n" in content:
            _, _, tail = content.partition("\n\n")
            if tail.strip():
                spoken = tail.strip()

    if spoken:
        return spoken
    if action_start is not None:
        # No spoken text was found after the reasoning block, but there's
        # an [ACTION] block - drop it and return whatever came before it.
        return text[:action_start].strip()
    return text.strip()


def _is_possible_marker_prefix(candidate: str) -> bool:
    """True if `candidate` (already lstripped) could still grow into a
    full [INTENT_ANALYSIS] or [REASONING_STEPS] marker with more tokens,
    or is empty so far."""
    if not candidate:
        return True
    return any(marker.startswith(candidate) for marker in _REASONING_MARKERS)


async def groq_complete(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.2,
    timeout: float = 20.0,
    images: list[Any] | None = None,
    video_frames: list[Any] | None = None,
    response_policy: dict[str, Any] | None = None,
    strip_reasoning: bool = True,
) -> str:
    """Run a non-streaming local Saarthi model completion and return the
    reply text.

    Args:
        messages: OpenAI-style chat messages.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        timeout: Overall wall-clock timeout in seconds for this call.
        images: Optional list of raw bytes / base64 strings / PIL.Image
            objects to attach, validated via media_processor.process_image.
        video_frames: Optional list of video frames, same accepted types
            as `images`.
        response_policy: Optional structured response-policy dict (see
            ai_services._build_policy_preamble) folded into the prompt.
        strip_reasoning: If True (default), strips any leading
            [INTENT_ANALYSIS]/[REASONING_STEPS] preamble and any stray
            [ACTION] block from the returned text. Set to False for
            callers that expect the raw/JSON model output as-is - e.g.
            the search planner.

    Returns:
        The (optionally reasoning-stripped) reply text.

    Raises:
        AIServiceError: If the call times out or otherwise fails. Never
            raises for a missing GROQ_API_KEY - there is none to check
            anymore; inference is local.
    """
    try:
        raw_text = await asyncio.wait_for(
            ai_services.call_saarthi_model(
                messages,
                response_policy=response_policy,
                images=images,
                video_frames=video_frames,
                max_new_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Saarthi model call timed out after %.1fs", timeout)
        raise AIServiceError(_MSG_BUSY) from None
    except AIServiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak internals to the user
        logger.warning("Saarthi model call failed: %s", type(exc).__name__)
        raise AIServiceError(_MSG_GENERIC) from None

    if not isinstance(raw_text, str):
        return ""

    return _strip_reasoning_headers(raw_text) if strip_reasoning else raw_text


async def groq_stream(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float = 0.6,
    timeout: float = 60.0,
    images: list[Any] | None = None,
    video_frames: list[Any] | None = None,
    response_policy: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from the local Saarthi model, yielding
    answer text chunks.

    Smart reasoning filter: if the reply opens with [INTENT_ANALYSIS] or
    [REASONING_STEPS], tokens are buffered internally (never yielded)
    until the blank line ("\\n\\n") that ends that block, and only the
    clean user-facing text is streamed from there on. If the reply does
    not open with either marker, tokens are streamed immediately with no
    buffering delay.

    ``timeout`` is enforced PER CHUNK (max silence between tokens), not
    as one overall deadline for the whole stream.

    Args:
        messages: OpenAI-style chat messages.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        timeout: Max seconds to wait for each next token.
        images: Optional list of raw bytes / base64 strings / PIL.Image
            objects to attach, validated via media_processor.process_image.
        video_frames: Optional list of video frames, same accepted types
            as `images`.
        response_policy: Optional structured response-policy dict folded
            into the prompt.

    Yields:
        Non-empty text chunks, with internal reasoning headers filtered
        out as described above.

    Raises:
        AIServiceError: If a chunk times out or generation otherwise
            fails mid-stream. ``_MSG_INTERRUPTED`` if some text had
            already been yielded to the caller, ``_MSG_GENERIC`` if the
            failure happened before anything was yielded.
    """
    yielded = False
    buffer = ""
    decided = False
    reasoning_mode = False

    agen = ai_services.stream_saarthi_model(
        messages,
        response_policy=response_policy,
        images=images,
        video_frames=video_frames,
        max_new_tokens=max_tokens,
        temperature=temperature,
    ).__aiter__()

    try:
        while True:
            try:
                token = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.warning("Saarthi stream stalled for over %.1fs", timeout)
                if yielded:
                    raise AIServiceError(_MSG_INTERRUPTED) from None
                raise AIServiceError(_MSG_GENERIC) from None

            if not decided:
                buffer += token
                lead = buffer.lstrip()
                if lead.startswith(_REASONING_MARKERS[0]) or lead.startswith(_REASONING_MARKERS[1]):
                    decided = True
                    reasoning_mode = True
                elif _is_possible_marker_prefix(lead) and len(lead) < _MARKER_PROBE_LEN:
                    continue  # still ambiguous - wait for more tokens before deciding
                else:
                    decided = True
                    reasoning_mode = False
                    if buffer:
                        yielded = True
                        yield buffer
                    buffer = ""
                    continue
            elif reasoning_mode:
                # Already decided this reply opens with a reasoning header -
                # keep accumulating tokens (still nothing yielded) until the
                # blank line that ends the reasoning block shows up below.
                buffer += token

            if reasoning_mode:
                if "\n\n" in buffer:
                    _, _, tail = buffer.partition("\n\n")
                    if tail:
                        yielded = True
                        yield tail
                    buffer = ""
                    reasoning_mode = False
                # else: keep silently buffering until the blank line arrives
                continue

            # decided and not in reasoning_mode: plain passthrough
            yielded = True
            yield token
    except AIServiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak internals to the user
        logger.warning("Saarthi stream failed: %s, tokens_started=%s", type(exc).__name__, yielded)
        if yielded:
            raise AIServiceError(_MSG_INTERRUPTED) from None
        raise AIServiceError(_MSG_GENERIC) from None
    finally:
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - cleanup must never raise
                pass


# Clean new names for new call sites; identical behavior to groq_complete /
# groq_stream.
saarthi_complete = groq_complete
saarthi_stream = groq_stream
