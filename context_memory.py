"""
context_memory.py

Lightweight, bounded short-term conversation memory manager for
StudyBot.

Multiple WebSocket sessions run concurrently against this server, so
memory here is strictly isolated per session/user identifier — one
session can never see or affect another's history.

This module:
  - stores only recent conversational turns (bounded, never unlimited)
  - stores no secrets, API keys, tokens, or credentials
  - has no WebSocket, database, or AI/LLM API code — it's a pure
    in-memory structure that callers read/write
  - is safe to use from an asyncio-based server: per-session locking
    prevents interleaved concurrent writes to the same session from
    corrupting history, without blocking unrelated sessions.

Persistence (if ever needed) is the caller's responsibility via the
export/import functions — this module itself never touches disk or a
database.

--------------------------------------------------------------------
PHASE 3B ADDITIONS (Memory & Orchestration)
--------------------------------------------------------------------
This phase adds, on top of the original bounded ring-buffer memory:

  - optional, bounded per-turn metadata (importance/source/intent/
    category/confidence/etc.) and an optional `modality` tag so a
    turn can eventually represent text/voice/camera/sensor input
    without redesigning this module (see build_context / MemoryTurn).
  - explicit context budgeting via `build_context()` /
    `ContextBudget`, so a single huge message or a long session can
    never blow past a caller-defined turn/character budget.
  - a persistence-neutral snapshot (`get_snapshot()`) and handoff
    payload (`build_persistence_handoff()`) intended for an
    orchestration layer to forward to a separate persistence module.
    This module still never imports or depends on any database code.

Every existing public class, function, and behavior from the prior
version of this module is preserved unchanged; all of the above is
strictly additive.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_TURNS = 20          # a "turn" = one user message OR one assistant reply
_HARD_MAX_TURNS = 200            # absolute ceiling regardless of configuration
_MAX_MESSAGE_LENGTH = 8000       # guard against pathological single-message bloat
_VALID_ROLES = frozenset({"user", "assistant"})

# -- Phase 3B: metadata bounds ------------------------------------------------
# Metadata is optional, caller-supplied, and opaque to this module (this
# module never interprets it — see module docstring / architectural rule
# in the spec: no AI-based importance classification, no LLM calls here).
_MAX_METADATA_KEYS = 10
_MAX_METADATA_KEY_LENGTH = 64
_MAX_METADATA_VALUE_LENGTH = 500
_METADATA_SCALAR_TYPES = (str, int, float, bool, type(None))

# -- Phase 3B: future input modalities ---------------------------------------
# A turn is conceptually a "normalized interaction event". Today only
# "text" is produced by callers, but the field exists now so voice/camera/
# sensor inputs can be represented later without a redesign (see spec
# section 13). No STT/TTS/audio/vision code is implemented here.
_VALID_MODALITIES = frozenset({"text", "voice", "camera", "sensor"})
_DEFAULT_MODALITY = "text"

# -- Phase 3B: context budgeting ---------------------------------------------
_DEFAULT_CONTEXT_MAX_CHARS = 6000     # default total-context character budget
_HARD_CONTEXT_MAX_CHARS = 40000       # absolute ceiling regardless of caller input
_SNAPSHOT_SCHEMA_VERSION = 1


class ContextMemoryError(ValueError):
    """Raised on invalid memory operations or malformed input/state."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryTurn:
    """One turn of conversation memory (either a user message or an
    assistant response).

    `metadata` and `modality` are optional (Phase 3B). Older callers
    that never pass them get identical behavior to before: `metadata`
    defaults to None and `modality` defaults to "text". This module
    stores/forwards metadata only — it never interprets, scores, or
    generates it (no importance classification, no embeddings, no
    LLM calls happen here).
    """

    role: str            # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    intent: Optional[str] = None       # optional tag, e.g. routed intent for a user turn
    metadata: Optional[Dict[str, Any]] = None   # optional bounded metadata (Phase 3B)
    modality: str = _DEFAULT_MODALITY  # optional input modality tag (Phase 3B)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "intent": self.intent,
        }
        # Only emit the new fields when non-default, so payloads produced
        # by callers that never touch Phase 3B features remain identical
        # to the pre-Phase-3B shape.
        if self.metadata:
            data["metadata"] = self.metadata
        if self.modality != _DEFAULT_MODALITY:
            data["modality"] = self.modality
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemoryTurn":
        role = data.get("role")
        content = data.get("content")
        if role not in _VALID_ROLES:
            raise ContextMemoryError(f"Invalid role in turn data: {role}")
        if not isinstance(content, str):
            raise ContextMemoryError("Turn content must be a string")

        # Backward compatibility: old serialized turns have neither
        # "metadata" nor "modality" — both fall back to safe defaults.
        raw_metadata = data.get("metadata")
        metadata = _sanitize_metadata(raw_metadata) if raw_metadata else None

        raw_modality = data.get("modality") or _DEFAULT_MODALITY
        modality = raw_modality if raw_modality in _VALID_MODALITIES else _DEFAULT_MODALITY

        return MemoryTurn(
            role=role,
            content=_truncate(content),
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            intent=data.get("intent"),
            metadata=metadata,
            modality=modality,
        )


def _truncate(content: str) -> str:
    if len(content) > _MAX_MESSAGE_LENGTH:
        return content[:_MAX_MESSAGE_LENGTH]
    return content


def _sanitize_metadata(metadata: Any) -> Optional[Dict[str, Any]]:
    """Defensively bound and clean caller-supplied metadata.

    This never raises on malformed *values* — bad individual entries
    are dropped rather than corrupting the whole turn or crashing the
    pipeline (spec section 15). It DOES raise ContextMemoryError if the
    top-level shape itself is unusable (not a dict at all), since that
    is a programmer/configuration error rather than "one bad field".

    Only plain scalar values (str/int/float/bool/None) are kept —
    this module deliberately stores/forwards metadata without
    interpreting it, so nested structures, embeddings, or arbitrary
    objects are out of scope by design (spec sections 5, 7, 17).
    """
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ContextMemoryError("metadata must be a dict")

    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if len(key) > _MAX_METADATA_KEY_LENGTH:
            key = key[:_MAX_METADATA_KEY_LENGTH]
        if not isinstance(value, _METADATA_SCALAR_TYPES):
            # Never store arbitrary objects; coerce defensively.
            value = str(value)[:_MAX_METADATA_VALUE_LENGTH]
        elif isinstance(value, str) and len(value) > _MAX_METADATA_VALUE_LENGTH:
            value = value[:_MAX_METADATA_VALUE_LENGTH]

        cleaned[key] = value
        if len(cleaned) >= _MAX_METADATA_KEYS:
            break

    return cleaned or None


class _SessionMemory:
    """Internal bounded ring-buffer of turns for a single session."""

    __slots__ = ("turns", "max_turns", "lock")

    def __init__(self, max_turns: int):
        self.turns: Deque[MemoryTurn] = deque(maxlen=max_turns)
        self.max_turns = max_turns
        self.lock = threading.Lock()


# ---------------------------------------------------------------------------
# Context budgeting (Phase 3B)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBudget:
    """Explicit, independent limits used when building AI working
    context from recent turns (spec section 4).

    All limits are optional; omitted ones fall back to this module's
    existing defaults/ceilings. Every limit is still clamped to a
    hard ceiling so a bad caller-supplied value can never produce
    unbounded context.
    """

    max_turns: Optional[int] = None
    max_total_chars: Optional[int] = None
    max_message_chars: Optional[int] = None

    def resolved(self, session_max_turns: int) -> "ContextBudget":
        turns = self.max_turns if self.max_turns is not None else session_max_turns
        turns = max(1, min(turns, _HARD_MAX_TURNS))

        total_chars = self.max_total_chars if self.max_total_chars is not None else _DEFAULT_CONTEXT_MAX_CHARS
        total_chars = max(1, min(total_chars, _HARD_CONTEXT_MAX_CHARS))

        msg_chars = self.max_message_chars if self.max_message_chars is not None else _MAX_MESSAGE_LENGTH
        msg_chars = max(1, min(msg_chars, _MAX_MESSAGE_LENGTH))

        return ContextBudget(max_turns=turns, max_total_chars=total_chars, max_message_chars=msg_chars)


def _apply_budget(turns: List[MemoryTurn], budget: ContextBudget) -> List[Dict[str, Any]]:
    """Turn a chronological list of MemoryTurn into a bounded context.

    Deterministic algorithm:
      1. Take at most `max_turns` most-recent turns.
      2. Per-message-cap each turn's content to `max_message_chars`
         (so one pathological message can't consume the whole budget).
      3. Walk from most-recent to oldest, greedily including whole
         turns while the running character total stays within
         `max_total_chars`. This preserves complete role/content
         boundaries (never splits a turn) and always keeps the most
         recent turns over older ones.
      4. Re-sort the kept turns back into chronological order.

    Never mutates the stored MemoryTurn objects or session state —
    truncation here is purely a view produced for the caller.
    """
    recent = turns[-budget.max_turns:] if budget.max_turns else list(turns)

    kept: List[MemoryTurn] = []
    running_chars = 0
    for turn in reversed(recent):
        content = turn.content
        if len(content) > budget.max_message_chars:
            content = content[: budget.max_message_chars]

        turn_chars = len(content) + len(turn.role)
        if kept and running_chars + turn_chars > budget.max_total_chars:
            # Budget exhausted; older turns are dropped, not the
            # current one — keeps output non-empty even for one very
            # large recent turn to avoid returning nothing.
            break

        running_chars += turn_chars
        if content != turn.content:
            capped = MemoryTurn(
                role=turn.role,
                content=content,
                timestamp=turn.timestamp,
                intent=turn.intent,
                metadata=turn.metadata,
                modality=turn.modality,
            )
            kept.append(capped)
        else:
            kept.append(turn)

    kept.reverse()
    return [t.to_dict() for t in kept]


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------

class ContextMemoryManager:
    """
    Manages bounded, per-session short-term conversation memory.

    Safe for use from a small async Python server: each session has
    its own lock, so concurrent access to DIFFERENT sessions never
    blocks each other, and concurrent access to the SAME session is
    serialized to prevent interleaved writes from corrupting order.

    This class holds no global lock on the whole manager during reads,
    only a lightweight guard around the session registry itself.
    """

    def __init__(self, max_turns: int = _DEFAULT_MAX_TURNS):
        if not isinstance(max_turns, int) or max_turns <= 0:
            raise ContextMemoryError("max_turns must be a positive integer")
        self._max_turns = min(max_turns, _HARD_MAX_TURNS)
        self._sessions: Dict[str, _SessionMemory] = {}
        self._registry_lock = threading.Lock()

    # -- internal helpers ---------------------------------------------------

    def _get_or_create_session(self, session_id: str) -> _SessionMemory:
        _validate_session_id(session_id)
        with self._registry_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = _SessionMemory(max_turns=self._max_turns)
                self._sessions[session_id] = session
            return session

    def _get_session_if_exists(self, session_id: str) -> Optional[_SessionMemory]:
        _validate_session_id(session_id)
        with self._registry_lock:
            return self._sessions.get(session_id)

    # -- write operations -----------------------------------------------------

    def add_message(
        self,
        session_id: str,
        content: str,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        modality: str = _DEFAULT_MODALITY,
    ) -> None:
        """Record a user message for the given session.

        `metadata` and `modality` are optional (Phase 3B) and fully
        backward compatible — omit them for identical pre-3B behavior.
        """
        self._add_turn(session_id, role="user", content=content, intent=intent,
                        metadata=metadata, modality=modality)

    def add_response(
        self,
        session_id: str,
        content: str,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        modality: str = _DEFAULT_MODALITY,
    ) -> None:
        """Record an assistant response for the given session.

        `metadata` and `modality` are optional (Phase 3B) and fully
        backward compatible — omit them for identical pre-3B behavior.
        """
        self._add_turn(session_id, role="assistant", content=content, intent=intent,
                        metadata=metadata, modality=modality)

    def _add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        modality: str = _DEFAULT_MODALITY,
    ) -> None:
        if role not in _VALID_ROLES:
            raise ContextMemoryError(f"Invalid role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ContextMemoryError("content must be a non-empty string")
        if modality not in _VALID_MODALITIES:
            raise ContextMemoryError(f"Invalid modality: {modality}")

        safe_metadata = _sanitize_metadata(metadata)

        turn = MemoryTurn(
            role=role,
            content=_truncate(content),
            intent=intent,
            metadata=safe_metadata,
            modality=modality,
        )
        session = self._get_or_create_session(session_id)
        with session.lock:
            # deque(maxlen=...) automatically evicts the oldest turn,
            # enforcing bounded memory with no manual trimming needed.
            session.turns.append(turn)

    # -- read operations ------------------------------------------------------

    def get_recent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent turns for a session, oldest-first within the
        returned slice (i.e. chronological order), most recent last.

        - limit: if given, returns only the last `limit` turns.
        - role: if given ("user" or "assistant"), filters to that role
          before applying the limit.

        Returns an empty list if the session has no history yet.
        """
        session = self._get_session_if_exists(session_id)
        if session is None:
            return []

        with session.lock:
            turns = list(session.turns)

        if role is not None:
            if role not in _VALID_ROLES:
                raise ContextMemoryError(f"Invalid role filter: {role}")
            turns = [t for t in turns if t.role == role]

        if limit is not None:
            if limit < 0:
                raise ContextMemoryError("limit must be non-negative")
            turns = turns[-limit:]

        return [t.to_dict() for t in turns]

    def get_turn_count(self, session_id: str) -> int:
        """Return the number of turns currently stored for a session."""
        session = self._get_session_if_exists(session_id)
        if session is None:
            return 0
        with session.lock:
            return len(session.turns)

    def has_session(self, session_id: str) -> bool:
        with self._registry_lock:
            return session_id in self._sessions

    # -- context orchestration (Phase 3B) --------------------------------------

    def build_context(
        self,
        session_id: str,
        budget: Optional[ContextBudget] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build a bounded AI "working context" from recent turns.

        This is the module's context-orchestration helper (spec
        sections 3-4): it wraps get_recent_messages-style access with
        explicit, independent budgeting so a single huge message or a
        long-running session can never produce unbounded context.

        - budget: a ContextBudget with optional max_turns /
          max_total_chars / max_message_chars. Any field left as None
          falls back to this manager's defaults; every field is
          clamped to a hard ceiling regardless of input.
        - role: optional role filter, same semantics as
          get_recent_messages.

        Returns turns in chronological order (oldest first), each as
        a plain dict (role/content/timestamp/intent/[metadata]/
        [modality]) — the same shape as get_recent_messages, so
        existing callers of get_recent_messages can adopt this
        directly.

        Returns an empty list for a session with no history yet.
        """
        session = self._get_session_if_exists(session_id)
        if session is None:
            return []

        with session.lock:
            turns = list(session.turns)
            session_max_turns = session.max_turns

        if role is not None:
            if role not in _VALID_ROLES:
                raise ContextMemoryError(f"Invalid role filter: {role}")
            turns = [t for t in turns if t.role == role]

        resolved_budget = (budget or ContextBudget()).resolved(session_max_turns)
        return _apply_budget(turns, resolved_budget)

    # -- lifecycle / cleanup ---------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        """
        Clear all memory for a single session (e.g. on logout, session
        end, or explicit reset). Isolated to that session only — no
        other session is affected.
        """
        session = self._get_session_if_exists(session_id)
        if session is None:
            return
        with session.lock:
            session.turns.clear()

    def remove_session(self, session_id: str) -> None:
        """
        Fully remove a session's memory object from the manager (e.g.
        when a WebSocket disconnects for good). Safe to call even if
        the session doesn't exist.
        """
        _validate_session_id(session_id)
        with self._registry_lock:
            self._sessions.pop(session_id, None)

    def active_session_ids(self) -> List[str]:
        """Return a snapshot list of currently tracked session ids."""
        with self._registry_lock:
            return list(self._sessions.keys())

    # -- export / import (serialization only, no persistence logic) -----------

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """
        Export a single session's memory as a plain, JSON-serializable
        dict. Caller is responsible for actually persisting this
        (e.g. writing it to a database) — this module has no
        persistence logic itself.
        """
        session = self._get_session_if_exists(session_id)
        if session is None:
            return {"session_id": session_id, "max_turns": self._max_turns, "turns": []}

        with session.lock:
            turns = [t.to_dict() for t in session.turns]

        return {
            "session_id": session_id,
            "max_turns": session.max_turns,
            "turns": turns,
        }

    def import_session(self, data: Dict[str, Any]) -> None:
        """
        Restore a session's memory from a previously exported dict.
        Overwrites any existing in-memory history for that session id.
        Truncates to this manager's configured max_turns if the
        imported data exceeds it.
        """
        if not isinstance(data, dict):
            raise ContextMemoryError("import data must be a dict")

        session_id = data.get("session_id")
        _validate_session_id(session_id)

        raw_turns = data.get("turns", []) or []
        if not isinstance(raw_turns, list):
            raise ContextMemoryError("import data.turns must be a list")

        parsed_turns = [MemoryTurn.from_dict(t) for t in raw_turns]

        session = self._get_or_create_session(session_id)
        with session.lock:
            session.turns.clear()
            for turn in parsed_turns[-session.max_turns:]:
                session.turns.append(turn)

    def export_all(self) -> Dict[str, Any]:
        """Export every active session's memory in one serializable dict."""
        for_ids = self.active_session_ids()
        return {
            "max_turns": self._max_turns,
            "sessions": {sid: self.export_session(sid) for sid in for_ids},
        }

    def import_all(self, data: Dict[str, Any]) -> None:
        """Restore multiple sessions at once from an export_all() payload."""
        if not isinstance(data, dict):
            raise ContextMemoryError("import data must be a dict")

        sessions = data.get("sessions", {}) or {}
        if not isinstance(sessions, dict):
            raise ContextMemoryError("import data.sessions must be a dict")

        for session_id, session_data in sessions.items():
            if not isinstance(session_data, dict):
                raise ContextMemoryError(f"Invalid session data for {session_id}")
            # Ensure session_id consistency even if payload omits it.
            session_data.setdefault("session_id", session_id)
            self.import_session(session_data)

    # -- snapshot / persistence handoff (Phase 3B) -----------------------------
    #
    # These do NOT persist anything themselves and do NOT import
    # database.py. They only produce plain, JSON-serializable dicts
    # for an orchestration layer to forward to a separate persistence
    # module, per the architectural rule at the top of this file.

    def get_snapshot(self, session_id: str, max_turns: Optional[int] = None) -> Dict[str, Any]:
        """
        Produce a safe, persistence-neutral snapshot of one session's
        short-term memory (spec section 5).

        Contains only: session_id, bounded recent turns, turn count,
        and a schema_version marker. Never contains embeddings, API
        keys, credentials, secrets, or database connections — those
        concepts don't exist in this module at all.

        `max_turns`, if given, further bounds the snapshot below the
        manager's normal per-session limit (e.g. for a smaller
        handoff payload); it never increases it.
        """
        session = self._get_session_if_exists(session_id)
        if session is None:
            return {
                "schema_version": _SNAPSHOT_SCHEMA_VERSION,
                "session_id": session_id,
                "turn_count": 0,
                "turns": [],
            }

        with session.lock:
            turns = list(session.turns)
            turn_count = len(turns)

        if max_turns is not None:
            if not isinstance(max_turns, int) or max_turns <= 0:
                raise ContextMemoryError("max_turns must be a positive integer")
            turns = turns[-max_turns:]

        return {
            "schema_version": _SNAPSHOT_SCHEMA_VERSION,
            "session_id": session_id,
            "turn_count": turn_count,
            "turns": [t.to_dict() for t in turns],
        }

    def build_persistence_handoff(self, session_id: str) -> Dict[str, Any]:
        """
        Build the persistence-neutral payload described in spec
        section 6: a plain dict another layer (orchestration ->
        database.py) can consume to persist this session, without
        this module ever knowing anything about how or whether
        persistence happens.

        This module intentionally has no knowledge of database.py —
        the caller/orchestration layer decides what, if anything, to
        do with this payload.
        """
        return self.get_snapshot(session_id)


# ---------------------------------------------------------------------------
# Async-friendly convenience wrapper
# ---------------------------------------------------------------------------
#
# The core ContextMemoryManager uses only fast, non-blocking
# threading.Lock sections (no I/O under the lock), so it is safe to
# call directly from async code without blocking the event loop
# meaningfully. This wrapper exists for callers who prefer an
# `await`-based API for consistency with the rest of an async server.

class AsyncContextMemoryManager:
    """
    Thin async-facing wrapper around ContextMemoryManager.

    Uses an asyncio.Lock per manager instance only to serialize the
    (very fast, in-memory) operations from coroutine call sites;
    the underlying manager already guarantees per-session thread
    safety independently.
    """

    def __init__(self, max_turns: int = _DEFAULT_MAX_TURNS):
        self._manager = ContextMemoryManager(max_turns=max_turns)
        self._async_lock = asyncio.Lock()

    async def add_message(
        self,
        session_id: str,
        content: str,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        modality: str = _DEFAULT_MODALITY,
    ) -> None:
        async with self._async_lock:
            self._manager.add_message(session_id, content, intent=intent,
                                       metadata=metadata, modality=modality)

    async def add_response(
        self,
        session_id: str,
        content: str,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        modality: str = _DEFAULT_MODALITY,
    ) -> None:
        async with self._async_lock:
            self._manager.add_response(session_id, content, intent=intent,
                                        metadata=metadata, modality=modality)

    async def get_recent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        async with self._async_lock:
            return self._manager.get_recent_messages(session_id, limit=limit, role=role)

    async def build_context(
        self,
        session_id: str,
        budget: Optional[ContextBudget] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        async with self._async_lock:
            return self._manager.build_context(session_id, budget=budget, role=role)

    async def clear_session(self, session_id: str) -> None:
        async with self._async_lock:
            self._manager.clear_session(session_id)

    async def remove_session(self, session_id: str) -> None:
        async with self._async_lock:
            self._manager.remove_session(session_id)

    async def export_session(self, session_id: str) -> Dict[str, Any]:
        async with self._async_lock:
            return self._manager.export_session(session_id)

    async def import_session(self, data: Dict[str, Any]) -> None:
        async with self._async_lock:
            self._manager.import_session(data)

    async def get_snapshot(self, session_id: str, max_turns: Optional[int] = None) -> Dict[str, Any]:
        async with self._async_lock:
            return self._manager.get_snapshot(session_id, max_turns=max_turns)

    async def build_persistence_handoff(self, session_id: str) -> Dict[str, Any]:
        async with self._async_lock:
            return self._manager.build_persistence_handoff(session_id)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_session_id(session_id: Any) -> None:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ContextMemoryError("session_id must be a non-empty string")
