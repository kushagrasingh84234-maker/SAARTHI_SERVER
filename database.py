"""
database.py

Persistence / storage / memory layer for the StudyBot AI robot.

RECOVERY NOTE: this file had been accidentally overwritten by a draft of
server.py (full FastAPI app, WebSocket route, rate limiting, response
cache, emotion/personality tag mapping, and a self-import of database.py
itself). No git history was available in this archive to recover the
original implementation from, so this is a clean rebuild driven strictly
by the public API contract server.py (the real one) and the personalization
modules (user_profile.py, preferences.py) already depend on:

    add_to_history(role, text, session_id)
    route_and_save_bg(role, text, session_id)
    retrieve_long_term_context(user_message, session_id) -> str
    build_groq_messages(long_term_context, session_id) -> list[dict]
    CHAT_HISTORY
    SUPABASE_ENABLED
    ADVANCED_DB_ENABLED
    save_personalization_profile_bg(session_id, category_counts, recent_intents,
        preferred_response_length, preferred_interaction_style, interests,
        preference_scores)
    load_personalization_profile(session_id) -> dict | None

This module owns ONLY persistence/storage/memory primitives. It does NOT
own: FastAPI app/routes, WebSocket connection management, response cache,
rate limiting, game logic, intent detection, personalization decisions,
or emotion/personality interpretation - those live in server.py,
game_manager.py, intent_router.py, personalization.py, emotion_engine.py,
and personality_engine.py respectively. It does not import any of those
modules, so no circular dependency is introduced (in particular, it never
imports logic.py or server.py).

--------------------------------------------------------------------
PHASE 3B ADDITIONS (Memory & Orchestration)
--------------------------------------------------------------------
Everything above this section is the original, preserved contract.
Phase 3B adds long-term/semantic-memory infrastructure on top of it,
strictly additively:

  - Optional, bounded memory-classification metadata (memory_type,
    importance, confidence, source, category) can now travel with a
    persisted turn. Existing callers that never pass these get
    exactly the previous behavior (memory_type defaults to
    "conversation", matching what was always persisted).
  - A new `save_semantic_memory_bg()` entry point lets the
    orchestration layer explicitly persist a classified memory
    (preference/fact/interest/interaction_pattern) without every raw
    chat turn being auto-promoted to that status.
  - `retrieve_long_term_context()` keeps its exact old signature and
    return type (a bounded string), but is now backed by a
    multi-signal ranking pipeline: semantic similarity, recency,
    importance, and confidence, followed by deduplication and a
    strict character/row budget.
  - `retrieve_relevant_memories()` is a new, additive API exposing
    the same ranked candidates in structured form (content, score,
    per-signal breakdown, metadata) for callers that want more than
    a flattened string - deterministic, bounded, and explainable.
  - The advanced/vector table's schema is probed defensively: if the
    live table doesn't yet have the new metadata columns, reads and
    writes transparently fall back to the original minimal shape.
    No destructive migration is required to deploy this file.
  - Fire-and-forget writes now go through a small bounded thread
    pool instead of spawning one raw thread per call, so write load
    can never create unbounded background threads.

This module still does not classify memories itself (no LLM calls,
no importance inference beyond what a caller supplies, no embeddings
model of its own beyond the existing ai_services.generate_embedding),
still never imports context_memory.py, logic.py, or server.py, and
still degrades to safe no-ops/empty results whenever Supabase or
embeddings are unavailable.

--------------------------------------------------------------------
PHASE 3C ADDITIONS (Advanced Semantic Memory)
--------------------------------------------------------------------
Everything above remains the preserved contract (Phase 3B included).
Phase 3C strengthens the ranked-retrieval pipeline that already
existed, strictly additively, with no public API changes:

  - A configurable, documented MINIMUM RELEVANCE THRESHOLD is now
    applied against each candidate's raw semantic similarity (kept
    separate from the blended score so recency/importance/confidence
    can never paper over a topically irrelevant match). "Nothing
    relevant found" is a valid, expected result - low-quality
    candidates are no longer forced into the prompt merely because
    the database returned rows for the session.
  - Memory type and explicit-vs-inferred source now contribute two
    additional small, bounded terms to the ranking formula (on top
    of the existing similarity/recency/importance/confidence terms).
    Both are capped at a combined 8% of the total score weight so
    semantic similarity remains the dominant signal, never
    overridden by type or source alone.
  - A bounded CONFLICT-RESOLUTION pass runs after deduplication: for
    preference/fact memories that share a category (and therefore
    plausibly describe the same topic without being textually
    near-identical, so dedup alone wouldn't catch them), only the
    strongest candidate survives, preferring explicit over inferred
    source, then higher confidence, then more recent. Superseded
    candidates are never deleted from storage - only excluded from
    this retrieval's result.
  - A bounded RELEVANCE-DIVERSITY pass runs last: the final selection
    caps how many memories may come from the same category/type so
    one dominant topic can't crowd out other genuinely relevant
    memories, while still filling remaining slots by score if
    diversity alone can't reach the configured result size.

None of this changes retrieve_long_term_context's signature/return
type, retrieve_relevant_memories's return shape, or any other public
API. All new behavior is deterministic given the same query,
database state, and reference time, and every new step degrades to
"no candidates survive" rather than raising - semantic memory
remains strictly optional relative to the rest of the chat pipeline.
"""

import math
import time
import logging
import threading
import concurrent.futures
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from config import (
    SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE,
    SUPABASE_SEARCH_TOP_K, SUPABASE_SEARCH_TIMEOUT_SECONDS, SUPABASE_MAX_CONTEXT_CHARS,
    ADVANCED_DB_URL, ADVANCED_DB_KEY, ADVANCED_DB_TABLE,
    MAX_HISTORY_ELEMENTS, SYSTEM_PROMPT,
)
from ai_services import generate_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase client initialization (failure isolated)
# ---------------------------------------------------------------------------
# Two independent Supabase projects/clients are supported, matching
# config.py's separate SUPABASE_* (standard chat log) and ADVANCED_DB_*
# (vector/embedding memory) settings. If the `supabase` package is
# missing, or credentials aren't configured, both flags simply go False
# and every function below degrades to a safe no-op - it never raises
# into the chat pipeline.

try:
    from supabase import create_client, Client  # type: ignore
except ImportError:
    create_client = None  # type: ignore
    Client = None  # type: ignore
    logger.warning("supabase package not importable; persistence disabled.")

_supabase_client: Optional["Client"] = None
_advanced_client: Optional["Client"] = None

if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase (standard) client: {e}")
        _supabase_client = None

if create_client and ADVANCED_DB_URL and ADVANCED_DB_KEY:
    try:
        _advanced_client = create_client(ADVANCED_DB_URL, ADVANCED_DB_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase (advanced/vector) client: {e}")
        _advanced_client = None

SUPABASE_ENABLED = _supabase_client is not None
ADVANCED_DB_ENABLED = _advanced_client is not None

# Table used to persist personalization profiles. Not previously exposed
# in config.py; kept as a local, easily-relocated constant rather than
# touching config.py for this task.
PERSONALIZATION_TABLE = "personalization_profiles"

# Bounded timeout for any blocking Supabase network call made from this
# module. Chat must never hang because a database is slow/unreachable.
PERSONALIZATION_TIMEOUT_SECONDS = SUPABASE_SEARCH_TIMEOUT_SECONDS

# Single small worker pool used only to enforce hard timeouts around
# otherwise-unbounded blocking network calls (the `supabase` client has
# no built-in per-call timeout). Callers of this module already run our
# sync functions via asyncio.to_thread, so this is an extra safety net,
# not the primary offload mechanism.
_timeout_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="db-timeout"
)

# -- Phase 3B: bounded fire-and-forget background execution -----------------
# route_and_save_bg / save_personalization_profile_bg / save_semantic_memory_bg
# previously spawned a brand-new raw threading.Thread on every call. Under
# heavy chat load that has no ceiling. A small bounded pool preserves the
# exact "fire and forget, never blocks the caller" contract while capping
# total concurrent background writes.
_BACKGROUND_MAX_WORKERS = 8
_background_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_BACKGROUND_MAX_WORKERS, thread_name_prefix="db-bg"
)


def _run_in_background(func, *args, **kwargs) -> None:
    """
    Submit a fire-and-forget task to the bounded background pool.
    Never raises - if the pool itself is somehow unable to accept the
    task, this logs and returns rather than propagating into the
    caller's (likely async request) code path.
    """
    try:
        _background_executor.submit(func, *args, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to submit background task {getattr(func, '__name__', func)}: {e}")


def _run_with_timeout(func, timeout_seconds: float, default, *args, **kwargs):
    """
    Run a blocking callable with a hard timeout. Returns `default` (and
    logs) on timeout OR on any exception - this is the single choke
    point that guarantees a slow/unavailable database can never surface
    as an uncaught exception or an indefinite hang.
    """
    try:
        future = _timeout_executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        logger.warning(f"{getattr(func, '__name__', func)} timed out after {timeout_seconds}s")
        return default
    except Exception as e:
        logger.warning(f"{getattr(func, '__name__', func)} failed: {e}")
        return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data validation / sanitization helpers
# ---------------------------------------------------------------------------

_MAX_CONTENT_CHARS = 4000  # bound any single stored message/content string
_MAX_RECENT_INTENTS_STORED = 25  # mirrors user_profile.py's own cap
_MAX_INTERESTS_STORED = 50       # mirrors user_profile.py's own cap
_MAX_INTEREST_LENGTH = 60
_MAX_CATEGORY_COUNT_ENTRIES = 20
_MAX_PREFERENCE_SCORE_ENTRIES = 20


def _safe_str(value: Any, max_len: int = _MAX_CONTENT_CHARS) -> str:
    """Coerce arbitrary input into a bounded, safe string. Never raises."""
    if value is None:
        return ""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return ""
    return text[:max_len]


def _safe_role(role: Any) -> str:
    role_str = _safe_str(role, 32).strip().lower()
    return role_str if role_str in ("user", "model", "assistant", "system") else "user"


def _sanitize_category_counts(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in list(value.items())[:_MAX_CATEGORY_COUNT_ENTRIES]:
        try:
            key = _safe_str(k, 40)
            count = int(v)
            if key and count >= 0:
                out[key] = count
        except (TypeError, ValueError):
            continue
    return out


def _sanitize_str_list(value: Any, max_items: int, max_len: int) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value[:max_items]:
        s = _safe_str(item, max_len).strip()
        if s:
            out.append(s)
    return out


def _sanitize_preference_scores(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in list(value.items())[:_MAX_PREFERENCE_SCORE_ENTRIES]:
        try:
            key = _safe_str(k, 40)
            score = float(v)
            if key:
                out[key] = score
        except (TypeError, ValueError):
            continue
    return out


# -- Phase 3B: semantic-memory metadata bounds/validation -------------------
# These values are opaque classification metadata supplied by the caller
# (orchestration layer). This module stores/ranks/forwards them; it never
# infers, classifies, or judges them itself (no LLM calls, no
# psychological inference - see module docstring).

_VALID_MEMORY_TYPES = frozenset({
    "conversation", "preference", "fact", "interest", "interaction_pattern",
})
_DEFAULT_MEMORY_TYPE = "conversation"
_DEFAULT_IMPORTANCE = 0.5
_DEFAULT_CONFIDENCE = 0.5
_MAX_CATEGORY_LENGTH = 40
_MAX_SOURCE_LENGTH = 40

# Candidate/result bounds for retrieval (spec section 6/9/19: never fetch
# or return unbounded memory).
_MAX_CANDIDATE_ROWS = 50          # rows pulled from the advanced table per query
_HARD_MAX_FINAL_MEMORIES = 10     # absolute ceiling on returned memories
_MAX_MEMORY_ITEM_CHARS = 1000     # per-memory cap before joining into context

# Deterministic ranking weights (must be non-negative; need not sum to 1,
# but are chosen to sum to 1 here for interpretability). Similarity
# dominates so an old, low-quality memory can't outrank a highly
# relevant recent one; importance/confidence provide a bounded nudge
# rather than an override (spec sections 6, 10, 11).
_WEIGHT_SIMILARITY = 0.55
_WEIGHT_RECENCY = 0.18
_WEIGHT_IMPORTANCE = 0.14
_WEIGHT_CONFIDENCE = 0.05
_WEIGHT_MEMORY_TYPE = 0.05
_WEIGHT_SOURCE = 0.03
# Sum == 1.00, interpretable as a weighted blend. Similarity alone carries
# more than half the score (spec section 4/20): memory_type and source are
# deliberately small nudges (8% combined) that can break near-ties but can
# never let a topically-irrelevant memory outrank a relevant one.
_RECENCY_HALF_LIFE_DAYS = 14.0  # a memory's recency signal halves every 2 weeks

# -- Phase 3C: relevance threshold, type/source priors, diversity cap -------

# Spec section 6: "nothing relevant found" must be a valid result, and
# section 4 requires semantic relevance to stay the primary signal that
# recency/importance/confidence/type/source can never override. Gating
# on the blended score alone would let a topically-unrelated-but-recent
# row (similarity ~0) sneak past on the strength of the other signals -
# so this threshold is checked against raw cosine similarity, before the
# other signals are blended in. A candidate must clear this bar to be
# considered "relevant" at all; the blended score then only orders the
# candidates that already passed.
_MIN_SIMILARITY_THRESHOLD = 0.20

# Spec section 5: memory type may influence ranking "modestly" - these are
# plain, non-psychological priors about how durably useful each type
# tends to be as long-term memory (a stored preference is more reusable
# than a raw conversational turn), not a judgement about the user.
_MEMORY_TYPE_PRIORITY: Dict[str, float] = {
    "preference": 0.80,
    "fact": 0.75,
    "interest": 0.60,
    "interaction_pattern": 0.50,
    "conversation": 0.40,
}

# Spec section 12: explicit information is generally stronger than weak
# inferred behavior, but this must stay a bounded nudge, not a
# categorical override. Unclassified/missing source is treated as
# neutral - it is not assumed to be either explicit or inferred.
_SOURCE_PRIORITY_EXPLICIT = 1.0
_SOURCE_PRIORITY_INFERRED = 0.35
_SOURCE_PRIORITY_UNKNOWN = 0.55

# Spec section 11: memory_type/category combinations treated as capable of
# describing conflicting claims about the same topic (e.g. two different
# "preference" rows in the same category). Raw "conversation" rows are
# intentionally excluded - ordinary chat turns aren't reconciled as
# competing claims.
_CONFLICT_PRONE_MEMORY_TYPES = frozenset({"preference", "fact"})

# Spec section 15: bounded diversity cap. No more than this many of the
# final memories may share the same category (or memory_type, when a
# candidate has no category) before later, lower-scored slots are
# preferred from a different group.
_MAX_PER_DIVERSITY_GROUP = 2


def _safe_float(value: Any, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Coerce arbitrary input into a finite float clamped to [lo, hi].
    Never raises, never returns NaN/Infinity - used for importance,
    confidence, and every derived ranking signal (spec section 8/11).
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return max(lo, min(hi, f))


def _sanitize_memory_type(value: Any) -> str:
    s = _safe_str(value, 32).strip().lower()
    return s if s in _VALID_MEMORY_TYPES else _DEFAULT_MEMORY_TYPE


def _is_finite_vector(vec: Any) -> bool:
    """True only for a non-empty list of finite (non-NaN/Inf) numbers."""
    if not isinstance(vec, list) or not vec:
        return False
    try:
        for x in vec:
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return False
        return True
    except (TypeError, ValueError):
        return False


def _parse_age_days(created_at: Any) -> float:
    """
    Return a memory's age in days from an ISO timestamp string. Malformed
    or missing timestamps default to a large age (so they neither crash
    ranking nor get an undeserved recency boost) - spec section 8.
    """
    _VERY_OLD_DAYS = 3650.0  # ~10 years; effectively zero recency signal
    if not isinstance(created_at, str) or not created_at.strip():
        return _VERY_OLD_DAYS
    try:
        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
        return max(0.0, age_seconds / 86400.0)
    except (ValueError, TypeError):
        return _VERY_OLD_DAYS


def _recency_score(age_days: float) -> float:
    """Exponential decay: 1.0 for a brand-new memory, halving every
    _RECENCY_HALF_LIFE_DAYS. Bounded to [0, 1]. Deterministic."""
    try:
        return _safe_float(0.5 ** (age_days / _RECENCY_HALF_LIFE_DAYS), default=0.0)
    except OverflowError:
        return 0.0


def _normalize_for_dedup(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _memory_type_priority(memory_type: str) -> float:
    """Bounded [0,1] prior for a (already-sanitized) memory_type. Unknown
    values fall back to the "conversation" prior rather than raising."""
    return _MEMORY_TYPE_PRIORITY.get(memory_type, _MEMORY_TYPE_PRIORITY[_DEFAULT_MEMORY_TYPE])


def _source_priority(source: Optional[str]) -> float:
    """Bounded [0,1] prior for explicit-vs-inferred provenance. Never
    raises; anything that isn't recognized is treated as neutral rather
    than penalized as "inferred" (spec section 12)."""
    s = _safe_str(source, _MAX_SOURCE_LENGTH).strip().lower()
    if s == "explicit":
        return _SOURCE_PRIORITY_EXPLICIT
    if s == "inferred":
        return _SOURCE_PRIORITY_INFERRED
    return _SOURCE_PRIORITY_UNKNOWN


# ---------------------------------------------------------------------------
# Episodic memory: short-term, in-memory, session-isolated chat history
# ---------------------------------------------------------------------------
# Kept intentionally simple and session-scoped: CHAT_HISTORY[session_id]
# is its own bounded deque, so there is no possibility of one session's
# turns leaking into another's context window. Role naming ("user" /
# "model") is preserved as-is for backward compatibility with existing
# callers (server.py calls add_to_history("model", ...) for AI replies);
# build_groq_messages translates "model" -> "assistant" only in the
# outgoing Groq-formatted payload.

CHAT_HISTORY: Dict[str, Deque[Dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_ELEMENTS)
)
_chat_history_lock = threading.Lock()


def add_to_history(role: str, text: str, session_id: str) -> None:
    """
    Append one turn to a session's bounded in-memory history. Never
    raises - a bad session_id/role/text degrades to a safe no-op rather
    than breaking the chat pipeline.
    """
    if not session_id or not isinstance(session_id, str):
        logger.warning("add_to_history called with invalid session_id; skipping.")
        return

    safe_role = _safe_role(role)
    safe_text = _safe_str(text)
    if not safe_text:
        return

    with _chat_history_lock:
        CHAT_HISTORY[session_id].append({"role": safe_role, "content": safe_text})


def build_groq_messages(long_term_context: str, session_id: str) -> List[dict]:
    """
    Build the outgoing Groq/OpenAI-compatible messages list: a system
    message (base SYSTEM_PROMPT, optionally extended with bounded
    retrieved long-term context) followed by this session's short-term
    history, translated into Groq's user/assistant role vocabulary.

    Never raises - an invalid/missing session simply yields history-free
    messages built from the system prompt alone.
    """
    system_content = SYSTEM_PROMPT
    context_text = _safe_str(long_term_context, SUPABASE_MAX_CONTEXT_CHARS).strip()
    if context_text:
        system_content = (
            f"{SYSTEM_PROMPT}\n\nRelevant memory from earlier conversations "
            f"(for your reference only, do not quote it verbatim):\n{context_text}"
        )

    messages: List[dict] = [{"role": "system", "content": system_content}]

    if session_id and isinstance(session_id, str):
        with _chat_history_lock:
            history_snapshot = list(CHAT_HISTORY.get(session_id, ()))
        for turn in history_snapshot:
            role = "assistant" if turn.get("role") in ("model", "assistant") else "user"
            content = turn.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

    return messages


# ---------------------------------------------------------------------------
# Long-term persistence: standard chat log + advanced/vector memory
# ---------------------------------------------------------------------------

def _insert_standard_log(role: str, text: str, session_id: str) -> None:
    if not SUPABASE_ENABLED:
        return
    try:
        _supabase_client.table(SUPABASE_TABLE).insert({
            "session_id": session_id,
            "role": role,
            "content": text,
            "created_at": _utc_now_iso(),
        }).execute()
    except Exception as e:
        logger.warning(f"Standard chat log insert failed for session_id={session_id}: {e}")


def _estimate_complexity_level(text: str) -> str:
    """
    Lightweight, bounded heuristic used only to populate the advanced
    table's existing "complexity" column. Purely descriptive metadata
    about message length - NOT an emotional, psychological, or intent
    judgement (that stays out of database.py entirely). Never raises.
    """
    try:
        word_count = len((text or "").split())
    except Exception:
        return "LOW"
    if word_count >= 40:
        return "HIGH"
    if word_count >= 12:
        return "MEDIUM"
    return "LOW"


# -- Phase 3B: advanced-table schema probing ---------------------------------
# The live advanced_memories table may or may not yet have the new
# metadata columns (memory_type/importance/confidence/category/source).
# Rather than requiring a migration before this file can deploy, both the
# insert and select paths try the extended shape first and transparently
# fall back to the original minimal shape on failure - caching the
# outcome so a table that's missing the columns doesn't pay a failed
# round-trip on every single call.
_advanced_schema_supports_metadata: Optional[bool] = None
_advanced_schema_lock = threading.Lock()

_EXTENDED_SELECT_COLUMNS = "content, embedding, created_at, memory_type, importance, confidence, category, source"
_BASE_SELECT_COLUMNS = "content, embedding, created_at"


def _insert_advanced_memory(
    role: str,
    text: str,
    session_id: str,
    memory_type: str = _DEFAULT_MEMORY_TYPE,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    global _advanced_schema_supports_metadata

    if not ADVANCED_DB_ENABLED:
        return
    try:
        embedding = generate_embedding(text)
    except Exception as e:
        logger.warning(f"Embedding generation failed for session_id={session_id}: {e}")
        embedding = None

    if embedding is None or not _is_finite_vector(embedding):
        # No usable embedding - skip the vector table rather than writing
        # a row that can never be retrieved by similarity search.
        return

    base_payload = {
        "session_id": session_id,
        "role": role,
        "content": text,
        "embedding": embedding,
        "complexity": _estimate_complexity_level(text),
        "created_at": _utc_now_iso(),
    }
    extended_payload = dict(base_payload)
    extended_payload.update({
        "memory_type": _sanitize_memory_type(memory_type),
        "importance": _safe_float(importance, _DEFAULT_IMPORTANCE),
        "confidence": _safe_float(confidence, _DEFAULT_CONFIDENCE),
        "category": _safe_str(category, _MAX_CATEGORY_LENGTH) if category else None,
        "source": _safe_str(source, _MAX_SOURCE_LENGTH) if source else None,
    })

    with _advanced_schema_lock:
        schema_state = _advanced_schema_supports_metadata

    if schema_state is not False:
        try:
            _advanced_client.table(ADVANCED_DB_TABLE).insert(extended_payload).execute()
            with _advanced_schema_lock:
                _advanced_schema_supports_metadata = True
            return
        except Exception as e:
            if schema_state is True:
                # Previously confirmed to work; treat as a transient
                # failure rather than a schema mismatch, and give up for
                # this call (matches original single-attempt behavior).
                logger.warning(f"Advanced/vector memory insert failed for session_id={session_id}: {e}")
                return
            logger.info(
                f"Extended advanced-memory insert failed (likely missing metadata "
                f"columns); falling back to base schema: {e}"
            )

    try:
        _advanced_client.table(ADVANCED_DB_TABLE).insert(base_payload).execute()
        with _advanced_schema_lock:
            if _advanced_schema_supports_metadata is None:
                _advanced_schema_supports_metadata = False
    except Exception as e:
        logger.warning(f"Advanced/vector memory insert failed for session_id={session_id}: {e}")


def _route_and_save_sync(
    role: str,
    text: str,
    session_id: str,
    memory_type: str = _DEFAULT_MEMORY_TYPE,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    safe_role = _safe_role(role)
    safe_text = _safe_str(text)
    safe_session = _safe_str(session_id, 128)
    if not safe_text or not safe_session:
        return
    # Failure isolation: standard log and advanced memory are independent
    # writes - a failure in one must never prevent the other.
    try:
        _insert_standard_log(safe_role, safe_text, safe_session)
    except Exception as e:
        logger.warning(f"route_and_save_bg standard-log path failed: {e}")
    try:
        _insert_advanced_memory(
            safe_role, safe_text, safe_session,
            memory_type=memory_type, importance=importance,
            confidence=confidence, source=source, category=category,
        )
    except Exception as e:
        logger.warning(f"route_and_save_bg advanced-memory path failed: {e}")


def route_and_save_bg(
    role: str,
    text: str,
    session_id: str,
    memory_type: str = _DEFAULT_MEMORY_TYPE,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """
    Fire-and-forget persistence of one chat turn to the standard chat
    log and (if configured) the advanced/vector memory table. Submits to
    a small bounded background thread pool and returns immediately -
    callers do NOT need to wrap this in asyncio.to_thread, though doing
    so (as server.py does) is harmless.

    If both SUPABASE_ENABLED and ADVANCED_DB_ENABLED are False, this is
    a cheap no-op.

    Phase 3B (optional, backward compatible): `memory_type`,
    `importance`, `confidence`, `source`, and `category` let an
    orchestration layer attach bounded classification metadata to this
    turn for later ranked retrieval. Every existing call site that omits
    them gets identical behavior to before (memory_type defaults to
    "conversation", matching what was always persisted).
    """
    if not SUPABASE_ENABLED and not ADVANCED_DB_ENABLED:
        return
    _run_in_background(
        _route_and_save_sync, role, text, session_id,
        memory_type=memory_type, importance=importance,
        confidence=confidence, source=source, category=category,
    )


def save_semantic_memory_bg(
    session_id: str,
    content: str,
    memory_type: str = "fact",
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """
    Phase 3B (new, additive): explicitly persist a single classified
    long-term memory (e.g. a distilled preference, fact, or interest)
    into the advanced/vector table, independent of raw per-turn chat
    logging.

    This exists so the orchestration layer can choose to promote a
    specific, already-classified piece of information to long-term
    memory WITHOUT every ordinary chat message being auto-promoted -
    the automatic per-turn persistence in route_and_save_bg/
    _route_and_save_sync is unchanged and continues to run
    independently of this function.

    Fire-and-forget, bounded, sanitized, and a safe no-op if the
    advanced database isn't configured. Never raises.
    """
    if not ADVANCED_DB_ENABLED:
        return
    safe_session = _safe_str(session_id, 128)
    safe_content = _safe_str(content)
    if not safe_session or not safe_content:
        return
    _run_in_background(
        _insert_advanced_memory,
        "system", safe_content, safe_session,
        memory_type=memory_type, importance=importance,
        confidence=confidence, source=source, category=category,
    )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    try:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return -1.0
        return dot / (norm_a * norm_b)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Phase 3B: relevance-retrieval pipeline
# ---------------------------------------------------------------------------
# QUERY -> EMBEDDING -> SCOPED CANDIDATE RETRIEVAL -> VALIDATION/FILTERING
# -> SEMANTIC SIMILARITY -> RECENCY -> IMPORTANCE -> CONFIDENCE
# -> DEDUPLICATION -> CONTEXT BUDGET -> FINAL RELEVANT MEMORIES
#
# Structured so a future migration to server-side pgvector/RPC retrieval
# only needs to replace `_fetch_candidate_rows` - everything downstream
# (validation, ranking, dedup, budgeting) is storage-agnostic and works
# on plain dicts.

class _RankedMemory:
    __slots__ = ("content", "similarity", "recency", "importance", "confidence",
                 "score", "memory_type", "category", "source", "created_at")

    def __init__(self, content: str, similarity: float, recency: float,
                 importance: float, confidence: float, memory_type: str,
                 category: Optional[str], source: Optional[str], created_at: str):
        self.content = content
        self.similarity = similarity
        self.recency = recency
        self.importance = importance
        self.confidence = confidence
        self.memory_type = memory_type
        self.category = category
        self.source = source
        self.created_at = created_at
        # Phase 3C: two additional small, bounded terms (memory_type and
        # source priors) on top of the original four-signal formula.
        # _safe_float guarantees the final score is always finite and
        # clamped, even if a future signal is added carelessly upstream
        # (spec section 3/20 - never NaN/Infinity/out-of-range).
        self.score = _safe_float(
            _WEIGHT_SIMILARITY * similarity
            + _WEIGHT_RECENCY * recency
            + _WEIGHT_IMPORTANCE * importance
            + _WEIGHT_CONFIDENCE * confidence
            + _WEIGHT_MEMORY_TYPE * _memory_type_priority(memory_type)
            + _WEIGHT_SOURCE * _source_priority(source),
            default=0.0, lo=-1.0, hi=1.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": round(self.score, 6),
            "similarity": round(self.similarity, 6),
            "recency": round(self.recency, 6),
            "importance": round(self.importance, 6),
            "confidence": round(self.confidence, 6),
            "memory_type": self.memory_type,
            "category": self.category,
            "source": self.source,
            "created_at": self.created_at,
        }


def _fetch_candidate_rows(safe_session: str) -> List[Dict[str, Any]]:
    """
    Scoped candidate retrieval (spec section 6/7): bounded row fetch,
    filtered by session_id at the query level so retrieval can never
    surface another session's memory. Tries the extended metadata
    columns first, falling back to the original minimal shape if the
    live table doesn't have them yet (see schema-probing note above).
    """
    global _advanced_schema_supports_metadata

    with _advanced_schema_lock:
        schema_state = _advanced_schema_supports_metadata

    if schema_state is not False:
        try:
            resp = (
                _advanced_client.table(ADVANCED_DB_TABLE)
                .select(_EXTENDED_SELECT_COLUMNS)
                .eq("session_id", safe_session)
                .order("created_at", desc=True)
                .limit(_MAX_CANDIDATE_ROWS)
                .execute()
            )
            with _advanced_schema_lock:
                _advanced_schema_supports_metadata = True
            return resp.data or []
        except Exception as e:
            if schema_state is True:
                logger.warning(f"Advanced memory query failed for session_id={safe_session}: {e}")
                return []
            logger.info(
                f"Extended advanced-memory select failed (likely missing metadata "
                f"columns); falling back to base schema: {e}"
            )

    try:
        resp = (
            _advanced_client.table(ADVANCED_DB_TABLE)
            .select(_BASE_SELECT_COLUMNS)
            .eq("session_id", safe_session)
            .order("created_at", desc=True)
            .limit(_MAX_CANDIDATE_ROWS)
            .execute()
        )
        with _advanced_schema_lock:
            if _advanced_schema_supports_metadata is None:
                _advanced_schema_supports_metadata = False
        return resp.data or []
    except Exception as e:
        logger.warning(f"Advanced memory query failed for session_id={safe_session}: {e}")
        return []


def _validate_and_score_rows(rows: List[Dict[str, Any]], query_embedding: List[float]) -> List[_RankedMemory]:
    """
    Validation/filtering + semantic/recency/importance/confidence
    scoring (spec sections 6, 8). A single corrupted row is dropped,
    never allowed to crash retrieval.
    """
    ranked: List[_RankedMemory] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        content = row.get("content")
        embedding = row.get("embedding")
        if not isinstance(content, str) or not content.strip():
            continue
        if not _is_finite_vector(embedding):
            continue

        similarity = _cosine_similarity(query_embedding, embedding)
        if similarity <= -1.0:
            # Dimension mismatch or degenerate vector - unusable.
            continue

        age_days = _parse_age_days(row.get("created_at"))
        recency = _recency_score(age_days)
        importance = _safe_float(row.get("importance"), _DEFAULT_IMPORTANCE)
        confidence = _safe_float(row.get("confidence"), _DEFAULT_CONFIDENCE)
        memory_type = _sanitize_memory_type(row.get("memory_type"))
        category = _safe_str(row.get("category"), _MAX_CATEGORY_LENGTH) or None
        source = _safe_str(row.get("source"), _MAX_SOURCE_LENGTH) or None
        created_at = _safe_str(row.get("created_at"), 64)

        ranked.append(_RankedMemory(
            content=content[:_MAX_CONTENT_CHARS],
            similarity=_safe_float(similarity, 0.0, lo=-1.0, hi=1.0),
            recency=recency,
            importance=importance,
            confidence=confidence,
            memory_type=memory_type,
            category=category,
            source=source,
            created_at=created_at,
        ))

    return ranked


def _deduplicate(ranked: List[_RankedMemory]) -> List[_RankedMemory]:
    """
    Bounded, deterministic deduplication (spec section 12): drop a
    candidate whose normalized content exactly matches, or is wholly
    contained in, an already-kept (higher-scored) candidate. No ML
    clustering - candidates are already bounded to a small list by the
    time this runs, so this stays cheap.
    """
    kept: List[_RankedMemory] = []
    seen_normalized: List[str] = []
    for candidate in ranked:
        normalized = _normalize_for_dedup(candidate.content)
        if not normalized:
            continue
        is_duplicate = any(
            normalized == existing or normalized in existing or existing in normalized
            for existing in seen_normalized
        )
        if is_duplicate:
            continue
        kept.append(candidate)
        seen_normalized.append(normalized)
    return kept


def _conflict_group_key(m: "_RankedMemory") -> Optional[tuple]:
    """Groups candidates that can plausibly make competing claims about
    the same topic (spec section 11). Only preference/fact rows with an
    actual category participate - everything else is left alone."""
    if m.memory_type in _CONFLICT_PRONE_MEMORY_TYPES and m.category:
        return (m.memory_type, _normalize_for_dedup(m.category))
    return None


def _conflict_priority(m: "_RankedMemory") -> tuple:
    """Tie-break used to pick the single surviving memory within a
    conflict group: explicit beats inferred/unknown, then higher
    confidence, then more recent, then higher blended score. Purely a
    selection order - never mutates or deletes anything in storage."""
    source_rank = 1 if _safe_str(m.source, _MAX_SOURCE_LENGTH).strip().lower() == "explicit" else 0
    return (source_rank, m.confidence, m.recency, m.score)


def _resolve_conflicts(ranked: List["_RankedMemory"]) -> List["_RankedMemory"]:
    """
    Bounded conflict resolution (spec section 11). `ranked` is expected
    already sorted best-first by score. For each conflict-prone
    (memory_type, category) group, keep only the strongest candidate per
    _conflict_priority; every other candidate (any memory_type/category
    outside the conflict-prone set, or with no category at all) passes
    through untouched. Historical rows are never deleted - this only
    narrows what a single retrieval call surfaces.
    """
    if not ranked:
        return []

    best_by_key: Dict[tuple, "_RankedMemory"] = {}
    passthrough: List["_RankedMemory"] = []

    for m in ranked:
        key = _conflict_group_key(m)
        if key is None:
            passthrough.append(m)
            continue
        current = best_by_key.get(key)
        if current is None or _conflict_priority(m) > _conflict_priority(current):
            best_by_key[key] = m

    resolved = passthrough + list(best_by_key.values())
    resolved.sort(key=lambda m: m.score, reverse=True)
    return resolved


def _diversity_group_key(m: "_RankedMemory") -> str:
    """Grouping used only for the diversity cap (spec section 15) - falls
    back to memory_type when a candidate has no category, since category
    is often unset."""
    return m.category or m.memory_type


def _diversify_selection(ranked: List["_RankedMemory"], k: int) -> List["_RankedMemory"]:
    """
    Bounded relevance-diversity selection (spec section 15). `ranked` is
    expected already sorted best-first by score. Greedily takes the
    best-scoring candidates while capping how many may share a diversity
    group at _MAX_PER_DIVERSITY_GROUP; if that leaves unused slots (not
    enough distinct groups to fill k under the cap), the remaining slots
    are backfilled from the skipped candidates in score order, so k is
    still reached whenever enough candidates exist. Deterministic and
    never returns more than k or more than len(ranked) items.
    """
    if not ranked or k <= 0:
        return []

    selected: List["_RankedMemory"] = []
    overflow: List["_RankedMemory"] = []
    group_counts: Dict[str, int] = defaultdict(int)

    for m in ranked:
        if len(selected) >= k:
            break
        key = _diversity_group_key(m)
        if group_counts[key] < _MAX_PER_DIVERSITY_GROUP:
            selected.append(m)
            group_counts[key] += 1
        else:
            overflow.append(m)

    if len(selected) < k:
        for m in overflow:
            if len(selected) >= k:
                break
            selected.append(m)

    return selected


def _final_k() -> int:
    return max(1, min(SUPABASE_SEARCH_TOP_K, _HARD_MAX_FINAL_MEMORIES))


def _rank_and_select(rows: List[Dict[str, Any]], query_embedding: List[float]) -> List[_RankedMemory]:
    """
    Full pipeline (spec sections 4-15): validate/score -> dedup ->
    resolve same-topic conflicts -> apply the minimum relevance
    threshold -> bounded-diversity final selection. Returns the final,
    ordered (best first) list of memories - deterministic, and an empty
    list is a valid, expected outcome at every stage (a database or
    embedding failure, no candidates, everything below threshold, etc.
    all safely collapse to []).
    """
    ranked = _validate_and_score_rows(rows, query_embedding)
    if not ranked:
        return []

    ranked.sort(key=lambda m: m.score, reverse=True)
    deduped = _deduplicate(ranked)
    if not deduped:
        return []

    resolved = _resolve_conflicts(deduped)
    if not resolved:
        return []

    relevant = [m for m in resolved if m.similarity >= _MIN_SIMILARITY_THRESHOLD]
    if not relevant:
        return []

    return _diversify_selection(relevant, _final_k())


def _retrieve_ranked_memories_sync(user_message: str, session_id: str) -> List[_RankedMemory]:
    if not ADVANCED_DB_ENABLED:
        return []

    safe_session = _safe_str(session_id, 128)
    if not safe_session:
        return []

    safe_message = _safe_str(user_message)
    if not safe_message.strip():
        return []

    try:
        query_embedding = generate_embedding(safe_message)
    except Exception as e:
        logger.warning(f"Embedding generation failed during retrieval for session_id={safe_session}: {e}")
        return []

    if not _is_finite_vector(query_embedding):
        return []

    rows = _fetch_candidate_rows(safe_session)
    if not rows:
        return []

    final_memories = _rank_and_select(rows, query_embedding)
    logger.info(
        f"Long-term retrieval for session_id={safe_session}: "
        f"{len(rows)} candidates -> {len(final_memories)} final memories"
    )
    return final_memories


def retrieve_relevant_memories(
    user_message: str,
    session_id: str,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 3B (new, additive): structured version of long-term memory
    retrieval. Runs the full QUERY -> EMBEDDING -> CANDIDATE RETRIEVAL ->
    VALIDATION -> SIMILARITY/RECENCY/IMPORTANCE/CONFIDENCE -> DEDUP ->
    BUDGET pipeline and returns each surviving memory as a plain dict
    (content, score, per-signal breakdown, memory_type, category,
    source, created_at) instead of a flattened string.

    `top_k`, if given, further bounds the number of returned memories
    below the configured default; it never increases it beyond
    _HARD_MAX_FINAL_MEMORIES.

    Deterministic, bounded, session-isolated, and failure-tolerant:
    returns an empty list (never raises) if the advanced database is
    disabled, unreachable, empty, or every candidate is malformed.
    Synchronous and potentially slow (network + embedding call) -
    callers should invoke this via asyncio.to_thread, exactly like
    retrieve_long_term_context.
    """
    memories = _run_with_timeout(
        _retrieve_ranked_memories_sync,
        SUPABASE_SEARCH_TIMEOUT_SECONDS,
        [],
        user_message,
        session_id,
    )
    if top_k is not None:
        try:
            k = max(1, min(int(top_k), _HARD_MAX_FINAL_MEMORIES))
        except (TypeError, ValueError):
            k = _final_k()
        memories = memories[:k]
    return [m.to_dict() for m in memories]


def _retrieve_long_term_context_sync(user_message: str, session_id: str) -> str:
    memories = _retrieve_ranked_memories_sync(user_message, session_id)
    if not memories:
        return ""

    chunks = [m.content[:_MAX_MEMORY_ITEM_CHARS] for m in memories]
    context = "\n".join(chunks)
    return context[:SUPABASE_MAX_CONTEXT_CHARS]


def retrieve_long_term_context(user_message: str, session_id: str) -> str:
    """
    Retrieve bounded, session-isolated long-term context relevant to
    user_message via the ranked retrieval pipeline over the
    advanced/vector memory table (semantic similarity + recency +
    importance + confidence, deduplicated, then budgeted). Synchronous
    and potentially slow (network + embedding call); callers should
    invoke this via asyncio.to_thread, exactly as server.py already
    does.

    Timeout-protected, failure-tolerant, and safe when the advanced
    database is disabled, unreachable, empty, or returns malformed rows
    - always returns a (possibly empty) string, never raises.

    Return type/signature unchanged from the pre-Phase-3B version;
    ranking quality improved underneath (see retrieve_relevant_memories
    for the structured, explainable equivalent).
    """
    return _run_with_timeout(
        _retrieve_long_term_context_sync,
        SUPABASE_SEARCH_TIMEOUT_SECONDS,
        "",
        user_message,
        session_id,
    )


# ---------------------------------------------------------------------------
# Personalization persistence
# ---------------------------------------------------------------------------
# Stores/restores ONLY the fields user_profile.py's UserProfile /
# profile_from_dict already define - no API keys, passwords, tokens, or
# other secrets ever pass through here. All incoming data is bounded and
# sanitized before being written, and all outgoing (loaded) data is
# bounded and sanitized again before being handed back to the caller, so
# a corrupted or tampered row can never crash the caller or blow up
# memory.

def _save_personalization_profile_sync(
    session_id: str,
    category_counts: Dict[str, int],
    recent_intents: List[str],
    preferred_response_length: str,
    preferred_interaction_style: str,
    interests: List[str],
    preference_scores: Dict[str, float],
) -> None:
    if not SUPABASE_ENABLED:
        return

    safe_session = _safe_str(session_id, 128)
    if not safe_session:
        return

    payload = {
        "session_id": safe_session,
        "category_counts": _sanitize_category_counts(category_counts),
        "recent_intents": _sanitize_str_list(recent_intents, _MAX_RECENT_INTENTS_STORED, 40),
        "preferred_response_length": _safe_str(preferred_response_length, 20) or "MEDIUM",
        "preferred_interaction_style": _safe_str(preferred_interaction_style, 20) or "FRIENDLY",
        "interests": _sanitize_str_list(interests, _MAX_INTERESTS_STORED, _MAX_INTEREST_LENGTH),
        "preference_scores": _sanitize_preference_scores(preference_scores),
        "updated_at": _utc_now_iso(),
    }

    try:
        _supabase_client.table(PERSONALIZATION_TABLE).upsert(
            payload, on_conflict="session_id"
        ).execute()
    except Exception as e:
        logger.warning(f"Personalization profile save failed for session_id={safe_session}: {e}")


def save_personalization_profile_bg(
    session_id: str,
    category_counts: Optional[Dict[str, int]] = None,
    recent_intents: Optional[List[str]] = None,
    preferred_response_length: str = "MEDIUM",
    preferred_interaction_style: str = "FRIENDLY",
    interests: Optional[List[str]] = None,
    preference_scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Fire-and-forget, bounded, sanitized upsert of a session's
    personalization profile. Submits to the bounded background thread
    pool and returns immediately - safe to call directly from an async
    context without asyncio.to_thread. A no-op if Supabase isn't
    configured.
    """
    if not SUPABASE_ENABLED:
        return
    _run_in_background(
        _save_personalization_profile_sync,
        session_id,
        category_counts or {},
        recent_intents or [],
        preferred_response_length,
        preferred_interaction_style,
        interests or [],
        preference_scores or {},
    )


def _load_personalization_profile_sync(session_id: str) -> Optional[Dict[str, Any]]:
    if not SUPABASE_ENABLED:
        return None

    safe_session = _safe_str(session_id, 128)
    if not safe_session:
        return None

    try:
        resp = (
            _supabase_client.table(PERSONALIZATION_TABLE)
            .select("*")
            .eq("session_id", safe_session)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        logger.warning(f"Personalization profile load failed for session_id={safe_session}: {e}")
        return None

    if not rows or not isinstance(rows[0], dict):
        return None

    row = rows[0]
    # Never trust the returned JSON blindly - sanitize/bound every field
    # again on the way out, same as on the way in. Any field absent from
    # an older row (or any new field a future writer might add) safely
    # falls back to its default here, so both old and new profile
    # records remain loadable without a migration.
    return {
        "category_counts": _sanitize_category_counts(row.get("category_counts")),
        "total_interactions": max(0, int(row.get("total_interactions") or 0)) if str(row.get("total_interactions") or "0").lstrip("-").isdigit() else 0,
        "recent_intents": _sanitize_str_list(row.get("recent_intents"), _MAX_RECENT_INTENTS_STORED, 40),
        "preferred_response_length": _safe_str(row.get("preferred_response_length"), 20) or "MEDIUM",
        "preferred_interaction_style": _safe_str(row.get("preferred_interaction_style"), 20) or "FRIENDLY",
        "interests": _sanitize_str_list(row.get("interests"), _MAX_INTERESTS_STORED, _MAX_INTEREST_LENGTH),
        "preference_scores": _sanitize_preference_scores(row.get("preference_scores")),
        "updated_at": _safe_str(row.get("updated_at"), 64),
    }


def load_personalization_profile(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a session's persisted personalization profile as a bounded,
    sanitized dict (or None if unavailable/not found/disabled/corrupted
    beyond repair). Synchronous and timeout-protected; intended to be
    called via asyncio.to_thread by whichever module owns session-start
    hydration - never raises.
    """
    return _run_with_timeout(
        _load_personalization_profile_sync,
        PERSONALIZATION_TIMEOUT_SECONDS,
        None,
        session_id,
    )
