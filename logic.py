import re
import math
import ast
import operator
import datetime
import logging
import threading
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---- All functions in this original section are pure/CPU-bound (regex + safe-eval math)
# with no I/O, no shared mutable state, and no per-session data - so they are
# already safe to call directly from a FastAPI WebSocket handler (or offloaded
# to a worker thread via asyncio.to_thread, as server.py already does for
# handle_local_queries) without freezing the event loop. Nothing here needed
# to become `async def`; the safety work below is defensive exception handling
# so a malformed/unexpected input can never bubble up and crash a connection.
#
# NONE OF THIS SECTION IS MODIFIED. All existing public functions, signatures,
# and behavior (date/time queries, local math evaluation, query
# categorization/classification) are preserved exactly as-is for full
# backward compatibility with server.py and database.py.

_ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {"sqrt", "abs", "round"}

_SQRT_PHRASE_RE = re.compile(r'square\s*root\s*of\s*(-?\d+(?:\.\d+)?)')
_TIME_QUERY_RE = re.compile(r"(what(?:'s| is)?\s+(?:the\s+)?(?:current\s+)?time(?:\s+is\s+it)?|time\s+is\s+it|current\s+time|tell\s+me\s+the\s+time|give\s+me\s+the\s+time)\s*[\?!.]*$")
_DATE_QUERY_RE = re.compile(r"(what(?:'s| is)?\s+(?:the\s+)?(?:current\s+)?date(?:\s+today)?|what\s+day\s+is\s+it|today'?s\s+date|current\s+date|tell\s+me\s+the\s+date|give\s+me\s+the\s+date)\s*[\?!.]*$")
_MATH_SAFE_CHARS_RE = re.compile(r'^[\d\s.\+\-\*/\(\)]*$')
_HAS_DIGIT_RE = re.compile(r'\d')
_HAS_OPERATOR_RE = re.compile(r'[\+\-\*/]')

def _eval_math_node(node):
    if isinstance(node, ast.Expression): return _eval_math_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool): return node.value
        raise ValueError("Unsupported constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_math_node(node.left), _eval_math_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_math_node(node.operand))
    if isinstance(node, ast.Call):
        func_name = getattr(node.func, "id", None)
        if func_name in _ALLOWED_FUNCS and len(node.args) == 1 and not node.keywords:
            arg = _eval_math_node(node.args[0])
            if func_name == "sqrt":
                if arg < 0: raise ValueError("negative sqrt")
                return math.sqrt(arg)
            if func_name == "abs": return abs(arg)
            if func_name == "round": return round(arg)
    raise ValueError("Unsupported node")

def _safe_eval_math(expr: str):
    parsed = ast.parse(expr, mode="eval")
    return _eval_math_node(parsed)

def handle_local_queries(query: str):
    try:
        if not query: return None
        lowered = query.strip().lower()

        if _DATE_QUERY_RE.search(lowered):
            ist_now = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
            return f"NORMAL|Today's date is {ist_now.strftime('%A, %B %d, %Y')}."
        if _TIME_QUERY_RE.search(lowered):
            ist_now = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
            return f"NORMAL|The current time is {ist_now.strftime('%I:%M %p')} IST."

        expr = _SQRT_PHRASE_RE.sub(r'sqrt(\1)', lowered).replace("square root of", "sqrt")
        for phrase in ("what is", "what's", "whats", "calculate", "solve", "compute", "evaluate"):
            expr = expr.replace(phrase, "")
        expr = expr.strip(" ?=.").replace("^", "**")

        if not _HAS_DIGIT_RE.search(expr): return None
        check = expr.replace("sqrt", "")
        if not _MATH_SAFE_CHARS_RE.match(check): return None
        if not _HAS_OPERATOR_RE.search(expr) and "sqrt" not in expr: return None

        try:
            result = _safe_eval_math(expr)
        except Exception:
            return None
        if isinstance(result, float): result = int(result) if result.is_integer() else round(result, 4)
        return f"EXCITED|The answer is {result}."
    except Exception as e:
        # Any unexpected failure here (bad input type, regex edge case, etc.)
        # falls through to the normal AI path instead of crashing the caller.
        logger.warning(f"handle_local_queries failed unexpectedly, falling back to AI: {e}")
        return None

_PROGRAMMING_KEYWORDS = ("code", "program", "python", "java", "c++", "javascript", "function", "variable", "sql", "esp32", "arduino")
_SCIENCE_KEYWORDS = ("force", "energy", "atom", "molecule", "physics", "chemistry", "biology", "velocity", "acceleration", "dna", "electron")

def categorize_query(text: str) -> str:
    try:
        lowered = (text or "").lower()
        if any(kw in lowered for kw in _PROGRAMMING_KEYWORDS): return "Programming"
        if any(kw in lowered for kw in _SCIENCE_KEYWORDS): return "Science"
        return "General"
    except Exception as e:
        logger.warning(f"categorize_query failed unexpectedly: {e}")
        return "General"

def estimate_token_count(text: str) -> int:
    try:
        return len(text.split()) if text else 0
    except Exception as e:
        logger.warning(f"estimate_token_count failed unexpectedly: {e}")
        return 0

_HIGH_COMPLEXITY_KEYWORDS = ("integral", "derivative", "differential equation", "quantum", "hybridization", "thermodynamics")
_MEDIUM_COMPLEXITY_KEYWORDS = ("velocity", "acceleration", "force", "momentum", "reaction", "organic", "algebra")
_COMPLEX_MATH_SYMBOL_RE = re.compile(r'∫|∑|∏|∂|√|±|≤|≥|≠|→|\bmatrix\b|\bdeterminant\b|\\frac|\\int|\\sum|\\prod')
_CALCULUS_SHORTHAND_RE = re.compile(r'\bd\^?\d*[a-z]\^?\d*\s*/\s*d\^?\d*[a-z]\^?\d*\b|\blim(?:it)?\b')
_CHEM_FORMULA_RE = re.compile(r'\b(?=[A-Za-z]*\d)(?:[A-Z][a-z]?\d*){2,}\b')
_DEFINITION_PATTERN_RE = re.compile(r'^\s*(?:what\s+(?:is|are|does)\b|define\b|meaning\s+of\b|explain\b)')
_PROBLEM_SOLVING_VERBS_RE = re.compile(r'\b(?:find|calculate|compute|solve|derive|evaluate|prove|integrate|differentiate)\b')
_OPERATOR_CHAR_RE = re.compile(r'[\+\-\*/\^=<>]')

def classify_query(query: str) -> str:
    try:
        if not query or not query.strip(): return "NORMAL"
        lowered = query.lower()
        stripped_lowered = lowered.strip()
        word_count = len(query.split())

        high_hits = sum(1 for kw in _HIGH_COMPLEXITY_KEYWORDS if kw in lowered)
        medium_hits = sum(1 for kw in _MEDIUM_COMPLEXITY_KEYWORDS if kw in lowered)
        symbol_hits = len(_COMPLEX_MATH_SYMBOL_RE.findall(lowered))
        calculus_hits = len(_CALCULUS_SHORTHAND_RE.findall(lowered))
        chem_hits = len(_CHEM_FORMULA_RE.findall(query))
        distinct_operators = len(set(_OPERATOR_CHAR_RE.findall(query)))
        is_definition = bool(_DEFINITION_PATTERN_RE.match(stripped_lowered))
        is_problem_verb = bool(_PROBLEM_SOLVING_VERBS_RE.search(lowered))
        dense_notation = symbol_hits > 0 or calculus_hits > 0 or chem_hits > 0

        if is_definition and not is_problem_verb and not dense_notation:
            if high_hits >= 2: return "HIGH"
            if high_hits >= 1 or medium_hits >= 1: return "MEDIUM"
            return "NORMAL"

        score = (high_hits * 3.0 + medium_hits * 1.5 + symbol_hits * 2.5 + calculus_hits * 3.0 + chem_hits * 2.0 + min(distinct_operators, 5) * 0.5 + min(word_count / 6.0, 2.0))
        if is_problem_verb and (high_hits or medium_hits or dense_notation): score += 2.5

        if score >= 4.0: return "HIGH"
        if score >= 1.5: return "MEDIUM"
        return "NORMAL"
    except Exception as e:
        logger.warning(f"classify_query failed unexpectedly: {e}")
        return "NORMAL"


# ===========================================================================
# NEW: Personalization orchestration
# ===========================================================================
#
# Everything below is ADDITIVE. It introduces a new, optional pipeline that
# wires together:
#
#   intent_router.py    -> deterministic intent detection (ground truth)
#   user_profile.py     -> per-session profile (history, response style)
#   preferences.py      -> per-session category preference weights
#   personalization.py  -> combines intent + profile + preferences + context
#   response_policy.py  -> turns personalization into structured response
#                           instructions (length/depth/tone/examples/etc.)
#   context_memory.py   -> bounded, session-isolated short-term memory
#   emotion_engine.py   -> per-turn conversational emotion/situation signal
#   personality_engine.py -> turns emotion+intent+preferences into a
#                           concrete tone/style decision
#
# HARD RULE (matches personalization.py / response_policy.py contracts):
# explicit user intent, as determined by intent_router.route_intent, is
# NEVER overridden by personalization, the Emotion Engine, or the
# Personality Engine. This module only ever calls `personalize()`,
# `build_response_policy()`, `analyze_emotion()`, and `build_personality()`
# - none of which are capable of changing the routed intent. Emotion and
# Personality only ever shape HOW the reply sounds (tone/style/length/
# opening/closing/humor/encouragement) around the intent that
# intent_router already decided; they never touch WHAT is answered.
#
# This module remains free of:
#   - WebSocket code (server.py owns the socket)
#   - persistence/storage internals (database.py owns those: Supabase
#     clients, embeddings storage, ranking-pipeline internals). The
#     rebuilt database.py confirms it never imports logic.py or server.py,
#     so the dependency direction is one-way (logic.py -> database.py) and
#     safe: this module MAY consume database.py's already-bounded,
#     already-session-scoped read APIs (see the Phase 3C semantic-memory
#     section below), but must never be imported back by database.py.
#   - Fast Math / game logic (game_manager.py + games/ own that; server.py
#     already intercepts game messages via handle_game_message BEFORE
#     falling through to the normal chat path, and that ordering is
#     unchanged - active game turns never reach emotion_engine,
#     personality_engine, personalization, semantic memory retrieval, or
#     the AI provider)
#
# ---------------------------------------------------------------------------
# Phase 3C: Advanced Semantic Memory Orchestration (additive)
# ---------------------------------------------------------------------------
# This module now also wires in database.py's existing, already-implemented
# semantic-memory retrieval (retrieve_relevant_memories: embedding ->
# candidate fetch -> similarity/recency/importance/confidence ranking ->
# relevance threshold -> conflict resolution -> diversity -> budget) as one
# more optional, best-effort enrichment stage in run_personalized_pipeline,
# strictly between USER PROFILE and PERSONALIZATION:
#
#     ... -> user profile lookup -> semantic memory retrieval ->
#     personalization -> response policy -> emotion -> personality ->
#     combined response guidance -> AI model
#
# Ground rules for this stage (all enforced below):
#   - Optional: any failure (import, network, embedding, timeout,
#     malformed data) degrades to "no semantic context" and the pipeline
#     continues exactly as it did before this integration.
#   - Never overrides explicit user intent, routing, safety, or the
#     current request - it is only ever folded into the AI-facing
#     messages as clearly-labeled supporting evidence, never treated as
#     ground truth, and never fed into intent_router, personalization's
#     routed intent, response_policy, emotion_engine, or personality_engine.
#   - Session-scoped only: the same session_id used everywhere else in
#     this module is the only identifier ever passed to retrieval.
#   - Skipped entirely for GAME-routed turns (belt-and-suspenders on top
#     of server.py's existing game-message interception) and for
#     empty/invalid input (already short-circuited above).
#   - Bounded: capped item count, capped characters per item, capped
#     total characters, deduplicated against what's already in short-term
#     context, so it can never crowd out the current message, recent
#     history, the system prompt, or response policy/personality guidance
#     within the AI provider's context budget.
#   - Never logs raw retrieved memory content or per-signal scores -
#     only counts/attempted/used flags (see _retrieve_semantic_memory_safe
#     and the "semantic_memory" key in run_personalized_pipeline's return
#     value).
#
# ai_services.call_groq is imported directly here since ai_services.py has
# no dependency back on logic.py, so no cycle is introduced. The AI
# provider call itself (retries, backoff, error handling) is left
# entirely to ai_services.py - this module only supplies it with messages
# and an optional structured response_policy dict (now enriched with
# Personality Engine style instructions, still capped/sanitized the same
# way by ai_services.py's own _build_policy_preamble).

from intent_router import route_intent, Intent  # noqa: E402
from personalization import (  # noqa: E402
    ConversationContext,
    personalize,
    PersonalizationError,
)
from response_policy import (  # noqa: E402
    PolicyContext,
    build_response_policy,
    ResponsePolicyError,
)
from user_profile import (  # noqa: E402
    UserProfile,
    create_default_profile,
    ProfileValidationError,
)
from preferences import (  # noqa: E402
    PreferenceProfile,
    get_default_profile,
    PreferenceValidationError,
)
from context_memory import ContextMemoryManager  # noqa: E402

# ---------------------------------------------------------------------------
# Emotion Engine / Personality Engine (additive, optional)
# ---------------------------------------------------------------------------
# Imported defensively, the same way SYSTEM_PROMPT/ai_services are below:
# if either module is ever unavailable, the pipeline degrades to its
# pre-existing behavior (neutral emotion / response_policy-only guidance)
# instead of failing to import.

try:
    from emotion_engine import (
        analyze_emotion,
        EmotionResult,
        Emotion as _Emotion,
        Situation as _Situation,
    )
except Exception:  # pragma: no cover - emotion_engine should always be importable
    analyze_emotion = None
    EmotionResult = None
    _Emotion = None
    _Situation = None

try:
    from personality_engine import (
        build_personality,
        PersonalityContext,
        PersonalityDecision,
    )
except Exception:  # pragma: no cover - personality_engine should always be importable
    build_personality = None
    PersonalityContext = None
    PersonalityDecision = None

try:
    from config import SYSTEM_PROMPT
except Exception:  # pragma: no cover - config should always provide this
    SYSTEM_PROMPT = (
        "You are a helpful, encouraging AI study companion. Keep replies concise."
    )

try:
    from config import ROBOT_CONTROL_SYSTEM_PROMPT, CONVERSATIONAL_SYSTEM_PROMPT
except Exception:  # pragma: no cover - config should always provide these (Phase 4 / dual-mode)
    ROBOT_CONTROL_SYSTEM_PROMPT = (
        "You are SAARTHI, an embodied desktop robot assistant. "
        "[MODE: ROBOT_CONTROL] Analyze sensor data, reason step-by-step, then "
        "output a precise JSON action command."
    )
    CONVERSATIONAL_SYSTEM_PROMPT = SYSTEM_PROMPT

try:
    from ai_services import call_groq as _default_ai_call
except Exception:  # pragma: no cover - ai_services should always be importable
    _default_ai_call = None

# ---------------------------------------------------------------------------
# Semantic memory retrieval (additive, optional; Phase 3C)
# ---------------------------------------------------------------------------
# Imported defensively, same pattern as emotion_engine/personality_engine
# above: if database.py (or its own Supabase/embedding dependencies) is
# ever unavailable, semantic memory silently degrades to "no context"
# rather than breaking the chat pipeline. database.py's own docstring
# confirms it never imports this module (or server.py), so this one-way
# import (logic.py -> database.py) introduces no circular dependency.
try:
    from database import retrieve_relevant_memories as _retrieve_relevant_memories_db
except Exception:  # pragma: no cover - database should always be importable
    _retrieve_relevant_memories_db = None


# ---------------------------------------------------------------------------
# Phase 4: adaptive_learning integration (additive, optional, fail-safe)
# ---------------------------------------------------------------------------
# Imported defensively, the same pattern as emotion_engine/personality_engine
# above: if adaptive_learning (or any of its three factories) is ever
# unavailable, this degrades to "no adaptive adjustment" and
# run_personalized_pipeline behaves exactly as it did before Phase 4 -
# response_guidance is built from response_policy + Personality Engine only.
#
# NOTE: the exact adaptive_learning API for creating a fresh per-session
# learning_state was not specified up front, so _get_or_create_learning_state
# below tries a short list of plausible factory method names on
# _learning_engine and gives up cleanly (returns None) if none of them
# exist. Verify/adjust that helper against your actual adaptive_learning
# package - everywhere else in this integration is defensive enough that a
# wrong guess there just disables Phase 4 for this process, it never
# crashes the chat pipeline (see _apply_adaptive_policy_safe below).
try:
    from adaptive_learning import (
        get_learning_engine,
        get_adaptation_engine,
        get_adaptive_policy,
    )

    _learning_engine = get_learning_engine()
    _adaptation_engine = get_adaptation_engine()
    _adaptive_policy = get_adaptive_policy()
except Exception:  # pragma: no cover - adaptive_learning (Phase 4) is optional
    _learning_engine = None
    _adaptation_engine = None
    _adaptive_policy = None

# Per-session_id registry of adaptive-learning state, mirroring
# _user_profiles/_preference_profiles below (same _registry_lock, cleared
# together in reset_session_state).
_learning_states: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Per-session in-memory state
# ---------------------------------------------------------------------------
# Lightweight, process-local registries keyed by session_id. This mirrors
# the isolation pattern already used by context_memory.py (per-session
# locking, no cross-session leakage). This is NOT a database - it is
# ephemeral personalization state that can be swapped for a persisted
# store later (via user_profile.profile_to_dict/from_dict and
# preferences' raw_scores dict) without changing this module's public
# functions.

_PERSONALIZATION_MAX_TURNS = 12

# Semantic-memory context budget (Phase 3C). Deliberately small and
# independent of database.py's own internal budget constants: this is the
# *second*, module-local bound applied on top of whatever database.py
# already returns, so the AI-facing context can never balloon even if
# database.py's own limits are loosened later. Reserves the vast majority
# of the AI provider's context for the current message, recent short-term
# history, the system prompt, and response/personality guidance.
_SEMANTIC_MEMORY_TOP_K = 3
_MAX_SEMANTIC_MEMORY_ITEM_CHARS = 160
_MAX_SEMANTIC_MEMORY_CONTEXT_CHARS = 500
# Intents that must never trigger semantic retrieval, even though
# server.py's game-message interception should already keep GAME turns
# from reaching this pipeline at all (defense in depth, not the primary
# control - see requirement 15/16 in the integration spec).
_SEMANTIC_MEMORY_SKIP_INTENTS = {"GAME"}

_user_profiles: Dict[str, UserProfile] = {}
_preference_profiles: Dict[str, PreferenceProfile] = {}
_registry_lock = threading.Lock()

conversation_memory = ContextMemoryManager(max_turns=_PERSONALIZATION_MAX_TURNS)


def get_session_state(session_id: str) -> Tuple[UserProfile, PreferenceProfile]:
    """
    Return the (UserProfile, PreferenceProfile) pair for a session,
    creating fresh defaults on first use. Thread-safe; safe to call from
    multiple concurrent sessions without cross-talk.
    """
    if not session_id or not isinstance(session_id, str):
        raise PersonalizationError("session_id must be a non-empty string")

    with _registry_lock:
        user_profile = _user_profiles.get(session_id)
        if user_profile is None:
            user_profile = create_default_profile(session_id)
            _user_profiles[session_id] = user_profile

        preference_profile = _preference_profiles.get(session_id)
        if preference_profile is None:
            preference_profile = get_default_profile()
            _preference_profiles[session_id] = preference_profile

    return user_profile, preference_profile


def reset_session_state(session_id: str) -> None:
    """
    Clear all personalization state for a session (profile, preferences,
    and short-term memory). Safe to call even if the session was never
    tracked. Intended for logout / long-idle cleanup - does not touch
    database.py's persisted chat history in any way.
    """
    with _registry_lock:
        _user_profiles.pop(session_id, None)
        _preference_profiles.pop(session_id, None)
        _learning_states.pop(session_id, None)
    conversation_memory.remove_session(session_id)


def detect_intent(user_message: str) -> Dict[str, Any]:
    """
    Thin convenience wrapper exposing raw, deterministic intent
    detection (no personalization applied). Useful for callers that
    only need routing information without running the full
    personalization pipeline.
    """
    return route_intent(user_message).to_dict()


def _analyze_emotion_safe(user_message: str) -> Optional[Dict[str, Any]]:
    """
    Run the Emotion Engine on a single user turn and return its result as
    a plain dict (already-serialized via EmotionResult.to_dict()).

    Never raises: emotion detection is a "nice to have" tone signal, not
    a critical-path dependency. If the module failed to import, if the
    detector raises for any reason, or if it comes back with weak
    confidence, this gracefully falls back to a neutral reading instead
    of ever bubbling an exception up into the chat pipeline.

    NOTE: this is a conversational-tone signal only - never treated as a
    medical/psychological diagnosis, and its result is never allowed to
    change `personalization_result.intent` (see run_personalized_pipeline,
    which only ever feeds this into PersonalityContext, not into intent
    routing).
    """
    if analyze_emotion is None:
        return None
    try:
        result = analyze_emotion(user_message)
        emotion_dict = result.to_dict()
        # Weak-confidence readings gracefully fall back to neutral behavior
        # (requirement 6): keep the low-confidence data for observability,
        # but don't let a shaky guess drive strong personality swings -
        # PersonalityContext below reads primary_emotion/situation either
        # way, but a low confidence value lets downstream consumers choose
        # to discount it if they want to.
        return emotion_dict
    except Exception as e:
        logger.warning(f"Emotion Engine failed, falling back to neutral: {e}")
        if EmotionResult is not None:
            try:
                return EmotionResult().to_dict()
            except Exception:
                return None
        return None


def _decide_personality_safe(
    emotion_result: Optional[Dict[str, Any]],
    intent: str,
    user_profile: UserProfile,
    preference_score: float,
    conversation_depth: int,
) -> Optional[Dict[str, Any]]:
    """
    Run the Personality Engine for this turn and return its decision as a
    plain dict (via PersonalityDecision.to_dict()).

    Never raises: on any failure (module unavailable, bad input, internal
    error) this returns None, and the caller falls back to using
    response_policy's own tone/length/instructions untouched - exactly
    the pre-existing behavior before this integration.

    Personality is fed the detected emotion/situation, the *routed*
    intent (read-only, informational), and the user's stated style
    preferences - it never decides or overrides intent itself.
    """
    if build_personality is None or PersonalityContext is None:
        return None
    try:
        detected_emotion = ""
        situation = ""
        if emotion_result:
            detected_emotion = str(emotion_result.get("primary_emotion") or "")
            situation = str(emotion_result.get("situation") or "")

        context = PersonalityContext(
            detected_user_emotion=detected_emotion,
            situation=situation,
            intent=intent or "",
            preferred_interaction_style=getattr(user_profile, "preferred_interaction_style", ""),
            preferred_response_length=getattr(user_profile, "preferred_response_length", ""),
            preference_strength=preference_score,
            conversation_depth=conversation_depth,
        )
        decision = build_personality(context)
        return decision.to_dict()
    except Exception as e:
        logger.warning(f"Personality Engine failed, falling back to response_policy defaults: {e}")
        return None


_MAX_GUIDANCE_NOTES = 8
_MAX_GUIDANCE_NOTE_LENGTH = 200


def _combine_response_guidance(
    policy_dict: Dict[str, Any],
    personality_decision: Optional[Dict[str, Any]],
    emotion_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge response_policy's structured policy with the Personality
    Engine's style decision into a single, bounded internal
    response-guidance structure to hand to the AI service.

    response_policy remains the authoritative policy layer: every field
    it already produces (response_length, explanation_depth, tone,
    educational_emphasis, example_usage, follow_up_behavior,
    requires_current_information, recommendation_behavior) is passed
    through unchanged. This function only ADDS concise, plain-language
    style instructions on top - it never rewrites or duplicates
    response_policy's own routing/content decisions.

    Only short, human-readable style instructions are added (e.g. "Use a
    congratulatory tone.", "Be supportive.", "Avoid humor."). No raw
    emotion/personality scores, confidence values, or internal scoring
    math are ever included here - those stay out of what's sent to the
    AI provider, matching response_policy's own "notes" contract in
    ai_services.py.
    """
    combined: Dict[str, Any] = dict(policy_dict) if policy_dict else {}
    notes: List[str] = list(combined.get("notes") or [])

    if personality_decision:
        for instruction in personality_decision.get("instructions") or []:
            text = str(instruction).strip()
            if text and text not in notes:
                notes.append(text)

        # A short, explicit tone hint derived from personality (in addition
        # to - never replacing - response_policy's own `tone` field), since
        # personality's tone is situation-aware (e.g. CONGRATULATORY,
        # SUPPORTIVE) in a way the base policy tone is not.
        personality_tone = personality_decision.get("tone")
        if personality_tone:
            tone_note = f"Emotional tone: {str(personality_tone).replace('_', ' ').lower()}."
            if tone_note not in notes:
                notes.append(tone_note)

    if emotion_result:
        # Situation-specific safety net: never let the reply sound
        # disappointed in the user after a failure/setback, even if this
        # note wasn't already produced by the Personality Engine (e.g.
        # personality_engine failed and this is the only signal left).
        situation = str(emotion_result.get("situation") or "").upper()
        if situation == "FAILURE":
            note = "Do not sound disappointed in the user; focus on recovery and next steps."
            if note not in notes:
                notes.append(note)

    # Bound the final note count/length so this can never balloon the
    # payload sent to the AI provider, mirroring ai_services.py's own
    # _MAX_POLICY_NOTES/_MAX_POLICY_NOTE_LENGTH caps.
    bounded_notes = [n[:_MAX_GUIDANCE_NOTE_LENGTH] for n in notes[:_MAX_GUIDANCE_NOTES]]
    combined["notes"] = bounded_notes
    return combined


def _build_context(session_id: str) -> ConversationContext:
    """Assemble a ConversationContext snapshot from short-term memory."""
    recent_user_turns = conversation_memory.get_recent_messages(session_id, role="user")
    last_intents = [
        turn["intent"] for turn in recent_user_turns if turn.get("intent")
    ]
    return ConversationContext(
        last_intents=last_intents,
        turns_in_session=conversation_memory.get_turn_count(session_id),
        last_recommended_categories=[],
    )


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_dedup(text: str) -> str:
    """Cheap, local normalization for comparing memory text against
    short-term context (lowercased, whitespace-collapsed). Intentionally
    simple - this only needs to catch near-identical restatements, not do
    semantic dedup (database.py's own ranking pipeline already handles
    dedup within its candidate set)."""
    return _WHITESPACE_RE.sub(" ", str(text or "").strip().lower())


async def _retrieve_semantic_memory_safe(
    user_message: str,
    session_id: str,
    routed_intent: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Best-effort semantic long-term memory retrieval for one turn, via
    database.py's already-implemented ranked pipeline
    (embedding -> candidates -> similarity/recency/importance/confidence
    -> relevance threshold -> conflict resolution -> diversity -> budget).

    Never raises and never blocks the event loop: the underlying call is
    synchronous and potentially slow (network + embedding request), so it
    is offloaded via asyncio.to_thread, exactly as database.py's own
    docstrings instruct callers to do. On any failure - missing/unwired
    database.py, network error, timeout, malformed response - this
    returns an empty list and the caller falls back to running without
    semantic context, matching the rest of this module's "optional
    enrichment" contract.

    Skipped up front (no retrieval attempted at all) for GAME-routed
    turns and for missing message/session_id, matching requirements
    4/5/15 of the integration spec: no unnecessary searches, no retrieval
    without a valid session scope, no interference with game isolation.
    """
    if _retrieve_relevant_memories_db is None:
        return []
    if not user_message or not isinstance(user_message, str) or not user_message.strip():
        return []
    if not session_id or not isinstance(session_id, str):
        return []
    if routed_intent and str(routed_intent).upper() in _SEMANTIC_MEMORY_SKIP_INTENTS:
        return []

    try:
        memories = await asyncio.to_thread(
            _retrieve_relevant_memories_db,
            user_message,
            session_id,
            _SEMANTIC_MEMORY_TOP_K,
        )
        return memories if isinstance(memories, list) else []
    except Exception as e:
        # Deliberately does not log user_message or memory content -
        # only the failure itself (requirement 20: no private retrieved
        # memories or full prompt content in logs).
        logger.warning(
            f"Semantic memory retrieval failed for session_id={session_id}, "
            f"continuing without it: {e}"
        )
        return []


def _format_semantic_memory_context(
    memories: List[Dict[str, Any]],
    recent_short_term_texts: List[str],
) -> Optional[str]:
    """
    Turn ranked semantic memories (already relevance-filtered, conflict-
    resolved, and diversified by database.py) into a small, bounded,
    plainly-labeled context block for the AI provider - or None if there
    is nothing usable, which is a valid and expected outcome (requirement
    12: never fabricate memory when retrieval returns nothing relevant).

    - Deduplicates against what's already present in short-term context
      (requirement 11) so the same fact is never injected twice.
    - Caps items and total characters (requirement 10) well below what
      database.py itself would allow, since this budget must also leave
      room for the current message, short-term history, the system
      prompt, and response/personality guidance.
    - Never includes similarity/recency/importance/confidence scores,
      memory_type, category, or source - only the plain content
      (requirement 7: no internal ranking scores exposed to the AI).
    - Explicitly frames the content as non-authoritative supporting
      evidence that the current explicit request always overrides
      (requirements 7/8/9), instead of instructing the AI to obey it.
    """
    if not memories:
        return None

    seen = {_normalize_for_dedup(t) for t in recent_short_term_texts if t}
    chunks: List[str] = []
    total_len = 0

    for memory in memories[:_SEMANTIC_MEMORY_TOP_K]:
        if not isinstance(memory, dict):
            continue
        content = str(memory.get("content") or "").strip()
        if not content:
            continue

        key = _normalize_for_dedup(content)
        if key in seen:
            continue
        seen.add(key)

        snippet = content[:_MAX_SEMANTIC_MEMORY_ITEM_CHARS]
        remaining_budget = _MAX_SEMANTIC_MEMORY_CONTEXT_CHARS - total_len
        if remaining_budget <= 20:
            break
        snippet = snippet[:remaining_budget]

        chunks.append(f"- {snippet}")
        total_len += len(snippet)
        if total_len >= _MAX_SEMANTIC_MEMORY_CONTEXT_CHARS:
            break

    if not chunks:
        return None

    return (
        "Relevant past user context (supporting evidence only, may be "
        "outdated or only loosely related - if it conflicts with the "
        "user's current explicit request, always follow the current "
        "request instead):\n" + "\n".join(chunks)
    )


def _merge_semantic_context_into_messages(
    messages: List[dict],
    semantic_context_text: Optional[str],
) -> List[dict]:
    """
    Fold the bounded semantic-memory context block into the outgoing
    message list's system message (creating one if none exists), without
    ever mutating the caller-supplied list in place - mirrors
    ai_services._apply_policy_to_messages's own merge behavior so the
    two additive enrichments (response policy, semantic memory) compose
    safely regardless of call order. A no-op (returns the same list
    object) when there is no semantic context to add, so callers that
    never get a hit see zero behavior change.
    """
    if not semantic_context_text:
        return messages

    new_messages = list(messages)
    if new_messages and isinstance(new_messages[0], dict) and new_messages[0].get("role") == "system":
        merged = dict(new_messages[0])
        existing_content = merged.get("content", "") or ""
        merged["content"] = f"{existing_content}\n\n{semantic_context_text}".strip()
        new_messages[0] = merged
    else:
        new_messages.insert(0, {"role": "system", "content": semantic_context_text})
    return new_messages


def _get_or_create_learning_state(session_id: str):
    """
    Return this session's adaptive-learning state, creating one on first
    use via whichever factory method _learning_engine actually exposes.
    Returns None if adaptive_learning (Phase 4) isn't available, or if
    state creation fails for any reason - callers must treat None as
    "adaptive learning unavailable this turn" and fall back gracefully,
    never raise.
    """
    if _learning_engine is None:
        return None

    with _registry_lock:
        state = _learning_states.get(session_id)
        if state is not None:
            return state

        state = None
        for factory_name in ("get_or_create_state", "create_state", "new_state"):
            factory = getattr(_learning_engine, factory_name, None)
            if not callable(factory):
                continue
            try:
                state = factory(session_id)
            except TypeError:
                try:
                    state = factory()
                except Exception:
                    state = None
            except Exception:
                state = None
            if state is not None:
                break

        _learning_states[session_id] = state
        return state


def _apply_adaptive_policy_safe(
    policy,
    session_id: str,
    intent: str,
) -> Dict[str, Any]:
    """
    Best-effort Phase 4 adaptive-learning pass over a freshly-built
    response_policy. Runs adaptation_engine.process(learning_state) to
    get adaptation decisions, then adaptive_policy.build(policy,
    adaptation_decisions=decisions, intent=intent,
    is_game_context=(intent == "GAME")) to fold them into a new policy
    dict.

    Never raises: on any failure (module unavailable, no learning_state,
    adaptation_engine/adaptive_policy raising, or an unexpected return
    type), this returns policy.to_dict() unchanged - the exact base
    policy run_personalized_pipeline would have used before Phase 4 was
    wired in.
    """
    base_dict = policy.to_dict()
    if _adaptation_engine is None or _adaptive_policy is None:
        return base_dict

    try:
        learning_state = _get_or_create_learning_state(session_id)
        if learning_state is None:
            return base_dict

        decisions = _adaptation_engine.process(learning_state)
        adapted = _adaptive_policy.build(
            policy,
            adaptation_decisions=decisions,
            intent=intent,
            is_game_context=(intent == "GAME"),
        )
        return adapted if isinstance(adapted, dict) else base_dict
    except Exception as e:
        logger.warning(
            f"Adaptive learning (Phase 4) failed for session_id={session_id}, "
            f"falling back to base response policy: {e}"
        )
        return base_dict


# ---------------------------------------------------------------------------
# Robot sensor formatting + multimodal/document message helpers (additive)
# ---------------------------------------------------------------------------

def format_robot_sensor_prompt(
    sensor_data: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None,
) -> str:
    """
    Format a robot's sensor readings into the "[SENSOR: key = value]"
    block ROBOT_CONTROL_SYSTEM_PROMPT expects, e.g.:

        [SENSOR: camera_vla = USER_DETECTED (est. 60cm)]

        [SENSOR: tof_obstacle_sensor = MEASURING (58cm)]

        User: <user_message or 'None'>

    Each "[SENSOR: ...]" line (and the final "User: ..." line) is
    separated from the next by a blank line (\\n\\n).

    Idempotent: if `sensor_data` is empty/None and `user_message` already
    looks like it was pre-formatted this way (starts with "[SENSOR:"),
    `user_message` is returned unchanged instead of being wrapped again.

    Pure function: no I/O, no session/global state - safe to call from
    anywhere, including concurrently across sessions.
    """
    if not sensor_data and isinstance(user_message, str) and user_message.lstrip().startswith("[SENSOR:"):
        return user_message

    parts = [f"[SENSOR: {key} = {value}]" for key, value in (sensor_data or {}).items()]
    parts.append(f"User: {user_message if user_message else 'None'}")
    return "\n\n".join(parts)


def _set_system_prompt(messages: List[dict], system_prompt: str) -> List[dict]:
    """
    Return a new messages list with the system message's content
    replaced by `system_prompt` (or one inserted at the front if none
    exists). Never mutates the caller's list. Used to switch between
    ROBOT_CONTROL_SYSTEM_PROMPT and whatever the caller's own
    build_messages_fn / _default_build_messages already put there, so
    style guidance never leaks into the robot's strict [ACTION] JSON
    output.
    """
    new_messages = list(messages)
    if new_messages and isinstance(new_messages[0], dict) and new_messages[0].get("role") == "system":
        replaced = dict(new_messages[0])
        replaced["content"] = system_prompt
        new_messages[0] = replaced
    else:
        new_messages.insert(0, {"role": "system", "content": system_prompt})
    return new_messages


def _merge_document_context_into_messages(
    messages: List[dict],
    document_text: Optional[str],
) -> List[dict]:
    """
    Fold an attached document's extracted text into the outgoing message
    list's system message (creating one if none exists), labeled clearly
    as "Attached document context:" so it's never confused with the
    user's own words. Mirrors _merge_semantic_context_into_messages's own
    merge behavior (append, never overwrite) so document context, semantic
    memory, and mode-specific system prompts all compose safely regardless
    of call order. Never mutates the caller's list; a no-op when there is
    no document_text.

    NOTE: media_processor.process_document already truncates document_text
    to DOC_MAX_EXTRACTED_CHARS before it ever reaches this function, so no
    additional length capping is applied here.
    """
    if not document_text or not str(document_text).strip():
        return messages

    block = f"Attached document context:\n{str(document_text).strip()}"
    new_messages = list(messages)
    if new_messages and isinstance(new_messages[0], dict) and new_messages[0].get("role") == "system":
        merged = dict(new_messages[0])
        existing_content = merged.get("content", "") or ""
        merged["content"] = f"{existing_content}\n\n{block}".strip()
        new_messages[0] = merged
    else:
        new_messages.insert(0, {"role": "system", "content": block})
    return new_messages


def _default_build_messages(session_id: str, user_message: str) -> List[dict]:
    """
    Minimal, dependency-free message builder used only when the caller
    doesn't supply its own (e.g. database.py's build_groq_messages,
    which this module intentionally does not import to avoid a
    circular dependency). Uses only this module's own short-term
    memory, never database.py's persisted long-term memory.
    """
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in conversation_memory.get_recent_messages(session_id):
        role = "assistant" if turn["role"] == "assistant" else "user"
        messages.append({"role": role, "content": turn["content"]})

    messages.append({"role": "user", "content": user_message})
    return messages


async def _invoke_ai(
    ai_call_fn: Callable[..., Any],
    messages: List[dict],
    response_policy: Optional[Dict[str, Any]],
    images: Optional[List[Any]] = None,
    video_frames: Optional[List[Any]] = None,
    mode: str = "conversational",
) -> str:
    """
    Call the AI provider function, passing the structured response
    policy, media attachments, and mode if the callable supports them.
    Falls back gracefully through progressively simpler signatures for
    any caller-supplied ai_call_fn that hasn't been upgraded to accept
    these newer keywords - so this still works with the original
    `ai_call_fn(messages, response_policy=...)` contract, and even with
    a bare `ai_call_fn(messages)` callable.
    """
    try:
        return await ai_call_fn(
            messages,
            response_policy=response_policy,
            images=images,
            video_frames=video_frames,
            mode=mode,
        )
    except TypeError:
        try:
            return await ai_call_fn(messages, response_policy=response_policy)
        except TypeError:
            return await ai_call_fn(messages)


async def run_personalized_pipeline(
    user_message: str,
    session_id: str,
    build_messages_fn: Optional[Callable[[str, str], List[dict]]] = None,
    ai_call_fn: Optional[Callable[..., Any]] = None,
    images: Optional[List[Any]] = None,
    video_frames: Optional[List[Any]] = None,
    document_text: Optional[str] = None,
    sensor_data: Optional[Dict[str, Any]] = None,
    mode: str = "conversational",
) -> Optional[Dict[str, Any]]:
    """
    New pipeline (additive; does not replace anything):

        message
          -> intent detection            (intent_router)
          -> user profile/context lookup (user_profile, context_memory)
          -> semantic memory retrieval    (database.retrieve_relevant_memories,
                                            optional/best-effort, Phase 3C)
          -> personalization              (personalization)
          -> response policy              (response_policy)
          -> emotion analysis             (emotion_engine)
          -> personality decision         (personality_engine)
          -> combined response guidance   (response_policy + personality
                                            instructions, bounded)
          -> existing AI service          (ai_services.call_groq), with
                                            bounded semantic-memory context
                                            folded into the outgoing
                                            messages as supporting evidence
          -> response
          -> update context/profile       (context_memory, user_profile,
                                            preferences - via personalize())

    Semantic memory retrieval never changes what category a message is
    routed as, never runs for GAME-routed turns, and is entirely optional:
    on any failure (import, network, embedding, timeout) the pipeline
    continues exactly as it did before this integration, with no semantic
    context folded in.

    Explicit user intent (from intent_router) always takes priority:
    this function never lets preferences, emotion, or personality change
    what category a message is routed as - only how the eventual reply
    is styled.

    This is intentionally NOT wired into server.py's existing
    `_process_chat_message` flow automatically, to guarantee zero
    behavior change for current deployments. server.py may opt in to
    calling this instead of (or before) its current Groq call path.

    Parameters
    ----------
    user_message: raw user text (already stripped/validated by caller,
        same contract as handle_local_queries).
    session_id: the session/user identifier; state is fully isolated
        per session_id, consistent with context_memory.py's guarantees.
    build_messages_fn: optional callable(session_id, user_message) ->
        list[dict] for building the AI provider's message list (e.g.
        database.py's build_groq_messages, wired in by the caller to
        avoid this module importing database.py directly). Defaults to
        a minimal internal builder using only short-term memory.
    ai_call_fn: optional async callable(messages, response_policy=...)
        -> str. Defaults to ai_services.call_groq. Any exception raised
        propagates to the caller so existing error handling in server.py
        (e.g. its try/except around call_groq) continues to apply
        unchanged.
    images: optional list of raw bytes / base64 strings / PIL.Image
        objects to attach (validated/compressed by media_processor.py
        inside ai_services.call_groq before reaching the model).
    video_frames: optional list of video frames, same accepted types as
        `images`.
    document_text: optional extracted document text (already truncated
        by media_processor.process_document) folded into the outgoing
        system message as "Attached document context:\\n...".
    sensor_data: optional dict of robot sensor readings (e.g.
        {"tof_obstacle_sensor": "MEASURING (58cm)", ...}). Supplying this
        - or passing mode="robot_control" - switches this call into
        robot-control mode: `user_message` is run through
        format_robot_sensor_prompt(sensor_data, user_message) before
        anything else, the outgoing system prompt becomes
        ROBOT_CONTROL_SYSTEM_PROMPT (replacing whatever build_messages_fn
        / the default builder would otherwise use), and no
        response_guidance (style/tone instructions) is sent to the model,
        so nothing interferes with its strict [ACTION] JSON output.
    mode: "conversational" (default) or "robot_control". Backward
        compatible: existing callers that never pass sensor_data and
        never set mode see byte-for-byte identical behavior to before
        this parameter existed.

    Returns
    -------
    A dict with intent, confidence, response_policy, personalization
    instructions, emotion/personality signals, the combined response
    guidance, a bounded semantic-memory summary, and the raw AI reply -
    or None if the message was empty/invalid, so callers can fall back
    to their existing empty-message handling exactly as before. All
    original keys from before this integration are unchanged; `emotion`,
    `personality`, `response_guidance`, and `semantic_memory` are new,
    additive keys.
    """
    if not user_message and not sensor_data:
        return None

    is_robot_mode = (mode == "robot_control") or bool(sensor_data)
    effective_user_message = (
        format_robot_sensor_prompt(sensor_data, user_message) if is_robot_mode else user_message
    )

    if not effective_user_message or not str(effective_user_message).strip():
        return None

    ai_call_fn = ai_call_fn or _default_ai_call
    if ai_call_fn is None:
        raise PersonalizationError(
            "No AI call function available: ai_services.call_groq could not "
            "be imported and no ai_call_fn was supplied."
        )

    try:
        user_profile, preference_profile = get_session_state(session_id)
        context = _build_context(session_id)

        # --- Semantic Memory Retrieval (Phase 3C, additive) -------------
        # Read-only, best-effort enrichment sourced from database.py's
        # own ranked pipeline. Runs after user-profile lookup and before
        # personalization, matching the target architecture, but its
        # result is only ever folded into the outgoing AI messages later
        # - it is never passed into personalize(), build_response_policy(),
        # analyze_emotion(), or build_personality(), so it can never
        # influence routing, policy, emotion, or personality decisions.
        routed_intent_probe = route_intent(effective_user_message).intent
        semantic_memories = await _retrieve_semantic_memory_safe(
            effective_user_message, session_id, routed_intent_probe
        )
        recent_short_term_texts = [
            turn.get("content", "")
            for turn in conversation_memory.get_recent_messages(session_id)
        ]
        semantic_context_text = _format_semantic_memory_context(
            semantic_memories, recent_short_term_texts
        )

        personalization_result = personalize(
            message=effective_user_message,
            user_profile=user_profile,
            preference_profile=preference_profile,
            context=context,
        )

        policy_context = PolicyContext(
            turns_in_session=context.turns_in_session,
            category_lockin_detected=bool(
                personalization_result.response_profile.get("category_lockin_detected")
            ),
            recent_intents=context.last_intents,
        )

        policy = build_response_policy(
            intent=personalization_result.intent,
            preference_score=personalization_result.preference_score,
            user_profile=user_profile,
            context=policy_context,
        )

        # --- Phase 4: adaptive_learning (additive, optional) ------------
        # Best-effort adjustment of the base response_policy using this
        # session's learning_state + adaptation_engine's decisions. Falls
        # back to policy.to_dict() unchanged on any failure - see
        # _apply_adaptive_policy_safe's own docstring.
        adaptive_policy_dict = _apply_adaptive_policy_safe(
            policy, session_id, personalization_result.intent
        )

        # --- Emotion Engine + Personality Engine -----------------------
        # Additive turn-level signals layered on top of the existing
        # intent/personalization/response_policy result. Both are
        # best-effort: on any failure they fall back to None and the
        # pipeline continues exactly as it did before this integration
        # (response_policy's own tone/length/instructions, unassisted).
        # Neither is ever allowed to change `personalization_result.intent`
        # - it is only ever passed to them read-only, for style context.
        emotion_result = _analyze_emotion_safe(effective_user_message)
        personality_decision = _decide_personality_safe(
            emotion_result=emotion_result,
            intent=personalization_result.intent,
            user_profile=user_profile,
            preference_score=personalization_result.preference_score,
            conversation_depth=context.turns_in_session,
        )

        # Robot-control turns (mode="robot_control" or sensor_data given)
        # never get style/tone guidance folded into the prompt - it would
        # only interfere with the model's strict [ACTION] JSON output.
        response_guidance = (
            None
            if is_robot_mode
            else _combine_response_guidance(adaptive_policy_dict, personality_decision, emotion_result)
        )

        conversation_memory.add_message(
            session_id, effective_user_message, intent=personalization_result.intent
        )

        messages = (
            build_messages_fn(session_id, effective_user_message)
            if build_messages_fn is not None
            else _default_build_messages(session_id, effective_user_message)
        )
        if is_robot_mode:
            messages = _set_system_prompt(messages, ROBOT_CONTROL_SYSTEM_PROMPT)
        messages = _merge_semantic_context_into_messages(messages, semantic_context_text)
        messages = _merge_document_context_into_messages(messages, document_text)

        raw_reply = await _invoke_ai(
            ai_call_fn,
            messages,
            response_guidance,
            images=images,
            video_frames=video_frames,
            mode=mode,
        )

        conversation_memory.add_response(
            session_id, raw_reply, intent=personalization_result.intent
        )

        return {
            "intent": personalization_result.intent,
            "confidence": personalization_result.confidence,
            "signals": personalization_result.signals,
            "preference_score": personalization_result.preference_score,
            "response_profile": personalization_result.response_profile,
            "response_policy": policy.to_dict(),
            "personalization_instructions": personalization_result.personalization_instructions,
            "emotion": emotion_result,
            "personality": personality_decision,
            "response_guidance": response_guidance,
            # Bounded observability only (requirement 20): counts/flags,
            # never the retrieved memory content itself or per-signal
            # scores - those never leave database.py/this function.
            "semantic_memory": {
                "attempted": _retrieve_relevant_memories_db is not None,
                "retrieved_count": len(semantic_memories),
                "used": bool(semantic_context_text),
            },
            "raw_reply": raw_reply,
        }

    except (
        PersonalizationError,
        ResponsePolicyError,
        ProfileValidationError,
        PreferenceValidationError,
    ) as e:
        # Invalid/corrupted personalization state must never crash chat -
        # log and let the caller fall back to its existing non-personalized
        # path (e.g. server.py's current call_groq(messages) flow).
        logger.warning(f"Personalization pipeline failed for session_id={session_id}: {e}")
        return None
