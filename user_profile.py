"""
user_profile.py

Lightweight, database-independent UserProfile model for the StudyBot ->
general-purpose Indian AI robot/assistant.

This module models per-user/session personalization state: which intent
categories a user interacts with (and how recently/often), their
preferred response length/style/language/depth, a bounded recent-intent
history, optional interests, and a confidence-aware preference model
that distinguishes EXPLICIT statements from INFERRED behavior and
SYSTEM/DEFAULT values.

It intentionally stores NOTHING sensitive: no passwords, API keys,
tokens, secrets, emails, phone numbers, or other PII. It is a pure
in-memory data model - persistence (database), transport (WebSocket),
semantic/vector memory, and intelligence (AI calls, Fast Math) all live
elsewhere and simply read/write instances of this class.

Phase 3 additions (see module sections below) layer a
production-oriented, explainable personalization model on top of the
original Phase 1/2 profile without removing or renaming any existing
public API. Every new field has a safe default so that old serialized
profiles (missing the new keys entirely) continue to load unchanged,
and every new piece of state is bounded so this module can never grow
an unlimited in-memory log.

Explicit vs. inferred vs. default, in one sentence: EXPLICIT means the
user said it, INFERRED means we noticed a pattern, SYSTEM means another
part of the app set it programmatically, and DEFAULT means nobody has
set it yet. These are never treated as equally certain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = frozenset({
    "EDUCATION",
    "GENERAL",
    "NEWS",
    "INFORMATION",
    "TECHNOLOGY",
    "SHOPPING",
    "BUSINESS",
    "ENTERTAINMENT",
    "PERSONAL",
    "GAME",
    "UNKNOWN",
})

_VALID_RESPONSE_LENGTHS = frozenset({"SHORT", "MEDIUM", "LONG"})
_VALID_INTERACTION_STYLES = frozenset({"FORMAL", "CASUAL", "FRIENDLY", "CONCISE", "PLAYFUL"})

_DEFAULT_RESPONSE_LENGTH = "MEDIUM"
_DEFAULT_INTERACTION_STYLE = "FRIENDLY"
_MAX_RECENT_INTENTS = 25
_MAX_INTERESTS = 50
_MAX_INTEREST_LENGTH = 60

# --- Phase 3: preference source model ---------------------------------------
#
# EXPLICIT   -> the user directly stated this ("I prefer short answers")
# INFERRED   -> learned from repeated behavior, never presented as certain
# SYSTEM     -> set programmatically by another module (not the user's voice)
# DEFAULT    -> nobody has set this yet; distinguishable from a real choice
_SOURCE_DEFAULT = "DEFAULT"
_SOURCE_EXPLICIT = "EXPLICIT"
_SOURCE_INFERRED = "INFERRED"
_SOURCE_SYSTEM = "SYSTEM"
_VALID_PREFERENCE_SOURCES = frozenset({_SOURCE_DEFAULT, _SOURCE_EXPLICIT, _SOURCE_INFERRED, _SOURCE_SYSTEM})

# How much authority each source carries when computing explainable
# "strength" scores. Explicit statements always outweigh weak inferred
# behavior. Kept as a plain constant (not a formula) so it stays
# inspectable/testable.
_SOURCE_EXPLICITNESS_WEIGHT = {
    _SOURCE_EXPLICIT: 1.0,
    _SOURCE_INFERRED: 0.5,
    _SOURCE_SYSTEM: 0.3,
    _SOURCE_DEFAULT: 0.1,
}

# --- Phase 3: bounds for the new preference / affinity stores ---------------
_MAX_PREFERENCES = 100
_MAX_PREFERENCE_KEY_LENGTH = 100
_MAX_PREFERENCE_VALUE_STR_LENGTH = 200

_MAX_TOPIC_AFFINITIES = 100
_MAX_TOPIC_LENGTH = 60

# --- Phase 3: communication profile (unknown != a real preference) ----------
_VALID_EXPLANATION_DEPTHS = frozenset({"BRIEF", "STANDARD", "DEEP"})
_VALID_TECHNICALITY_LEVELS = frozenset({"BEGINNER", "INTERMEDIATE", "ADVANCED"})
_MAX_LANGUAGE_LENGTH = 40

# --- Phase 3: lightweight, aggregate-only feedback signal -------------------
_VALID_FEEDBACK_TYPES = frozenset({"POSITIVE", "NEGATIVE", "ACCEPTED", "CORRECTED"})
_MAX_FEEDBACK_COUNT = 1_000_000  # sanity ceiling; prevents unbounded/overflow-style growth

# --- Phase 3: deterministic, configurable recency/frequency/strength model --
#
# strength = confidence*W_CONFIDENCE + frequency*W_FREQUENCY
#          + recency*W_RECENCY       + explicitness*W_EXPLICITNESS
#
# All four inputs are pre-clamped to [0.0, 1.0], the weights sum to 1.0,
# and the result is clamped to [0.0, 1.0]. Changing personalization
# behavior means changing these constants, not hunting for a magic
# number buried in a function body.
_WEIGHT_CONFIDENCE = 0.25
_WEIGHT_FREQUENCY = 0.30
_WEIGHT_RECENCY = 0.25
_WEIGHT_EXPLICITNESS = 0.20

_RECENCY_HALF_LIFE_DAYS = 7.0   # recency score halves every N days of inactivity
_FREQUENCY_SCALE = 5.0          # saturating scale for frequency scoring

_STRONG_PREFERENCE_THRESHOLD = 0.5
_MAX_STRONG_PREFERENCES = 10
_SUMMARY_TOP_CATEGORIES = 5


class ProfileValidationError(ValueError):
    """Raised when profile data fails validation."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Phase 3: small, explainable value objects
# ---------------------------------------------------------------------------
#
# These are intentionally plain dataclasses (not free-form dicts) so that
# every preference signal in the system carries the same four pieces of
# metadata: value, confidence, source, updated_at. That is what lets us
# tell "the user said they love mathematics" (EXPLICIT, confidence ~1.0)
# apart from "the user asked about mathematics 20 times" (INFERRED,
# confidence derived from behavior) instead of collapsing both into one
# undifferentiated "likes mathematics" flag.

@dataclass
class PreferenceSignal:
    """
    A single confidence-aware preference or topic-affinity signal.

    `value` is intentionally restricted to simple, JSON-safe primitives
    (str/int/float/bool/None) - this is a personalization signal store,
    not a place to stash arbitrary free-form (and potentially sensitive)
    user data.
    """

    value: Any
    confidence: float = 0.5
    source: str = _SOURCE_DEFAULT
    updated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)
        _validate_preference_source(self.source)
        _validate_preference_value(self.value)
        if not isinstance(self.updated_at, str) or not self.updated_at:
            self.updated_at = _utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PreferenceSignal":
        if not isinstance(data, dict):
            raise ProfileValidationError("preference signal data must be a dict")
        return PreferenceSignal(
            value=data.get("value"),
            confidence=data.get("confidence", 0.5),
            source=data.get("source", _SOURCE_DEFAULT),
            updated_at=data.get("updated_at") or _utc_now_iso(),
        )


@dataclass
class CategoryActivity:
    """
    Recency-aware activity metadata for one intent category.

    Deliberately separate from `category_counts` (which only tracks raw
    frequency) so callers can answer "how often" and "how recently /
    how strongly right now" as two distinct questions.
    """

    count: int = 0
    first_seen: str = field(default_factory=_utc_now_iso)
    last_seen: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 0:
            raise ProfileValidationError("CategoryActivity.count must be a non-negative int")
        if not isinstance(self.first_seen, str) or not self.first_seen:
            self.first_seen = _utc_now_iso()
        if not isinstance(self.last_seen, str) or not self.last_seen:
            self.last_seen = _utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "first_seen": self.first_seen, "last_seen": self.last_seen}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CategoryActivity":
        if not isinstance(data, dict):
            raise ProfileValidationError("category activity data must be a dict")
        now = _utc_now_iso()
        return CategoryActivity(
            count=data.get("count", 0),
            first_seen=data.get("first_seen") or now,
            last_seen=data.get("last_seen") or now,
        )


@dataclass
class FeedbackSignals:
    """
    Bounded, aggregate-only record of whether personalization decisions
    have landed well. Counters only - never a per-event log - so this
    can be updated on every interaction indefinitely without growing
    memory usage. Absence of feedback is NOT treated as negative
    feedback anywhere in this module; it simply means no signal yet.
    """

    positive: int = 0
    negative: int = 0
    accepted: int = 0
    corrected: int = 0

    def __post_init__(self) -> None:
        for name in ("positive", "negative", "accepted", "corrected"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ProfileValidationError(f"FeedbackSignals.{name} must be a non-negative int")
            if v > _MAX_FEEDBACK_COUNT:
                setattr(self, name, _MAX_FEEDBACK_COUNT)

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FeedbackSignals":
        if not isinstance(data, dict):
            return FeedbackSignals()
        return FeedbackSignals(
            positive=data.get("positive", 0),
            negative=data.get("negative", 0),
            accepted=data.get("accepted", 0),
            corrected=data.get("corrected", 0),
        )


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """
    Personalization profile for a single user or session.

    Safe to hold multiple independent instances in memory, one per
    active user/session (no global registry lives in this module -
    concurrency/threading is the calling layer's job). Contains no
    secrets and no unnecessary PII by construction.
    """

    # --- Phase 1/2 fields (unchanged; do not rename) ---
    user_id: str
    category_counts: Dict[str, int] = field(default_factory=dict)
    total_interactions: int = 0
    recent_intents: Deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_RECENT_INTENTS))
    preferred_response_length: str = _DEFAULT_RESPONSE_LENGTH
    preferred_interaction_style: str = _DEFAULT_INTERACTION_STYLE
    interests: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    # --- Phase 3 additions (all optional / safely defaulted) ---

    # Generic confidence-aware preferences, e.g. "language" -> English,
    # "response_style" -> concise. Explicit statements and inferred
    # behavior both live here, distinguished by PreferenceSignal.source.
    preferences: Dict[str, PreferenceSignal] = field(default_factory=dict)

    # Topic/subject affinity, distinct from category_counts: this can
    # hold "the user explicitly said they love mathematics" (EXPLICIT,
    # high confidence) separately from raw interaction frequency.
    topic_affinity: Dict[str, PreferenceSignal] = field(default_factory=dict)

    # Recency metadata mirroring category_counts' keys. Bounded by the
    # same fixed category vocabulary, so this can never grow unbounded.
    category_activity: Dict[str, CategoryActivity] = field(default_factory=dict)

    # Lightweight, aggregate-only interaction success signal.
    feedback: FeedbackSignals = field(default_factory=FeedbackSignals)

    # Adaptive communication profile. `None` means "unknown" and must
    # stay distinguishable from any real, resolved preference.
    preferred_language: Optional[str] = None
    explanation_depth: Optional[str] = None
    technicality_preference: Optional[str] = None

    # Bumped only if the on-disk shape changes again in the future;
    # purely informational, never required by callers.
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.user_id or not isinstance(self.user_id, str):
            raise ProfileValidationError("user_id must be a non-empty string")

        if not isinstance(self.recent_intents, deque):
            self.recent_intents = deque(self.recent_intents, maxlen=_MAX_RECENT_INTENTS)
        elif self.recent_intents.maxlen != _MAX_RECENT_INTENTS:
            self.recent_intents = deque(self.recent_intents, maxlen=_MAX_RECENT_INTENTS)

        if self.preferred_response_length not in _VALID_RESPONSE_LENGTHS:
            raise ProfileValidationError(
                f"Invalid preferred_response_length: {self.preferred_response_length}"
            )
        if self.preferred_interaction_style not in _VALID_INTERACTION_STYLES:
            raise ProfileValidationError(
                f"Invalid preferred_interaction_style: {self.preferred_interaction_style}"
            )

        self.interests = self.interests[:_MAX_INTERESTS]

        # --- Phase 3 validation / bounding ---

        if not isinstance(self.preferences, dict):
            raise ProfileValidationError("preferences must be a dict")
        for key, sig in self.preferences.items():
            _validate_preference_key(key)
            if not isinstance(sig, PreferenceSignal):
                raise ProfileValidationError(f"preferences[{key}] must be a PreferenceSignal")
        if len(self.preferences) > _MAX_PREFERENCES:
            # Deterministic trim: keep the most recently updated entries.
            trimmed = sorted(self.preferences.items(), key=lambda kv: kv[1].updated_at, reverse=True)
            self.preferences = dict(trimmed[:_MAX_PREFERENCES])

        if not isinstance(self.topic_affinity, dict):
            raise ProfileValidationError("topic_affinity must be a dict")
        for topic, sig in self.topic_affinity.items():
            _validate_topic(topic)
            if not isinstance(sig, PreferenceSignal):
                raise ProfileValidationError(f"topic_affinity[{topic}] must be a PreferenceSignal")
        if len(self.topic_affinity) > _MAX_TOPIC_AFFINITIES:
            trimmed_topics = sorted(self.topic_affinity.items(), key=lambda kv: kv[1].updated_at, reverse=True)
            self.topic_affinity = dict(trimmed_topics[:_MAX_TOPIC_AFFINITIES])

        if not isinstance(self.category_activity, dict):
            raise ProfileValidationError("category_activity must be a dict")
        for category, activity in self.category_activity.items():
            _validate_category(category)
            if not isinstance(activity, CategoryActivity):
                raise ProfileValidationError(f"category_activity[{category}] must be a CategoryActivity")

        if not isinstance(self.feedback, FeedbackSignals):
            raise ProfileValidationError("feedback must be a FeedbackSignals instance")

        if self.preferred_language is not None:
            if not isinstance(self.preferred_language, str) or not self.preferred_language.strip():
                raise ProfileValidationError("preferred_language must be a non-empty string or None")
            if len(self.preferred_language) > _MAX_LANGUAGE_LENGTH:
                raise ProfileValidationError(
                    f"preferred_language exceeds max length of {_MAX_LANGUAGE_LENGTH} characters"
                )

        if self.explanation_depth is not None and self.explanation_depth not in _VALID_EXPLANATION_DEPTHS:
            raise ProfileValidationError(f"Invalid explanation_depth: {self.explanation_depth}")

        if self.technicality_preference is not None and self.technicality_preference not in _VALID_TECHNICALITY_LEVELS:
            raise ProfileValidationError(f"Invalid technicality_preference: {self.technicality_preference}")

        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            self.schema_version = 2


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def create_default_profile(user_id: str) -> UserProfile:
    """
    Create a fresh UserProfile for a new user/session with safe defaults.
    """
    if not user_id or not isinstance(user_id, str):
        raise ProfileValidationError("user_id must be a non-empty string")

    now = _utc_now_iso()
    return UserProfile(
        user_id=user_id,
        category_counts={},
        total_interactions=0,
        recent_intents=deque(maxlen=_MAX_RECENT_INTENTS),
        preferred_response_length=_DEFAULT_RESPONSE_LENGTH,
        preferred_interaction_style=_DEFAULT_INTERACTION_STYLE,
        interests=[],
        created_at=now,
        updated_at=now,
        preferences={},
        topic_affinity={},
        category_activity={},
        feedback=FeedbackSignals(),
        preferred_language=None,
        explanation_depth=None,
        technicality_preference=None,
        schema_version=2,
    )


# ---------------------------------------------------------------------------
# Category activity
# ---------------------------------------------------------------------------

def update_category_activity(profile: UserProfile, category: str) -> UserProfile:
    """
    Record that the user interacted with a given intent category.

    Increments the category's counter, the total interaction counter,
    updates the timestamp, AND (Phase 3) maintains bounded recency
    metadata for that category in `category_activity`. Does not push to
    recent_intents - use record_recent_intent for that (kept separate
    so callers can choose to track category counts and/or history
    independently, same as before).
    """
    _validate_category(category)

    profile.category_counts[category] = profile.category_counts.get(category, 0) + 1
    profile.total_interactions += 1
    now = _utc_now_iso()

    existing_activity = profile.category_activity.get(category)
    if existing_activity is None:
        profile.category_activity[category] = CategoryActivity(count=1, first_seen=now, last_seen=now)
    else:
        existing_activity.count += 1
        existing_activity.last_seen = now

    profile.updated_at = now
    return profile


def get_category_counts(profile: UserProfile) -> Dict[str, int]:
    """Return a copy of category interaction counts, sorted descending."""
    return dict(
        sorted(profile.category_counts.items(), key=lambda item: item[1], reverse=True)
    )


def get_top_categories(profile: UserProfile, top_n: int = 3) -> List[str]:
    """Return the top N most-interacted-with categories, by count."""
    ranked = sorted(
        profile.category_counts.items(), key=lambda item: item[1], reverse=True
    )
    return [category for category, _ in ranked[:max(0, top_n)]]


def get_category_activity(profile: UserProfile, category: str) -> Dict[str, Any]:
    """
    Return recency-aware activity metadata for one category:
    count, first_seen, last_seen. Returns zeroed/None defaults if the
    category has never been recorded - this never raises for a known,
    valid category with no activity yet.
    """
    _validate_category(category)
    activity = profile.category_activity.get(category)
    if activity is None:
        return {"count": 0, "first_seen": None, "last_seen": None}
    return activity.to_dict()


# ---------------------------------------------------------------------------
# Recent intent history
# ---------------------------------------------------------------------------

def record_recent_intent(profile: UserProfile, intent: str) -> UserProfile:
    """
    Append an intent to the bounded recent-intent history (most recent
    last). Automatically drops the oldest entry once the max length is
    exceeded, so this is safe to call on every interaction indefinitely.
    """
    _validate_category(intent)

    profile.recent_intents.append(intent)
    profile.updated_at = _utc_now_iso()
    return profile


def get_recent_intents(profile: UserProfile, limit: Optional[int] = None) -> List[str]:
    """
    Return recent intents, most recent last. If limit is given, returns
    only the last `limit` entries.
    """
    history = list(profile.recent_intents)
    if limit is not None and limit >= 0:
        return history[-limit:]
    return history


# ---------------------------------------------------------------------------
# Response preferences
# ---------------------------------------------------------------------------

def update_response_preferences(
    profile: UserProfile,
    response_length: Optional[str] = None,
    interaction_style: Optional[str] = None,
) -> UserProfile:
    """
    Update the user's preferred response length and/or interaction
    style. Either argument may be omitted to leave that field
    unchanged.
    """
    if response_length is not None:
        if response_length not in _VALID_RESPONSE_LENGTHS:
            raise ProfileValidationError(
                f"Invalid response_length: {response_length}. "
                f"Must be one of {sorted(_VALID_RESPONSE_LENGTHS)}"
            )
        profile.preferred_response_length = response_length

    if interaction_style is not None:
        if interaction_style not in _VALID_INTERACTION_STYLES:
            raise ProfileValidationError(
                f"Invalid interaction_style: {interaction_style}. "
                f"Must be one of {sorted(_VALID_INTERACTION_STYLES)}"
            )
        profile.preferred_interaction_style = interaction_style

    if response_length is not None or interaction_style is not None:
        profile.updated_at = _utc_now_iso()

    return profile


def get_response_preferences(profile: UserProfile) -> Dict[str, str]:
    """Read the user's current response-shaping preferences."""
    return {
        "preferred_response_length": profile.preferred_response_length,
        "preferred_interaction_style": profile.preferred_interaction_style,
    }


# ---------------------------------------------------------------------------
# Interests (optional, lightweight, non-sensitive tags only)
# ---------------------------------------------------------------------------

def add_interest(profile: UserProfile, interest: str) -> UserProfile:
    """
    Add a lightweight interest tag (e.g. "space", "guitar", "soccer").

    Interests are meant for casual personalization only - short topic
    tags, never free-form sensitive personal data. Enforces a max
    count and max length per tag, and avoids duplicates.
    """
    if not interest or not isinstance(interest, str):
        raise ProfileValidationError("interest must be a non-empty string")

    cleaned = interest.strip()
    if not cleaned:
        raise ProfileValidationError("interest must be a non-empty string")
    if len(cleaned) > _MAX_INTEREST_LENGTH:
        raise ProfileValidationError(
            f"interest exceeds max length of {_MAX_INTEREST_LENGTH} characters"
        )

    normalized_existing = {i.lower() for i in profile.interests}
    if cleaned.lower() in normalized_existing:
        return profile

    if len(profile.interests) >= _MAX_INTERESTS:
        profile.interests.pop(0)

    profile.interests.append(cleaned)
    profile.updated_at = _utc_now_iso()
    return profile


def remove_interest(profile: UserProfile, interest: str) -> UserProfile:
    """Remove an interest tag (case-insensitive match), if present."""
    target = interest.strip().lower()
    filtered = [i for i in profile.interests if i.lower() != target]
    if len(filtered) != len(profile.interests):
        profile.interests = filtered
        profile.updated_at = _utc_now_iso()
    return profile


def get_interests(profile: UserProfile) -> List[str]:
    """Return a copy of the user's interest tags."""
    return list(profile.interests)


# ---------------------------------------------------------------------------
# Phase 3: explicit vs. inferred preferences (confidence + source)
# ---------------------------------------------------------------------------

def set_preference(
    profile: UserProfile,
    key: str,
    value: Any,
    confidence: float = 0.5,
    source: str = _SOURCE_INFERRED,
) -> UserProfile:
    """
    Set (or overwrite) a confidence-aware preference signal, e.g.:

        set_preference(profile, "language", "Hindi", confidence=0.94, source="EXPLICIT")
        set_preference(profile, "response_style", "concise", confidence=0.71, source="INFERRED")

    EXPLICIT statements should generally be given high confidence (the
    user said it directly); INFERRED behavior should reflect genuine
    uncertainty. This function does not itself enforce that EXPLICIT
    always "wins" over INFERRED for the same key - callers that need
    that policy should check `get_preference(...).source` first, since
    that decision belongs to the personalization/response-policy layer,
    not to this data model.
    """
    _validate_preference_key(key)
    signal = PreferenceSignal(value=value, confidence=confidence, source=source, updated_at=_utc_now_iso())

    if key not in profile.preferences and len(profile.preferences) >= _MAX_PREFERENCES:
        # Deterministic eviction: drop the least recently updated entry.
        oldest_key = min(profile.preferences.items(), key=lambda kv: kv[1].updated_at)[0]
        del profile.preferences[oldest_key]

    profile.preferences[key] = signal
    profile.updated_at = signal.updated_at
    return profile


def get_preference(profile: UserProfile, key: str) -> Optional[PreferenceSignal]:
    """Return the PreferenceSignal for `key`, or None if unset."""
    return profile.preferences.get(key)


def get_preferences(profile: UserProfile) -> Dict[str, PreferenceSignal]:
    """Return a shallow copy of all preference signals."""
    return dict(profile.preferences)


# ---------------------------------------------------------------------------
# Phase 3: topic affinity (distinct from raw category_counts frequency)
# ---------------------------------------------------------------------------

def set_topic_affinity(
    profile: UserProfile,
    topic: str,
    confidence: float = 0.5,
    source: str = _SOURCE_INFERRED,
    value: Any = True,
) -> UserProfile:
    """
    Record affinity for a free-text topic/subject, e.g. "mathematics".

    This is deliberately separate from `category_counts`: category
    counts answer "how often did the user ask about X", while topic
    affinity can hold "the user explicitly said they love X" as its own
    confidence-and-source-tagged signal. This module stores only that
    minimal signal - full semantic/vector topic matching belongs in the
    memory/database layer, not here.
    """
    _validate_topic(topic)
    normalized = topic.strip()
    signal = PreferenceSignal(value=value, confidence=confidence, source=source, updated_at=_utc_now_iso())

    if normalized not in profile.topic_affinity and len(profile.topic_affinity) >= _MAX_TOPIC_AFFINITIES:
        oldest_topic = min(profile.topic_affinity.items(), key=lambda kv: kv[1].updated_at)[0]
        del profile.topic_affinity[oldest_topic]

    profile.topic_affinity[normalized] = signal
    profile.updated_at = signal.updated_at
    return profile


def get_topic_affinity(profile: UserProfile, topic: str) -> Optional[PreferenceSignal]:
    """Return the PreferenceSignal for a topic, or None if unset."""
    return profile.topic_affinity.get(topic.strip())


def get_topic_affinities(profile: UserProfile) -> Dict[str, PreferenceSignal]:
    """Return a shallow copy of all topic-affinity signals."""
    return dict(profile.topic_affinity)


# ---------------------------------------------------------------------------
# Phase 3: communication profile (language / depth / technicality)
# ---------------------------------------------------------------------------

def update_communication_profile(
    profile: UserProfile,
    preferred_language: Optional[str] = None,
    explanation_depth: Optional[str] = None,
    technicality_preference: Optional[str] = None,
) -> UserProfile:
    """
    Update adaptive communication characteristics. Any argument left as
    None is treated as "no change" for that field - passing None here
    never resets a field back to unknown. Existing
    preferred_response_length / preferred_interaction_style are
    untouched by this function; use update_response_preferences for
    those, as before.
    """
    changed = False

    if preferred_language is not None:
        if not isinstance(preferred_language, str) or not preferred_language.strip():
            raise ProfileValidationError("preferred_language must be a non-empty string")
        if len(preferred_language) > _MAX_LANGUAGE_LENGTH:
            raise ProfileValidationError(
                f"preferred_language exceeds max length of {_MAX_LANGUAGE_LENGTH} characters"
            )
        profile.preferred_language = preferred_language.strip()
        changed = True

    if explanation_depth is not None:
        if explanation_depth not in _VALID_EXPLANATION_DEPTHS:
            raise ProfileValidationError(f"Invalid explanation_depth: {explanation_depth}")
        profile.explanation_depth = explanation_depth
        changed = True

    if technicality_preference is not None:
        if technicality_preference not in _VALID_TECHNICALITY_LEVELS:
            raise ProfileValidationError(f"Invalid technicality_preference: {technicality_preference}")
        profile.technicality_preference = technicality_preference
        changed = True

    if changed:
        profile.updated_at = _utc_now_iso()
    return profile


def get_communication_profile(profile: UserProfile) -> Dict[str, Optional[str]]:
    """
    Return the full adaptive communication profile, combining the
    original response-length/style fields with the Phase 3 additions.
    Unknown fields are returned as None - never silently guessed.
    """
    return {
        "preferred_response_length": profile.preferred_response_length,
        "preferred_interaction_style": profile.preferred_interaction_style,
        "preferred_language": profile.preferred_language,
        "explanation_depth": profile.explanation_depth,
        "technicality_preference": profile.technicality_preference,
    }


# ---------------------------------------------------------------------------
# Phase 3: lightweight interaction success signal
# ---------------------------------------------------------------------------

def record_interaction_feedback(profile: UserProfile, feedback_type: str) -> UserProfile:
    """
    Record a bounded, aggregate-only feedback signal: POSITIVE,
    NEGATIVE, ACCEPTED (a suggested preference was accepted), or
    CORRECTED (the user corrected a personalization decision). Never
    stores per-event history - only running counters - and absence of
    a call here is never interpreted as negative feedback anywhere in
    this module.
    """
    if feedback_type not in _VALID_FEEDBACK_TYPES:
        raise ProfileValidationError(
            f"Invalid feedback_type: {feedback_type}. Must be one of {sorted(_VALID_FEEDBACK_TYPES)}"
        )

    if feedback_type == "POSITIVE":
        profile.feedback.positive = min(profile.feedback.positive + 1, _MAX_FEEDBACK_COUNT)
    elif feedback_type == "NEGATIVE":
        profile.feedback.negative = min(profile.feedback.negative + 1, _MAX_FEEDBACK_COUNT)
    elif feedback_type == "ACCEPTED":
        profile.feedback.accepted = min(profile.feedback.accepted + 1, _MAX_FEEDBACK_COUNT)
    elif feedback_type == "CORRECTED":
        profile.feedback.corrected = min(profile.feedback.corrected + 1, _MAX_FEEDBACK_COUNT)

    profile.updated_at = _utc_now_iso()
    return profile


def get_feedback_summary(profile: UserProfile) -> Dict[str, int]:
    """
    Return raw feedback counters plus a total. Deliberately does not
    compute a "success rate" or similar derived judgment here - that
    interpretation belongs to a higher layer that knows the full
    context, and a low count is a lack of signal, not evidence of
    failure.
    """
    summary = profile.feedback.to_dict()
    summary["total_signals"] = sum(summary.values())
    return summary


# ---------------------------------------------------------------------------
# Phase 3: deterministic recency / frequency / strength scoring
# ---------------------------------------------------------------------------
#
# All helpers here accept an optional `reference_time` so tests (and any
# other deterministic caller) never depend on wall-clock time.

def compute_recency_score(
    last_seen_iso: Optional[str],
    reference_time: Optional[datetime] = None,
    half_life_days: float = _RECENCY_HALF_LIFE_DAYS,
) -> float:
    """
    Exponential-decay recency score in [0.0, 1.0]. Halves every
    `half_life_days` of inactivity. Returns 0.0 for missing/unparseable
    timestamps rather than raising, since historical records may not
    have this field.
    """
    if not last_seen_iso:
        return 0.0
    try:
        last_seen = datetime.fromisoformat(last_seen_iso)
    except (ValueError, TypeError):
        return 0.0

    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)

    ref = reference_time if reference_time is not None else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (ref - last_seen).total_seconds() / 86400.0)
    if half_life_days <= 0:
        half_life_days = _RECENCY_HALF_LIFE_DAYS

    score = 0.5 ** (age_days / half_life_days)
    return _clamp(score)


def compute_frequency_score(count: int, scale: float = _FREQUENCY_SCALE) -> float:
    """
    Saturating frequency score in [0.0, 1.0): count / (count + scale).
    Never reaches 1.0 exactly, avoiding a false sense of certainty from
    frequency alone.
    """
    if not isinstance(count, (int, float)) or count <= 0:
        return 0.0
    if scale <= 0:
        scale = _FREQUENCY_SCALE
    return _clamp(count / (count + scale))


def compute_explicitness_score(source: str) -> float:
    """Map a preference source to an explicitness weight in [0.0, 1.0]."""
    return _SOURCE_EXPLICITNESS_WEIGHT.get(source, _SOURCE_EXPLICITNESS_WEIGHT[_SOURCE_DEFAULT])


def compute_preference_strength(
    confidence: float = 0.0,
    frequency: float = 0.0,
    recency: float = 0.0,
    explicitness: float = 0.0,
    weights: Optional[Tuple[float, float, float, float]] = None,
) -> float:
    """
    Combine confidence + frequency + recency + explicitness into a
    single bounded, deterministic, explainable strength score in
    [0.0, 1.0]. `weights` is (w_confidence, w_frequency, w_recency,
    w_explicitness); defaults to the module-level constants above so
    behavior can be tuned in one place.
    """
    confidence = _clamp(confidence if isinstance(confidence, (int, float)) and not math.isnan(confidence) else 0.0)
    frequency = _clamp(frequency if isinstance(frequency, (int, float)) and not math.isnan(frequency) else 0.0)
    recency = _clamp(recency if isinstance(recency, (int, float)) and not math.isnan(recency) else 0.0)
    explicitness = _clamp(explicitness if isinstance(explicitness, (int, float)) and not math.isnan(explicitness) else 0.0)

    w_c, w_f, w_r, w_e = weights or (_WEIGHT_CONFIDENCE, _WEIGHT_FREQUENCY, _WEIGHT_RECENCY, _WEIGHT_EXPLICITNESS)

    strength = (confidence * w_c) + (frequency * w_f) + (recency * w_r) + (explicitness * w_e)
    return _clamp(strength)


def get_category_strength(
    profile: UserProfile,
    category: str,
    reference_time: Optional[datetime] = None,
) -> float:
    """
    Bounded, explainable "how strong is this category right now" score,
    combining raw frequency with how recently it was touched. Categories
    have no confidence/source of their own (they're pure behavior), so
    this uses frequency + recency only, via the same shared weighting
    helper for consistency with preference strength.
    """
    _validate_category(category)
    count = profile.category_counts.get(category, 0)
    frequency = compute_frequency_score(count)

    activity = profile.category_activity.get(category)
    recency = compute_recency_score(activity.last_seen if activity else None, reference_time)

    return compute_preference_strength(confidence=0.0, frequency=frequency, recency=recency, explicitness=0.0)


def get_strong_preferences(
    profile: UserProfile,
    threshold: float = _STRONG_PREFERENCE_THRESHOLD,
    top_n: int = _MAX_STRONG_PREFERENCES,
    reference_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Return preferences AND topic affinities whose computed strength
    meets `threshold`, sorted strongest-first and capped at `top_n`.
    Each entry is a small, JSON-safe, explainable dict - never the raw
    internal PreferenceSignal object.
    """
    scored: List[Dict[str, Any]] = []

    for key, sig in profile.preferences.items():
        strength = compute_preference_strength(
            confidence=sig.confidence,
            frequency=0.0,
            recency=compute_recency_score(sig.updated_at, reference_time),
            explicitness=compute_explicitness_score(sig.source),
        )
        if strength >= threshold:
            scored.append({
                "kind": "preference",
                "key": key,
                "value": sig.value,
                "confidence": sig.confidence,
                "source": sig.source,
                "strength": strength,
            })

    for topic, sig in profile.topic_affinity.items():
        strength = compute_preference_strength(
            confidence=sig.confidence,
            frequency=0.0,
            recency=compute_recency_score(sig.updated_at, reference_time),
            explicitness=compute_explicitness_score(sig.source),
        )
        if strength >= threshold:
            scored.append({
                "kind": "topic",
                "key": topic,
                "value": sig.value,
                "confidence": sig.confidence,
                "source": sig.source,
                "strength": strength,
            })

    scored.sort(key=lambda item: item["strength"], reverse=True)
    return scored[:max(0, top_n)]


# ---------------------------------------------------------------------------
# Phase 3: personalization summary (safe for prompt construction)
# ---------------------------------------------------------------------------

def build_personalization_summary(
    profile: UserProfile,
    top_n: int = _SUMMARY_TOP_CATEGORIES,
    reference_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Produce a compact, bounded, JSON-serializable personalization
    summary suitable for feeding into a prompt or response-policy layer.
    Deliberately does NOT dump the entire profile - only the fields
    useful for adapting a response, all already free of secrets/PII by
    construction.
    """
    return {
        "top_categories": get_top_categories(profile, top_n=top_n),
        "strong_preferences": get_strong_preferences(profile, reference_time=reference_time),
        "response_preferences": get_communication_profile(profile),
        "interests": get_interests(profile),
        "confidence": {
            **{f"preference:{k}": v.confidence for k, v in profile.preferences.items()},
            **{f"topic:{k}": v.confidence for k, v in profile.topic_affinity.items()},
        },
        "feedback_summary": get_feedback_summary(profile),
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def profile_to_dict(profile: UserProfile) -> Dict[str, Any]:
    """
    Convert a UserProfile into a plain, JSON-serializable dictionary.

    Contains no secrets or sensitive PII by construction, since none
    are ever stored on the profile in the first place. Phase 3 fields
    are always serialized (with safe/empty defaults for a brand-new
    profile) so a full round-trip preserves everything.
    """
    return {
        "user_id": profile.user_id,
        "category_counts": dict(profile.category_counts),
        "total_interactions": profile.total_interactions,
        "recent_intents": list(profile.recent_intents),
        "preferred_response_length": profile.preferred_response_length,
        "preferred_interaction_style": profile.preferred_interaction_style,
        "interests": list(profile.interests),
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        # Phase 3 additions:
        "preferences": {k: v.to_dict() for k, v in profile.preferences.items()},
        "topic_affinity": {k: v.to_dict() for k, v in profile.topic_affinity.items()},
        "category_activity": {k: v.to_dict() for k, v in profile.category_activity.items()},
        "feedback": profile.feedback.to_dict(),
        "preferred_language": profile.preferred_language,
        "explanation_depth": profile.explanation_depth,
        "technicality_preference": profile.technicality_preference,
        "schema_version": profile.schema_version,
    }


def profile_from_dict(data: Dict[str, Any]) -> UserProfile:
    """
    Restore a UserProfile from a plain dictionary (e.g. loaded from a
    database record or cache by calling code). Performs validation on
    all fields so corrupted or tampered data cannot silently produce
    an inconsistent profile.

    Backward compatible by construction: every Phase 3 key is read with
    `.get(key, <safe default>)`, so pre-Phase-3 records (which simply
    don't have these keys) load exactly as before, just with empty/
    unknown Phase 3 state.
    """
    if not isinstance(data, dict):
        raise ProfileValidationError("data must be a dict")

    user_id = data.get("user_id")
    if not user_id or not isinstance(user_id, str):
        raise ProfileValidationError("data.user_id must be a non-empty string")

    category_counts = data.get("category_counts", {}) or {}
    if not isinstance(category_counts, dict):
        raise ProfileValidationError("data.category_counts must be a dict")
    for category, count in category_counts.items():
        _validate_category(category)
        if not isinstance(count, int) or count < 0:
            raise ProfileValidationError(
                f"category_counts[{category}] must be a non-negative int"
            )

    total_interactions = data.get("total_interactions", 0)
    if not isinstance(total_interactions, int) or total_interactions < 0:
        raise ProfileValidationError("data.total_interactions must be a non-negative int")

    recent_intents_raw = data.get("recent_intents", []) or []
    if not isinstance(recent_intents_raw, list):
        raise ProfileValidationError("data.recent_intents must be a list")
    for intent in recent_intents_raw:
        _validate_category(intent)

    preferred_response_length = data.get("preferred_response_length", _DEFAULT_RESPONSE_LENGTH)
    preferred_interaction_style = data.get("preferred_interaction_style", _DEFAULT_INTERACTION_STYLE)

    interests = data.get("interests", []) or []
    if not isinstance(interests, list) or not all(isinstance(i, str) for i in interests):
        raise ProfileValidationError("data.interests must be a list of strings")

    created_at = data.get("created_at") or _utc_now_iso()
    updated_at = data.get("updated_at") or _utc_now_iso()

    # --- Phase 3 fields: all optional, all safely defaulted ---

    preferences_raw = data.get("preferences", {}) or {}
    if not isinstance(preferences_raw, dict):
        raise ProfileValidationError("data.preferences must be a dict")
    preferences = {}
    for key, sig_data in preferences_raw.items():
        _validate_preference_key(key)
        preferences[key] = PreferenceSignal.from_dict(sig_data)

    topic_affinity_raw = data.get("topic_affinity", {}) or {}
    if not isinstance(topic_affinity_raw, dict):
        raise ProfileValidationError("data.topic_affinity must be a dict")
    topic_affinity = {}
    for topic, sig_data in topic_affinity_raw.items():
        _validate_topic(topic)
        topic_affinity[topic] = PreferenceSignal.from_dict(sig_data)

    category_activity_raw = data.get("category_activity", {}) or {}
    if not isinstance(category_activity_raw, dict):
        raise ProfileValidationError("data.category_activity must be a dict")
    category_activity = {}
    for category, activity_data in category_activity_raw.items():
        _validate_category(category)
        category_activity[category] = CategoryActivity.from_dict(activity_data)

    feedback = FeedbackSignals.from_dict(data.get("feedback", {}) or {})

    preferred_language = data.get("preferred_language")
    explanation_depth = data.get("explanation_depth")
    technicality_preference = data.get("technicality_preference")
    schema_version = data.get("schema_version", 2)
    if not isinstance(schema_version, int):
        schema_version = 2

    profile = UserProfile(
        user_id=user_id,
        category_counts=dict(category_counts),
        total_interactions=total_interactions,
        recent_intents=deque(recent_intents_raw[-_MAX_RECENT_INTENTS:], maxlen=_MAX_RECENT_INTENTS),
        preferred_response_length=preferred_response_length,
        preferred_interaction_style=preferred_interaction_style,
        interests=interests[:_MAX_INTERESTS],
        created_at=created_at,
        updated_at=updated_at,
        preferences=preferences,
        topic_affinity=topic_affinity,
        category_activity=category_activity,
        feedback=feedback,
        preferred_language=preferred_language,
        explanation_depth=explanation_depth,
        technicality_preference=technicality_preference,
        schema_version=schema_version,
    )
    return profile


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_category(category: str) -> None:
    if not isinstance(category, str) or category not in _VALID_CATEGORIES:
        raise ProfileValidationError(f"Invalid category: {category}")


def _validate_confidence(confidence: Any) -> None:
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ProfileValidationError("confidence must be a number")
    if math.isnan(confidence) or math.isinf(confidence):
        raise ProfileValidationError("confidence must be a finite number")
    if not (0.0 <= confidence <= 1.0):
        raise ProfileValidationError("confidence must be between 0.0 and 1.0")


def _validate_preference_source(source: Any) -> None:
    if not isinstance(source, str) or source not in _VALID_PREFERENCE_SOURCES:
        raise ProfileValidationError(
            f"Invalid preference source: {source}. Must be one of {sorted(_VALID_PREFERENCE_SOURCES)}"
        )


def _validate_preference_value(value: Any) -> None:
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ProfileValidationError("preference value must be a finite number")
        return
    if isinstance(value, str):
        if len(value) > _MAX_PREFERENCE_VALUE_STR_LENGTH:
            raise ProfileValidationError(
                f"preference value exceeds max length of {_MAX_PREFERENCE_VALUE_STR_LENGTH} characters"
            )
        return
    raise ProfileValidationError(
        "preference value must be a str, int, float, bool, or None "
        "(no nested/free-form structures are allowed in this store)"
    )


def _validate_preference_key(key: Any) -> None:
    if not isinstance(key, str) or not key.strip():
        raise ProfileValidationError("preference key must be a non-empty string")
    if len(key) > _MAX_PREFERENCE_KEY_LENGTH:
        raise ProfileValidationError(
            f"preference key exceeds max length of {_MAX_PREFERENCE_KEY_LENGTH} characters"
        )


def _validate_topic(topic: Any) -> None:
    if not isinstance(topic, str) or not topic.strip():
        raise ProfileValidationError("topic must be a non-empty string")
    if len(topic) > _MAX_TOPIC_LENGTH:
        raise ProfileValidationError(f"topic exceeds max length of {_MAX_TOPIC_LENGTH} characters")
