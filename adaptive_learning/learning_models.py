"""
Adaptive Intelligence & Learning — Phase 4 Shared Contract Layer
====================================================================

This module defines the strongly typed data contracts shared by all
Phase 4 internal modules:

    - feedback_engine.py     (4A Feedback)
    - learning_engine.py     (4B Behavior Learning, 4D Memory
                              Reinforcement/Decay)
    - adaptation_engine.py   (4C Preference Adaptation)
    - adaptive_policy.py     (4E Response Optimization,
                              4F Adaptive Decision Layer)

RESPONSIBILITY
--------------
This file is a CONTRACT LAYER ONLY. It defines the shapes that data
takes as it moves between Phase 4 modules. It does not:

    - implement any learning, scoring, or adaptation algorithm,
    - access a database, cache, or any persistence backend,
    - call any AI/ML model or external service,
    - perform orchestration or decide what should happen next,
    - perform any I/O.

The only "logic" present here is structural validation (type/range/shape
checks performed at construction time) and pure, side-effect-free
serialization helpers. Everything else is left to the modules that
consume these contracts.

PROVENANCE
----------
Every learning-relevant model carries enough information for a
downstream system to answer, without guessing:

    - where the signal originated (`source`),
    - whether it was explicit or implicit (`origin`),
    - when it occurred (`occurred_at`, always timezone-aware UTC),
    - how much evidence backs it (`evidence_count`),
    - how confident the system is (`confidence`),
    - how recent/reinforced/decayed it currently is
      (`recency_weight`, `reinforcement_count`, `decay_factor`).

CONTRADICTIONS
---------------
Nothing here overwrites history. `PreferenceEvidence` and
`LearningSignal` are individually immutable, append-only records;
reconciling conflicting evidence (e.g. "likes quiet mode" vs "likes loud
mode") is the responsibility of `learning_engine.py`'s aggregation logic,
which is expected to fold a *sequence* of these records rather than
mutate them in place.

PRIVACY
-------
No contract in this file requires raw conversation text as a mandatory
field. Free-text fields are optional and intended for short, already
redacted/summarized context, not full transcripts. Prefer references
(IDs), categories (enums), and structured metadata.

No contract in this file has a field for secrets. `to_dict()` /
`to_json()` never serialize API keys, tokens, passwords, or credentials
of any kind, because no such fields exist on these models — this is
enforced by construction, not by a scrubbing step.

BACKWARD COMPATIBILITY
------------------------
This module imports only the Python standard library. It does not
import, and must never import, server.py, database.py,
context_memory.py, user_profile.py, emotion_engine.py,
personality_engine.py, or logic.py. This avoids any circular
dependency between Phase 4 and the existing application.

FUTURE ML READINESS
---------------------
Every model here is a plain, introspectable, serializable record
(dataclass + enums + primitive/structured fields). This makes them
directly usable as feature-engineering inputs or training/inference
records for a future statistical or ML-based learning system, without
requiring any redesign of the contracts themselves.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

__all__ = [
    # Enums
    "FeedbackSource",
    "FeedbackOrigin",
    "FeedbackPolarity",
    "FeedbackCategory",
    "EvidenceStrengthTier",
    "AdaptationAction",
    "PolicyRecommendationType",
    # Models
    "FeedbackEvent",
    "FeedbackClassification",
    "LearningSignal",
    "PreferenceEvidence",
    "BehaviorObservation",
    "LearningState",
    "AdaptationDecision",
    "AdaptivePolicyRecommendation",
    # Errors
    "ContractValidationError",
]

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ContractValidationError(ValueError):
    """Raised when a Phase 4 contract is constructed with invalid data."""


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-:.]{1,128}$")

# Reserved field names that must never appear in metadata payloads,
# as a structural guard against accidentally carrying secrets through
# an otherwise-generic metadata dict.
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api_secret",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "passwd",
        "secret",
        "credential",
        "credentials",
        "auth",
        "authorization",
        "private_key",
    }
)


def _require_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.match(value):
        raise ContractValidationError(
            f"{field_name!r} must be a non-empty id string matching "
            f"{_ID_PATTERN.pattern!r}, got {value!r}"
        )
    return value


def _require_utc_datetime(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ContractValidationError(f"{field_name!r} must be a datetime, got {type(value)!r}")
    if value.tzinfo is None:
        raise ContractValidationError(
            f"{field_name!r} must be timezone-aware (UTC); got a naive datetime"
        )
    return value.astimezone(timezone.utc)


def _require_bounded_float(
    value: float,
    field_name: str,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name!r} must be numeric, got {type(value)!r}")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ContractValidationError(f"{field_name!r} must be finite, got {value!r}")
    if not (minimum <= value <= maximum):
        raise ContractValidationError(
            f"{field_name!r} must be within [{minimum}, {maximum}], got {value!r}"
        )
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{field_name!r} must be an int, got {type(value)!r}")
    if value < 0:
        raise ContractValidationError(f"{field_name!r} must be >= 0, got {value!r}")
    return value


def _require_metadata(value: Optional[Mapping[str, Any]], field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name!r} must be a mapping, got {type(value)!r}")
    normalized: dict[str, Any] = {}
    for key, val in value.items():
        if not isinstance(key, str):
            raise ContractValidationError(f"{field_name!r} keys must be strings, got {key!r}")
        if key.lower() in _FORBIDDEN_METADATA_KEYS:
            raise ContractValidationError(
                f"{field_name!r} must not contain secret-like key {key!r}"
            )
        normalized[key] = val
    return normalized


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _to_dict(instance: Any) -> dict[str, Any]:
    """
    Shared serialization boundary: converts a dataclass instance into a
    plain dict suitable for JSON, a relational row, an event-queue
    payload, or a log line.

    - Enums are reduced to their `.value`.
    - datetimes are reduced to ISO-8601 strings (already UTC by
      construction).
    - Nested dataclasses are recursively converted the same way.
    """

    def convert(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.astimezone(timezone.utc).isoformat()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    raw = asdict(instance)
    return {k: convert(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FeedbackSource(str, Enum):
    """Where a piece of feedback originated."""

    USER_UTTERANCE = "user_utterance"
    USER_ACTION = "user_action"
    USER_CORRECTION = "user_correction"
    SENSOR = "sensor"
    SYSTEM_HEURISTIC = "system_heuristic"
    OPERATOR = "operator"
    UNKNOWN = "unknown"


class FeedbackOrigin(str, Enum):
    """Whether feedback was explicitly given or implicitly inferred."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class FeedbackPolarity(str, Enum):
    """Coarse direction of a feedback signal."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class FeedbackCategory(str, Enum):
    """What aspect of behavior the feedback concerns."""

    TONE = "tone"
    PACING = "pacing"
    CONTENT_RELEVANCE = "content_relevance"
    INITIATIVE = "initiative"
    HUMOR = "humor"
    ACCURACY = "accuracy"
    SAFETY = "safety"
    OTHER = "other"


class EvidenceStrengthTier(str, Enum):
    """Coarse bucket describing how much evidence backs a conclusion."""

    ANECDOTAL = "anecdotal"          # 1 observation
    EMERGING = "emerging"            # a few observations
    ESTABLISHED = "established"      # a consistent pattern
    STRONG = "strong"                # large, consistent body of evidence


class AdaptationAction(str, Enum):
    """Kinds of behavioral adjustment a decision may recommend."""

    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    RESET_TO_DEFAULT = "reset_to_default"
    SUPPRESS = "suppress"


class PolicyRecommendationType(str, Enum):
    """Kinds of recommendation the adaptive policy layer may emit."""

    ADJUST_TONE = "adjust_tone"
    ADJUST_PACING = "adjust_pacing"
    ADJUST_INITIATIVE = "adjust_initiative"
    PREFER_CONTENT_CATEGORY = "prefer_content_category"
    AVOID_CONTENT_CATEGORY = "avoid_content_category"
    NO_CHANGE = "no_change"


# ---------------------------------------------------------------------------
# 4A Feedback
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedbackEvent:
    """
    A single, immutable, raw feedback occurrence.

    This is the entry point of the feedback pipeline (4A). It captures
    that *something happened* worth learning from, without yet judging
    what it means — that judgment belongs to `FeedbackClassification`.
    """

    event_id: str
    occurred_at: datetime
    source: FeedbackSource
    origin: FeedbackOrigin
    session_reference: Optional[str] = None
    interaction_reference: Optional[str] = None
    summary: Optional[str] = None  # short, redacted context — never full transcript
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.event_id, "event_id")
        object.__setattr__(self, "occurred_at", _require_utc_datetime(self.occurred_at, "occurred_at"))
        if not isinstance(self.source, FeedbackSource):
            raise ContractValidationError(f"source must be a FeedbackSource, got {self.source!r}")
        if not isinstance(self.origin, FeedbackOrigin):
            raise ContractValidationError(f"origin must be a FeedbackOrigin, got {self.origin!r}")
        if self.session_reference is not None:
            _require_id(self.session_reference, "session_reference")
        if self.interaction_reference is not None:
            _require_id(self.interaction_reference, "interaction_reference")
        if self.summary is not None and len(self.summary) > 512:
            raise ContractValidationError("summary must be at most 512 characters")
        object.__setattr__(self, "metadata", _require_metadata(self.metadata, "metadata"))

    @staticmethod
    def new_id() -> str:
        return _new_id("fbevt")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


@dataclass(frozen=True)
class FeedbackClassification:
    """
    An interpretation of a `FeedbackEvent`: what it means, produced by
    feedback_engine.py. Multiple classifications may reference the same
    event over time (e.g. re-classified after a model update) — this is
    why classification is a separate, additional record rather than a
    mutation of the event.
    """

    classification_id: str
    event_id: str
    classified_at: datetime
    category: FeedbackCategory
    polarity: FeedbackPolarity
    confidence: float
    classifier_reference: str  # identifies which logic/model produced this
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.classification_id, "classification_id")
        _require_id(self.event_id, "event_id")
        object.__setattr__(self, "classified_at", _require_utc_datetime(self.classified_at, "classified_at"))
        if not isinstance(self.category, FeedbackCategory):
            raise ContractValidationError(f"category must be a FeedbackCategory, got {self.category!r}")
        if not isinstance(self.polarity, FeedbackPolarity):
            raise ContractValidationError(f"polarity must be a FeedbackPolarity, got {self.polarity!r}")
        object.__setattr__(self, "confidence", _require_bounded_float(self.confidence, "confidence"))
        _require_id(self.classifier_reference, "classifier_reference")
        object.__setattr__(self, "metadata", _require_metadata(self.metadata, "metadata"))

    @staticmethod
    def new_id() -> str:
        return _new_id("fbcls")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


# ---------------------------------------------------------------------------
# 4B Behavior Learning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LearningSignal:
    """
    A normalized unit of learning input derived from one or more
    feedback classifications. This is what learning_engine.py actually
    consumes — it is deliberately decoupled from raw feedback so that
    future non-feedback signal sources (e.g. sensor-derived signals)
    can produce `LearningSignal` records too.
    """

    signal_id: str
    derived_at: datetime
    category: FeedbackCategory
    polarity: FeedbackPolarity
    strength: float  # bounded [0, 1]: how strongly this signal should move belief
    confidence: float  # bounded [0, 1]: how trustworthy this signal is
    source_classification_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.signal_id, "signal_id")
        object.__setattr__(self, "derived_at", _require_utc_datetime(self.derived_at, "derived_at"))
        if not isinstance(self.category, FeedbackCategory):
            raise ContractValidationError(f"category must be a FeedbackCategory, got {self.category!r}")
        if not isinstance(self.polarity, FeedbackPolarity):
            raise ContractValidationError(f"polarity must be a FeedbackPolarity, got {self.polarity!r}")
        object.__setattr__(self, "strength", _require_bounded_float(self.strength, "strength"))
        object.__setattr__(self, "confidence", _require_bounded_float(self.confidence, "confidence"))
        ids = tuple(self.source_classification_ids)
        for cid in ids:
            _require_id(cid, "source_classification_ids[*]")
        object.__setattr__(self, "source_classification_ids", ids)
        object.__setattr__(self, "metadata", _require_metadata(self.metadata, "metadata"))

    @staticmethod
    def new_id() -> str:
        return _new_id("lsig")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


@dataclass(frozen=True)
class BehaviorObservation:
    """
    A raw observation about robot/user behavior that may or may not
    have accompanying explicit feedback (e.g. "user interrupted mid-
    response", "user re-engaged after silence"). Distinct from
    FeedbackEvent because an observation is descriptive, not yet judged
    to be positive or negative.
    """

    observation_id: str
    observed_at: datetime
    description_code: str  # short stable code, e.g. "user_interrupted"
    session_reference: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.observation_id, "observation_id")
        object.__setattr__(self, "observed_at", _require_utc_datetime(self.observed_at, "observed_at"))
        _require_id(self.description_code, "description_code")
        if self.session_reference is not None:
            _require_id(self.session_reference, "session_reference")
        object.__setattr__(self, "metadata", _require_metadata(self.metadata, "metadata"))

    @staticmethod
    def new_id() -> str:
        return _new_id("bobs")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


# ---------------------------------------------------------------------------
# 4C Preference Adaptation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreferenceEvidence:
    """
    A single, immutable piece of evidence about a user preference.

    Multiple, potentially conflicting `PreferenceEvidence` records may
    exist for the same `preference_key` — this is intentional: history
    is never overwritten. Reconciliation into a current belief happens
    in `LearningState`, produced by learning_engine.py /
    adaptation_engine.py, not by mutating these records.
    """

    evidence_id: str
    preference_key: str  # e.g. "humor_level", "preferred_greeting_style"
    observed_at: datetime
    origin: FeedbackOrigin
    polarity: FeedbackPolarity
    confidence: float
    supporting_signal_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.evidence_id, "evidence_id")
        _require_id(self.preference_key, "preference_key")
        object.__setattr__(self, "observed_at", _require_utc_datetime(self.observed_at, "observed_at"))
        if not isinstance(self.origin, FeedbackOrigin):
            raise ContractValidationError(f"origin must be a FeedbackOrigin, got {self.origin!r}")
        if not isinstance(self.polarity, FeedbackPolarity):
            raise ContractValidationError(f"polarity must be a FeedbackPolarity, got {self.polarity!r}")
        object.__setattr__(self, "confidence", _require_bounded_float(self.confidence, "confidence"))
        ids = tuple(self.supporting_signal_ids)
        for sid in ids:
            _require_id(sid, "supporting_signal_ids[*]")
        object.__setattr__(self, "supporting_signal_ids", ids)
        object.__setattr__(self, "metadata", _require_metadata(self.metadata, "metadata"))

    @staticmethod
    def new_id() -> str:
        return _new_id("pref")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


# ---------------------------------------------------------------------------
# 4D Memory Reinforcement / Decay  (+ aggregate learning state)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LearningState:
    """
    The current, reconciled belief about one preference/behavior
    dimension, derived from some number of underlying evidence/signal
    records. This is the object that carries reinforcement/decay state
    and full provenance summary for a downstream consumer.

    This is a snapshot, not a log: learning_engine.py is expected to
    produce a new `LearningState` each time it reconciles evidence,
    rather than mutating a prior one in place, so history can still be
    reconstructed from a sequence of snapshots if persisted.
    """

    state_id: str
    preference_key: str
    computed_at: datetime
    belief_value: float  # bounded [-1, 1]: reconciled directional belief
    confidence: float  # bounded [0, 1]
    evidence_count: int
    evidence_strength_tier: EvidenceStrengthTier
    reinforcement_count: int = 0
    decay_factor: float = 1.0  # bounded (0, 1]: 1.0 = no decay applied yet
    recency_weight: float = 1.0  # bounded [0, 1]
    has_conflicting_evidence: bool = False
    contributing_evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.state_id, "state_id")
        _require_id(self.preference_key, "preference_key")
        object.__setattr__(self, "computed_at", _require_utc_datetime(self.computed_at, "computed_at"))
        object.__setattr__(
            self, "belief_value", _require_bounded_float(self.belief_value, "belief_value", -1.0, 1.0)
        )
        object.__setattr__(self, "confidence", _require_bounded_float(self.confidence, "confidence"))
        object.__setattr__(
            self, "evidence_count", _require_non_negative_int(self.evidence_count, "evidence_count")
        )
        if not isinstance(self.evidence_strength_tier, EvidenceStrengthTier):
            raise ContractValidationError(
                f"evidence_strength_tier must be an EvidenceStrengthTier, got {self.evidence_strength_tier!r}"
            )
        object.__setattr__(
            self,
            "reinforcement_count",
            _require_non_negative_int(self.reinforcement_count, "reinforcement_count"),
        )
        object.__setattr__(
            self,
            "decay_factor",
            _require_bounded_float(self.decay_factor, "decay_factor", minimum=1e-9, maximum=1.0),
        )
        object.__setattr__(
            self, "recency_weight", _require_bounded_float(self.recency_weight, "recency_weight")
        )
        if not isinstance(self.has_conflicting_evidence, bool):
            raise ContractValidationError("has_conflicting_evidence must be a bool")
        ids = tuple(self.contributing_evidence_ids)
        for eid in ids:
            _require_id(eid, "contributing_evidence_ids[*]")
        object.__setattr__(self, "contributing_evidence_ids", ids)
        object.__setattr__(self, "metadata", _require_metadata(self.metadata, "metadata"))

    @staticmethod
    def new_id() -> str:
        return _new_id("lstate")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


# ---------------------------------------------------------------------------
# 4E / 4F Response Optimization & Adaptive Decision Layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptationDecision:
    """
    A single decision about how to adjust a behavior parameter, produced
    by adaptation_engine.py from one or more `LearningState` snapshots.
    """

    decision_id: str
    decided_at: datetime
    parameter_key: str  # e.g. "tone_warmth", "response_length"
    action: AdaptationAction
    magnitude: float  # bounded [0, 1]: how large the adjustment is
    confidence: float
    based_on_state_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.decision_id, "decision_id")
        object.__setattr__(self, "decided_at", _require_utc_datetime(self.decided_at, "decided_at"))
        _require_id(self.parameter_key, "parameter_key")
        if not isinstance(self.action, AdaptationAction):
            raise ContractValidationError(f"action must be an AdaptationAction, got {self.action!r}")
        object.__setattr__(self, "magnitude", _require_bounded_float(self.magnitude, "magnitude"))
        object.__setattr__(self, "confidence", _require_bounded_float(self.confidence, "confidence"))
        ids = tuple(self.based_on_state_ids)
        for sid in ids:
            _require_id(sid, "based_on_state_ids[*]")
        object.__setattr__(self, "based_on_state_ids", ids)
        object.__setattr__(self, "metadata", _require_metadata(self.metadata, "metadata"))

    @staticmethod
    def new_id() -> str:
        return _new_id("adec")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


@dataclass(frozen=True)
class AdaptivePolicyRecommendation:
    """
    The final, consumable output of the Phase 4 pipeline: a concrete
    recommendation that adaptive_policy.py hands to the response
    generation path. This is intentionally the narrowest, most
    decision-ready contract in the file.
    """

    recommendation_id: str
    recommended_at: datetime
    recommendation_type: PolicyRecommendationType
    target: Optional[str] = None  # e.g. a content category or parameter name
    confidence: float = 0.0
    rationale_decision_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.recommendation_id, "recommendation_id")
        object.__setattr__(
            self, "recommended_at", _require_utc_datetime(self.recommended_at, "recommended_at")
        )
        if not isinstance(self.recommendation_type, PolicyRecommendationType):
            raise ContractValidationError(
                f"recommendation_type must be a PolicyRecommendationType, got {self.recommendation_type!r}"
            )
        if self.target is not None:
            _require_id(self.target, "target")
        object.__setattr__(self, "confidence", _require_bounded_float(self.confidence, "confidence"))
        ids = tuple(self.rationale_decision_ids)
        for did in ids:
            _require_id(did, "rationale_decision_ids[*]")
        object.__setattr__(self, "rationale_decision_ids", ids)
        object.__setattr__(self, "metadata", _require_metadata(self.metadata, "metadata"))

    @staticmethod
    def new_id() -> str:
        return _new_id("prec")

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)
