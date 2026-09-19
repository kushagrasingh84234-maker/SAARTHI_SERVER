"""
Adaptive Intelligence & Learning — Phase 4A Feedback Engine
====================================================================

This module is the FEEDBACK PROCESSING boundary of Phase 4. It converts
raw, untrusted interaction feedback into validated, normalized
`LearningSignal` objects (see `learning_models.py`) for consumption by
`learning_engine.py`.

PIPELINE
--------
    Interaction
        |
        v
    FeedbackEvent              (raw occurrence, validated shape)
        |
        v
    Feedback Engine            (this module)
        |
        v
    FeedbackClassification     (what the event means)
        |
        v
    LearningSignal              (normalized, bounded, deterministic)
        |
        v
    learning_engine.py

RESPONSIBILITY BOUNDARY
-------------------------
This module produces EVIDENCE. It never mutates, and never imports,
`user_profile.py`, `context_memory.py`, `database.py`,
`personality_engine.py`, `emotion_engine.py`, or `response_policy.py`.
A single interaction can therefore never permanently change user
behavior or profile by itself — that requires the downstream learning
and adaptation stages to reconcile evidence over time. This module also
never implements long-term learning, profile mutation, preference
persistence, adaptation decisions, policy mutation, or database writes —
all of that is explicitly out of scope and belongs to `learning_engine.py`
and beyond.

SECURITY MODEL
---------------
All incoming feedback is treated as untrusted input. This module NEVER
accepts a client-provided confidence, learning score, importance,
authority, or system-level flag — such fields, if present on raw input,
are dropped before processing and confidence/strength are always
derived internally from a fixed, auditable rule set that considers:
feedback origin, feedback type, evidence count, provenance quality, and
event validity (staleness). All derived confidence and strength values
are deterministic and bounded to [0, 1].

IDEMPOTENCY
------------
This module has NO hidden global mutable state. Instead, it exposes a
pure `compute_idempotency_key()` function that derives a deterministic
key from event identity fields. Callers (e.g. the eventual persistence
layer) are responsible for storing/checking keys; this module accepts
an optional, caller-supplied collection of already-seen keys and will
report a duplicate rather than emit a second signal for the same
underlying event.

SPAM / REPETITION RESISTANCE
------------------------------
Corroborating evidence (`evidence_count`) strengthens a signal only via
a diminishing-returns curve with a hard ceiling — no amount of
repetition can push strength or confidence past their bounded maximums,
and each additional occurrence contributes less than the last. Separately,
callers may report `burst_occurrences_in_window` — how many times this
same feedback was observed in a short recent window (window ownership
and counting are the caller's responsibility; this module holds no
timers or counters) — which is used to *dampen* strength and to surface
a `repetition_risk` classification ("none" / "elevated" / "high") for
downstream systems to act on. This module never bans, blocks, or
otherwise enforces anything — it only produces evidence and metadata
that make burst/duplicate/suspicious repetition detectable downstream.

CONTRADICTORY FEEDBACK
------------------------
This module never overwrites, merges, or invalidates prior evidence. A
signal that contradicts an earlier one is simply new evidence with its
own id and timestamp; reconciling contradictory evidence over time is
`learning_engine.py`'s job, not this module's.

FAILURE BEHAVIOR
------------------
    - Invalid/malformed required input   -> rejected safely, never raises
                                             out of `process_feedback`.
    - Malformed optional metadata        -> sanitized (offending keys
                                             dropped) rather than failing
                                             the whole event.
    - Unexpected internal error          -> NOT swallowed; this module
                                             does not wrap arbitrary
                                             internal bugs in a broad
                                             `except Exception`, so such
                                             errors remain diagnosable
                                             instead of silently
                                             corrupting learning state.

EXTENSIBILITY
--------------
Classification and strength/confidence derivation are both implemented
behind small strategy interfaces (`FeedbackClassifierStrategy`,
`SignalStrengthModel`). The current deterministic, rule-based
implementations remain the default and the safe baseline. A future
statistical, ML-assisted, or reinforcement-based implementation can be
supplied via optional parameters on `process_feedback` /
`get_feedback_engine` without changing any existing public function's
name or default behavior.

NO EXTERNAL DEPENDENCIES
---------------------------
This module imports only the Python standard library plus
`learning_models.py`. It performs no database, network, filesystem, or
secret access of any kind.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Collection, Dict, Mapping, Optional, Tuple

from .learning_models import (
    ContractValidationError,
    FeedbackCategory,
    FeedbackClassification,
    FeedbackEvent,
    FeedbackOrigin,
    FeedbackPolarity,
    FeedbackSource,
    LearningSignal,
)

__all__ = [
    "ExplicitFeedbackKind",
    "ImplicitFeedbackKind",
    "FeedbackRejectionReason",
    "FeedbackProcessingResult",
    "FeedbackRejected",
    "FeedbackClassifierStrategy",
    "RuleBasedFeedbackClassifier",
    "SignalStrengthModel",
    "RuleBasedSignalStrengthModel",
    "FeedbackEngineConfig",
    "compute_idempotency_key",
    "build_feedback_event",
    "classify_explicit_feedback",
    "classify_implicit_feedback",
    "derive_learning_signal",
    "assess_repetition_risk",
    "process_feedback",
    "get_feedback_engine",
]

__version__ = "0.2.0"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class FeedbackRejected(Exception):
    """
    Raised internally to signal that raw input was safely rejected.

    This is caught within `process_feedback` and turned into a
    `FeedbackProcessingResult(accepted=False, ...)` — it is not expected
    to escape this module under normal operation. It is distinct from
    `ContractValidationError` (a data-shape problem in an already-built
    contract) and from an unexpected internal bug (which is deliberately
    left to propagate).
    """

    def __init__(self, reason: "FeedbackRejectionReason", detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class FeedbackRejectionReason(str, Enum):
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_SOURCE = "invalid_source"
    INVALID_ORIGIN = "invalid_origin"
    INVALID_KIND = "invalid_kind"
    KIND_ORIGIN_MISMATCH = "kind_origin_mismatch"
    INVALID_TIMESTAMP = "invalid_timestamp"
    SUMMARY_TOO_LONG = "summary_too_long"
    DUPLICATE_EVENT = "duplicate_event"
    INVALID_EVIDENCE_COUNT = "invalid_evidence_count"


# ---------------------------------------------------------------------------
# Feedback kind vocabularies
# ---------------------------------------------------------------------------

class ExplicitFeedbackKind(str, Enum):
    """Concepts the caller may report as EXPLICIT feedback."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    SATISFACTION = "satisfaction"
    DISSATISFACTION = "dissatisfaction"
    EXPLICIT_PREFERENCE = "explicit_preference"


class ImplicitFeedbackKind(str, Enum):
    """Concepts the caller may report as IMPLICIT (inferred) feedback."""

    REPEATED_REQUEST = "repeated_request"
    CLARIFICATION_REQUESTED = "clarification_requested"
    CORRECTION_AFTER_RESPONSE = "correction_after_response"
    ABANDONMENT = "abandonment"
    REPEATED_SUCCESSFUL_INTERACTION = "repeated_successful_interaction"
    REPEATED_UNSUCCESSFUL_INTERACTION = "repeated_unsuccessful_interaction"


# Fixed, auditable mapping from explicit kind -> (category, polarity, base
# classification confidence, base signal strength). These base values are
# the ONLY source of strength/confidence for explicit feedback; client
# input can never override them. Explicit feedback always carries
# strictly stronger base trust than any implicit counterpart, preserving
# the EXPLICIT > IMPLICIT distinction end-to-end.
_EXPLICIT_RULES: Dict[ExplicitFeedbackKind, Tuple[FeedbackCategory, FeedbackPolarity, float, float]] = {
    ExplicitFeedbackKind.POSITIVE: (FeedbackCategory.OTHER, FeedbackPolarity.POSITIVE, 0.90, 0.70),
    ExplicitFeedbackKind.NEGATIVE: (FeedbackCategory.OTHER, FeedbackPolarity.NEGATIVE, 0.90, 0.70),
    ExplicitFeedbackKind.CORRECTION: (FeedbackCategory.ACCURACY, FeedbackPolarity.NEGATIVE, 0.95, 0.80),
    ExplicitFeedbackKind.SATISFACTION: (FeedbackCategory.OTHER, FeedbackPolarity.POSITIVE, 0.85, 0.60),
    ExplicitFeedbackKind.DISSATISFACTION: (FeedbackCategory.OTHER, FeedbackPolarity.NEGATIVE, 0.85, 0.60),
    ExplicitFeedbackKind.EXPLICIT_PREFERENCE: (FeedbackCategory.OTHER, FeedbackPolarity.NEUTRAL, 0.95, 0.50),
}

# Same idea for implicit feedback, but with lower base confidence since
# these are inferred rather than stated, and require corroborating
# evidence_count from the caller (e.g. "this is the 3rd repeated
# request") to reach meaningful strength.
_IMPLICIT_RULES: Dict[ImplicitFeedbackKind, Tuple[FeedbackCategory, FeedbackPolarity, float, float]] = {
    ImplicitFeedbackKind.REPEATED_REQUEST: (FeedbackCategory.CONTENT_RELEVANCE, FeedbackPolarity.NEGATIVE, 0.55, 0.35),
    ImplicitFeedbackKind.CLARIFICATION_REQUESTED: (FeedbackCategory.ACCURACY, FeedbackPolarity.NEGATIVE, 0.50, 0.30),
    ImplicitFeedbackKind.CORRECTION_AFTER_RESPONSE: (FeedbackCategory.ACCURACY, FeedbackPolarity.NEGATIVE, 0.65, 0.45),
    ImplicitFeedbackKind.ABANDONMENT: (FeedbackCategory.CONTENT_RELEVANCE, FeedbackPolarity.NEGATIVE, 0.45, 0.30),
    ImplicitFeedbackKind.REPEATED_SUCCESSFUL_INTERACTION: (FeedbackCategory.OTHER, FeedbackPolarity.POSITIVE, 0.55, 0.35),
    ImplicitFeedbackKind.REPEATED_UNSUCCESSFUL_INTERACTION: (FeedbackCategory.OTHER, FeedbackPolarity.NEGATIVE, 0.55, 0.35),
}

_MAX_SUMMARY_LENGTH = 512

# --- Diminishing-returns repetition bonus (replaces a naive linear bump).
# Each additional corroborating occurrence contributes less than the last;
# the bonus asymptotically approaches, but can NEVER reach or exceed,
# `_MAX_REPETITION_BONUS`. This is the "hard ceiling" required for both
# strength and confidence repetition bumps.
_MAX_EVIDENCE_COUNT_BONUS_OCCURRENCES = 50  # occurrences considered at all
_MAX_REPETITION_BONUS = 0.15                # hard ceiling on the bump itself
_REPETITION_SATURATION_CONSTANT = 6.0       # higher = slower to saturate

# --- Burst / suspicious-repetition dampening (spam resistance).
# `burst_occurrences_in_window` is entirely caller-supplied and caller-
# counted (this module has no timers or counters of its own); occurrences
# up to `_BURST_GRACE` are treated as normal corroboration and dampen
# nothing. Beyond that, strength is dampened multiplicatively, and beyond
# `_BURST_HIGH_RISK_THRESHOLD` the repetition is flagged "high" risk.
_BURST_GRACE = 2
_BURST_DAMPENING_STEP = 0.15
_BURST_ELEVATED_RISK_THRESHOLD = 3
_BURST_HIGH_RISK_THRESHOLD = 8

# --- Provenance-quality confidence bonus (small, bounded).
_PROVENANCE_FIELD_BONUS = 0.02
_MAX_PROVENANCE_BONUS = 0.04

# --- Event-validity (staleness) confidence factor. This is the ONLY
# place current time indirectly matters, and only via the event's own
# `occurred_at` vs. when it is being processed — a legitimate use of
# timestamps, not incidental "now"-dependence.
_STALE_EVENT_AGE_DAYS_THRESHOLD = 90.0
_STALE_EVENT_CONFIDENCE_FACTOR = 0.85


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


# ---------------------------------------------------------------------------
# Result contract for this module's orchestration function
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedbackProcessingResult:
    """
    Outcome of `process_feedback`. Exactly one of (`signal`) or
    (`rejection_reason`) is populated, depending on `accepted`.

    `repetition_risk`, when populated, is one of "none" / "elevated" /
    "high" — a downstream-facing summary of how suspicious the reported
    repetition pattern looks, derived purely from caller-supplied
    `evidence_count` / `burst_occurrences_in_window`. It is informational
    only; this module takes no enforcement action based on it.
    """

    accepted: bool
    idempotency_key: str
    event: Optional[FeedbackEvent] = None
    classification: Optional[FeedbackClassification] = None
    signal: Optional[LearningSignal] = None
    rejection_reason: Optional[FeedbackRejectionReason] = None
    rejection_detail: Optional[str] = None
    repetition_risk: Optional[str] = None


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def compute_idempotency_key(
    *,
    source: FeedbackSource,
    origin: FeedbackOrigin,
    kind: str,
    session_reference: Optional[str],
    interaction_reference: Optional[str],
) -> str:
    """
    Deterministically derive an idempotency key for a feedback event.

    Deliberately excludes `occurred_at`, `summary`, and `metadata`: two
    reports of "the same feedback about the same interaction" should
    collide even if timestamps differ slightly or metadata varies,
    since what identifies a piece of feedback is *what happened and
    where*, not incidental details of how it was reported.

    This function is pure and has no side effects; callers own storing
    and checking keys against whatever persistence they use.
    """
    basis = "|".join(
        [
            source.value,
            origin.value,
            kind,
            session_reference or "-",
            interaction_reference or "-",
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage 1: raw input -> FeedbackEvent
# ---------------------------------------------------------------------------

def build_feedback_event(
    *,
    source: str,
    origin: str,
    session_reference: Optional[str] = None,
    interaction_reference: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    occurred_at: Optional[datetime] = None,
) -> FeedbackEvent:
    """
    Validate untrusted raw fields and construct a `FeedbackEvent`.

    Raises `FeedbackRejected` for any safely-rejectable problem. Does
    NOT accept confidence/strength/importance fields at all — they are
    not parameters of this function, by design.
    """
    if not source:
        raise FeedbackRejected(FeedbackRejectionReason.MISSING_REQUIRED_FIELD, "source is required")
    if not origin:
        raise FeedbackRejected(FeedbackRejectionReason.MISSING_REQUIRED_FIELD, "origin is required")

    try:
        source_enum = FeedbackSource(source)
    except ValueError as exc:
        raise FeedbackRejected(FeedbackRejectionReason.INVALID_SOURCE, str(exc)) from exc

    try:
        origin_enum = FeedbackOrigin(origin)
    except ValueError as exc:
        raise FeedbackRejected(FeedbackRejectionReason.INVALID_ORIGIN, str(exc)) from exc

    if summary is not None and len(summary) > _MAX_SUMMARY_LENGTH:
        raise FeedbackRejected(
            FeedbackRejectionReason.SUMMARY_TOO_LONG,
            f"summary exceeds {_MAX_SUMMARY_LENGTH} characters",
        )

    resolved_occurred_at = occurred_at or datetime.now(timezone.utc)

    safe_metadata = _sanitize_metadata(metadata)

    try:
        return FeedbackEvent(
            event_id=FeedbackEvent.new_id(),
            occurred_at=resolved_occurred_at,
            source=source_enum,
            origin=origin_enum,
            session_reference=session_reference,
            interaction_reference=interaction_reference,
            summary=summary,
            metadata=safe_metadata,
        )
    except ContractValidationError as exc:
        raise FeedbackRejected(FeedbackRejectionReason.INVALID_TIMESTAMP, str(exc)) from exc


def _sanitize_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Best-effort sanitation of optional metadata: malformed metadata is
    ignored/repaired rather than failing the whole event.

    `learning_models` already rejects secret-like keys and non-mapping
    input at construction time; this function additionally drops any
    keys the caller might use to smuggle in trust-sensitive fields
    (confidence, score, importance, authority, system flags), since
    those must only ever be derived internally by this engine.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        return {}

    untrusted_keys = {
        "confidence",
        "learning_score",
        "score",
        "importance",
        "authority",
        "is_system",
        "system",
        "trusted",
        "override",
        "strength",
    }

    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if key.lower() in untrusted_keys:
            continue
        cleaned[key] = value
    return cleaned


# ---------------------------------------------------------------------------
# Stage 2: FeedbackEvent -> FeedbackClassification (pluggable strategy)
# ---------------------------------------------------------------------------

class FeedbackClassifierStrategy(ABC):
    """
    Pluggable classification strategy. `process_feedback` depends only on
    this interface, so a future statistical or ML-assisted classifier can
    be supplied without changing this module's public function names or
    default behavior. The current deterministic rule table
    (`RuleBasedFeedbackClassifier`) remains the default and safe baseline.
    """

    @abstractmethod
    def classify_explicit(
        self, event: FeedbackEvent, kind: ExplicitFeedbackKind
    ) -> FeedbackClassification:
        raise NotImplementedError

    @abstractmethod
    def classify_implicit(
        self, event: FeedbackEvent, kind: ImplicitFeedbackKind
    ) -> FeedbackClassification:
        raise NotImplementedError

    @abstractmethod
    def base_strength(self, kind: str) -> float:
        """Base [0,1] signal strength for a given (explicit or implicit) kind string."""
        raise NotImplementedError


class RuleBasedFeedbackClassifier(FeedbackClassifierStrategy):
    """Deterministic, explainable default classifier implementing the fixed rule tables."""

    def classify_explicit(
        self, event: FeedbackEvent, kind: ExplicitFeedbackKind
    ) -> FeedbackClassification:
        if event.origin is not FeedbackOrigin.EXPLICIT:
            raise FeedbackRejected(
                FeedbackRejectionReason.KIND_ORIGIN_MISMATCH,
                "classify_explicit_feedback requires an EXPLICIT-origin event",
            )
        if not isinstance(kind, ExplicitFeedbackKind):
            raise FeedbackRejected(FeedbackRejectionReason.INVALID_KIND, f"unknown explicit kind {kind!r}")

        category, polarity, base_confidence, _base_strength = _EXPLICIT_RULES[kind]

        return FeedbackClassification(
            classification_id=FeedbackClassification.new_id(),
            event_id=event.event_id,
            classified_at=datetime.now(timezone.utc),
            category=category,
            polarity=polarity,
            confidence=base_confidence,
            classifier_reference="feedback_engine.explicit_rules.v1",
            metadata={"kind": kind.value},
        )

    def classify_implicit(
        self, event: FeedbackEvent, kind: ImplicitFeedbackKind
    ) -> FeedbackClassification:
        if event.origin is not FeedbackOrigin.IMPLICIT:
            raise FeedbackRejected(
                FeedbackRejectionReason.KIND_ORIGIN_MISMATCH,
                "classify_implicit_feedback requires an IMPLICIT-origin event",
            )
        if not isinstance(kind, ImplicitFeedbackKind):
            raise FeedbackRejected(FeedbackRejectionReason.INVALID_KIND, f"unknown implicit kind {kind!r}")

        category, polarity, base_confidence, _base_strength = _IMPLICIT_RULES[kind]

        return FeedbackClassification(
            classification_id=FeedbackClassification.new_id(),
            event_id=event.event_id,
            classified_at=datetime.now(timezone.utc),
            category=category,
            polarity=polarity,
            confidence=base_confidence,
            classifier_reference="feedback_engine.implicit_rules.v1",
            metadata={"kind": kind.value},
        )

    def base_strength(self, kind: str) -> float:
        for enum_cls, table in ((ExplicitFeedbackKind, _EXPLICIT_RULES), (ImplicitFeedbackKind, _IMPLICIT_RULES)):
            try:
                member = enum_cls(kind)
            except ValueError:
                continue
            return table[member][3]
        raise FeedbackRejected(FeedbackRejectionReason.INVALID_KIND, f"unknown feedback kind {kind!r}")


_DEFAULT_CLASSIFIER = RuleBasedFeedbackClassifier()


def classify_explicit_feedback(
    event: FeedbackEvent,
    kind: ExplicitFeedbackKind,
) -> FeedbackClassification:
    """Classify an event known to carry EXPLICIT feedback. Public API — unchanged semantics."""
    return _DEFAULT_CLASSIFIER.classify_explicit(event, kind)


def classify_implicit_feedback(
    event: FeedbackEvent,
    kind: ImplicitFeedbackKind,
) -> FeedbackClassification:
    """Classify an event known to carry IMPLICIT (inferred) feedback. Public API — unchanged semantics."""
    return _DEFAULT_CLASSIFIER.classify_implicit(event, kind)


# ---------------------------------------------------------------------------
# Stage 3: FeedbackClassification -> LearningSignal (pluggable strength model)
# ---------------------------------------------------------------------------

def _repetition_bonus(evidence_count: int) -> float:
    """
    Diminishing-returns bonus for corroborating occurrences. Deterministic,
    bounded in [0, `_MAX_REPETITION_BONUS`), and resistant to both
    single-event overreaction (the first repeat contributes only a small
    fraction of the ceiling) and unlimited repeated-event spam (the curve
    asymptotically approaches, but never reaches, the ceiling).
    """
    bonus_occurrences = max(0, min(evidence_count - 1, _MAX_EVIDENCE_COUNT_BONUS_OCCURRENCES))
    if bonus_occurrences <= 0:
        return 0.0
    return _MAX_REPETITION_BONUS * (
        bonus_occurrences / (bonus_occurrences + _REPETITION_SATURATION_CONSTANT)
    )


def _provenance_quality_bonus(event: FeedbackEvent) -> float:
    """Small, bounded confidence bonus for richer provenance on the event."""
    bonus = 0.0
    if event.session_reference:
        bonus += _PROVENANCE_FIELD_BONUS
    if event.interaction_reference:
        bonus += _PROVENANCE_FIELD_BONUS
    return min(bonus, _MAX_PROVENANCE_BONUS)


def _event_validity_factor(event: FeedbackEvent, evaluated_at: datetime) -> float:
    """
    Mild, deterministic confidence discount for stale events (evidence
    reported long after it occurred is somewhat less trustworthy). Never
    below `_STALE_EVENT_CONFIDENCE_FACTOR`, never a function of "now" in
    an unbounded way — purely `evaluated_at - event.occurred_at`.
    """
    age_days = max(0.0, (evaluated_at - event.occurred_at).total_seconds() / 86400.0)
    if age_days <= _STALE_EVENT_AGE_DAYS_THRESHOLD:
        return 1.0
    return _STALE_EVENT_CONFIDENCE_FACTOR


def assess_repetition_risk(evidence_count: int, burst_occurrences_in_window: int) -> str:
    """
    Deterministic, informational classification of how suspicious a
    repetition pattern looks, for downstream duplicate/burst/spam
    handling. Returns one of "none", "elevated", "high". This function
    takes no enforcement action — it only labels the pattern.
    """
    if burst_occurrences_in_window >= _BURST_HIGH_RISK_THRESHOLD:
        return "high"
    if burst_occurrences_in_window >= _BURST_ELEVATED_RISK_THRESHOLD:
        return "elevated"
    if evidence_count > (_MAX_EVIDENCE_COUNT_BONUS_OCCURRENCES // 2):
        return "elevated"
    return "none"


def _burst_dampening_factor(burst_occurrences_in_window: int) -> float:
    """
    Multiplicative strength dampening once burst repetition exceeds a
    small grace allowance. Bounded to (0, 1]; never amplifies strength.
    """
    excess = max(0, burst_occurrences_in_window - _BURST_GRACE)
    if excess <= 0:
        return 1.0
    return 1.0 / (1.0 + excess * _BURST_DAMPENING_STEP)


class SignalStrengthModel(ABC):
    """
    Pluggable strength/confidence derivation strategy. `derive_learning_signal`
    depends only on this interface, so a future statistical/ML/reinforcement
    model can be swapped in without changing this module's public function
    names or default behavior. The current deterministic rule-based model
    (`RuleBasedSignalStrengthModel`) remains the default and safe baseline.
    """

    @abstractmethod
    def compute(
        self,
        *,
        event: FeedbackEvent,
        classification: FeedbackClassification,
        kind: str,
        evidence_count: int,
        burst_occurrences_in_window: int,
        evaluated_at: datetime,
    ) -> Tuple[float, float]:
        """Return (strength, confidence), each bounded to [0, 1]."""
        raise NotImplementedError


class RuleBasedSignalStrengthModel(SignalStrengthModel):
    """
    Deterministic, explainable default strength/confidence model.

    strength    = base_strength(kind) + diminishing_repetition_bonus,
                  then dampened by burst-repetition suspicion.
    confidence  = classification.confidence + diminishing_repetition_bonus
                  + provenance_quality_bonus, then discounted for stale
                  events.

    Both outputs are clamped to [0, 1] at every stage; neither can ever
    exceed 1.0 regardless of how much evidence or corroboration is
    supplied.
    """

    def __init__(self, classifier: Optional[FeedbackClassifierStrategy] = None):
        self._classifier = classifier or _DEFAULT_CLASSIFIER

    def compute(
        self,
        *,
        event: FeedbackEvent,
        classification: FeedbackClassification,
        kind: str,
        evidence_count: int,
        burst_occurrences_in_window: int,
        evaluated_at: datetime,
    ) -> Tuple[float, float]:
        if classification.category is FeedbackCategory.ACCURACY and kind in (
            ExplicitFeedbackKind.CORRECTION.value,
            ImplicitFeedbackKind.CORRECTION_AFTER_RESPONSE.value,
        ):
            base_strength = (
                _EXPLICIT_RULES[ExplicitFeedbackKind.CORRECTION][3]
                if kind == ExplicitFeedbackKind.CORRECTION.value
                else _IMPLICIT_RULES[ImplicitFeedbackKind.CORRECTION_AFTER_RESPONSE][3]
            )
        else:
            base_strength = self._classifier.base_strength(kind)

        repetition_bonus = _repetition_bonus(evidence_count)
        provenance_bonus = _provenance_quality_bonus(event)
        validity_factor = _event_validity_factor(event, evaluated_at)
        burst_factor = _burst_dampening_factor(burst_occurrences_in_window)

        strength = _clamp(base_strength + repetition_bonus) * burst_factor
        confidence = _clamp(classification.confidence + repetition_bonus + provenance_bonus) * validity_factor

        return _clamp(strength), _clamp(confidence)


_DEFAULT_STRENGTH_MODEL = RuleBasedSignalStrengthModel()


def derive_learning_signal(
    event: FeedbackEvent,
    classification: FeedbackClassification,
    *,
    kind: str,
    evidence_count: int = 1,
    burst_occurrences_in_window: int = 0,
    strength_model: Optional[SignalStrengthModel] = None,
) -> LearningSignal:
    """
    Deterministically derive a bounded `LearningSignal` from a
    classification. Public API — existing positional/keyword contract for
    `event`, `classification`, `kind`, and `evidence_count` is unchanged;
    `burst_occurrences_in_window` and `strength_model` are new, optional,
    backward-compatible parameters.

    `evidence_count` represents how many corroborating occurrences the
    *caller* observed (e.g. "this is the 3rd repeated request in this
    session") — it is clamped and used only to modestly scale strength
    and confidence within their fixed bounds via a diminishing-returns
    curve; it can never push either value outside [0, 1], and it is never
    trusted as a pre-computed strength/confidence value itself.

    `burst_occurrences_in_window` represents how many times this same
    feedback was observed in a short recent window, as counted and owned
    entirely by the caller (this module keeps no timers/counters). It is
    used only to dampen strength and to compute `metadata["repetition_risk"]`
    — never to raise strength above what the base rules and evidence_count
    already allow.
    """
    if evidence_count < 1:
        raise ContractValidationError("evidence_count must be >= 1")
    if burst_occurrences_in_window < 0:
        raise ContractValidationError("burst_occurrences_in_window must be >= 0")

    model = strength_model or _DEFAULT_STRENGTH_MODEL
    evaluated_at = datetime.now(timezone.utc)

    strength, confidence = model.compute(
        event=event,
        classification=classification,
        kind=kind,
        evidence_count=evidence_count,
        burst_occurrences_in_window=burst_occurrences_in_window,
        evaluated_at=evaluated_at,
    )

    repetition_risk = assess_repetition_risk(evidence_count, burst_occurrences_in_window)

    return LearningSignal(
        signal_id=LearningSignal.new_id(),
        derived_at=evaluated_at,
        category=classification.category,
        polarity=classification.polarity,
        strength=strength,
        confidence=confidence,
        source_classification_ids=(classification.classification_id,),
        metadata={
            "kind": kind,
            "event_id": event.event_id,
            "evidence_count": evidence_count,
            "burst_occurrences_in_window": burst_occurrences_in_window,
            "repetition_risk": repetition_risk,
        },
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class FeedbackEngineConfig:
    """
    Optional configuration bundle for a `FeedbackEngineHandle`. Entirely
    optional and additive — omitting it preserves today's default,
    rule-based behavior exactly.
    """
    classifier: Optional[FeedbackClassifierStrategy] = None
    strength_model: Optional[SignalStrengthModel] = None


def process_feedback(
    *,
    source: str,
    origin: str,
    kind: str,
    session_reference: Optional[str] = None,
    interaction_reference: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    occurred_at: Optional[datetime] = None,
    evidence_count: int = 1,
    burst_occurrences_in_window: int = 0,
    already_seen_idempotency_keys: Optional[Collection[str]] = None,
    classifier: Optional[FeedbackClassifierStrategy] = None,
    strength_model: Optional[SignalStrengthModel] = None,
) -> FeedbackProcessingResult:
    """
    Full pipeline entry point: raw, untrusted fields in, a
    `FeedbackProcessingResult` out. Never raises for expected,
    safely-rejectable problems — those are reported via
    `result.accepted is False`. Unexpected internal errors (e.g. a bug
    in this module) are NOT caught here and will propagate, by design.

    `already_seen_idempotency_keys` is optional and caller-owned: this
    function holds no internal memory of prior calls. If the computed
    idempotency key for this event is present in the supplied
    collection, the event is reported as a duplicate and no
    classification/signal is produced.

    `burst_occurrences_in_window`, `classifier`, and `strength_model` are
    new, optional, backward-compatible parameters; omitting all three
    reproduces the engine's original default behavior exactly.
    """
    active_classifier = classifier or _DEFAULT_CLASSIFIER

    idempotency_key = compute_idempotency_key(
        source=source or "",
        origin=origin or "",
        kind=kind or "",
        session_reference=session_reference,
        interaction_reference=interaction_reference,
    ) if _looks_like_valid_enum_value(source, FeedbackSource) and _looks_like_valid_enum_value(origin, FeedbackOrigin) else _fallback_key(
        source, origin, kind, session_reference, interaction_reference
    )

    if already_seen_idempotency_keys is not None and idempotency_key in already_seen_idempotency_keys:
        return FeedbackProcessingResult(
            accepted=False,
            idempotency_key=idempotency_key,
            rejection_reason=FeedbackRejectionReason.DUPLICATE_EVENT,
            rejection_detail="an event with this idempotency key was already processed",
        )

    # --- Guard evidence/burst counts up front so a caller mistake here is
    # a safe rejection, not an uncaught ContractValidationError bubbling
    # out of this orchestration function.
    if not isinstance(evidence_count, int) or evidence_count < 1:
        return FeedbackProcessingResult(
            accepted=False,
            idempotency_key=idempotency_key,
            rejection_reason=FeedbackRejectionReason.INVALID_EVIDENCE_COUNT,
            rejection_detail=f"evidence_count must be an int >= 1, got {evidence_count!r}",
        )
    if not isinstance(burst_occurrences_in_window, int) or burst_occurrences_in_window < 0:
        return FeedbackProcessingResult(
            accepted=False,
            idempotency_key=idempotency_key,
            rejection_reason=FeedbackRejectionReason.INVALID_EVIDENCE_COUNT,
            rejection_detail=(
                f"burst_occurrences_in_window must be an int >= 0, got {burst_occurrences_in_window!r}"
            ),
        )

    try:
        event = build_feedback_event(
            source=source,
            origin=origin,
            session_reference=session_reference,
            interaction_reference=interaction_reference,
            summary=summary,
            metadata=metadata,
            occurred_at=occurred_at,
        )

        if event.origin is FeedbackOrigin.EXPLICIT:
            try:
                explicit_kind = ExplicitFeedbackKind(kind)
            except ValueError as exc:
                raise FeedbackRejected(FeedbackRejectionReason.INVALID_KIND, str(exc)) from exc
            classification = active_classifier.classify_explicit(event, explicit_kind)
        elif event.origin is FeedbackOrigin.IMPLICIT:
            try:
                implicit_kind = ImplicitFeedbackKind(kind)
            except ValueError as exc:
                raise FeedbackRejected(FeedbackRejectionReason.INVALID_KIND, str(exc)) from exc
            classification = active_classifier.classify_implicit(event, implicit_kind)
        else:  # pragma: no cover - FeedbackOrigin is exhaustively EXPLICIT/IMPLICIT today
            raise FeedbackRejected(FeedbackRejectionReason.INVALID_ORIGIN, "unsupported origin")

        signal = derive_learning_signal(
            event,
            classification,
            kind=kind,
            evidence_count=evidence_count,
            burst_occurrences_in_window=burst_occurrences_in_window,
            strength_model=strength_model,
        )

    except FeedbackRejected as rejection:
        return FeedbackProcessingResult(
            accepted=False,
            idempotency_key=idempotency_key,
            rejection_reason=rejection.reason,
            rejection_detail=rejection.detail,
        )

    return FeedbackProcessingResult(
        accepted=True,
        idempotency_key=idempotency_key,
        event=event,
        classification=classification,
        signal=signal,
        repetition_risk=signal.metadata.get("repetition_risk"),
    )


def _looks_like_valid_enum_value(value: str, enum_cls: type) -> bool:
    try:
        enum_cls(value)
        return True
    except ValueError:
        return False


def _fallback_key(
    source: Optional[str],
    origin: Optional[str],
    kind: Optional[str],
    session_reference: Optional[str],
    interaction_reference: Optional[str],
) -> str:
    """
    Idempotency key for input too malformed to build a proper key from
    validated enums. Still deterministic given the same raw strings, so
    repeated identical garbage input still dedupes cleanly.
    """
    basis = "|".join(
        [
            "invalid",
            source or "-",
            origin or "-",
            kind or "-",
            session_reference or "-",
            interaction_reference or "-",
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Package boundary integration
# ---------------------------------------------------------------------------

def get_feedback_engine(config: Optional[FeedbackEngineConfig] = None) -> "FeedbackEngineHandle":
    """
    Public factory expected by `__init__.py`'s lazy component resolution
    (`_COMPONENT_MODULES["feedback_engine"]`). Called with zero arguments
    by that resolver, so `config` is optional and defaults to today's
    rule-based behavior exactly.

    Returns a lightweight, stateless handle exposing this module's pure
    functions as methods, so callers obtained via the package boundary
    don't need to import this module directly.
    """
    return FeedbackEngineHandle(config or FeedbackEngineConfig())


@dataclass(frozen=True)
class FeedbackEngineHandle:
    """Stateless facade over this module's functions."""

    config: FeedbackEngineConfig = FeedbackEngineConfig()

    def process(self, **kwargs: Any) -> FeedbackProcessingResult:
        kwargs.setdefault("classifier", self.config.classifier)
        kwargs.setdefault("strength_model", self.config.strength_model)
        return process_feedback(**kwargs)

    def compute_idempotency_key(self, **kwargs: Any) -> str:
        return compute_idempotency_key(**kwargs)

    def assess_repetition_risk(self, evidence_count: int, burst_occurrences_in_window: int) -> str:
        return assess_repetition_risk(evidence_count, burst_occurrences_in_window)
