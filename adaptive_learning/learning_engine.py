"""
Adaptive Intelligence & Learning / learning_engine.py
======================================================

Role
----
Transforms validated learning evidence into controlled, explainable
`LearningState` that downstream consumers (adaptation_engine.py) can
safely act on.

PRIMARY PIPELINE
-----------------
    feedback_engine.py
            |
            v
    LearningSignal (learning_models.py — the canonical contract)
            |
            v
    learning_engine.py   <-- THIS MODULE
            |
            v
    LearningState
            |
            v
    adaptation_engine.py

This module does NOT:
  - call any LLM or network endpoint
  - touch the database
  - import or mutate user_profile.py, context_memory.py, database.py,
    personality_engine.py, emotion_engine.py, response_policy.py, or
    server.py
  - import feedback_engine.py's internal implementation — the boundary
    with feedback_engine.py is the `learning_models.LearningSignal`
    contract only (plus `FeedbackOrigin`/`FeedbackPolarity`/
    `FeedbackCategory`, all from `learning_models.py`)
  - hold hidden global mutable state

It communicates only through explicit, serializable data contracts.

CONTRACT LAYERING
-------------------
`learning_models.py` is the canonical, cross-module contract layer.
This module's OWN `LearningSignal` / `LearningState` / `DimensionState` /
`EvidenceRecord` classes below are intentionally NOT duplicates of
`learning_models.py`'s same-named classes — they are this engine's
internal working representation (dimension-keyed, decay/stability-aware,
provenance-bearing) that `adaptation_engine.py` already depends on
structurally (by attribute, not by class identity; see that module's own
"Contract note"). The bridge between the two layers is explicit and
one-directional:

    learning_models.LearningSignal  -->  LearningSignal.from_feedback_signal(...)  -->  this engine

`ingest_feedback_signal()` is the single supported entry point for that
bridge; nothing in this module ever assumes feedback_engine.py's raw
kind/enum vocabulary directly.

EVIDENCE HIERARCHY (never reversed)
--------------------------------------
    Explicit user preference
        >
    Repeated strong behavioral evidence
        >
    Weak behavioral inference

A single implicit event can move `net_score` only a small, capped amount
and can raise `confidence` only a small, saturating amount — nowhere
near enough on its own to be mistaken for an established preference (see
`RuleBasedStrategyConfig.max_single_event_delta` and
`max_single_event_confidence_gain`). Explicit/correction evidence is
trusted more per-event, but even it accumulates rather than instantly
overwriting existing belief, keeping the pipeline evidence-driven and
resistant to feedback-loop runaway (response -> inferred reaction ->
adaptation -> changed response -> ... ).

Design summary
---------------
- LearningEngine.process(signal, state) is the single entrypoint.
  It is a pure function of (signal, state) -> (new_state, transition).
  No shared global mutable state is required to call it correctly, and
  no state mutation is ever partially applied: either a full new
  `LearningState` is returned, or the original `state` is returned
  unchanged alongside a rejection/no-op `TransitionRecord`.
- Optional lightweight anti-spam/anti-duplicate bookkeeping is kept in a
  small, thread-safe, per-engine-instance store (NOT global module state),
  scoped by (subject) key. Callers that shard by user should hold one
  LearningEngine instance per user OR rely on the fact that this
  bookkeeping is keyed internally, so a single shared instance is also
  safe across users as long as `subject` keys don't collide across users.
  If subject keys can collide across users, namespace them
  (e.g. f"{user_id}:{subject}") before calling `process`.
- The actual scoring math lives behind a `LearningStrategy` interface so a
  future ML/statistical learner can be swapped in without touching
  LearningEngine's public surface or LearningState's shape.

SECURITY
---------
All `LearningSignal` input is treated as untrusted, regardless of
whether it arrived via `LearningSignal(...)`, `LearningSignal.from_raw(...)`,
or `LearningSignal.from_feedback_signal(...)`. `value` and
`signal_confidence` are validated to be finite, non-NaN real numbers
before anything else happens; a malformed value is rejected safely
(`TransitionOutcome.REJECTED_INVALID`) rather than propagating NaN/Inf
into accumulated state or raising out of `process()`. Every bounded
field (`net_score`, `confidence`, learning-rate multipliers, etc.) is
clamped at every mutation point, so no accumulation path can overflow
regardless of how much or how skewed the input evidence is.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical contract layer (the ONLY sanctioned link to feedback_engine.py's
# output shape). Imported defensively: this module must still be usable
# standalone if learning_models.py is mid-development or briefly broken.
# ---------------------------------------------------------------------------
try:
    from .learning_models import (  # type: ignore
        FeedbackCategory as _FBCategory,
        FeedbackOrigin as _FBOrigin,
        FeedbackPolarity as _FBPolarity,
        LearningSignal as _CanonicalFeedbackSignal,
    )
except Exception:  # pragma: no cover - fallback for standalone use/testing
    class _FBCategory(str, Enum):
        TONE = "tone"
        PACING = "pacing"
        CONTENT_RELEVANCE = "content_relevance"
        INITIATIVE = "initiative"
        HUMOR = "humor"
        ACCURACY = "accuracy"
        SAFETY = "safety"
        OTHER = "other"

    class _FBOrigin(str, Enum):
        EXPLICIT = "explicit"
        IMPLICIT = "implicit"

    class _FBPolarity(str, Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"
        MIXED = "mixed"

    _CanonicalFeedbackSignal = Any  # type: ignore


# ======================================================================
# Utilities
# ======================================================================

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_finite_real(value: Any) -> bool:
    """True only for a real int/float that is neither NaN nor +/-inf."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return not (math.isnan(value) or math.isinf(value))


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sign(value: float, epsilon: float = 1e-9) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


# ======================================================================
# Input contract: LearningSignal (this engine's INTERNAL working shape,
# produced either directly or bridged from learning_models.LearningSignal)
# ======================================================================

class SignalType(str, Enum):
    EXPLICIT = "explicit"          # user directly stated a preference
    CORRECTION = "correction"      # user corrected a prior behavior
    IMPLICIT_POSITIVE = "implicit_positive"  # inferred positive reaction
    IMPLICIT_NEGATIVE = "implicit_negative"  # inferred negative reaction
    SYSTEM = "system"              # engine-internal/derived signal


# Explicit provenance-weight ordering: explicit statements outrank weak
# inference, per the mandated evidence hierarchy ("explicit > repeated
# strong behavioral evidence > weak inference").
_SIGNAL_TYPE_BASE_TRUST: Dict[SignalType, float] = {
    SignalType.EXPLICIT: 1.0,
    SignalType.CORRECTION: 0.95,
    SignalType.IMPLICIT_POSITIVE: 0.55,
    SignalType.IMPLICIT_NEGATIVE: 0.55,
    SignalType.SYSTEM: 0.4,
}

# Default, overridable mapping from a feedback category to the learning
# "dimension" this engine tracks belief for. Mirrors the vocabulary
# adaptation_engine.py's DEFAULT_DIMENSION_TO_AREA already expects
# (verbosity, explanation_depth, formality, topic_affinity, ...).
DEFAULT_CATEGORY_TO_DIMENSION: Dict[_FBCategory, str] = {
    _FBCategory.PACING: "verbosity",
    _FBCategory.TONE: "formality",
    _FBCategory.CONTENT_RELEVANCE: "topic_affinity",
    _FBCategory.INITIATIVE: "proactivity",
    _FBCategory.HUMOR: "communication_style",
    _FBCategory.ACCURACY: "explanation_depth",
    _FBCategory.SAFETY: "general_assistance_emphasis",
    _FBCategory.OTHER: "general_assistance_emphasis",
}


class LearningRejectionReason(str, Enum):
    """Reasons a signal can be safely rejected without raising."""
    NON_FINITE_VALUE = "non_finite_value"
    NON_FINITE_CONFIDENCE = "non_finite_confidence"
    UNMAPPED_CATEGORY = "unmapped_category"
    EMPTY_DIMENSION = "empty_dimension"


@dataclass(frozen=True)
class LearningSignal:
    """
    A single unit of validated evidence, in this engine's internal shape.

    Fields
    ------
    signal_id:
        Unique id for this signal. Used for de-duplication. If produced
        via `from_raw`/`from_feedback_signal` without an explicit id, a
        fresh uuid4 is generated — callers who need cross-call dedup
        should supply a stable id derived upstream (e.g. feedback_engine's
        own `signal_id` or idempotency key) so retries don't get
        double-counted.
    subject:
        The entity/topic this evidence is about, e.g. "user_123" or
        "user_123:communication_style". Callers that share one
        LearningEngine across users MUST namespace this to avoid
        cross-user bleed (see module docstring).
    dimension:
        The specific trait/preference being updated, e.g.
        "verbosity", "formality", "humor_tolerance". Keeps subjects
        multi-dimensional without needing separate LearningState trees.
    value:
        Directional magnitude of the evidence in [-1.0, 1.0]. Positive
        values push the dimension one way, negative the other. Must be
        a finite real number — NaN/Inf is rejected upstream in
        `LearningEngine.process`, never silently clamped into a
        misleading finite value.
    signal_confidence:
        Upstream confidence that this signal is valid, in [0.0, 1.0].
        This is independent of signal_type trust weight, and — like
        `value` — is treated as untrusted client-adjacent input and
        validated for finiteness before use.
    signal_type:
        Category of evidence (see SignalType). Governs base trust.
    timestamp:
        When the underlying event occurred (not when it was processed).
    source:
        Free-text origin, e.g. "chat_message", "feedback_engine".
    context:
        Arbitrary metadata for audit/debugging. Not interpreted here.
    """

    subject: str
    dimension: str
    value: float
    signal_type: SignalType
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_confidence: float = 0.8
    timestamp: datetime = field(default_factory=_now)
    source: str = "unknown"
    context: Dict[str, Any] = field(default_factory=dict)

    def is_value_finite(self) -> bool:
        return _is_finite_real(self.value)

    def is_confidence_finite(self) -> bool:
        return _is_finite_real(self.signal_confidence)

    @staticmethod
    def from_raw(raw: Dict[str, Any]) -> "LearningSignal":
        """
        Adapter for a plain-dict signal representation (e.g. a
        persistence round-trip or a hand-built test fixture). Deliberately
        permissive on shape; numeric validity is still enforced later by
        `LearningEngine.process`, not here, so a malformed dict can be
        constructed but will be safely rejected at processing time rather
        than raising during construction.
        """
        signal_type_raw = raw.get("signal_type", raw.get("type", "implicit_positive"))
        try:
            signal_type = SignalType(signal_type_raw)
        except ValueError:
            signal_type = SignalType.SYSTEM

        ts = raw.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = _now()

        raw_value = raw.get("value", 0.0)
        raw_confidence = raw.get("signal_confidence", raw.get("confidence", 0.8))

        return LearningSignal(
            subject=str(raw["subject"]),
            dimension=str(raw.get("dimension", "general")),
            value=raw_value if _is_finite_real(raw_value) else 0.0,
            signal_type=signal_type,
            signal_id=str(raw.get("signal_id", uuid.uuid4())),
            signal_confidence=raw_confidence if _is_finite_real(raw_confidence) else 0.0,
            timestamp=timestamp,
            source=str(raw.get("source", "unknown")),
            context=dict(raw.get("context", {})),
        )

    @staticmethod
    def from_feedback_signal(
        fb_signal: Any,
        *,
        origin: Any,
        dimension: Optional[str] = None,
        category_to_dimension: Optional[Dict[Any, str]] = None,
        source: str = "feedback_engine",
    ) -> "LearningSignal":
        """
        THE sanctioned bridge from feedback_engine.py's actual output
        (`learning_models.LearningSignal`, carrying `category`, `polarity`,
        `strength`, `confidence`, `metadata`) into this engine's internal
        `LearningSignal` shape.

        This function imports only `learning_models.py` types (never
        feedback_engine.py's internal enums/functions), matching the
        required module boundary: "the connection to feedback_engine.py
        must happen through LearningSignal contracts, not by importing
        its internal implementation."

        `origin` must be a `learning_models.FeedbackOrigin` (or the
        fallback stand-in) — feedback_engine's `FeedbackEvent.origin` —
        since the canonical `LearningSignal` contract itself does not
        carry origin; the caller is expected to have it in hand from the
        same `FeedbackProcessingResult` that produced `fb_signal`.

        `dimension`, if omitted, is resolved from `fb_signal.category`
        via `category_to_dimension` (defaulting to
        `DEFAULT_CATEGORY_TO_DIMENSION`). An unmapped category with no
        explicit `dimension` override still produces a signal (mapped to
        a generic "general" dimension) rather than raising — validity is
        judged at `LearningEngine.process` time, consistently with every
        other construction path in this module.

        Directionality: `polarity` POSITIVE/NEGATIVE combines with
        `strength` to form the signed `value`; NEUTRAL/MIXED polarity
        produces `value = 0.0` (no directional push) while still carrying
        through as evidence for spam/oscillation bookkeeping — mixed or
        neutral feedback should not silently masquerade as a directional
        preference in either direction.
        """
        mapping = category_to_dimension or DEFAULT_CATEGORY_TO_DIMENSION
        category = getattr(fb_signal, "category", None)
        polarity = getattr(fb_signal, "polarity", None)
        strength = getattr(fb_signal, "strength", 0.0)
        confidence = getattr(fb_signal, "confidence", 0.0)
        metadata = dict(getattr(fb_signal, "metadata", {}) or {})

        resolved_dimension = dimension or mapping.get(category, "general")

        if polarity == _FBPolarity.POSITIVE:
            signed_value = strength if _is_finite_real(strength) else 0.0
        elif polarity == _FBPolarity.NEGATIVE:
            signed_value = -strength if _is_finite_real(strength) else 0.0
        else:  # NEUTRAL, MIXED, or unrecognized -> no directional push
            signed_value = 0.0

        kind = str(metadata.get("kind", ""))
        is_correction_kind = "correction" in kind
        if origin == _FBOrigin.EXPLICIT:
            signal_type = SignalType.CORRECTION if is_correction_kind else SignalType.EXPLICIT
        elif origin == _FBOrigin.IMPLICIT:
            if polarity == _FBPolarity.POSITIVE:
                signal_type = SignalType.IMPLICIT_POSITIVE
            elif polarity == _FBPolarity.NEGATIVE:
                signal_type = SignalType.IMPLICIT_NEGATIVE
            else:
                signal_type = SignalType.SYSTEM
        else:
            signal_type = SignalType.SYSTEM

        derived_at = getattr(fb_signal, "derived_at", None)
        timestamp = derived_at if isinstance(derived_at, datetime) else _now()

        return LearningSignal(
            subject=str(metadata.get("subject", "")) or "unknown_subject",
            dimension=resolved_dimension,
            value=signed_value,
            signal_type=signal_type,
            signal_id=str(getattr(fb_signal, "signal_id", uuid.uuid4())),
            signal_confidence=confidence if _is_finite_real(confidence) else 0.0,
            timestamp=timestamp,
            source=source,
            context={
                "bridged_from": "learning_models.LearningSignal",
                "category": getattr(category, "value", str(category)),
                "polarity": getattr(polarity, "value", str(polarity)),
                "kind": kind,
                "repetition_risk": metadata.get("repetition_risk"),
                "evidence_count": metadata.get("evidence_count"),
            },
        )


# ======================================================================
# Internal evidence bookkeeping (for provenance + explainability)
# ======================================================================

@dataclass(frozen=True)
class EvidenceRecord:
    """One accepted piece of evidence, kept for provenance/explainability."""
    signal_id: str
    dimension: str
    value: float
    effective_weight: float
    signal_type: SignalType
    source: str
    timestamp: datetime


@dataclass(frozen=True)
class CompactedEvidence:
    """
    Aggregate summary of evidence that aged out of full detail retention.
    History is never destroyed outright — old detail is folded into a
    running summary so decay reduces *influence*, not *record*.
    """
    count: int = 0
    weighted_value_sum: float = 0.0
    weighted_weight_sum: float = 0.0
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None

    def folded_with(self, rec: EvidenceRecord) -> "CompactedEvidence":
        return CompactedEvidence(
            count=self.count + 1,
            weighted_value_sum=self.weighted_value_sum + rec.value * rec.effective_weight,
            weighted_weight_sum=self.weighted_weight_sum + rec.effective_weight,
            earliest_timestamp=min(
                [t for t in (self.earliest_timestamp, rec.timestamp) if t is not None]
            ),
            latest_timestamp=max(
                [t for t in (self.latest_timestamp, rec.timestamp) if t is not None]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "weighted_value_sum": self.weighted_value_sum,
            "weighted_weight_sum": self.weighted_weight_sum,
            "earliest_timestamp": self.earliest_timestamp.isoformat() if self.earliest_timestamp else None,
            "latest_timestamp": self.latest_timestamp.isoformat() if self.latest_timestamp else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CompactedEvidence":
        return CompactedEvidence(
            count=d.get("count", 0),
            weighted_value_sum=d.get("weighted_value_sum", 0.0),
            weighted_weight_sum=d.get("weighted_weight_sum", 0.0),
            earliest_timestamp=datetime.fromisoformat(d["earliest_timestamp"]) if d.get("earliest_timestamp") else None,
            latest_timestamp=datetime.fromisoformat(d["latest_timestamp"]) if d.get("latest_timestamp") else None,
        )


# ======================================================================
# Per-dimension and top-level learning state (the output contract)
# ======================================================================

@dataclass(frozen=True)
class DimensionState:
    """
    Learning state for one (subject, dimension) pair.

    net_score:
        Current believed value in [-1.0, 1.0]. Bounded — cannot run away
        to infinity regardless of how much evidence accumulates.
    confidence:
        How sure the engine is in net_score, in [0.0, 1.0]. Saturates;
        never claims certainty from a single event.
    evidence_count:
        Total number of accepted evidence events ever folded in (full
        detail + compacted), i.e. never decreases.
    positive_evidence_count / negative_evidence_count:
        Running counts of accepted evidence whose *signed value* was
        strictly positive / negative, for explainability ("why does this
        preference exist") without needing to replay full history.
    last_updated:
        Timestamp of last accepted update, used to compute decay on the
        next call.
    recent_history:
        Bounded deque of full-detail EvidenceRecord for explainability.
        Oldest entries are folded into `compacted` rather than discarded
        outright when the deque overflows.
    compacted:
        Aggregate of evidence that aged out of `recent_history`.
    recent_signs:
        Bounded window of recent value signs, used for oscillation /
        stability detection.
    contradiction_flag:
        True if the most recent update conflicted with the prevailing
        direction of evidence (kept visible for explainability /
        downstream caution, not just silently resolved).
    stability_score:
        1.0 = very stable (consistent direction over time), 0.0 = highly
        oscillating. Downstream consumers can use this to gate how
        aggressively to adapt behavior.
    is_stable:
        Convenience flag: True once evidence_count, confidence, and
        stability_score all clear their configured thresholds — i.e. this
        dimension has crossed from "one-off signal" into an established
        tendency (see LearningEngineConfig.stability_*).
    seen_signal_ids:
        Small bounded set of recently seen signal_ids for duplicate
        rejection. Not a full audit log (that's `recent_history`).
    """
    dimension: str
    net_score: float = 0.0
    confidence: float = 0.0
    evidence_count: int = 0
    positive_evidence_count: int = 0
    negative_evidence_count: int = 0
    last_updated: Optional[datetime] = None
    recent_history: Tuple[EvidenceRecord, ...] = field(default_factory=tuple)
    compacted: CompactedEvidence = field(default_factory=CompactedEvidence)
    recent_signs: Tuple[int, ...] = field(default_factory=tuple)
    contradiction_flag: bool = False
    stability_score: float = 1.0
    is_stable: bool = False
    seen_signal_ids: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "net_score": self.net_score,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "positive_evidence_count": self.positive_evidence_count,
            "negative_evidence_count": self.negative_evidence_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "recent_history": [
                {
                    "signal_id": r.signal_id,
                    "dimension": r.dimension,
                    "value": r.value,
                    "effective_weight": r.effective_weight,
                    "signal_type": r.signal_type.value,
                    "source": r.source,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.recent_history
            ],
            "compacted": self.compacted.to_dict(),
            "recent_signs": list(self.recent_signs),
            "contradiction_flag": self.contradiction_flag,
            "stability_score": self.stability_score,
            "is_stable": self.is_stable,
            "seen_signal_ids": list(self.seen_signal_ids),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DimensionState":
        return DimensionState(
            dimension=d["dimension"],
            net_score=d.get("net_score", 0.0),
            confidence=d.get("confidence", 0.0),
            evidence_count=d.get("evidence_count", 0),
            positive_evidence_count=d.get("positive_evidence_count", 0),
            negative_evidence_count=d.get("negative_evidence_count", 0),
            last_updated=datetime.fromisoformat(d["last_updated"]) if d.get("last_updated") else None,
            recent_history=tuple(
                EvidenceRecord(
                    signal_id=r["signal_id"],
                    dimension=r["dimension"],
                    value=r["value"],
                    effective_weight=r["effective_weight"],
                    signal_type=SignalType(r["signal_type"]),
                    source=r["source"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                for r in d.get("recent_history", [])
            ),
            compacted=CompactedEvidence.from_dict(d.get("compacted", {})),
            recent_signs=tuple(d.get("recent_signs", [])),
            contradiction_flag=d.get("contradiction_flag", False),
            stability_score=d.get("stability_score", 1.0),
            is_stable=d.get("is_stable", False),
            seen_signal_ids=tuple(d.get("seen_signal_ids", [])),
        )


@dataclass(frozen=True)
class LearningState:
    """
    Full learning state for a subject, keyed by dimension.
    This is the object handed to adaptation_engine.

    Persistence boundary: this class and everything it contains
    (`DimensionState`, `EvidenceRecord`, `CompactedEvidence`) is a plain,
    serializable dataclass tree with `to_dict()`/`from_dict()` on every
    level. A future persistence layer can store/load it as JSON (or any
    row/document shape derived from that dict) without this module ever
    importing database.py or knowing that a database exists.
    """
    subject: str
    dimensions: Dict[str, DimensionState] = field(default_factory=dict)
    version: int = 0
    last_updated: Optional[datetime] = None

    def get_dimension(self, dimension: str) -> DimensionState:
        return self.dimensions.get(dimension, DimensionState(dimension=dimension))

    def with_dimension(self, dim_state: DimensionState) -> "LearningState":
        new_dims = dict(self.dimensions)
        new_dims[dim_state.dimension] = dim_state
        return replace(
            self,
            dimensions=new_dims,
            version=self.version + 1,
            last_updated=_now(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "version": self.version,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LearningState":
        return LearningState(
            subject=d["subject"],
            dimensions={
                k: DimensionState.from_dict(v) for k, v in d.get("dimensions", {}).items()
            },
            version=d.get("version", 0),
            last_updated=datetime.fromisoformat(d["last_updated"]) if d.get("last_updated") else None,
        )

    @staticmethod
    def empty(subject: str) -> "LearningState":
        return LearningState(subject=subject)


# ======================================================================
# Explainability contract: what happened and why
# ======================================================================

class TransitionOutcome(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_DAMPENED = "accepted_dampened"      # applied, but reduced impact
    REJECTED_DUPLICATE = "rejected_duplicate"
    REJECTED_RATE_LIMITED = "rejected_rate_limited"
    REJECTED_INVALID = "rejected_invalid"        # malformed/non-finite signal


@dataclass(frozen=True)
class TransitionRecord:
    """
    Explains a single call to LearningEngine.process(): what evidence came
    in, what the engine decided, and why. Intended for logging/debugging/
    audit — not required by adaptation_engine but cheap to keep. Exposes
    structured metadata only; never free-text reasoning or chain-of-thought.
    """
    subject: str
    dimension: str
    signal_id: str
    outcome: TransitionOutcome
    previous_score: float
    new_score: float
    previous_confidence: float
    new_confidence: float
    raw_signal_value: float
    effective_weight: float
    reason_codes: Tuple[str, ...]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "dimension": self.dimension,
            "signal_id": self.signal_id,
            "outcome": self.outcome.value,
            "previous_score": self.previous_score,
            "new_score": self.new_score,
            "previous_confidence": self.previous_confidence,
            "new_confidence": self.new_confidence,
            "raw_signal_value": self.raw_signal_value,
            "effective_weight": self.effective_weight,
            "reason_codes": list(self.reason_codes),
            "details": self.details,
        }


# ======================================================================
# Strategy interface (so a future ML learner can replace the rule engine)
# ======================================================================

@dataclass(frozen=True)
class StrategyResult:
    """What the strategy thinks should happen to the score/confidence."""
    delta_score: float
    delta_confidence: float
    effective_weight: float
    reason_codes: Tuple[str, ...]


class LearningStrategy(ABC):
    """
    Pluggable scoring strategy. LearningEngine depends only on this
    interface, so a statistical/ML-based strategy can be swapped in later
    without touching LearningEngine, LearningState, or any consumer.
    """

    @abstractmethod
    def compute_update(
        self,
        signal: LearningSignal,
        dim_state: DimensionState,
        now: datetime,
    ) -> StrategyResult:
        raise NotImplementedError


@dataclass
class RuleBasedStrategyConfig:
    # Recency weighting: exponential decay half-life for a single event's
    # relevance to *this update* is not needed (we weight at ingest time);
    # this half-life governs decay of the *accumulated state* between
    # updates, applied by the engine before invoking the strategy.
    state_decay_half_life_days: float = 21.0
    confidence_floor: float = 0.05  # decay never erases all confidence
    score_decay_floor_fraction: float = 0.15  # decay pulls toward 0 but leaves a remainder

    # Reinforcement: diminishing-returns learning rate. Higher = faster
    # adaptation per event, but events approach (not reach) the target.
    base_learning_rate: float = 0.28

    # Bounded learning: no single event may move score by more than this.
    max_single_event_delta: float = 0.35

    # Confidence growth is saturating; this is the max confidence any
    # single event can contribute even at full trust/agreement.
    max_single_event_confidence_gain: float = 0.18

    # Contradiction handling: if new evidence opposes the prevailing
    # score sign, dampen its score impact and confidence impact.
    contradiction_score_dampening: float = 0.5
    contradiction_confidence_penalty: float = 0.12

    # Stability / oscillation detection.
    oscillation_window: int = 6
    oscillation_sign_change_threshold: int = 3  # sign flips in window -> unstable
    unstable_learning_rate_multiplier: float = 0.4

    # Anti-oscillation reversal gate: on top of the general oscillation
    # dampening above, a *full sign reversal* of an already-confident
    # net_score requires the incoming evidence to be at least this
    # confident, or the reversal itself is additionally dampened. This is
    # what stops "SHORT -> LONG -> SHORT -> LONG" from tracking every
    # individual interaction once a tendency is established.
    established_confidence_for_reversal_gate: float = 0.5
    reversal_confidence_floor: float = 0.6
    reversal_dampening_multiplier: float = 0.35


class RuleBasedLearningStrategy(LearningStrategy):
    """
    Deterministic, explainable rule-based strategy implementing the
    evidence-hierarchy, decay, contradiction, and anti-oscillation
    requirements. This is the default/initial "brain" — replaceable later
    via the LearningStrategy interface without touching LearningEngine.
    """

    def __init__(self, config: Optional[RuleBasedStrategyConfig] = None):
        self.config = config or RuleBasedStrategyConfig()

    def compute_update(
        self,
        signal: LearningSignal,
        dim_state: DimensionState,
        now: datetime,
    ) -> StrategyResult:
        cfg = self.config
        reason_codes: List[str] = []

        # --- Base trust from signal type + upstream signal confidence.
        base_trust = _SIGNAL_TYPE_BASE_TRUST.get(signal.signal_type, 0.5)
        trust = _clamp(base_trust * signal.signal_confidence, 0.0, 1.0)
        reason_codes.append(f"base_trust={base_trust:.2f}")
        reason_codes.append(f"signal_confidence={signal.signal_confidence:.2f}")

        # --- Recency weighting relative to the event's own timestamp.
        # Evidence reported very late (stale at ingest) counts for less.
        age_seconds = max(0.0, (now - signal.timestamp).total_seconds())
        # Soft recency penalty: full weight inside 1 hour, gently decays
        # afterward. This is distinct from *state* decay (applied by the
        # engine before calling the strategy) — this is about the
        # freshness of the incoming evidence itself.
        recency_factor = 1.0 / (1.0 + age_seconds / (6 * 3600.0))
        reason_codes.append(f"recency_factor={recency_factor:.2f}")

        raw_weight = trust * recency_factor
        raw_weight = _clamp(raw_weight, 0.0, 1.0)

        # --- Contradiction detection: does this oppose the prevailing score?
        prevailing_sign = _sign(dim_state.net_score)
        incoming_sign = _sign(signal.value)
        is_contradiction = (
            prevailing_sign != 0
            and incoming_sign != 0
            and prevailing_sign != incoming_sign
            and dim_state.confidence > 0.2  # only meaningful once we have some belief
        )
        if is_contradiction:
            raw_weight *= cfg.contradiction_score_dampening
            reason_codes.append("contradiction_detected")

        # --- Anti-oscillation reversal gate: a full sign flip against an
        # already-*established* (confident) belief needs stronger, more
        # trustworthy evidence than routine corroboration does.
        is_established_reversal_attempt = (
            is_contradiction
            and dim_state.confidence >= cfg.established_confidence_for_reversal_gate
        )
        if is_established_reversal_attempt:
            reason_codes.append("established_reversal_attempt")
            if signal.signal_confidence < cfg.reversal_confidence_floor:
                raw_weight *= cfg.reversal_dampening_multiplier
                reason_codes.append("reversal_gate_dampening_applied")
            else:
                reason_codes.append("reversal_gate_cleared_by_high_confidence_signal")

        # --- Stability / oscillation detection.
        signs_after = list(dim_state.recent_signs[-(cfg.oscillation_window - 1):]) + [incoming_sign]
        sign_changes = sum(
            1 for a, b in zip(signs_after, signs_after[1:]) if a != 0 and b != 0 and a != b
        )
        is_oscillating = sign_changes >= cfg.oscillation_sign_change_threshold
        if is_oscillating:
            raw_weight *= cfg.unstable_learning_rate_multiplier
            reason_codes.append(f"oscillation_detected(sign_changes={sign_changes})")

        # --- Reinforcement with diminishing returns: move toward the
        # signal's value, not toward +-inf. Distance-to-target shrinks the
        # step as the state already agrees with the evidence, and repeated
        # *consistent* evidence still keeps nudging (not stuck at first hit).
        target = signal.value
        distance = target - dim_state.net_score
        learning_rate = cfg.base_learning_rate * raw_weight
        delta_score = distance * learning_rate

        # --- Bounded learning: cap any single event's influence. This is
        # what guarantees "one implicit event must never permanently
        # establish a strong preference" — the cap applies uniformly,
        # explicit evidence just reaches it faster via higher raw_weight.
        if abs(delta_score) > cfg.max_single_event_delta:
            delta_score = math.copysign(cfg.max_single_event_delta, delta_score)
            reason_codes.append("single_event_cap_applied")

        # --- Confidence: saturating growth, reduced by contradiction/oscillation.
        # Agreement with existing state increases confidence faster than
        # a lone contradicting event does.
        agreement_bonus = 1.0 if prevailing_sign == 0 or prevailing_sign == incoming_sign else 0.3
        delta_confidence = cfg.max_single_event_confidence_gain * raw_weight * agreement_bonus
        # Saturate confidence gain harder as we approach full confidence
        # (evidence gets harder to move an already-confident state).
        headroom = max(0.0, 1.0 - dim_state.confidence)
        delta_confidence *= headroom

        if is_contradiction:
            delta_confidence -= cfg.contradiction_confidence_penalty
            reason_codes.append("confidence_contradiction_penalty")

        return StrategyResult(
            delta_score=delta_score,
            delta_confidence=delta_confidence,
            effective_weight=raw_weight,
            reason_codes=tuple(reason_codes),
        )


# ======================================================================
# Anti-spam / anti-duplicate bookkeeping (thread-safe, instance-scoped)
# ======================================================================

class _RateLimiter:
    """
    Simple sliding-window counter + duplicate-id cache, keyed by
    (subject, dimension). Instance-scoped and thread-safe — NOT a global.
    This is also this engine's idempotency mechanism: the same
    `signal_id` seen twice for the same (subject, dimension) is reported
    as a duplicate and never learned from twice.
    """

    def __init__(
        self,
        max_events_per_window: int = 20,
        window_seconds: float = 60.0,
        duplicate_id_cache_size: int = 128,
    ):
        self._max_events = max_events_per_window
        self._window = window_seconds
        self._dup_cache_size = duplicate_id_cache_size
        self._lock = threading.Lock()
        self._event_times: Dict[Tuple[str, str], Deque[float]] = {}
        self._seen_ids: Dict[Tuple[str, str], Deque[str]] = {}
        self._seen_ids_set: Dict[Tuple[str, str], set] = {}

    def check_and_record(self, key: Tuple[str, str], signal_id: str) -> Tuple[bool, bool]:
        """
        Returns (is_duplicate, is_rate_limited). Records the event if
        neither condition trips (rate-limit check happens even if not a
        duplicate; duplicates don't consume rate-limit budget twice).
        """
        now = time.monotonic()
        with self._lock:
            seen_ids = self._seen_ids.setdefault(key, deque(maxlen=self._dup_cache_size))
            seen_set = self._seen_ids_set.setdefault(key, set())

            if signal_id in seen_set:
                return True, False

            times = self._event_times.setdefault(key, deque())
            cutoff = now - self._window
            while times and times[0] < cutoff:
                times.popleft()

            if len(times) >= self._max_events:
                return False, True

            times.append(now)
            seen_ids.append(signal_id)
            seen_set.add(signal_id)
            # keep seen_set bounded in lockstep with the deque
            if len(seen_ids) == seen_ids.maxlen and len(seen_set) > seen_ids.maxlen:
                # rebuild set from deque occasionally to drop evicted ids
                self._seen_ids_set[key] = set(seen_ids)

            return False, False


# ======================================================================
# The engine
# ======================================================================

@dataclass
class LearningEngineConfig:
    max_full_history_per_dimension: int = 25
    rate_limit_max_events: int = 20
    rate_limit_window_seconds: float = 60.0
    strategy: Optional[LearningStrategy] = None
    state_decay_half_life_days: float = 21.0
    confidence_floor: float = 0.05
    score_decay_floor_fraction: float = 0.15

    # Stability promotion thresholds (task 7): a dimension is only
    # reported `is_stable=True` once ALL of these are met simultaneously.
    stability_min_evidence_count: int = 8
    stability_min_confidence: float = 0.55
    stability_min_stability_score: float = 0.7


class LearningEngine:
    """
    Stateless processor over explicit LearningState in/out. Safe to share
    a single instance across requests/threads as long as `subject` values
    passed to `process` are namespaced per-user where relevant (see module
    docstring) — internal bookkeeping (rate limiting / dedup) is keyed by
    (subject, dimension), not held per-call.

    Usage:
        engine = LearningEngine()
        new_state, transition = engine.process(signal, current_state)
        # persist new_state via your own persistence layer (not this module)

    Or, bridging directly from feedback_engine.py's output:

        signal = LearningSignal.from_feedback_signal(
            fb_result.signal, origin=fb_result.event.origin
        )
        new_state, transition = engine.process(signal, current_state)
    """

    def __init__(self, config: Optional[LearningEngineConfig] = None):
        self.config = config or LearningEngineConfig()
        self._strategy: LearningStrategy = self.config.strategy or RuleBasedLearningStrategy()
        self._rate_limiter = _RateLimiter(
            max_events_per_window=self.config.rate_limit_max_events,
            window_seconds=self.config.rate_limit_window_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        signal: LearningSignal,
        state: LearningState,
    ) -> Tuple[LearningState, TransitionRecord]:
        """
        Apply one signal to the given state and return (new_state, transition).

        Never mutates `state` in place — always returns either a brand
        new `LearningState` (on acceptance) or the exact original `state`
        object (on any rejection), so no partial mutation is ever visible
        to a caller. `signal.subject != state.subject` is treated as a
        caller programming error (mismatched routing) and raises
        `ValueError`, matching prior behavior; every other invalid-input
        case is reported via `TransitionOutcome`, never raised.
        """
        if signal.subject != state.subject:
            raise ValueError(
                f"Signal subject '{signal.subject}' does not match "
                f"state subject '{state.subject}'"
            )

        dim_state = state.get_dimension(signal.dimension)
        now = _now()

        # --- Untrusted-input validation: reject non-finite numeric input
        # safely rather than letting NaN/Inf propagate into accumulated
        # state (which `_clamp`'s min/max cannot reliably catch, since
        # NaN comparisons are always False in Python).
        if not signal.is_value_finite():
            return state, self._rejection(
                signal, dim_state, TransitionOutcome.REJECTED_INVALID,
                (LearningRejectionReason.NON_FINITE_VALUE.value,),
            )
        if not signal.is_confidence_finite():
            return state, self._rejection(
                signal, dim_state, TransitionOutcome.REJECTED_INVALID,
                (LearningRejectionReason.NON_FINITE_CONFIDENCE.value,),
            )
        if not signal.dimension:
            return state, self._rejection(
                signal, dim_state, TransitionOutcome.REJECTED_INVALID,
                (LearningRejectionReason.EMPTY_DIMENSION.value,),
            )

        # --- Duplicate / rate-limit guard (also this engine's idempotency
        # mechanism: the same signal_id is never learned from twice).
        key = (signal.subject, signal.dimension)
        is_duplicate, is_rate_limited = self._rate_limiter.check_and_record(key, signal.signal_id)

        if is_duplicate:
            return state, self._rejection(
                signal, dim_state, TransitionOutcome.REJECTED_DUPLICATE,
                ("duplicate_signal_id",),
            )

        if is_rate_limited:
            return state, self._rejection(
                signal, dim_state, TransitionOutcome.REJECTED_RATE_LIMITED,
                ("rate_limit_exceeded",),
            )

        # --- Apply controlled decay to the existing state before scoring
        # new evidence, so stale preferences lose influence over time but
        # are never wiped to exactly zero / erased from history.
        decayed_dim_state = self._apply_decay(dim_state, now)

        # --- Delegate the actual scoring to the (replaceable) strategy.
        # Any bug inside a *custom* strategy implementation is NOT caught
        # here — an implementation error must remain diagnosable rather
        # than being reinterpreted as "invalid signal".
        result = self._strategy.compute_update(signal, decayed_dim_state, now)

        # --- Defensive re-validation of strategy output: a well-behaved
        # strategy already respects bounds, but this engine never trusts
        # ANY single component (including its own default strategy) to
        # keep accumulated state finite/bounded on its own.
        safe_delta_score = result.delta_score if _is_finite_real(result.delta_score) else 0.0
        safe_delta_confidence = result.delta_confidence if _is_finite_real(result.delta_confidence) else 0.0
        safe_effective_weight = _clamp(
            result.effective_weight if _is_finite_real(result.effective_weight) else 0.0,
            0.0, 1.0,
        )

        new_score = _clamp(decayed_dim_state.net_score + safe_delta_score)
        new_confidence = _clamp(decayed_dim_state.confidence + safe_delta_confidence, 0.0, 1.0)

        reason_codes = result.reason_codes
        outcome = (
            TransitionOutcome.ACCEPTED_DAMPENED
            if "contradiction_detected" in reason_codes
            or any("oscillation_detected" in r for r in reason_codes)
            else TransitionOutcome.ACCEPTED
        )

        record = EvidenceRecord(
            signal_id=signal.signal_id,
            dimension=signal.dimension,
            value=signal.value,
            effective_weight=safe_effective_weight,
            signal_type=signal.signal_type,
            source=signal.source,
            timestamp=signal.timestamp,
        )

        new_history, new_compacted = self._append_with_compaction(
            decayed_dim_state.recent_history,
            decayed_dim_state.compacted,
            record,
        )

        new_signs = (decayed_dim_state.recent_signs + (_sign(signal.value),))[
            -self._strategy_window():
        ]

        stability_score = self._compute_stability(new_signs)

        new_seen_ids = (decayed_dim_state.seen_signal_ids + (signal.signal_id,))[-64:]

        incoming_sign = _sign(signal.value)
        new_positive_count = decayed_dim_state.positive_evidence_count + (1 if incoming_sign > 0 else 0)
        new_negative_count = decayed_dim_state.negative_evidence_count + (1 if incoming_sign < 0 else 0)
        new_evidence_count = decayed_dim_state.evidence_count + 1

        is_stable = (
            new_evidence_count >= self.config.stability_min_evidence_count
            and new_confidence >= self.config.stability_min_confidence
            and stability_score >= self.config.stability_min_stability_score
        )

        new_dim_state = DimensionState(
            dimension=signal.dimension,
            net_score=new_score,
            confidence=new_confidence,
            evidence_count=new_evidence_count,
            positive_evidence_count=new_positive_count,
            negative_evidence_count=new_negative_count,
            last_updated=now,
            recent_history=new_history,
            compacted=new_compacted,
            recent_signs=new_signs,
            contradiction_flag="contradiction_detected" in reason_codes,
            stability_score=stability_score,
            is_stable=is_stable,
            seen_signal_ids=new_seen_ids,
        )

        new_state = state.with_dimension(new_dim_state)

        transition = TransitionRecord(
            subject=signal.subject,
            dimension=signal.dimension,
            signal_id=signal.signal_id,
            outcome=outcome,
            previous_score=dim_state.net_score,
            new_score=new_score,
            previous_confidence=dim_state.confidence,
            new_confidence=new_confidence,
            raw_signal_value=signal.value,
            effective_weight=safe_effective_weight,
            reason_codes=reason_codes,
            details={
                "decay_applied": decayed_dim_state.net_score != dim_state.net_score
                or decayed_dim_state.confidence != dim_state.confidence,
                "stability_score": stability_score,
                "is_stable": is_stable,
            },
        )

        return new_state, transition

    def ingest_feedback_signal(
        self,
        fb_signal: Any,
        state: LearningState,
        *,
        origin: Any,
        dimension: Optional[str] = None,
        category_to_dimension: Optional[Dict[Any, str]] = None,
        source: str = "feedback_engine",
    ) -> Tuple[LearningState, TransitionRecord]:
        """
        Convenience entrypoint that bridges a `learning_models.LearningSignal`
        (feedback_engine.py's actual output shape) directly into `process()`,
        without the caller needing to construct this engine's internal
        `LearningSignal` by hand. This IS the sanctioned feedback_engine.py
        connection point referenced in the module docstring — it only
        depends on the `learning_models.LearningSignal` contract plus a
        caller-supplied `origin`, never on feedback_engine.py's internals.

        `state.subject` must already be set to the correct (namespaced)
        subject by the caller; this function has no way to derive a
        subject from `fb_signal` alone (the canonical contract carries no
        subject field), so it borrows `state.subject` for the bridged
        signal.
        """
        bridged = LearningSignal.from_feedback_signal(
            fb_signal,
            origin=origin,
            dimension=dimension,
            category_to_dimension=category_to_dimension,
            source=source,
        )
        bridged = replace(bridged, subject=state.subject)
        return self.process(bridged, state)

    def decay_only(self, state: LearningState, dimension: str) -> LearningState:
        """
        Optional maintenance hook: apply time-based decay to a dimension
        without any new evidence (e.g. a scheduled housekeeping job).
        Does not touch other dimensions in the state.
        """
        dim_state = state.get_dimension(dimension)
        decayed = self._apply_decay(dim_state, _now())
        if decayed == dim_state:
            return state
        return state.with_dimension(decayed)

    def explain(self, state: LearningState, dimension: str) -> Dict[str, Any]:
        """
        Human-readable, structured-only explanation of the current state
        for a dimension, built entirely from stored provenance metadata —
        no free-text reasoning or chain-of-thought is ever exposed here.
        """
        dim = state.get_dimension(dimension)
        return {
            "dimension": dimension,
            "net_score": dim.net_score,
            "confidence": dim.confidence,
            "evidence_count": dim.evidence_count,
            "positive_evidence_count": dim.positive_evidence_count,
            "negative_evidence_count": dim.negative_evidence_count,
            "stability_score": dim.stability_score,
            "is_stable": dim.is_stable,
            "contradiction_flag": dim.contradiction_flag,
            "last_updated": dim.last_updated.isoformat() if dim.last_updated else None,
            "recent_evidence": [
                {
                    "value": r.value,
                    "weight": r.effective_weight,
                    "type": r.signal_type.value,
                    "source": r.source,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in dim.recent_history
            ],
            "older_evidence_summary": dim.compacted.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rejection(
        self,
        signal: LearningSignal,
        dim_state: DimensionState,
        outcome: TransitionOutcome,
        reason_codes: Tuple[str, ...],
    ) -> TransitionRecord:
        """Build a TransitionRecord for a safely-rejected signal, no state change."""
        safe_value = signal.value if _is_finite_real(signal.value) else 0.0
        safe_weight = 0.0
        return TransitionRecord(
            subject=signal.subject,
            dimension=signal.dimension or "unknown",
            signal_id=signal.signal_id,
            outcome=outcome,
            previous_score=dim_state.net_score,
            new_score=dim_state.net_score,
            previous_confidence=dim_state.confidence,
            new_confidence=dim_state.confidence,
            raw_signal_value=safe_value,
            effective_weight=safe_weight,
            reason_codes=reason_codes,
        )

    def _strategy_window(self) -> int:
        if isinstance(self._strategy, RuleBasedLearningStrategy):
            return self._strategy.config.oscillation_window
        return 6

    def _compute_stability(self, signs: Tuple[int, ...]) -> float:
        if len(signs) < 2:
            return 1.0
        changes = sum(
            1 for a, b in zip(signs, signs[1:]) if a != 0 and b != 0 and a != b
        )
        max_possible = len(signs) - 1
        if max_possible <= 0:
            return 1.0
        return _clamp(1.0 - (changes / max_possible), 0.0, 1.0)

    def _apply_decay(self, dim_state: DimensionState, now: datetime) -> DimensionState:
        """
        Controlled decay: reduces the *magnitude* of net_score and
        confidence toward a floor as time passes without new evidence.
        Never erases evidence_count/history/provenance — decay affects
        influence, not the record.
        """
        if dim_state.last_updated is None:
            return dim_state

        elapsed_days = max(0.0, (now - dim_state.last_updated).total_seconds() / 86400.0)
        if elapsed_days <= 0:
            return dim_state

        half_life = max(1e-6, self.config.state_decay_half_life_days)
        decay_factor = 0.5 ** (elapsed_days / half_life)  # in (0, 1]

        floor_fraction = self.config.score_decay_floor_fraction
        # Score decays toward zero but only down to a floor fraction of its
        # own magnitude — i.e. long-unconfirmed evidence weakens but a
        # trace persists rather than snapping to neutral.
        min_retained = dim_state.net_score * floor_fraction
        decayed_score = min_retained + (dim_state.net_score - min_retained) * decay_factor
        decayed_score = _clamp(decayed_score)

        decayed_confidence = max(
            self.config.confidence_floor,
            dim_state.confidence * decay_factor,
        )

        # Decayed confidence can also drop a dimension out of "stable"
        # status if it falls back below threshold — recomputed on the
        # next process() call via new_confidence/stability_score, so no
        # extra bookkeeping is needed here beyond leaving is_stable as-is
        # (a stale True is corrected the next time evidence arrives; a
        # pure decay_only() call recomputes it explicitly below).
        is_stable = (
            dim_state.evidence_count >= self.config.stability_min_evidence_count
            and decayed_confidence >= self.config.stability_min_confidence
            and dim_state.stability_score >= self.config.stability_min_stability_score
        )

        if (
            decayed_score == dim_state.net_score
            and decayed_confidence == dim_state.confidence
            and is_stable == dim_state.is_stable
        ):
            return dim_state

        return replace(
            dim_state,
            net_score=decayed_score,
            confidence=decayed_confidence,
            is_stable=is_stable,
        )

    def _append_with_compaction(
        self,
        history: Tuple[EvidenceRecord, ...],
        compacted: CompactedEvidence,
        new_record: EvidenceRecord,
    ) -> Tuple[Tuple[EvidenceRecord, ...], CompactedEvidence]:
        """
        Append new_record to full-detail history; if over the cap, fold
        the oldest full-detail record into `compacted` rather than
        dropping it outright.
        """
        updated = history + (new_record,)
        max_len = self.config.max_full_history_per_dimension
        new_compacted = compacted
        while len(updated) > max_len:
            oldest, *rest = updated
            new_compacted = new_compacted.folded_with(oldest)
            updated = tuple(rest)
        return updated, new_compacted


# ======================================================================
# Package boundary integration
# ======================================================================

def get_learning_engine(config: Optional[LearningEngineConfig] = None) -> LearningEngine:
    """
    Public factory expected by the package `__init__.py`'s lazy
    component resolution (`_COMPONENT_MODULES["learning_engine"]`).
    Called with zero arguments by that resolver, so `config` is optional
    and defaults to today's rule-based behavior exactly. Returns a fresh
    `LearningEngine` instance; construct and hold your own instance
    directly if you need config sharing or per-user isolation beyond
    what `process()`'s subject-namespacing already provides.
    """
    return LearningEngine(config)


# ======================================================================
# Example integration sketch (not executed; for reference only)
# ======================================================================
#
# from adaptive_intelligence_and_learning.feedback_engine import process_feedback
# from adaptive_intelligence_and_learning.learning_engine import (
#     LearningEngine, LearningState,
# )
#
# engine = LearningEngine()
# state = LearningState.from_dict(loaded_state_dict)  # loaded via YOUR persistence layer
#
# fb_result = process_feedback(source="user_utterance", origin="explicit", kind="explicit_preference", ...)
# if fb_result.accepted:
#     state, transition = engine.ingest_feedback_signal(
#         fb_result.signal, state, origin=fb_result.event.origin,
#     )
#     persist(state.to_dict())        # hand off to your persistence layer
#     log(transition.to_dict())       # explainability / audit trail
#
# adaptation_engine.process(state)    # downstream consumer
