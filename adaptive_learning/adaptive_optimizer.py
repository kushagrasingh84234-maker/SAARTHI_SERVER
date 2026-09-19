"""
Adaptive Intelligence & Learning / adaptive_optimizer.py
=========================================================

Phase 4F — Intelligent Adaptive Optimization Layer
----------------------------------------------------

Role
----
Sits AFTER the existing 4A-4E pipeline and BEFORE the existing policy
application layer:

    4A Feedback              -> LearningSignal
    4B Learning               -> LearningState
    4C Adaptation              -> AdaptationDecision
    4D Reinforcement/Decay     -> Stable LearningState
    4E Adaptive Response Policy -> Adaptive Policy
    4F Intelligent Adaptive Optimization (THIS MODULE)
            |
            v
    OptimizationRecommendation
            |
            v
    4E / existing policy application layer

4F is a DECISION-OPTIMIZATION LAYER ONLY. It evaluates whether the
already-learned/adaptive behavior for one (subject, area) pair can be
improved further, and returns a bounded, structured
`OptimizationRecommendation`. It never generates the final AI response,
and it never applies its own recommendation.

THIS MODULE IS NOT AN AUTONOMOUS AGENT. It must not, and does not:
  - execute arbitrary actions or shell commands
  - access credentials or change security settings
  - modify database records, user profile, memory, or response policy
  - call external APIs, an LLM, or the network
  - make uncontrolled/unbounded policy changes
  - hold hidden global mutable state (all bookkeeping is instance-scoped
    and size-bounded, mirroring adaptation_engine.py)

It only evaluates validated, already-computed structured system state
(AdaptationDecision objects plus bounded feedback-outcome summaries) and
returns structured, machine-readable `OptimizationRecommendation`
objects. No chain-of-thought, free-text reasoning, or raw conversation
transcript is ever accepted, produced, or stored.

CONTRACT NOTE
-------------
This module expects `AdaptationDecision`, `AdaptationArea`,
`DecisionOutcome`, and `Durability` from `adaptation_engine.py` (4C),
and `EvidenceStrengthTier` from `learning_models.py` (the cross-phase
contract layer). Both are imported defensively: if the sibling module
isn't importable, or its shapes differ, a structurally-compatible
fallback is defined so this file still runs standalone (the same
pattern `adaptation_engine.py` itself uses for `learning_engine.py`).
Everything below relies on attribute access, not on class identity.

PRIORITY HIERARCHY (never reversed, enforced unconditionally)
-----------------------------------------------------------------
    SYSTEM CONSTRAINT
        >
    EXPLICIT USER PREFERENCE
        >
    STABLE LEARNED PREFERENCE   (Durability.EXPLICIT / ESTABLISHED)
        >
    ADAPTATION                  (Durability.TENTATIVE)
        >
    WEAK INFERENCE

4F never recommends an optimization that would fight a hard constraint,
a locked area, or an explicitly-locked user preference field. At most
it may recommend reinforcing (STRENGTHEN / PROMOTE_TO_STABLE) such a
preference; it can never recommend REDUCE/STABILIZE/REJECT against one.

SAFETY ENVELOPE
----------------
Every numeric field on every recommendation is unconditionally clamped
to a finite, bounded range before it is returned -- regardless of what
the (possibly future, ML-based) `OptimizationStrategy` computed. NaN,
Infinity, and out-of-range values never leave this module. A locked
area, an explicitly-preferred field, or a `HELD_AREA_LOCKED` decision
can never be overridden by strategy confidence.

DETERMINISM / PURITY
----------------------
Given identical inputs (`AdaptationDecision`, feedback-outcome history,
recent-decision history, config, and wall-clock `now`), the default
`DeterministicOptimizationStrategy` produces the same recommendation.
No randomness is introduced anywhere in this module. The optimizer
performs no I/O, no LLM calls, no network calls, and no database
access; it operates purely on already-computed structured state passed
in by the caller.

FUTURE ML COMPATIBILITY
--------------------------
`OptimizationStrategy` is an abstract, pluggable interface.
`DeterministicOptimizationStrategy` is the safe, always-available
baseline. A future statistical, contextual-bandit, RL-based, or
learned-ranking strategy can implement the same interface and be
injected via `AdaptiveOptimizerConfig.strategy` without changing this
module's public surface, its safety envelope, or the
`OptimizationRecommendation` contract -- the unconditional safety
clamp and lock checks in `AdaptiveOptimizer.evaluate` apply to every
strategy's output identically.

DO NOT TOUCH
------------
This module does not import, and must never import, server.py,
logic.py, database.py, context_memory.py, user_profile.py,
emotion_engine.py, personality_engine.py, response_policy.py,
ai_services.py, feedback_engine.py, or learning_engine.py. It only
reads the public contracts exposed by adaptation_engine.py (4C) and
learning_models.py (the shared contract layer).
"""

from __future__ import annotations

import math
import threading
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FeedbackOutcomeSignal",
    "OptimizationDirection",
    "RiskLevel",
    "FeedbackOutcomeRecord",
    "OptimizationRecommendation",
    "OptimizationContext",
    "StrategyVerdict",
    "OptimizationStrategy",
    "DeterministicOptimizationStrategy",
    "DeterministicStrategyConfig",
    "AdaptiveOptimizerConfig",
    "AdaptiveOptimizer",
    "OptimizerValidationError",
    "get_adaptive_optimizer",
]

__version__ = "1.0.0"


# ======================================================================
# Shared data contracts (from adaptation_engine.py / learning_models.py)
# ======================================================================
try:
    from .adaptation_engine import (  # type: ignore
        AdaptationArea,
        AdaptationDecision,
        DecisionOutcome,
        Durability,
    )
except Exception:  # pragma: no cover - fallback for standalone use/testing
    class AdaptationArea(str, Enum):
        RESPONSE_LENGTH = "response_length"
        EXPLANATION_DEPTH = "explanation_depth"
        COMMUNICATION_STYLE = "communication_style"
        LANGUAGE_PREFERENCE = "language_preference"
        TOPIC_AFFINITY = "topic_affinity"
        EDUCATIONAL_EMPHASIS = "educational_emphasis"
        INFORMATIONAL_EMPHASIS = "informational_emphasis"
        GENERAL_ASSISTANCE_EMPHASIS = "general_assistance_emphasis"
        INTERACTION_STYLE = "interaction_style"
        INITIATIVE_LEVEL = "initiative_level"

    class DecisionOutcome(str, Enum):
        PROPOSED = "proposed"
        PROPOSED_TENTATIVE = "proposed_tentative"
        HELD_INSUFFICIENT_EVIDENCE = "held_insufficient_evidence"
        HELD_INSUFFICIENT_CONFIDENCE = "held_insufficient_confidence"
        HELD_OSCILLATION_GUARD = "held_oscillation_guard"
        HELD_CONTRADICTION = "held_contradiction"
        HELD_AREA_LOCKED = "held_area_locked"
        HELD_INVALID_STATE = "held_invalid_state"
        NO_CHANGE = "no_change"

    class Durability(str, Enum):
        TENTATIVE = "tentative"
        ESTABLISHED = "established"
        EXPLICIT = "explicit"

    @dataclass(frozen=True)
    class AdaptationDecision:  # minimal structural fallback
        decision_id: str
        subject: str
        area: AdaptationArea
        outcome: DecisionOutcome
        previous_value: float = 0.0
        proposed_value: float = 0.0
        delta: float = 0.0
        confidence: float = 0.0
        evidence_count: int = 0
        durability: Durability = Durability.TENTATIVE
        is_reversal: bool = False
        reason_codes: Tuple[str, ...] = ()
        provenance: Tuple[Any, ...] = ()
        generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

        @property
        def is_actionable(self) -> bool:
            return self.outcome in (DecisionOutcome.PROPOSED, DecisionOutcome.PROPOSED_TENTATIVE)

try:
    from .learning_models import EvidenceStrengthTier  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone use/testing
    class EvidenceStrengthTier(str, Enum):
        ANECDOTAL = "anecdotal"
        EMERGING = "emerging"
        ESTABLISHED = "established"
        STRONG = "strong"


# ======================================================================
# Errors
# ======================================================================

class OptimizerValidationError(ValueError):
    """Raised only for programming-error-grade malformed *construction*
    input (e.g. a non-finite weight passed to a frozen dataclass at
    __post_init__ time). Malformed *runtime* input to `evaluate()` is
    NOT raised -- it is handled as safely-rejectable input and mapped to
    a HOLD/REJECT recommendation instead (see rule 24, FAILURE SAFETY)."""


# ======================================================================
# Small pure helpers
# ======================================================================

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_finite_real(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return not (math.isnan(value) or math.isinf(value))


def _clamp01(value: float) -> float:
    if not _is_finite_real(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _clamp(value: float, lo: float, hi: float) -> float:
    if not _is_finite_real(value):
        return lo
    return max(lo, min(hi, float(value)))


def _safe_area_name(area: Any) -> str:
    return area.value if isinstance(area, Enum) else str(area)


def _new_id() -> str:
    return f"optrec_{uuid.uuid4().hex}"


# ======================================================================
# Enums
# ======================================================================

class FeedbackOutcomeSignal(str, Enum):
    """Structured evaluation of whether a prior adaptation appears to be
    working (rule 5). `UNKNOWN` must never be treated as `FAILURE` -- a
    missing feedback signal must not automatically punish an
    adaptation; it is simply excluded from the evidence tally."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NEUTRAL = "neutral"
    FAILURE = "failure"
    UNKNOWN = "unknown"


# Directional weight used only for effectiveness scoring; UNKNOWN is
# intentionally absent -- it is filtered out before scoring, never
# scored as zero-with-weight (that would still subtly count against a
# decision by diluting its confidence).
_OUTCOME_WEIGHT: Dict[FeedbackOutcomeSignal, float] = {
    FeedbackOutcomeSignal.SUCCESS: 1.0,
    FeedbackOutcomeSignal.PARTIAL_SUCCESS: 0.5,
    FeedbackOutcomeSignal.NEUTRAL: 0.0,
    FeedbackOutcomeSignal.FAILURE: -1.0,
}


class OptimizationDirection(str, Enum):
    """What 4F is recommending 4E/the policy application layer consider.
    This is a RECOMMENDATION, not an instruction -- 4F never applies any
    of these itself (rule 25, NO DIRECT SIDE EFFECTS)."""

    HOLD = "hold"                        # do nothing; insufficient/contradictory signal
    ADAPT = "adapt"                       # under-adaptation: begin/continue adapting
    STRENGTHEN = "strengthen"             # under-adaptation: increase confidence/effect
    PROMOTE_TO_STABLE = "promote_to_stable"  # under-adaptation: treat as stable/established
    REDUCE = "reduce"                     # over-adaptation / negative effectiveness
    STABILIZE = "stabilize"               # over-adaptation: stop changing, let it settle
    REJECT = "reject"                     # invalid/unsafe input; never act on it


# Directions that represent an actual proposed change (vs. inaction).
_ACTIONABLE_DIRECTIONS = frozenset(
    {
        OptimizationDirection.ADAPT,
        OptimizationDirection.STRENGTHEN,
        OptimizationDirection.PROMOTE_TO_STABLE,
        OptimizationDirection.REDUCE,
        OptimizationDirection.STABILIZE,
    }
)

# Directions that would work AGAINST an existing preference (as opposed
# to merely reinforcing or leaving it alone). Rule 11 forbids ever
# reaching these for a Durability.EXPLICIT decision.
_AGAINST_PREFERENCE_DIRECTIONS = frozenset(
    {
        OptimizationDirection.REDUCE,
        OptimizationDirection.STABILIZE,
    }
)


class RiskLevel(str, Enum):
    """Coarse risk bucket for a proposed optimization (rule 6). Higher
    risk requires stronger evidence before 4F will recommend anything
    beyond HOLD."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_RISK_ORDER: Dict[RiskLevel, int] = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
_DURABILITY_ORDER: Dict[Durability, int] = {
    Durability.TENTATIVE: 0,
    Durability.ESTABLISHED: 1,
    Durability.EXPLICIT: 2,
}
_EVIDENCE_TIER_ORDER: Dict[EvidenceStrengthTier, int] = {
    EvidenceStrengthTier.ANECDOTAL: 0,
    EvidenceStrengthTier.EMERGING: 1,
    EvidenceStrengthTier.ESTABLISHED: 2,
    EvidenceStrengthTier.STRONG: 3,
}


# ======================================================================
# Input contract: bounded feedback-outcome evidence
# ======================================================================

@dataclass(frozen=True)
class FeedbackOutcomeRecord:
    """
    One bounded, already-summarized piece of evidence about whether a
    past adaptation for a given area appears to be working. This is
    deliberately NOT raw feedback or a transcript (rule 21, PRIVACY) --
    it is the kind of already-validated, structured summary a caller
    would derive from `learning_models.FeedbackClassification` /
    `LearningSignal` records upstream, or roll up from several of them.

    Multiple records may exist for the same area/decision; 4F never
    mutates them, only folds a *sequence* of them (mirrors the
    append-only philosophy of `learning_models.py`).
    """

    record_id: str
    area: str  # AdaptationArea value or raw dimension name
    outcome: FeedbackOutcomeSignal
    observed_at: datetime
    weight: float = 1.0  # bounded [0, 1]: how much this record should count
    decision_id: Optional[str] = None  # which AdaptationDecision this evidences, if known
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise OptimizerValidationError("record_id must be a non-empty string")
        if not isinstance(self.area, str) or not self.area:
            raise OptimizerValidationError("area must be a non-empty string")
        if not isinstance(self.outcome, FeedbackOutcomeSignal):
            raise OptimizerValidationError(f"outcome must be a FeedbackOutcomeSignal, got {self.outcome!r}")
        if not isinstance(self.observed_at, datetime):
            raise OptimizerValidationError("observed_at must be a datetime")
        observed_at = self.observed_at
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "observed_at", observed_at.astimezone(timezone.utc))
        if not _is_finite_real(self.weight):
            raise OptimizerValidationError(f"weight must be a finite number, got {self.weight!r}")
        object.__setattr__(self, "weight", _clamp01(self.weight))
        if self.decision_id is not None and not isinstance(self.decision_id, str):
            raise OptimizerValidationError("decision_id must be a string or None")
        if not isinstance(self.metadata, Mapping):
            raise OptimizerValidationError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "area": self.area,
            "outcome": self.outcome.value,
            "observed_at": self.observed_at.isoformat(),
            "weight": self.weight,
            "decision_id": self.decision_id,
            "metadata": dict(self.metadata),
        }


# ======================================================================
# Output contract: OptimizationRecommendation
# ======================================================================

@dataclass(frozen=True)
class OptimizationRecommendation:
    """
    The complete, bounded, machine-readable output of Phase 4F (rule 3).
    Every field here is either an id/enum/timestamp or a numeric value
    clamped into a finite, documented range -- never free text
    explanation and never chain-of-thought.
    """

    recommendation_id: str
    generated_at: datetime
    subject: str
    target_area: str  # AdaptationArea value
    current_value: float  # snapshot of the decision's proposed_value at eval time
    current_durability: str  # Durability value
    proposed_direction: OptimizationDirection
    confidence: float  # bounded [0, 1]: how confident 4F is in this recommendation
    evidence_strength: EvidenceStrengthTier
    expected_benefit: float  # bounded [0, 1]
    risk_level: RiskLevel
    stability: float  # bounded [0, 1]: 1.0 == no recent oscillation
    reversible: bool  # always True today (rule 12); kept explicit for future strategies
    reason_codes: Tuple[str, ...]
    source_decision_ids: Tuple[str, ...]
    source_feedback_ids: Tuple[str, ...]
    cooldown_active: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.proposed_direction, OptimizationDirection):
            raise OptimizerValidationError("proposed_direction must be an OptimizationDirection")
        if not isinstance(self.evidence_strength, EvidenceStrengthTier):
            raise OptimizerValidationError("evidence_strength must be an EvidenceStrengthTier")
        if not isinstance(self.risk_level, RiskLevel):
            raise OptimizerValidationError("risk_level must be a RiskLevel")
        object.__setattr__(self, "confidence", _clamp01(self.confidence))
        object.__setattr__(self, "expected_benefit", _clamp01(self.expected_benefit))
        object.__setattr__(self, "stability", _clamp01(self.stability))
        if not _is_finite_real(self.current_value):
            object.__setattr__(self, "current_value", 0.0)
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "source_decision_ids", tuple(self.source_decision_ids))
        object.__setattr__(self, "source_feedback_ids", tuple(self.source_feedback_ids))
        if not isinstance(self.metadata, Mapping):
            raise OptimizerValidationError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_actionable(self) -> bool:
        """Convenience for the consuming layer: does this recommendation
        propose an actual change, as opposed to HOLD/REJECT?"""
        return self.proposed_direction in _ACTIONABLE_DIRECTIONS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "subject": self.subject,
            "target_area": self.target_area,
            "current_value": self.current_value,
            "current_durability": self.current_durability,
            "proposed_direction": self.proposed_direction.value,
            "confidence": self.confidence,
            "evidence_strength": self.evidence_strength.value,
            "expected_benefit": self.expected_benefit,
            "risk_level": self.risk_level.value,
            "stability": self.stability,
            "reversible": self.reversible,
            "reason_codes": list(self.reason_codes),
            "source_decision_ids": list(self.source_decision_ids),
            "source_feedback_ids": list(self.source_feedback_ids),
            "cooldown_active": self.cooldown_active,
            "metadata": dict(self.metadata),
        }


# ======================================================================
# Internal evaluation context / strategy verdict
# ======================================================================

@dataclass(frozen=True)
class OptimizationContext:
    """
    Validated, immutable bundle of everything one `evaluate()` call
    needs. Built internally by `AdaptiveOptimizer.evaluate` -- callers
    do not need to construct this directly, but a strategy implementation
    receives exactly this.
    """

    subject: str
    area: AdaptationArea
    decision: Optional[AdaptationDecision]
    feedback_history: Tuple[FeedbackOutcomeRecord, ...]
    recent_decisions: Tuple[AdaptationDecision, ...]
    explicit_locked_fields: frozenset
    now: datetime


@dataclass(frozen=True)
class StrategyVerdict:
    """
    What an `OptimizationStrategy` thinks should happen, BEFORE the
    unconditional safety clamp, lock check, hysteresis, and cooldown
    logic in `AdaptiveOptimizer.evaluate` are applied. A future ML
    strategy only ever needs to produce this shape -- everything else is
    enforced centrally and cannot be bypassed.
    """

    direction: OptimizationDirection
    confidence: float
    evidence_strength: EvidenceStrengthTier
    expected_benefit: float
    risk_level: RiskLevel
    stability: float
    reason_codes: Tuple[str, ...]
    source_feedback_ids: Tuple[str, ...] = ()


# ======================================================================
# Strategy interface (so RL / bandit / statistical strategies can be
# swapped in without touching AdaptiveOptimizer's public surface)
# ======================================================================

class OptimizationStrategy(ABC):
    """
    Pluggable optimization strategy. `AdaptiveOptimizer` depends only on
    this interface. A future statistical, contextual-bandit,
    reinforcement-learning, or learned-ranking strategy can implement it
    and be swapped in via `AdaptiveOptimizerConfig.strategy` without
    changing `AdaptiveOptimizer`'s public API, the safety envelope it
    enforces afterward, or the `OptimizationRecommendation` contract.
    """

    @abstractmethod
    def evaluate(
        self,
        context: OptimizationContext,
        previous: Optional[OptimizationRecommendation],
    ) -> StrategyVerdict:
        raise NotImplementedError


@dataclass
class DeterministicStrategyConfig:
    """Tunable thresholds for `DeterministicOptimizationStrategy`. All
    values are plain, application-supplied configuration -- never
    derived from client input or learned state."""

    # Evidence-count thresholds for EvidenceStrengthTier classification.
    emerging_at: int = 2
    established_at: int = 4
    strong_at: int = 8

    # Effectiveness-score thresholds (score is in [-1, 1]).
    strong_positive_effectiveness: float = 0.6
    positive_effectiveness: float = 0.25
    negative_effectiveness: float = -0.25
    strong_negative_effectiveness: float = -0.6

    # Minimum evidence required before any actionable (non-HOLD)
    # recommendation may be produced at all -- mirrors adaptation_engine's
    # "sufficient evidence" gate (rule 4: never conclude from one datum).
    min_evidence_for_action: int = 2
    min_evidence_for_promote: int = 6

    # Over-adaptation detection (rule 7): number of reversals/direction
    # flips within `oscillation_window` recent decisions that triggers
    # STABILIZE/REDUCE instead of a further change.
    oscillation_window: int = 6
    max_reversals_before_stabilize: int = 2
    max_reversals_before_reduce: int = 3

    # Recency weighting: most-recent feedback record gets this multiplier
    # of the oldest one within the same evaluation (linear ramp).
    recency_weight_floor: float = 0.4


class DeterministicOptimizationStrategy(OptimizationStrategy):
    """
    The safe, always-available rule-based baseline strategy (rule 17).
    Deterministic and side-effect free: identical input always produces
    an identical `StrategyVerdict`.
    """

    def __init__(self, config: Optional[DeterministicStrategyConfig] = None) -> None:
        self.config = config or DeterministicStrategyConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        context: OptimizationContext,
        previous: Optional[OptimizationRecommendation],
    ) -> StrategyVerdict:
        cfg = self.config
        decision = context.decision
        area_name = _safe_area_name(context.area)

        # --- Rule 24 FAILURE SAFETY: missing required information -> HOLD.
        if decision is None:
            return StrategyVerdict(
                direction=OptimizationDirection.HOLD,
                confidence=0.0,
                evidence_strength=EvidenceStrengthTier.ANECDOTAL,
                expected_benefit=0.0,
                risk_level=RiskLevel.LOW,
                stability=1.0,
                reason_codes=("missing_adaptation_decision",),
            )

        # --- Rule 9/10/11/19 hard boundaries: a locked area or an
        # explicitly-locked field is never optimized against. This is
        # re-checked unconditionally in AdaptiveOptimizer.evaluate too
        # (defense in depth), but the strategy itself must never even
        # propose something that would need overriding.
        if decision.outcome == DecisionOutcome.HELD_AREA_LOCKED:
            return StrategyVerdict(
                direction=OptimizationDirection.HOLD,
                confidence=1.0,
                evidence_strength=EvidenceStrengthTier.ANECDOTAL,
                expected_benefit=0.0,
                risk_level=RiskLevel.LOW,
                stability=1.0,
                reason_codes=("area_locked_by_safety_envelope",),
            )
        if area_name in context.explicit_locked_fields:
            return StrategyVerdict(
                direction=OptimizationDirection.HOLD,
                confidence=1.0,
                evidence_strength=EvidenceStrengthTier.ANECDOTAL,
                expected_benefit=0.0,
                risk_level=RiskLevel.LOW,
                stability=1.0,
                reason_codes=("explicit_user_preference_locked",),
            )

        # --- Rule 24: an invalid/unsafe underlying decision -> nothing
        # actionable can be built on top of it.
        if decision.outcome == DecisionOutcome.HELD_INVALID_STATE:
            return StrategyVerdict(
                direction=OptimizationDirection.HOLD,
                confidence=0.0,
                evidence_strength=EvidenceStrengthTier.ANECDOTAL,
                expected_benefit=0.0,
                risk_level=RiskLevel.LOW,
                stability=0.5,
                reason_codes=("underlying_decision_invalid",),
            )

        effectiveness, evidence_count, feedback_ids = self._score_effectiveness(context)
        evidence_strength = self._evidence_tier(evidence_count)
        reversals = self._count_reversals(context)
        stability = self._stability_from_reversals(reversals)

        reason_codes: List[str] = []

        # --- Rule 16 contradiction check: decision itself flags a
        # contradiction in its own upstream evidence.
        has_contradiction = "held_contradiction" in decision.reason_codes or any(
            "contradiction" in code for code in decision.reason_codes
        )
        if has_contradiction:
            reason_codes.append("upstream_contradiction_detected")

        # --- Rule 7 over-adaptation detection: too much recent churn
        # takes priority over anything the fresh feedback score suggests
        # -- an oscillating dimension needs to be stabilized before more
        # evidence can even be trusted.
        if reversals >= cfg.max_reversals_before_reduce:
            reason_codes.append("over_adaptation_frequent_reversals")
            direction = (
                OptimizationDirection.REDUCE
                if effectiveness <= cfg.negative_effectiveness
                else OptimizationDirection.STABILIZE
            )
            return StrategyVerdict(
                direction=direction,
                confidence=_clamp01(0.4 + 0.1 * reversals),
                evidence_strength=evidence_strength,
                expected_benefit=_clamp01(0.3),
                risk_level=RiskLevel.MEDIUM,
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )
        if reversals >= cfg.max_reversals_before_stabilize:
            reason_codes.append("over_adaptation_repeated_reversals")
            return StrategyVerdict(
                direction=OptimizationDirection.STABILIZE,
                confidence=_clamp01(0.3 + 0.1 * reversals),
                evidence_strength=evidence_strength,
                expected_benefit=_clamp01(0.2),
                risk_level=RiskLevel.LOW,
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )

        # --- Rule 5: not enough non-UNKNOWN evidence to conclude
        # anything -- and a missing signal must never be treated as
        # failure. Stay at HOLD, evidence tier ANECDOTAL at worst.
        if evidence_count == 0:
            reason_codes.append("no_scoreable_feedback_evidence")
            return StrategyVerdict(
                direction=OptimizationDirection.HOLD,
                confidence=0.0,
                evidence_strength=EvidenceStrengthTier.ANECDOTAL,
                expected_benefit=0.0,
                risk_level=RiskLevel.LOW,
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )

        if evidence_count < cfg.min_evidence_for_action:
            reason_codes.append("insufficient_evidence_for_action")
            return StrategyVerdict(
                direction=OptimizationDirection.HOLD,
                confidence=_clamp01(0.15 * evidence_count),
                evidence_strength=evidence_strength,
                expected_benefit=0.0,
                risk_level=RiskLevel.LOW,
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )

        if has_contradiction:
            # Rule 16: contradictory signal with no way to safely
            # resolve it here -> HOLD, never guess.
            return StrategyVerdict(
                direction=OptimizationDirection.HOLD,
                confidence=_clamp01(0.2),
                evidence_strength=evidence_strength,
                expected_benefit=0.0,
                risk_level=RiskLevel.MEDIUM,
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )

        # --- Rule 8 under-adaptation detection: strong, stable, positive
        # evidence on a dimension that's still only tentatively adapted.
        if (
            decision.durability == Durability.TENTATIVE
            and effectiveness >= cfg.strong_positive_effectiveness
            and evidence_count >= cfg.min_evidence_for_promote
            and reversals == 0
        ):
            reason_codes.append("under_adaptation_strong_stable_evidence")
            return StrategyVerdict(
                direction=OptimizationDirection.PROMOTE_TO_STABLE,
                confidence=_clamp01(0.5 + 0.05 * evidence_count),
                evidence_strength=evidence_strength,
                expected_benefit=_clamp01(effectiveness),
                risk_level=self._risk_for(evidence_strength, decision.durability),
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )

        if (
            effectiveness >= cfg.positive_effectiveness
            and evidence_count >= cfg.min_evidence_for_action
            and decision.is_actionable
        ):
            reason_codes.append("positive_adaptation_effectiveness")
            direction = (
                OptimizationDirection.STRENGTHEN
                if decision.outcome == DecisionOutcome.PROPOSED
                else OptimizationDirection.ADAPT
            )
            return StrategyVerdict(
                direction=direction,
                confidence=_clamp01(0.35 + 0.08 * evidence_count),
                evidence_strength=evidence_strength,
                expected_benefit=_clamp01(effectiveness),
                risk_level=self._risk_for(evidence_strength, decision.durability),
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )

        if effectiveness <= cfg.strong_negative_effectiveness and evidence_count >= cfg.min_evidence_for_action:
            reason_codes.append("strong_negative_adaptation_effectiveness")
            return StrategyVerdict(
                direction=OptimizationDirection.REDUCE,
                confidence=_clamp01(0.4 + 0.08 * evidence_count),
                evidence_strength=evidence_strength,
                expected_benefit=_clamp01(-effectiveness),
                risk_level=self._risk_for(evidence_strength, decision.durability),
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )

        if effectiveness <= cfg.negative_effectiveness and evidence_count >= cfg.min_evidence_for_action:
            reason_codes.append("mild_negative_adaptation_effectiveness")
            return StrategyVerdict(
                direction=OptimizationDirection.STABILIZE,
                confidence=_clamp01(0.25 + 0.05 * evidence_count),
                evidence_strength=evidence_strength,
                expected_benefit=_clamp01(0.15),
                risk_level=RiskLevel.LOW,
                stability=stability,
                reason_codes=tuple(reason_codes),
                source_feedback_ids=feedback_ids,
            )

        # --- Neutral zone: evidence exists but doesn't clearly justify
        # a change either way. Rule 13 hysteresis: this IS the stable
        # zone small fluctuations should land in.
        reason_codes.append("neutral_zone_no_clear_signal")
        return StrategyVerdict(
            direction=OptimizationDirection.HOLD,
            confidence=_clamp01(0.2 + 0.03 * evidence_count),
            evidence_strength=evidence_strength,
            expected_benefit=0.0,
            risk_level=RiskLevel.LOW,
            stability=stability,
            reason_codes=tuple(reason_codes),
            source_feedback_ids=feedback_ids,
        )

    # ------------------------------------------------------------------
    # Internal scoring helpers (pure, unit-testable in isolation)
    # ------------------------------------------------------------------

    def _score_effectiveness(
        self, context: OptimizationContext
    ) -> Tuple[float, int, Tuple[str, ...]]:
        """
        Rule 4/15: fold the feedback-outcome history for this area into
        a single effectiveness score in [-1, 1], weighted by each
        record's own `weight` and by recency (more recent evidence
        counts more), never by "latest event wins" alone. UNKNOWN
        records are excluded entirely -- they neither help nor punish.
        """
        area_name = _safe_area_name(context.area)
        relevant = [r for r in context.feedback_history if r.area == area_name and r.outcome != FeedbackOutcomeSignal.UNKNOWN]
        if not relevant:
            return 0.0, 0, ()

        ordered = sorted(relevant, key=lambda r: r.observed_at)
        n = len(ordered)
        floor = self.config.recency_weight_floor
        total_weight = 0.0
        total_score = 0.0
        for index, record in enumerate(ordered):
            recency_factor = floor if n == 1 else floor + (1.0 - floor) * (index / (n - 1))
            effective_weight = record.weight * recency_factor
            total_weight += effective_weight
            total_score += effective_weight * _OUTCOME_WEIGHT[record.outcome]

        if total_weight <= 0.0:
            return 0.0, len(ordered), tuple(r.record_id for r in ordered)

        score = _clamp(total_score / total_weight, -1.0, 1.0)
        return score, len(ordered), tuple(r.record_id for r in ordered)

    def _evidence_tier(self, evidence_count: int) -> EvidenceStrengthTier:
        cfg = self.config
        if evidence_count >= cfg.strong_at:
            return EvidenceStrengthTier.STRONG
        if evidence_count >= cfg.established_at:
            return EvidenceStrengthTier.ESTABLISHED
        if evidence_count >= cfg.emerging_at:
            return EvidenceStrengthTier.EMERGING
        return EvidenceStrengthTier.ANECDOTAL

    def _count_reversals(self, context: OptimizationContext) -> int:
        """
        Rule 7: count explicit reversal flags and outcome-direction
        flips within the most recent `oscillation_window` decisions
        supplied by the caller, oldest-first-in/most-recent-last
        ordering assumed. This never inspects global state -- only
        exactly what the caller passed for this evaluation.
        """
        window = list(context.recent_decisions)[-self.config.oscillation_window :]
        reversals = sum(1 for d in window if getattr(d, "is_reversal", False))
        signs = [1 if d.delta > 1e-9 else (-1 if d.delta < -1e-9 else 0) for d in window if _is_finite_real(getattr(d, "delta", 0.0))]
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != 0 and b != 0 and a != b)
        return max(reversals, flips)

    @staticmethod
    def _stability_from_reversals(reversals: int) -> float:
        return _clamp01(1.0 - 0.2 * reversals)

    @staticmethod
    def _risk_for(evidence_strength: EvidenceStrengthTier, durability: Durability) -> RiskLevel:
        """Rule 6: high-risk changes require stronger evidence. Risk here
        reflects how much confidence 4F itself should require, not how
        large the underlying magnitude is (that bound is
        adaptation_engine's job via AreaSafetyEnvelope)."""
        tier_rank = _EVIDENCE_TIER_ORDER[evidence_strength]
        if durability == Durability.EXPLICIT:
            return RiskLevel.LOW
        if tier_rank >= _EVIDENCE_TIER_ORDER[EvidenceStrengthTier.STRONG]:
            return RiskLevel.LOW
        if tier_rank >= _EVIDENCE_TIER_ORDER[EvidenceStrengthTier.ESTABLISHED]:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH


# ======================================================================
# AdaptiveOptimizer — public orchestration layer
# ======================================================================

@dataclass
class AdaptiveOptimizerConfig:
    """Application-supplied configuration. Never populated from client
    input or learned state -- mirrors `AdaptationEngineConfig` /
    `AdaptivePolicyConfig` in the sibling Phase 4 modules."""

    strategy: Optional[OptimizationStrategy] = None
    strategy_config: Optional[DeterministicStrategyConfig] = None

    # Rule 14 cooldown: minimum time between two actionable (non-HOLD,
    # non-REJECT) recommendations for the same (subject, area).
    cooldown_seconds: float = 900.0

    # Rule 13 hysteresis: a direction reversal relative to the previous
    # actionable recommendation is only honored if at least this much
    # wall-clock time has passed AND the new confidence clears
    # `hysteresis_min_confidence` -- otherwise it is downgraded to HOLD.
    hysteresis_seconds: float = 300.0
    hysteresis_min_confidence: float = 0.5

    # Bounded, instance-scoped bookkeeping cap (never a global cache).
    max_tracked_keys: int = 10_000

    # Optional structured observability hook (rule 29). Never receives
    # secrets -- only the event name and the recommendation's own
    # to_dict()-shaped payload.
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None


class AdaptiveOptimizer:
    """
    Phase 4F entry point.

    Public API
    ----------
    evaluate(...)        -> one OptimizationRecommendation for one
                             (subject, area) pair.
    evaluate_many(...)   -> OptimizationRecommendation per item, with
                             same-key conflicts resolved (rule 16).
    get_current(...)     -> last recommendation for (subject, area).
    explain(...)         -> that recommendation as a plain dict.

    Guarantees enforced HERE (never delegated to the strategy, so a
    future ML strategy cannot bypass them):
      - a locked area (`DecisionOutcome.HELD_AREA_LOCKED`) or an
        explicitly-locked field NEVER receives an actionable
        recommendation -- HOLD only, regardless of strategy confidence.
      - a `Durability.EXPLICIT` decision NEVER receives a REDUCE/
        STABILIZE recommendation (rule 11) -- downgraded to HOLD.
      - every numeric output field is clamped to a finite, bounded
        range (rule 19) -- NaN/Infinity/out-of-range values can never
        leave this module.
      - hysteresis (rule 13): a fresh reversal of direction relative to
        the last actionable recommendation is only honored once enough
        time has passed and confidence is high enough; otherwise HOLD.
      - cooldown (rule 14): no two actionable recommendations for the
        same (subject, area) within `cooldown_seconds` of each other.
      - any unexpected internal error is caught and mapped to a REJECT
        recommendation with a diagnosable reason code -- this module
        never raises out of `evaluate()` and never fabricates an
        actionable recommendation merely to avoid returning nothing
        (rule 24).
    """

    def __init__(self, config: Optional[AdaptiveOptimizerConfig] = None) -> None:
        self.config = config or AdaptiveOptimizerConfig()
        self._strategy: OptimizationStrategy = self.config.strategy or DeterministicOptimizationStrategy(
            self.config.strategy_config
        )
        self._lock = threading.Lock()
        # Instance-scoped, size-bounded bookkeeping only (rule: no hidden
        # global mutable state). Keyed by (subject, area value).
        self._last_recommendation: "OrderedDict[Tuple[str, str], OptimizationRecommendation]" = OrderedDict()
        self._last_actionable_at: "OrderedDict[Tuple[str, str], datetime]" = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        subject: str,
        area: AdaptationArea,
        decision: Optional[AdaptationDecision],
        feedback_history: Sequence[FeedbackOutcomeRecord] = (),
        recent_decisions: Sequence[AdaptationDecision] = (),
        explicit_locked_fields: Sequence[str] = (),
        now: Optional[datetime] = None,
    ) -> OptimizationRecommendation:
        """Evaluate one (subject, area) pair and return exactly one
        `OptimizationRecommendation`. Never raises for ordinary bad
        input -- see class docstring."""
        eval_time = now or _now()
        try:
            context = self._build_context(
                subject, area, decision, feedback_history, recent_decisions, explicit_locked_fields, eval_time
            )
        except OptimizerValidationError as exc:
            recommendation = self._reject(subject, area, eval_time, f"invalid_input:{exc}")
            self._emit("optimization_rejected", recommendation)
            return recommendation
        except Exception as exc:  # pragma: no cover - defensive, keeps this module diagnosable
            recommendation = self._reject(subject, area, eval_time, f"internal_error:{type(exc).__name__}")
            self._emit("optimization_rejected", recommendation)
            return recommendation

        key = (subject, _safe_area_name(area))
        previous = self._peek_previous(key)

        try:
            verdict = self._strategy.evaluate(context, previous)
        except Exception as exc:  # pragma: no cover - a strategy must never crash the pipeline
            recommendation = self._reject(subject, area, eval_time, f"strategy_error:{type(exc).__name__}")
            self._emit("optimization_rejected", recommendation)
            return recommendation

        recommendation = self._finalize(context, verdict, previous, key)
        self._store(key, recommendation)
        self._emit("optimization_evaluated", recommendation)
        if recommendation.proposed_direction == OptimizationDirection.REJECT:
            self._emit("optimization_rejected", recommendation)
        elif recommendation.proposed_direction == OptimizationDirection.HOLD:
            self._emit("optimization_held", recommendation)
        else:
            self._emit("optimization_applied", recommendation)
        return recommendation

    def evaluate_many(
        self,
        items: Sequence[
            Tuple[str, AdaptationArea, Optional[AdaptationDecision], Sequence[FeedbackOutcomeRecord], Sequence[AdaptationDecision], Sequence[str]]
        ],
        now: Optional[datetime] = None,
    ) -> List[OptimizationRecommendation]:
        """
        Evaluate several (subject, area, decision, feedback_history,
        recent_decisions, explicit_locked_fields) tuples. If more than
        one item resolves to the same (subject, area) key, rule 16's
        conflict-resolution order is applied and only the winner is
        stored as the current recommendation for that key -- every
        other candidate for that key is still returned to the caller,
        but with `reason_codes` annotated to show it was superseded.
        """
        eval_time = now or _now()
        by_key: "OrderedDict[Tuple[str, str], List[OptimizationRecommendation]]" = OrderedDict()
        order: List[Tuple[str, str]] = []
        for subject, area, decision, feedback_history, recent_decisions, locked in items:
            rec = self.evaluate(subject, area, decision, feedback_history, recent_decisions, locked, eval_time)
            key = (subject, _safe_area_name(area))
            by_key.setdefault(key, [])
            if key not in order:
                order.append(key)
            by_key[key].append(rec)

        results: List[OptimizationRecommendation] = []
        for key in order:
            candidates = by_key[key]
            if len(candidates) == 1:
                results.append(candidates[0])
                continue
            winner = self.resolve_conflicts(candidates)
            self._store(key, winner)
            for candidate in candidates:
                if candidate.recommendation_id == winner.recommendation_id:
                    results.append(winner)
                else:
                    superseded = self._supersede(candidate, winner)
                    results.append(superseded)
        return results

    def get_current(self, subject: str, area: AdaptationArea) -> Optional[OptimizationRecommendation]:
        with self._lock:
            return self._last_recommendation.get((subject, _safe_area_name(area)))

    def explain(self, subject: str, area: AdaptationArea) -> Dict[str, Any]:
        current = self.get_current(subject, area)
        if current is None:
            return {"subject": subject, "area": _safe_area_name(area), "status": "no_recommendation_yet"}
        return current.to_dict()

    # ------------------------------------------------------------------
    # Rule 16: conflict resolution across recommendations for one key
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_conflicts(candidates: Sequence[OptimizationRecommendation]) -> OptimizationRecommendation:
        """
        Deterministically pick exactly one winner among recommendations
        that target the same (subject, area), in the fixed order:
          1. hard constraints (HOLD from a lock always wins outright)
          2. explicit preference protection (HOLD from explicit-lock)
          3. highest durability tier implied by the recommendation
             (approximated here via risk_level: LOW risk from an
             EXPLICIT-durability decision ranks above others)
          4. strongest evidence (`evidence_strength`)
          5. highest confidence
          6. highest expected_benefit ("proven effectiveness")
          7. most recent `generated_at` (pure tie-breaker)
        If every criterion ties exactly, return a HOLD recommendation
        referencing every tied candidate rather than guessing (rule 16:
        "If conflict remains unresolved: RETURN HOLD. Never guess.").
        """
        if not candidates:
            raise OptimizerValidationError("resolve_conflicts requires at least one candidate")
        if len(candidates) == 1:
            return candidates[0]

        hard_locks = [c for c in candidates if "area_locked_by_safety_envelope" in c.reason_codes]
        if hard_locks:
            return hard_locks[0]
        explicit_locks = [c for c in candidates if "explicit_user_preference_locked" in c.reason_codes]
        if explicit_locks:
            return explicit_locks[0]

        def sort_key(rec: OptimizationRecommendation) -> Tuple[int, int, float, float, float]:
            return (
                _EVIDENCE_TIER_ORDER[rec.evidence_strength],
                -_RISK_ORDER[rec.risk_level],  # lower risk ranks higher
                rec.confidence,
                rec.expected_benefit,
                rec.generated_at.timestamp(),
            )

        ranked = sorted(candidates, key=sort_key, reverse=True)
        best, runner_up = ranked[0], ranked[1]
        if sort_key(best) == sort_key(runner_up):
            tied_ids = tuple(c.recommendation_id for c in candidates)
            return OptimizationRecommendation(
                recommendation_id=_new_id(),
                generated_at=best.generated_at,
                subject=best.subject,
                target_area=best.target_area,
                current_value=best.current_value,
                current_durability=best.current_durability,
                proposed_direction=OptimizationDirection.HOLD,
                confidence=0.0,
                evidence_strength=best.evidence_strength,
                expected_benefit=0.0,
                risk_level=RiskLevel.LOW,
                stability=min(c.stability for c in candidates),
                reversible=True,
                reason_codes=("unresolved_conflict_between_recommendations",),
                source_decision_ids=tuple(sorted(set(sum((c.source_decision_ids for c in candidates), ())))),
                source_feedback_ids=tuple(sorted(set(sum((c.source_feedback_ids for c in candidates), ())))),
                metadata={"tied_recommendation_ids": list(tied_ids)},
            )
        return best

    @staticmethod
    def _supersede(candidate: OptimizationRecommendation, winner: OptimizationRecommendation) -> OptimizationRecommendation:
        if candidate.recommendation_id == winner.recommendation_id:
            return candidate
        reason_codes = candidate.reason_codes + (f"superseded_by:{winner.recommendation_id}",)
        return OptimizationRecommendation(
            recommendation_id=candidate.recommendation_id,
            generated_at=candidate.generated_at,
            subject=candidate.subject,
            target_area=candidate.target_area,
            current_value=candidate.current_value,
            current_durability=candidate.current_durability,
            proposed_direction=OptimizationDirection.HOLD,
            confidence=candidate.confidence,
            evidence_strength=candidate.evidence_strength,
            expected_benefit=candidate.expected_benefit,
            risk_level=candidate.risk_level,
            stability=candidate.stability,
            reversible=candidate.reversible,
            reason_codes=reason_codes,
            source_decision_ids=candidate.source_decision_ids,
            source_feedback_ids=candidate.source_feedback_ids,
            cooldown_active=candidate.cooldown_active,
            metadata=dict(candidate.metadata),
        )

    # ------------------------------------------------------------------
    # Internal: validation / context construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        subject: Any,
        area: Any,
        decision: Optional[AdaptationDecision],
        feedback_history: Sequence[FeedbackOutcomeRecord],
        recent_decisions: Sequence[AdaptationDecision],
        explicit_locked_fields: Sequence[str],
        now: datetime,
    ) -> OptimizationContext:
        if not isinstance(subject, str) or not subject:
            raise OptimizerValidationError("subject must be a non-empty string")
        if not isinstance(area, AdaptationArea):
            raise OptimizerValidationError(f"area must be an AdaptationArea, got {area!r}")
        if decision is not None:
            if not isinstance(decision, AdaptationDecision):
                raise OptimizerValidationError("decision must be an AdaptationDecision or None")
            for numeric_name in ("previous_value", "proposed_value", "delta", "confidence"):
                value = getattr(decision, numeric_name, 0.0)
                if not _is_finite_real(value):
                    raise OptimizerValidationError(f"decision.{numeric_name} must be finite, got {value!r}")
            evidence_count = getattr(decision, "evidence_count", 0)
            if isinstance(evidence_count, bool) or not isinstance(evidence_count, int) or evidence_count < 0:
                raise OptimizerValidationError("decision.evidence_count must be a non-negative int")

        for record in feedback_history:
            if not isinstance(record, FeedbackOutcomeRecord):
                raise OptimizerValidationError("feedback_history entries must be FeedbackOutcomeRecord")
        for past in recent_decisions:
            if not isinstance(past, AdaptationDecision):
                raise OptimizerValidationError("recent_decisions entries must be AdaptationDecision")

        locked = frozenset(str(f) for f in explicit_locked_fields)

        if not isinstance(now, datetime):
            raise OptimizerValidationError("now must be a datetime")
        eval_time = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)

        return OptimizationContext(
            subject=subject,
            area=area,
            decision=decision,
            feedback_history=tuple(feedback_history),
            recent_decisions=tuple(recent_decisions),
            explicit_locked_fields=locked,
            now=eval_time.astimezone(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Internal: finalize a strategy verdict into a safe recommendation
    # ------------------------------------------------------------------

    def _finalize(
        self,
        context: OptimizationContext,
        verdict: StrategyVerdict,
        previous: Optional[OptimizationRecommendation],
        key: Tuple[str, str],
    ) -> OptimizationRecommendation:
        decision = context.decision
        reason_codes = list(verdict.reason_codes)
        direction = verdict.direction

        # --- Rule 11 (defense in depth): an EXPLICIT-durability decision
        # can never be optimized AGAINST, no matter what the strategy
        # computed. This check is unconditional and cannot be bypassed
        # by a future ML strategy's confidence.
        if decision is not None and decision.durability == Durability.EXPLICIT and direction in _AGAINST_PREFERENCE_DIRECTIONS:
            direction = OptimizationDirection.HOLD
            reason_codes.append("explicit_preference_protected_from_reduction")

        # --- Rule 13 hysteresis / anti-oscillation: a fresh reversal
        # relative to the last actionable recommendation for this key is
        # only honored once enough time has passed and confidence clears
        # the bar; otherwise it is downgraded to HOLD so minor evidence
        # fluctuations cannot cause A -> B -> A -> B churn.
        cooldown_active = False
        if previous is not None and direction in _ACTIONABLE_DIRECTIONS and previous.proposed_direction in _ACTIONABLE_DIRECTIONS:
            reversed_direction = self._is_reversal(previous.proposed_direction, direction)
            elapsed = (context.now - previous.generated_at).total_seconds()
            if reversed_direction and (
                elapsed < self.config.hysteresis_seconds or verdict.confidence < self.config.hysteresis_min_confidence
            ):
                direction = OptimizationDirection.HOLD
                reason_codes.append("hysteresis_guard_recent_reversal")

        # --- Rule 14 cooldown: no two actionable recommendations for the
        # same key within cooldown_seconds of each other.
        if direction in _ACTIONABLE_DIRECTIONS:
            last_actionable_at = self._peek_last_actionable(key)
            if last_actionable_at is not None:
                elapsed = (context.now - last_actionable_at).total_seconds()
                if elapsed < self.config.cooldown_seconds:
                    cooldown_active = True
                    reason_codes.append("cooldown_active")
                    direction = OptimizationDirection.HOLD

        current_value = getattr(decision, "proposed_value", 0.0) if decision is not None else 0.0
        current_durability = (
            decision.durability.value if decision is not None and isinstance(decision.durability, Enum) else "unknown"
        )
        source_decision_ids = tuple(
            d.decision_id for d in ((decision,) if decision is not None else ()) if getattr(d, "decision_id", None)
        )

        return OptimizationRecommendation(
            recommendation_id=_new_id(),
            generated_at=context.now,
            subject=context.subject,
            target_area=_safe_area_name(context.area),
            current_value=_clamp(current_value, -1e6, 1e6) if _is_finite_real(current_value) else 0.0,
            current_durability=current_durability,
            proposed_direction=direction,
            confidence=verdict.confidence,
            evidence_strength=verdict.evidence_strength,
            expected_benefit=verdict.expected_benefit,
            risk_level=verdict.risk_level,
            stability=verdict.stability,
            reversible=True,
            reason_codes=tuple(reason_codes),
            source_decision_ids=source_decision_ids,
            source_feedback_ids=verdict.source_feedback_ids,
            cooldown_active=cooldown_active,
        )

    @staticmethod
    def _is_reversal(previous_direction: OptimizationDirection, new_direction: OptimizationDirection) -> bool:
        growth = {OptimizationDirection.ADAPT, OptimizationDirection.STRENGTHEN, OptimizationDirection.PROMOTE_TO_STABLE}
        shrink = {OptimizationDirection.REDUCE, OptimizationDirection.STABILIZE}
        return (previous_direction in growth and new_direction in shrink) or (
            previous_direction in shrink and new_direction in growth
        )

    def _reject(self, subject: Any, area: Any, now: datetime, reason: str) -> OptimizationRecommendation:
        subject_str = subject if isinstance(subject, str) and subject else "unknown"
        area_str = _safe_area_name(area) if area is not None else "unknown"
        return OptimizationRecommendation(
            recommendation_id=_new_id(),
            generated_at=now,
            subject=subject_str,
            target_area=area_str,
            current_value=0.0,
            current_durability="unknown",
            proposed_direction=OptimizationDirection.REJECT,
            confidence=0.0,
            evidence_strength=EvidenceStrengthTier.ANECDOTAL,
            expected_benefit=0.0,
            risk_level=RiskLevel.HIGH,
            stability=0.0,
            reversible=True,
            reason_codes=(reason,),
            source_decision_ids=(),
            source_feedback_ids=(),
        )

    # ------------------------------------------------------------------
    # Internal: bounded, instance-scoped bookkeeping
    # ------------------------------------------------------------------

    def _peek_previous(self, key: Tuple[str, str]) -> Optional[OptimizationRecommendation]:
        with self._lock:
            rec = self._last_recommendation.get(key)
            if rec is not None:
                self._last_recommendation.move_to_end(key)
            return rec

    def _peek_last_actionable(self, key: Tuple[str, str]) -> Optional[datetime]:
        with self._lock:
            at = self._last_actionable_at.get(key)
            if at is not None:
                self._last_actionable_at.move_to_end(key)
            return at

    def _store(self, key: Tuple[str, str], recommendation: OptimizationRecommendation) -> None:
        with self._lock:
            self._last_recommendation[key] = recommendation
            self._last_recommendation.move_to_end(key)
            if recommendation.proposed_direction in _ACTIONABLE_DIRECTIONS:
                self._last_actionable_at[key] = recommendation.generated_at
                self._last_actionable_at.move_to_end(key)

            max_keys = max(1, self.config.max_tracked_keys)
            while len(self._last_recommendation) > max_keys:
                self._last_recommendation.popitem(last=False)
            while len(self._last_actionable_at) > max_keys:
                self._last_actionable_at.popitem(last=False)

    def _emit(self, event: str, recommendation: OptimizationRecommendation) -> None:
        """Rule 29: concise structured diagnostics only, via an
        application-supplied hook. Never logs secrets -- the payload is
        exactly `OptimizationRecommendation.to_dict()`, which by
        construction has no field capable of carrying one."""
        if self.config.on_event is None:
            return
        try:
            self.config.on_event(event, recommendation.to_dict())
        except Exception:  # pragma: no cover - a logging hook must never break the pipeline
            pass


# ======================================================================
# Package boundary integration
# ======================================================================

def get_adaptive_optimizer(config: Optional[AdaptiveOptimizerConfig] = None) -> AdaptiveOptimizer:
    """
    Public factory, mirroring `get_adaptation_engine` /
    `get_adaptive_policy` for the package `__init__.py`'s lazy
    component resolution. Returns a fresh `AdaptiveOptimizer`; hold your
    own instance directly if you need recommendation-history continuity
    (hysteresis/cooldown) across calls beyond what a single shared
    instance already provides.
    """
    return AdaptiveOptimizer(config)


# ======================================================================
# Deterministic self-tests (rule 28 TESTABILITY). Exercised via
# `python adaptive_optimizer.py`; not a pytest suite, but pure enough
# to be wrapped in one trivially.
# ======================================================================

def _run_self_tests() -> None:  # pragma: no cover - exercised via __main__
    failures: List[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            failures.append(f"{name}: {detail}")

    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def make_decision(
        outcome: DecisionOutcome = DecisionOutcome.PROPOSED,
        durability: Durability = Durability.TENTATIVE,
        evidence_count: int = 5,
        confidence: float = 0.6,
        is_reversal: bool = False,
        delta: float = 0.1,
        reason_codes: Tuple[str, ...] = (),
    ) -> AdaptationDecision:
        return AdaptationDecision(
            decision_id=f"adec_{uuid.uuid4().hex[:8]}",
            subject="user_1",
            area=AdaptationArea.RESPONSE_LENGTH,
            outcome=outcome,
            previous_value=0.0,
            proposed_value=0.2,
            delta=delta,
            confidence=confidence,
            evidence_count=evidence_count,
            durability=durability,
            is_reversal=is_reversal,
            reason_codes=reason_codes,
            provenance=(),
            generated_at=base_time,
        )

    def make_feedback(outcome: FeedbackOutcomeSignal, count: int, start_minute: int = 0) -> List[FeedbackOutcomeRecord]:
        return [
            FeedbackOutcomeRecord(
                record_id=f"fb_{uuid.uuid4().hex[:8]}",
                area=AdaptationArea.RESPONSE_LENGTH.value,
                outcome=outcome,
                observed_at=base_time.replace(minute=(start_minute + i) % 59),
            )
            for i in range(count)
        ]

    # 1. Strong successful adaptation -> PROMOTE_TO_STABLE.
    optimizer = AdaptiveOptimizer()
    decision = make_decision(evidence_count=8, confidence=0.8)
    feedback = make_feedback(FeedbackOutcomeSignal.SUCCESS, 8)
    rec = optimizer.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision, feedback, [], [], base_time)
    check("strong_success", rec.proposed_direction == OptimizationDirection.PROMOTE_TO_STABLE, rec.proposed_direction)

    # 2. Weak evidence -> HOLD.
    optimizer2 = AdaptiveOptimizer()
    decision2 = make_decision(evidence_count=1)
    feedback2 = make_feedback(FeedbackOutcomeSignal.SUCCESS, 1)
    rec2 = optimizer2.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision2, feedback2, [], [], base_time)
    check("weak_evidence_holds", rec2.proposed_direction == OptimizationDirection.HOLD, rec2.proposed_direction)

    # 3. Repeated failure -> REDUCE.
    optimizer3 = AdaptiveOptimizer()
    decision3 = make_decision(evidence_count=6, outcome=DecisionOutcome.PROPOSED)
    feedback3 = make_feedback(FeedbackOutcomeSignal.FAILURE, 6)
    rec3 = optimizer3.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision3, feedback3, [], [], base_time)
    check("repeated_failure_reduces", rec3.proposed_direction == OptimizationDirection.REDUCE, rec3.proposed_direction)

    # 4. Unknown feedback must not be treated as failure.
    optimizer4 = AdaptiveOptimizer()
    decision4 = make_decision(evidence_count=3)
    feedback4 = make_feedback(FeedbackOutcomeSignal.UNKNOWN, 5)
    rec4 = optimizer4.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision4, feedback4, [], [], base_time)
    check(
        "unknown_not_punished",
        rec4.proposed_direction == OptimizationDirection.HOLD and "no_scoreable_feedback_evidence" in rec4.reason_codes,
        rec4.reason_codes,
    )

    # 5. Contradictory evidence -> HOLD, not a guess.
    optimizer5 = AdaptiveOptimizer()
    decision5 = make_decision(evidence_count=6, reason_codes=("held_contradiction",))
    feedback5 = make_feedback(FeedbackOutcomeSignal.SUCCESS, 3) + make_feedback(FeedbackOutcomeSignal.FAILURE, 3, start_minute=10)
    rec5 = optimizer5.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision5, feedback5, [], [], base_time)
    check("contradiction_holds", rec5.proposed_direction == OptimizationDirection.HOLD, rec5.proposed_direction)

    # 6. Over-adaptation: frequent reversals -> STABILIZE/REDUCE, never a fresh change.
    optimizer6 = AdaptiveOptimizer()
    decision6 = make_decision(evidence_count=5)
    history = [make_decision(is_reversal=True) for _ in range(4)]
    feedback6 = make_feedback(FeedbackOutcomeSignal.SUCCESS, 5)
    rec6 = optimizer6.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision6, feedback6, history, [], base_time)
    check(
        "over_adaptation_detected",
        rec6.proposed_direction in (OptimizationDirection.STABILIZE, OptimizationDirection.REDUCE),
        rec6.proposed_direction,
    )

    # 7. Locked area -> always HOLD regardless of evidence.
    optimizer7 = AdaptiveOptimizer()
    decision7 = make_decision(outcome=DecisionOutcome.HELD_AREA_LOCKED, evidence_count=9)
    feedback7 = make_feedback(FeedbackOutcomeSignal.SUCCESS, 9)
    rec7 = optimizer7.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision7, feedback7, [], [], base_time)
    check("locked_area_holds", rec7.proposed_direction == OptimizationDirection.HOLD, rec7.proposed_direction)

    # 8. Explicit preference protection: EXPLICIT durability + failing
    # feedback must never yield REDUCE/STABILIZE.
    optimizer8 = AdaptiveOptimizer()
    decision8 = make_decision(durability=Durability.EXPLICIT, evidence_count=6)
    feedback8 = make_feedback(FeedbackOutcomeSignal.FAILURE, 6)
    rec8 = optimizer8.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision8, feedback8, [], [], base_time)
    check(
        "explicit_preference_protected",
        rec8.proposed_direction not in _AGAINST_PREFERENCE_DIRECTIONS,
        rec8.proposed_direction,
    )

    # 9. Invalid numeric values (NaN) -> REJECT, never propagate.
    optimizer9 = AdaptiveOptimizer()
    try:
        bad_decision = AdaptationDecision(
            decision_id="adec_bad",
            subject="user_1",
            area=AdaptationArea.RESPONSE_LENGTH,
            outcome=DecisionOutcome.PROPOSED,
            previous_value=0.0,
            proposed_value=float("nan"),
            delta=0.0,
            confidence=0.5,
            evidence_count=3,
            durability=Durability.TENTATIVE,
            is_reversal=False,
            reason_codes=(),
            provenance=(),
            generated_at=base_time,
        )
        rec9 = optimizer9.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, bad_decision, [], [], [], base_time)
        check("nan_rejected", rec9.proposed_direction == OptimizationDirection.REJECT, rec9.proposed_direction)
    except Exception as exc:  # AdaptationDecision's own contract may reject NaN at construction
        check("nan_rejected_at_construction", True, str(exc))

    # 10. Missing state -> HOLD, not REJECT (it's an expected input shape).
    optimizer10 = AdaptiveOptimizer()
    rec10 = optimizer10.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, None, [], [], [], base_time)
    check("missing_state_holds", rec10.proposed_direction == OptimizationDirection.HOLD, rec10.proposed_direction)

    # 11. Explicit locked field -> HOLD even with perfect evidence.
    optimizer11 = AdaptiveOptimizer()
    decision11 = make_decision(evidence_count=9)
    feedback11 = make_feedback(FeedbackOutcomeSignal.SUCCESS, 9)
    rec11 = optimizer11.evaluate(
        "user_1", AdaptationArea.RESPONSE_LENGTH, decision11, feedback11, [], ["response_length"], base_time
    )
    check("explicit_locked_field_holds", rec11.proposed_direction == OptimizationDirection.HOLD, rec11.proposed_direction)

    # 12. Cooldown: a second actionable recommendation shortly after the
    # first is held.
    optimizer12 = AdaptiveOptimizer(AdaptiveOptimizerConfig(cooldown_seconds=3600))
    decision12 = make_decision(evidence_count=8, confidence=0.8)
    feedback12 = make_feedback(FeedbackOutcomeSignal.SUCCESS, 8)
    first = optimizer12.evaluate("user_1", AdaptationArea.RESPONSE_LENGTH, decision12, feedback12, [], [], base_time)
    second = optimizer12.evaluate(
        "user_1", AdaptationArea.RESPONSE_LENGTH, decision12, feedback12, [], [], base_time.replace(minute=1)
    )
    check(
        "cooldown_blocks_second_change",
        first.is_actionable and second.proposed_direction == OptimizationDirection.HOLD and second.cooldown_active,
        (first.proposed_direction, second.proposed_direction, second.cooldown_active),
    )

    # 13. Conflicting recommendations for the same key resolve to one
    # winner (or HOLD if truly tied), never raise.
    rec_a = rec  # PROMOTE_TO_STABLE from test 1, strongest evidence
    rec_b = rec2  # HOLD from test 2, weak evidence
    winner = AdaptiveOptimizer.resolve_conflicts([rec_a, rec_b])
    check("conflict_resolution_prefers_stronger_evidence", winner.recommendation_id == rec_a.recommendation_id, winner.target_area)

    if failures:
        raise AssertionError("adaptive_optimizer self-tests failed:\n" + "\n".join(failures))


if __name__ == "__main__":  # pragma: no cover
    _run_self_tests()
    print("adaptive_optimizer.py: all self-tests passed")
