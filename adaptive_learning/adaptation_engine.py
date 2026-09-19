"""
Adaptive Intelligence & Learning / adaptation_engine.py
=========================================================

Role
----
Converts a validated `LearningState` (produced by learning_engine.py) into
bounded, explainable `AdaptationDecision` objects that `adaptive_policy.py`
may choose to apply.

PIPELINE
--------
    feedback_engine.py
            |
            v
    LearningSignal
            |
            v
    learning_engine.py
            |
            v
    LearningState
            |
            v
    adaptation_engine.py     <-- THIS MODULE
            |
            v
    AdaptationDecision
            |
            v
    adaptive_policy.py

This module ONLY converts learned state into bounded adaptation
decisions. It does NOT:
  - directly apply those decisions (that's adaptive_policy.py's job)
  - mutate user_profile.py, personality_engine.py, or response_policy.py
  - accept arbitrary client-supplied adaptation commands
  - touch the database, an AI provider, or the network
  - write files or access secrets
  - hold unbounded global mutable state

Contract note
--------------
This module expects `LearningState` / `DimensionState` / `EvidenceRecord` /
`SignalType` to come from `learning_engine.py` (this engine's canonical
working representation, itself bridged from `learning_models.py`'s
cross-phase contract layer). Those exact classes are imported
defensively: if `learning_engine` isn't importable, or its shapes
differ, a structurally-compatible fallback is used so this file still
runs standalone. This preserves the existing, already-required internal
working representation rather than inventing a duplicate contract —
everything below relies on attribute access (subject, dimensions,
net_score, confidence, evidence_count, recent_history, contradiction_flag,
stability_score, signal_type, source, timestamp, signal_id), not on any
particular class identity.

SECURITY
---------
All `LearningState` input is treated as potentially untrusted (it may
have been deserialized from storage, or produced by a future learning
strategy). Before any decision logic runs, every numeric field this
module reads (`net_score`, `confidence`, `stability_score`,
`evidence_count`) is validated to be a finite, in-range real number, and
`dimension`/`subject` identifiers are validated to be non-empty strings.
A dimension state that fails validation never reaches the policy layer —
it is held (`HELD_INVALID_STATE`) at its last known-good value instead,
so malformed or corrupted state can never silently produce a dangerous
adaptation.

IDEMPOTENCY
------------
The same, unchanged `LearningState` must not create uncontrolled,
repeated adaptation "churn" (fresh decision ids/timestamps for a
dimension whose evidence hasn't actually changed). This engine
fingerprints each `DimensionState` it evaluates and, when a
(subject, area) pair's evidence is unchanged since the last call,
returns the previously issued decision object as-is rather than
minting a new one. Internal bookkeeping (`_last_decision`,
`_last_fingerprint`) is instance-scoped (never global) and
size-bounded (oldest entries are evicted once a configurable cap is
reached), so this module can run for the lifetime of a process without
an unbounded memory cache.
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Shared data contracts (from learning_engine.py)
# ----------------------------------------------------------------------
try:
    from .learning_engine import (  # type: ignore
        LearningState,
        DimensionState,
        EvidenceRecord,
        SignalType,
    )
except Exception:  # pragma: no cover - fallback for standalone use/testing
    class SignalType(str, Enum):
        EXPLICIT = "explicit"
        CORRECTION = "correction"
        IMPLICIT_POSITIVE = "implicit_positive"
        IMPLICIT_NEGATIVE = "implicit_negative"
        SYSTEM = "system"

    @dataclass(frozen=True)
    class EvidenceRecord:
        signal_id: str
        dimension: str
        value: float
        effective_weight: float
        signal_type: SignalType
        source: str
        timestamp: datetime

    @dataclass(frozen=True)
    class DimensionState:
        dimension: str
        net_score: float = 0.0
        confidence: float = 0.0
        evidence_count: int = 0
        last_updated: Optional[datetime] = None
        recent_history: Tuple[EvidenceRecord, ...] = field(default_factory=tuple)
        contradiction_flag: bool = False
        stability_score: float = 1.0

    @dataclass(frozen=True)
    class LearningState:
        subject: str
        dimensions: Dict[str, DimensionState] = field(default_factory=dict)
        version: int = 0
        last_updated: Optional[datetime] = None

        def get_dimension(self, dimension: str) -> DimensionState:
            return self.dimensions.get(dimension, DimensionState(dimension=dimension))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sign(value: float, epsilon: float = 1e-9) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _is_finite_real(value: Any) -> bool:
    """True only for a real int/float that is neither NaN nor +/-inf."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return not (math.isnan(value) or math.isinf(value))


# ======================================================================
# Adaptation areas (the supported surface)
# ======================================================================
# Fixed, closed vocabulary — no arbitrary/guessed areas are ever invented.
# Every member here corresponds to a genuinely existing response
# capability surfaced downstream (via adaptive_policy.py / response_policy.py
# fields, or reserved for a documented future capability of the same name).

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


# Default mapping from learning_engine dimension names to adaptation areas.
# This is intentionally a plain, overridable dict (not a hardcoded switch)
# so callers can extend it without editing engine internals — e.g. if
# feedback_engine/learning_engine introduce new dimension names. Extending
# this map is the ONLY sanctioned way to add area coverage; nothing in
# this module infers or guesses a mapping at runtime.
DEFAULT_DIMENSION_TO_AREA: Dict[str, AdaptationArea] = {
    "verbosity": AdaptationArea.RESPONSE_LENGTH,
    "response_length": AdaptationArea.RESPONSE_LENGTH,
    "explanation_depth": AdaptationArea.EXPLANATION_DEPTH,
    "detail_level": AdaptationArea.EXPLANATION_DEPTH,
    "formality": AdaptationArea.COMMUNICATION_STYLE,
    "tone": AdaptationArea.COMMUNICATION_STYLE,
    "communication_style": AdaptationArea.COMMUNICATION_STYLE,
    "language_preference": AdaptationArea.LANGUAGE_PREFERENCE,
    "topic_affinity": AdaptationArea.TOPIC_AFFINITY,
    "educational_emphasis": AdaptationArea.EDUCATIONAL_EMPHASIS,
    "informational_emphasis": AdaptationArea.INFORMATIONAL_EMPHASIS,
    "general_assistance_emphasis": AdaptationArea.GENERAL_ASSISTANCE_EMPHASIS,
    "interaction_style": AdaptationArea.INTERACTION_STYLE,
    "initiative_level": AdaptationArea.INITIATIVE_LEVEL,
    "proactivity": AdaptationArea.INITIATIVE_LEVEL,
}


# ======================================================================
# Safety envelope — hard limits that no amount of learned evidence,
# confidence, or future ML/RL strategy may exceed.
# ======================================================================

@dataclass(frozen=True)
class AreaSafetyEnvelope:
    """
    Hard, non-negotiable bounds for one adaptation area. These are set by
    the application (not by client input, not by learned state) and are
    enforced unconditionally in AdaptationEngine — no code path in this
    module may bypass them, regardless of confidence or evidence. No
    metadata carried on a LearningState/DimensionState/EvidenceRecord can
    ever override these values; the envelope is supplied only at
    `AdaptationEngineConfig` construction time by trusted application code.
    """
    min_value: float = -1.0
    max_value: float = 1.0
    max_magnitude_per_decision: float = 0.35
    adaptable: bool = True  # if False, this area never adapts (locked)


DEFAULT_SAFETY_ENVELOPE: Dict[AdaptationArea, AreaSafetyEnvelope] = {
    area: AreaSafetyEnvelope() for area in AdaptationArea
}


# ======================================================================
# Output contract: AdaptationDecision
# ======================================================================

class DecisionOutcome(str, Enum):
    PROPOSED = "proposed"                    # a real, actionable adaptation
    PROPOSED_TENTATIVE = "proposed_tentative"  # actionable but low-durability
    HELD_INSUFFICIENT_EVIDENCE = "held_insufficient_evidence"
    HELD_INSUFFICIENT_CONFIDENCE = "held_insufficient_confidence"
    HELD_OSCILLATION_GUARD = "held_oscillation_guard"
    HELD_CONTRADICTION = "held_contradiction"
    HELD_AREA_LOCKED = "held_area_locked"
    HELD_INVALID_STATE = "held_invalid_state"  # malformed/non-finite learning state
    NO_CHANGE = "no_change"  # evaluated, but new value ~= previous value


class Durability(str, Enum):
    """
    Rule 8: "Do not permanently adapt from weak evidence." Decisions are
    tagged with how durable they should be treated by adaptive_policy.py.
    """
    TENTATIVE = "tentative"      # apply softly / reversible on next signal
    ESTABLISHED = "established"  # backed by enough evidence+confidence
    EXPLICIT = "explicit"        # user said so directly — treat as strong


@dataclass(frozen=True)
class ProvenanceEntry:
    """One piece of evidence that contributed to a decision (Rule 9)."""
    signal_id: str
    signal_type: str
    source: str
    value: float
    timestamp: Optional[str]  # ISO8601


@dataclass(frozen=True)
class AdaptationDecision:
    """
    A single proposed adaptation for one (subject, area). Immutable,
    serializable, and carries its own justification so adaptive_policy.py
    (or a human reviewer) can decide whether/how to apply it without
    needing to re-derive anything from raw learning state. Exposes ONLY
    structured, machine-readable explanation metadata — no free-text
    chain-of-thought is ever produced or stored here.
    """
    decision_id: str
    subject: str
    area: AdaptationArea
    outcome: DecisionOutcome
    previous_value: float
    proposed_value: float
    delta: float
    confidence: float
    evidence_count: int
    durability: Durability
    is_reversal: bool
    reason_codes: Tuple[str, ...]
    provenance: Tuple[ProvenanceEntry, ...]
    generated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "subject": self.subject,
            "area": self.area.value,
            "outcome": self.outcome.value,
            "previous_value": self.previous_value,
            "proposed_value": self.proposed_value,
            "delta": self.delta,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "durability": self.durability.value,
            "is_reversal": self.is_reversal,
            "reason_codes": list(self.reason_codes),
            "provenance": [
                {
                    "signal_id": p.signal_id,
                    "signal_type": p.signal_type,
                    "source": p.source,
                    "value": p.value,
                    "timestamp": p.timestamp,
                }
                for p in self.provenance
            ],
            "generated_at": self.generated_at.isoformat(),
        }

    @property
    def is_actionable(self) -> bool:
        """Convenience for adaptive_policy.py: should this even be considered?"""
        return self.outcome in (DecisionOutcome.PROPOSED, DecisionOutcome.PROPOSED_TENTATIVE)


# ======================================================================
# Policy interface (so RL / statistical / ML models can replace the
# rule-based decision policy without breaking adaptive_policy.py)
# ======================================================================

class AdaptationPolicy(ABC):
    """
    Pluggable decision policy. AdaptationEngine depends only on this
    interface. A future RL agent, bandit, or statistical preference model
    can implement this same interface and be swapped in without touching
    AdaptationEngine's public surface, the AdaptationDecision contract, or
    any downstream application code.
    """

    @abstractmethod
    def decide(
        self,
        area: AdaptationArea,
        dim_state: DimensionState,
        previous_decision: Optional[AdaptationDecision],
        envelope: AreaSafetyEnvelope,
    ) -> "PolicyVerdict":
        raise NotImplementedError


@dataclass(frozen=True)
class PolicyVerdict:
    """What the policy thinks should happen, before safety clamping."""
    proposed_value: float
    outcome: DecisionOutcome
    durability: Durability
    is_reversal: bool
    reason_codes: Tuple[str, ...]


@dataclass
class RuleBasedPolicyConfig:
    # Rule 1: sufficient evidence.
    min_evidence_count: int = 4
    min_evidence_count_explicit: int = 1  # explicit statements need less repetition

    # Rule 2: sufficient confidence.
    min_confidence: float = 0.35
    min_confidence_explicit: float = 0.15

    # Rule 3: magnitude limiting (soft cap; hard cap is the safety envelope).
    max_change_per_decision: float = 0.25

    # Rule 4: oscillation prevention — minimum time between direction
    # reversals for the same area, and a minimum stability_score.
    min_seconds_between_reversals: float = 6 * 3600.0
    min_stability_for_change: float = 0.4
    min_stability_for_change_explicit: float = 0.0  # explicit can override instability

    # Rule 4b: even without explicit evidence, a sufficiently strong AND
    # stable reversal must eventually be allowed — the system must never
    # get permanently stuck defending a stale tendency. If the dimension's
    # own confidence and stability both clear these (deliberately high)
    # bars, the reversal cooldown is bypassed on evidence strength alone.
    reversal_override_confidence: float = 0.85
    reversal_override_min_stability: float = 0.75

    # Rule 8: durability thresholds.
    established_evidence_count: int = 10
    established_confidence: float = 0.6

    # Ignore changes smaller than this — not worth a decision record.
    no_change_epsilon: float = 0.03


class RuleBasedAdaptationPolicy(AdaptationPolicy):
    """
    Deterministic, explainable default policy implementing Rules 1-8 plus
    the bounded reversal-override rule (4b). Replaceable via the
    AdaptationPolicy interface.
    """

    def __init__(self, config: Optional[RuleBasedPolicyConfig] = None):
        self.config = config or RuleBasedPolicyConfig()

    def decide(
        self,
        area: AdaptationArea,
        dim_state: DimensionState,
        previous_decision: Optional[AdaptationDecision],
        envelope: AreaSafetyEnvelope,
    ) -> PolicyVerdict:
        cfg = self.config
        reason_codes: List[str] = []

        # Rule 7: explicit preference > weak inference — detect whether
        # any *recent* evidence behind this state was an explicit/correction
        # signal, and relax evidence/confidence/stability thresholds for it.
        has_explicit_evidence = any(
            getattr(r, "signal_type", None) in (SignalType.EXPLICIT, SignalType.CORRECTION)
            for r in getattr(dim_state, "recent_history", ())
        )
        if has_explicit_evidence:
            reason_codes.append("explicit_evidence_present")

        min_evidence = (
            cfg.min_evidence_count_explicit if has_explicit_evidence else cfg.min_evidence_count
        )
        min_confidence = (
            cfg.min_confidence_explicit if has_explicit_evidence else cfg.min_confidence
        )
        min_stability = (
            cfg.min_stability_for_change_explicit
            if has_explicit_evidence
            else cfg.min_stability_for_change
        )

        # --- Rule 1: sufficient evidence.
        if dim_state.evidence_count < min_evidence:
            return PolicyVerdict(
                proposed_value=previous_decision.proposed_value if previous_decision else 0.0,
                outcome=DecisionOutcome.HELD_INSUFFICIENT_EVIDENCE,
                durability=Durability.TENTATIVE,
                is_reversal=False,
                reason_codes=tuple(reason_codes + [
                    f"evidence_count={dim_state.evidence_count}<{min_evidence}"
                ]),
            )

        # --- Rule 2: sufficient confidence.
        if dim_state.confidence < min_confidence:
            return PolicyVerdict(
                proposed_value=previous_decision.proposed_value if previous_decision else 0.0,
                outcome=DecisionOutcome.HELD_INSUFFICIENT_CONFIDENCE,
                durability=Durability.TENTATIVE,
                is_reversal=False,
                reason_codes=tuple(reason_codes + [
                    f"confidence={dim_state.confidence:.2f}<{min_confidence:.2f}"
                ]),
            )

        # --- Rule 6: contradiction support — don't suppress, but flag,
        # dampen, and preserve provenance so a single contradicting spike
        # can't flip behavior alone, while still letting later evidence
        # resolve the conflict on its own merits over subsequent calls.
        target_value = dim_state.net_score
        if getattr(dim_state, "contradiction_flag", False):
            reason_codes.append("contradiction_flag_set")
            if previous_decision is not None:
                # Pull the target halfway back toward the last accepted
                # decision rather than fully committing to the new value.
                target_value = (target_value + previous_decision.proposed_value) / 2.0
                reason_codes.append("contradiction_dampened_toward_previous")

        previous_value = previous_decision.proposed_value if previous_decision else 0.0
        raw_delta = target_value - previous_value

        if abs(raw_delta) < cfg.no_change_epsilon:
            return PolicyVerdict(
                proposed_value=previous_value,
                outcome=DecisionOutcome.NO_CHANGE,
                durability=previous_decision.durability if previous_decision else Durability.TENTATIVE,
                is_reversal=False,
                reason_codes=tuple(reason_codes + ["delta_below_epsilon"]),
            )

        is_reversal = (
            previous_decision is not None
            and _sign(raw_delta) != 0
            and _sign(previous_decision.delta) != 0
            and _sign(raw_delta) != _sign(previous_decision.delta)
        )

        # --- Rule 4: oscillation prevention, with a bounded override (4b)
        # so the system can never become permanently stuck defending a
        # stale tendency against sufficiently strong, stable evidence.
        if is_reversal and previous_decision is not None:
            stability = getattr(dim_state, "stability_score", 1.0)
            elapsed = (_now() - previous_decision.generated_at).total_seconds()

            if stability < min_stability:
                reason_codes.append(f"stability_score={stability:.2f}<{min_stability:.2f}")
                return PolicyVerdict(
                    proposed_value=previous_value,
                    outcome=DecisionOutcome.HELD_OSCILLATION_GUARD,
                    durability=previous_decision.durability,
                    is_reversal=False,
                    reason_codes=tuple(reason_codes),
                )

            cooldown_active = elapsed < cfg.min_seconds_between_reversals
            if cooldown_active and not has_explicit_evidence:
                strong_stable_override = (
                    dim_state.confidence >= cfg.reversal_override_confidence
                    and stability >= cfg.reversal_override_min_stability
                )
                if strong_stable_override:
                    reason_codes.append(
                        f"reversal_override_high_confidence(confidence={dim_state.confidence:.2f},"
                        f"stability={stability:.2f})"
                    )
                else:
                    reason_codes.append(
                        f"reversal_cooldown_active(elapsed={elapsed:.0f}s)"
                    )
                    return PolicyVerdict(
                        proposed_value=previous_value,
                        outcome=DecisionOutcome.HELD_OSCILLATION_GUARD,
                        durability=previous_decision.durability,
                        is_reversal=False,
                        reason_codes=tuple(reason_codes),
                    )

            reason_codes.append("reversal_accepted")

        # --- Rule 3: magnitude limiting (soft policy cap; hard cap applied
        # later by the engine's safety envelope regardless of this policy).
        delta = raw_delta
        if abs(delta) > cfg.max_change_per_decision:
            delta = cfg.max_change_per_decision if delta > 0 else -cfg.max_change_per_decision
            reason_codes.append("policy_magnitude_cap_applied")

        proposed_value = previous_value + delta

        # --- Rule 8: durability classification.
        if has_explicit_evidence:
            durability = Durability.EXPLICIT
        elif (
            dim_state.evidence_count >= cfg.established_evidence_count
            and dim_state.confidence >= cfg.established_confidence
        ):
            durability = Durability.ESTABLISHED
        else:
            durability = Durability.TENTATIVE

        outcome = (
            DecisionOutcome.PROPOSED
            if durability != Durability.TENTATIVE
            else DecisionOutcome.PROPOSED_TENTATIVE
        )

        return PolicyVerdict(
            proposed_value=proposed_value,
            outcome=outcome,
            durability=durability,
            is_reversal=is_reversal,
            reason_codes=tuple(reason_codes),
        )


# ======================================================================
# The engine
# ======================================================================

@dataclass
class AdaptationEngineConfig:
    policy: Optional[AdaptationPolicy] = None
    dimension_to_area: Dict[str, AdaptationArea] = field(
        default_factory=lambda: dict(DEFAULT_DIMENSION_TO_AREA)
    )
    safety_envelopes: Dict[AdaptationArea, AreaSafetyEnvelope] = field(
        default_factory=lambda: dict(DEFAULT_SAFETY_ENVELOPE)
    )
    max_provenance_entries: int = 8
    # Rule 10 (idempotency / bounded memory): maximum number of distinct
    # (subject, area) keys this engine will retain decision history for.
    # Once exceeded, the least-recently-touched key is evicted. This
    # bounds memory for a long-lived process without requiring an
    # external persistence layer for in-flight decision tracking.
    max_tracked_keys: int = 10_000


class AdaptationEngine:
    """
    Stateless-in/state-out processor: given a `LearningState` and the
    engine's own prior decisions (held internally, instance-scoped, never
    global), produces `AdaptationDecision` objects for each mapped area.

    Security posture:
      - The only input accepted is a `LearningState` object produced by
        learning_engine.py. There is no code path that accepts a raw
        client-supplied "set adaptation to X" command — decisions are only
        ever derived from accumulated, validated learning state.
      - Every `DimensionState` is validated (finite, in-range numeric
        fields; non-empty identifiers) BEFORE reaching the policy layer.
        A dimension that fails validation is held at its last known-good
        value (`HELD_INVALID_STATE`) rather than ever reaching
        `AdaptationPolicy.decide` with potentially NaN/Infinite input.
      - `AreaSafetyEnvelope` bounds are supplied by application config at
        construction time, not by any per-call input or state metadata,
        and are enforced unconditionally in `_clamp_to_envelope` after
        the policy runs — no policy output can exceed them, including a
        future RL/ML policy.
      - Locked areas (`envelope.adaptable == False`) never produce a
        change regardless of evidence/confidence.
      - Idempotency: a (subject, area) pair whose `DimensionState`
        fingerprint is unchanged since the last call returns the exact
        previously issued `AdaptationDecision` object rather than minting
        a new one, preventing uncontrolled decision churn from repeated
        processing of the same learning state.
      - Internal history (`_last_decision`, `_last_fingerprint`) is
        instance-scoped and capped at `config.max_tracked_keys` entries
        (oldest-touched evicted first) — never an unbounded global cache.
    """

    def __init__(self, config: Optional[AdaptationEngineConfig] = None):
        self.config = config or AdaptationEngineConfig()
        self._policy: AdaptationPolicy = self.config.policy or RuleBasedAdaptationPolicy()
        self._lock = threading.Lock()
        # Instance-scoped (not global), size-bounded history of the last
        # accepted decision AND the fingerprint of the DimensionState that
        # produced it, per (subject, area) — needed for oscillation/
        # reversal detection and for idempotent re-evaluation.
        self._last_decision: "OrderedDict[Tuple[str, AdaptationArea], AdaptationDecision]" = OrderedDict()
        self._last_fingerprint: "OrderedDict[Tuple[str, AdaptationArea], Tuple[Any, ...]]" = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, state: LearningState) -> List[AdaptationDecision]:
        """
        Evaluate every dimension in `state` that maps to a known
        AdaptationArea and return one AdaptationDecision per area.
        Dimensions with no mapping are silently skipped (not an error) —
        extend `dimension_to_area` to support new dimensions. A malformed
        `state` (missing/invalid `subject`) is treated the same as any
        other invalid input: nothing is processed and an empty list is
        returned rather than raising, since there is no (subject, area)
        key to safely attach a HELD_INVALID_STATE decision to.
        """
        subject = getattr(state, "subject", None)
        if not isinstance(subject, str) or not subject:
            return []

        decisions: List[AdaptationDecision] = []
        for dimension, dim_state in getattr(state, "dimensions", {}).items():
            area = self.config.dimension_to_area.get(dimension)
            if area is None:
                continue
            decisions.append(self._process_one(subject, area, dim_state))
        return decisions

    def process_area(
        self, state: LearningState, dimension: str, area: AdaptationArea
    ) -> AdaptationDecision:
        """Evaluate a single explicit (dimension -> area) pair on demand."""
        dim_state = state.get_dimension(dimension)
        return self._process_one(state.subject, area, dim_state)

    def get_current_decision(
        self, subject: str, area: AdaptationArea
    ) -> Optional[AdaptationDecision]:
        """Last accepted (non-held) decision for (subject, area), if any."""
        with self._lock:
            return self._last_decision.get((subject, area))

    def explain(self, subject: str, area: AdaptationArea) -> Dict[str, Any]:
        decision = self.get_current_decision(subject, area)
        if decision is None:
            return {"subject": subject, "area": area.value, "status": "no_decision_yet"}
        return decision.to_dict()

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dim_state(dim_state: Any) -> Optional[str]:
        """
        Returns None if `dim_state` is safe to hand to the policy layer,
        otherwise a short machine-readable reason code. Rejects NaN/
        Infinity and out-of-range values on every numeric field this
        engine reads, and malformed identifiers, without ever raising —
        malformed learning state is expected, safely-rejectable input,
        not a programming error.
        """
        dimension = getattr(dim_state, "dimension", None)
        if not isinstance(dimension, str) or not dimension:
            return "invalid_dimension_identifier"

        net_score = getattr(dim_state, "net_score", 0.0)
        if not _is_finite_real(net_score) or not (-1.0 - 1e-9 <= net_score <= 1.0 + 1e-9):
            return "net_score_not_finite_or_out_of_range"

        confidence = getattr(dim_state, "confidence", 0.0)
        if not _is_finite_real(confidence) or not (0.0 - 1e-9 <= confidence <= 1.0 + 1e-9):
            return "confidence_not_finite_or_out_of_range"

        stability_score = getattr(dim_state, "stability_score", 1.0)
        if not _is_finite_real(stability_score) or not (0.0 - 1e-9 <= stability_score <= 1.0 + 1e-9):
            return "stability_score_not_finite_or_out_of_range"

        evidence_count = getattr(dim_state, "evidence_count", 0)
        if isinstance(evidence_count, bool) or not isinstance(evidence_count, int) or evidence_count < 0:
            return "evidence_count_invalid"

        for record in getattr(dim_state, "recent_history", ()):
            value = getattr(record, "value", 0.0)
            weight = getattr(record, "effective_weight", 0.0)
            if not _is_finite_real(value) or not _is_finite_real(weight):
                return "evidence_record_value_not_finite"

        return None

    @staticmethod
    def _fingerprint(dim_state: Any) -> Tuple[Any, ...]:
        """
        Cheap, deterministic fingerprint of the evidence behind a
        DimensionState, used purely for idempotency (has anything
        actually changed since the last time we evaluated this
        dimension?). Not a security boundary — validation happens
        separately in `_validate_dim_state`.
        """
        last_updated = getattr(dim_state, "last_updated", None)
        return (
            getattr(dim_state, "net_score", None),
            getattr(dim_state, "confidence", None),
            getattr(dim_state, "evidence_count", None),
            getattr(dim_state, "stability_score", None),
            getattr(dim_state, "contradiction_flag", None),
            last_updated.isoformat() if isinstance(last_updated, datetime) else last_updated,
        )

    # ------------------------------------------------------------------
    # Internal: bounded LRU-style bookkeeping
    # ------------------------------------------------------------------

    def _touch(
        self,
        key: Tuple[str, AdaptationArea],
        decision: AdaptationDecision,
        fingerprint: Tuple[Any, ...],
    ) -> None:
        with self._lock:
            self._last_decision[key] = decision
            self._last_decision.move_to_end(key)
            self._last_fingerprint[key] = fingerprint
            self._last_fingerprint.move_to_end(key)

            max_keys = max(1, self.config.max_tracked_keys)
            while len(self._last_decision) > max_keys:
                self._last_decision.popitem(last=False)
            while len(self._last_fingerprint) > max_keys:
                self._last_fingerprint.popitem(last=False)

    def _get_previous(
        self, key: Tuple[str, AdaptationArea]
    ) -> Tuple[Optional[AdaptationDecision], Optional[Tuple[Any, ...]]]:
        with self._lock:
            decision = self._last_decision.get(key)
            if decision is not None:
                self._last_decision.move_to_end(key)
            fingerprint = self._last_fingerprint.get(key)
            if fingerprint is not None:
                self._last_fingerprint.move_to_end(key)
            return decision, fingerprint

    # ------------------------------------------------------------------
    # Internal: core evaluation
    # ------------------------------------------------------------------

    def _process_one(
        self, subject: str, area: AdaptationArea, dim_state: DimensionState
    ) -> AdaptationDecision:
        envelope = self.config.safety_envelopes.get(area, AreaSafetyEnvelope())
        key = (subject, area)

        previous_decision, previous_fingerprint = self._get_previous(key)

        # --- Security/validation gate: malformed state never reaches the
        # policy layer. Held at the last known-good value.
        invalid_reason = self._validate_dim_state(dim_state)
        if invalid_reason is not None:
            decision = self._build_decision(
                subject=subject,
                area=area,
                dim_state=dim_state,
                previous_value=previous_decision.proposed_value if previous_decision else 0.0,
                verdict=PolicyVerdict(
                    proposed_value=previous_decision.proposed_value if previous_decision else 0.0,
                    outcome=DecisionOutcome.HELD_INVALID_STATE,
                    durability=previous_decision.durability if previous_decision else Durability.TENTATIVE,
                    is_reversal=False,
                    reason_codes=(f"invalid_learning_state:{invalid_reason}",),
                ),
                safe_dim_state=False,
            )
            # Do not update fingerprint/decision history from invalid
            # input — a corrupt read must never become the new baseline.
            return decision

        # --- Hard lock: never adapt this area, no matter what.
        if not envelope.adaptable:
            decision = self._build_decision(
                subject=subject,
                area=area,
                dim_state=dim_state,
                previous_value=previous_decision.proposed_value if previous_decision else 0.0,
                verdict=PolicyVerdict(
                    proposed_value=previous_decision.proposed_value if previous_decision else 0.0,
                    outcome=DecisionOutcome.HELD_AREA_LOCKED,
                    durability=Durability.TENTATIVE,
                    is_reversal=False,
                    reason_codes=("area_locked_by_safety_config",),
                ),
            )
            return decision

        # --- Idempotency: unchanged evidence since last evaluation ->
        # return the exact previously issued decision, no new id/timestamp,
        # no re-invocation of the policy. This is what prevents "the same
        # learning state creating uncontrolled repeated adaptation
        # decisions" (Rule 10) while still allowing genuinely new evidence
        # to be evaluated immediately.
        fingerprint = self._fingerprint(dim_state)
        if previous_decision is not None and fingerprint == previous_fingerprint:
            return previous_decision

        verdict = self._policy.decide(area, dim_state, previous_decision, envelope)

        previous_value = previous_decision.proposed_value if previous_decision else 0.0
        decision = self._build_decision(
            subject=subject,
            area=area,
            dim_state=dim_state,
            previous_value=previous_value,
            verdict=verdict,
            envelope=envelope,
        )

        # Track fingerprint regardless of outcome (so a HELD_* result for
        # this exact evidence is also idempotent next call), but only
        # advance the "current decision" pointer for actionable outcomes;
        # held/no-change outcomes leave prior actionable state as-is so a
        # temporary dip in evidence/confidence doesn't erase adaptation
        # history.
        if decision.is_actionable:
            self._touch(key, decision, fingerprint)
        else:
            with self._lock:
                self._last_fingerprint[key] = fingerprint
                self._last_fingerprint.move_to_end(key)
                max_keys = max(1, self.config.max_tracked_keys)
                while len(self._last_fingerprint) > max_keys:
                    self._last_fingerprint.popitem(last=False)

        return decision

    def _build_decision(
        self,
        subject: str,
        area: AdaptationArea,
        dim_state: DimensionState,
        previous_value: float,
        verdict: PolicyVerdict,
        envelope: Optional[AreaSafetyEnvelope] = None,
        safe_dim_state: bool = True,
    ) -> AdaptationDecision:
        import uuid

        reason_codes = list(verdict.reason_codes)
        proposed_value = verdict.proposed_value

        # --- Unconditional safety clamp — applies even if a future
        # RL/ML policy produced the verdict. Nothing bypasses this.
        if envelope is not None and verdict.outcome in (
            DecisionOutcome.PROPOSED,
            DecisionOutcome.PROPOSED_TENTATIVE,
        ):
            clamped = _clamp(proposed_value, envelope.min_value, envelope.max_value)
            magnitude = clamped - previous_value
            if abs(magnitude) > envelope.max_magnitude_per_decision:
                magnitude = (
                    envelope.max_magnitude_per_decision
                    if magnitude > 0
                    else -envelope.max_magnitude_per_decision
                )
                clamped = previous_value + magnitude
                reason_codes.append("safety_envelope_magnitude_clamp_applied")
            if clamped != proposed_value:
                reason_codes.append("safety_envelope_value_clamp_applied")
            proposed_value = clamped

        delta = proposed_value - previous_value

        confidence = getattr(dim_state, "confidence", 0.0) if safe_dim_state else 0.0
        evidence_count = getattr(dim_state, "evidence_count", 0) if safe_dim_state else 0
        # Even when the incoming state failed validation, surface
        # whatever finite confidence/evidence_count values ARE readable,
        # for explainability, without letting non-finite ones through.
        if not _is_finite_real(confidence):
            confidence = 0.0
        if isinstance(evidence_count, bool) or not isinstance(evidence_count, int) or evidence_count < 0:
            evidence_count = 0

        provenance = tuple(
            ProvenanceEntry(
                signal_id=getattr(r, "signal_id", ""),
                signal_type=getattr(getattr(r, "signal_type", None), "value", str(getattr(r, "signal_type", ""))),
                source=getattr(r, "source", "unknown"),
                value=getattr(r, "value", 0.0) if _is_finite_real(getattr(r, "value", 0.0)) else 0.0,
                timestamp=getattr(r, "timestamp", None).isoformat()
                if getattr(r, "timestamp", None)
                else None,
            )
            for r in list(getattr(dim_state, "recent_history", ()))[
                -self.config.max_provenance_entries:
            ]
        )

        return AdaptationDecision(
            decision_id=str(uuid.uuid4()),
            subject=subject,
            area=area,
            outcome=verdict.outcome,
            previous_value=previous_value,
            proposed_value=proposed_value,
            delta=delta,
            confidence=confidence,
            evidence_count=evidence_count,
            durability=verdict.durability,
            is_reversal=verdict.is_reversal,
            reason_codes=tuple(reason_codes),
            provenance=provenance,
        )


# ======================================================================
# Package boundary integration
# ======================================================================

def get_adaptation_engine(config: Optional[AdaptationEngineConfig] = None) -> AdaptationEngine:
    """
    Public factory expected by the package `__init__.py`'s lazy
    component resolution (`_COMPONENT_MODULES["adaptation_engine"]`).
    Called with zero arguments by that resolver, so `config` is optional
    and defaults to today's rule-based behavior exactly. Returns a fresh
    `AdaptationEngine` instance; hold your own instance directly if you
    need decision-history continuity across calls beyond what a single
    shared instance already provides.
    """
    return AdaptationEngine(config)


# ======================================================================
# Example integration sketch (not executed; for reference only)
# ======================================================================
#
# from learning_engine import LearningEngine
# from adaptation_engine import AdaptationEngine
#
# learning_engine = LearningEngine()
# adaptation_engine = AdaptationEngine()
#
# state = LearningState.from_dict(loaded_state_dict)  # via your persistence layer
# state, transition = learning_engine.process(signal, state)
# persist(state.to_dict())
#
# decisions = adaptation_engine.process(state)
# for d in decisions:
#     if d.is_actionable:
#         adaptive_policy.consider(d)   # adaptive_policy.py decides whether/how to apply it
#     audit_log(d.to_dict())            # explainability trail
