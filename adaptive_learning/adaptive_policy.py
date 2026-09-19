"""
Adaptive Intelligence & Learning / adaptive_policy.py
======================================================

Role
----
This module is the FINAL ADAPTIVE DECISION BOUNDARY of Phase 4. It sits
between learned/adaptive behavior and the existing response-generation
architecture:

    Current Context
          +
    Existing Response Policy      (response_policy.py — duck-typed input)
          +
    Learning State                (learning_engine.py — optional)
          +
    Adaptation Decision(s)        (adaptation_engine.py — optional)
          v
    Adaptive Policy                <-- THIS MODULE
          v
    Existing AI orchestration
          v
    Final response

CRITICAL RULE
-------------
This module does NOT generate an AI response and does NOT call any model,
database, or network service. It ONLY produces a policy/recommendation
object (`AdaptivePolicyResult`) for a downstream response generator to
follow. The existing AI service remains solely responsible for model
execution.

BACKWARD COMPATIBILITY
-----------------------
`response_policy.py` remains authoritative. This module:
  - never imports response_policy.py, user_profile.py, context_memory.py,
    personality_engine.py, emotion_engine.py, ai_services.py, server.py,
    logic.py, or database.py (Phase 4 -> application is a forbidden
    dependency direction; see the package __init__.py docstring),
  - never mutates the `ResponsePolicy`-like object it is given,
  - treats it as an opaque, duck-typed input and only ever *adds* a
    derived, separate output on top of it,
  - falls back to a value-for-value equivalent of the original policy
    whenever adaptive information is missing, invalid, disabled, or the
    adaptation pipeline errors in any way.

ADAPTATION GUARANTEES
----------------------
Every adjustment this module makes is:
  - bounded       — at most one ordinal "step" per dimension per call,
                     regardless of how large the underlying learning
                     signal claims to be; booleans only flip on strong,
                     high-confidence evidence.
  - validated     — malformed decisions/learning state are skipped
                     individually rather than corrupting the whole
                     result; malformed *base* input still fails safely
                     (see FAIL-SAFE below).
  - confidence-aware — durability (EXPLICIT > ESTABLISHED > TENTATIVE)
                     and per-decision confidence gate how much (if any)
                     influence a decision gets.
  - reversible    — this module holds NO persistent state of its own.
                     Every call recomputes the adaptive policy fresh from
                     whatever `AdaptationDecision`s it is given, so a
                     later contradicting decision fully supersedes an
                     earlier one on the very next call.
  - explainable internally — every applied or skipped adaptation is
                     recorded with a machine-readable reason, exposed via
                     `AdaptivePolicyResult.adaptive_meta`.

PRIORITY HIERARCHY (never reversed)
------------------------------------
    Hard system constraints
        >
    Explicit user preferences
        >
    Stable learned preferences      (AdaptationDecision.durability == EXPLICIT)
        >
    Strong adaptation decisions     (AdaptationDecision.durability == ESTABLISHED)
        >
    Weak inferred behavior          (AdaptationDecision.durability == TENTATIVE /
                                      outcome == PROPOSED_TENTATIVE)

"Hard system constraints" here means: dimensions response_policy.py has
already fixed for safety/trust reasons (fact-sensitive or game intents),
dimensions locked by the adaptation engine's own safety envelope
(`HELD_AREA_LOCKED`), and GAME isolation (see below). "Explicit user
preferences" means any field name the caller marks as explicitly set by
the user (via `explicit_locked_fields`) — this module will never
override those, no matter how strong the learned signal is.

GAME ISOLATION
---------------
General personalization/adaptation is NEVER applied when the base
policy's intent is "GAME" (or the caller marks `is_game_context=True`),
unless the caller explicitly passes `adaptation_authorized_for_game=True`
(reserved for a future, deliberately-scoped game-behavior policy). This
prevents general learned behavior from silently bleeding into gameplay.

SAME-FIELD CONFLICT RESOLUTION
--------------------------------
More than one `AdaptationDecision` can legally target the same output
field in a single `build()` call (e.g. two decisions both mapping to
"response_length"). This module NEVER resolves that by "whichever is
later in the list wins" — order of the input sequence carries no
priority weight. Instead, all decisions that would otherwise apply to
the same field are grouped and exactly ONE winner is chosen, in this
fixed order:

    1. hard constraints            (already excluded before grouping)
    2. explicit user preference    (already excluded before grouping,
                                     via `explicit_locked_fields`)
    3. highest durability tier     (EXPLICIT > ESTABLISHED > TENTATIVE)
    4. highest `confidence`
    5. highest `evidence_count`    ("strongest evidence")
    6. most recent `generated_at`  (a pure tie-breaker, used ONLY when
                                     1-5 are exactly equal)

Every non-winning decision for that field is recorded in `skipped` with
reason `superseded_by_higher_priority_decision:<winning_decision_id>` —
it is not silently dropped. Only the winning decision is ever passed
to the `AdaptationApplier`, so a single `build()` call still moves any
one ordinal field by at most one step, regardless of how many competing
decisions were supplied.

FAIL-SAFE
---------
If `adaptation_decisions` is missing/None/empty/not a list, if
`learning_state` is invalid, if this module is disabled (`ENABLED =
False`), or if anything in the adaptation path raises an unexpected
error, `build()` returns a result that is value-for-value equivalent to
the given base policy (same fields, same values) with the reason
recorded in `adaptive_meta`. The only case this module refuses to
paper over is a base policy that is not usable at all (missing required
fields) — that is a caller programming error, not an adaptive-state
problem, and raises `AdaptivePolicyError` rather than being silently
guessed at.

NO DATABASE ACCESS. NO LLM CALLS. NO NETWORK CALLS. NO SECRETS.

FUTURE (4E/4F)
---------------
The actual "how do I turn one AdaptationDecision into one field change"
logic lives behind the `AdaptationApplier` interface. A future
optimization system (statistical, bandit, or RL-based) can implement
that interface and be passed into `AdaptivePolicyEngine` / `build()`
without changing this module's public surface, its safety envelope, or
the `AdaptivePolicyResult` contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Kill switch — checked by the package __init__.py's component resolver.
# Flip to False to make this entire module report as DISABLED, in which
# case __init__.get_adaptive_policy() returns None and callers are
# expected to use response_policy.py's output unmodified.
# ---------------------------------------------------------------------------
ENABLED: bool = True


class AdaptivePolicyError(ValueError):
    """
    Raised only for a genuinely unusable *base* policy input (missing
    required fields) — a caller programming error, not an adaptive-state
    problem. Everything downstream of a valid base policy fails soft.
    """


# ---------------------------------------------------------------------------
# Defensive imports of sibling Phase 4 modules.
#
# This module depends only on OTHER Phase 4 modules (never on the
# application). Those modules may not exist yet, may be mid-development,
# or may raise on import for unrelated reasons — any of that is treated
# as "adaptation info unavailable", never as a reason to fail the caller.
# ---------------------------------------------------------------------------
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
        NO_CHANGE = "no_change"

    class Durability(str, Enum):
        TENTATIVE = "tentative"
        ESTABLISHED = "established"
        EXPLICIT = "explicit"

    AdaptationDecision = Any  # type: ignore


# ---------------------------------------------------------------------------
# Priority hierarchy
# ---------------------------------------------------------------------------

class PriorityTier(str, Enum):
    """
    Explicit representation of the mandated safety hierarchy. Higher tiers
    always beat lower tiers; this ordering must never be reversed anywhere
    in this module.
    """
    HARD_CONSTRAINT = "hard_constraint"                # never adaptable
    EXPLICIT_USER_PREFERENCE = "explicit_user_preference"  # caller-locked field
    STABLE_LEARNED_PREFERENCE = "stable_learned_preference"  # durability EXPLICIT
    STRONG_ADAPTATION_DECISION = "strong_adaptation_decision"  # durability ESTABLISHED
    WEAK_INFERRED_BEHAVIOR = "weak_inferred_behavior"  # durability TENTATIVE


_TIER_RANK: Dict[PriorityTier, int] = {
    PriorityTier.HARD_CONSTRAINT: 4,
    PriorityTier.EXPLICIT_USER_PREFERENCE: 3,
    PriorityTier.STABLE_LEARNED_PREFERENCE: 2,
    PriorityTier.STRONG_ADAPTATION_DECISION: 1,
    PriorityTier.WEAK_INFERRED_BEHAVIOR: 0,
}


def _tier_for_durability(durability: Any) -> PriorityTier:
    if durability == Durability.EXPLICIT:
        return PriorityTier.STABLE_LEARNED_PREFERENCE
    if durability == Durability.ESTABLISHED:
        return PriorityTier.STRONG_ADAPTATION_DECISION
    return PriorityTier.WEAK_INFERRED_BEHAVIOR


# ---------------------------------------------------------------------------
# Hard constraints (mirrors response_policy.py's own invariants).
#
# These constants intentionally DUPLICATE (rather than import) the
# equivalent private sets in response_policy.py, because Phase 4 must
# never import back into the application layer (see package __init__.py).
# If response_policy.py's invariants change, update these to match.
# ---------------------------------------------------------------------------

NON_PERSONALIZABLE_INTENTS: frozenset = frozenset(
    {"NEWS", "SHOPPING", "GAME", "UNKNOWN"}
)
FIXED_TONE_INTENTS: frozenset = frozenset({"NEWS", "SHOPPING", "GAME", "UNKNOWN"})
GAME_INTENT: str = "GAME"

# Fields this module will NEVER adapt, under any circumstances, for any
# intent. `intent` itself is ground truth (set upstream); recommendation
# behavior and the current-information requirement are safety/trust
# sensitive and are fixed by response_policy.py's base policy alone.
ALWAYS_LOCKED_FIELDS: frozenset = frozenset(
    {"intent", "recommendation_behavior", "requires_current_information"}
)

# Ordinal scales for fields that can move by one bounded "step" in a
# known direction. Order matters: index 0 is the "least" of the
# dimension, the last index is the "most".
_ORDINAL_SCALES: Dict[str, Tuple[str, ...]] = {
    "response_length": ("SHORT", "MEDIUM", "LONG"),
    "explanation_depth": ("SURFACE", "MODERATE", "DEEP"),
    "example_usage": ("NONE", "MINIMAL", "SOME", "RICH"),
    "follow_up_behavior": ("NONE", "OPTIONAL_LIGHT", "SUGGEST_RELATED", "ASK_CLARIFYING"),
}

# Tone is categorical, not strictly ordinal, but a small "warmth" ladder
# is safe to nudge along. Tones outside this ladder (SUPPORTIVE,
# PRACTICAL, CONCISE) are intent-purpose-bound and are left untouched.
_TONE_WARMTH_LADDER: Tuple[str, ...] = ("FORMAL", "NEUTRAL", "FRIENDLY", "PLAYFUL")

# Default mapping from an AdaptationArea to the ResponsePolicy-shaped
# field it may influence, and what kind of movement is allowed for it.
# "ordinal" -> use _ORDINAL_SCALES[field]; "tone_ladder" -> use
# _TONE_WARMTH_LADDER; "boolean" -> threshold-gated flip.
_DEFAULT_AREA_FIELD_MAP: Dict[AdaptationArea, Tuple[str, str]] = {
    AdaptationArea.RESPONSE_LENGTH: ("response_length", "ordinal"),
    AdaptationArea.EXPLANATION_DEPTH: ("explanation_depth", "ordinal"),
    AdaptationArea.COMMUNICATION_STYLE: ("tone", "tone_ladder"),
    AdaptationArea.INTERACTION_STYLE: ("tone", "tone_ladder"),
    AdaptationArea.INITIATIVE_LEVEL: ("follow_up_behavior", "ordinal"),
    AdaptationArea.EDUCATIONAL_EMPHASIS: ("educational_emphasis", "boolean"),
}
# LANGUAGE_PREFERENCE, TOPIC_AFFINITY, INFORMATIONAL_EMPHASIS, and
# GENERAL_ASSISTANCE_EMPHASIS have no corresponding ResponsePolicy field
# today and are intentionally left unmapped (silently skipped) rather
# than guessed at. Extend `_DEFAULT_AREA_FIELD_MAP` when/if such fields
# are added to response_policy.py.

# Intents for which ANY field-level personalization is allowed at all.
# Mirrors response_policy.py's `_PERSONALIZABLE_INTENTS`.
_DEFAULT_PERSONALIZABLE_INTENTS: frozenset = frozenset(
    {"EDUCATION", "GENERAL", "INFORMATION", "TECHNOLOGY", "BUSINESS",
     "ENTERTAINMENT", "PERSONAL"}
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sign(value: float, epsilon: float = 1e-9) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppliedAdaptation:
    """One field-level change this module actually made, fully explained."""
    field: str
    area: str
    previous_value: Any
    new_value: Any
    priority_tier: str
    durability: str
    confidence: float
    decision_id: Optional[str]
    reason_codes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "area": self.area,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "priority_tier": self.priority_tier,
            "durability": self.durability,
            "confidence": self.confidence,
            "decision_id": self.decision_id,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class SkippedAdaptation:
    """One candidate adaptation this module deliberately did NOT apply."""
    field: Optional[str]
    area: Optional[str]
    reason: str
    decision_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "area": self.area,
            "reason": self.reason,
            "decision_id": self.decision_id,
        }


@dataclass(frozen=True)
class AdaptivePolicyResult:
    """
    The final, consumable output of this boundary.

    `to_dict()` intentionally emits the SAME top-level keys as
    `response_policy.ResponsePolicy.to_dict()` (intent, response_length,
    explanation_depth, tone, educational_emphasis, example_usage,
    follow_up_behavior, requires_current_information,
    recommendation_behavior, notes) so existing AI orchestration can use
    this result as a drop-in, additive replacement — plus one extra
    `adaptive_meta` key carrying full explainability detail that
    unaware callers can safely ignore.
    """
    intent: str
    response_length: str
    explanation_depth: str
    tone: str
    educational_emphasis: bool
    example_usage: str
    follow_up_behavior: str
    requires_current_information: bool
    recommendation_behavior: str
    notes: List[str]

    adaptive_enabled: bool
    fell_back_to_base: bool
    fallback_reason: Optional[str]
    applied: Tuple[AppliedAdaptation, ...] = ()
    skipped: Tuple[SkippedAdaptation, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "response_length": self.response_length,
            "explanation_depth": self.explanation_depth,
            "tone": self.tone,
            "educational_emphasis": self.educational_emphasis,
            "example_usage": self.example_usage,
            "follow_up_behavior": self.follow_up_behavior,
            "requires_current_information": self.requires_current_information,
            "recommendation_behavior": self.recommendation_behavior,
            "notes": list(self.notes),
            "adaptive_meta": {
                "adaptive_enabled": self.adaptive_enabled,
                "fell_back_to_base": self.fell_back_to_base,
                "fallback_reason": self.fallback_reason,
                "applied": [a.to_dict() for a in self.applied],
                "skipped": [s.to_dict() for s in self.skipped],
            },
        }


# ---------------------------------------------------------------------------
# Pluggable applier interface (for future 4E/4F optimization systems)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApplierVerdict:
    """What an AdaptationApplier thinks a field's new value should be."""
    changed: bool
    new_value: Any = None
    reason_codes: Tuple[str, ...] = ()


class AdaptationApplier(ABC):
    """
    Pluggable strategy for turning ONE (field, current_value,
    AdaptationDecision) into an ApplierVerdict. `AdaptivePolicyEngine`
    depends only on this interface, so a future statistical/bandit/RL
    optimization system (4E/4F) can be swapped in without touching this
    module's public surface, its priority hierarchy enforcement, or the
    `AdaptivePolicyResult` contract. Every implementation MUST still
    respect the one-bounded-step contract; the engine does not re-clamp
    ordinal moves beyond what the applier itself decides.
    """

    @abstractmethod
    def apply(
        self,
        field: str,
        kind: str,
        current_value: Any,
        decision: Any,
        max_step_fraction: float,
    ) -> ApplierVerdict:
        raise NotImplementedError


class DefaultAdaptationApplier(AdaptationApplier):
    """
    Deterministic, explainable default applier:
      - ordinal fields move at most one step in the sign(delta) direction.
      - tone_ladder fields move at most one step along the warmth ladder,
        and only if the current tone is actually on that ladder.
      - boolean fields flip only when the decision's proposed_value
        magnitude clears a confidence-scaled threshold in the flip's
        direction; otherwise no change.
    `max_step_fraction` (0.0-1.0) lets tentative (weak) evidence take a
    softer action than strong/stable evidence; the default applier uses
    it to decide whether a boolean flip is warranted at all, and always
    caps ordinal movement to exactly one step regardless (never more).
    """

    BOOLEAN_FLIP_THRESHOLD = 0.5

    def apply(
        self,
        field: str,
        kind: str,
        current_value: Any,
        decision: Any,
        max_step_fraction: float,
    ) -> ApplierVerdict:
        delta = getattr(decision, "delta", 0.0)
        proposed_value = getattr(decision, "proposed_value", 0.0)
        direction = _sign(delta)

        if direction == 0:
            return ApplierVerdict(changed=False, reason_codes=("no_directional_signal",))

        if kind == "ordinal":
            scale = _ORDINAL_SCALES.get(field)
            return self._step_ordinal(scale, current_value, direction)

        if kind == "tone_ladder":
            return self._step_ordinal(_TONE_WARMTH_LADDER, current_value, direction, strict=True)

        if kind == "boolean":
            threshold = self.BOOLEAN_FLIP_THRESHOLD * (2.0 - _clamp01(max_step_fraction))
            if abs(proposed_value) < threshold:
                return ApplierVerdict(
                    changed=False,
                    reason_codes=(f"boolean_threshold_not_met(|{proposed_value:.2f}|<{threshold:.2f})",),
                )
            new_value = direction > 0
            if bool(current_value) == new_value:
                return ApplierVerdict(changed=False, reason_codes=("boolean_already_at_target",))
            return ApplierVerdict(
                changed=True,
                new_value=new_value,
                reason_codes=(f"boolean_flip_threshold_met(|{proposed_value:.2f}|>={threshold:.2f})",),
            )

        return ApplierVerdict(changed=False, reason_codes=(f"unsupported_field_kind:{kind}",))

    @staticmethod
    def _step_ordinal(
        scale: Optional[Sequence[str]],
        current_value: Any,
        direction: int,
        strict: bool = False,
    ) -> ApplierVerdict:
        if not scale:
            return ApplierVerdict(changed=False, reason_codes=("no_scale_defined",))
        if current_value not in scale:
            if strict:
                # e.g. tone currently SUPPORTIVE/PRACTICAL/CONCISE — not on
                # this ladder; leave it alone rather than guessing a slot.
                return ApplierVerdict(changed=False, reason_codes=("current_value_off_scale",))
            return ApplierVerdict(changed=False, reason_codes=("current_value_off_scale",))

        idx = scale.index(current_value)
        new_idx = max(0, min(len(scale) - 1, idx + direction))
        if new_idx == idx:
            return ApplierVerdict(changed=False, reason_codes=("already_at_scale_boundary",))
        return ApplierVerdict(
            changed=True,
            new_value=scale[new_idx],
            reason_codes=(f"ordinal_step({idx}->{new_idx})",),
        )


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptivePolicyConfig:
    applier: Optional[AdaptationApplier] = None
    area_field_map: Dict[AdaptationArea, Tuple[str, str]] = field(
        default_factory=lambda: dict(_DEFAULT_AREA_FIELD_MAP)
    )
    personalizable_intents: frozenset = field(
        default_factory=lambda: _DEFAULT_PERSONALIZABLE_INTENTS
    )
    non_personalizable_intents: frozenset = field(
        default_factory=lambda: NON_PERSONALIZABLE_INTENTS
    )
    fixed_tone_intents: frozenset = field(
        default_factory=lambda: FIXED_TONE_INTENTS
    )
    always_locked_fields: frozenset = field(
        default_factory=lambda: ALWAYS_LOCKED_FIELDS
    )
    # Weak (TENTATIVE) evidence gets a reduced "license to act" fraction,
    # used by the applier for threshold-gated (boolean) decisions. Ordinal
    # moves are always exactly one step regardless of this fraction — the
    # bound on ordinal fields is structural, not a magnitude scale.
    tentative_step_fraction: float = 0.5
    stable_step_fraction: float = 1.0
    # Minimum decision.confidence required before even considering a
    # WEAK_INFERRED_BEHAVIOR (tentative) decision at all. Stronger tiers
    # are governed by adaptation_engine.py's own confidence gates already.
    min_confidence_for_tentative: float = 0.35


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class AdaptivePolicyEngine:
    """
    Stateless (per-instance, no persisted mutable state) processor that
    turns (base_policy, adaptation_decisions) into an AdaptivePolicyResult.

    Safe to share a single instance across requests/threads: `build()`
    reads only its arguments and `self.config` (set once at construction),
    and writes nothing.
    """

    def __init__(self, config: Optional[AdaptivePolicyConfig] = None):
        self.config = config or AdaptivePolicyConfig()
        self._applier: AdaptationApplier = self.config.applier or DefaultAdaptationApplier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        base_policy: Any,
        *,
        adaptation_decisions: Optional[Sequence[Any]] = None,
        learning_state: Optional[Any] = None,  # accepted for future/explainability use only
        context: Optional[Any] = None,          # accepted for future contextual gating
        intent: Optional[str] = None,
        is_game_context: bool = False,
        adaptation_authorized_for_game: bool = False,
        explicit_locked_fields: Optional[Sequence[str]] = None,
    ) -> AdaptivePolicyResult:
        """
        Build the adaptive policy. Never raises for adaptive-state
        problems (missing/invalid/disabled adaptation input, malformed
        individual decisions, or unexpected internal errors while
        applying adaptation) — all of those fall back to a base-policy-
        equivalent result with the reason recorded. Raises
        `AdaptivePolicyError` ONLY if `base_policy` itself cannot be
        read as a response-policy-shaped object at all.
        """
        base_fields = self._extract_base_fields(base_policy)  # may raise AdaptivePolicyError
        resolved_intent = intent or base_fields["intent"]

        # --- FAIL-SAFE / disablement checks -> immediate equivalent-to-base.
        if not ENABLED:
            return self._fallback(base_fields, "adaptive_policy_disabled")

        if is_game_context or resolved_intent == GAME_INTENT:
            if not adaptation_authorized_for_game:
                return self._fallback(base_fields, "game_isolation_not_authorized")

        if not adaptation_decisions:
            return self._fallback(base_fields, "no_adaptation_decisions_supplied")

        if not isinstance(adaptation_decisions, (list, tuple)):
            return self._fallback(base_fields, "adaptation_decisions_not_a_sequence")

        locked_fields = set(self.config.always_locked_fields)
        if explicit_locked_fields:
            try:
                locked_fields |= {str(f) for f in explicit_locked_fields}
            except TypeError:
                # Malformed caller input for this optional parameter is
                # itself an "invalid adaptive state" -> ignore, don't fail.
                pass

        # --- Hard constraint: intent not personalizable at all.
        if resolved_intent not in self.config.personalizable_intents:
            skipped = tuple(
                SkippedAdaptation(field=None, area=None,
                                   reason=f"intent_not_personalizable:{resolved_intent}")
                for _ in [0]
            )
            return self._fallback(
                base_fields, "intent_not_personalizable", skipped=skipped
            )

        # --- Everything below is best-effort: any unexpected error here
        # falls back to the base policy rather than becoming a single
        # point of failure for the assistant.
        try:
            return self._apply_decisions(base_fields, adaptation_decisions, locked_fields, resolved_intent)
        except Exception as exc:  # noqa: BLE001 - deliberate, documented fail-safe boundary
            return self._fallback(
                base_fields, f"unexpected_error:{type(exc).__name__}"
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_base_fields(base_policy: Any) -> Dict[str, Any]:
        """
        Duck-types the base ResponsePolicy so this module has zero import
        dependency on response_policy.py. Accepts either an object with
        the expected attributes/`to_dict()` or an already-plain dict.
        """
        if base_policy is None:
            raise AdaptivePolicyError("base_policy must not be None")

        if isinstance(base_policy, dict):
            source: Dict[str, Any] = base_policy
        elif hasattr(base_policy, "to_dict"):
            source = base_policy.to_dict()
        else:
            source = {
                "intent": getattr(base_policy, "intent", None),
                "response_length": getattr(base_policy, "response_length", None),
                "explanation_depth": getattr(base_policy, "explanation_depth", None),
                "tone": getattr(base_policy, "tone", None),
                "educational_emphasis": getattr(base_policy, "educational_emphasis", None),
                "example_usage": getattr(base_policy, "example_usage", None),
                "follow_up_behavior": getattr(base_policy, "follow_up_behavior", None),
                "requires_current_information": getattr(base_policy, "requires_current_information", None),
                "recommendation_behavior": getattr(base_policy, "recommendation_behavior", None),
                "notes": getattr(base_policy, "notes", None),
            }

        required = (
            "intent", "response_length", "explanation_depth", "tone",
            "educational_emphasis", "example_usage", "follow_up_behavior",
            "requires_current_information", "recommendation_behavior",
        )
        missing = [f for f in required if source.get(f) is None and f != "educational_emphasis"
                   and f != "requires_current_information"]
        # booleans are allowed to be legitimately False, so check presence
        # rather than truthiness for the two boolean fields.
        if "educational_emphasis" not in source:
            missing.append("educational_emphasis")
        if "requires_current_information" not in source:
            missing.append("requires_current_information")
        if missing:
            raise AdaptivePolicyError(
                f"base_policy is missing required field(s): {missing}"
            )

        return {
            "intent": str(source["intent"]),
            "response_length": str(source["response_length"]),
            "explanation_depth": str(source["explanation_depth"]),
            "tone": str(source["tone"]),
            "educational_emphasis": bool(source["educational_emphasis"]),
            "example_usage": str(source["example_usage"]),
            "follow_up_behavior": str(source["follow_up_behavior"]),
            "requires_current_information": bool(source["requires_current_information"]),
            "recommendation_behavior": str(source["recommendation_behavior"]),
            "notes": list(source.get("notes") or []),
        }

    @staticmethod
    def _fallback(
        base_fields: Dict[str, Any],
        reason: str,
        skipped: Tuple[SkippedAdaptation, ...] = (),
    ) -> AdaptivePolicyResult:
        """Equivalent-to-base-policy result: no adaptation applied."""
        return AdaptivePolicyResult(
            intent=base_fields["intent"],
            response_length=base_fields["response_length"],
            explanation_depth=base_fields["explanation_depth"],
            tone=base_fields["tone"],
            educational_emphasis=base_fields["educational_emphasis"],
            example_usage=base_fields["example_usage"],
            follow_up_behavior=base_fields["follow_up_behavior"],
            requires_current_information=base_fields["requires_current_information"],
            recommendation_behavior=base_fields["recommendation_behavior"],
            notes=list(base_fields["notes"]),
            adaptive_enabled=ENABLED,
            fell_back_to_base=True,
            fallback_reason=reason,
            applied=(),
            skipped=skipped,
        )

    def _apply_decisions(
        self,
        base_fields: Dict[str, Any],
        adaptation_decisions: Sequence[Any],
        locked_fields: set,
        intent: str,
    ) -> AdaptivePolicyResult:
        working = dict(base_fields)
        working["notes"] = list(base_fields["notes"])
        applied: List[AppliedAdaptation] = []
        skipped: List[SkippedAdaptation] = []

        fixed_tone = intent in self.config.fixed_tone_intents

        # --- Phase 1: pre-filter every decision independently. Anything
        # that is unconditionally excluded (hard-locked field, explicit
        # user lock, non-actionable outcome, under-confident tentative
        # evidence, unmapped area, malformed decision) is resolved here
        # and never reaches conflict resolution — those exclusions sit
        # ABOVE durability/confidence/evidence in the priority hierarchy,
        # so no amount of "strong evidence" can revive them.
        candidates_by_field: Dict[str, List[Tuple]] = {}
        for decision in adaptation_decisions:
            try:
                outcome = self._prefilter_one(decision, locked_fields, fixed_tone)
            except Exception as exc:  # noqa: BLE001 - one bad decision must not sink the batch
                skipped.append(
                    SkippedAdaptation(
                        field=None, area=None,
                        reason=f"invalid_decision:{type(exc).__name__}",
                        decision_id=getattr(decision, "decision_id", None),
                    )
                )
                continue

            if outcome is None:
                continue  # unmapped area — nothing to record, not an error
            if isinstance(outcome, SkippedAdaptation):
                skipped.append(outcome)
                continue

            # A live candidate: (decision, field_name, kind, area_name,
            # tier, confidence, evidence_count, generated_at, decision_id)
            field_name = outcome[1]
            candidates_by_field.setdefault(field_name, []).append(outcome)

        # --- Phase 2: resolve same-field conflicts (never "latest wins" —
        # see module docstring "SAME-FIELD CONFLICT RESOLUTION") and apply
        # exactly one winner per field.
        for field_name, candidates in candidates_by_field.items():
            if len(candidates) > 1:
                candidates_sorted = sorted(
                    candidates,
                    key=lambda c: (_TIER_RANK[c[4]], c[5], c[6], c[7]),
                    reverse=True,
                )
                winner = candidates_sorted[0]
                for loser in candidates_sorted[1:]:
                    skipped.append(
                        SkippedAdaptation(
                            loser[1], loser[3],
                            f"superseded_by_higher_priority_decision:{winner[8]}",
                            loser[8],
                        )
                    )
            else:
                winner = candidates[0]

            (
                decision, _field_name, kind, area_name,
                tier, confidence, _evidence_count, _generated_at, decision_id,
            ) = winner

            step_fraction = (
                self.config.tentative_step_fraction
                if tier == PriorityTier.WEAK_INFERRED_BEHAVIOR
                else self.config.stable_step_fraction
            )
            current_value = working.get(field_name)
            verdict = self._applier.apply(field_name, kind, current_value, decision, step_fraction)

            if not verdict.changed:
                skipped.append(
                    SkippedAdaptation(
                        field_name, area_name,
                        ";".join(verdict.reason_codes) or "no_change",
                        decision_id,
                    )
                )
                continue

            durability = getattr(decision, "durability", Durability.TENTATIVE)
            applied.append(
                AppliedAdaptation(
                    field=field_name,
                    area=area_name,
                    previous_value=current_value,
                    new_value=verdict.new_value,
                    priority_tier=tier.value,
                    durability=getattr(durability, "value", str(durability)),
                    confidence=confidence,
                    decision_id=decision_id,
                    reason_codes=verdict.reason_codes,
                )
            )
            working[field_name] = verdict.new_value

        if applied:
            summary = ", ".join(f"{a.field}->{a.new_value}" for a in applied)
            working["notes"].append(f"Adaptive layer applied: {summary}.")

        return AdaptivePolicyResult(
            intent=working["intent"],
            response_length=working["response_length"],
            explanation_depth=working["explanation_depth"],
            tone=working["tone"],
            educational_emphasis=working["educational_emphasis"],
            example_usage=working["example_usage"],
            follow_up_behavior=working["follow_up_behavior"],
            requires_current_information=working["requires_current_information"],
            recommendation_behavior=working["recommendation_behavior"],
            notes=working["notes"],
            adaptive_enabled=ENABLED,
            fell_back_to_base=False,
            fallback_reason=None,
            applied=tuple(applied),
            skipped=tuple(skipped),
        )

    @staticmethod
    def _normalized_generated_at(decision: Any) -> datetime:
        """
        Extracts a tz-aware `generated_at` for tie-breaking, defaulting to
        the earliest possible timestamp (never crashes/never favors a
        decision just because it's missing a timestamp).
        """
        gen_at = getattr(decision, "generated_at", None)
        if not isinstance(gen_at, datetime):
            return datetime.min.replace(tzinfo=timezone.utc)
        if gen_at.tzinfo is None:
            return gen_at.replace(tzinfo=timezone.utc)
        return gen_at

    def _prefilter_one(
        self,
        decision: Any,
        locked_fields: set,
        fixed_tone: bool,
    ) -> Optional[Any]:
        """
        Applies every exclusion that is NOT a same-field conflict:
        unmapped area (returns None), hard-locked field, fixed-tone
        field, explicit-user-locked field, non-actionable outcome, and
        under-confident tentative evidence (all return a
        `SkippedAdaptation`). Anything that survives is returned as a
        plain candidate tuple for Phase 2 conflict resolution — this
        method never itself applies a change or consults `working`
        (the current field value), since that depends on which decision
        ultimately wins for the field.
        """
        area = getattr(decision, "area", None)
        decision_id = getattr(decision, "decision_id", None)

        mapping = self.config.area_field_map.get(area)
        if mapping is None:
            return None  # this area has no corresponding response field
        field_name, kind = mapping
        area_name = getattr(area, "value", str(area))

        # --- Hard constraints (never adaptable).
        if field_name in self.config.always_locked_fields:
            return SkippedAdaptation(field_name, area_name, "field_is_hard_locked", decision_id)
        if field_name == "tone" and fixed_tone:
            return SkippedAdaptation(field_name, area_name, "tone_fixed_for_intent", decision_id)

        # --- Explicit user preference (caller-declared lock) always wins.
        if field_name in locked_fields:
            return SkippedAdaptation(field_name, area_name, "locked_by_explicit_user_preference", decision_id)

        outcome = getattr(decision, "outcome", None)
        is_actionable = getattr(decision, "is_actionable", None)
        if is_actionable is None:
            is_actionable = outcome in (DecisionOutcome.PROPOSED, DecisionOutcome.PROPOSED_TENTATIVE)
        if not is_actionable:
            return SkippedAdaptation(
                field_name, area_name,
                f"decision_not_actionable:{getattr(outcome, 'value', outcome)}",
                decision_id,
            )

        durability = getattr(decision, "durability", Durability.TENTATIVE)
        tier = _tier_for_durability(durability)
        confidence = float(getattr(decision, "confidence", 0.0) or 0.0)

        if tier == PriorityTier.WEAK_INFERRED_BEHAVIOR and confidence < self.config.min_confidence_for_tentative:
            return SkippedAdaptation(
                field_name, area_name,
                f"tentative_confidence_too_low({confidence:.2f})",
                decision_id,
            )

        evidence_count = getattr(decision, "evidence_count", 0)
        try:
            evidence_count = int(evidence_count)
        except (TypeError, ValueError):
            evidence_count = 0
        generated_at = self._normalized_generated_at(decision)

        return (
            decision, field_name, kind, area_name,
            tier, confidence, evidence_count, generated_at, decision_id,
        )


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default_engine = AdaptivePolicyEngine()


def build_adaptive_policy(
    base_policy: Any,
    *,
    adaptation_decisions: Optional[Sequence[Any]] = None,
    learning_state: Optional[Any] = None,
    context: Optional[Any] = None,
    intent: Optional[str] = None,
    is_game_context: bool = False,
    adaptation_authorized_for_game: bool = False,
    explicit_locked_fields: Optional[Sequence[str]] = None,
) -> AdaptivePolicyResult:
    """Module-level convenience wrapper around a shared default engine."""
    return _default_engine.build(
        base_policy,
        adaptation_decisions=adaptation_decisions,
        learning_state=learning_state,
        context=context,
        intent=intent,
        is_game_context=is_game_context,
        adaptation_authorized_for_game=adaptation_authorized_for_game,
        explicit_locked_fields=explicit_locked_fields,
    )


def build_adaptive_policy_dict(
    base_policy: Any,
    *,
    adaptation_decisions: Optional[Sequence[Any]] = None,
    learning_state: Optional[Any] = None,
    context: Optional[Any] = None,
    intent: Optional[str] = None,
    is_game_context: bool = False,
    adaptation_authorized_for_game: bool = False,
    explicit_locked_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper returning a plain dict for JSON serialization."""
    return build_adaptive_policy(
        base_policy,
        adaptation_decisions=adaptation_decisions,
        learning_state=learning_state,
        context=context,
        intent=intent,
        is_game_context=is_game_context,
        adaptation_authorized_for_game=adaptation_authorized_for_game,
        explicit_locked_fields=explicit_locked_fields,
    ).to_dict()


# ---------------------------------------------------------------------------
# Package boundary integration
# ---------------------------------------------------------------------------

def get_adaptive_policy() -> AdaptivePolicyEngine:
    """
    Public factory expected by the package `__init__.py`'s lazy
    component resolution (`_COMPONENT_MODULES["adaptive_policy"]`).

    Returns a fresh, stateless `AdaptivePolicyEngine` instance backed by
    the default configuration and the default rule-based applier. Safe
    to call repeatedly; callers needing a custom `AdaptivePolicyConfig`
    (e.g. a future 4E/4F applier) should construct `AdaptivePolicyEngine`
    directly instead of going through this factory.
    """
    return AdaptivePolicyEngine()


# ---------------------------------------------------------------------------
# Self-tests (task 14). Exercised only when this module is run directly —
# never imported or executed as a side effect of `import adaptive_policy`.
# Uses lightweight local stand-ins for AdaptationDecision/base-policy so
# this module stays testable standalone, with zero dependency on the
# application layer or a test framework.
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:  # pragma: no cover - exercised via __main__
    from datetime import timedelta

    failures: List[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            failures.append(f"{name}: {detail}")

    @dataclass(frozen=True)
    class _FakeDecision:
        decision_id: str
        area: Any
        outcome: Any = DecisionOutcome.PROPOSED
        previous_value: float = 0.0
        proposed_value: float = 1.0
        delta: float = 1.0
        confidence: float = 0.8
        evidence_count: int = 5
        durability: Any = Durability.ESTABLISHED
        generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def base_policy(**overrides: Any) -> Dict[str, Any]:
        policy = {
            "intent": "EDUCATION",
            "response_length": "MEDIUM",
            "explanation_depth": "MODERATE",
            "tone": "NEUTRAL",
            "educational_emphasis": False,
            "example_usage": "SOME",
            "follow_up_behavior": "OPTIONAL_LIGHT",
            "requires_current_information": False,
            "recommendation_behavior": "NONE",
            "notes": [],
        }
        policy.update(overrides)
        return policy

    engine = AdaptivePolicyEngine()

    # 1) no adaptation supplied -> fails soft to base, unchanged fields.
    result = engine.build(base_policy(), adaptation_decisions=None)
    check("no_adaptation", result.fell_back_to_base and result.response_length == "MEDIUM",
          result.fallback_reason)

    # 2) one adaptation -> exactly one bounded ordinal step.
    result = engine.build(
        base_policy(),
        adaptation_decisions=[_FakeDecision("d1", AdaptationArea.RESPONSE_LENGTH)],
    )
    check("one_adaptation", not result.fell_back_to_base and result.response_length == "LONG",
          result.response_length)
    check("one_adaptation_applied_count", len(result.applied) == 1, len(result.applied))

    # 3) multiple adaptations on different fields -> both apply independently.
    result = engine.build(
        base_policy(),
        adaptation_decisions=[
            _FakeDecision("d1", AdaptationArea.RESPONSE_LENGTH),
            _FakeDecision("d2", AdaptationArea.EXPLANATION_DEPTH),
        ],
    )
    check(
        "multiple_adaptations",
        result.response_length == "LONG" and result.explanation_depth == "DEEP",
        (result.response_length, result.explanation_depth),
    )

    # 4) conflicting adaptations on the SAME field -> higher tier/confidence
    # wins, loser recorded as superseded, never "latest wins".
    weak = _FakeDecision(
        "d_weak", AdaptationArea.RESPONSE_LENGTH, durability=Durability.TENTATIVE,
        confidence=0.9, evidence_count=1,
        generated_at=datetime.now(timezone.utc),  # newest, but still lower tier
    )
    strong = _FakeDecision(
        "d_strong", AdaptationArea.RESPONSE_LENGTH, durability=Durability.ESTABLISHED,
        confidence=0.5, evidence_count=2,
        generated_at=datetime.now(timezone.utc) - timedelta(hours=1),  # older, but higher tier
    )
    result = engine.build(base_policy(), adaptation_decisions=[weak, strong])
    check("conflicting_adaptations_applied", len(result.applied) == 1, len(result.applied))
    if result.applied:
        check(
            "conflicting_adaptations_winner",
            result.applied[0].decision_id == "d_strong",
            result.applied[0].decision_id,
        )
    check(
        "conflicting_adaptations_loser_recorded",
        any(s.decision_id == "d_weak" and "superseded_by_higher_priority_decision" in s.reason
            for s in result.skipped),
        [s.to_dict() for s in result.skipped],
    )

    # 5) explicit user preference beats any learned/adapted signal.
    result = engine.build(
        base_policy(),
        adaptation_decisions=[_FakeDecision("d1", AdaptationArea.RESPONSE_LENGTH, confidence=1.0,
                                             durability=Durability.EXPLICIT)],
        explicit_locked_fields={"response_length"},
    )
    check(
        "explicit_user_preference",
        result.response_length == "MEDIUM"
        and any(s.reason == "locked_by_explicit_user_preference" for s in result.skipped),
        result.to_dict(),
    )

    # 6) weak inference: below threshold is skipped, above threshold applies
    # with a reduced (tentative) step fraction but still exactly one step.
    result = engine.build(
        base_policy(),
        adaptation_decisions=[
            _FakeDecision("d1", AdaptationArea.RESPONSE_LENGTH, durability=Durability.TENTATIVE, confidence=0.1)
        ],
    )
    check(
        "weak_inference_below_threshold",
        result.response_length == "MEDIUM"
        and any("tentative_confidence_too_low" in s.reason for s in result.skipped),
        result.to_dict(),
    )
    result = engine.build(
        base_policy(),
        adaptation_decisions=[
            _FakeDecision("d1", AdaptationArea.RESPONSE_LENGTH, durability=Durability.TENTATIVE, confidence=0.8)
        ],
    )
    check("weak_inference_above_threshold", result.response_length == "LONG", result.response_length)

    # 7) hard system constraint: a field in always_locked_fields can never
    # move, regardless of decision strength.
    hard_cfg = AdaptivePolicyConfig(
        area_field_map={**_DEFAULT_AREA_FIELD_MAP, AdaptationArea.TOPIC_AFFINITY: ("response_length", "ordinal")},
        always_locked_fields=frozenset(ALWAYS_LOCKED_FIELDS | {"response_length"}),
    )
    hard_engine = AdaptivePolicyEngine(hard_cfg)
    result = hard_engine.build(
        base_policy(),
        adaptation_decisions=[_FakeDecision("d1", AdaptationArea.TOPIC_AFFINITY, confidence=1.0,
                                             durability=Durability.EXPLICIT)],
    )
    check(
        "hard_system_constraint",
        result.response_length == "MEDIUM"
        and any(s.reason == "field_is_hard_locked" for s in result.skipped),
        result.to_dict(),
    )

    # 8) GAME intent -> isolated from general personalization by default.
    result = engine.build(
        base_policy(intent="GAME"),
        adaptation_decisions=[_FakeDecision("d1", AdaptationArea.RESPONSE_LENGTH)],
    )
    check(
        "game_intent_isolated",
        result.fell_back_to_base and result.fallback_reason == "game_isolation_not_authorized",
        result.fallback_reason,
    )

    # 9) invalid decision (malformed confidence) is skipped individually,
    # not fatal to the batch.
    bad = _FakeDecision("d_bad", AdaptationArea.RESPONSE_LENGTH, confidence="not_a_number")  # type: ignore[arg-type]
    good = _FakeDecision("d_good", AdaptationArea.EXPLANATION_DEPTH)
    result = engine.build(base_policy(), adaptation_decisions=[bad, good])
    check(
        "invalid_decision_skipped",
        any(s.decision_id == "d_bad" and "invalid_decision" in s.reason for s in result.skipped),
        [s.to_dict() for s in result.skipped],
    )
    check("invalid_decision_does_not_sink_batch", result.explanation_depth == "DEEP", result.explanation_depth)

    # 10) unsupported/unmapped area is silently skipped, not an error.
    result = engine.build(
        base_policy(),
        adaptation_decisions=[
            _FakeDecision("d1", AdaptationArea.TOPIC_AFFINITY),  # unmapped by default
            _FakeDecision("d2", AdaptationArea.RESPONSE_LENGTH),
        ],
    )
    check(
        "unsupported_area_silently_skipped",
        not any(s.decision_id == "d1" for s in result.skipped) and result.response_length == "LONG",
        result.to_dict(),
    )

    # 11) disabled adaptive mode -> always falls back, even with strong decisions.
    global ENABLED
    previous_enabled = ENABLED
    ENABLED = False
    try:
        result = engine.build(
            base_policy(),
            adaptation_decisions=[_FakeDecision("d1", AdaptationArea.RESPONSE_LENGTH)],
        )
        check(
            "disabled_adaptive_mode",
            result.fell_back_to_base and result.fallback_reason == "adaptive_policy_disabled",
            result.fallback_reason,
        )
    finally:
        ENABLED = previous_enabled

    # 12) fallback preserves base-policy values exactly (value-for-value).
    base = base_policy(response_length="SHORT", tone="FORMAL")
    result = engine.build(base, adaptation_decisions=None)
    check(
        "fallback_value_for_value",
        result.response_length == "SHORT" and result.tone == "FORMAL" and result.applied == (),
        result.to_dict(),
    )

    # 13) bounded field changes: an extreme delta still moves an ordinal
    # field by exactly one step, never more.
    extreme = _FakeDecision("d1", AdaptationArea.RESPONSE_LENGTH, delta=999.0, proposed_value=999.0)
    result = engine.build(base_policy(), adaptation_decisions=[extreme])
    check("bounded_ordinal_change", result.response_length == "LONG", result.response_length)

    if failures:
        raise AssertionError("adaptive_policy self-tests failed:\n" + "\n".join(failures))
    print(f"adaptive_policy: all self-tests passed.")


# ---------------------------------------------------------------------------
# Example integration sketch (not executed; for reference only)
# ---------------------------------------------------------------------------
#
# from response_policy import build_response_policy          # existing app code
# from adaptive_intelligence_and_learning import (            # Phase 4 boundary
#     get_adaptation_engine, get_adaptive_policy,
# )
#
# base_policy = build_response_policy(intent, preference_score, user_profile, ctx)
# adaptation_engine = get_adaptation_engine()
# adaptive_policy = get_adaptive_policy()
#
# decisions = adaptation_engine.process(learning_state) if adaptation_engine else []
# result = adaptive_policy.build(
#     base_policy,
#     adaptation_decisions=decisions,
#     intent=intent,
#     explicit_locked_fields={"response_length"} if user_set_length_explicitly else None,
# ) if adaptive_policy else base_policy
#
# final_instructions = result.to_dict() if hasattr(result, "to_dict") else result
# ai_services.generate(final_instructions)   # existing AI orchestration, unchanged

if __name__ == "__main__":
    _run_self_tests()
