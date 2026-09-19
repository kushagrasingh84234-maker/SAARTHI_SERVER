"""
response_policy.py

Response-policy layer for the StudyBot personalized AI robot.

This module takes an already-determined intent (never re-derives or
overrides it), a preference score, a user profile, and conversation
context, and produces a structured set of RESPONSE INSTRUCTIONS across
fixed dimensions: length, explanation depth, tone, educational
emphasis, use of examples, follow-up behavior, current-information
requirement, and recommendation behavior.

This module does NOT:
  - call Groq, Hugging Face, Supabase, or any external service
  - use WebSocket
  - generate or invent factual content
  - determine or change intent
  - let personalization override or dilute explicit user intent

It only produces policy/instructions for a downstream response
generator to follow. It is pure, deterministic, and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums for response dimensions
# ---------------------------------------------------------------------------

class ResponseLength(str, Enum):
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"


class ExplanationDepth(str, Enum):
    SURFACE = "SURFACE"
    MODERATE = "MODERATE"
    DEEP = "DEEP"


class Tone(str, Enum):
    NEUTRAL = "NEUTRAL"
    FRIENDLY = "FRIENDLY"
    FORMAL = "FORMAL"
    CONCISE = "CONCISE"
    PLAYFUL = "PLAYFUL"
    SUPPORTIVE = "SUPPORTIVE"
    PRACTICAL = "PRACTICAL"


class ExampleUsage(str, Enum):
    NONE = "NONE"
    MINIMAL = "MINIMAL"
    SOME = "SOME"
    RICH = "RICH"


class FollowUpBehavior(str, Enum):
    NONE = "NONE"
    OPTIONAL_LIGHT = "OPTIONAL_LIGHT"
    SUGGEST_RELATED = "SUGGEST_RELATED"
    ASK_CLARIFYING = "ASK_CLARIFYING"


class RecommendationBehavior(str, Enum):
    NONE = "NONE"
    REQUIREMENTS_FIRST = "REQUIREMENTS_FIRST"
    COMPARISON_ORIENTED = "COMPARISON_ORIENTED"
    LIGHT_SUGGESTION = "LIGHT_SUGGESTION"


_VALID_INTENTS = frozenset({
    "EDUCATION", "GENERAL", "NEWS", "INFORMATION", "TECHNOLOGY",
    "SHOPPING", "BUSINESS", "ENTERTAINMENT", "PERSONAL", "GAME", "UNKNOWN",
})

_HIGH_PREFERENCE_THRESHOLD = 0.35
_LOW_PREFERENCE_THRESHOLD = 0.10


class ResponsePolicyError(ValueError):
    """Raised when inputs to the response policy engine are invalid."""


# ---------------------------------------------------------------------------
# Input context (minimal, decoupled from other modules' concrete types)
# ---------------------------------------------------------------------------

@dataclass
class PolicyContext:
    """
    Minimal, duck-typed snapshot of conversation context needed for
    policy decisions. Decoupled from any specific ConversationContext
    class so this module has zero import dependency on the rest of the
    personalization stack.
    """
    turns_in_session: int = 0
    category_lockin_detected: bool = False
    recent_intents: List[str] = field(default_factory=list)


@dataclass
class ResponsePolicy:
    """Structured response instructions for a downstream generator."""

    intent: str
    response_length: str
    explanation_depth: str
    tone: str
    educational_emphasis: bool
    example_usage: str
    follow_up_behavior: str
    requires_current_information: bool
    recommendation_behavior: str
    notes: List[str] = field(default_factory=list)

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
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_intent(intent: str) -> None:
    if not isinstance(intent, str) or intent not in _VALID_INTENTS:
        raise ResponsePolicyError(f"Invalid or unknown intent: {intent}")


def _validate_preference_score(preference_score: float) -> float:
    if not isinstance(preference_score, (int, float)):
        raise ResponsePolicyError("preference_score must be numeric")
    return max(0.0, min(1.0, float(preference_score)))


def _extract_user_profile_fields(user_profile: Any) -> Dict[str, str]:
    """
    Duck-types the user profile so this module doesn't need to import
    user_profile.py directly. Falls back to safe defaults if fields
    are missing.
    """
    preferred_length = getattr(user_profile, "preferred_response_length", "MEDIUM")
    preferred_style = getattr(user_profile, "preferred_interaction_style", "FRIENDLY")

    if preferred_length not in {"SHORT", "MEDIUM", "LONG"}:
        preferred_length = "MEDIUM"
    if preferred_style not in {"FORMAL", "CASUAL", "FRIENDLY", "CONCISE", "PLAYFUL"}:
        preferred_style = "FRIENDLY"

    return {
        "preferred_response_length": preferred_length,
        "preferred_interaction_style": preferred_style,
    }


# ---------------------------------------------------------------------------
# Per-intent base policies
# ---------------------------------------------------------------------------
# Each base policy defines the DEFAULT shape of a response for that
# intent, independent of preference/personalization. Personalization
# only nudges depth/examples/tone/follow-up within safe bounds — it
# never changes `requires_current_information`, `educational_emphasis`
# for unrelated intents, or the intent itself.

def _base_policy_education() -> ResponsePolicy:
    return ResponsePolicy(
        intent="EDUCATION",
        response_length="MEDIUM",
        explanation_depth="MODERATE",
        tone="FRIENDLY",
        educational_emphasis=True,
        example_usage="SOME",
        follow_up_behavior="SUGGEST_RELATED",
        requires_current_information=False,
        recommendation_behavior="NONE",
        notes=["Use clear explanations.", "Use step-by-step structure when appropriate."],
    )


def _base_policy_general() -> ResponsePolicy:
    return ResponsePolicy(
        intent="GENERAL",
        response_length="SHORT",
        explanation_depth="SURFACE",
        tone="FRIENDLY",
        educational_emphasis=False,
        example_usage="NONE",
        follow_up_behavior="NONE",
        requires_current_information=False,
        recommendation_behavior="NONE",
        notes=["Respond naturally and conversationally.", "Do not force structure onto small talk."],
    )


def _base_policy_news() -> ResponsePolicy:
    return ResponsePolicy(
        intent="NEWS",
        response_length="SHORT",
        explanation_depth="SURFACE",
        tone="NEUTRAL",
        educational_emphasis=False,
        example_usage="NONE",
        follow_up_behavior="OPTIONAL_LIGHT",
        requires_current_information=True,
        recommendation_behavior="NONE",
        notes=[
            "Prioritize the most current information available.",
            "Clearly distinguish confirmed facts from uncertain or developing details.",
            "Keep the summary concise.",
        ],
    )


def _base_policy_information() -> ResponsePolicy:
    return ResponsePolicy(
        intent="INFORMATION",
        response_length="SHORT",
        explanation_depth="MODERATE",
        tone="NEUTRAL",
        educational_emphasis=False,
        example_usage="MINIMAL",
        follow_up_behavior="OPTIONAL_LIGHT",
        requires_current_information=False,
        recommendation_behavior="NONE",
        notes=["Answer directly and factually.", "Avoid unnecessary elaboration."],
    )


def _base_policy_technology() -> ResponsePolicy:
    return ResponsePolicy(
        intent="TECHNOLOGY",
        response_length="MEDIUM",
        explanation_depth="MODERATE",
        tone="NEUTRAL",
        educational_emphasis=False,
        example_usage="SOME",
        follow_up_behavior="OPTIONAL_LIGHT",
        requires_current_information=True,
        notes=[
            "Prefer up-to-date technical details where relevant.",
            "Clarify version/context-specific caveats when applicable.",
        ],
        recommendation_behavior="NONE",
    )


def _base_policy_shopping() -> ResponsePolicy:
    return ResponsePolicy(
        intent="SHOPPING",
        response_length="MEDIUM",
        explanation_depth="MODERATE",
        tone="NEUTRAL",
        educational_emphasis=False,
        example_usage="SOME",
        follow_up_behavior="ASK_CLARIFYING",
        requires_current_information=True,
        recommendation_behavior="REQUIREMENTS_FIRST",
        notes=[
            "Clarify user requirements and constraints before recommending.",
            "Present comparisons rather than a single blind recommendation.",
            "Avoid recommending a specific product without sufficient context.",
        ],
    )


def _base_policy_business() -> ResponsePolicy:
    return ResponsePolicy(
        intent="BUSINESS",
        response_length="MEDIUM",
        explanation_depth="MODERATE",
        tone="PRACTICAL",
        educational_emphasis=False,
        example_usage="SOME",
        follow_up_behavior="SUGGEST_RELATED",
        requires_current_information=False,
        recommendation_behavior="LIGHT_SUGGESTION",
        notes=["Be practical and structured.", "Frame guidance around concrete next decisions or actions."],
    )


def _base_policy_entertainment() -> ResponsePolicy:
    return ResponsePolicy(
        intent="ENTERTAINMENT",
        response_length="SHORT",
        explanation_depth="SURFACE",
        tone="PLAYFUL",
        educational_emphasis=False,
        example_usage="MINIMAL",
        follow_up_behavior="OPTIONAL_LIGHT",
        requires_current_information=False,
        recommendation_behavior="NONE",
        notes=["Keep it light and fun.", "Do not over-explain."],
    )


def _base_policy_personal() -> ResponsePolicy:
    return ResponsePolicy(
        intent="PERSONAL",
        response_length="SHORT",
        explanation_depth="SURFACE",
        tone="SUPPORTIVE",
        educational_emphasis=False,
        example_usage="NONE",
        follow_up_behavior="OPTIONAL_LIGHT",
        requires_current_information=False,
        recommendation_behavior="NONE",
        notes=["Respond with warmth and attentiveness.", "Avoid turning a personal moment into a lecture."],
    )


def _base_policy_game() -> ResponsePolicy:
    return ResponsePolicy(
        intent="GAME",
        response_length="SHORT",
        explanation_depth="SURFACE",
        tone="PLAYFUL",
        educational_emphasis=False,
        example_usage="NONE",
        follow_up_behavior="NONE",
        requires_current_information=False,
        recommendation_behavior="NONE",
        notes=["Hand off to the game engine's flow.", "Keep any narration brief."],
    )


def _base_policy_unknown() -> ResponsePolicy:
    return ResponsePolicy(
        intent="UNKNOWN",
        response_length="SHORT",
        explanation_depth="SURFACE",
        tone="NEUTRAL",
        educational_emphasis=False,
        example_usage="NONE",
        follow_up_behavior="ASK_CLARIFYING",
        requires_current_information=False,
        recommendation_behavior="NONE",
        notes=["Intent is unclear; ask a brief clarifying question rather than guessing."],
    )


_BASE_POLICY_BUILDERS = {
    "EDUCATION": _base_policy_education,
    "GENERAL": _base_policy_general,
    "NEWS": _base_policy_news,
    "INFORMATION": _base_policy_information,
    "TECHNOLOGY": _base_policy_technology,
    "SHOPPING": _base_policy_shopping,
    "BUSINESS": _base_policy_business,
    "ENTERTAINMENT": _base_policy_entertainment,
    "PERSONAL": _base_policy_personal,
    "GAME": _base_policy_game,
    "UNKNOWN": _base_policy_unknown,
}

# Intents where preference-driven depth/example nudging is allowed at
# all. Fact-sensitive or safety-sensitive intents (NEWS, SHOPPING,
# GAME, UNKNOWN) are excluded so personalization can never soften
# factual rigor, push blind recommendations, or bleed into gameplay.
_PERSONALIZABLE_INTENTS = frozenset({
    "EDUCATION", "GENERAL", "INFORMATION", "TECHNOLOGY", "BUSINESS",
    "ENTERTAINMENT", "PERSONAL",
})


# ---------------------------------------------------------------------------
# Personalization nudges (bounded, never overriding base policy invariants)
# ---------------------------------------------------------------------------

def _apply_preference_nudge(
    policy: ResponsePolicy,
    preference_score: float,
    lockin_detected: bool,
) -> None:
    """
    Mutates `policy` in place with small, bounded adjustments based on
    preference strength. Never touches `intent`,
    `requires_current_information`, or `recommendation_behavior` for
    fact-sensitive/shopping intents — those are fixed by the base
    policy regardless of preference.
    """
    if policy.intent not in _PERSONALIZABLE_INTENTS:
        return

    if lockin_detected:
        # Safeguard: never let personalization intensify further once
        # lock-in is detected; keep things at base/moderate levels.
        policy.explanation_depth = "MODERATE"
        policy.example_usage = "SOME"
        policy.notes.append(
            "Category lock-in detected in recent history; personalization "
            "intensity dampened to avoid over-fitting to past behavior."
        )
        return

    if preference_score >= _HIGH_PREFERENCE_THRESHOLD:
        if policy.intent == "EDUCATION":
            policy.explanation_depth = "DEEP"
            policy.example_usage = "RICH"
            policy.notes.append("Strong educational affinity: deepen explanation and add richer examples.")
        else:
            # For non-education intents, a high EDUCATION-adjacent
            # affinity elsewhere must not force teaching mode; only
            # mildly enrich examples for the CURRENT intent.
            if policy.example_usage == "NONE":
                policy.example_usage = "MINIMAL"
            elif policy.example_usage == "MINIMAL":
                policy.example_usage = "SOME"
    elif preference_score <= _LOW_PREFERENCE_THRESHOLD:
        # Low affinity: keep things lighter, don't over-invest depth.
        if policy.explanation_depth == "DEEP":
            policy.explanation_depth = "MODERATE"


def _apply_response_length_preference(
    policy: ResponsePolicy,
    preferred_length: Optional[str],
) -> None:
    """
    Applies the user's explicitly stated response-length preference
    (from their UserProfile) as a soft override of the base policy's
    default length, since this is an explicit style preference rather
    than an inferred one.
    """
    if preferred_length in {"SHORT", "MEDIUM", "LONG"}:
        policy.response_length = preferred_length


def _apply_interaction_style_preference(
    policy: ResponsePolicy,
    preferred_style: Optional[str],
) -> None:
    """
    Maps the user's explicitly stated interaction style onto tone,
    but never for intents with a fixed, purpose-driven tone that
    matters for trust/safety (NEWS stays NEUTRAL; SHOPPING stays
    NEUTRAL; GAME stays PLAYFUL).
    """
    _FIXED_TONE_INTENTS = {"NEWS", "SHOPPING", "GAME", "UNKNOWN"}
    if policy.intent in _FIXED_TONE_INTENTS:
        return

    style_to_tone = {
        "FORMAL": "FORMAL",
        "CASUAL": "FRIENDLY",
        "FRIENDLY": "FRIENDLY",
        "CONCISE": "CONCISE",
        "PLAYFUL": "PLAYFUL",
    }
    mapped = style_to_tone.get(preferred_style)
    if mapped:
        policy.tone = mapped


def _apply_context_adjustments(policy: ResponsePolicy, context: PolicyContext) -> None:
    """
    Light contextual adjustments that don't depend on preferences at
    all — e.g. trimming verbosity in a long-running session, or never
    repeating an ASK_CLARIFYING loop indefinitely.
    """
    if context.turns_in_session > 15 and policy.response_length == "LONG":
        policy.response_length = "MEDIUM"
        policy.notes.append("Long session detected; trimming response length to stay concise.")

    if (
        policy.follow_up_behavior == "ASK_CLARIFYING"
        and context.recent_intents.count("UNKNOWN") >= 2
    ):
        policy.follow_up_behavior = "OPTIONAL_LIGHT"
        policy.notes.append(
            "Repeated unclear intents detected; avoiding another clarifying "
            "question loop in favor of a light, best-effort response."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_response_policy(
    intent: str,
    preference_score: float,
    user_profile: Any,
    context: Optional[PolicyContext] = None,
    category_lockin_detected: bool = False,
) -> ResponsePolicy:
    """
    Build a structured ResponsePolicy for the given (already-determined)
    intent.

    Parameters
    ----------
    intent: the final routed intent (ground truth; never changed here).
    preference_score: normalized preference weight (0.0-1.0) for this
        intent's category, as produced by preferences.get_preference.
    user_profile: duck-typed user profile object exposing
        `preferred_response_length` and `preferred_interaction_style`.
    context: optional PolicyContext with session/history signals.
    category_lockin_detected: optional explicit override if the caller
        (e.g. personalization.py) already computed lock-in detection;
        if provided, it is honored in addition to any lock-in implied
        by `context.category_lockin_detected`.

    Returns
    -------
    ResponsePolicy: structured instructions for a downstream response
    generator. This function never invents facts and never strengthens
    personalization beyond the user's explicit intent.
    """
    _validate_intent(intent)
    preference_score = _validate_preference_score(preference_score)
    context = context or PolicyContext()

    lockin = bool(category_lockin_detected) or bool(context.category_lockin_detected)

    profile_fields = _extract_user_profile_fields(user_profile)

    base_builder = _BASE_POLICY_BUILDERS[intent]
    policy = base_builder()

    # Apply bounded preference-based nudges (no-op for non-personalizable intents).
    _apply_preference_nudge(policy, preference_score, lockin)

    # Apply explicit user-stated style preferences (from UserProfile).
    _apply_response_length_preference(policy, profile_fields["preferred_response_length"])
    _apply_interaction_style_preference(policy, profile_fields["preferred_interaction_style"])

    # Apply session/context-driven adjustments.
    _apply_context_adjustments(policy, context)

    return policy


def build_response_policy_dict(
    intent: str,
    preference_score: float,
    user_profile: Any,
    context: Optional[PolicyContext] = None,
    category_lockin_detected: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper returning a plain dict for JSON serialization."""
    policy = build_response_policy(
        intent=intent,
        preference_score=preference_score,
        user_profile=user_profile,
        context=context,
        category_lockin_detected=category_lockin_detected,
    )
    return policy.to_dict()
