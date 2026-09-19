"""
personalization.py

Adaptive Personalization Engine for StudyBot.

Combines:
  1. User message intent           (intent_router.route_intent)
  2. User profile                  (user_profile.UserProfile)
  3. Historical interaction behavior (profile.recent_intents / category_counts)
  4. Category preferences          (preferences.PreferenceProfile)
  5. Current conversation context  (caller-supplied ConversationContext)

HARD RULE: this engine NEVER overrides explicit user intent. The intent
produced by the router is treated as ground truth and is echoed back
unchanged. Personalization only ever shapes HOW a response is built
(style, depth, examples, follow-ups, recommendation priorities) —
never WHAT the request is routed as.

This module is intentionally independent of WebSocket transport, Fast
Math, database persistence, and any external/AI API. It operates on
plain Python objects and dicts so the underlying AI service or
front-end transport can be swapped freely without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from intent_router import IntentResult, Intent, route_intent
from preferences import (
    PreferenceProfile,
    PreferenceValidationError,
    get_preference,
    get_all_preferences,
    get_dominant_category,
    suggest_related_categories,
    style_profile_for_category,
    learn_from_interaction,
)
from user_profile import (
    UserProfile,
    ProfileValidationError,
    update_category_activity,
    record_recent_intent,
    get_recent_intents,
    get_top_categories,
)


# ---------------------------------------------------------------------------
# Conversation context (lightweight, caller-supplied, no persistence)
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    """
    Snapshot of the current conversation, supplied by the caller
    (e.g. the chat-flow orchestrator). This module never fetches or
    stores this itself — it only reads what it's given.
    """

    last_intents: List[str] = field(default_factory=list)  # most recent last
    turns_in_session: int = 0
    last_recommended_categories: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safeguard constants
# ---------------------------------------------------------------------------

_MAX_PREFERENCE_INFLUENCE = 0.65          # cap on how much preference score
                                            # can push style intensity (0-1)
_CATEGORY_LOCKIN_WINDOW = 5                 # look back this many recent intents
_CATEGORY_LOCKIN_THRESHOLD = 4              # if >= this many of the window are
                                            # the same category, dampen it
_REPETITIVE_RECOMMENDATION_WINDOW = 3       # recent recommendations to compare against
_MAX_SINGLE_TURN_PREFERENCE_DELTA = 0.20    # sudden-change guard, informational
_MIN_VALID_INTENT_CONFIDENCE = 0.0          # engine accepts UNKNOWN too


class PersonalizationError(ValueError):
    """Raised when inputs to the personalization engine are invalid."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PersonalizationResult:
    intent: str
    confidence: float
    signals: List[str]
    preference_score: float
    response_profile: Dict[str, Any]
    personalization_instructions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
            "preference_score": round(self.preference_score, 4),
            "response_profile": self.response_profile,
            "personalization_instructions": self.personalization_instructions,
        }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    message: str,
    user_profile: UserProfile,
    preference_profile: PreferenceProfile,
) -> None:
    if not isinstance(message, str) or not message.strip():
        raise PersonalizationError("message must be a non-empty string")

    if not isinstance(user_profile, UserProfile):
        raise PersonalizationError("user_profile must be a UserProfile instance")

    if not isinstance(preference_profile, PreferenceProfile):
        raise PersonalizationError(
            "preference_profile must be a PreferenceProfile instance"
        )

    try:
        get_all_preferences(preference_profile)
    except PreferenceValidationError as exc:
        raise PersonalizationError(f"Invalid preference_profile: {exc}") from exc


# ---------------------------------------------------------------------------
# Safeguard 1: preference domination cap
# ---------------------------------------------------------------------------

def _capped_preference_influence(raw_weight: float) -> float:
    """
    Prevent a single dominant preference from having outsized influence
    on response style. The RAW weight is still reported truthfully in
    the result (`preference_score`); this cap only affects how strongly
    style/depth decisions react to it.
    """
    return min(raw_weight, _MAX_PREFERENCE_INFLUENCE)


# ---------------------------------------------------------------------------
# Safeguard 2: category lock-in detection
# ---------------------------------------------------------------------------

def _detect_category_lockin(recent_intents: List[str], routed_intent: str) -> bool:
    """
    Detect whether the user's recent history is dominated by a single
    category, which risks the engine over-personalizing toward it
    (e.g. always assuming EDUCATION framing even as topics vary).

    Returns True if lock-in is detected for the routed intent's
    dominant neighbor — used to soften (not remove) style intensity.
    """
    if len(recent_intents) < _CATEGORY_LOCKIN_WINDOW:
        return False

    window = recent_intents[-_CATEGORY_LOCKIN_WINDOW:]
    counts: Dict[str, int] = {}
    for intent in window:
        counts[intent] = counts.get(intent, 0) + 1

    if not counts:
        return False

    dominant_category, dominant_count = max(counts.items(), key=lambda kv: kv[1])
    return dominant_count >= _CATEGORY_LOCKIN_THRESHOLD and dominant_category == routed_intent


# ---------------------------------------------------------------------------
# Safeguard 3: repetitive recommendation avoidance
# ---------------------------------------------------------------------------

def _filter_repetitive_recommendations(
    candidates: Dict[str, float],
    last_recommended: List[str],
) -> Dict[str, float]:
    """
    Drop recommendation candidates that were already suggested in the
    last few turns, so the engine doesn't nag the user with the same
    follow-up suggestion repeatedly.
    """
    recent_set = set(last_recommended[-_REPETITIVE_RECOMMENDATION_WINDOW:])
    filtered = {
        category: weight
        for category, weight in candidates.items()
        if category not in recent_set
    }
    # If filtering removed everything, fall back to the original
    # candidates rather than returning nothing useful.
    return filtered if filtered else candidates


# ---------------------------------------------------------------------------
# Safeguard 4: sudden preference change dampening
# ---------------------------------------------------------------------------

def _dampen_sudden_change(
    previous_weight: Optional[float],
    current_weight: float,
) -> float:
    """
    If a preference weight jumped sharply since the last known value
    (e.g. due to an upstream bug or burst of similar messages), clamp
    the effective weight used THIS turn to move at most
    _MAX_SINGLE_TURN_PREFERENCE_DELTA from the previous value. This
    only affects the effective weight used for style shaping this
    turn — it does not mutate the underlying PreferenceProfile.
    """
    if previous_weight is None:
        return current_weight

    delta = current_weight - previous_weight
    if abs(delta) <= _MAX_SINGLE_TURN_PREFERENCE_DELTA:
        return current_weight

    direction = 1.0 if delta > 0 else -1.0
    return previous_weight + (direction * _MAX_SINGLE_TURN_PREFERENCE_DELTA)


# ---------------------------------------------------------------------------
# Personalization instruction builder
# ---------------------------------------------------------------------------

def _build_instructions(
    routed_intent: str,
    effective_weight: float,
    style_profile: Dict[str, Any],
    lockin_detected: bool,
    related_recommendations: Dict[str, float],
    dominant_category: Optional[str],
) -> List[str]:
    instructions: List[str] = []

    instructions.append(f"Respond primarily as intent={routed_intent}; do not reframe the topic.")

    depth = style_profile.get("suggested_depth", "moderate")
    examples = style_profile.get("suggested_examples", "some")

    if lockin_detected:
        instructions.append(
            "User has shown heavy recent focus on this category; "
            "keep explanation grounded in the current topic without "
            "forcing extra thematic callbacks to avoid over-personalization."
        )
        instructions.append("Use moderate depth and standard examples to avoid overfitting to history.")
    else:
        instructions.append(f"Use {depth} depth for this response.")
        instructions.append(f"Include {examples} illustrative examples where natural.")

    if routed_intent == Intent.EDUCATION.value and effective_weight >= 0.30 and not lockin_detected:
        instructions.append(
            "User shows strong educational affinity; frame explanations "
            "with a teaching tone and step-by-step structure when helpful."
        )

    if dominant_category and dominant_category != routed_intent and not lockin_detected:
        instructions.append(
            f"User's general interests skew toward {dominant_category}; "
            "a brief, optional connection MAY be offered but must not "
            "replace or dilute the direct answer to the current request."
        )

    if related_recommendations:
        top_related = ", ".join(related_recommendations.keys())
        instructions.append(
            f"If appropriate, offer at most one lightweight follow-up "
            f"related to: {top_related}. Do not force it."
        )
    else:
        instructions.append("No follow-up recommendation is necessary this turn.")

    instructions.append(
        "Preferences must influence tone/depth/examples only — never "
        "the determined intent or factual content of the answer."
    )

    return instructions


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def personalize(
    message: str,
    user_profile: UserProfile,
    preference_profile: PreferenceProfile,
    context: Optional[ConversationContext] = None,
    previous_effective_weight: Optional[float] = None,
    record_history: bool = True,
    learn: bool = True,
) -> PersonalizationResult:
    """
    Run the full personalization pipeline:

        message -> intent -> user preferences -> contextual adjustment
                -> response personalization instructions

    Parameters
    ----------
    message: the raw user message.
    user_profile: the user's UserProfile (interaction history, style prefs).
    preference_profile: the user's PreferenceProfile (category weights).
    context: optional ConversationContext for the current session.
    previous_effective_weight: optional last-known effective weight for
        the eventual routed category, used only for sudden-change damping.
    record_history: if True, updates user_profile's category counters
        and recent-intent history as a side effect of this call.
    learn: if True, feeds the routed intent into preference_profile's
        gradual learning mechanism as a side effect of this call.

    Returns
    -------
    PersonalizationResult — intent is guaranteed to match exactly what
    the router determined; personalization never changes it.
    """
    _validate_inputs(message, user_profile, preference_profile)
    context = context or ConversationContext()

    # Step 1: intent (ground truth, never overridden)
    intent_result: IntentResult = route_intent(message)
    routed_intent = intent_result.intent

    # Only categories the preference model understands can be scored;
    # GAME and UNKNOWN are valid routing outcomes but sit outside the
    # preference taxonomy, so we treat their preference score as neutral.
    try:
        raw_weight = get_preference(preference_profile, routed_intent)
    except PreferenceValidationError:
        raw_weight = 0.0

    # Safeguard 1: preference domination cap
    capped_weight = _capped_preference_influence(raw_weight)

    # Safeguard 4: sudden preference change dampening
    effective_weight = _dampen_sudden_change(previous_effective_weight, capped_weight)

    # Step 2 + 3: contextual adjustment using history + live context
    recent_intents = get_recent_intents(user_profile) + list(context.last_intents)
    lockin_detected = _detect_category_lockin(recent_intents, routed_intent)

    dominant_category = get_dominant_category(preference_profile)

    try:
        style_profile = style_profile_for_category(preference_profile, routed_intent)
    except PreferenceValidationError:
        # Categories outside the preference taxonomy (GAME, UNKNOWN)
        # get a neutral, safe default style profile.
        style_profile = {
            "routed_category": routed_intent,
            "category_affinity": 0.0,
            "suggested_depth": "moderate",
            "suggested_examples": "some",
            "related_recommendations": {},
        }

    if lockin_detected:
        # Soften style intensity so the engine doesn't keep pushing the
        # same framing turn after turn.
        style_profile["suggested_depth"] = "moderate"
        style_profile["suggested_examples"] = "some"

    related_candidates = suggest_related_categories(
        preference_profile, exclude=routed_intent, top_n=3
    )
    # Safeguard 3: repetitive recommendation avoidance
    related_recommendations = _filter_repetitive_recommendations(
        related_candidates, context.last_recommended_categories
    )
    # Keep at most 2 after filtering to avoid overwhelming the user.
    related_recommendations = dict(list(related_recommendations.items())[:2])

    # Step 4: build personalization instructions
    instructions = _build_instructions(
        routed_intent=routed_intent,
        effective_weight=effective_weight,
        style_profile=style_profile,
        lockin_detected=lockin_detected,
        related_recommendations=related_recommendations,
        dominant_category=dominant_category,
    )

    response_profile = {
        "suggested_depth": style_profile.get("suggested_depth"),
        "suggested_examples": style_profile.get("suggested_examples"),
        "preferred_response_length": user_profile.preferred_response_length,
        "preferred_interaction_style": user_profile.preferred_interaction_style,
        "related_recommendations": related_recommendations,
        "category_lockin_detected": lockin_detected,
        "dominant_interest_category": dominant_category,
    }

    # Side effects: update history / learning. Kept optional so this
    # function can be called in a dry-run/preview mode without mutating
    # state (e.g. for tests or admin tooling).
    if record_history:
        try:
            update_category_activity(user_profile, routed_intent)
            record_recent_intent(user_profile, routed_intent)
        except ProfileValidationError:
            # Category not recognized by the profile's category set
            # (shouldn't happen given shared taxonomy) — skip silently
            # rather than failing the whole personalization pipeline.
            pass

    if learn:
        try:
            learn_from_interaction(preference_profile, routed_intent)
        except PreferenceValidationError:
            pass

    return PersonalizationResult(
        intent=routed_intent,
        confidence=intent_result.confidence,
        signals=intent_result.signals,
        preference_score=raw_weight,
        response_profile=response_profile,
        personalization_instructions=instructions,
    )


def personalize_dict(
    message: str,
    user_profile: UserProfile,
    preference_profile: PreferenceProfile,
    context: Optional[ConversationContext] = None,
    previous_effective_weight: Optional[float] = None,
    record_history: bool = True,
    learn: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper returning a plain dict for JSON serialization."""
    result = personalize(
        message=message,
        user_profile=user_profile,
        preference_profile=preference_profile,
        context=context,
        previous_effective_weight=previous_effective_weight,
        record_history=record_history,
        learn=learn,
    )
    return result.to_dict()
