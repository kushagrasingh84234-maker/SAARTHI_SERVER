"""
preferences.py

Lightweight, pure-Python preference model for the StudyBot personalized
AI robot.

This module tracks how interested a user is in each supported category
over time, and exposes normalized weights that can be used to influence
RESPONSE STYLE, DEPTH, EXAMPLES, and RELATED RECOMMENDATIONS.

Hard rule: preferences must NEVER override explicit user intent/routing.
If a router (e.g. intent_router.py) determines the user asked for NEWS,
the request is handled as NEWS regardless of the user's EDUCATION
preference weight. This module only informs *how* a response is
shaped, never *what* category a request is routed to.

No database code. No WebSocket code. No AI/LLM API calls. Pure Python,
fully unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class Category(str, Enum):
    EDUCATION = "EDUCATION"
    GENERAL = "GENERAL"
    NEWS = "NEWS"
    INFORMATION = "INFORMATION"
    TECHNOLOGY = "TECHNOLOGY"
    SHOPPING = "SHOPPING"
    BUSINESS = "BUSINESS"
    ENTERTAINMENT = "ENTERTAINMENT"
    PERSONAL = "PERSONAL"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Raw default scores (unnormalized). These express relative starting
# interest across categories for a brand-new user profile. They are
# normalized on load/use, so the raw numbers only need to be internally
# consistent relative to one another, not sum to 100.
DEFAULT_RAW_PREFERENCES: Dict[str, float] = {
    Category.EDUCATION.value: 30.0,
    Category.GENERAL.value: 20.0,
    Category.INFORMATION.value: 15.0,
    Category.NEWS.value: 10.0,
    Category.TECHNOLOGY.value: 10.0,
    Category.SHOPPING.value: 5.0,
    Category.BUSINESS.value: 5.0,
    Category.ENTERTAINMENT.value: 5.0,
    Category.PERSONAL.value: 10.0,
}

_VALID_CATEGORIES = frozenset(c.value for c in Category)

# Tuning constants
_MIN_SCORE = 0.0
_MAX_SCORE = 1000.0          # hard ceiling to prevent unbounded growth
_MAX_STEP_RATIO = 0.20       # a single learning update can move a score
                              # by at most 20% of the current total mass
_DEFAULT_LEARNING_RATE = 0.10  # fraction of reinforcement applied per event
_MIN_NORMALIZED_FLOOR = 0.0


class PreferenceValidationError(ValueError):
    """Raised when preference data fails validation."""


@dataclass
class PreferenceProfile:
    """
    Holds raw (unnormalized) preference scores for one user.

    Raw scores are the internal "memory" of accumulated interest.
    Normalized weights (0-1, summing to 1.0) are what callers should
    use for shaping response style/depth/examples.
    """

    raw_scores: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RAW_PREFERENCES)
    )

    def __post_init__(self) -> None:
        validate_preferences(self.raw_scores)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_preferences(raw_scores: Dict[str, float]) -> None:
    """
    Validate a raw preference score dict.

    Raises PreferenceValidationError on any problem. Does not mutate
    the input.
    """
    if not isinstance(raw_scores, dict):
        raise PreferenceValidationError("raw_scores must be a dict")

    for category, score in raw_scores.items():
        if category not in _VALID_CATEGORIES:
            raise PreferenceValidationError(f"Unknown category: {category}")
        if not isinstance(score, (int, float)):
            raise PreferenceValidationError(
                f"Score for {category} must be numeric, got {type(score)}"
            )
        if score < _MIN_SCORE:
            raise PreferenceValidationError(
                f"Score for {category} cannot be negative: {score}"
            )
        if score > _MAX_SCORE:
            raise PreferenceValidationError(
                f"Score for {category} exceeds max allowed ({_MAX_SCORE}): {score}"
            )

    missing = _VALID_CATEGORIES - set(raw_scores.keys())
    if missing:
        raise PreferenceValidationError(
            f"Missing required categories: {sorted(missing)}"
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_preferences(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw accumulated scores into normalized weights that sum to 1.0.

    If all scores are zero (degenerate case), falls back to a uniform
    distribution across all categories so callers never divide by zero
    or receive an empty/skewed result.
    """
    validate_preferences(raw_scores)

    total = sum(raw_scores.values())
    if total <= 0:
        uniform = 1.0 / len(raw_scores)
        return {category: uniform for category in raw_scores}

    return {
        category: max(_MIN_NORMALIZED_FLOOR, score / total)
        for category, score in raw_scores.items()
    }


# ---------------------------------------------------------------------------
# Core preference operations
# ---------------------------------------------------------------------------

def get_default_profile() -> PreferenceProfile:
    """Return a fresh preference profile using system defaults."""
    return PreferenceProfile(raw_scores=dict(DEFAULT_RAW_PREFERENCES))


def get_preference(profile: PreferenceProfile, category: str) -> float:
    """
    Get the normalized weight (0.0-1.0) for a single category.

    Raises PreferenceValidationError if the category is unknown.
    """
    if category not in _VALID_CATEGORIES:
        raise PreferenceValidationError(f"Unknown category: {category}")

    normalized = normalize_preferences(profile.raw_scores)
    return normalized[category]


def get_all_preferences(profile: PreferenceProfile) -> Dict[str, float]:
    """Get normalized weights for every category, sorted by weight descending."""
    normalized = normalize_preferences(profile.raw_scores)
    return dict(
        sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    )


def update_preference(
    profile: PreferenceProfile,
    category: str,
    raw_value: float,
) -> PreferenceProfile:
    """
    Directly set the raw score for a category (e.g. admin/manual override,
    or restoring a saved profile). Value is clamped to the valid range.

    Returns the same profile instance, mutated, for convenience chaining.
    """
    if category not in _VALID_CATEGORIES:
        raise PreferenceValidationError(f"Unknown category: {category}")

    clamped = _clamp(float(raw_value), _MIN_SCORE, _MAX_SCORE)
    profile.raw_scores[category] = clamped
    validate_preferences(profile.raw_scores)
    return profile


def reset_preferences(profile: PreferenceProfile) -> PreferenceProfile:
    """Reset a profile back to system defaults in place."""
    profile.raw_scores = dict(DEFAULT_RAW_PREFERENCES)
    return profile


# ---------------------------------------------------------------------------
# Gradual learning from observed behavior
# ---------------------------------------------------------------------------
#
# IMPORTANT: This learning mechanism should be fed the CATEGORY THAT WAS
# ACTUALLY ROUTED (e.g. by intent_router.py) for a given interaction —
# never a preference-biased guess. Preferences observe and adapt to
# behavior; they do not decide behavior. Callers must not use this
# module's weights as an input to routing decisions.

def learn_from_interaction(
    profile: PreferenceProfile,
    observed_category: str,
    learning_rate: float = _DEFAULT_LEARNING_RATE,
) -> PreferenceProfile:
    """
    Nudge preferences based on one observed, already-routed interaction.

    This reinforces the observed category slightly and lets others
    decay slightly relative to it, while capping any single update so
    one interaction can never cause a runaway swing in preferences.

    - observed_category: the category the request was ACTUALLY routed
      to (never inferred from current preferences).
    - learning_rate: fraction (0-1) of the reinforcement step to apply.
      Smaller = slower, more stable learning.

    Returns the same profile instance, mutated.
    """
    if observed_category not in _VALID_CATEGORIES:
        raise PreferenceValidationError(f"Unknown category: {observed_category}")

    learning_rate = _clamp(float(learning_rate), 0.0, 1.0)

    total_mass = sum(profile.raw_scores.values())
    if total_mass <= 0:
        total_mass = float(len(profile.raw_scores))

    # Maximum absolute change allowed for the reinforced category in
    # this single update — prevents any one interaction from dominating.
    max_step = total_mass * _MAX_STEP_RATIO

    current = profile.raw_scores[observed_category]
    reinforcement = min(learning_rate * total_mass * 0.1, max_step)
    profile.raw_scores[observed_category] = _clamp(
        current + reinforcement, _MIN_SCORE, _MAX_SCORE
    )

    # Mild decay on all other categories, also capped, so the profile
    # gradually shifts toward observed behavior without erasing history.
    decay_rate = learning_rate * 0.05
    for category in profile.raw_scores:
        if category == observed_category:
            continue
        current_other = profile.raw_scores[category]
        max_decay_step = current_other * _MAX_STEP_RATIO
        decay_amount = min(current_other * decay_rate, max_decay_step)
        profile.raw_scores[category] = _clamp(
            current_other - decay_amount, _MIN_SCORE, _MAX_SCORE
        )

    validate_preferences(profile.raw_scores)
    return profile


# ---------------------------------------------------------------------------
# Response-shaping helpers (style/depth/examples only — NEVER routing)
# ---------------------------------------------------------------------------

def get_dominant_category(profile: PreferenceProfile) -> Optional[str]:
    """
    Return the category with the highest normalized weight.

    Intended ONLY for shaping style/depth/examples/related
    recommendations in a response that has already been routed to a
    specific intent by the router. This must never be used to decide
    what the user's request IS about.
    """
    normalized = normalize_preferences(profile.raw_scores)
    if not normalized:
        return None
    return max(normalized.items(), key=lambda item: item[1])[0]


def suggest_related_categories(
    profile: PreferenceProfile,
    exclude: Optional[str] = None,
    top_n: int = 2,
) -> Dict[str, float]:
    """
    Suggest up to `top_n` categories (by normalized weight) that could be
    offered as related/follow-up recommendations, excluding the category
    already being served for the current request.

    Purely additive/suggestive — never used to override or reroute the
    current request's intent.
    """
    normalized = get_all_preferences(profile)
    if exclude and exclude in normalized:
        normalized = {k: v for k, v in normalized.items() if k != exclude}

    return dict(list(normalized.items())[:max(0, top_n)])


def style_profile_for_category(
    profile: PreferenceProfile,
    routed_category: str,
) -> Dict[str, object]:
    """
    Build a small style-guidance dict for a response, given the category
    the router already decided on (`routed_category`).

    This NEVER changes `routed_category` itself — it only returns
    guidance on tone/depth/examples that the response generator MAY use.
    """
    if routed_category not in _VALID_CATEGORIES:
        raise PreferenceValidationError(f"Unknown category: {routed_category}")

    weight = get_preference(profile, routed_category)

    if weight >= 0.25:
        depth = "deep"
        examples = "rich"
    elif weight >= 0.10:
        depth = "moderate"
        examples = "some"
    else:
        depth = "brief"
        examples = "minimal"

    return {
        "routed_category": routed_category,
        "category_affinity": round(weight, 4),
        "suggested_depth": depth,
        "suggested_examples": examples,
        "related_recommendations": suggest_related_categories(
            profile, exclude=routed_category
        ),
}
