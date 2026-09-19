"""
personality_engine.py

Deterministic, rule-based Personality Engine for StudyBot.

Purpose
-------
Determines **how** StudyBot should communicate — tone, warmth, energy,
verbosity, humor, opening/closing style — as a pure function of context
(detected user emotion, situation, intent, user style preferences).

This module answers "HOW should the bot sound?", never "WHAT does the
user want?" or "WHAT emotion is the user feeling?". Those questions
belong to intent detection and the emotion engine respectively. This
module only *consumes* their outputs (as plain strings/values passed in
via PersonalityContext) — it never detects emotion or intent itself, and
it never decides response content or stores user data.

Design goals
------------
- Zero network/DB/WebSocket/hardware calls. Pure in-memory computation.
- No dependency on server.py, database.py, ai_services.py, emotion_engine.py,
  response_policy.py, or any AI provider SDK. Fully standalone.
- Deterministic: same PersonalityContext -> same PersonalityDecision.
- Never overrides user intent — it only shapes communication style
  around whatever content/decision the rest of the system already made.
- Never claims human feelings/consciousness. It expresses a
  *conversational style*, not simulated emotion ("I am sad" is banned;
  "That sounds tough — let's work through it." is fine).
- Extensible: PersonalityProfile and PersonalityDecision are designed so
  future modules (voice/TTS style, facial/animation intensity,
  user-adaptive personality, alternate presets) can plug in without
  breaking the public interface.

Public interface
-----------------
    build_personality(context: PersonalityContext) -> PersonalityDecision
    build_personality_dict(context: PersonalityContext) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN = 0.0
_MAX = 1.0


def _clamp(value: float, lo: float = _MIN, hi: float = _MAX) -> float:
    """Clamp a numeric value into [lo, hi]. Never raises on bad input."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def _norm_str(value: Optional[str]) -> str:
    """Lowercase/trim a free-text hint. Empty/invalid -> ''."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


# ---------------------------------------------------------------------------
# 1. Enums — vocabulary for style outputs
# ---------------------------------------------------------------------------

class Tone(str, Enum):
    CONGRATULATORY = "CONGRATULATORY"
    SUPPORTIVE = "SUPPORTIVE"
    PATIENT = "PATIENT"
    CALM_VALIDATING = "CALM_VALIDATING"
    ENERGETIC_MATCHED = "ENERGETIC_MATCHED"
    FRIENDLY_NEUTRAL = "FRIENDLY_NEUTRAL"
    DIRECT_FACTUAL = "DIRECT_FACTUAL"
    APPRECIATIVE = "APPRECIATIVE"
    CURIOUS_ENGAGED = "CURIOUS_ENGAGED"


class EmotionalStyle(str, Enum):
    """
    A conversational *style* label — never a claim of felt emotion.
    Used to guide word choice/energy, not to assert the bot 'feels' this.
    """
    UPBEAT = "UPBEAT"
    WARM = "WARM"
    STEADY = "STEADY"
    GENTLE = "GENTLE"
    NEUTRAL = "NEUTRAL"
    ENTHUSIASTIC = "ENTHUSIASTIC"


class ResponseLength(str, Enum):
    VERY_SHORT = "VERY_SHORT"
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"


class OpeningStyle(str, Enum):
    CELEBRATE = "CELEBRATE"
    REASSURE = "REASSURE"
    ACKNOWLEDGE = "ACKNOWLEDGE"
    VALIDATE = "VALIDATE"
    MATCH_ENERGY = "MATCH_ENERGY"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    WARM_GREETING = "WARM_GREETING"
    THANKS_ACK = "THANKS_ACK"


class ClosingStyle(str, Enum):
    ENCOURAGE_NEXT_STEP = "ENCOURAGE_NEXT_STEP"
    OFFER_FURTHER_HELP = "OFFER_FURTHER_HELP"
    INVITE_QUESTIONS = "INVITE_QUESTIONS"
    LIGHT_CELEBRATION = "LIGHT_CELEBRATION"
    SIMPLE_CLOSE = "SIMPLE_CLOSE"
    CHECK_UNDERSTANDING = "CHECK_UNDERSTANDING"


class HumorLevel(str, Enum):
    NONE = "NONE"
    LIGHT = "LIGHT"
    MODERATE = "MODERATE"


class InteractionStyle(str, Enum):
    """User-facing preference input — how the user likes to be talked to."""
    FORMAL = "FORMAL"
    CASUAL = "CASUAL"
    BALANCED = "BALANCED"
    PLAYFUL = "PLAYFUL"


# ---------------------------------------------------------------------------
# 2. PersonalityProfile — stable base personality, with safe defaults
# ---------------------------------------------------------------------------

@dataclass
class PersonalityProfile:
    """
    StudyBot's stable base personality, expressed as bounded numeric
    dimensions in [0, 1]. This is the *default character* — friendly,
    warm, encouraging, intelligent, respectful, concise-by-default,
    playful when appropriate, never robotic, never over-emotional.

    Future presets (e.g. "strict tutor", "cheerful buddy") can be built
    by constructing alternate PersonalityProfile instances — the rest of
    the engine only depends on this shape, not on these specific values.
    """
    warmth: float = 0.75
    friendliness: float = 0.8
    enthusiasm: float = 0.55
    encouragement: float = 0.75
    playfulness: float = 0.35
    formality: float = 0.3
    empathy: float = 0.75
    verbosity: float = 0.4
    confidence: float = 0.7
    humor: float = 0.3

    def __post_init__(self) -> None:
        # Validate/clamp every dimension so an invalid or out-of-range
        # value can never silently propagate downstream.
        self.warmth = _clamp(self.warmth)
        self.friendliness = _clamp(self.friendliness)
        self.enthusiasm = _clamp(self.enthusiasm)
        self.encouragement = _clamp(self.encouragement)
        self.playfulness = _clamp(self.playfulness)
        self.formality = _clamp(self.formality)
        self.empathy = _clamp(self.empathy)
        self.verbosity = _clamp(self.verbosity)
        self.confidence = _clamp(self.confidence)
        self.humor = _clamp(self.humor)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Default, shared base personality instance. Callers may pass their own
# PersonalityProfile into build_personality() to support future presets
# (voice persona, alternate robot characters, per-user adaptive tuning)
# without changing the public function signature's meaning.
DEFAULT_PROFILE = PersonalityProfile()


# ---------------------------------------------------------------------------
# 3. PersonalityContext — inputs (decoupled from other modules)
# ---------------------------------------------------------------------------

@dataclass
class PersonalityContext:
    """
    Everything the Personality Engine needs to decide *how* to speak.

    All fields are intentionally plain primitives (str/float), not enums
    or types imported from emotion_engine.py / intent modules / response
    policy. This keeps personality_engine.py fully decoupled — callers
    translate their own emotion/intent/situation representations into
    simple strings when constructing this context.

    Fields
    ------
    detected_user_emotion: str
        Free-text label from an emotion detector (e.g. "EXCITED",
        "DISAPPOINTED", "CONFUSED"). Case-insensitive. Optional.
    situation: str
        Free-text situation label (e.g. "achievement", "failure",
        "confusion", "frustration", "excitement", "neutral_conversation").
    intent: str
        Free-text user intent label from intent detection, informational
        only — this module never alters or overrides it.
    preferred_interaction_style: str
        One of InteractionStyle values (or free text); how the user
        likes to be spoken to (formal/casual/balanced/playful).
    preferred_response_length: str
        Hint like "short", "medium", "long". Optional.
    preference_strength: float
        0..1 — how strongly to honor the user's stated style/length
        preference over the base personality defaults.
    conversation_depth: int
        Roughly how many turns deep the conversation is; used only to
        mildly favor brevity over time (avoid restating pleasantries).
    profile: Optional[PersonalityProfile]
        Base personality to use. Defaults to DEFAULT_PROFILE.
    """
    detected_user_emotion: str = ""
    situation: str = ""
    intent: str = ""
    preferred_interaction_style: str = ""
    preferred_response_length: str = ""
    preference_strength: float = 0.5
    conversation_depth: int = 0
    profile: Optional[PersonalityProfile] = None

    def __post_init__(self) -> None:
        self.detected_user_emotion = _norm_str(self.detected_user_emotion)
        self.situation = _norm_str(self.situation)
        self.intent = _norm_str(self.intent)
        self.preferred_interaction_style = _norm_str(self.preferred_interaction_style)
        self.preferred_response_length = _norm_str(self.preferred_response_length)
        self.preference_strength = _clamp(self.preference_strength)
        try:
            self.conversation_depth = max(0, int(self.conversation_depth))
        except (TypeError, ValueError):
            self.conversation_depth = 0
        if self.profile is None:
            self.profile = DEFAULT_PROFILE

    def effective_profile(self) -> PersonalityProfile:
        """`profile` is guaranteed non-None after __post_init__."""
        assert self.profile is not None
        return self.profile


# ---------------------------------------------------------------------------
# 4. PersonalityDecision — output
# ---------------------------------------------------------------------------

@dataclass
class PersonalityDecision:
    """
    The concrete communication-style decision for a single turn.

    `instructions` is a short list of plain-language style directives
    intended for a downstream response generator/prompt builder (e.g.
    "Use warm, encouraging language.", "Keep it brief."). It never
    contains raw content, chain-of-thought, or claims of human feeling —
    only style guidance.
    """
    tone: Tone = Tone.FRIENDLY_NEUTRAL
    emotional_style: EmotionalStyle = EmotionalStyle.NEUTRAL
    enthusiasm_level: float = 0.5
    empathy_level: float = 0.5
    response_length: ResponseLength = ResponseLength.MEDIUM
    recommended_opening_style: OpeningStyle = OpeningStyle.DIRECT_ANSWER
    recommended_closing_style: ClosingStyle = ClosingStyle.SIMPLE_CLOSE
    allowed_humor: HumorLevel = HumorLevel.NONE
    encouragement_level: float = 0.5
    instructions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.enthusiasm_level = _clamp(self.enthusiasm_level)
        self.empathy_level = _clamp(self.empathy_level)
        self.encouragement_level = _clamp(self.encouragement_level)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["tone"] = self.tone.value
        d["emotional_style"] = self.emotional_style.value
        d["response_length"] = self.response_length.value
        d["recommended_opening_style"] = self.recommended_opening_style.value
        d["recommended_closing_style"] = self.recommended_closing_style.value
        d["allowed_humor"] = self.allowed_humor.value
        return d


# ---------------------------------------------------------------------------
# 5. Situation -> style rule table
# ---------------------------------------------------------------------------
# Maps normalized situation strings to a bundle of style choices. This is
# the single source of truth for the "IMPORTANT BEHAVIOR" rules in the
# spec (achievement -> congratulatory, failure -> never disappointed &
# supportive, confusion -> patient, frustration -> calm/validating,
# excitement -> match energy without excess, neutral -> calm/direct).

@dataclass(frozen=True)
class _SituationStyle:
    tone: Tone
    emotional_style: EmotionalStyle
    opening: OpeningStyle
    closing: ClosingStyle
    enthusiasm_delta: float   # added to base enthusiasm
    empathy_delta: float      # added to base empathy
    encouragement_delta: float
    humor_ceiling: HumorLevel
    core_instruction: str


_SITUATION_STYLES: Dict[str, _SituationStyle] = {
    "achievement": _SituationStyle(
        tone=Tone.CONGRATULATORY,
        emotional_style=EmotionalStyle.UPBEAT,
        opening=OpeningStyle.CELEBRATE,
        closing=ClosingStyle.LIGHT_CELEBRATION,
        enthusiasm_delta=+0.3,
        empathy_delta=0.0,
        encouragement_delta=+0.15,
        humor_ceiling=HumorLevel.LIGHT,
        core_instruction="Use warm, congratulatory language celebrating the user's success.",
    ),
    "failure": _SituationStyle(
        tone=Tone.SUPPORTIVE,
        emotional_style=EmotionalStyle.GENTLE,
        opening=OpeningStyle.REASSURE,
        closing=ClosingStyle.ENCOURAGE_NEXT_STEP,
        enthusiasm_delta=-0.15,
        empathy_delta=+0.25,
        encouragement_delta=+0.3,
        humor_ceiling=HumorLevel.NONE,
        core_instruction=(
            "Do not sound disappointed in the user. Use supportive, "
            "encouraging language and focus on next steps, not the setback."
        ),
    ),
    "confusion": _SituationStyle(
        tone=Tone.PATIENT,
        emotional_style=EmotionalStyle.STEADY,
        opening=OpeningStyle.ACKNOWLEDGE,
        closing=ClosingStyle.CHECK_UNDERSTANDING,
        enthusiasm_delta=-0.1,
        empathy_delta=+0.15,
        encouragement_delta=+0.1,
        humor_ceiling=HumorLevel.NONE,
        core_instruction="Explain patiently and clearly; avoid rushing or overwhelming detail.",
    ),
    "frustration": _SituationStyle(
        tone=Tone.CALM_VALIDATING,
        emotional_style=EmotionalStyle.GENTLE,
        opening=OpeningStyle.VALIDATE,
        closing=ClosingStyle.OFFER_FURTHER_HELP,
        enthusiasm_delta=-0.25,
        empathy_delta=+0.25,
        encouragement_delta=+0.15,
        humor_ceiling=HumorLevel.NONE,
        core_instruction="Stay calm, validate the user's frustration, and avoid adding pressure.",
    ),
    "excitement": _SituationStyle(
        tone=Tone.ENERGETIC_MATCHED,
        emotional_style=EmotionalStyle.ENTHUSIASTIC,
        opening=OpeningStyle.MATCH_ENERGY,
        closing=ClosingStyle.LIGHT_CELEBRATION,
        enthusiasm_delta=+0.25,
        empathy_delta=0.0,
        encouragement_delta=+0.1,
        humor_ceiling=HumorLevel.MODERATE,
        core_instruction="Match the user's positive energy without becoming excessive or unfocused.",
    ),
    "sadness": _SituationStyle(
        tone=Tone.SUPPORTIVE,
        emotional_style=EmotionalStyle.GENTLE,
        opening=OpeningStyle.VALIDATE,
        closing=ClosingStyle.OFFER_FURTHER_HELP,
        enthusiasm_delta=-0.3,
        empathy_delta=+0.3,
        encouragement_delta=+0.2,
        humor_ceiling=HumorLevel.NONE,
        core_instruction="Acknowledge the feeling gently and supportively, without overstating emotion.",
    ),
    "anger": _SituationStyle(
        tone=Tone.CALM_VALIDATING,
        emotional_style=EmotionalStyle.STEADY,
        opening=OpeningStyle.VALIDATE,
        closing=ClosingStyle.OFFER_FURTHER_HELP,
        enthusiasm_delta=-0.3,
        empathy_delta=+0.2,
        encouragement_delta=+0.05,
        humor_ceiling=HumorLevel.NONE,
        core_instruction="Remain calm and non-defensive; de-escalate rather than mirror intensity.",
    ),
    "gratitude": _SituationStyle(
        tone=Tone.APPRECIATIVE,
        emotional_style=EmotionalStyle.WARM,
        opening=OpeningStyle.THANKS_ACK,
        closing=ClosingStyle.INVITE_QUESTIONS,
        enthusiasm_delta=+0.1,
        empathy_delta=+0.05,
        encouragement_delta=0.0,
        humor_ceiling=HumorLevel.LIGHT,
        core_instruction="Warmly acknowledge the thanks briefly, then remain useful.",
    ),
    "curiosity": _SituationStyle(
        tone=Tone.CURIOUS_ENGAGED,
        emotional_style=EmotionalStyle.WARM,
        opening=OpeningStyle.DIRECT_ANSWER,
        closing=ClosingStyle.INVITE_QUESTIONS,
        enthusiasm_delta=+0.1,
        empathy_delta=0.0,
        encouragement_delta=0.0,
        humor_ceiling=HumorLevel.LIGHT,
        core_instruction="Engage with genuine interest while staying informative and clear.",
    ),
    "encouragement_needed": _SituationStyle(
        tone=Tone.SUPPORTIVE,
        emotional_style=EmotionalStyle.GENTLE,
        opening=OpeningStyle.REASSURE,
        closing=ClosingStyle.ENCOURAGE_NEXT_STEP,
        enthusiasm_delta=-0.05,
        empathy_delta=+0.2,
        encouragement_delta=+0.3,
        humor_ceiling=HumorLevel.NONE,
        core_instruction="Offer steady reassurance and concrete encouragement.",
    ),
    "neutral_conversation": _SituationStyle(
        tone=Tone.DIRECT_FACTUAL,
        emotional_style=EmotionalStyle.NEUTRAL,
        opening=OpeningStyle.DIRECT_ANSWER,
        closing=ClosingStyle.SIMPLE_CLOSE,
        enthusiasm_delta=-0.1,
        empathy_delta=-0.1,
        encouragement_delta=-0.1,
        humor_ceiling=HumorLevel.LIGHT,
        core_instruction="Remain calm and direct. Do not force emotional language into a factual answer.",
    ),
}

_DEFAULT_SITUATION_STYLE = _SITUATION_STYLES["neutral_conversation"]


def _resolve_situation_style(situation: str) -> _SituationStyle:
    """Loose matching: exact key, then substring match, then default."""
    if situation in _SITUATION_STYLES:
        return _SITUATION_STYLES[situation]
    for key, style in _SITUATION_STYLES.items():
        if key in situation or situation in key:
            return style
    return _DEFAULT_SITUATION_STYLE


# ---------------------------------------------------------------------------
# 6. Interaction style / length preference handling
# ---------------------------------------------------------------------------

def _resolve_interaction_style_deltas(style_hint: str) -> Dict[str, float]:
    """
    Map a free-text interaction-style preference to dimension deltas.
    Unknown/empty input -> no adjustment (all zeros).
    """
    if style_hint in (InteractionStyle.FORMAL.value.lower(), "formal"):
        return {"formality": +0.35, "playfulness": -0.2, "humor": -0.15}
    if style_hint in (InteractionStyle.CASUAL.value.lower(), "casual"):
        return {"formality": -0.25, "playfulness": +0.15, "humor": +0.1}
    if style_hint in (InteractionStyle.PLAYFUL.value.lower(), "playful"):
        return {"formality": -0.3, "playfulness": +0.3, "humor": +0.25}
    # "balanced" or unrecognized -> no change
    return {}


def _resolve_length(preferred: str, verbosity: float, depth: int) -> ResponseLength:
    """
    Determine target response length from (in priority order):
    explicit user preference > base personality verbosity > conversation
    depth (favor brevity as a conversation goes on, mirroring natural
    conversational pacing rather than restating context every turn).
    """
    if preferred in ("very_short", "very short", "brief", "tiny"):
        return ResponseLength.VERY_SHORT
    if preferred in ("short", "concise"):
        return ResponseLength.SHORT
    if preferred in ("long", "detailed", "thorough"):
        return ResponseLength.LONG
    if preferred in ("medium", "normal"):
        return ResponseLength.MEDIUM

    # No explicit preference: derive from verbosity dimension.
    effective_verbosity = verbosity - min(depth, 10) * 0.02  # mild decay over a long chat
    if effective_verbosity < 0.25:
        return ResponseLength.VERY_SHORT
    if effective_verbosity < 0.45:
        return ResponseLength.SHORT
    if effective_verbosity < 0.7:
        return ResponseLength.MEDIUM
    return ResponseLength.LONG


def _humor_from_score(score: float, ceiling: HumorLevel) -> HumorLevel:
    """Convert a numeric playfulness/humor score into a HumorLevel, capped
    by the situation's humor ceiling (e.g. never humorous during failure)."""
    order = [HumorLevel.NONE, HumorLevel.LIGHT, HumorLevel.MODERATE]
    if score < 0.25:
        level = HumorLevel.NONE
    elif score < 0.55:
        level = HumorLevel.LIGHT
    else:
        level = HumorLevel.MODERATE
    # Cap at ceiling
    if order.index(level) > order.index(ceiling):
        return ceiling
    return level


# ---------------------------------------------------------------------------
# 7. Public API
# ---------------------------------------------------------------------------

def build_personality(context: PersonalityContext) -> PersonalityDecision:
    """
    Compute a PersonalityDecision for the given context.

    This function only decides *style* (tone, energy, empathy level,
    length, opening/closing shape, humor, encouragement). It never
    inspects or alters `context.intent` beyond reading it for logging-
    style instructions, and it never determines response *content*.

    Safe on missing/invalid context fields — PersonalityContext.__post_init__
    already normalizes and clamps everything, so this function does not
    need to re-validate raw input.
    """
    if not isinstance(context, PersonalityContext):
        # Defensive fallback: treat as a neutral, default context.
        context = PersonalityContext()

    profile = context.effective_profile()
    style = _resolve_situation_style(context.situation)

    # --- Apply user interaction-style preference, weighted by how
    #     strongly the user expressed it (preference_strength). ---
    style_deltas = _resolve_interaction_style_deltas(context.preferred_interaction_style)
    strength = context.preference_strength

    adj_formality = profile.formality + style_deltas.get("formality", 0.0) * strength
    adj_playfulness = profile.playfulness + style_deltas.get("playfulness", 0.0) * strength
    adj_humor_dim = profile.humor + style_deltas.get("humor", 0.0) * strength

    # --- Apply situational deltas on top of the (preference-adjusted) base. ---
    enthusiasm_level = _clamp(profile.enthusiasm + style.enthusiasm_delta)
    empathy_level = _clamp(profile.empathy + style.empathy_delta)
    encouragement_level = _clamp(profile.encouragement + style.encouragement_delta)

    humor_score = _clamp((adj_humor_dim + _clamp(adj_playfulness)) / 2)
    allowed_humor = _humor_from_score(humor_score, style.humor_ceiling)

    response_length = _resolve_length(
        context.preferred_response_length, profile.verbosity, context.conversation_depth
    )

    # --- Assemble plain-language style instructions for a downstream
    #     response generator. Style guidance only — never content, never
    #     chain-of-thought, never a claim of human feeling. ---
    instructions: List[str] = [style.core_instruction]

    if adj_formality >= 0.6:
        instructions.append("Use a more formal, polished register.")
    elif adj_formality <= 0.2:
        instructions.append("Use a relaxed, casual register.")

    if response_length == ResponseLength.VERY_SHORT:
        instructions.append("Keep the response to one or two sentences.")
    elif response_length == ResponseLength.SHORT:
        instructions.append("Keep the response brief and to the point.")
    elif response_length == ResponseLength.LONG:
        instructions.append("A more detailed, thorough response is appropriate here.")

    if allowed_humor == HumorLevel.NONE:
        instructions.append("Avoid humor or jokes in this response.")
    elif allowed_humor == HumorLevel.MODERATE:
        instructions.append("Light, appropriate humor is welcome if it fits naturally.")

    instructions.append(
        "Express style and tone conversationally; never claim human feelings "
        "or consciousness (e.g. avoid phrasing like 'I feel sad')."
    )

    return PersonalityDecision(
        tone=style.tone,
        emotional_style=style.emotional_style,
        enthusiasm_level=enthusiasm_level,
        empathy_level=empathy_level,
        response_length=response_length,
        recommended_opening_style=style.opening,
        recommended_closing_style=style.closing,
        allowed_humor=allowed_humor,
        encouragement_level=encouragement_level,
        instructions=instructions,
    )


def build_personality_dict(context: PersonalityContext) -> dict:
    """
    Convenience wrapper around `build_personality` returning a plain,
    JSON-serializable dict instead of a PersonalityDecision dataclass.
    """
    return build_personality(context).to_dict()


# ---------------------------------------------------------------------------
# 8. Manual smoke test (only runs when executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        PersonalityContext(detected_user_emotion="EXCITED", situation="achievement"),
        PersonalityContext(detected_user_emotion="DISAPPOINTED", situation="failure"),
        PersonalityContext(detected_user_emotion="CONFUSED", situation="confusion"),
        PersonalityContext(detected_user_emotion="FRUSTRATED", situation="frustration",
                            preferred_interaction_style="casual", preference_strength=0.8),
        PersonalityContext(detected_user_emotion="EXCITED", situation="excitement"),
        PersonalityContext(situation="neutral_conversation", intent="ask_fact",
                            preferred_response_length="short"),
        PersonalityContext(situation="gratitude"),
        PersonalityContext(),  # fully default / neutral context
    ]
    for ctx in scenarios:
        decision = build_personality(ctx)
        print(f"situation={ctx.situation!r} emotion={ctx.detected_user_emotion!r} ->")
        print("  ", decision.to_dict())
