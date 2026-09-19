"""
emotion_engine.py

Deterministic, rule-based Emotion Engine for StudyBot.

Purpose
-------
Infers lightweight *conversational* emotional signals from a user's text
message. This module is NOT a medical, psychological, or clinical
diagnostic tool. It does not detect, store, or reason about mental health
conditions — it only classifies short-term conversational tone (e.g. did
the user sound excited, frustrated, confused) so the bot can pick an
appropriate tone in response.

Design goals
------------
- Zero network calls. Zero DB calls. Zero WebSocket usage.
- Fully deterministic (same input -> same output), unit-test friendly.
- Clear separation of concerns:
    1. DETECTION      -> raw signal extraction from text
    2. INTERPRETATION -> turning signals into an EmotionResult
    3. RECOMMENDATION -> mapping user emotion/situation to a robot
                          response emotion + response mode
- Safe on empty/invalid input (always resolves to NEUTRAL, never raises
  on bad input types).
- Does not persist any data, does not touch user profiles/personality/
  memory/response-policy subsystems. This module is intentionally
  self-contained and side-effect free so it can later be swapped for an
  ML/LLM-based classifier without changing its public interface.

Public interface
-----------------
    analyze_emotion(message: str) -> EmotionResult
    analyze_emotion_dict(message: str) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Tuple
import re


# ---------------------------------------------------------------------------
# 1. Enums / Constants
# ---------------------------------------------------------------------------

class Emotion(str, Enum):
    """Neutral, non-clinical conversational emotion labels."""
    HAPPY = "HAPPY"
    EXCITED = "EXCITED"
    SAD = "SAD"
    FRUSTRATED = "FRUSTRATED"
    CONFUSED = "CONFUSED"
    ANGRY = "ANGRY"
    DISAPPOINTED = "DISAPPOINTED"
    WORRIED = "WORRIED"
    CURIOUS = "CURIOUS"
    GRATEFUL = "GRATEFUL"
    CALM = "CALM"
    NEUTRAL = "NEUTRAL"


class Situation(str, Enum):
    """Conversational situation / context category."""
    ACHIEVEMENT = "ACHIEVEMENT"
    FAILURE = "FAILURE"
    FRUSTRATION = "FRUSTRATION"
    CONFUSION = "CONFUSION"
    EXCITEMENT = "EXCITEMENT"
    SADNESS = "SADNESS"
    ANGER = "ANGER"
    GRATITUDE = "GRATITUDE"
    CURIOSITY = "CURIOSITY"
    ENCOURAGEMENT_NEEDED = "ENCOURAGEMENT_NEEDED"
    NEUTRAL_CONVERSATION = "NEUTRAL_CONVERSATION"


class ResponseMode(str, Enum):
    """Suggested overall tone/mode the bot's reply generator should use."""
    CONGRATULATORY = "CONGRATULATORY"
    ENCOURAGING = "ENCOURAGING"
    CALMING = "CALMING"
    CLARIFYING = "CLARIFYING"
    EMPATHETIC = "EMPATHETIC"
    SUPPORTIVE = "SUPPORTIVE"
    APPRECIATIVE = "APPRECIATIVE"
    INFORMATIVE = "INFORMATIVE"
    NEUTRAL = "NEUTRAL"


# Bounds shared across the module
_MIN_SCORE = 0.0
_MAX_SCORE = 1.0


def _clamp(value: float, lo: float = _MIN_SCORE, hi: float = _MAX_SCORE) -> float:
    """Clamp a numeric value into [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 2. Result container
# ---------------------------------------------------------------------------

@dataclass
class EmotionResult:
    """
    Structured result of emotion analysis for a single user message.

    Attributes
    ----------
    primary_emotion: Emotion
        The dominant detected USER emotion.
    confidence: float
        Confidence in `primary_emotion`, bounded [0, 1].
    secondary_emotions: List[Emotion]
        Any additional co-occurring USER emotions (e.g. mixed emotions
        like "nervous but excited"). Does not include primary_emotion.
    intensity: float
        How strongly the emotion is expressed, bounded [0, 1].
    situation: Situation
        Best-guess conversational situation/context.
    suggested_robot_emotion: Emotion
        Recommended emotion for the ROBOT's response. This is
        deliberately a *separate* field from primary_emotion/user
        emotion — the robot's emotion is a reaction/complement to the
        user's emotion, not a mirror of it.
    response_mode: ResponseMode
        Suggested high-level tone for the bot's reply generator to use.
    signals: List[str]
        Human-readable list of the lightweight lexical/pattern signals
        that contributed to the result (for debugging/tuning only —
        never exposes internal scoring math or chain-of-thought).
    """
    primary_emotion: Emotion = Emotion.NEUTRAL
    confidence: float = 0.0
    secondary_emotions: List[Emotion] = field(default_factory=list)
    intensity: float = 0.0
    situation: Situation = Situation.NEUTRAL_CONVERSATION
    suggested_robot_emotion: Emotion = Emotion.CALM
    response_mode: ResponseMode = ResponseMode.NEUTRAL
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to a plain JSON-safe dict (enums -> their string value)."""
        d = asdict(self)
        d["primary_emotion"] = self.primary_emotion.value
        d["secondary_emotions"] = [e.value for e in self.secondary_emotions]
        d["situation"] = self.situation.value
        d["suggested_robot_emotion"] = self.suggested_robot_emotion.value
        d["response_mode"] = self.response_mode.value
        return d


# ---------------------------------------------------------------------------
# 3. Lexicon / pattern tables (deterministic rule base)
# ---------------------------------------------------------------------------
# Each entry maps a keyword/phrase (already lowercase) to a weight.
# Weights are intentionally modest so a single weak word never triggers
# a maxed-out confidence/intensity (requirement: don't overreact).

_STRONG = 0.55
_MEDIUM = 0.35
_WEAK = 0.18

_EMOTION_LEXICON: Dict[Emotion, Dict[str, float]] = {
    Emotion.EXCITED: {
        "i finally": _MEDIUM, "finally passed": _STRONG, "i passed": _MEDIUM,
        "yes!": _MEDIUM, "amazing": _MEDIUM, "awesome": _MEDIUM,
        "can't wait": _MEDIUM, "so excited": _STRONG, "excited": _MEDIUM,
        "thrilled": _STRONG, "pumped": _MEDIUM, "yay": _WEAK,
        "woohoo": _STRONG, "can't believe it": _MEDIUM,
    },
    Emotion.HAPPY: {
        "happy": _MEDIUM, "glad": _WEAK, "great": _WEAK, "good news": _MEDIUM,
        "love this": _WEAK, "nice": _WEAK, ":)": _WEAK, "😊": _WEAK,
        "😄": _WEAK, "🎉": _MEDIUM,
    },
    Emotion.SAD: {
        "sad": _MEDIUM, "down": _WEAK, "unhappy": _MEDIUM, "depressed mood": _MEDIUM,
        "crying": _STRONG, "heartbroken": _STRONG, "miserable": _STRONG,
        ":(": _WEAK, "😢": _WEAK, "😭": _MEDIUM,
    },
    Emotion.DISAPPOINTED: {
        "failed": _MEDIUM, "failed again": _STRONG, "i failed": _MEDIUM,
        "disappointed": _STRONG, "let down": _MEDIUM, "didn't pass": _MEDIUM,
        "not good enough": _MEDIUM, "again": _WEAK,
    },
    Emotion.FRUSTRATED: {
        "frustrated": _STRONG, "annoying": _MEDIUM, "annoyed": _MEDIUM,
        "ugh": _WEAK, "this is stupid": _MEDIUM, "so frustrating": _STRONG,
        "fed up": _STRONG, "sick of": _MEDIUM, "keeps failing": _MEDIUM,
        "not working": _WEAK, "doesn't work": _WEAK,
    },
    Emotion.CONFUSED: {
        "confused": _STRONG, "i don't understand": _STRONG, "don't get it": _MEDIUM,
        "what does this mean": _MEDIUM, "unclear": _MEDIUM, "lost": _WEAK,
        "huh?": _WEAK, "makes no sense": _MEDIUM, "not sure what": _WEAK,
    },
    Emotion.ANGRY: {
        "angry": _STRONG, "furious": _STRONG, "mad": _MEDIUM, "pissed": _STRONG,
        "hate this": _MEDIUM, "this is ridiculous": _MEDIUM, "unacceptable": _MEDIUM,
    },
    Emotion.WORRIED: {
        "worried": _STRONG, "nervous": _MEDIUM, "anxious": _MEDIUM,
        "scared": _MEDIUM, "afraid": _MEDIUM, "stressed": _MEDIUM,
        "what if i fail": _STRONG, "i'm worried": _STRONG,
    },
    Emotion.CURIOUS: {
        "curious": _MEDIUM, "how does": _WEAK, "why does": _WEAK,
        "i wonder": _MEDIUM, "what if": _WEAK, "how do i": _WEAK,
        "tell me more": _WEAK, "interesting": _WEAK,
    },
    Emotion.GRATEFUL: {
        "thank you": _MEDIUM, "thanks": _WEAK, "appreciate it": _MEDIUM,
        "grateful": _STRONG, "you helped": _WEAK, "that helped a lot": _MEDIUM,
    },
    Emotion.CALM: {
        "i'm fine": _WEAK, "no worries": _WEAK, "all good": _WEAK,
        "relaxed": _MEDIUM, "calm": _MEDIUM,
    },
}

# Weak, generic single words that must never *alone* push intensity/confidence
# high (requirement 7). These get down-weighted further when found in
# isolation with no other supporting signal.
_GENERIC_WEAK_WORDS = {"bad", "ok", "okay", "fine", "meh", "alright"}

# Situation is largely inferred from which emotion(s) fired, with a couple
# of extra situational-only cues (e.g. explicit achievement/failure phrasing
# independent of a strong emotion word).
_SITUATION_CUES: Dict[Situation, Dict[str, float]] = {
    Situation.ACHIEVEMENT: {
        "i passed": _STRONG, "i finally passed": _STRONG, "i did it": _MEDIUM,
        "i got an a": _STRONG, "i won": _MEDIUM, "i succeeded": _MEDIUM,
    },
    Situation.FAILURE: {
        "i failed": _STRONG, "failed again": _STRONG, "didn't pass": _MEDIUM,
        "i lost": _WEAK, "i got an f": _STRONG,
    },
}

_NEGATIONS = {"not", "n't", "never", "no"}

_INTENSIFIERS = {"very": 1.3, "really": 1.25, "so": 1.25, "extremely": 1.4,
                 "totally": 1.2, "absolutely": 1.3}

_DIMINISHERS = {"a bit": 0.7, "a little": 0.7, "slightly": 0.6, "kind of": 0.75,
                "kinda": 0.75, "somewhat": 0.75}


# ---------------------------------------------------------------------------
# 4. DETECTION — raw signal extraction
# ---------------------------------------------------------------------------

def _normalize(message: str) -> str:
    """Lowercase + collapse whitespace. Never raises on odd input."""
    text = message.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_modifiers(text: str, phrase: str, base_weight: float) -> float:
    """
    Adjust a phrase's contribution based on nearby intensifiers/diminishers
    and simple negation. This is a shallow, deterministic heuristic — not
    full NLP — by design (lightweight, explainable, dependency-free).
    """
    weight = base_weight

    idx = text.find(phrase)
    if idx == -1:
        return weight

    window_start = max(0, idx - 15)
    window = text[window_start:idx]

    for word, mult in _INTENSIFIERS.items():
        if word in window:
            weight *= mult
            break
    for word, mult in _DIMINISHERS.items():
        if word in window:
            weight *= mult
            break

    for neg in _NEGATIONS:
        if neg in window:
            weight *= 0.4  # negated sentiment: sharply reduce, don't flip
            break

    return weight


def _detect_emotion_scores(text: str) -> Tuple[Dict[Emotion, float], List[str]]:
    """
    DETECTION step.

    Scans normalized text against the lexicon and returns a raw score per
    emotion plus a list of human-readable matched signals.
    """
    scores: Dict[Emotion, float] = {}
    signals: List[str] = []

    if not text:
        return scores, signals

    words = text.split()
    is_single_generic_word = len(words) == 1 and words[0] in _GENERIC_WEAK_WORDS

    for emotion, phrase_weights in _EMOTION_LEXICON.items():
        emotion_score = 0.0
        for phrase, base_weight in phrase_weights.items():
            if phrase in text:
                weight = _apply_modifiers(text, phrase, base_weight)
                emotion_score += weight
                signals.append(f"matched:{emotion.value}:'{phrase}'")

        if emotion_score > 0:
            scores[emotion] = _clamp(emotion_score)

    # Requirement 7: a single weak/generic word must not create a highly
    # emotional reading. If the *entire* message is just a generic weak
    # word, force everything down regardless of any accidental lexicon hit.
    if is_single_generic_word:
        scores = {e: min(s, 0.2) for e, s in scores.items()}
        signals.append("dampened:single_generic_weak_word")

    return scores, signals


def _detect_situation(text: str, top_emotions: List[Emotion]) -> Situation:
    """
    DETECTION step for situation/context.

    Uses explicit situational cue phrases first; falls back to mapping
    from the dominant detected emotion(s).
    """
    situation_scores: Dict[Situation, float] = {}
    for situation, phrase_weights in _SITUATION_CUES.items():
        for phrase, weight in phrase_weights.items():
            if phrase in text:
                situation_scores[situation] = situation_scores.get(situation, 0.0) + weight

    if situation_scores:
        return max(situation_scores.items(), key=lambda kv: kv[1])[0]

    if not top_emotions:
        return Situation.NEUTRAL_CONVERSATION

    emotion_to_situation = {
        Emotion.EXCITED: Situation.EXCITEMENT,
        Emotion.HAPPY: Situation.EXCITEMENT,
        Emotion.SAD: Situation.SADNESS,
        Emotion.DISAPPOINTED: Situation.FAILURE,
        Emotion.FRUSTRATED: Situation.FRUSTRATION,
        Emotion.CONFUSED: Situation.CONFUSION,
        Emotion.ANGRY: Situation.ANGER,
        Emotion.WORRIED: Situation.ENCOURAGEMENT_NEEDED,
        Emotion.CURIOUS: Situation.CURIOSITY,
        Emotion.GRATEFUL: Situation.GRATITUDE,
        Emotion.CALM: Situation.NEUTRAL_CONVERSATION,
    }
    return emotion_to_situation.get(top_emotions[0], Situation.NEUTRAL_CONVERSATION)


# ---------------------------------------------------------------------------
# 5. INTERPRETATION — turn raw scores into a coherent picture
# ---------------------------------------------------------------------------

def _rank_emotions(scores: Dict[Emotion, float]) -> List[Tuple[Emotion, float]]:
    """Sort detected emotions by score, descending, deterministic tie-break
    by enum name so results are stable across runs."""
    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1], kv[0].value),
    )


def _interpret(scores: Dict[Emotion, float], signals: List[str]) -> Tuple[Emotion, float, float, List[Emotion]]:
    """
    INTERPRETATION step.

    Turns raw per-emotion scores into:
      - primary_emotion
      - confidence (bounded 0..1)
      - intensity  (bounded 0..1)
      - secondary_emotions (mixed-emotion support)
    """
    if not scores:
        return Emotion.NEUTRAL, 0.0, 0.0, []

    ranked = _rank_emotions(scores)
    primary_emotion, primary_score = ranked[0]

    # Confidence: how dominant the top emotion is vs. total signal mass.
    total_mass = sum(s for _, s in ranked)
    confidence = _clamp(primary_score / total_mass) if total_mass > 0 else 0.0
    # Blend in the raw strength so a single faint hit doesn't get
    # artificially inflated to confidence=1.0 just by being alone.
    confidence = _clamp((confidence * 0.6) + (primary_score * 0.4))

    intensity = _clamp(primary_score)

    # Mixed emotions: any other emotion scoring at least 40% of the
    # primary's score is considered a genuine co-occurring secondary
    # emotion (e.g. "nervous but excited").
    secondary_emotions = [
        e for e, s in ranked[1:]
        if s >= primary_score * 0.4 and e != primary_emotion
    ]

    return primary_emotion, confidence, intensity, secondary_emotions


# ---------------------------------------------------------------------------
# 6. ROBOT RESPONSE RECOMMENDATION
# ---------------------------------------------------------------------------
# Explicitly separate table: user_emotion/situation -> (robot_emotion, mode).
# This is where the "never confuse user emotion with robot emotion"
# requirement is enforced structurally — the robot's emotion is always
# looked up, never copied from the user's.

_RECOMMENDATION_TABLE: Dict[Situation, Tuple[Emotion, ResponseMode]] = {
    Situation.ACHIEVEMENT: (Emotion.EXCITED, ResponseMode.CONGRATULATORY),
    Situation.FAILURE: (Emotion.CALM, ResponseMode.ENCOURAGING),
    Situation.FRUSTRATION: (Emotion.CALM, ResponseMode.CALMING),
    Situation.CONFUSION: (Emotion.CALM, ResponseMode.CLARIFYING),
    Situation.EXCITEMENT: (Emotion.HAPPY, ResponseMode.CONGRATULATORY),
    Situation.SADNESS: (Emotion.CALM, ResponseMode.EMPATHETIC),
    Situation.ANGER: (Emotion.CALM, ResponseMode.CALMING),
    Situation.GRATITUDE: (Emotion.HAPPY, ResponseMode.APPRECIATIVE),
    Situation.CURIOSITY: (Emotion.CURIOUS, ResponseMode.INFORMATIVE),
    Situation.ENCOURAGEMENT_NEEDED: (Emotion.CALM, ResponseMode.SUPPORTIVE),
    Situation.NEUTRAL_CONVERSATION: (Emotion.CALM, ResponseMode.NEUTRAL),
}


def _recommend_robot_response(situation: Situation) -> Tuple[Emotion, ResponseMode]:
    """
    RECOMMENDATION step.

    Looks up the appropriate ROBOT emotion + response mode for a given
    situation. Intentionally decoupled from `primary_emotion` (the user's
    emotion) — the robot reacts to the *situation*, it does not mirror the
    user's raw feeling. E.g. a FAILURE situation (user = DISAPPOINTED)
    recommends a SUPPORTIVE/CALM robot response, not a disappointed one.
    """
    return _RECOMMENDATION_TABLE.get(
        situation, (Emotion.CALM, ResponseMode.NEUTRAL)
    )


# ---------------------------------------------------------------------------
# 7. Public API
# ---------------------------------------------------------------------------

def analyze_emotion(message: str) -> EmotionResult:
    """
    Analyze a single user message and return a structured EmotionResult.

    This function never raises on bad input (None, non-str, empty string,
    whitespace-only) — it safely falls back to a NEUTRAL result instead.

    Parameters
    ----------
    message: str
        The raw user message text.

    Returns
    -------
    EmotionResult
    """
    # --- Validation / safe fallback (requirement 13) ---
    if not isinstance(message, str):
        return EmotionResult(signals=["invalid_input:non_string"])

    text = _normalize(message)
    if not text:
        return EmotionResult(signals=["empty_input"])

    # --- 1. DETECTION ---
    emotion_scores, signals = _detect_emotion_scores(text)

    # --- 2. INTERPRETATION ---
    primary_emotion, confidence, intensity, secondary_emotions = _interpret(
        emotion_scores, signals
    )

    ranked_emotions = [e for e, _ in _rank_emotions(emotion_scores)]
    situation = _detect_situation(text, ranked_emotions)

    # --- 3. ROBOT RESPONSE RECOMMENDATION ---
    suggested_robot_emotion, response_mode = _recommend_robot_response(situation)

    if not emotion_scores:
        signals.append("no_signals_detected:defaulted_neutral")

    return EmotionResult(
        primary_emotion=primary_emotion,
        confidence=round(_clamp(confidence), 3),
        secondary_emotions=secondary_emotions,
        intensity=round(_clamp(intensity), 3),
        situation=situation,
        suggested_robot_emotion=suggested_robot_emotion,
        response_mode=response_mode,
        signals=signals,
    )


def analyze_emotion_dict(message: str) -> dict:
    """
    Convenience wrapper around `analyze_emotion` that returns a plain,
    JSON-serializable dict instead of an EmotionResult dataclass.
    Useful for API responses / logging without leaking internal types.
    """
    return analyze_emotion(message).to_dict()


# ---------------------------------------------------------------------------
# 8. Manual smoke test (only runs when executed directly — no test
#    framework dependency is imposed on the module itself)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "I finally passed my exam!",
        "I failed my exam again.",
        "I'm nervous but excited about the exam.",
        "",
        None,
        "bad",
        "This is so frustrating, nothing works!",
        "I don't understand this at all, can you explain?",
        "Thank you so much, that really helped.",
        "How does photosynthesis work?",
    ]
    for s in samples:
        result = analyze_emotion(s)  # type: ignore[arg-type]
        print(repr(s), "->", result.to_dict())
