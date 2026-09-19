"""
intent_router.py

Lightweight, deterministic intent router for StudyBot.

This module ONLY classifies the likely intent/category of an incoming
user message. It does not touch WebSocket, database, AI service,
game engine, Fast Math, config, or chat-flow code in any way.

Designed to be swapped out later for an ML/LLM-based classifier by
implementing the same `route_intent(message) -> IntentResult` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Pattern, Tuple


class Intent(str, Enum):
    EDUCATION = "EDUCATION"
    GENERAL = "GENERAL"
    NEWS = "NEWS"
    INFORMATION = "INFORMATION"
    TECHNOLOGY = "TECHNOLOGY"
    SHOPPING = "SHOPPING"
    BUSINESS = "BUSINESS"
    ENTERTAINMENT = "ENTERTAINMENT"
    PERSONAL = "PERSONAL"
    GAME = "GAME"
    UNKNOWN = "UNKNOWN"


@dataclass
class IntentResult:
    intent: str
    confidence: float
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
        }


@dataclass
class _Rule:
    pattern: Pattern
    weight: float
    label: str


def _compile(phrases: List[Tuple[str, float]]) -> List[_Rule]:
    rules = []
    for phrase, weight in phrases:
        pattern = re.compile(r"\b" + phrase + r"\b", re.IGNORECASE)
        rules.append(_Rule(pattern=pattern, weight=weight, label=phrase))
    return rules


# ---------------------------------------------------------------------------
# Signal definitions per intent.
# Each tuple is (regex-ready phrase, weight).
# Weights are tuned so multi-signal matches build strong confidence while
# single generic words stay weak.
# ---------------------------------------------------------------------------

_GAME_RULES = _compile([
    (r"let'?s play", 3.0),
    (r"play a game", 3.0),
    (r"play game", 2.5),
    (r"start (a |the )?game", 2.5),
    (r"fast math", 2.5),
    (r"quiz me", 2.0),
    (r"trivia", 2.0),
    (r"guess (the|a|my)", 1.5),
    (r"riddle", 1.5),
    (r"play", 1.0),
    (r"game", 1.0),
    (r"score", 0.5),
    (r"level up", 1.0),
])

_EDUCATION_RULES = _compile([
    (r"teach me", 2.5),
    (r"explain", 2.0),
    (r"homework", 2.5),
    (r"study", 2.0),
    (r"exam", 2.0),
    (r"algebra", 2.0),
    (r"geometry", 2.0),
    (r"calculus", 2.0),
    (r"photosynthesis", 2.0),
    (r"chemistry", 1.8),
    (r"physics", 1.8),
    (r"biology", 1.8),
    (r"history lesson", 2.0),
    (r"how does .* work", 1.5),
    (r"solve (this|for)", 1.8),
    (r"equation", 1.8),
    (r"grammar", 1.5),
    (r"essay", 1.5),
    (r"tutor", 2.0),
    (r"lesson", 1.5),
    (r"learn", 1.2),
])

_NEWS_RULES = _compile([
    (r"latest news", 3.0),
    (r"breaking news", 3.0),
    (r"what happened today", 2.5),
    (r"what'?s happening", 2.0),
    (r"current events", 2.5),
    (r"headline", 2.0),
    (r"news", 1.8),
    (r"today'?s events", 2.0),
    (r"update on", 1.2),
])

_TECHNOLOGY_RULES = _compile([
    (r"laptop", 1.8),
    (r"smartphone", 1.8),
    (r"software", 1.6),
    (r"hardware", 1.6),
    (r"programming", 1.8),
    (r"python", 1.5),
    (r"javascript", 1.5),
    (r"ai model", 1.8),
    (r"artificial intelligence", 1.8),
    (r"computer", 1.4),
    (r"app(lication)?", 1.2),
    (r"code", 1.2),
    (r"bug", 1.2),
    (r"tech", 1.0),
    (r"gadget", 1.5),
])

_SHOPPING_RULES = _compile([
    (r"which laptop should i buy", 3.0),
    (r"should i buy", 2.5),
    (r"buy", 1.8),
    (r"purchase", 1.8),
    (r"price of", 1.8),
    (r"cheapest", 1.8),
    (r"best deal", 2.0),
    (r"discount", 1.8),
    (r"shop(ping)?", 1.6),
    (r"cart", 1.5),
    (r"order", 1.2),
    (r"compare .* (price|prices|model|models)", 2.0),
])

_BUSINESS_RULES = _compile([
    (r"my business", 2.5),
    (r"start(ing)? a business", 2.5),
    (r"marketing plan", 2.2),
    (r"business plan", 2.5),
    (r"revenue", 2.0),
    (r"startup", 2.0),
    (r"investor", 2.0),
    (r"invoice", 1.8),
    (r"client", 1.5),
    (r"company", 1.4),
    (r"sales", 1.4),
    (r"profit", 1.6),
    (r"budget", 1.4),
])

_ENTERTAINMENT_RULES = _compile([
    (r"tell me a joke", 3.0),
    (r"joke", 2.0),
    (r"movie", 1.8),
    (r"tv show", 1.8),
    (r"song", 1.6),
    (r"music", 1.6),
    (r"funny", 1.5),
    (r"story", 1.4),
    (r"celebrity", 1.6),
    (r"meme", 1.6),
    (r"entertain me", 2.5),
])

_PERSONAL_RULES = _compile([
    (r"how i feel", 2.0),
    (r"i feel", 1.8),
    (r"i'?m sad", 2.2),
    (r"i'?m tired", 1.8),
    (r"my day", 1.8),
    (r"advice", 1.6),
    (r"relationship", 1.8),
    (r"my friend", 1.6),
    (r"my family", 1.6),
    (r"how are you", 1.5),
    (r"talk to (me|you)", 1.4),
])

_INFORMATION_RULES = _compile([
    (r"who is", 1.6),
    (r"who was", 1.6),
    (r"define", 1.8),
    (r"definition of", 2.0),
    (r"meaning of", 1.8),
    (r"capital of", 2.0),
    (r"population of", 2.0),
    (r"how far is", 1.8),
    (r"how many", 1.4),
    (r"when did", 1.6),
    (r"where is", 1.6),
    (r"fact about", 1.8),
])

_GENERAL_RULES = _compile([
    (r"hi", 1.2),
    (r"hello", 1.2),
    (r"hey", 1.0),
    (r"thanks", 1.2),
    (r"thank you", 1.4),
    (r"ok(ay)?", 0.6),
    (r"good morning", 1.4),
    (r"good night", 1.4),
    (r"how'?s it going", 1.4),
])


_INTENT_RULES: Dict[Intent, List[_Rule]] = {
    Intent.GAME: _GAME_RULES,
    Intent.EDUCATION: _EDUCATION_RULES,
    Intent.NEWS: _NEWS_RULES,
    Intent.TECHNOLOGY: _TECHNOLOGY_RULES,
    Intent.SHOPPING: _SHOPPING_RULES,
    Intent.BUSINESS: _BUSINESS_RULES,
    Intent.ENTERTAINMENT: _ENTERTAINMENT_RULES,
    Intent.PERSONAL: _PERSONAL_RULES,
    Intent.INFORMATION: _INFORMATION_RULES,
    Intent.GENERAL: _GENERAL_RULES,
}


# Precedence order used only for tie-breaking when scores are close.
# GAME is deliberately first so an explicit "let's play" wins over
# generic overlaps (e.g. "play" also loosely resembles entertainment).
_PRECEDENCE: List[Intent] = [
    Intent.GAME,
    Intent.EDUCATION,
    Intent.NEWS,
    Intent.SHOPPING,
    Intent.BUSINESS,
    Intent.TECHNOLOGY,
    Intent.ENTERTAINMENT,
    Intent.PERSONAL,
    Intent.INFORMATION,
    Intent.GENERAL,
]

_MIN_CONFIDENCE = 0.35
_TIE_MARGIN = 0.15


def _normalize(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


def _score_intent(text: str, rules: List[_Rule]) -> Tuple[float, List[str]]:
    total = 0.0
    signals: List[str] = []
    for rule in rules:
        if rule.pattern.search(text):
            total += rule.weight
            signals.append(rule.label)
    return total, signals


def _confidence_from_score(score: float) -> float:
    if score <= 0:
        return 0.0
    # Squash raw additive score into a 0-1 confidence range.
    confidence = 1.0 - (1.0 / (1.0 + score))
    return min(confidence, 0.99)


def route_intent(message: str) -> IntentResult:
    """
    Determine the most likely intent for a given user message.

    This is a pure function with no side effects: no I/O, no network,
    no database access, no WebSocket usage. Safe to call from any
    part of the server that needs a quick routing hint.
    """
    if not message or not message.strip():
        return IntentResult(intent=Intent.UNKNOWN.value, confidence=0.0, signals=[])

    text = _normalize(message)

    scored: Dict[Intent, Tuple[float, List[str]]] = {}
    for intent, rules in _INTENT_RULES.items():
        score, signals = _score_intent(text, rules)
        if score > 0:
            scored[intent] = (score, signals)

    if not scored:
        return IntentResult(intent=Intent.UNKNOWN.value, confidence=0.0, signals=[])

    ranked = sorted(
        scored.items(),
        key=lambda item: (item[1][0], -_PRECEDENCE.index(item[0])),
        reverse=True,
    )

    top_intent, (top_score, top_signals) = ranked[0]

    if len(ranked) > 1:
        second_intent, (second_score, _) = ranked[1]
        if (top_score - second_score) < _TIE_MARGIN:
            top_precedence = _PRECEDENCE.index(top_intent)
            second_precedence = _PRECEDENCE.index(second_intent)
            if second_precedence < top_precedence:
                top_intent, top_score, top_signals = (
                    second_intent,
                    second_score,
                    scored[second_intent][1],
                )

    confidence = _confidence_from_score(top_score)

    if confidence < _MIN_CONFIDENCE:
        return IntentResult(
            intent=Intent.UNKNOWN.value,
            confidence=confidence,
            signals=top_signals,
        )

    return IntentResult(
        intent=top_intent.value,
        confidence=confidence,
        signals=top_signals,
    )


def route_intent_dict(message: str) -> Dict:
    """Convenience wrapper returning a plain dict for easy JSON serialization."""
    return route_intent(message).to_dict()
