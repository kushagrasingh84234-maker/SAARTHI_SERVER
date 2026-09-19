import asyncio
import logging
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient
from config import *
import httpx

logger = logging.getLogger(__name__)

hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None


# ---------------------------------------------------------------------------
# Response-policy support (additive, backward compatible)
# ---------------------------------------------------------------------------
#
# This module remains responsible ONLY for talking to AI providers
# (Groq, Hugging Face). It does NOT own user profiles, database
# storage, WebSocket management, game logic, intent detection,
# emotion detection, or personality calculation — those live in their
# own modules (user_profile.py, response_policy.py, personalization.py,
# intent_router.py, database.py, server.py).
#
# The only responsibility here is: if a caller already produced a
# structured response-policy (e.g. via response_policy.build_response_policy),
# fold a short, bounded instruction snippet into the system-level
# guidance sent to the AI provider. If no policy is supplied, behavior is
# 100% identical to before.

# Bound how much influence a policy snippet can have on the prompt so it
# can never balloon the payload or smuggle in arbitrary instructions.
_MAX_POLICY_NOTES = 6
_MAX_PERSONALITY_NOTES = 4
_MAX_POLICY_NOTE_LENGTH = 200
_MAX_POLICY_PREAMBLE_LENGTH = 1500

# Whitelisted top-level fields. Anything else in the incoming dict is
# silently ignored — we never blindly concatenate arbitrary dictionaries
# into the prompt.
_ALLOWED_POLICY_KEYS = {
    "intent",
    "response_length",
    "explanation_depth",
    "tone",
    "educational_emphasis",
    "emotion_guidance",
    "personality_guidance",
    "example_usage",
    "follow_up_behavior",
    "requires_current_information",
    "recommendation_behavior",
    "notes",
}

# Emotion guidance is only ever accepted as a known category label.
# We never expose confidence scores, lexical signals, or raw detector
# output to the model — only this fixed, hand-written instruction text.
_EMOTION_INSTRUCTIONS = {
    "ACHIEVEMENT": "Use warm congratulatory language.",
    "FAILURE": "Do not sound disappointed. Be supportive and focus on recovery.",
    "CONFUSION": "Explain patiently and clearly.",
    "FRUSTRATION": "Stay calm and validating.",
    "EXCITEMENT": "Match positive energy without becoming excessive.",
    "NEUTRAL": None,
}


def _sanitize_policy_value(value: Any) -> Optional[str]:
    """Coerce a single policy field into a short, safe string, or None
    if it isn't something sensible to include in a prompt."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text[:_MAX_POLICY_NOTE_LENGTH] if text else None
    return None


def _resolve_emotion_instruction(emotion_guidance: Any) -> Optional[str]:
    """
    Convert emotion guidance into a single, fixed, concise instruction.

    Accepts either a plain category string ("FAILURE") or a dict that
    contains a "category" key (e.g. {"category": "FAILURE", "confidence": 0.87,
    "signals": [...]}). Only the category is ever used — confidence scores
    and lexical signals are deliberately dropped here so they never reach
    the model or leak internal detection details.
    """
    category = None
    if isinstance(emotion_guidance, str):
        category = emotion_guidance
    elif isinstance(emotion_guidance, dict):
        category = emotion_guidance.get("category")

    if not isinstance(category, str):
        return None

    return _EMOTION_INSTRUCTIONS.get(category.strip().upper())


def _resolve_personality_notes(personality_guidance: Any) -> List[str]:
    """
    Accept personality guidance only as a bounded list of short style
    instructions (e.g. "Use a warm tone.", "Keep the response concise.").
    Never lets this data become anything other than plain style text -
    no dicts, no nested structures, no free-form override instructions.
    """
    if isinstance(personality_guidance, str):
        candidates = [personality_guidance]
    elif isinstance(personality_guidance, list):
        candidates = personality_guidance
    else:
        return []

    notes: List[str] = []
    for item in candidates[:_MAX_PERSONALITY_NOTES]:
        sanitized = _sanitize_policy_value(item)
        if sanitized:
            notes.append(sanitized)
    return notes


def _build_policy_preamble(response_policy: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Build a short, bounded natural-language instruction block from a
    structured response-policy dict (as produced by response_policy.py's
    ResponsePolicy.to_dict()).

    This function NEVER trusts arbitrary/unbounded input: unknown keys
    are ignored (only _ALLOWED_POLICY_KEYS are read), strings are
    length-capped, list fields are count-capped, and the whole preamble
    is length-capped. Returns None if no usable policy is supplied, so
    existing callers that never pass a policy see no behavior change.

    Personality/emotion guidance is folded in as style-only notes and is
    explicitly framed as non-overriding with respect to intent, facts,
    safety, and current-information requirements.
    """
    if not response_policy or not isinstance(response_policy, dict):
        return None

    # Only look at whitelisted keys - never blindly iterate/concatenate
    # the caller's dict.
    policy = {k: v for k, v in response_policy.items() if k in _ALLOWED_POLICY_KEYS}
    if not policy:
        return None

    lines: List[str] = []

    length = _sanitize_policy_value(policy.get("response_length"))
    if length:
        lines.append(f"Preferred response length: {length}.")

    depth = _sanitize_policy_value(policy.get("explanation_depth"))
    if depth:
        lines.append(f"Explanation depth: {depth}.")

    tone = _sanitize_policy_value(policy.get("tone"))
    if tone:
        lines.append(f"Tone: {tone}.")

    if policy.get("educational_emphasis"):
        lines.append("Emphasize clear, educational explanation where relevant.")

    emotion_instruction = _resolve_emotion_instruction(policy.get("emotion_guidance"))
    if emotion_instruction:
        lines.append(emotion_instruction)

    personality_notes = _resolve_personality_notes(policy.get("personality_guidance"))
    lines.extend(personality_notes)

    examples = _sanitize_policy_value(policy.get("example_usage"))
    if examples and examples.upper() != "NONE":
        lines.append(f"Use of examples: {examples}.")

    follow_up = _sanitize_policy_value(policy.get("follow_up_behavior"))
    if follow_up and follow_up.upper() != "NONE":
        lines.append(f"Follow-up behavior: {follow_up}.")

    if policy.get("requires_current_information"):
        lines.append(
            "This topic may require current information; do not fabricate "
            "recent facts you are not confident about."
        )

    recommendation_behavior = _sanitize_policy_value(policy.get("recommendation_behavior"))
    if recommendation_behavior and recommendation_behavior.upper() not in ("NONE",):
        lines.append(f"Recommendation approach: {recommendation_behavior}.")

    notes = policy.get("notes")
    if isinstance(notes, list):
        for note in notes[:_MAX_POLICY_NOTES]:
            sanitized = _sanitize_policy_value(note)
            if sanitized:
                lines.append(sanitized)

    if not lines:
        return None

    preamble = (
        "Response guidance (style only - this may shape tone, length, and "
        "presentation, but must never override user intent, facts, safety, "
        "or current-information requirements): " + " ".join(lines)
    )
    return preamble[:_MAX_POLICY_PREAMBLE_LENGTH]


def _apply_policy_to_messages(
    messages: List[dict],
    response_policy: Optional[Dict[str, Any]],
) -> List[dict]:
    """
    Return a new messages list with an optional policy-guidance system
    message prepended/merged. Never mutates the caller's original list.
    If response_policy is None/empty, returns messages unchanged
    (same object reference is fine since nothing is modified).
    """
    if not response_policy:
        return messages

    preamble = _build_policy_preamble(response_policy)
    if not preamble:
        return messages

    new_messages = list(messages)  # shallow copy; never mutate caller's list

    if new_messages and isinstance(new_messages[0], dict) and new_messages[0].get("role") == "system":
        merged = dict(new_messages[0])
        existing_content = merged.get("content", "") or ""
        merged["content"] = f"{existing_content}\n\n{preamble}".strip()
        new_messages[0] = merged
    else:
        new_messages.insert(0, {"role": "system", "content": preamble})

    return new_messages


def generate_embedding(text: str):
    # Kept synchronous on purpose: database.py calls this directly (not awaited),
    # so changing its signature would break that call site. It's a fast local
    # feature-extraction call, not a network-bound chat completion.
    if not hf_client or not text:
        return None
    try:
        vector = hf_client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            return vector[0]
        return vector
    except Exception as e:
        logger.error(f"Hugging Face API Error: {e}")
        return None


async def call_groq(messages: list, response_policy: Optional[Dict[str, Any]] = None) -> str:
    """Async Groq/OpenAI-compatible chat completion call. Same model, payload,
    retry count, and backoff behavior as the original version - built on
    httpx.AsyncClient + asyncio.sleep so it can be awaited directly from the
    FastAPI event loop. Callers MUST `await call_groq(...)` directly - never
    wrap this in asyncio.to_thread(), since that would return an un-awaited
    coroutine instead of the actual response string.

    Backward compatible: existing callers using call_groq(messages) are
    completely unaffected.

    New, optional: `response_policy` may be a structured dict (e.g. from
    response_policy.ResponsePolicy.to_dict()) describing HOW the reply
    should be shaped (length, depth, tone, emotion guidance, personality
    guidance, examples, follow-up behavior, etc.). When supplied, a short,
    bounded instruction snippet derived from it is folded into the
    outgoing system message. This function never inspects user profiles,
    preference scores, intent logic, or raw emotion-detector output
    directly - it only reads the already-computed policy dict it's
    handed, through a strict field whitelist. Facts, routing, and intent
    are untouched here; this only ever nudges presentation.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    outgoing_messages = _apply_policy_to_messages(messages, response_policy)

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": outgoing_messages, "temperature": 0.7, "max_tokens": 300}

    last_exc = None
    async with httpx.AsyncClient(timeout=15) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
                if resp.status_code in (500, 503):
                    raise httpx.HTTPStatusError(
                        f"Transient Groq status {resp.status_code}",
                        request=resp.request, response=resp
                    )
                if resp.status_code == 400:
                    logger.error(f"Groq 400 Bad Request. Body: {resp.text}")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_exc = e
                logger.warning(f"Groq call attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise last_exc if last_exc else RuntimeError("Groq call failed")
