import json
import re
from config import MAX_DISPLAY_CHARS, DEFAULT_TAG, VALID_TAGS

GREEK_MAP = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'π': 'pi', 'ρ': 'rho', 'ς': 'sigma', 'σ': 'sigma', 'τ': 'tau',
    'Ω': 'Omega', 'Δ': 'Delta', # (बाकी सारे ग्रीक सिंबल जो तुम्हारे कोड में थे)
}

MATH_SYMBOL_MAP = {
    '×': '*', '÷': '/', '·': '*', '⋅': '*', '∗': '*',
    '±': '+/-', '∓': '-/+', '≈': '~', '≅': '~', '≃': '~',
    '≠': '!=', '≤': '<=', '≥': '>=', '≡': '==',
    '∞': 'infinity', '√': 'sqrt',
    '∑': 'sum', '∏': 'product', '∫': 'integral', '∂': 'd', '∇': 'del',
    '°': 'deg', '′': "'", '″': '"',
    '→': '->', '←': '<-', '↔': '<->', '⇒': '=>', '⇐': '<=',
    '∝': 'proportional to', '∴': 'therefore', '∵': 'because',
}

FRACTION_MAP = {'½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4'}

SUPERSCRIPT_MAP = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁺': '+', '⁻': '-'}
SUBSCRIPT_MAP = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₊': '+', '₋': '-'}

_SUPERSCRIPT_RUN = re.compile('[' + ''.join(SUPERSCRIPT_MAP.keys()) + ']+')
_SUBSCRIPT_RUN = re.compile('[' + ''.join(SUBSCRIPT_MAP.keys()) + ']+')

_LATEX_FRAC = re.compile(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}')
_LATEX_SQRT = re.compile(r'\\sqrt\s*\{([^{}]*)\}')
_LATEX_WRAP = re.compile(r'\\(?:left|right|text|mathrm|mathbf|boldsymbol|displaystyle)\b')
_LATEX_DELIMS = re.compile(r'\${1,2}|\\\(|\\\)|\\\[|\\\]')
_LATEX_CMD_MAP = {r'\times': '*', r'\cdot': '*', r'\div': '/', r'\pm': '+/-', r'\infty': 'infinity'}
_LATEX_GENERIC_CMD = re.compile(r'\\([A-Za-z]+)')

_MD_HEADER = re.compile(r'^[ \t]{0,3}#{1,6}[ \t]+', re.MULTILINE)
_MD_BOLD_STAR = re.compile(r'\*\*(.+?)\*\*')
_MD_BOLD_UNDER = re.compile(r'__(.+?)__')
_NEWLINE_RUN = re.compile(r'\s*\n+\s*')
_BULLET_PREFIX = re.compile(r'^[ \t]*[-*•][ \t]+', re.MULTILINE)

# Any raw C0/DEL control character (other than the newlines already normalized
# to spaces above) is stripped before the reply is handed back to server.py -
# these are still valid ASCII bytes and would survive the ascii-encode step
# below, but have no business inside a single-line WebSocket JSON payload or
# on the ILI9341 display.
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x1f\x7f]')

def _convert_scripts(text: str) -> str:
    text = _SUPERSCRIPT_RUN.sub(lambda m: '^' + ''.join(SUPERSCRIPT_MAP.get(c, c) for c in m.group(0)), text)
    text = _SUBSCRIPT_RUN.sub(lambda m: '_' + ''.join(SUBSCRIPT_MAP.get(c, c) for c in m.group(0)), text)
    return text

def _latex_to_plain(text: str) -> str:
    if '\\' not in text and '$' not in text:
        return text
    text = _LATEX_FRAC.sub(r'(\1/\2)', text)
    text = _LATEX_SQRT.sub(r'sqrt(\1)', text)
    for cmd, rep in _LATEX_CMD_MAP.items():
        text = text.replace(cmd, rep)
    text = _LATEX_WRAP.sub('', text)
    text = _LATEX_DELIMS.sub('', text)
    text = _LATEX_GENERIC_CMD.sub(r'\1', text)
    text = text.replace('{', '').replace('}', '')
    return text

def _replace_with_spacing(text: str, symbol: str, replacement: str) -> str:
    out = []
    i, n = 0, len(text)
    while True:
        j = text.find(symbol, i)
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        prev_char = text[j - 1] if j > 0 else ''
        next_char = text[j + len(symbol)] if j + len(symbol) < n else ''
        left = ' ' if prev_char.isalnum() else ''
        right = ' ' if next_char.isalnum() else ''
        out.append(f"{left}{replacement}{right}")
        i = j + len(symbol)
    return ''.join(out)

def format_symbols(text: str) -> str:
    if not text:
        return text or ""
    text = _latex_to_plain(text)
    text = _convert_scripts(text)
    combined_map = {}
    combined_map.update(MATH_SYMBOL_MAP)
    combined_map.update(GREEK_MAP)
    combined_map.update(FRACTION_MAP)
    for symbol, replacement in combined_map.items():
        if symbol in text:
            text = _replace_with_spacing(text, symbol, replacement)
    text = _BULLET_PREFIX.sub('', text)
    text = _NEWLINE_RUN.sub(' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def strip_markdown(text: str) -> str:
    if not text:
        return text or ""
    text = text.replace('`', '')
    text = _MD_HEADER.sub('', text)
    text = _MD_BOLD_STAR.sub(r'\1', text)
    text = _MD_BOLD_UNDER.sub(r'\1', text)
    return text

def _truncate_preserving_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    budget = max(limit - 3, 0)
    truncated = text[:budget]
    last_space = truncated.rfind(' ')
    if last_space > budget * 0.5:
        truncated = truncated[:last_space]
    return truncated.rstrip(' ,.;:') + "..."

# ===========================================================================
# NEW: structured [MODE: ROBOT_CONTROL] / [MODE: CONVERSATIONAL] output
# ===========================================================================
#
# Our fine-tuned saarthi_v2_perfect model can now emit a structured,
# 3-part reply instead of a single "TAG|text" line:
#
#   [INTENT_ANALYSIS]
#   ...
#   [REASONING_STEPS]
#   ...
#   [ACTION]
#   {"schema": "motor_control", "payload": {...}}
#
# ...or, in [MODE: CONVERSATIONAL], the same [INTENT_ANALYSIS]/
# [REASONING_STEPS] preamble followed directly by a plain user-facing
# paragraph (no [ACTION] block), separated from the reasoning by a blank
# line. parse_saarthi_structured_output() below understands both shapes,
# as well as a plain conversational reply or a legacy "TAG|text" reply
# with no structured markers at all.

_SECTION_MARKERS = ("[INTENT_ANALYSIS]", "[REASONING_STEPS]", "[ACTION]")

# tft_display action payload -> screen emotion tag. NOTE: some of these
# (HAPPY, CURIOUS, CALM) are NOT in VALID_TAGS/DEFAULT_TAG's legacy set -
# they're new tags for the saarthi_v2_perfect robot's own TFT firmware.
# Verify your ESP32/TFT firmware understands these before relying on them;
# if it only understands the legacy VALID_TAGS set, narrow this map down
# to tags that are also in VALID_TAGS.
_TFT_DISPLAY_TO_EMOTION = {
    "show_happy_eyes": "HAPPY",
    "show_curious_eyes": "CURIOUS",
    "show_focused_eyes": "THINKING",
    "show_battery_critical_eyes": "SAD",
    "show_alert_eyes": "SHOCKED",
    "show_calm_eyes": "CALM",
}
_KNOWN_EMOTION_TAGS = VALID_TAGS | set(_TFT_DISPLAY_TO_EMOTION.values())

_JSON_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```\s*$', re.IGNORECASE)


def _extract_sections(raw_reply: str) -> dict:
    """Split raw_reply on whichever of _SECTION_MARKERS are present, in
    the order they actually appear, and return {marker: content} for each
    one found. A marker absent from raw_reply is simply absent from the
    returned dict."""
    positions = [(raw_reply.find(marker), marker) for marker in _SECTION_MARKERS if marker in raw_reply]
    positions.sort()
    sections = {}
    for i, (start, marker) in enumerate(positions):
        content_start = start + len(marker)
        content_end = positions[i + 1][0] if i + 1 < len(positions) else len(raw_reply)
        sections[marker] = raw_reply[content_start:content_end].strip()
    return sections


def _parse_action_json(action_raw: str):
    """Best-effort parse of the JSON object following [ACTION]. Tolerates
    an optional ```json ... ``` markdown fence and trailing text after the
    JSON object by using json.JSONDecoder().raw_decode instead of a plain
    json.loads. Returns None (never raises) if no JSON object can be
    found."""
    if not action_raw:
        return None
    text = _JSON_FENCE_RE.sub('', action_raw.strip()).strip()
    start = text.find('{')
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except (ValueError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _friendly_status_from_action(intent_analysis: str, action: dict) -> str:
    """Derive a short, friendly status string for the TFT screen when a
    [MODE: ROBOT_CONTROL] reply has no separate spoken text of its own -
    just [INTENT_ANALYSIS]/[REASONING_STEPS]/[ACTION]. Prefers a short
    excerpt of intent_analysis; falls back to describing the action
    payload; never raises and never returns an empty string."""
    if intent_analysis:
        first_line = intent_analysis.strip().splitlines()[0].strip()
        first_sentence = first_line.split('.')[0].strip()
        if first_sentence:
            return first_sentence

    payload = action.get("payload") if isinstance(action, dict) else None
    if isinstance(payload, dict):
        parts = []
        tft_display = payload.get("tft_display")
        if isinstance(tft_display, str) and tft_display.strip():
            parts.append(tft_display.replace("show_", "").replace("_", " ").strip())
        speaker = payload.get("speaker")
        if isinstance(speaker, str) and speaker.strip():
            parts.append(speaker.replace("_", " ").strip())
        if parts:
            return "On it - " + ", ".join(parts) + "."

    return "Okay, on it!"


def _resolve_emotion_tag(raw_reply: str, action) -> str:
    """Resolve the screen emotion tag: action["payload"]["tft_display"]
    (mapped through _TFT_DISPLAY_TO_EMOTION) takes priority; otherwise a
    leading "TAG|" prefix on raw_reply is honored if it names a known tag;
    otherwise DEFAULT_TAG."""
    if isinstance(action, dict):
        payload = action.get("payload")
        if isinstance(payload, dict):
            tft_display = payload.get("tft_display")
            if isinstance(tft_display, str):
                mapped = _TFT_DISPLAY_TO_EMOTION.get(tft_display.strip())
                if mapped:
                    return mapped

    stripped = raw_reply.lstrip()
    if "|" in stripped:
        potential_tag, _, _ = stripped.partition("|")
        potential_tag = potential_tag.strip().upper()
        if potential_tag in _KNOWN_EMOTION_TAGS:
            return potential_tag

    return DEFAULT_TAG


def parse_saarthi_structured_output(raw_reply: str) -> dict:
    """Parse a saarthi_v2_perfect reply that may contain [INTENT_ANALYSIS],
    [REASONING_STEPS], and/or [ACTION] sections, a plain conversational
    reply, or a legacy "TAG|text" reply.

    Returns a dict with:
        intent_analysis: str content under [INTENT_ANALYSIS], or "".
        reasoning_steps:  str content under [REASONING_STEPS], or "".
        action:           parsed dict from the [ACTION] JSON, or None.
        spoken_text:      the user-facing reply text, with no internal
                           reasoning headers or raw JSON leaked into it.
        display_text:     spoken_text run through the same
                           strip_markdown/format_symbols/ASCII-safe/
                           _truncate_preserving_words pipeline parse_ai_reply
                           already uses, capped at MAX_DISPLAY_CHARS.
        emotion:          the resolved screen emotion tag (see
                           _resolve_emotion_tag).
    """
    raw_reply = raw_reply or ""
    sections = _extract_sections(raw_reply)

    intent_analysis = sections.get("[INTENT_ANALYSIS]", "")
    reasoning_steps = sections.get("[REASONING_STEPS]", "")
    action_raw = sections.get("[ACTION]", "")
    action = _parse_action_json(action_raw) if action_raw else None

    spoken_text = ""

    if sections:
        if action_raw:
            # [MODE: ROBOT_CONTROL]: [REASONING_STEPS]'s content is pure
            # reasoning - the JSON action is already isolated as `action`.
            pass
        elif reasoning_steps:
            # [MODE: CONVERSATIONAL]: the model writes straight from
            # [REASONING_STEPS] into its final paragraph, separated by a
            # blank line. Split on the FIRST blank line so the bullet/
            # numbered reasoning never leaks into what the screen shows.
            reasoning_part, sep, spoken_part = reasoning_steps.partition("\n\n")
            reasoning_steps = reasoning_part.strip()
            spoken_text = spoken_part.strip() if sep else ""
        elif intent_analysis:
            # Only [INTENT_ANALYSIS] present: apply the same split there.
            intent_part, sep, spoken_part = intent_analysis.partition("\n\n")
            intent_analysis = intent_part.strip()
            spoken_text = spoken_part.strip() if sep else ""

        if not spoken_text and action is not None:
            spoken_text = _friendly_status_from_action(intent_analysis, action)
    else:
        # No structured markers at all: plain conversational reply, or a
        # legacy "TAG|text" reply - strip a legacy tag prefix if present.
        text = raw_reply.strip()
        if "|" in text:
            potential_tag, _, remainder = text.partition("|")
            if potential_tag.strip().upper() in VALID_TAGS:
                text = remainder.strip()
        spoken_text = text

    if not spoken_text:
        spoken_text = intent_analysis or reasoning_steps

    emotion = _resolve_emotion_tag(raw_reply, action)

    display_text = strip_markdown(spoken_text)
    display_text = format_symbols(display_text)
    display_text = display_text.replace('–', '-').replace('—', '-').replace('−', '-')
    display_text = display_text.encode('ascii', 'ignore').decode('ascii')
    display_text = _CONTROL_CHAR_RE.sub('', display_text)
    display_text = _truncate_preserving_words(display_text, MAX_DISPLAY_CHARS)

    return {
        "intent_analysis": intent_analysis,
        "reasoning_steps": reasoning_steps,
        "action": action,
        "spoken_text": spoken_text,
        "display_text": display_text,
        "emotion": emotion,
    }


def parse_ai_reply(raw_reply: str) -> str:
    raw_reply = (raw_reply or "").strip()
    if not raw_reply:
        return "SAD|I didn't get a response, please try again."

    # NEW: structured saarthi_v2_perfect output. If none of
    # [INTENT_ANALYSIS]/[REASONING_STEPS]/[ACTION] are present, fall through
    # unchanged to the exact legacy "TAG|text" parsing below.
    if any(marker in raw_reply for marker in _SECTION_MARKERS):
        structured = parse_saarthi_structured_output(raw_reply)
        emotion = structured["emotion"]
        text = structured["display_text"]
        if not text or not text.strip(' |'):
            text = "I couldn't format a clean answer for that one."
        # Re-truncate if needed so "{emotion}|{text}" (not text alone) still
        # fits the same MAX_DISPLAY_CHARS budget the legacy path enforces.
        text_budget = max(MAX_DISPLAY_CHARS - len(emotion) - 1, 20)
        text = _truncate_preserving_words(text, text_budget)
        return f"{emotion}|{text}"

    tag = DEFAULT_TAG
    text = raw_reply
    if "|" in raw_reply:
        potential_tag, _, remainder = raw_reply.partition("|")
        potential_tag = potential_tag.strip().upper()
        remainder = remainder.strip()
        if potential_tag in VALID_TAGS:
            tag = potential_tag
            text = remainder
        elif remainder:
            text = remainder

    text = strip_markdown(text)
    text = format_symbols(text)

    # ILI9341 Screen Fix
    text = text.replace('–', '-').replace('—', '-').replace('−', '-')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = _CONTROL_CHAR_RE.sub('', text)

    if not text or not text.strip(' |'):
        text = "I couldn't format a clean answer for that one."

    text_budget = max(MAX_DISPLAY_CHARS - len(tag) - 1, 20)
    text = _truncate_preserving_words(text, text_budget)
    return f"{tag}|{text}"


# ===========================================================================
# NEW: lightweight preprocessing helpers for intent detection / personalization
# ===========================================================================
#
# Everything above this line is UNCHANGED - same functions, same regexes,
# same behavior, same output for existing callers (parse_ai_reply's output
# format and content are untouched).
#
# The helpers below are ADDITIVE utilities used by intent_router.py and
# related modules for consistent, deterministic INPUT normalization
# (never output formatting). They are pure string/regex operations: no
# AI calls, no network calls, no database calls, and no new third-party
# NLP dependencies. They intentionally do NOT reuse format_symbols/
# strip_markdown, since those are display-formatting transforms for
# OUTGOING AI replies and would alter user-visible text in ways that
# don't belong in silent, incoming-message preprocessing for routing.

_WHITESPACE_RUN_RE = re.compile(r'\s+')
_LEADING_TRAILING_PUNCT_RE = re.compile(r'^[\s"\'.,!?;:]+|[\s"\'.,!?;:]+$')


def collapse_whitespace(text: str) -> str:
    """
    Collapse any run of whitespace (spaces, tabs, newlines) into a
    single space and trim the ends. Pure, deterministic, no side
    effects. Safe to call on already-clean text (no-op in that case).
    """
    if not text:
        return text or ""
    return _WHITESPACE_RUN_RE.sub(' ', text).strip()


def strip_control_characters(text: str) -> str:
    """
    Remove raw C0/DEL control characters from text. Exposed publicly
    (the parsing pipeline above already relies on the equivalent
    private regex for outgoing replies) so incoming-message
    preprocessing can apply the same safety guarantee before a message
    ever reaches intent detection or logging.
    """
    if not text:
        return text or ""
    return _CONTROL_CHAR_RE.sub('', text)


def normalize_for_matching(text: str) -> str:
    """
    Produce a lowercase, whitespace-collapsed version of text suitable
    for deterministic keyword/phrase matching (e.g. intent_router.py's
    scoring rules). This is a MATCHING-ONLY normalization - it must
    never be used to overwrite what's shown back to the user or stored
    as the canonical message text, since it discards case information.
    """
    if not text:
        return ""
    cleaned = strip_control_characters(text)
    return collapse_whitespace(cleaned).lower()


def clean_user_message(text: str, max_length: int = 2000) -> str:
    """
    General-purpose preprocessing for an INCOMING user message before
    it reaches intent detection, personalization, or the AI pipeline.
    Unlike normalize_for_matching, this preserves original casing and
    punctuation (so it's still fine to display back, log, or store) -
    it only strips control characters, collapses excess whitespace,
    and enforces a defensive max length so a pathological input can't
    balloon downstream processing (regex scoring, message history,
    etc.). Does not touch math/markdown formatting - that remains
    exclusive to the outgoing-reply pipeline (format_symbols/
    strip_markdown/parse_ai_reply) so this function can never change
    user-visible text in unexpected ways.
    """
    if not text:
        return ""
    cleaned = strip_control_characters(text)
    cleaned = collapse_whitespace(cleaned)
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned


def strip_edge_punctuation(text: str) -> str:
    """
    Strip leading/trailing quotes and common sentence punctuation
    (periods, commas, question/exclamation marks). Useful as a small
    phrase-matching aid (e.g. matching "algebra" against "algebra?" or
    "algebra."). Does not touch punctuation in the middle of the text.
    """
    if not text:
        return text or ""
    return _LEADING_TRAILING_PUNCT_RE.sub('', text)


def contains_phrase(text: str, phrase: str) -> bool:
    """
    Deterministic, case-insensitive whole-word/phrase containment
    check using regex word boundaries (e.g. "play" won't match inside
    "playground"). Intended as a small, reusable building block for
    lightweight intent/phrase matchers so callers don't each hand-roll
    their own boundary-aware regex.
    """
    if not text or not phrase:
        return False
    pattern = r'\b' + re.escape(phrase.strip().lower()) + r'\b'
    return re.search(pattern, normalize_for_matching(text)) is not None


def contains_any_phrase(text: str, phrases) -> bool:
    """
    True if ANY of the given phrases appears as a whole word/phrase
    match in text. Short-circuits on first match for efficiency.
    """
    if not text or not phrases:
        return False
    normalized = normalize_for_matching(text)
    for phrase in phrases:
        if not phrase:
            continue
        pattern = r'\b' + re.escape(phrase.strip().lower()) + r'\b'
        if re.search(pattern, normalized):
            return True
    return False


def find_matching_phrases(text: str, phrases) -> list:
    """
    Return the subset of `phrases` that match as whole words/phrases
    in text, preserving the input order. Useful for building the
    "signals"/"matched reasons" list intent_router.py and similar
    lightweight classifiers report alongside their decision.
    """
    if not text or not phrases:
        return []
    normalized = normalize_for_matching(text)
    matches = []
    for phrase in phrases:
        if not phrase:
            continue
        pattern = r'\b' + re.escape(phrase.strip().lower()) + r'\b'
        if re.search(pattern, normalized):
            matches.append(phrase)
    return matches
