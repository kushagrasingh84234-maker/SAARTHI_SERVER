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

def parse_ai_reply(raw_reply: str) -> str:
    raw_reply = (raw_reply or "").strip()
    if not raw_reply:
        return "SAD|I didn't get a response, please try again."
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
