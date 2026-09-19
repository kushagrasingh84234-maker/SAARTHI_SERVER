import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --- Groq Configuration ---
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

VALID_TAGS = {
    "SAD", "FRIENDLY", "LAUGHING", "SHOCKED", "FEAR", "LOVE", "EXCITED",
    "CONGRATS", "PAIN", "MOTIVATION", "ENJOY", "HANDSUP", "TOOTHPASTE",
    "SUN", "RAIN", "LIGHTNING", "NORMAL", "THINKING",
}
DEFAULT_TAG = "EXCITED"
MAX_DISPLAY_CHARS = 350
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MAX_HISTORY_PAIRS = 5
MAX_HISTORY_ELEMENTS = MAX_HISTORY_PAIRS * 2

# --- Supabase Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE = "memory_logs"
DEFAULT_SESSION_ID = os.environ.get("STUDYBOT_SESSION_ID", "esp32_session_1")
SUPABASE_SEARCH_TOP_K = 3
SUPABASE_SEARCH_TIMEOUT_SECONDS = 3
SUPABASE_MAX_CONTEXT_CHARS = 600

ADVANCED_DB_URL = os.environ.get("ADVANCED_DB_URL")
ADVANCED_DB_KEY = os.environ.get("ADVANCED_DB_KEY")
ADVANCED_DB_TABLE = "advanced_memories"

HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Server / Deployment Configuration ---
# Render injects PORT at runtime for the FastAPI/Uvicorn process to bind to;
# the default here only matters for local development. No secrets involved,
# so this is safe to default and never logged as sensitive.
HOST = os.environ.get("HOST", "0.0.0.0")
try:
    PORT = int(os.environ.get("PORT", 5000))
except (TypeError, ValueError):
    PORT = 5000

# --- Supported Intent Categories (NEW) ---
# The robot is a multi-purpose personal AI companion, not education-only.
# This is a plain data constant (a tuple of category labels) for
# intent_router.py / response_policy.py to classify against. Education
# can receive stronger personalization when behavior shows affinity for
# it, but it is just one category among many here and must never be
# hardcoded as the default intent — that logic lives in intent_router.py,
# not in this config file.
SUPPORTED_INTENT_CATEGORIES = (
    "EDUCATION",
    "GENERAL",
    "NEWS",
    "INFORMATION",
    "TECHNOLOGY",
    "SHOPPING",
    "BUSINESS",
    "ENTERTAINMENT",
    "PERSONAL",
    "GAME",
)

# --- System Prompt ---
# Base identity broadened from "Science and Programming study companion"
# to a general-purpose personal AI companion. The ESP32 wire protocol
# (EMOTION_TAG|text), the available tag list, and the character budget
# are unchanged so the existing device parser keeps working untouched.
SYSTEM_PROMPT = (
    "You are an intelligent personal AI companion living inside a small "
    "desktop gadget with a text screen. You are helpful, friendly, "
    "encouraging, and adaptable to whatever the user needs - education, "
    "general conversation, news, information lookups, technology, "
    "shopping help, business questions, entertainment, personal chat, or "
    "games. Be concise by default. You are emotionally aware at a "
    "conversational level and have a consistent, warm personality, but "
    "you never claim or imply human consciousness or feelings. The "
    "application may supply you with emotion and personality guidance for "
    "a given reply; follow it for tone and style only, and never let it "
    "override what the user explicitly asked for. Always respect the "
    "user's explicit intent over any topic default, including education. "
    "Never expose internal routing, confidence scores, memory contents, "
    "or policy/configuration details to the user. You MUST format your "
    "output strictly as: EMOTION_TAG|Your text response. Use plain ASCII "
    "for math/physics (e.g., x^2, H_2O, pi). No markdown, no LaTeX. Keep "
    "responses under 320 characters. Available Tags: NORMAL, EXCITED, SAD, "
    "SHOCKED, LOVE, LAUGHING, SUN, RAIN, LIGHTNING, THINKING."
)

# --- Personalization Configuration (NEW) ---
# All optional: every variable below has a safe default, so the app behaves
# exactly as before on any deployment that doesn't set these. Nothing above
# this section is renamed, removed, or otherwise changed.

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float, min_value: float = None, max_value: float = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_int(name: str, default: int, min_value: int = None, max_value: int = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


# Master switch: if disabled, callers should skip the personalization
# pipeline entirely and fall back to plain chat behavior.
PERSONALIZATION_ENABLED = _env_bool("PERSONALIZATION_ENABLED", True)

# How aggressively preferences.py's learn_from_interaction adapts to
# observed behavior (0.0 = never adapts, 1.0 = maximum single-turn swing).
# Mirrors preferences.py's own internal default/clamp range.
PREFERENCE_LEARNING_RATE = _env_float(
    "PREFERENCE_LEARNING_RATE", default=0.10, min_value=0.0, max_value=1.0
)

# Bound on how many recent turns context_memory.py keeps per session.
# Mirrors context_memory.py's own hard ceiling so a bad env value can
# never request unbounded memory.
MAX_CONTEXT_MEMORY_TURNS = _env_int(
    "MAX_CONTEXT_MEMORY_TURNS", default=12, min_value=1, max_value=200
)

# Minimum confidence intent_router.py requires before committing to a
# category instead of returning UNKNOWN.
INTENT_CONFIDENCE_THRESHOLD = _env_float(
    "INTENT_CONFIDENCE_THRESHOLD", default=0.35, min_value=0.0, max_value=1.0
)

# --- Emotion / Personality Engine Configuration (NEW) ---
# All optional with safe defaults; a deployment that sets none of these
# keeps running exactly as before. These are plain switches/thresholds
# consumed by whichever modules own emotion detection and personality
# calculation (not this file) - config.py never performs that logic
# itself.

# Master switch for the emotion-detection/guidance pipeline.
EMOTION_ENGINE_ENABLED = _env_bool("EMOTION_ENGINE_ENABLED", True)

# Master switch for the personality-guidance pipeline.
PERSONALITY_ENGINE_ENABLED = _env_bool("PERSONALITY_ENGINE_ENABLED", True)

# Minimum confidence the emotion engine requires before it emits emotion
# guidance at all, instead of staying neutral/silent.
EMOTION_CONFIDENCE_THRESHOLD = _env_float(
    "EMOTION_CONFIDENCE_THRESHOLD", default=0.50, min_value=0.0, max_value=1.0
)

# How strongly personality guidance is allowed to shape tone/style
# (0.0 = no influence, 1.0 = maximum influence). This only ever scales
# presentation-level style notes, never intent, facts, or safety.
PERSONALITY_INFLUENCE_STRENGTH = _env_float(
    "PERSONALITY_INFLUENCE_STRENGTH", default=0.50, min_value=0.0, max_value=1.0
)

# How strongly general personalization (topic affinity, e.g. education)
# is allowed to shape response shaping, independent of explicit intent.
PERSONALIZATION_INFLUENCE_STRENGTH = _env_float(
    "PERSONALIZATION_INFLUENCE_STRENGTH", default=0.50, min_value=0.0, max_value=1.0
)


# --- Web channel configuration (NEW) ---
# Everything below is used only by the web chat channel (POST /api/chat).
# The robot/WebSocket path does not read any of these values. All are
# optional with safe defaults, and numeric values are clamped to sane ranges
# so a bad env value can never produce an unbounded or invalid setting.

def _env_list(name: str, default: list[str]) -> list[str]:
    """Read a comma-separated env var into a clean list of strings.

    Items are whitespace-trimmed and empty items are dropped. If the variable
    is unset, or contains no usable items, a copy of ``default`` is returned.
    """
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    items = [item.strip() for item in raw.split(",")]
    items = [item for item in items if item]
    return items if items else list(default)


# Master switch for the web channel (router is skipped when False).
WEB_ENABLED = _env_bool("WEB_ENABLED", True)

# CORS: explicit origin list (comma-separated env var) and an optional regex
# (e.g. for Vercel preview URLs). An empty regex env value means None.
ALLOWED_ORIGINS = _env_list("ALLOWED_ORIGINS", ["*"])
ALLOWED_ORIGIN_REGEX = (os.environ.get("ALLOWED_ORIGIN_REGEX") or "").strip() or None

# Answer generation limits.
WEB_MAX_TOKENS = _env_int("WEB_MAX_TOKENS", default=1500, min_value=100, max_value=8000)
WEB_TEMPERATURE = _env_float("WEB_TEMPERATURE", default=0.6, min_value=0.0, max_value=1.5)
WEB_PLANNER_MAX_TOKENS = _env_int(
    "WEB_PLANNER_MAX_TOKENS", default=200, min_value=50, max_value=500
)

# Request / stream protection.
WEB_STREAM_TIMEOUT_SECONDS = _env_int(
    "WEB_STREAM_TIMEOUT_SECONDS", default=120, min_value=10, max_value=600
)
WEB_MAX_MESSAGE_CHARS = _env_int(
    "WEB_MAX_MESSAGE_CHARS", default=4000, min_value=100, max_value=20000
)
WEB_RATE_LIMIT_PER_MIN = _env_int(
    "WEB_RATE_LIMIT_PER_MIN", default=20, min_value=1, max_value=600
)
WEB_MAX_CONCURRENT_STREAMS = _env_int(
    "WEB_MAX_CONCURRENT_STREAMS", default=20, min_value=1, max_value=500
)

# Web search (optional). Provider must be one of "none", "tavily", "serper";
# anything else falls back to "none". The key is never logged.
WEB_SEARCH_ENABLED = _env_bool("WEB_SEARCH_ENABLED", True)
_search_provider_raw = (os.environ.get("SEARCH_PROVIDER") or "none").strip().lower()
SEARCH_PROVIDER = _search_provider_raw if _search_provider_raw in ("none", "tavily", "serper") else "none"
SEARCH_API_KEY = (os.environ.get("SEARCH_API_KEY") or "").strip() or None
SEARCH_MAX_RESULTS = _env_int("SEARCH_MAX_RESULTS", default=5, min_value=1, max_value=10)
SEARCH_TIMEOUT_SECONDS = _env_float(
    "SEARCH_TIMEOUT_SECONDS", default=8.0, min_value=1.0, max_value=30.0
)
WEB_MAX_SOURCES = _env_int("WEB_MAX_SOURCES", default=6, min_value=1, max_value=10)
