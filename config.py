import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --- Groq Configuration (LEGACY NAMES, KEPT FOR BACKWARD COMPATIBILITY) ---
# The server no longer calls the real Groq cloud API - inference now runs
# locally (see "Local Model / Kaggle GPU Configuration" below). These three
# names are kept exactly as-is, unused by the new inference path, purely so
# none of the other ~31 files that import them break with an ImportError.
# GROQ_MODEL is repurposed to carry the local model identifier instead of a
# Groq model string.
GROQ_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Local Model / Kaggle GPU Configuration (NEW) ---
# Base vision-language model plus our fine-tuned LoRA adapter, loaded
# locally instead of calling out to Groq. Everything here is optional with
# safe defaults so importing this file never requires the model files to
# be present.
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
SAARTHI_ADAPTER_PATH = os.environ.get("SAARTHI_ADAPTER_PATH", "./saarthi_v2_perfect")

# 4-bit quantized loading (bitsandbytes) to fit Kaggle's free GPU memory.
# (Defined with a plain parse, not _env_bool, since that helper is declared
# further down in this file and isn't available yet at this point.)
LOAD_IN_4BIT = (os.environ.get("LOAD_IN_4BIT", "true").strip().lower() in ("1", "true", "yes", "on"))

# How many inference requests may run on the GPU at once. Kaggle typically
# offers 1 or 2 GPUs; keep this at 1 unless the deployment explicitly
# configures a dual-GPU setup.
GPU_MAX_CONCURRENT_INFERENCES = int(os.environ.get("GPU_MAX_CONCURRENT_INFERENCES", 1))

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

# --- System Prompt (LEGACY NAME, KEPT FOR BACKWARD COMPATIBILITY) ---
# Base identity broadened from "Science and Programming study companion"
# to a general-purpose personal AI companion. The old hard EMOTION_TAG|text
# wire-format requirement, the "no markdown/no LaTeX" rule, and the 320
# character cap have been removed from here: they were breaking the
# robot channel's [ACTION] JSON command output and the web channel's
# markdown rendering. Channel-specific formatting now belongs to
# ROBOT_CONTROL_SYSTEM_PROMPT / CONVERSATIONAL_SYSTEM_PROMPT below; this
# variable name is kept only so any of the other ~31 files still importing
# SYSTEM_PROMPT directly keep working.
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
    "or policy/configuration details to the user."
)

# --- Dual-Mode System Prompts (NEW) ---
# Replaces the single rigid SYSTEM_PROMPT contract above with two
# channel-specific prompts: one for the embodied robot (which must emit a
# structured JSON action command) and one for plain conversational
# channels (web / mobile / desktop), which are free to use normal
# markdown-formatted prose.
ROBOT_CONTROL_SYSTEM_PROMPT = (
    "You are SAARTHI, an embodied desktop robot assistant. "
    "[MODE: ROBOT_CONTROL] Analyze sensor data, reason step-by-step, then "
    "output a precise JSON action command."
)

CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are SAARTHI, a helpful and warm desktop companion robot. "
    "[MODE: CONVERSATIONAL] Respond naturally and helpfully."
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
WEB_MAX_TOKENS = _env_int("WEB_MAX_TOKENS", default=2500, min_value=100, max_value=8000)
WEB_TEMPERATURE = _env_float("WEB_TEMPERATURE", default=0.6, min_value=0.0, max_value=1.5)
WEB_PLANNER_MAX_TOKENS = _env_int(
    "WEB_PLANNER_MAX_TOKENS", default=600, min_value=50, max_value=2000
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
    "WEB_MAX_CONCURRENT_STREAMS", default=4, min_value=1, max_value=500
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

# Reasoning effort sent to Groq for gpt-oss models ("low", "medium", "high").
# An empty string disables the field. Now disabled by default: this field
# was specific to Groq's gpt-oss models and has no meaning for the local
# Qwen3-VL model, but the name/validation is kept so nothing importing it
# breaks. Unknown non-empty values still fall back to "low".
_reasoning_raw = (os.environ.get("WEB_REASONING_EFFORT", "") or "").strip().lower()
WEB_REASONING_EFFORT = _reasoning_raw if _reasoning_raw in ("", "low", "medium", "high") else "low"

# Local shortcuts (date, time, simple maths) on the web channel. Off by
# default: the regex answers "what is time?" with the clock time.
WEB_LOCAL_SHORTCUTS = _env_bool("WEB_LOCAL_SHORTCUTS", False)


# --- Image Ingestion Configuration (NEW) ---
# Every upload is downsized/recompressed before it ever reaches the model,
# so a single oversized image can't blow up VRAM or the prompt payload.
IMAGE_MAX_UPLOAD_MB = _env_float(
    "IMAGE_MAX_UPLOAD_MB", default=10.0, min_value=0.1, max_value=100.0
)
IMAGE_TARGET_RESOLUTION = _env_int(
    "IMAGE_TARGET_RESOLUTION", default=448, min_value=64, max_value=2048
)
IMAGE_COMPRESSION_QUALITY = _env_int(
    "IMAGE_COMPRESSION_QUALITY", default=80, min_value=1, max_value=100
)
IMAGE_MAX_COMPRESSED_KB = _env_int(
    "IMAGE_MAX_COMPRESSED_KB", default=100, min_value=1, max_value=10000
)

# --- Video Ingestion Configuration (NEW) ---
# Videos are capped hard on duration/size and reduced to a handful of
# low-res frames rather than sent to the model whole. A tight per-second
# rate limit stops a burst of video uploads from starving the GPU queue.
VIDEO_MAX_DURATION_SECONDS = _env_float(
    "VIDEO_MAX_DURATION_SECONDS", default=5.0, min_value=0.5, max_value=120.0
)
VIDEO_MAX_UPLOAD_MB = _env_float(
    "VIDEO_MAX_UPLOAD_MB", default=3.0, min_value=0.1, max_value=100.0
)
VIDEO_TARGET_FPS = _env_int("VIDEO_TARGET_FPS", default=1, min_value=1, max_value=30)
VIDEO_MAX_FRAMES = _env_int("VIDEO_MAX_FRAMES", default=5, min_value=1, max_value=64)
VIDEO_FRAME_RESOLUTION = _env_int(
    "VIDEO_FRAME_RESOLUTION", default=336, min_value=64, max_value=2048
)
VIDEO_RATE_LIMIT_SECONDS = _env_float(
    "VIDEO_RATE_LIMIT_SECONDS", default=1.0, min_value=0.1, max_value=60.0
)
VIDEO_MAX_PER_SECOND = _env_int(
    "VIDEO_MAX_PER_SECOND", default=1, min_value=1, max_value=100
)

# --- Document Ingestion Configuration (NEW) ---
# Uploaded documents are capped in size, restricted to a known-safe
# extension list, and truncated in both page count and extracted
# character count before anything from them enters a prompt/RAG context.
DOC_MAX_UPLOAD_MB = _env_float(
    "DOC_MAX_UPLOAD_MB", default=2.0, min_value=0.1, max_value=100.0
)
DOC_ALLOWED_EXTENSIONS = tuple(
    _env_list("DOC_ALLOWED_EXTENSIONS", [".pdf", ".txt", ".md", ".docx"])
)
DOC_MAX_PAGES = _env_int("DOC_MAX_PAGES", default=10, min_value=1, max_value=1000)
DOC_MAX_EXTRACTED_CHARS = _env_int(
    "DOC_MAX_EXTRACTED_CHARS", default=2000, min_value=100, max_value=100000
)
