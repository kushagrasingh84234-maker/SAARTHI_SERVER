import json
import time
import asyncio
import logging
import threading
import hashlib
from contextlib import asynccontextmanager
from collections import defaultdict, deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from cachetools import TTLCache

from config import (
    GROQ_MODEL, DEFAULT_SESSION_ID, HOST, PORT, PERSONALIZATION_ENABLED,
    ALLOWED_ORIGINS, ALLOWED_ORIGIN_REGEX, WEB_ENABLED, SEARCH_PROVIDER,
)
from text_utils import parse_ai_reply
from logic import handle_local_queries, run_personalized_pipeline, get_session_state
from ai_services import call_groq
from game_manager import handle_game_message
from web_channel.web_api import router as web_router
from database import (
    add_to_history, route_and_save_bg, retrieve_long_term_context,
    build_groq_messages, CHAT_HISTORY, SUPABASE_ENABLED, ADVANCED_DB_ENABLED,
    save_personalization_profile_bg,
)

# IMPROVEMENT: added timestamp/logger-name to the default format for easier
# correlation of related log lines when reading Render's log stream.
# Level and destination (stdout) are unchanged.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# IMPROVEMENT: process start time, used only to report uptime on /health.
START_TIME = time.time()

# ---- Response cache (now session-aware) ----
response_cache = TTLCache(maxsize=500, ttl=300)
cache_lock = threading.Lock()


def _cache_key(session_id: str, user_message: str) -> str:
    normalized = f"{session_id}:{user_message.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---- Manual rate limiting ----
RATE_LIMIT_MAX = 15
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_store = defaultdict(deque)
_rate_limit_lock = threading.Lock()

# IMPROVEMENT: without this, _rate_limit_store keeps one deque forever for
# every distinct key it has ever seen (client IPs, or the session-id fallback
# below), even long after that client stops connecting. The deques themselves
# stay small (capped at RATE_LIMIT_MAX), but the number of *keys* grows
# without bound over the life of the process. This periodic sweep drops keys
# whose most recent timestamp has already aged out of the window. It runs
# under the same lock every call already takes, so it adds no new locking,
# and it only ever deletes entries that are already expired for rate-limiting
# purposes - it cannot make an active client incorrectly un-limited or
# limited.
_RATE_LIMIT_SWEEP_INTERVAL = 500
_rate_limit_call_counter = 0


def _is_rate_limited(key: str) -> bool:
    global _rate_limit_call_counter
    now = time.time()
    with _rate_limit_lock:
        timestamps = _rate_limit_store[key]
        while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
            timestamps.popleft()

        limited = len(timestamps) >= RATE_LIMIT_MAX
        if not limited:
            timestamps.append(now)

        _rate_limit_call_counter += 1
        if _rate_limit_call_counter >= _RATE_LIMIT_SWEEP_INTERVAL:
            _rate_limit_call_counter = 0
            stale_keys = [
                k for k, dq in _rate_limit_store.items()
                if not dq or now - dq[-1] > RATE_LIMIT_WINDOW_SECONDS
            ]
            for k in stale_keys:
                del _rate_limit_store[k]

        return limited


def _split_emotion_text(pipe_formatted: str):
    pipe_formatted = pipe_formatted or ""
    emotion, sep, text = pipe_formatted.partition("|")
    if not sep:
        return "NEUTRAL", pipe_formatted
    return (emotion.strip() or "NEUTRAL"), text


def _persist_personalization_bg(session_id: str) -> None:
    """
    NEW: best-effort, fire-and-forget sync of the current in-memory
    personalization state (logic.py) into persistent storage
    (database.py). save_personalization_profile_bg already spawns its
    own background thread and is a safe no-op if Supabase isn't
    configured, so this call never blocks the event loop and never
    raises into the caller.
    """
    try:
        user_profile, preference_profile = get_session_state(session_id)
        save_personalization_profile_bg(
            session_id=session_id,
            category_counts=user_profile.category_counts,
            recent_intents=list(user_profile.recent_intents),
            preferred_response_length=user_profile.preferred_response_length,
            preferred_interaction_style=user_profile.preferred_interaction_style,
            interests=user_profile.interests,
            preference_scores=preference_profile.raw_scores,
        )
    except Exception as e:
        logger.warning(f"Skipping personalization persistence for session_id={session_id}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up.")
    logger.info(f"Groq model: {GROQ_MODEL} | Supabase enabled: {SUPABASE_ENABLED} | "
                f"Advanced vector memory enabled: {ADVANCED_DB_ENABLED} | "
                f"Personalization enabled: {PERSONALIZATION_ENABLED}")
    logger.info(
        f"Web channel enabled: {WEB_ENABLED} | Search provider: {SEARCH_PROVIDER} | "
        f"Allowed origins: {len(ALLOWED_ORIGINS)}"
    )
    yield
    # IMPROVEMENT: surface in-memory state size on shutdown, useful when
    # correlating a Render restart/deploy with how much session state existed
    # at the time.
    logger.info(
        f"Server shutting down. active_websocket_connections={len(active_connections)} "
        f"tracked_chat_sessions={len(CHAT_HISTORY)}"
    )


app = FastAPI(lifespan=lifespan)

if ALLOWED_ORIGINS == ["*"]:
    logger.warning("ALLOWED_ORIGINS is not set; CORS is open. Set it in production.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- NEW: web chat channel (POST /api/chat, GET /api/health) ----
if WEB_ENABLED:
    app.include_router(web_router)


@app.get("/")
async def root():
    return {"service": "saarthi", "status": "ok"}


# ---- NEW: one active WebSocket per session_id ----
# Maps session_id -> the currently-registered WebSocket instance for
# that robot. When a second connection shows up for the same
# session_id (e.g. a duplicate/stale ESP32 connection), the old one
# is closed so only one connection is ever "live" per session.
active_connections: dict[str, WebSocket] = {}
active_connections_lock = asyncio.Lock()


async def _run_ai_pipeline(user_message: str, session_id: str, long_term_context: str) -> str:
    """
    NEW: orchestrates the Adaptive Personalization Engine on top of the
    existing AI call path:

        logic.py -> intent_router -> personalization -> response_policy
        -> ai_services (call_groq)

    This function OWNS NONE of that logic itself - it only wires
    already-built modules together and decides how to fall back if
    personalization is disabled or fails. Game routing has already
    happened by the time this is ever called (see _process_chat_message),
    and Fast Math / any active game session never reaches this function
    at all, since handle_game_message short-circuits before it.

    Returns the raw AI reply string (still in EMOTION|text pipe format,
    exactly like the original direct call_groq(messages) path), so the
    caller's existing parse_ai_reply/caching/history logic is unchanged.
    """
    # build_messages_fn closure captures the already-fetched
    # long_term_context so logic.py's pipeline never has to perform its
    # own blocking Supabase lookup inside the event loop.
    def _build_messages(sid: str, _msg: str):
        return build_groq_messages(long_term_context, sid)

    if PERSONALIZATION_ENABLED:
        try:
            result = await run_personalized_pipeline(
                user_message=user_message,
                session_id=session_id,
                build_messages_fn=_build_messages,
                ai_call_fn=call_groq,
            )
        except Exception as e:
            # Personalization must never break chat. Any unexpected
            # failure here falls back to the original direct path below.
            logger.warning(f"Personalization pipeline error for session_id={session_id}: {e}")
            result = None

        if result is not None:
            logger.info(
                f"session_id={session_id} routed intent={result.get('intent')} "
                f"confidence={result.get('confidence')}"
            )
            # Fire-and-forget persistence of updated profile/preferences.
            _persist_personalization_bg(session_id)
            return result["raw_reply"]

    # Fallback: personalization disabled, returned None (e.g. empty
    # message, invalid profile state), or failed - use the exact
    # original direct AI call path with the same messages.
    messages = build_groq_messages(long_term_context, session_id)
    return await call_groq(messages)


async def _process_chat_message(user_message: str, session_id: str) -> dict:
    # NEW: Game Manager intercept. If the session is chatting and this
    # message isn't game-related, handle_game_message returns None and
    # we fall through to the existing pipeline untouched. If the session
    # is mid-game (or this message starts/continues one), it returns a
    # ready-to-send response dict and we short-circuit everything below
    # (personalization, Groq, cache, history, long-term memory) for this
    # message. Game handling always stays higher priority than normal
    # chat / AI processing - an active Fast Math session never reaches
    # the AI pipeline.
    game_reply = await asyncio.to_thread(handle_game_message, session_id, user_message)
    if game_reply is not None:
        return game_reply

    if not user_message:
        emotion, text = _split_emotion_text(parse_ai_reply(""))
        return {"type": "response", "emotion": emotion, "text": text}

    local_reply = await asyncio.to_thread(handle_local_queries, user_message)
    if local_reply is not None:
        _, local_text_only = _split_emotion_text(local_reply)
        await asyncio.to_thread(add_to_history, "user", user_message, session_id)
        await asyncio.to_thread(add_to_history, "model", local_text_only or local_reply, session_id)
        await asyncio.to_thread(route_and_save_bg, "user", user_message, session_id)
        await asyncio.to_thread(route_and_save_bg, "assistant", local_text_only or local_reply, session_id)
        emotion, text = _split_emotion_text(local_reply)
        return {"type": "response", "emotion": emotion, "text": text}

    long_term_context = await asyncio.to_thread(retrieve_long_term_context, user_message, session_id)
    await asyncio.to_thread(add_to_history, "user", user_message, session_id)

    cache_key = _cache_key(session_id, user_message)
    with cache_lock:
        cached_reply = response_cache.get(cache_key)

    if cached_reply is not None:
        formatted_reply = cached_reply
    else:
        try:
            # NEW: routes through logic.py's Adaptive Personalization
            # Engine (intent_router -> personalization -> response_policy)
            # before reaching ai_services, with an automatic fallback to
            # the original direct call_groq path if personalization is
            # disabled or fails for any reason.
            raw_reply = await _run_ai_pipeline(user_message, session_id, long_term_context)
        except Exception as e:
            logger.error(f"Groq call failed after retries: {e}")
            emotion, text = _split_emotion_text("SAD|My brain is offline right now, try again in a bit.")
            return {"type": "response", "emotion": emotion, "text": text}

        formatted_reply = parse_ai_reply(raw_reply)
        with cache_lock:
            response_cache[cache_key] = formatted_reply

    _, reply_text_only = _split_emotion_text(formatted_reply)
    await asyncio.to_thread(add_to_history, "model", reply_text_only or formatted_reply, session_id)
    await asyncio.to_thread(route_and_save_bg, "user", user_message, session_id)
    await asyncio.to_thread(route_and_save_bg, "assistant", reply_text_only or formatted_reply, session_id)

    emotion, text = _split_emotion_text(formatted_reply)
    return {"type": "response", "emotion": emotion, "text": text}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # IMPROVEMENT: accept() can fail if the client drops mid-handshake.
    # Previously an exception here would propagate as an unhandled error;
    # now it's logged cleanly and the handler exits before touching
    # active_connections (nothing to clean up yet at this point either way).
    try:
        await websocket.accept()
    except Exception as e:
        logger.warning(f"WebSocket accept failed for session_id={session_id}: {e}")
        return

    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"Client connected: session_id={session_id}, ip={client_ip}")

    # NEW: duplicate-session guard. If another connection is already
    # registered for this session_id, it's stale (the ESP32 only ever
    # intends one connection per session) — take over as the active
    # connection first, then close the stale one.
    #
    # FIX (production-safety review): the registry swap happens under
    # the lock (fast, in-memory, atomic), but closing the old socket
    # does NOT — active_connections_lock is shared by every session on
    # the server, so awaiting a slow/unresponsive old_ws.close() while
    # holding it would stall connect/disconnect handling for every
    # OTHER session too. The registry is already fully consistent by
    # the time we close(), so this reordering changes no behavior; it
    # only stops one stale socket from stalling the whole server. The
    # disconnect handler's identity check below (SPECIAL CHECK) still
    # guarantees a newer connection can never be deleted by an older
    # one's cleanup, regardless of close() ordering.
    async with active_connections_lock:
        old_ws = active_connections.get(session_id)
        active_connections[session_id] = websocket

    if old_ws is not None and old_ws is not websocket:
        logger.info(f"Duplicate connection for session_id={session_id} — closing previous one")
        try:
            await old_ws.close()
        except Exception:
            pass

    try:
        while True:
            try:
                raw_data = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: session_id={session_id}, ip={client_ip}")
                break

            try:
                data = json.loads(raw_data)
                if not isinstance(data, dict):
                    raise ValueError("Message must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Malformed message from session_id={session_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "emotion": "CONFUSED",
                    "text": "Malformed message received.",
                })
                continue

            msg_type = data.get("type", "chat")
            if msg_type != "chat":
                await websocket.send_json({
                    "type": "error",
                    "emotion": "CONFUSED",
                    "text": f"Unsupported message type: {msg_type}",
                })
                continue

            user_message = (data.get("text") or "").strip()
            effective_session_id = data.get("session_id") or session_id or DEFAULT_SESSION_ID

            # IMPROVEMENT: session-id consistency diagnostic. The URL's
            # session_id is what active_connections/dedup keys on; the
            # per-message session_id (if the payload sends one) is what
            # actually drives history/cache/game state. These are expected
            # to match for the ESP32 protocol as-is; this only logs at
            # DEBUG (silent under the default INFO level) so it adds zero
            # log noise in normal operation but is available if someone
            # bumps the log level while chasing a session-mismatch bug.
            if effective_session_id != session_id:
                logger.debug(
                    f"session_id mismatch: url_session_id={session_id} "
                    f"payload_session_id={effective_session_id}"
                )

            # IMPROVEMENT: fall back to a session-scoped rate-limit key when
            # the client IP isn't available, instead of bucketing every
            # such connection together under the single literal string
            # "unknown" (which previously meant one misbehaving session
            # with no visible IP could rate-limit every other session in
            # the same situation).
            rate_limit_key = client_ip if client_ip != "unknown" else f"session:{session_id}"
            if _is_rate_limited(rate_limit_key):
                await websocket.send_json({
                    "type": "response",
                    "emotion": "SAD",
                    "text": "Too many requests, slow down a bit.",
                })
                continue

            # IMPROVEMENT: measure and log per-message processing latency.
            # Pure diagnostic addition — does not affect control flow or
            # what gets sent back to the client.
            start_ts = time.monotonic()
            try:
                result = await _process_chat_message(user_message, effective_session_id)
            except Exception as e:
                logger.exception(f"Unhandled error processing message for session_id={session_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "emotion": "SAD",
                    "text": "Something went wrong processing that message.",
                })
                continue
            finally:
                elapsed_ms = (time.monotonic() - start_ts) * 1000
                logger.info(
                    f"session_id={effective_session_id} ip={client_ip} "
                    f"processed in {elapsed_ms:.1f}ms"
                )

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: session_id={session_id}, ip={client_ip}")
    except Exception as e:
        logger.exception(f"WebSocket connection error for session_id={session_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        # SPECIAL CHECK: only remove this session's entry if it's still
        # THIS websocket instance. If a newer connection has already
        # taken over active_connections[session_id], this cleanup must
        # not delete it. Verified correct as-is — unchanged.
        async with active_connections_lock:
            if active_connections.get(session_id) is websocket:
                del active_connections[session_id]


@app.get("/health")
async def health():
    # IMPROVEMENT: read shared state under the same locks used elsewhere so
    # these counts can't be corrupted by a concurrent connect/disconnect or
    # rate-limit check — matters more now that /health reports on them.
    async with active_connections_lock:
        active_ws_count = len(active_connections)
    with _rate_limit_lock:
        rate_limit_keys_tracked = len(_rate_limit_store)

    return {
        "status": "ok",
        "transport": "websocket (fastapi native)",
        "groq_model": GROQ_MODEL,
        "long_term_memory_enabled": SUPABASE_ENABLED,
        "advanced_vector_memory_enabled": ADVANCED_DB_ENABLED,
        "personalization_enabled": PERSONALIZATION_ENABLED,
        "active_sessions": len(CHAT_HISTORY),
        "cache_size": len(response_cache),
        # IMPROVEMENT: new diagnostic fields, additive only — existing keys
        # above are all unchanged.
        "active_websocket_connections": active_ws_count,
        "rate_limit_keys_tracked": rate_limit_keys_tracked,
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
