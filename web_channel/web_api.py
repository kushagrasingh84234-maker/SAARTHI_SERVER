import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

import database

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
HISTORY_COLUMNS = "id, session_id, role, content, created_at"
HISTORY_LIMIT = 100
MAX_MESSAGE_CHARS = 4000

# Keeps references to background save tasks so they are not garbage-collected
# before they finish.
_background_tasks: set = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _supabase_ready() -> bool:
    """True only if Supabase is enabled and the client actually exists."""
    return bool(getattr(database, "SUPABASE_ENABLED", False)) and (
        getattr(database, "_supabase_client", None) is not None
    )


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _save_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    """Best-effort save of one user/assistant turn to memory_logs.

    Never raises: a database problem must not break the chat stream.
    """
    if not _supabase_ready():
        return

    client = database._supabase_client
    rows = [{"session_id": session_id, "role": "user", "content": user_text}]
    if assistant_text:
        rows.append(
            {"session_id": session_id, "role": "assistant", "content": assistant_text}
        )

    def _insert():
        return client.table("memory_logs").insert(rows).execute()

    try:
        await asyncio.to_thread(_insert)
    except Exception as e:
        logger.error(f"Error saving chat turn for {session_id}: {e}")


def _schedule_save(session_id: str, user_text: str, assistant_text: str) -> None:
    """Run _save_turn in the background without blocking the stream."""
    task = asyncio.create_task(_save_turn(session_id, user_text, assistant_text))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ---------------------------------------------------------------------------
# Token source (PLACEHOLDER)
# ---------------------------------------------------------------------------
# Replace this function with your real AI streaming call (Groq, etc.).
# It must be an async generator that yields plain text chunks (tokens).
async def _token_source(
    message: str, session_id: str, conversation_id: str, channel: str
) -> AsyncGenerator[str, None]:
    reply = f"Namaste! I am Saarthi. This is a placeholder reply to: {message}"
    for word in reply.split(" "):
        await asyncio.sleep(0.03)
        yield word + " "


async def generate_reply_stream(
    message: str, session_id: str, conversation_id: str, channel: str
) -> AsyncGenerator[str, None]:
    """Yield SSE frames: status -> token(s) -> done (or error).

    The full reply is saved to the database in the background once the
    stream ends, even if the client disconnects midway.
    """
    parts: list = []
    try:
        yield _sse("status", {"message": "Thinking..."})

        async for token in _token_source(message, session_id, conversation_id, channel):
            parts.append(token)
            yield _sse("token", {"text": token})

        yield _sse("done", {"conversation_id": conversation_id})

    except asyncio.CancelledError:
        # Client disconnected; the finally block still saves what we have.
        raise
    except Exception as e:
        logger.error(f"Chat stream error for {session_id}: {e}")
        yield _sse("error", {"message": "Something went wrong. Please try again."})
    finally:
        _schedule_save(session_id, message, "".join(parts).strip())


# ---------------------------------------------------------------------------
# POST /chat  (SSE stream)
# ---------------------------------------------------------------------------
@router.post("/chat")
async def chat(request: Request):
    """Main Saarthi chat endpoint. Returns a Server-Sent Events stream."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "JSON body must be an object."}, status_code=400)

    message = str(payload.get("message") or "").strip()
    session_id = str(payload.get("session_id") or "").strip()
    conversation_id = str(payload.get("conversation_id") or "").strip() or "default"
    channel = str(payload.get("channel") or "web").strip() or "web"

    if not message:
        return JSONResponse({"error": "'message' is required."}, status_code=400)
    if not session_id:
        return JSONResponse({"error": "'session_id' is required."}, status_code=400)
    if len(message) > MAX_MESSAGE_CHARS:
        return JSONResponse(
            {"error": f"'message' is too long (max {MAX_MESSAGE_CHARS} characters)."},
            status_code=413,
        )

    return StreamingResponse(
        generate_reply_stream(message, session_id, conversation_id, channel),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop proxies from buffering the stream
        },
    )


# ---------------------------------------------------------------------------
# GET /history/{session_id}
# ---------------------------------------------------------------------------
@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """Fetch chat history for a specific session from Supabase."""
    if not _supabase_ready():
        return {"messages": []}

    client = database._supabase_client

    def _fetch():
        # Newest N rows (desc + limit); reversed below to chronological order.
        return (
            client.table("memory_logs")
            .select(HISTORY_COLUMNS)
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(HISTORY_LIMIT)
            .execute()
        )

    try:
        # The Supabase client is synchronous, so run it in a worker thread.
        resp = await asyncio.to_thread(_fetch)
        messages = list(reversed(resp.data or []))
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Error fetching history for {session_id}: {e}")
        return JSONResponse({"messages": [], "error": True}, status_code=500)
      
