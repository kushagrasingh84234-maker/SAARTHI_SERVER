"""storage/chat_layer_manager.py

Chat-side Layers 1+2 of SAARTHI_SERVER's 3-layer data structure, for
thousands of web/app users (user_id / session_id). Built on top of
storage/db_config.py's execute_chat_query and get_mongo_collection.

record_chat_turn() is the single write entry point: it decides whether a
turn is an error (-> chat_error_logs) or a normal exchange (->
chat_normal_logs, mirrored into MongoDB's "chat_sessions" collection when
available). get_recent_chat_history()/get_user_permanent_facts() are the
read side, used to rebuild context for a session or surface a user's
long-term "golden facts" (Layer 3, storage/layer3_monthly_cleaner.py).
"""

import logging
import time
from typing import Any, Dict, List

from storage.db_config import execute_chat_query, get_mongo_collection

logger = logging.getLogger(__name__)


def _sync_to_mongo(
    user_id: str,
    session_id: str,
    channel: str,
    user_message: str,
    assistant_reply: str,
    intent: str,
    ts: float,
) -> bool:
    """Best-effort mirror of one chat exchange into MongoDB Atlas's
    "chat_sessions" collection - one document per session_id, with a
    growing "turns" array. Returns False (never raises) if MongoDB isn't
    configured/reachable, or if the write itself fails."""
    collection = get_mongo_collection("chat_sessions")
    if collection is None:
        return False
    turn = {
        "ts": ts,
        "channel": channel,
        "intent": intent,
        "user_message": user_message or "",
        "assistant_reply": assistant_reply or "",
    }
    try:
        collection.update_one(
            {"session_id": session_id},
            {
                "$set": {"user_id": user_id, "session_id": session_id, "last_active_ts": ts},
                "$push": {"turns": turn},
            },
            upsert=True,
        )
        return True
    except Exception as e:
        logger.warning(f"Chat Layer: MongoDB sync failed for session_id={session_id}: {e}")
        return False


def record_chat_turn(
    user_id: str,
    session_id: str,
    channel: str,
    user_message: str,
    assistant_reply: str,
    intent: str = "chat",
    error_flag: bool = False,
    error_type: str = "",
    error_details: str = "",
) -> dict:
    """Record one chat exchange (or one chat-turn error) for a user/session.

    If error_flag is True, writes a single row to chat_error_logs
    describing the failure (a model timeout, an empty reply, a parsing
    error, etc.) - user_message/assistant_reply are not additionally
    stored as chat_normal_logs rows for that turn.

    Otherwise, writes the user's message and the assistant's reply as up
    to two rows in chat_normal_logs (role="user" / role="assistant"),
    silently skipping either one if it's empty/whitespace-only noise,
    and - if MongoDB is available - mirrors the same exchange into a
    per-session "chat_sessions" document too.

    Never raises: every write is independently guarded (a MongoDB sync
    failure never affects whether the SQL write itself succeeded, and
    vice versa).

    Returns:
        {
            "stored_as": "error" | "normal" | "skipped",
            "sql_rows_written": int,
            "mongo_synced": bool,
        }
    """
    now = time.time()

    if error_flag:
        cur = execute_chat_query(
            "INSERT INTO chat_error_logs (user_id, session_id, channel, error_type, "
            "user_message, error_details, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, session_id, channel, error_type or "UNKNOWN_ERROR",
             user_message or "", error_details or "", now),
        )
        return {
            "stored_as": "error",
            "sql_rows_written": 1 if cur is not None else 0,
            "mongo_synced": False,
        }

    rows = []
    if user_message and user_message.strip():
        rows.append((user_id, session_id, channel, "user", user_message.strip(), intent, now))
    if assistant_reply and assistant_reply.strip():
        rows.append((user_id, session_id, channel, "assistant", assistant_reply.strip(), intent, now))

    if not rows:
        return {"stored_as": "skipped", "sql_rows_written": 0, "mongo_synced": False}

    written = 0
    for row in rows:
        cur = execute_chat_query(
            "INSERT INTO chat_normal_logs (user_id, session_id, channel, role, content, "
            "intent, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        if cur is not None:
            written += 1
        else:
            logger.error(f"Chat Layer: failed to write a chat_normal_logs row for session_id={session_id}")

    mongo_synced = _sync_to_mongo(user_id, session_id, channel, user_message, assistant_reply, intent, now)

    return {"stored_as": "normal", "sql_rows_written": written, "mongo_synced": mongo_synced}


def get_recent_chat_history(session_id: str, limit: int = 20) -> List[dict]:
    """Return up to `limit` most recent chat_normal_logs rows for
    `session_id`, oldest-first (so callers can feed them straight into a
    prompt-building function in chronological order). Returns [] on any
    query failure - never raises.

    Ordered by (created_at, id) rather than created_at alone: a user
    turn and its assistant reply are written back-to-back and can share
    the exact same created_at timestamp, so `id` (insertion order) is
    used as the tiebreaker to keep the user's message before its own
    reply.
    """
    cur = execute_chat_query(
        "SELECT role, content, intent, created_at FROM chat_normal_logs "
        "WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
        (session_id, limit),
    )
    if cur is None:
        return []
    try:
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Chat Layer: failed to fetch recent history for session_id={session_id}: {e}")
        return []
    history = [
        {"role": r[0], "content": r[1], "intent": r[2], "created_at": r[3]}
        for r in rows
    ]
    history.reverse()  # was newest-first (for LIMIT to keep the right rows); return oldest-first
    return history


def get_user_permanent_facts(user_id: str, limit: int = 20) -> List[dict]:
    """Return up to `limit` chat_permanent_memory rows for `user_id`
    (Layer 3's "golden facts"), most-important-first. Returns [] on any
    query failure - never raises."""
    cur = execute_chat_query(
        "SELECT session_id, fact_category, golden_fact, importance_score, created_at "
        "FROM chat_permanent_memory WHERE user_id = ? "
        "ORDER BY importance_score DESC, created_at DESC LIMIT ?",
        (user_id, limit),
    )
    if cur is None:
        return []
    try:
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Chat Layer: failed to fetch permanent facts for user_id={user_id}: {e}")
        return []
    return [
        {
            "session_id": r[0], "fact_category": r[1], "golden_fact": r[2],
            "importance_score": r[3], "created_at": r[4],
        }
        for r in rows
    ]
