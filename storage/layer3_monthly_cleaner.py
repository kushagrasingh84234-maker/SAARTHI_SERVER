"""storage/layer3_monthly_cleaner.py

Layer 3 of SAARTHI_SERVER's 3-layer data structure: a lifecycle job (run
monthly, or on demand) that keeps the Robot Fleet DB and User Chat SQL DB
from growing without bound.

For every robot_id/user_id with data older than `retention_days`, the
most valuable ~keep_top_ratio (default 10%) of it is extracted into a
permanent-memory table (robot_permanent_memory / chat_permanent_memory),
then ALL of the old raw rows (the 90%+ that wasn't kept) are deleted from
their source tables.

"Most valuable" is decided by simple, cheap, non-ML heuristics (see
_sensor... no - see run_robot_monthly_cleanup's own docstring and
_score_chat_message below) - this job is meant to run cheaply and
offline over potentially large historical volumes, not to call out to
any model.
"""

import json
import logging
import re
import time
from collections import defaultdict
from math import ceil
from typing import Any, Dict, List, Tuple

from storage.db_config import execute_chat_query, execute_robot_query

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86400.0
# SQLite-family connections cap how many placeholders one statement can
# have; delete in bounded-size chunks well under typical limits (~999).
_DELETE_CHUNK_SIZE = 500


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _month_bucket_of(ts: float) -> str:
    """Format a unix timestamp as its "YYYY-MM" month bucket (UTC)."""
    return time.strftime("%Y-%m", time.gmtime(ts))


def _safe_json_loads(text: Any) -> Any:
    """Parse `text` as JSON, returning the raw value unchanged (never
    raising) if it isn't valid JSON or is empty/None."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _delete_by_ids(table: str, ids: List[int]) -> int:
    """Delete rows from `table` (any of robot_normal_logs/
    robot_error_logs/chat_normal_logs/chat_error_logs - all use `id` as
    their primary key) whose id is in `ids`, in bounded-size chunks.

    Returns how many ids were part of a chunk that executed
    successfully (an approximation of rows actually deleted, since a
    failed chunk is logged and skipped rather than raised).
    """
    if not ids:
        return 0
    execute = execute_chat_query if table.startswith("chat_") else execute_robot_query
    deleted = 0
    for i in range(0, len(ids), _DELETE_CHUNK_SIZE):
        chunk = ids[i:i + _DELETE_CHUNK_SIZE]
        placeholders = ",".join("?" * len(chunk))
        cur = execute(f"DELETE FROM {table} WHERE id IN ({placeholders})", tuple(chunk))
        if cur is not None:
            deleted += len(chunk)
        else:
            logger.error(f"Layer 3: failed to delete a chunk of {len(chunk)} row(s) from {table}")
    return deleted


# ---------------------------------------------------------------------------
# Robot side: run_robot_monthly_cleanup
# ---------------------------------------------------------------------------

def _distinct_old_robot_ids(cutoff_ts: float) -> List[str]:
    """All robot_ids with at least one robot_normal_logs or
    robot_error_logs row older than cutoff_ts."""
    ids = set()
    cur1 = execute_robot_query(
        "SELECT DISTINCT robot_id FROM robot_normal_logs WHERE created_at < ?", (cutoff_ts,)
    )
    if cur1 is not None:
        ids.update(row[0] for row in cur1.fetchall())
    cur2 = execute_robot_query(
        "SELECT DISTINCT robot_id FROM robot_error_logs WHERE timestamp < ?", (cutoff_ts,)
    )
    if cur2 is not None:
        ids.update(row[0] for row in cur2.fetchall())
    return sorted(ids)


def _cleanup_robot_normal_logs(robot_id: str, cutoff_ts: float, keep_top_ratio: float) -> Tuple[int, int]:
    """For one robot_id: keep the top `keep_top_ratio` of old
    robot_normal_logs rows PER MONTH (grouped by each row's own
    created_at) by significant_events_count (the busiest/most eventful
    windows), write those as robot_permanent_memory rows, then delete
    every old row regardless of whether it was kept. Returns (rows
    written to permanent memory, rows deleted from robot_normal_logs)."""
    cur = execute_robot_query(
        "SELECT id, window_start_ts, window_end_ts, summary_json, significant_events_count, "
        "created_at FROM robot_normal_logs WHERE robot_id = ? AND created_at < ? ORDER BY created_at",
        (robot_id, cutoff_ts),
    )
    rows = cur.fetchall() if cur is not None else []
    if not rows:
        return 0, 0

    by_month: Dict[str, list] = defaultdict(list)
    for row in rows:
        by_month[_month_bucket_of(row[5])].append(row)

    written = 0
    for month_bucket, month_rows in by_month.items():
        keep_n = max(1, ceil(len(month_rows) * keep_top_ratio))
        top_rows = sorted(month_rows, key=lambda r: r[4], reverse=True)[:keep_n]
        for row in top_rows:
            golden_summary = {
                "window_start_ts": row[1],
                "window_end_ts": row[2],
                "summary": _safe_json_loads(row[3]),
                "significant_events_count": row[4],
            }
            cur2 = execute_robot_query(
                "INSERT INTO robot_permanent_memory (robot_id, month_bucket, pattern_type, "
                "golden_summary_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (robot_id, month_bucket, "eventful_window", json.dumps(golden_summary), time.time()),
            )
            if cur2 is not None:
                written += 1

    deleted = _delete_by_ids("robot_normal_logs", [row[0] for row in rows])
    return written, deleted


def _cleanup_robot_error_logs(robot_id: str, cutoff_ts: float, keep_top_ratio: float) -> Tuple[int, int]:
    """For one robot_id: within each month bucket, group old
    robot_error_logs rows by error_type and keep the top
    `keep_top_ratio` of DISTINCT error_type PATTERNS by how often they
    recur (a type seen 40 times is a real recurring issue worth
    remembering; a one-off rare type is not), writing one
    robot_permanent_memory row per kept pattern, then delete every old
    error row regardless of whether its pattern was kept. Returns (rows
    written to permanent memory, rows deleted from robot_error_logs)."""
    cur = execute_robot_query(
        "SELECT id, timestamp, error_type, sensor_snapshot_json, action_attempted_json, "
        "error_message, severity FROM robot_error_logs WHERE robot_id = ? AND timestamp < ? "
        "ORDER BY timestamp",
        (robot_id, cutoff_ts),
    )
    rows = cur.fetchall() if cur is not None else []
    if not rows:
        return 0, 0

    by_month: Dict[str, list] = defaultdict(list)
    for row in rows:
        by_month[_month_bucket_of(row[1])].append(row)

    written = 0
    for month_bucket, month_rows in by_month.items():
        by_type: Dict[str, list] = defaultdict(list)
        for row in month_rows:
            by_type[row[2] or "UNKNOWN"].append(row)

        keep_n = max(1, ceil(len(by_type) * keep_top_ratio))
        top_types = sorted(by_type.items(), key=lambda kv: len(kv[1]), reverse=True)[:keep_n]

        for error_type, type_rows in top_types:
            most_recent = type_rows[-1]
            golden_summary = {
                "occurrences": len(type_rows),
                "example_error_message": most_recent[5],
                "example_sensor_snapshot": _safe_json_loads(most_recent[3]),
                "example_action_attempted": _safe_json_loads(most_recent[4]),
                "severities_seen": sorted({r[6] for r in type_rows if r[6]}),
            }
            cur2 = execute_robot_query(
                "INSERT INTO robot_permanent_memory (robot_id, month_bucket, pattern_type, "
                "golden_summary_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (robot_id, month_bucket, f"recurring_error:{error_type}", json.dumps(golden_summary), time.time()),
            )
            if cur2 is not None:
                written += 1

    deleted = _delete_by_ids("robot_error_logs", [row[0] for row in rows])
    return written, deleted


def run_robot_monthly_cleanup(retention_days: int = 30, keep_top_ratio: float = 0.10) -> dict:
    """For every robot_id with robot_normal_logs/robot_error_logs older
    than `retention_days`, extract the most valuable ~keep_top_ratio
    (default 10%) of that old data per (robot_id, month) into
    robot_permanent_memory, then DELETE all of the old raw rows - so the
    Robot Fleet DB's log tables never grow without bound, while the most
    useful long-term patterns are kept forever.

    "Most valuable" means:
      - robot_normal_logs: the rows with the highest
        significant_events_count (the busiest/most eventful windows).
      - robot_error_logs: the most frequently-recurring error_type
        patterns, not just the most recent individual errors.

    Never raises: each robot_id is processed independently in its own
    try/except, so one robot's bad/unexpected data can't abort cleanup
    for the rest of the fleet.

    Returns:
        {
            "cutoff_ts": float, "robots_processed": int,
            "golden_rows_written": int, "normal_rows_deleted": int,
            "error_rows_deleted": int, "errors": [robot_id, ...],
        }
    """
    cutoff_ts = time.time() - (retention_days * _SECONDS_PER_DAY)
    robot_ids = _distinct_old_robot_ids(cutoff_ts)

    golden_rows_written = 0
    normal_rows_deleted = 0
    error_rows_deleted = 0
    errored_robots: List[str] = []

    for robot_id in robot_ids:
        try:
            n_written, n_deleted = _cleanup_robot_normal_logs(robot_id, cutoff_ts, keep_top_ratio)
            golden_rows_written += n_written
            normal_rows_deleted += n_deleted

            e_written, e_deleted = _cleanup_robot_error_logs(robot_id, cutoff_ts, keep_top_ratio)
            golden_rows_written += e_written
            error_rows_deleted += e_deleted
        except Exception as e:
            logger.error(f"Layer 3: robot monthly cleanup failed for robot_id={robot_id}: {e}")
            errored_robots.append(robot_id)

    return {
        "cutoff_ts": cutoff_ts,
        "robots_processed": len(robot_ids),
        "golden_rows_written": golden_rows_written,
        "normal_rows_deleted": normal_rows_deleted,
        "error_rows_deleted": error_rows_deleted,
        "errors": errored_robots,
    }


# ---------------------------------------------------------------------------
# Chat side: run_chat_monthly_cleanup
# ---------------------------------------------------------------------------

_NAME_PATTERN = re.compile(r"\bmy name is\b|\bmera naam\b|\bi'?m called\b", re.IGNORECASE)
_PREFERENCE_PATTERN = re.compile(r"\bi (?:like|love|prefer|hate|dislike)\b|\bmujhe pasand\b", re.IGNORECASE)
_GOAL_PATTERN = re.compile(r"\bmy goal\b|\bi want to\b|\bi'?m trying to\b|\bmera lakshy\b", re.IGNORECASE)


def _score_chat_message(role: str, content: str) -> float:
    """Small heuristic (no ML/embedding call - Layer 3 must run cheaply
    and offline over potentially huge historical volumes) scoring how
    likely a chat_normal_logs row is to carry a durable fact worth
    remembering forever, versus routine back-and-forth chit-chat:
      - assistant turns score lower by default (a golden "user fact" is
        almost always something the USER said, not Saarthi's reply)
      - a message naming the user, or stating a preference or a goal,
        scores highest
      - a substantive question (contains "?" and more than 4 words)
        scores moderately
      - short/empty content scores near zero

    Returns a float roughly in [0.0, 1.0]. Never raises.
    """
    text = (content or "").strip()
    if not text:
        return 0.0

    score = 0.05 if role == "assistant" else 0.15

    if _NAME_PATTERN.search(text):
        score += 0.5
    if _PREFERENCE_PATTERN.search(text):
        score += 0.4
    if _GOAL_PATTERN.search(text):
        score += 0.4
    if "?" in text and len(text.split()) > 4:
        score += 0.2

    score += min(len(text) / 500.0, 0.1)  # small bonus for substantive length

    return round(min(score, 1.0), 3)


def _distinct_old_chat_user_ids(cutoff_ts: float) -> List[str]:
    """All user_ids with at least one chat_normal_logs row older than
    cutoff_ts."""
    cur = execute_chat_query(
        "SELECT DISTINCT user_id FROM chat_normal_logs WHERE created_at < ?", (cutoff_ts,)
    )
    if cur is None:
        return []
    return sorted(row[0] for row in cur.fetchall())


def _cleanup_chat_normal_logs_for_user(user_id: str, cutoff_ts: float, keep_top_ratio: float) -> Tuple[int, int]:
    """For one user_id: score every old chat_normal_logs row via
    _score_chat_message, keep the top `keep_top_ratio` as golden facts in
    chat_permanent_memory, then delete every old row regardless of
    whether it was kept. Returns (rows written to permanent memory, rows
    deleted from chat_normal_logs)."""
    cur = execute_chat_query(
        "SELECT id, session_id, role, content, intent, created_at FROM chat_normal_logs "
        "WHERE user_id = ? AND created_at < ? ORDER BY created_at",
        (user_id, cutoff_ts),
    )
    rows = cur.fetchall() if cur is not None else []
    if not rows:
        return 0, 0

    scored = [(row, _score_chat_message(row[2], row[3])) for row in rows]
    keep_n = max(1, ceil(len(scored) * keep_top_ratio))
    top = sorted(scored, key=lambda pair: pair[1], reverse=True)[:keep_n]

    written = 0
    for row, score in top:
        row_id, session_id, role, content, intent, created_at = row
        cur2 = execute_chat_query(
            "INSERT INTO chat_permanent_memory (user_id, session_id, fact_category, "
            "golden_fact, importance_score, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, session_id, intent or "general", content, score, time.time()),
        )
        if cur2 is not None:
            written += 1

    deleted = _delete_by_ids("chat_normal_logs", [row[0] for row in rows])
    return written, deleted


def _cleanup_all_old_chat_error_logs(cutoff_ts: float) -> int:
    """Delete every chat_error_logs row older than cutoff_ts, across all
    users in one pass - no golden-fact extraction applies to error logs
    (they're operational noise, not durable user facts), so this is a
    simple bulk delete rather than a per-user pass. This also correctly
    cleans up error rows whose user_id is null/unknown, which a
    per-user-from-chat_normal_logs pass would otherwise never reach."""
    cur = execute_chat_query("SELECT id FROM chat_error_logs WHERE created_at < ?", (cutoff_ts,))
    if cur is None:
        return 0
    ids = [row[0] for row in cur.fetchall()]
    return _delete_by_ids("chat_error_logs", ids)


def run_chat_monthly_cleanup(retention_days: int = 30, keep_top_ratio: float = 0.10) -> dict:
    """For every user_id with chat_normal_logs older than
    `retention_days`, extract the most valuable ~keep_top_ratio (default
    10%) of that old content into chat_permanent_memory (see
    _score_chat_message for how "most valuable" is decided), then DELETE
    all of the old chat_normal_logs/chat_error_logs rows.

    Never raises: each user_id is processed independently in its own
    try/except; the bulk chat_error_logs cleanup runs separately and its
    own failure doesn't affect the per-user results already collected.

    Returns:
        {
            "cutoff_ts": float, "users_processed": int,
            "golden_facts_written": int, "normal_rows_deleted": int,
            "error_rows_deleted": int, "errors": [user_id, ...],
        }
    """
    cutoff_ts = time.time() - (retention_days * _SECONDS_PER_DAY)
    user_ids = _distinct_old_chat_user_ids(cutoff_ts)

    golden_written = 0
    normal_deleted = 0
    errored_users: List[str] = []

    for user_id in user_ids:
        try:
            written, deleted = _cleanup_chat_normal_logs_for_user(user_id, cutoff_ts, keep_top_ratio)
            golden_written += written
            normal_deleted += deleted
        except Exception as e:
            logger.error(f"Layer 3: chat monthly cleanup failed for user_id={user_id}: {e}")
            errored_users.append(user_id)

    try:
        error_deleted = _cleanup_all_old_chat_error_logs(cutoff_ts)
    except Exception as e:
        logger.error(f"Layer 3: chat_error_logs cleanup failed: {e}")
        error_deleted = 0

    return {
        "cutoff_ts": cutoff_ts,
        "users_processed": len(user_ids),
        "golden_facts_written": golden_written,
        "normal_rows_deleted": normal_deleted,
        "error_rows_deleted": error_deleted,
        "errors": errored_users,
    }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_full_layer3_cleanup(retention_days: int = 30) -> dict:
    """Run both run_robot_monthly_cleanup and run_chat_monthly_cleanup
    with the same retention window - meant to be called from a
    background monthly timer, or triggered on demand from an admin API
    endpoint. Each half runs independently: a robot-side failure doesn't
    prevent the chat-side cleanup from running, or vice versa."""
    started_at = time.time()
    robot_result = run_robot_monthly_cleanup(retention_days=retention_days)
    chat_result = run_chat_monthly_cleanup(retention_days=retention_days)
    return {
        "started_at": started_at,
        "finished_at": time.time(),
        "robot": robot_result,
        "chat": chat_result,
    }
