"""storage/db_config.py

Multi-tenant, 3-layer storage configuration for SAARTHI_SERVER, built to
scale to thousands of robots (robot_id) and thousands of web/app users
(user_id / session_id).

Three independent stores, each individually optional/degradable:

  1. Robot Fleet DB        - Turso/libSQL; falls back to local SQLite at
                              ./local_data/saarthi_robots.db.
  2. User Chat SQL DB      - Turso/libSQL; falls back to local SQLite at
                              ./local_data/saarthi_chats.db.
  3. User Chat Document DB - MongoDB Atlas (optional; lazy connection,
                              degrades to None if pymongo isn't installed
                              or MONGODB_URI isn't set).

Zero-crash design: nothing in this module raises on import, on a missing
package, or on missing/bad credentials. Every connection attempt that
fails is caught, logged, and degrades to the next safest option (Turso ->
local SQLite; MongoDB -> disabled). Call init_all_databases() once at
app startup to create every table/index this module owns.

Environment variables:
    TURSO_ROBOT_DB_URL, TURSO_ROBOT_AUTH_TOKEN  - Robot Fleet DB (Turso)
    TURSO_CHAT_DB_URL,  TURSO_CHAT_AUTH_TOKEN   - User Chat SQL DB (Turso)
    MONGODB_URI                                 - User Chat Document DB
    MONGODB_DB_NAME                             - default "saarthi_chat_db"
    SAARTHI_LOCAL_DATA_DIR                       - default "./local_data"
"""

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    # Turso/libSQL client. Optional: if it isn't installed, both SQL
    # stores below simply run on local SQLite instead.
    import libsql_experimental as libsql
except ImportError:  # pragma: no cover - optional dependency
    libsql = None

try:
    import pymongo
except ImportError:  # pragma: no cover - optional dependency
    pymongo = None


_LOCAL_DATA_DIR = Path(os.environ.get("SAARTHI_LOCAL_DATA_DIR", "./local_data"))
_ROBOT_DB_LOCAL_PATH = _LOCAL_DATA_DIR / "saarthi_robots.db"
_CHAT_DB_LOCAL_PATH = _LOCAL_DATA_DIR / "saarthi_chats.db"
_MONGODB_DEFAULT_DB_NAME = "saarthi_chat_db"


# ---------------------------------------------------------------------------
# Connection resolution (Turso/libSQL, with automatic local SQLite fallback)
# ---------------------------------------------------------------------------

def _connect_local_sqlite(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) a local SQLite file with WAL mode and
    check_same_thread=False, so it can be shared across threads under our
    own locking (see execute_robot_query/execute_chat_query below)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.commit()
    return conn


def _connect_turso(url: str, token: Optional[str], label: str):
    """Attempt a Turso/libSQL remote connection.

    Returns None (never raises) if libsql_experimental isn't installed,
    or if connecting fails for any reason - callers must fall back to
    local SQLite in that case, per this module's zero-crash design.

    NOTE: libsql_experimental is an actively-evolving package. The exact
    call here - connect(url, auth_token=token), returning a
    sqlite3-compatible connection with .execute()/.executemany()/
    .commit() - matches its documented usage as of this writing but has
    not been verified against whatever version you have installed. If
    it doesn't match, this simply fails closed and falls back to local
    SQLite (by design) rather than crashing the server; check your
    installed libsql_experimental version's own docs if the Turso path
    itself isn't connecting and you want to fix that specifically.
    """
    if libsql is None:
        logger.warning(f"{label}: libsql_experimental not installed - using local SQLite instead.")
        return None
    try:
        return libsql.connect(url, auth_token=token)
    except Exception as e:
        logger.warning(f"{label}: Turso/libSQL connection failed ({e}) - using local SQLite instead.")
        return None


def _resolve_db_connection(url_env: str, token_env: str, local_path: Path, label: str):
    """Resolve one logical database's connection: try Turso/libSQL first
    if TURSO_*_DB_URL is set, else (or on any failure) fall back to a
    local SQLite file. Returns (connection, is_remote)."""
    url = os.environ.get(url_env)
    token = os.environ.get(token_env)
    if url:
        conn = _connect_turso(url, token, label)
        if conn is not None:
            logger.info(f"{label}: connected to Turso/libSQL ({url_env}).")
            return conn, True
    conn = _connect_local_sqlite(local_path)
    logger.info(f"{label}: using local SQLite fallback at {local_path}.")
    return conn, False


_robot_conn: Optional[Any] = None
_robot_is_remote: bool = False
_robot_conn_lock = threading.Lock()

_chat_conn: Optional[Any] = None
_chat_is_remote: bool = False
_chat_conn_lock = threading.Lock()

_mongo_client: Optional[Any] = None
_mongo_db: Optional[Any] = None
_mongo_lock = threading.Lock()


def _get_robot_connection():
    """Lazily resolve (and cache) the Robot Fleet DB connection."""
    global _robot_conn, _robot_is_remote
    if _robot_conn is not None:
        return _robot_conn
    with _robot_conn_lock:
        if _robot_conn is None:
            _robot_conn, _robot_is_remote = _resolve_db_connection(
                "TURSO_ROBOT_DB_URL", "TURSO_ROBOT_AUTH_TOKEN",
                _ROBOT_DB_LOCAL_PATH, "Robot Fleet DB",
            )
        return _robot_conn


def _get_chat_connection():
    """Lazily resolve (and cache) the User Chat SQL DB connection."""
    global _chat_conn, _chat_is_remote
    if _chat_conn is not None:
        return _chat_conn
    with _chat_conn_lock:
        if _chat_conn is None:
            _chat_conn, _chat_is_remote = _resolve_db_connection(
                "TURSO_CHAT_DB_URL", "TURSO_CHAT_AUTH_TOKEN",
                _CHAT_DB_LOCAL_PATH, "User Chat SQL DB",
            )
        return _chat_conn


def _get_mongo_db():
    """Lazily resolve (and cache) the MongoDB Atlas document database.
    Returns None (never raises) if pymongo isn't installed, MONGODB_URI
    isn't set, or the connection/ping fails."""
    global _mongo_client, _mongo_db
    if _mongo_db is not None:
        return _mongo_db
    with _mongo_lock:
        if _mongo_db is not None:
            return _mongo_db
        if pymongo is None:
            logger.info("User Chat Document DB: pymongo not installed - MongoDB disabled.")
            return None
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            logger.info("User Chat Document DB: MONGODB_URI not set - MongoDB disabled.")
            return None
        db_name = os.environ.get("MONGODB_DB_NAME", _MONGODB_DEFAULT_DB_NAME)
        try:
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")  # fail fast here rather than on first real query
            _mongo_client = client
            _mongo_db = client[db_name]
            logger.info(f"User Chat Document DB: connected to MongoDB Atlas database '{db_name}'.")
            return _mongo_db
        except Exception as e:
            logger.warning(f"User Chat Document DB: MongoDB connection failed ({e}) - disabled.")
            return None


# ---------------------------------------------------------------------------
# Thread-safe, exception-safe query helpers
# ---------------------------------------------------------------------------

def execute_robot_query(sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Cursor]:
    """Execute one statement (DDL/INSERT/UPDATE/SELECT) against the Robot
    Fleet DB and commit. Thread-safe (serialized via a single lock for
    this connection - simple and correct for a SQLite-family connection,
    local or Turso). Returns the cursor (so SELECT rows can still be
    fetched from it) on success, or None on any failure - the exception
    is logged, never raised, so one bad/failed write degrades instead of
    crashing the caller.
    """
    conn = _get_robot_connection()
    if conn is None:
        logger.error("Robot Fleet DB: no connection available - query skipped.")
        return None
    with _robot_conn_lock:
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur
        except Exception as e:
            logger.error(f"Robot Fleet DB: query failed: {e}")
            return None


def execute_robot_many(sql: str, seq_of_params: Sequence[Sequence[Any]]) -> bool:
    """Execute `sql` once per parameter tuple in seq_of_params (bulk
    insert/update) against the Robot Fleet DB and commit. Returns True on
    success, False (logged, never raised) on any failure."""
    conn = _get_robot_connection()
    if conn is None:
        logger.error("Robot Fleet DB: no connection available - batch write skipped.")
        return False
    with _robot_conn_lock:
        try:
            conn.executemany(sql, seq_of_params)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Robot Fleet DB: batch write failed: {e}")
            return False


def execute_chat_query(sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Cursor]:
    """Same contract as execute_robot_query, against the User Chat SQL DB."""
    conn = _get_chat_connection()
    if conn is None:
        logger.error("User Chat SQL DB: no connection available - query skipped.")
        return None
    with _chat_conn_lock:
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur
        except Exception as e:
            logger.error(f"User Chat SQL DB: query failed: {e}")
            return None


def get_mongo_collection(name: str):
    """Return a pymongo Collection named `name`, connecting lazily to
    MongoDB Atlas on first use. Returns None (never raises) if pymongo
    isn't installed, MONGODB_URI isn't set, or the connection fails -
    callers must treat None as "document storage unavailable" and skip
    whatever optional document-store write/read they were about to do.
    """
    db = _get_mongo_db()
    if db is None:
        return None
    try:
        return db[name]
    except Exception as e:
        logger.warning(f"User Chat Document DB: could not get collection '{name}': {e}")
        return None


# ---------------------------------------------------------------------------
# Schema (tables + indexes), created automatically by init_all_databases()
# ---------------------------------------------------------------------------

_ROBOT_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS robot_normal_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        robot_id TEXT NOT NULL,
        window_start_ts REAL NOT NULL,
        window_end_ts REAL NOT NULL,
        summary_json TEXT NOT NULL,
        significant_events_count INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_robot_normal_logs_robot_created
        ON robot_normal_logs (robot_id, created_at);
    """,
    """
    CREATE TABLE IF NOT EXISTS robot_error_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        robot_id TEXT NOT NULL,
        timestamp REAL NOT NULL,
        error_type TEXT,
        sensor_snapshot_json TEXT,
        action_attempted_json TEXT,
        error_message TEXT,
        severity TEXT
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_robot_error_logs_robot_ts
        ON robot_error_logs (robot_id, timestamp);
    """,
    # Layer 3: the top ~10% "golden" patterns kept after each month's 90%
    # cleanup of robot_normal_logs/robot_error_logs - long-lived,
    # per-robot, per-month-bucket permanent memory.
    """
    CREATE TABLE IF NOT EXISTS robot_permanent_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        robot_id TEXT NOT NULL,
        month_bucket TEXT NOT NULL,
        pattern_type TEXT,
        golden_summary_json TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_robot_permanent_memory_robot_month
        ON robot_permanent_memory (robot_id, month_bucket);
    """,
)

_CHAT_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS chat_normal_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        channel TEXT,
        role TEXT NOT NULL,
        content TEXT,
        intent TEXT,
        created_at REAL NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_normal_logs_user_session_created
        ON chat_normal_logs (user_id, session_id, created_at);
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_error_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        session_id TEXT NOT NULL,
        channel TEXT,
        error_type TEXT,
        user_message TEXT,
        error_details TEXT,
        created_at REAL NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_error_logs_session_created
        ON chat_error_logs (session_id, created_at);
    """,
    # Layer 3: the top ~10% "golden" user facts/profile bits kept after
    # each month's 90% cleanup of chat_normal_logs/chat_error_logs.
    """
    CREATE TABLE IF NOT EXISTS chat_permanent_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        session_id TEXT,
        fact_category TEXT,
        golden_fact TEXT NOT NULL,
        importance_score REAL,
        created_at REAL NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_permanent_memory_user_session
        ON chat_permanent_memory (user_id, session_id);
    """,
)


def init_all_databases() -> dict:
    """Create every table/index this module owns, in both the Robot
    Fleet DB and the User Chat SQL DB, if they don't already exist.

    Safe to call multiple times (every statement is CREATE ... IF NOT
    EXISTS). Never raises: each statement is attempted independently, so
    one bad/unsupported statement (e.g. a SQL dialect quirk on a
    not-yet-verified Turso/libSQL version) can't block the rest of
    initialization - see execute_robot_query/execute_chat_query's own
    exception handling.

    Returns a small status dict for observability/health checks:
        {
            "robot_db": {"backend": "turso" | "sqlite", "ok": bool},
            "chat_db":  {"backend": "turso" | "sqlite", "ok": bool},
            "mongo":    {"enabled": bool},
        }
    ``ok`` is True only if every schema statement for that database
    succeeded.
    """
    robot_ok = True
    for stmt in _ROBOT_SCHEMA_STATEMENTS:
        if execute_robot_query(stmt) is None:
            robot_ok = False

    chat_ok = True
    for stmt in _CHAT_SCHEMA_STATEMENTS:
        if execute_chat_query(stmt) is None:
            chat_ok = False

    mongo_db = _get_mongo_db()

    return {
        "robot_db": {"backend": "turso" if _robot_is_remote else "sqlite", "ok": robot_ok},
        "chat_db": {"backend": "turso" if _chat_is_remote else "sqlite", "ok": chat_ok},
        "mongo": {"enabled": mongo_db is not None},
    }
