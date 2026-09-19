"""
game_manager.py

Bridge layer between the WebSocket server and the modular game engines
living under `games/`. This file owns per-session game state so that
multiple ESP32 clients / users can be mid-game simultaneously without
stepping on each other.

Nothing in this file touches Groq, Supabase, or any other external
service — it is pure routing/state-machine logic, and it is completely
deterministic aside from delegating gameplay to the individual game
engine classes (e.g. FastMathGame), which are themselves deterministic.

Usage from server.py (not modified here, shown for context only):

    from game_manager import handle_game_message

    game_response = handle_game_message(session_id, user_message)
    if game_response is not None:
        # send game_response straight to the ESP32
        ...
    else:
        # not game-related -> fall through to the normal AI/chat pipeline
        ...
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import logging
import re
import threading

from games.fast_math import FastMathGame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session modes
# ---------------------------------------------------------------------------

class GameMode:
    """String constants for the per-session state machine."""
    CHATTING = "CHATTING"
    GAME_MENU = "GAME_MENU"
    PLAYING_FAST_MATH = "PLAYING_FAST_MATH"
    GAME_OVER_MENU = "GAME_OVER_MENU"


# ---------------------------------------------------------------------------
# Keyword lists (Hinglish + English)
# ---------------------------------------------------------------------------
# Matching is case-insensitive substring matching (see _contains_keyword),
# so shorter phrases like "game" or "khelo" deliberately double as catch-alls
# for longer ones ("game lagao", "chalo khelo") without needing every
# permutation spelled out.

# 58 unique keywords/phrases that trigger entry into the game menu.
_GAME_TRIGGER_KEYWORDS = (
    # --- English ---
    "play a game",
    "let's play",
    "lets play",
    "start game",
    "start a game",
    "new game",
    "wanna play",
    "want to play",
    "can we play",
    "i want to play",
    "play game",
    "begin game",
    "launch game",
    "fire up a game",
    "i'm bored, play a game",
    "i am bored, play a game",
    "quiz me",
    "quiz time",
    "math game",
    "maths game",
    "fast math",
    "speed math",
    "math quiz",
    "test me",
    "challenge me",
    "let's do a quiz",
    "lets do a quiz",
    "play something",
    "any games",
    "got any games",
    "wanna play a game",
    "i'm bored",
    "im bored",
    "bored, let's play",
    "start playing",
    "load a game",
    "activate game mode",
    "game mode on",
    "enter game mode",
    "play mode",
    "time to play",
    "let's have some fun",
    "entertain me",
    "game",
    # --- Hinglish ---
    "khelna hai",
    "chalo khelte hai",
    "chalo khelte hain",
    "game khelna hai",
    "game lagao",
    "game start karo",
    "khel shuru karo",
    "maths game khelna hai",
    "bore ho raha hu game start karo",
    "bore ho raha hoon",
    "kuch khelte hai",
    "ek game khelo",
    "game chalu karo",
    "mujhe khelna hai",
    "chal game khel",
    "thoda khel lete hai",
    "dimag ka khel",
    "dimaag ka game",
    "ganit ka khel",
)

# 24 unique keywords/phrases that stop/exit the game from any game state.
_EXIT_KEYWORDS = (
    # --- English ---
    "exit",
    "quit",
    "stop game",
    "stop playing",
    "end game",
    "leave game",
    "cancel game",
    "no more game",
    "i'm done",
    "im done",
    "go back",
    "back to chat",
    "stop it",
    "enough",
    "that's enough",
    "finish game",
    "close game",
    # --- Hinglish ---
    "band karo",
    "bas karo",
    "khatam karo",
    "ruk jao",
    "bahar nikal",
    "chodo",
    "nahi khelna",
)

# Keywords that select "Fast Math" from the game menu.
_FAST_MATH_KEYWORDS = ("1", "fast math", "fastmath", "math")


def _compile_keyword_patterns(keywords: tuple) -> "list[re.Pattern]":
    """
    FIX (game trigger false positives / input normalization): the original
    matching was a raw case-insensitive substring check, so very short
    keywords like the single-word "game" or the numeric selector "1" would
    fire on completely unrelated text (e.g. "1" matching inside "at 1pm" or
    "room 101"; "game" matching inside "endgame"). Compiling each keyword
    to a word-boundary regex keeps every existing keyword/phrase exactly as
    written (nothing added or removed) while requiring it to appear as a
    standalone word/phrase rather than an arbitrary substring.
    """
    return [re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE) for keyword in keywords]


_GAME_TRIGGER_PATTERNS = _compile_keyword_patterns(_GAME_TRIGGER_KEYWORDS)
_EXIT_PATTERNS = _compile_keyword_patterns(_EXIT_KEYWORDS)
_FAST_MATH_PATTERNS = _compile_keyword_patterns(_FAST_MATH_KEYWORDS)

# Precompiled patterns for the GAME_OVER_MENU "again"/"menu" commands, for
# the same word-boundary reasoning as above (e.g. so "menu" doesn't fire on
# some future keyword that merely contains "menu" as a substring).
_AGAIN_PATTERN = re.compile(r"\bagain\b", re.IGNORECASE)
_MENU_PATTERN = re.compile(r"\bmenu\b", re.IGNORECASE)


@dataclass
class SessionState:
    """Per-user/session game state. One of these lives per session_id."""
    mode: str = GameMode.CHATTING
    game: Optional[object] = None  # holds the active game engine instance


# The single source of truth for all active sessions.
# Keyed by session_id -> SessionState. NEVER share state across sessions.
user_states: Dict[str, SessionState] = {}

# FIX (session isolation / multi-session safety): server.py dispatches
# handle_game_message via asyncio.to_thread, so calls for two DIFFERENT
# session_ids can genuinely execute concurrently on different worker
# threads. Plain dict get-or-create ("if key not in d: d[key] = ...") is
# not atomic, so two concurrent first-messages for the same session_id
# could each construct a fresh SessionState and one write could be lost.
# This lock only guards the dict lookup/creation/reset itself (a few
# in-memory operations) - it is never held while running game logic, so
# different sessions never block on each other's turns.
_user_states_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_session_state(session_id: str) -> SessionState:
    """Fetch (or lazily create) the SessionState for a given session_id."""
    with _user_states_lock:
        if session_id not in user_states:
            user_states[session_id] = SessionState()
        return user_states[session_id]


def _reset_session(session_id: str) -> None:
    """Wipe a session back to a fresh CHATTING state."""
    with _user_states_lock:
        user_states[session_id] = SessionState()


def _normalize_message(user_message) -> str:
    """
    FIX (input normalization): centralizes turning whatever came in off the
    wire into a clean string once, instead of the ad-hoc `if None: ""`
    check that used to live inline. Also defensively coerces any
    unexpected non-string type to str rather than raising, since a
    malformed/odd payload should degrade to "no keyword match", never to
    an exception.
    """
    if user_message is None:
        return ""
    if not isinstance(user_message, str):
        try:
            user_message = str(user_message)
        except Exception:
            return ""
    return user_message.strip()


def _contains_keyword(message: str, patterns: "list[re.Pattern]") -> bool:
    """Word-boundary, case-insensitive match against precompiled patterns."""
    return any(pattern.search(message) for pattern in patterns)


def _game_menu_response() -> dict:
    """The standard 'which game do you want to play' prompt."""
    return {
        "type": "response",
        "emotion": "EXCITED",
        "text": "Let's play! Choose a game: 1. Fast Math",
    }


def _start_fast_math(state: SessionState) -> dict:
    """
    Instantiate a fresh FastMathGame for this session and start it.

    FIX (safe handling of missing/broken game instances): construction and
    start() are both wrapped so a failure here (however unlikely, since
    FastMathGame internals aren't touched) can never bubble out of
    game_manager as an uncaught exception. On failure the session is put
    back in a known-good CHATTING state and a friendly fallback is
    returned, instead of leaving state.mode pointed at a game with no
    (or a broken) instance behind it.
    """
    try:
        game = FastMathGame()
        response = game.start()
    except Exception as e:
        logger.warning(f"Fast Math failed to start: {e}")
        state.mode = GameMode.CHATTING
        state.game = None
        return {
            "type": "response",
            "emotion": "SAD",
            "text": "Couldn't start Fast Math right now — let's just chat instead.",
        }

    state.game = game
    state.mode = GameMode.PLAYING_FAST_MATH
    return response


def _game_over_menu_response() -> dict:
    return {
        "type": "response",
        "emotion": "NORMAL",
        "text": "Game over! Say 'again' to replay, 'menu' for game choices, or 'exit' to stop playing.",
    }


def _exit_to_chatting(state: SessionState) -> dict:
    """Instantly reset a session to normal CHATTING mode, clearing any game."""
    state.mode = GameMode.CHATTING
    state.game = None
    return {
        "type": "response",
        "emotion": "NORMAL",
        "text": "Okay, back to chatting! Let me know if you want to play again later.",
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle_game_message(
    session_id: str, user_message: str
) -> Optional[Dict[str, Union[str, int, float, None]]]:
    """
    Route an incoming user message through the per-session game state
    machine.

    Returns:
        - A response dict if this message was handled by the game system.
          Plain chat-style replies (menus, exit confirmations, timeouts)
          use {"type": "response", "emotion": ..., "text": ...}.
          Active Fast Math rounds use the structured
          {"type": "game_fast_math", "equation": ..., "timer": ...} (or
          "game_fast_math_over") payloads defined in games/fast_math.py.
        - None if the session is in CHATTING mode and the message was
          NOT a game-related trigger -- this tells server.py to route
          the message to the normal AI/chat pipeline instead.
    """
    user_message = _normalize_message(user_message)

    state = _get_session_state(session_id)

    # ------------------------------------------------------------------
    # Global exit check: an exit keyword mid-game (or anywhere in the
    # game flow) instantly resets state and drops back to chat, no
    # matter which mode we're in.
    # ------------------------------------------------------------------
    if state.mode != GameMode.CHATTING and _contains_keyword(user_message, _EXIT_PATTERNS):
        return _exit_to_chatting(state)

    # ------------------------------------------------------------------
    # 1. CHATTING: only intercept messages that look like a game request.
    # ------------------------------------------------------------------
    if state.mode == GameMode.CHATTING:
        if _contains_keyword(user_message, _GAME_TRIGGER_PATTERNS):
            state.mode = GameMode.GAME_MENU
            return _game_menu_response()
        # Not game-related -> let server.py handle this normally.
        return None

    # ------------------------------------------------------------------
    # 2. GAME_MENU: waiting for the user to pick which game to play.
    # ------------------------------------------------------------------
    if state.mode == GameMode.GAME_MENU:
        if _contains_keyword(user_message, _FAST_MATH_PATTERNS):
            return _start_fast_math(state)

        # Unrecognized selection -> re-prompt instead of silently failing.
        return {
            "type": "response",
            "emotion": "THINKING_E",
            "text": "I didn't catch that. Type '1' or 'Fast Math' to start, "
            "or 'exit' to stop.",
        }

    # ------------------------------------------------------------------
    # 3. PLAYING_FAST_MATH: forward the input to the active game engine.
    # ------------------------------------------------------------------
    if state.mode == GameMode.PLAYING_FAST_MATH:
        game = state.game

        if game is None:
            # Defensive recovery: state says we're playing but there's no
            # game instance (e.g. server restart). Bounce back to the menu.
            state.mode = GameMode.GAME_MENU
            return _game_menu_response()

        # FIX (prevent accidental routing into normal AI chat / safe
        # handling of a broken game instance): process_input is the one
        # call in this whole module that delegates to engine code outside
        # our control. If it ever raised, the old code would let the
        # exception escape handle_game_message entirely - which for the
        # server means falling into its generic error handler and losing
        # the "this was a game answer" context (and never returning None
        # here, so no risk of a game answer like "42" leaking to the AI
        # pipeline either way, but the session would be stuck in
        # PLAYING_FAST_MATH with an indeterminate/undefined game_active
        # state). This now can never happen: any failure is treated as a
        # stale/broken game and recovered back to the menu with a normal
        # response dict, same shape as every other branch here.
        try:
            response = game.process_input(user_message)
            game_finished = not game.game_active
        except Exception as e:
            logger.warning(f"Fast Math process_input failed for session_id={session_id}: {e}")
            state.mode = GameMode.GAME_MENU
            state.game = None
            return {
                "type": "response",
                "emotion": "SAD",
                "text": "That round hit a snag — let's pick a game again. "
                "Choose a game: 1. Fast Math",
            }

        # CRITICAL: game_active is the single source of truth for whether
        # the round-based game has finished.
        if game_finished:
            state.mode = GameMode.GAME_OVER_MENU
            # FIX (cleanup after game completion): the finished instance is
            # no longer needed - "again" always builds a brand-new
            # FastMathGame anyway, and "menu" already cleared this. Doing
            # it here too means a completed game is never left dangling in
            # state.game while we're sitting in GAME_OVER_MENU.
            state.game = None

        return response

    # ------------------------------------------------------------------
    # 4. GAME_OVER_MENU: offer to replay, go to menu, or exit.
    # ------------------------------------------------------------------
    if state.mode == GameMode.GAME_OVER_MENU:
        lowered = user_message.lower()

        if _AGAIN_PATTERN.search(lowered):
            return _start_fast_math(state)

        if _MENU_PATTERN.search(lowered):
            state.mode = GameMode.GAME_MENU
            state.game = None
            return _game_menu_response()

        # Unrecognized -> re-show the game-over options.
        # (Exit is already handled by the global exit check above.)
        return _game_over_menu_response()

    # ------------------------------------------------------------------
    # Fallback: unknown mode somehow got set -> reset the session safely.
    # ------------------------------------------------------------------
    _reset_session(session_id)
    return None
