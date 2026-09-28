"""storage/robot_layer1_buffer.py

Layer 1 of SAARTHI_SERVER's 3-layer robot data structure: a high-speed,
thread-safe, per-robot rolling 15-minute buffer held entirely in RAM.

Each robot transmits hardware sensor telemetry roughly every 0.5 seconds
(~2 packets/second, ~1,800 packets per 15 minutes). This module holds
only the most recent 15 minutes of that stream per robot_id, ready for
Layer 2 filtration to consume (and eventually persist the interesting
parts of via storage/db_config.py - this module never touches that
directly).

Designed to scale to hundreds or thousands of concurrent robot_ids: the
top-level registry is guarded by its own lock (only touched briefly, to
find-or-create a robot's state), while each robot's own packets/timers
are guarded by a *per-robot* lock, so one busy or slow robot never blocks
another's telemetry.
"""

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

# 2 packets/second target cadence.
TELEMETRY_INTERVAL_SEC = 0.5
# 15-minute rolling window.
BUFFER_WINDOW_SECONDS = 900.0
# 900s / 0.5s = 1800 packets - matches BUFFER_WINDOW_SECONDS at the
# expected cadence; also used as a hard deque ceiling so a robot sending
# faster than expected still can't grow its buffer without bound.
MAX_PACKETS_PER_ROBOT = 1800


class _RobotState:
    """Internal per-robot bookkeeping: its rolling deque of packets, plus
    last_seen_ts/last_flushed_ts, guarded by its own lock."""

    __slots__ = ("packets", "lock", "last_seen_ts", "last_flushed_ts")

    def __init__(self) -> None:
        """Start a new, empty per-robot buffer. last_seen_ts is stamped
        now; last_flushed_ts starts at 0.0 (meaning "never flushed yet"),
        not at the creation time - this guarantees a robot's very first
        packet is never skipped by pop_ready_for_filtration just because
        its timestamp happened to tie with this state's own creation
        instant (real epoch timestamps are always far greater than 0.0)."""
        self.packets: Deque[dict] = deque(maxlen=MAX_PACKETS_PER_ROBOT)
        self.lock = threading.Lock()
        self.last_seen_ts = time.time()
        self.last_flushed_ts = 0.0


class RobotLayer1Buffer:
    """Thread-safe registry of per-robot rolling 15-minute telemetry
    buffers, scaling to hundreds/thousands of concurrent robot_ids."""

    def __init__(self) -> None:
        """Start an empty registry."""
        self._robots: Dict[str, _RobotState] = {}
        self._registry_lock = threading.Lock()

    def _get_or_create_state(self, robot_id: str) -> _RobotState:
        """Find (or, under the registry lock, create) `robot_id`'s state."""
        state = self._robots.get(robot_id)
        if state is not None:
            return state
        with self._registry_lock:
            state = self._robots.get(robot_id)
            if state is None:
                state = _RobotState()
                self._robots[robot_id] = state
            return state

    def record_telemetry(
        self,
        robot_id: str,
        sensor_data: dict,
        user_message: str = "",
        action_taken: Optional[dict] = None,
        error_flag: bool = False,
        error_reason: str = "",
    ) -> dict:
        """Append one timestamped telemetry packet for `robot_id`.

        Evicts any packet older than BUFFER_WINDOW_SECONDS (15 minutes)
        from the left of that robot's deque on every call, so only the
        freshest 15 minutes of data ever stay in memory regardless of how
        fast or slow telemetry actually arrives.

        Args:
            robot_id: Which robot this packet is for.
            sensor_data: Raw hardware sensor readings for this instant.
            user_message: What the user said at this instant, if anything.
            action_taken: The action (e.g. a parsed [ACTION] JSON dict)
                the robot took in response, if any.
            error_flag: Whether this instant coincided with an error.
            error_reason: Human-readable detail when error_flag is True.

        Returns:
            The packet dict that was recorded (including its "ts").
        """
        now = time.time()
        packet = {
            "ts": now,
            "sensor_data": sensor_data or {},
            "user_message": user_message or "",
            "action_taken": action_taken,
            "error_flag": bool(error_flag),
            "error_reason": error_reason or "",
        }
        state = self._get_or_create_state(robot_id)
        with state.lock:
            state.packets.append(packet)
            state.last_seen_ts = now
            cutoff = now - BUFFER_WINDOW_SECONDS
            while state.packets and state.packets[0]["ts"] < cutoff:
                state.packets.popleft()
        return packet

    def get_robot_window(self, robot_id: str, since_ts: Optional[float] = None) -> List[dict]:
        """Return a snapshot list of `robot_id`'s packets in the current
        15-minute window, optionally only those newer than `since_ts` (so
        Layer 2 filtration can process just-accumulated packets without
        reprocessing ones it already filtered).

        The 15-minute cutoff is re-applied here (not just relied on from
        record_telemetry's own eviction), so a robot that has gone quiet
        - and so hasn't triggered fresh eviction on write - never has
        stale, out-of-window packets read back out as if they were current.

        Returns [] for an unknown robot_id; never raises.
        """
        state = self._robots.get(robot_id)
        if state is None:
            return []
        window_cutoff = time.time() - BUFFER_WINDOW_SECONDS
        effective_cutoff = max(window_cutoff, since_ts) if since_ts is not None else window_cutoff
        with state.lock:
            return [p for p in state.packets if p["ts"] > effective_cutoff]

    def pop_ready_for_filtration(self, flush_interval_sec: float = 60.0) -> Dict[str, List[dict]]:
        """Return packets accumulated since each robot's last flush, for
        every robot that is due for one.

        A robot is "due" once at least `flush_interval_sec` seconds have
        passed since its last_flushed_ts. Its last_flushed_ts is then
        advanced to now, whether or not it actually had new packets (so a
        quiet robot isn't re-checked packet-by-packet on every call).

        Returns:
            {robot_id: [new_packets...]} - only for robots that were both
            due for a flush AND had at least one new packet since their
            last one. Meant to be called on a timer (e.g. every few
            seconds); each robot only does real work once per
            flush_interval_sec.
        """
        now = time.time()
        ready: Dict[str, List[dict]] = {}
        # Snapshot the registry's keys first: a robot registered by another
        # thread mid-call is simply picked up on a later call instead of
        # risking a "dict changed size during iteration" error here.
        for robot_id in list(self._robots.keys()):
            state = self._robots.get(robot_id)
            if state is None:
                continue
            with state.lock:
                if now - state.last_flushed_ts < flush_interval_sec:
                    continue
                new_packets = [p for p in state.packets if p["ts"] > state.last_flushed_ts]
                state.last_flushed_ts = now
                if new_packets:
                    ready[robot_id] = new_packets
        return ready

    def cleanup_inactive_robots(self, max_idle_seconds: float = 1800.0) -> List[str]:
        """Evict every robot whose last_seen_ts is more than
        `max_idle_seconds` ago (default 30 minutes) from the registry
        entirely, so RAM stays bounded even across thousands of
        historical robot_ids that connected once and never came back.

        Returns:
            The list of robot_ids that were evicted.
        """
        now = time.time()
        evicted: List[str] = []
        with self._registry_lock:
            for robot_id in list(self._robots.keys()):
                state = self._robots.get(robot_id)
                if state is None:
                    continue
                # last_seen_ts is only ever written while holding
                # state.lock (in record_telemetry). Reading it here without
                # that lock is fine: it's a single float assignment (an
                # atomic operation in CPython) and eviction only needs an
                # approximately-current value, not a perfectly fresh one.
                if now - state.last_seen_ts > max_idle_seconds:
                    del self._robots[robot_id]
                    evicted.append(robot_id)
        return evicted


# Module-level singleton, mirroring how config.py/database.py are shared
# elsewhere in this project. server.py's WebSocket handler and Layer 2
# filtration should both import and use this one instance rather than
# creating their own (a second instance would just be an empty, unrelated
# buffer).
robot_layer1_buffer = RobotLayer1Buffer()
