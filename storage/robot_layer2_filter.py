"""storage/robot_layer2_filter.py

Layer 2 of SAARTHI_SERVER's 3-layer robot data structure: filters each
robot's freshly-flushed batch of Layer 1 telemetry (storage/
robot_layer1_buffer.py) and persists it via storage/db_config.py.

Smart deduplication: a robot standing still sends the same near-identical
packet every 0.5 seconds. Rather than writing hundreds of duplicate rows,
"quiet" packets (no user message, no action change, no meaningful sensor
change) are folded into one compact per-batch summary row in
robot_normal_logs. Anything that looks dangerous - or was already flagged
as an error by the caller - goes straight to robot_error_logs instead,
one row per event, since those are exactly the rows worth keeping
individually.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from storage.db_config import execute_robot_many, execute_robot_query
from storage.robot_layer1_buffer import robot_layer1_buffer

logger = logging.getLogger(__name__)

# --- Hardware danger thresholds/keywords ------------------------------------

_EDGE_DANGER_KEYWORDS = ("CLIFF", "DROP", "DANGER")
_IMU_DANGER_KEYWORDS = ("TILT", "FALL")
_TOF_DANGER_KEYWORD = "COLLISION"
_BATTERY_DANGER_KEYWORD = "CRITICAL"

_TOF_DANGER_CM = 20.0
_BATTERY_DANGER_PERCENT = 15.0

# Cap on how many significant events one normal_logs summary_json embeds,
# so a genuinely eventful window still produces a bounded row.
_MAX_SIGNIFICANT_EVENTS_IN_SUMMARY = 50


def _extract_number(text: str) -> Optional[float]:
    """Pull the first number (int or float, optionally negative) out of a
    sensor-reading string like "MEASURING (58cm)" or "NORMAL (85%)".
    Returns None if no number is found."""
    import re

    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group(0)) if match else None


def _sensor_indicates_danger(sensor_data: dict) -> Tuple[bool, List[str]]:
    """Inspect one packet's sensor_data dict for any configured danger
    condition (edge_sensor CLIFF/DROP/DANGER, tof_obstacle_sensor
    COLLISION or < 20cm, battery_monitor CRITICAL or < 15%, imu_gyro
    TILT/FALL). Unknown/missing sensor keys are simply not checked.

    Returns:
        (is_dangerous, reasons) - reasons is a short list of
        "sensor=value" strings describing which sensor(s) tripped, used
        to build the stored error_message. Never raises.
    """
    sensor_data = sensor_data or {}
    reasons: List[str] = []

    edge = str(sensor_data.get("edge_sensor", "")).upper()
    if any(kw in edge for kw in _EDGE_DANGER_KEYWORDS):
        reasons.append(f"edge_sensor={sensor_data.get('edge_sensor')}")

    tof_raw = sensor_data.get("tof_obstacle_sensor", "")
    tof = str(tof_raw)
    if _TOF_DANGER_KEYWORD in tof.upper():
        reasons.append(f"tof_obstacle_sensor={tof_raw}")
    else:
        distance_cm = _extract_number(tof)
        if distance_cm is not None and distance_cm < _TOF_DANGER_CM:
            reasons.append(f"tof_obstacle_sensor={tof_raw}")

    battery_raw = sensor_data.get("battery_monitor", "")
    battery = str(battery_raw)
    if _BATTERY_DANGER_KEYWORD in battery.upper():
        reasons.append(f"battery_monitor={battery_raw}")
    else:
        battery_pct = _extract_number(battery)
        if battery_pct is not None and battery_pct < _BATTERY_DANGER_PERCENT:
            reasons.append(f"battery_monitor={battery_raw}")

    imu = str(sensor_data.get("imu_gyro", "")).upper()
    if any(kw in imu for kw in _IMU_DANGER_KEYWORDS):
        reasons.append(f"imu_gyro={sensor_data.get('imu_gyro')}")

    return (len(reasons) > 0, reasons)


# --- "Is this packet a significant event?" (vs. quiet/duplicate) -----------

def _sensor_data_differs(a: Optional[dict], b: Optional[dict]) -> bool:
    """True if two sensor_data dicts differ in any key seen in either one."""
    a = a or {}
    b = b or {}
    keys = set(a.keys()) | set(b.keys())
    return any(a.get(k) != b.get(k) for k in keys)


def _is_significant(packet: dict, previous: Optional[dict]) -> bool:
    """A packet is a "significant event" - kept explicitly in the batch's
    summary rather than silently absorbed into the quiet-packet count -
    if it carries a user message, its action_taken differs from the
    previous packet's, or any sensor reading differs from the previous
    packet's. The very first packet in a batch (previous=None) always
    counts as significant, so a window's starting state is never
    silently dropped."""
    if previous is None:
        return True
    if packet.get("user_message"):
        return True
    if packet.get("action_taken") != previous.get("action_taken"):
        return True
    if _sensor_data_differs(packet.get("sensor_data"), previous.get("sensor_data")):
        return True
    return False


def _build_normal_summary(normal_packets: List[dict], significant_events: List[dict]) -> dict:
    """Build the compact JSON-able summary stored in one robot_normal_logs
    row for a batch: total/quiet packet counts plus the significant
    events themselves (trimmed to a few key fields each, and capped in
    count so a genuinely eventful window still produces a bounded row)."""
    trimmed_events = [
        {
            "ts": e.get("ts"),
            "user_message": e.get("user_message") or None,
            "action_taken": e.get("action_taken"),
            "sensor_data": e.get("sensor_data"),
        }
        for e in significant_events[:_MAX_SIGNIFICANT_EVENTS_IN_SUMMARY]
    ]
    return {
        "packet_count": len(normal_packets),
        "quiet_packet_count": len(normal_packets) - len(significant_events),
        "significant_event_count": len(significant_events),
        "significant_events": trimmed_events,
        "truncated": len(significant_events) > _MAX_SIGNIFICANT_EVENTS_IN_SUMMARY,
    }


def filter_and_store_robot_packets(robot_id: str, packets: List[dict]) -> dict:
    """Filter one robot's freshly-flushed batch of Layer 1 telemetry
    packets and persist the result.

    - Every packet that is a hardware danger condition, or already
      carried error_flag=True, is written as its own row in
      robot_error_logs.
    - Every other ("normal") packet is folded into a single compact
      summary row in robot_normal_logs covering the whole batch's time
      window, instead of one row per near-identical packet.

    Never raises: a storage failure is logged and reflected in the
    returned counts (e.g. error_rows_written=0), never propagated - one
    bad batch must not crash Layer 2 or the periodic flush loop.

    Returns:
        {
            "robot_id": str, "total_packets": int, "error_packets": int,
            "normal_packets": int, "significant_events_count": int,
            "error_rows_written": int, "normal_row_written": bool,
        }
    """
    if not packets:
        return {
            "robot_id": robot_id, "total_packets": 0, "error_packets": 0,
            "normal_packets": 0, "significant_events_count": 0,
            "error_rows_written": 0, "normal_row_written": False,
        }

    ordered = sorted(packets, key=lambda p: p.get("ts", 0.0))

    error_rows: List[tuple] = []
    normal_packets: List[dict] = []

    for packet in ordered:
        sensor_data = packet.get("sensor_data") or {}
        is_dangerous, reasons = _sensor_indicates_danger(sensor_data)
        if packet.get("error_flag") or is_dangerous:
            error_rows.append((
                robot_id,
                packet.get("ts", time.time()),
                "SENSOR_DANGER" if is_dangerous else "REPORTED_ERROR",
                json.dumps(sensor_data),
                json.dumps(packet.get("action_taken")),
                packet.get("error_reason") or "; ".join(reasons) or "error_flag set",
                "critical" if is_dangerous else "reported",
            ))
        else:
            normal_packets.append(packet)

    error_rows_written = 0
    if error_rows:
        ok = execute_robot_many(
            "INSERT INTO robot_error_logs (robot_id, timestamp, error_type, "
            "sensor_snapshot_json, action_attempted_json, error_message, severity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            error_rows,
        )
        error_rows_written = len(error_rows) if ok else 0
        if not ok:
            logger.error(f"Layer 2: failed to write {len(error_rows)} error row(s) for robot_id={robot_id}")

    significant_events: List[dict] = []
    previous: Optional[dict] = None
    for packet in normal_packets:
        if _is_significant(packet, previous):
            significant_events.append(packet)
        previous = packet

    normal_row_written = False
    if normal_packets:
        window_start_ts = ordered[0].get("ts", time.time())
        window_end_ts = ordered[-1].get("ts", time.time())
        summary = _build_normal_summary(normal_packets, significant_events)
        cur = execute_robot_query(
            "INSERT INTO robot_normal_logs (robot_id, window_start_ts, window_end_ts, "
            "summary_json, significant_events_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (robot_id, window_start_ts, window_end_ts, json.dumps(summary),
             len(significant_events), time.time()),
        )
        normal_row_written = cur is not None
        if not normal_row_written:
            logger.error(f"Layer 2: failed to write normal-log summary for robot_id={robot_id}")

    return {
        "robot_id": robot_id,
        "total_packets": len(packets),
        "error_packets": len(error_rows),
        "normal_packets": len(normal_packets),
        "significant_events_count": len(significant_events),
        "error_rows_written": error_rows_written,
        "normal_row_written": normal_row_written,
    }


def flush_all_ready_robots(flush_interval_sec: float = 60.0) -> Dict[str, dict]:
    """Pop every robot's newly-accumulated Layer 1 packets that are due
    for a flush (robot_layer1_buffer.pop_ready_for_filtration) and run
    each through filter_and_store_robot_packets.

    Meant to be called on a timer (e.g. every few seconds); each robot
    only does real filtering/storage work once per flush_interval_sec,
    matching Layer 1's own cadence.

    Never raises: one robot's filtering/storage failure is caught,
    logged, and simply excluded from the returned results, rather than
    aborting the whole flush pass for every other robot.

    Returns:
        {robot_id: <filter_and_store_robot_packets() result>, ...} for
        every robot that had ready packets this pass.
    """
    ready = robot_layer1_buffer.pop_ready_for_filtration(flush_interval_sec)
    results: Dict[str, dict] = {}
    for robot_id, packets in ready.items():
        try:
            results[robot_id] = filter_and_store_robot_packets(robot_id, packets)
        except Exception as e:
            logger.error(f"Layer 2: unexpected error filtering robot_id={robot_id}: {e}")
    return results
