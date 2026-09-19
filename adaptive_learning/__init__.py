"""
Adaptive Intelligence & Learning — Phase 4 Public Package Boundary
====================================================================

This package is the single public entry point for the Phase 4 "Adaptive
Intelligence & Learning" subsystem of the robot's software stack.

PURPOSE
-------
Phase 4 is organized into the following internal modules (present in this
same package directory, added incrementally as they are implemented):

    - learning_models.py    Shared data contracts/models used across the
                             other Phase 4 modules (not a runtime engine).
    - feedback_engine.py    Feedback Processing: ingests and normalizes
                             signals about how prior responses/actions
                             were received.
    - learning_engine.py    Learning: consumes processed feedback over
                             time to update internal preference/context
                             models.
    - adaptation_engine.py  Behavioral Adaptation: turns learned signals
                             into concrete behavior-parameter adjustments.
    - adaptive_policy.py    Adaptive Response Policy: applies current
                             adaptation state at decision time to shape
                             the robot's response/action.

This __init__.py contains NONE of that logic. It only defines:

    - a stable, minimal public API surface for the rest of the
      application to import against,
    - explicit, component-specific capability detection, and
    - safe lazy-loading of internal modules.

DEPENDENCY DIRECTION
---------------------
    Existing Application
            |
            v
    Phase 4 Public Boundary   (this file)
            |
            v
    Phase 4 Internal Modules
            |
            v
    Persistence / External Systems (via explicit interfaces in those
    internal modules only)

This direction is never reversed. This file does not import, and must
never import, application-level modules (server, logic, database,
context_memory, user_profile, emotion_engine, personality_engine,
response_policy, ai_services, database drivers, LLM SDKs, network
clients, or ML frameworks). Internal Phase 4 modules must likewise avoid
importing back into application modules, to prevent import cycles such
as:

    __init__.py -> internal module -> server.py -> __init__.py

IMPORT-TIME GUARANTEES
-----------------------
Importing this package performs NO I/O, does not open database or
network connections, does not initialize AI/ML models, does not start
background workers or threads, does not touch the filesystem, and does
not execute any business logic. Importing this package is deterministic
and side-effect free.

LAZY LOADING & AVAILABILITY SEMANTICS
---------------------------------------
Internal modules are optional at any point in time: they may not exist
yet, may be partially implemented, or may be intentionally disabled.
The package being importable does NOT imply any given component is
available — availability is always checked per-component, never
collapsed into a single package-wide boolean.

Capability states, per component, are one of:

    COMPONENT_AVAILABLE        module + expected factory found and usable
    COMPONENT_NOT_IMPLEMENTED  module does not exist yet, or lacks the
                                expected public factory
    COMPONENT_DISABLED         module exists but declares itself disabled
                                (via an internal `ENABLED = False` flag)

Only expected, well-understood conditions (missing module, missing
expected attribute, explicit self-reported disablement) are treated as
"unavailable". Unexpected runtime errors raised while importing or
initializing a present module are NOT swallowed — they propagate, so
real implementation bugs stay diagnosable instead of silently
degrading into "component unavailable".

CACHING
-------
Only successful resolutions are cached (the resolved factory callable).
NOT_IMPLEMENTED and DISABLED are recomputed on each status check rather
than being memoized, so a component that starts out unavailable can
still be picked up later in the same process (e.g. a module hot-added
during development) without a restart — a transient or currently-true
"unavailable" is never treated as a permanent, unrecoverable fact.
Resolution is guarded by a lock so concurrent first-time access from
multiple threads is safe, and no hidden singleton engines are created
here: each `get_*` call returns whatever the internal module's factory
itself returns, fresh, on every call.

BACKWARD COMPATIBILITY
------------------------
This file does not modify, rename, or otherwise affect any Phase 1,
Phase 2, or Phase 3 behavior or public API. It introduces no dependency
from Phase 4 back into core application modules.
"""

from __future__ import annotations

import enum
import importlib
import threading
from typing import Any, Optional

__all__ = [
    "PACKAGE_AVAILABLE",
    "ComponentStatus",
    "get_component_status",
    "is_component_available",
    "get_feedback_engine",
    "get_learning_engine",
    "get_adaptation_engine",
    "get_adaptive_policy",
]

__version__ = "0.1.0"

# The package module itself is always importable and reaches this point
# without error; that fact alone is what PACKAGE_AVAILABLE communicates.
# It intentionally says nothing about which internal components exist —
# the package being importable is NOT equivalent to every Phase 4
# component being available.
PACKAGE_AVAILABLE: bool = True


class ComponentStatus(enum.Enum):
    """Explicit, per-component capability state."""

    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"
    DISABLED = "disabled"


# Maps a public component name to (module name within this package,
# expected public factory attribute name).
#
# learning_models is deliberately excluded: it is a shared contract
# module, not a runtime engine, and has no factory to expose here.
_COMPONENT_MODULES: dict[str, tuple[str, str]] = {
    "feedback_engine": ("feedback_engine", "get_feedback_engine"),
    "learning_engine": ("learning_engine", "get_learning_engine"),
    "adaptation_engine": ("adaptation_engine", "get_adaptation_engine"),
    "adaptive_policy": ("adaptive_policy", "get_adaptive_policy"),
}

_cache_lock = threading.Lock()
_factory_cache: dict[str, Any] = {}


def _resolve_status(component: str) -> ComponentStatus:
    """
    Attempt to resolve one component fresh, with no negative caching.

    Only expected conditions are treated as non-fatal:
      - the submodule does not exist yet (ImportError)          -> NOT_IMPLEMENTED
      - the submodule exists but lacks the expected factory
        (AttributeError)                                        -> NOT_IMPLEMENTED
      - the submodule explicitly declares `ENABLED = False`     -> DISABLED

    Any other exception raised while importing or reading the module
    (syntax errors, errors raised by module-level code, etc.) is a real
    implementation problem and is allowed to propagate rather than
    being reinterpreted as "unavailable".

    On success, the resolved factory is cached so subsequent calls for
    this component short-circuit via get_component_status without
    re-importing or re-checking.
    """
    module_name, factory_attr = _COMPONENT_MODULES[component]

    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
    except ImportError:
        return ComponentStatus.NOT_IMPLEMENTED

    if getattr(module, "ENABLED", True) is False:
        return ComponentStatus.DISABLED

    try:
        factory = getattr(module, factory_attr)
    except AttributeError:
        return ComponentStatus.NOT_IMPLEMENTED

    _factory_cache[component] = factory
    return ComponentStatus.AVAILABLE


def get_component_status(component: str) -> ComponentStatus:
    """
    Return the current, explicit status of a named Phase 4 component.

    `component` must be one of the keys of the internal component map
    (currently: "feedback_engine", "learning_engine",
    "adaptation_engine", "adaptive_policy").

    Raises KeyError for an unknown component name — that is a caller
    programming error, not a runtime availability condition, and is
    not masked.

    Only a successful (AVAILABLE) resolution is cached; NOT_IMPLEMENTED
    and DISABLED are re-evaluated on every call so a component that
    becomes available later in the process lifetime is detected without
    requiring a restart.
    """
    if component not in _COMPONENT_MODULES:
        raise KeyError(f"Unknown Phase 4 component: {component!r}")

    with _cache_lock:
        if component in _factory_cache:
            return ComponentStatus.AVAILABLE
        return _resolve_status(component)


def is_component_available(component: str) -> bool:
    """Convenience wrapper: True only if the component's status is AVAILABLE."""
    return get_component_status(component) is ComponentStatus.AVAILABLE


def _get_factory(component: str) -> Optional[Any]:
    """Return the cached factory for a component if available, else None."""
    if get_component_status(component) is not ComponentStatus.AVAILABLE:
        return None
    with _cache_lock:
        return _factory_cache.get(component)


def get_feedback_engine() -> Optional[Any]:
    """
    Return the Feedback Processing engine handle, or None if that
    component is not implemented or is disabled.
    """
    factory = _get_factory("feedback_engine")
    return factory() if factory is not None else None


def get_learning_engine() -> Optional[Any]:
    """
    Return the Learning engine handle, or None if that component is
    not implemented or is disabled.
    """
    factory = _get_factory("learning_engine")
    return factory() if factory is not None else None


def get_adaptation_engine() -> Optional[Any]:
    """
    Return the Behavioral Adaptation engine handle, or None if that
    component is not implemented or is disabled.
    """
    factory = _get_factory("adaptation_engine")
    return factory() if factory is not None else None


def get_adaptive_policy() -> Optional[Any]:
    """
    Return the Adaptive Response Policy handle, or None if that
    component is not implemented or is disabled.
    """
    factory = _get_factory("adaptive_policy")
    return factory() if factory is not None else None
