"""Pluggable web search layer for the Saarthi web chat channel.

Supported providers (selected by ``SEARCH_PROVIDER``): ``tavily`` and
``serper``. Each provider lives in its own private function
(``_search_tavily`` / ``_search_serper``), so a change to one provider's API
means editing one function.

Contract:
    * ``web_search`` never raises. On any failure it logs the HTTP status code
      or the exception type only, and returns ``[]``.
    * Results are normalized: only http/https URLs, ``domain`` is the netloc
      without ``www.``, snippets are trimmed to 500 characters, entries with an
      empty title or URL are dropped, URLs are de-duplicated, and the list is
      capped at ``max_results``.
    * The API key and result contents are never logged.

API format verification: NOT verified live. The request and response shapes
below come from my knowledge of the Tavily and Serper docs, and this
environment had no access to either provider's documentation. Before
deploying, check them against:
    * Tavily:  https://docs.tavily.com  (Search endpoint, POST /search)
    * Serper:  https://serper.dev  (Google Search API, POST /search)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from config import (
    SEARCH_API_KEY,
    SEARCH_MAX_RESULTS,
    SEARCH_PROVIDER,
    SEARCH_TIMEOUT_SECONDS,
    WEB_SEARCH_ENABLED,
)

logger = logging.getLogger(__name__)

__all__ = ["SearchResult", "search_enabled", "web_search", "search_many"]

# Provider endpoints.
_TAVILY_URL = "https://api.tavily.com/search"
_SERPER_URL = "https://google.serper.dev/search"

# Normalization limits.
_MAX_SNIPPET_CHARS = 500
_MAX_QUERY_CHARS = 400
_MAX_RESULTS_CEILING = 10

# Total attempts per request (one retry on timeout).
_ATTEMPTS = 2

_WHITESPACE_RE = re.compile(r"\s+")

# Ensures the "search is disabled" message is logged only once per process.
_disabled_logged = False


@dataclass
class SearchResult:
    """One normalized web search result.

    Attributes:
        title: Page title (whitespace-collapsed).
        url: Absolute http/https URL.
        domain: URL host without a leading ``www.``.
        snippet: Short text excerpt, at most 500 characters.
    """

    title: str
    url: str
    domain: str
    snippet: str


def search_enabled() -> bool:
    """Return True only if web search is switched on and fully configured.

    Requires ``WEB_SEARCH_ENABLED``, a provider in {"tavily", "serper"} and a
    non-empty ``SEARCH_API_KEY``.
    """
    return bool(
        WEB_SEARCH_ENABLED
        and SEARCH_PROVIDER in ("tavily", "serper")
        and SEARCH_API_KEY
    )


def _clean_text(value: Any) -> str:
    """Return ``value`` as a whitespace-collapsed string ("" if not a string)."""
    if not isinstance(value, str):
        return ""
    return _WHITESPACE_RE.sub(" ", value).strip()


def _build_result(title: Any, url: Any, snippet: Any) -> SearchResult | None:
    """Build a ``SearchResult`` from raw provider fields, or None if unusable."""
    clean_title = _clean_text(title)
    clean_url = url.strip() if isinstance(url, str) else ""
    if not clean_title or not clean_url:
        return None
    try:
        parsed = urlparse(clean_url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return None
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[len("www."):]
    if not domain:
        return None
    return SearchResult(
        title=clean_title,
        url=clean_url,
        domain=domain,
        snippet=_clean_text(snippet)[:_MAX_SNIPPET_CHARS],
    )


def _normalize(results: list[SearchResult], limit: int) -> list[SearchResult]:
    """De-duplicate by URL (ignoring a trailing slash) and cap at ``limit``."""
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for item in results:
        key = item.url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


async def _post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> Any | None:
    """POST JSON and return the decoded response, or None on any failure.

    Retries once on timeout. Logs only status codes / exception type names.
    """
    timeout = httpx.Timeout(float(SEARCH_TIMEOUT_SECONDS))
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                logger.warning("Search request timed out (attempt %d/%d)", attempt, _ATTEMPTS)
                continue
            if response.status_code != 200:
                logger.warning("Search provider returned HTTP %d", response.status_code)
                return None
            try:
                return response.json()
            except ValueError:
                logger.warning("Search provider returned invalid JSON")
                return None
    return None


async def _search_tavily(query: str, n: int) -> list[SearchResult]:
    """Search via Tavily.

    Request:  POST https://api.tavily.com/search
              Authorization: Bearer <key>
              {"query", "max_results", "search_depth": "basic"}
    Response: {"results": [{"title", "url", "content", ...}, ...]}
    (Not verified live; see module docstring.)
    """
    data = await _post_json(
        _TAVILY_URL,
        headers={
            "Authorization": f"Bearer {SEARCH_API_KEY}",
            "Content-Type": "application/json",
        },
        payload={"query": query, "max_results": n, "search_depth": "basic"},
    )
    if not isinstance(data, dict):
        return []
    items = data.get("results")
    if not isinstance(items, list):
        return []
    found: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        built = _build_result(item.get("title"), item.get("url"), item.get("content"))
        if built:
            found.append(built)
    return found


async def _search_serper(query: str, n: int) -> list[SearchResult]:
    """Search via Serper (Google results).

    Request:  POST https://google.serper.dev/search
              X-API-KEY: <key>
              {"q", "num", "gl": "in"}
    Response: {"organic": [{"title", "link", "snippet", ...}, ...]}
    (Not verified live; see module docstring.)
    """
    data = await _post_json(
        _SERPER_URL,
        headers={
            "X-API-KEY": str(SEARCH_API_KEY),
            "Content-Type": "application/json",
        },
        payload={"q": query, "num": n, "gl": "in"},
    )
    if not isinstance(data, dict):
        return []
    items = data.get("organic")
    if not isinstance(items, list):
        return []
    found: list[SearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        built = _build_result(item.get("title"), item.get("link"), item.get("snippet"))
        if built:
            found.append(built)
    return found


_PROVIDERS: dict[str, Callable[[str, int], Awaitable[list[SearchResult]]]] = {
    "tavily": _search_tavily,
    "serper": _search_serper,
}


async def web_search(query: str, max_results: int | None = None) -> list[SearchResult]:
    """Search the web with the configured provider.

    Never raises. Returns ``[]`` if search is not enabled, the query is empty,
    or anything goes wrong (only status codes / exception types are logged).

    Args:
        query: Search query text.
        max_results: Maximum results to return; defaults to
            ``SEARCH_MAX_RESULTS`` and is clamped to 1..10.

    Returns:
        A normalized, de-duplicated list of ``SearchResult``.
    """
    global _disabled_logged

    if not search_enabled():
        if not _disabled_logged:
            _disabled_logged = True
            logger.info("Web search is not enabled or not configured; returning no results.")
        return []

    clean_query = _clean_text(query)[:_MAX_QUERY_CHARS]
    if not clean_query:
        return []

    limit = max_results if max_results is not None else SEARCH_MAX_RESULTS
    limit = max(1, min(int(limit), _MAX_RESULTS_CEILING))

    provider = _PROVIDERS.get(SEARCH_PROVIDER)
    if provider is None:
        return []

    try:
        raw = await provider(clean_query, limit)
        return _normalize(raw, limit)
    except Exception as exc:  # noqa: BLE001 - search must never break a chat reply
        logger.warning("Web search failed (%s)", type(exc).__name__)
        return []


async def search_many(
    queries: list[str], max_results_each: int | None = None
) -> list[list[SearchResult]]:
    """Run several searches concurrently.

    Args:
        queries: Search queries.
        max_results_each: Per-query result cap (see ``web_search``).

    Returns:
        One result list per query, in the same order as ``queries``.
    """
    if not queries:
        return []
    return list(
        await asyncio.gather(*(web_search(q, max_results_each) for q in queries))
    )
