"""Dubai Land Department transaction client - LIVE source (data.dubai / DDA).

This client is **live-only**: it fetches genuine DLD sale transactions from the
Digital Dubai Authority (DDA) OpenAPI and never returns synthetic data. DLD
records are historical and append-only, so callers use it for *incremental*
ingestion (pull only what is newer than what we already stored) to build the
training table for the yield model - not on a per-request hot path.

Endpoint (per `artifacts/dld_data.md`):
    https://apis.data.dubai/open/<entity>/<dataset>
    e.g. https://apis.data.dubai/open/dld/dld_transactions-open-api

Auth flow:
    1. POST .../gatewaytoken/1.0.0/getAccessToken  → Bearer token (1h TTL)
    2. GET  .../open/<entity>/<dataset>?page=...    → one page of records

Quota safety (approved application limits - never exceed):
    * 20 requests / minute
    * 500 requests / day
A process-local limiter throttles to stay within both; the daily counter is a
best-effort guard within a single process lifetime.

When DDA credentials are absent the client raises ``DLDNotConfigured`` rather
than inventing data. The ingest node treats that as a clean skip so the rest of
the app still boots and serves the already-trained model.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from datetime import date

import httpx
import structlog

from config import get_settings

log = structlog.get_logger(__name__)

_TOKEN_PATH = "/secure/ssis/dubaiai/gatewaytoken/1.0.0/getAccessToken"
_TOKEN_REFRESH_SKEW_S = 60
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4


class DLDNotConfigured(RuntimeError):
    """Raised when a live DLD fetch is requested without DDA credentials."""


class DLDQuotaExceeded(RuntimeError):
    """Raised when the approved daily request quota would be exceeded."""


class _RateLimiter:
    """Sliding-window per-minute limiter + hard daily cap (process-local).

    Thread-safe and asyncio-friendly: ``acquire`` sleeps (without holding the
    lock) until a request slot is free, then records the call.
    """

    def __init__(self, per_minute: int, per_day: int) -> None:
        self._per_minute = max(1, per_minute)
        self._per_day = max(1, per_day)
        self._calls: deque[float] = deque()
        self._day_count = 0
        self._day = date.today()
        self._lock = threading.Lock()

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self._day_count = 0

    async def acquire(self) -> None:
        while True:
            with self._lock:
                self._roll_day()
                if self._day_count >= self._per_day:
                    raise DLDQuotaExceeded(
                        f"Daily DLD request quota of {self._per_day} reached"
                    )
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self._per_minute:
                    self._calls.append(now)
                    self._day_count += 1
                    return
                wait_s = 60.0 - (now - self._calls[0]) + 0.01
            await asyncio.sleep(max(wait_s, 0.05))


_limiter: _RateLimiter | None = None
_limiter_lock = threading.Lock()
_token: str | None = None
_token_expiry_monotonic: float = 0.0
_token_lock = asyncio.Lock()


def _get_limiter() -> _RateLimiter:
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            s = get_settings()
            _limiter = _RateLimiter(
                s.dld_max_requests_per_minute, s.dld_max_requests_per_day
            )
        return _limiter


def _require_configured():
    s = get_settings()
    if not s.dld_configured:
        raise DLDNotConfigured(
            "DLD (data.dubai) credentials are not configured; cannot fetch live data"
        )
    return s


async def _get_token(http: httpx.AsyncClient) -> str:
    """Return a valid Bearer token, refreshing via the gateway when needed."""
    global _token, _token_expiry_monotonic
    async with _token_lock:
        if _token and time.monotonic() < _token_expiry_monotonic:
            return _token

        s = get_settings()
        url = f"{s.dld_base_url}{_TOKEN_PATH}"
        headers = {
            "Content-Type": "application/json",
            "x-DDA-SecurityApplicationIdentifier": s.dld_security_identifier,
        }
        body = {
            "grant_type": "client_credentials",
            "client_id": s.dld_client_id,
            "client_secret": s.dld_client_secret,
        }
        resp = await http.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError("DDA token response did not contain access_token")
        expires_in = int(data.get("expires_in", 3600))
        _token = token
        _token_expiry_monotonic = time.monotonic() + max(
            expires_in - _TOKEN_REFRESH_SKEW_S, 30
        )
        log.info("dld_token_refreshed", expires_in=expires_in)
        return _token


def _invalidate_token() -> None:
    global _token, _token_expiry_monotonic
    _token = None
    _token_expiry_monotonic = 0.0


async def _get_with_retry(
    http: httpx.AsyncClient, url: str, params: dict
) -> httpx.Response:
    """GET with token auth, rate limiting, and bounded retry/backoff."""
    limiter = _get_limiter()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        await limiter.acquire()
        token = await _get_token(http)
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = await http.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            last_exc = exc
            backoff = min(2 ** attempt, 30)
            log.warning("dld_request_error", attempt=attempt, error=str(exc))
            await asyncio.sleep(backoff)
            continue

        if resp.status_code == 401:
            log.warning("dld_unauthorized_refreshing_token", attempt=attempt)
            _invalidate_token()
            last_exc = httpx.HTTPStatusError(
                "401 Unauthorized", request=resp.request, response=resp
            )
            continue

        if resp.status_code in _RETRYABLE_STATUS:
            backoff = min(2 ** attempt, 30)
            log.warning(
                "dld_retryable_status",
                attempt=attempt,
                status=resp.status_code,
                backoff=backoff,
            )
            last_exc = httpx.HTTPStatusError(
                f"{resp.status_code}", request=resp.request, response=resp
            )
            await asyncio.sleep(backoff)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError(f"DLD request failed after {_MAX_RETRIES} retries") from last_exc


def _dataset_url(settings, dataset: str | None = None) -> str:
    return (
        f"{settings.dld_base_url}/open/{settings.dld_entity}/"
        f"{dataset or settings.dld_dataset}"
    )


async def fetch_raw(
    page: int = 1,
    page_size: int | None = None,
    order_by: str | None = None,
    order_dir: str | None = None,
    filter_expr: str | None = None,
    dataset: str | None = None,
) -> list[dict]:
    """Fetch one page of DLD records UNMAPPED - exactly as the API returns them.

    Used to inspect real column names and to build a table that mirrors the
    source schema faithfully (no field-name guessing). Optional ``order_by`` /
    ``order_dir`` sort server-side; ``filter_expr`` is a DDA filter expression
    (e.g. ``instance_date<='2025-11-07'``). ``dataset`` selects another dataset
    under the same entity (default: the sales transactions). Live-only.
    """
    s = _require_configured()
    page_size = page_size or s.dld_page_size
    url = _dataset_url(s, dataset)
    params: dict = {"page": page, "pageSize": page_size}
    if order_by:
        params["order_by"] = order_by
        params["order_dir"] = order_dir or "desc"
    if filter_expr:
        params["filter"] = filter_expr
    async with httpx.AsyncClient(timeout=s.dld_request_timeout_s) as http:
        resp = await _get_with_retry(http, url, params)
        raw = resp.json().get("results", [])
    log.info(
        "dld_raw_page_fetched",
        dataset=dataset or s.dld_dataset,
        page=page,
        rows=len(raw),
    )
    return raw


async def fetch_raw_limited(
    limit: int | None = None,
    order_by: str | None = None,
    order_dir: str | None = None,
    filter_expr: str | None = None,
    dataset: str | None = None,
) -> list[dict]:
    """Fetch up to ``limit`` raw records (default: ``DLD_FETCH_LIMIT``).

    Pages as needed but never requests more per page than the cap, so limit=1
    is a single 1-record request. 0/None cap means a single default-size page.
    ``order_by``/``order_dir`` sort server-side (e.g. instance_date desc for the
    latest transactions first). ``filter_expr`` passes a DDA filter expression.
    """
    s = _require_configured()
    cap = s.dld_fetch_limit if limit is None else limit
    cap = cap if cap and cap > 0 else None
    page_size = s.dld_page_size if cap is None else min(s.dld_page_size, cap)

    out: list[dict] = []
    page = 1
    while True:
        batch = await fetch_raw(
            page=page,
            page_size=page_size,
            order_by=order_by,
            order_dir=order_dir,
            filter_expr=filter_expr,
            dataset=dataset,
        )
        if not batch:
            break
        out.extend(batch)
        if cap is not None and len(out) >= cap:
            return out[:cap]
        if len(batch) < page_size:
            break
        page += 1
    return out
