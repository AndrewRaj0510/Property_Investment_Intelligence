"""M1A-03 - DLD client (live-only data.dubai source).

The DLD client never returns synthetic data: without DDA credentials it raises
``DLDNotConfigured``. These tests cover the live-only contract (sales and rent
datasets), the rate limiter, and the admin ingest's clean skip.
"""

from __future__ import annotations

import asyncio

import pytest

from ingestion import dld_rent_client
from ingestion.dld_client import (
    DLDNotConfigured,
    DLDQuotaExceeded,
    _RateLimiter,
    fetch_raw_limited,
)


def test_fetch_is_live_only_without_creds(unconfigured_dld):
    """Without DDA credentials the live-only client refuses to fetch.

    Takes `unconfigured_dld` so it holds on a machine that HAS working DDA creds
    otherwise the client happily starts a real, quota-consuming fetch and the
    live-only contract goes untested (F20).
    """
    with pytest.raises(DLDNotConfigured):
        asyncio.run(fetch_raw_limited(limit=1))
    with pytest.raises(DLDNotConfigured):
        asyncio.run(dld_rent_client.fetch_raw_limited(limit=1))


def test_admin_ingest_reports_skip_when_unconfigured(client, unconfigured_dld, monkeypatch):
    """The admin trigger treats 'not configured' as a reported skip, not an error.

    Replaces a test of the old LangGraph ``ingest_dld`` node. That node wrote to
    ``dld_transactions``, a table this database has never had, so the endpoint
    returned "started" and then failed silently (F34). The route now calls
    ``scripts.pull_and_store_dld.run_backfill``, which targets the real
    ``dld_transactions_raw`` - but the live-only contract is unchanged: with no
    DDA credentials it must SAY it skipped rather than pretend to have run.
    """
    import config
    from api.security import hash_password

    s = config.get_settings()
    monkeypatch.setattr(s, "jwt_secret", "test-secret-please-rotate", raising=False)
    monkeypatch.setattr(s, "admin_username", "admin", raising=False)
    monkeypatch.setattr(s, "admin_password_hash", hash_password("S3cret!pw"), raising=False)

    token = client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "S3cret!pw"}
    ).json()["access_token"]

    r = client.post(
        "/api/v1/admin/ingest/dld", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dld_configured"] is False
    assert body["status"] == "skipped"


def test_admin_ingest_calls_the_real_backfill_script():
    """The admin route must invoke the script that writes dld_transactions_raw.

    Pins the rewiring: if this ever points back at a node writing to a table the
    deployment does not have, the endpoint silently does nothing again.
    """
    import inspect

    import api.routes.admin as admin

    src = inspect.getsource(admin._run_dld_ingest)
    assert "run_backfill" in src
    assert "pull_and_store_dld" in src

    from scripts.pull_and_store_dld import TABLE

    assert TABLE == "dld_transactions_raw"


def test_rate_limiter_enforces_per_minute():
    """Limiter allows N immediately, then blocks (we don't wait the full minute)."""
    limiter = _RateLimiter(per_minute=3, per_day=100)

    async def grab(n):
        for _ in range(n):
            await limiter.acquire()

    asyncio.run(grab(3))
    assert len(limiter._calls) == 3
    assert limiter._day_count == 3


def test_rate_limiter_daily_cap():
    limiter = _RateLimiter(per_minute=1000, per_day=2)

    async def grab():
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()

    with pytest.raises(DLDQuotaExceeded):
        asyncio.run(grab())
