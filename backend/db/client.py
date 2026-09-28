"""Supabase client singleton - the single datastore (live-only).

The app reads an already-trained model and already-ingested data from Supabase;
the weekly backend job is what populates/refreshes it. There is no in-memory or
mock fallback: if Supabase is not configured, data access fails loudly rather
than silently serving fabricated data.
"""

from __future__ import annotations

import structlog

from config import get_settings

log = structlog.get_logger(__name__)

_client = None
_init_attempted = False


class SupabaseNotConfigured(RuntimeError):
    """Raised when a data operation is attempted without Supabase credentials."""


def get_supabase():
    """Return a cached Supabase client, or None if not configured/available."""
    global _client, _init_attempted

    if _client is not None:
        return _client
    if _init_attempted:
        return None

    _init_attempted = True
    settings = get_settings()

    if not settings.supabase_configured:
        log.error("supabase_not_configured", detail="SUPABASE_URL/KEY are required")
        return None

    try:
        from supabase import create_client

        _client = create_client(settings.supabase_url, settings.supabase_key)
        log.info("supabase_connected", url=settings.supabase_url)
        return _client
    except Exception as exc:  # noqa: BLE001
        log.error("supabase_init_failed", error=str(exc))
        return None


def get_db():
    """Return the Supabase client. Raises if Supabase is unavailable.

    All data access goes through this, so a missing/broken Supabase connection
    surfaces immediately instead of being masked by a fake store.
    """
    client = get_supabase()
    if client is None:
        raise SupabaseNotConfigured(
            "Supabase is not configured or unreachable; set SUPABASE_URL and "
            "SUPABASE_KEY in the environment."
        )
    return client


def is_db_available() -> bool:
    """Lightweight health check for the datastore the data endpoints actually use.

    Data access runs over a DIRECT psycopg connection (the Supabase ORM is not
    configured - see DISSERTATION_FINDINGS F6), so health must ping *that*
    connection, not the ORM. A cheap ``SELECT 1`` over the pooled Postgres
    connection reflects real availability; the ORM-based ping previously here made
    /health report ``db: false`` even when every data endpoint worked.
    """
    settings = get_settings()
    if not settings.db_parts_configured:
        return False
    try:
        from db.pg import connect as _connect

        with _connect(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("db_ping_failed", error=str(exc))
        return False
