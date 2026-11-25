"""Datastore helpers: the Postgres health ping and the optional Supabase client.

All data access runs over a direct psycopg connection (``db.pg``). The Supabase
client here is used only for optional model-artefact upload to Supabase Storage
(``ml.artefacts``); when ``SUPABASE_URL``/``KEY`` are unset it is simply absent and
artefacts stay on local disk.
"""

from __future__ import annotations

import structlog

from config import get_settings

log = structlog.get_logger(__name__)

_client = None
_init_attempted = False


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
        log.info(
            "supabase_not_configured",
            detail="SUPABASE_URL/KEY unset; model artefacts stay on local disk",
        )
        return None

    try:
        from supabase import create_client

        _client = create_client(settings.supabase_url, settings.supabase_key)
        log.info("supabase_connected", url=settings.supabase_url)
        return _client
    except Exception as exc:  # noqa: BLE001
        log.error("supabase_init_failed", error=str(exc))
        return None


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
