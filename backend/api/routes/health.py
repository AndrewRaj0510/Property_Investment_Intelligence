"""GET /api/v1/health - readiness and data-freshness probe.

Reports DB connectivity, whether an active model exists, and the timestamp of
the most recent pipeline run. Degrades cleanly in mock mode (no DB configured).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from api.rate_limit import limiter
from api.schemas.health import HealthResponse
from config import get_settings
from db.client import is_db_available
from ml.persist import active_run_row

log = structlog.get_logger(__name__)
router = APIRouter()


def _last_pipeline_run() -> str | None:
    """Timestamp of the most recent pipeline run, via psycopg (best-effort).

    The pipeline_runs table may not exist in a given deployment, so any failure is
    swallowed - health should never 500 on an optional freshness field.
    """
    try:
        from db.pg import connect as _connect

        with _connect(get_settings()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT started_at FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
            )
            row = cur.fetchone()
        return str(row[0]) if row else None
    except Exception as exc:  # noqa: BLE001
        log.warning("health_pipeline_check_failed", error=str(exc))
        return None


@router.get("/health", response_model=HealthResponse, tags=["system"])
@limiter.limit("120/minute")
async def health(request: Request) -> HealthResponse:
    db_ok = is_db_available()

    model_loaded = False
    last_pipeline_run: str | None = None

    if db_ok:
        try:
            model_loaded = active_run_row() is not None
        except Exception as exc:  # noqa: BLE001
            log.warning("health_model_check_failed", error=str(exc))
        last_pipeline_run = _last_pipeline_run()

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db=db_ok,
        model_loaded=model_loaded,
        last_pipeline_run=last_pipeline_run,
    )
