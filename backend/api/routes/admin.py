"""Secured admin endpoints (JWT-protected, rate-limited).

Exposes a manual trigger for a bounded, incremental DLD sales top-up. This is an
administrative, state-changing operation, so it sits behind:
  * JWT bearer auth (``require_admin``) - fail-closed,
  * a strict per-IP rate limit,
  * background execution so the request returns immediately.

**What this runs, and why it changed.** It used to call the LangGraph node
``pipeline.nodes.ingest_dld``, which upserted into ``dld_transactions`` - a table
that has never existed in this database. The write failed, the node swallowed the
error into its ``state["errors"]`` list, and the endpoint had already returned
``{"status": "started"}``. So the trigger reported success and did nothing, every
time.

It now calls ``scripts.pull_and_store_dld.run_backfill`` - the script that
actually produced the deployed sales data. It writes to ``dld_transactions_raw``
(what the feature matrix reads), is resumable, and is bounded per run by
``DLD_RECORDS_PER_RUN`` and ``DLD_MAX_PER_DATE`` so a trigger cannot exhaust the
approved DDA quota (500 requests/day, 20/minute).

It does NOT retrain: refreshing rows and rebuilding the model are separate
decisions. Retrain with ``python -m scripts.train_model`` once data is refreshed.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel

from api.rate_limit import limiter
from api.security import require_admin
from config import get_settings

log = structlog.get_logger(__name__)
router = APIRouter()


class DLDIngestResponse(BaseModel):
    run_id: str
    status: str
    dld_configured: bool
    detail: str


async def _run_dld_ingest(run_id: str) -> None:
    """Background task: run the bounded incremental backfill and log the outcome."""
    from scripts.pull_and_store_dld import run_backfill

    settings = get_settings()
    try:
        await run_backfill(settings)
        log.info("admin_dld_ingest_finished", run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - a background task must not crash the app
        log.error("admin_dld_ingest_failed", run_id=run_id, error=str(exc))


@router.post("/admin/ingest/dld", response_model=DLDIngestResponse, tags=["admin"])
@limiter.limit("3/minute")
async def trigger_dld_ingest(
    request: Request,
    background: BackgroundTasks,
    _admin: str = Depends(require_admin),
) -> DLDIngestResponse:
    settings = get_settings()
    run_id = str(uuid.uuid4())

    if not settings.dld_configured:
        return DLDIngestResponse(
            run_id=run_id,
            status="skipped",
            dld_configured=False,
            detail="DLD (data.dubai) credentials are not configured; ingest skipped.",
        )

    if not (settings.db_parts_configured or settings.supabase_db_url):
        return DLDIngestResponse(
            run_id=run_id,
            status="skipped",
            dld_configured=True,
            detail="No Postgres connection configured; ingest skipped.",
        )

    background.add_task(_run_dld_ingest, run_id)
    log.info("admin_dld_ingest_started", run_id=run_id, by=_admin)
    return DLDIngestResponse(
        run_id=run_id,
        status="started",
        dld_configured=True,
        detail=(
            f"Incremental DLD sales backfill started (max "
            f"{settings.dld_records_per_run} records this run, "
            f"{settings.dld_max_per_date}/date) into dld_transactions_raw. "
            "Does not retrain - run scripts.train_model afterwards. "
            "Check server logs for the outcome."
        ),
    )
