"""Pipeline-run log queries."""

from __future__ import annotations

from db.client import get_db
from db.models import PIPELINE_RUNS


def start_run(run_id: str) -> None:
    get_db().table(PIPELINE_RUNS).insert(
        {"id": run_id, "status": "running", "stage": "start"}
    ).execute()


def update_stage(run_id: str, stage: str) -> None:
    get_db().table(PIPELINE_RUNS).update({"stage": stage}).eq("id", run_id).execute()


def complete_run(run_id: str, completed_at: str) -> None:
    (
        get_db()
        .table(PIPELINE_RUNS)
        .update({"status": "completed", "completed_at": completed_at})
        .eq("id", run_id)
        .execute()
    )


def fail_run(run_id: str, error: str) -> None:
    (
        get_db()
        .table(PIPELINE_RUNS)
        .update({"status": "failed", "error_message": error})
        .eq("id", run_id)
        .execute()
    )


def latest_run() -> dict | None:
    rows = (
        get_db()
        .table(PIPELINE_RUNS)
        .select("*")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    return rows[0] if rows else None
