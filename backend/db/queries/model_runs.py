"""Model-run metadata queries."""

from __future__ import annotations

from db.client import get_db
from db.models import MODEL_RUNS


def insert_run(row: dict) -> dict:
    res = get_db().table(MODEL_RUNS).insert(row).execute()
    return res.data[0] if res.data else row


def active_run(phase: int | None = None) -> dict | None:
    q = get_db().table(MODEL_RUNS).select("*").eq("is_active", True)
    if phase is not None:
        q = q.eq("phase", phase)
    rows = q.execute().data
    return rows[0] if rows else None


def deactivate_all(phase: int | None = None) -> None:
    db = get_db()
    q = db.table(MODEL_RUNS).update({"is_active": False}).eq("is_active", True)
    if phase is not None:
        q = q.eq("phase", phase)
    q.execute()


def activate_run(run_id: str) -> None:
    (
        get_db()
        .table(MODEL_RUNS)
        .update({"is_active": True})
        .eq("id", run_id)
        .execute()
    )


def all_runs() -> list[dict]:
    return (
        get_db().table(MODEL_RUNS).select("*").order("trained_at", desc=True).execute().data
    )
