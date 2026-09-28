"""DLD transaction queries."""

from __future__ import annotations

from db.client import get_db
from db.models import DLD_TRANSACTIONS


def upsert_transactions(rows: list[dict]) -> int:
    if not rows:
        return 0
    db = get_db()
    db.table(DLD_TRANSACTIONS).upsert(rows, on_conflict="transaction_id").execute()
    return len(rows)


def all_transactions() -> list[dict]:
    return get_db().table(DLD_TRANSACTIONS).select("*").execute().data


def latest_transaction_date() -> str | None:
    """Return the most recent ``transaction_date`` we have stored, or None.

    Used to drive incremental DLD fetches (pull only newer-or-equal rows). Works
    against both the live Supabase client and the in-memory store; falls back to
    a Python-side max if the backend cannot order server-side.
    """
    db = get_db()
    try:
        res = (
            db.table(DLD_TRANSACTIONS)
            .select("transaction_date")
            .order("transaction_date", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0].get("transaction_date")
    except Exception:  # noqa: BLE001 - backend may not support order()
        rows = db.table(DLD_TRANSACTIONS).select("transaction_date").execute().data
        dates = [r.get("transaction_date") for r in rows if r.get("transaction_date")]
        if dates:
            return max(dates)
    return None
