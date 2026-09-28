"""Pull live DLD records from data.dubai and store them in Postgres.

Resumable backward backfill - just trigger it; it continues where it left off:
  * First run (empty table): pulls the newest DLD_FETCH_LIMIT records
    (ordered by instance_date desc), creates the table from the real columns,
    adds a UNIQUE(transaction_id) constraint, and upserts them.
  * Later runs: read the OLDEST instance_date already stored, then fetch
    instance_date <= that date (desc), going further back in history. The
    boundary day is re-fetched but deduped by the upsert, so no gaps / no dups.
  * When a run makes no progress (only already-stored rows, table didn't grow),
    it reports the backfill is complete and exits cleanly.

Run from the backend/ directory (so .env and imports resolve):
    .venv/Scripts/python.exe -m scripts.pull_and_store_dld

Requirements: DLD_* credentials and Supabase Postgres (parts or URL) in .env.
The data.dubai API is UAE-only, so this must run from a UAE connection.
"""

from __future__ import annotations

import asyncio
import json
import sys

import structlog

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 - older streams may not support reconfigure
    pass

from config import get_settings
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident
from ingestion.dld_client import fetch_raw_limited

log = structlog.get_logger(__name__)

TABLE = "dld_transactions_raw"
LOADED_AT_COL = "_loaded_at"
KEY_COL = "transaction_id"
ORDER_BY = "instance_date"
ORDER_DIR = "desc"


def _collect_columns(records: list[dict]) -> list[str]:
    """Ordered union of all keys across the pulled records (first-seen order)."""
    cols: list[str] = []
    for rec in records:
        for k in rec.keys():
            if k not in cols:
                cols.append(k)
    return cols


def _create_table_sql(columns: list[str]) -> str:
    """Build a CREATE TABLE IF NOT EXISTS with every source column as TEXT.

    One-time bootstrap: harmless/no-op once the table exists.
    """
    col_defs = ",\n    ".join(f"{_quote_ident(c)} TEXT" for c in columns)
    return (
        f"CREATE TABLE IF NOT EXISTS {_quote_ident(TABLE)} (\n"
        f"    {col_defs},\n"
        f"    {_quote_ident(LOADED_AT_COL)} TIMESTAMPTZ DEFAULT NOW()\n"
        f");"
    )


def _unique_constraint_sql() -> str:
    """Add a UNIQUE constraint on the natural key (idempotent) for upserts."""
    return (
        "DO $$ BEGIN "
        f"  ALTER TABLE {_quote_ident(TABLE)} "
        f"  ADD CONSTRAINT {_quote_ident(TABLE + '_' + KEY_COL + '_key')} "
        f"  UNIQUE ({_quote_ident(KEY_COL)}); "
        "EXCEPTION WHEN duplicate_table THEN NULL; "
        "WHEN duplicate_object THEN NULL; END $$;"
    )


def _insert_sql(columns: list[str]) -> str:
    """Upsert on the natural key so re-runs don't create duplicate rows."""
    cols_sql = ", ".join(_quote_ident(c) for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(
        f"{_quote_ident(c)} = EXCLUDED.{_quote_ident(c)}"
        for c in columns
        if c != KEY_COL
    )
    return (
        f"INSERT INTO {_quote_ident(TABLE)} ({cols_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT ({_quote_ident(KEY_COL)}) DO UPDATE SET {updates};"
    )


def _stringify(value) -> str | None:
    """Store everything as TEXT; serialize dict/list values to JSON."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _table_exists(cur) -> bool:
    cur.execute("SELECT to_regclass(%s);", (f"public.{TABLE}",))
    return cur.fetchone()[0] is not None


def _ensure_table(cur, columns: list[str]) -> None:
    cur.execute(_create_table_sql(columns))
    cur.execute(_unique_constraint_sql())


def _date_progress(cur, date_str: str) -> tuple[int, str | None]:
    """For a given instance_date: (rows stored, min transaction_id stored).

    The min transaction_id is the intra-day cursor - the next page fetches
    records on this date with transaction_id < that value (string ordering).
    """
    cur.execute(
        f"SELECT count(*), min({_quote_ident(KEY_COL)}) "
        f"FROM {_quote_ident(TABLE)} WHERE {_quote_ident(ORDER_BY)} = %s;",
        (date_str,),
    )
    cnt, min_id = cur.fetchone()
    return cnt, min_id


def _oldest_stored_date(cur) -> str | None:
    cur.execute(f"SELECT min({_quote_ident(ORDER_BY)}) FROM {_quote_ident(TABLE)};")
    return cur.fetchone()[0]


def _upsert_records(cur, columns: list[str], records: list[dict]) -> None:
    upsert = _insert_sql(columns)
    for rec in records:
        cur.execute(upsert, [_stringify(rec.get(c)) for c in columns])


async def _fetch_page(date_str: str, before_id: str | None) -> list[dict]:
    """Fetch one page (<=1000) for a single date, newest transaction_id first.

    ``before_id`` is the intra-day cursor: only records with transaction_id
    strictly less than it (string order) are returned - so we never re-read.
    """
    flt = f"{ORDER_BY}='{date_str}'"
    if before_id is not None:
        flt += f" and {KEY_COL}<'{before_id}'"
    return await fetch_raw_limited(
        order_by=KEY_COL, order_dir="desc", filter_expr=flt
    )


async def _newest_date_before(date_str: str) -> str | None:
    """Return the newest instance_date strictly older than ``date_str``."""
    rows = await fetch_raw_limited(
        limit=1,
        order_by=ORDER_BY,
        order_dir="desc",
        filter_expr=f"{ORDER_BY}<'{date_str}'",
    )
    return rows[0].get(ORDER_BY) if rows else None


async def _newest_date() -> str | None:
    """Return the newest instance_date in the dataset (for the first run)."""
    rows = await fetch_raw_limited(limit=1, order_by=ORDER_BY, order_dir="desc")
    return rows[0].get(ORDER_BY) if rows else None


async def run_backfill(settings) -> None:
    per_date_cap = settings.dld_max_per_date
    run_budget = settings.dld_records_per_run

    with _connect(settings) as conn:
        with conn.cursor() as cur:
            table_ready = _table_exists(cur)
            current_date = _oldest_stored_date(cur) if table_ready else None

    if current_date is None:
        current_date = await _newest_date()
        if current_date is None:
            print("No records available from the DLD API.")
            return
        print(f"First run: starting at newest date {current_date}.")
    else:
        print(f"Resuming at oldest stored date {current_date}.")

    fetched_this_run = 0
    added_this_run = 0
    columns: list[str] | None = None

    while fetched_this_run < run_budget:
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                have, min_id = (
                    _date_progress(cur, current_date) if _table_exists(cur) else (0, None)
                )

        if have >= per_date_cap:
            prev = await _newest_date_before(current_date)
            if prev is None:
                print(f"Backfill complete: reached the oldest date ({current_date}).")
                break
            print(f"Date {current_date} at cap ({have}/{per_date_cap}); "
                  f"moving back to {prev}.")
            current_date = prev
            continue

        page = await _fetch_page(current_date, min_id)
        fetched_this_run += len(page)

        if not page:
            prev = await _newest_date_before(current_date)
            if prev is None:
                print(f"Backfill complete: reached the oldest date ({current_date}).")
                break
            print(f"Date {current_date} exhausted ({have} rows); moving to {prev}.")
            current_date = prev
            continue

        room = per_date_cap - have
        keep = page[:room]

        if columns is None:
            columns = _collect_columns(keep)
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                _ensure_table(cur, columns)
                before = _date_progress(cur, current_date)[0]
                _upsert_records(cur, columns, keep)
                conn.commit()
                after = _date_progress(cur, current_date)[0]
        added_this_run += after - before
        print(f"  {current_date}: +{after - before} (now {after}/{per_date_cap}); "
              f"run total fetched {fetched_this_run}/{run_budget}.")

        if after >= per_date_cap:
            prev = await _newest_date_before(current_date)
            if prev is None:
                print(f"Backfill complete: reached the oldest date ({current_date}).")
                break
            current_date = prev

    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*), min({_quote_ident(ORDER_BY)}), "
                        f"max({_quote_ident(ORDER_BY)}) FROM {_quote_ident(TABLE)};")
            total, mn, mx = cur.fetchone()
    log.info("backfill_run_done", fetched=fetched_this_run, added=added_this_run,
             total=total)
    print(f"\nRun done: fetched {fetched_this_run}, added {added_this_run} new. "
          f"Table now {total} rows spanning {mn} .. {mx}.")


def main() -> None:
    settings = get_settings()
    if not settings.dld_configured:
        raise SystemExit("DLD credentials are not configured in .env.")
    if not (settings.db_parts_configured or settings.supabase_db_url):
        raise SystemExit(
            "No Postgres connection configured. Set SUPABASE_DB_HOST / "
            "SUPABASE_DB_USER / SUPABASE_DB_PASSWORD (recommended) or "
            "SUPABASE_DB_URL in .env."
        )
    asyncio.run(run_backfill(settings))


if __name__ == "__main__":
    main()
