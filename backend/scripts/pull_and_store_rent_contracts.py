"""Pull live DLD Ejari rent contracts from data.dubai and store them in Postgres.

The rental counterpart of ``pull_and_store_dld.py``, but it backfills by
**``load_timestamp``** (the DDA load-batch stamp) rather than by an event date.

Why load_timestamp: DDA loads rent contracts in **bulk batches** - every row in
a batch shares one ``load_timestamp`` (on staging the whole dataset is a single
batch, e.g. ``2025-11-13 23:25:34``). ``contract_start_date`` is scrambled on
staging (all ``2030-12-31``), so it can't drive a spread there; the load batch
is the meaningful bucket. In production, each periodic load gets its own
timestamp, so stepping back through load_timestamps walks the load history.

Strategy (two-level cursor, same shape as the sales script):
  * OUTER cursor = ``load_timestamp`` - process one batch, then step to the
    next-older batch.
  * INNER paging = ``contract_id`` desc within a batch, so we never re-read.
  * Each batch is capped at ``DLD_RENT_MAX_PER_LOAD_TS`` (5000) stored rows.
  * ``DLD_RENT_RECORDS_PER_RUN`` (5000) bounds records fetched per invocation.

Natural key is ``contract_id`` (string-sortable) with a UNIQUE constraint, so
re-runs upsert instead of duplicating. Reuses the sales script's pure helpers
(``_connect``, ``_collect_columns``, ``_stringify``, ``_quote_ident``).

Run from the backend/ directory (so .env and imports resolve):
    .venv/Scripts/python.exe -m scripts.pull_and_store_rent_contracts

Requirements: DLD_* credentials and Supabase Postgres in .env. The data.dubai
API is UAE-only, so this must run from a UAE connection.
"""

from __future__ import annotations

import asyncio
import sys

import structlog

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import get_settings
from ingestion.dld_rent_client import fetch_raw_limited

from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident

from scripts.pull_and_store_dld import _collect_columns, _stringify

log = structlog.get_logger(__name__)

TABLE = "dld_rent_contracts_raw"
LOADED_AT_COL = "_loaded_at"
KEY_COL = "contract_id"
CURSOR = "load_timestamp"
MAX_LOAD_TS_PER_RUN = 1


def _create_table_sql(columns: list[str]) -> str:
    col_defs = ",\n    ".join(f"{_quote_ident(c)} TEXT" for c in columns)
    return (
        f"CREATE TABLE IF NOT EXISTS {_quote_ident(TABLE)} (\n"
        f"    {col_defs},\n"
        f"    {_quote_ident(LOADED_AT_COL)} TIMESTAMPTZ DEFAULT NOW()\n"
        f");"
    )


def _unique_constraint_sql() -> str:
    return (
        "DO $$ BEGIN "
        f"  ALTER TABLE {_quote_ident(TABLE)} "
        f"  ADD CONSTRAINT {_quote_ident(TABLE + '_' + KEY_COL + '_key')} "
        f"  UNIQUE ({_quote_ident(KEY_COL)}); "
        "EXCEPTION WHEN duplicate_table THEN NULL; "
        "WHEN duplicate_object THEN NULL; END $$;"
    )


def _insert_sql(columns: list[str]) -> str:
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


def _table_exists(cur) -> bool:
    cur.execute("SELECT to_regclass(%s);", (f"public.{TABLE}",))
    return cur.fetchone()[0] is not None


def _ensure_table(cur, columns: list[str]) -> None:
    cur.execute(_create_table_sql(columns))
    cur.execute(_unique_constraint_sql())


def _batch_progress(cur, load_ts: str) -> tuple[int, str | None]:
    """For a load_timestamp batch: (rows stored, min contract_id stored).

    The min contract_id is the intra-batch cursor - the next page fetches rows
    in this batch with contract_id < that value (string ordering).
    """
    cur.execute(
        f"SELECT count(*), min({_quote_ident(KEY_COL)}) "
        f"FROM {_quote_ident(TABLE)} WHERE {_quote_ident(CURSOR)} = %s;",
        (load_ts,),
    )
    return cur.fetchone()


def _oldest_stored_load_ts(cur) -> str | None:
    cur.execute(f"SELECT min({_quote_ident(CURSOR)}) FROM {_quote_ident(TABLE)};")
    return cur.fetchone()[0]


def _upsert_records(cur, columns: list[str], records: list[dict]) -> None:
    upsert = _insert_sql(columns)
    for rec in records:
        cur.execute(upsert, [_stringify(rec.get(c)) for c in columns])


async def _fetch_page(load_ts: str, before_id: str | None) -> list[dict]:
    """One page (<=1000) for a single load_timestamp, newest contract_id first."""
    flt = f"{CURSOR}='{load_ts}'"
    if before_id is not None:
        flt += f" and {KEY_COL}<'{before_id}'"
    return await fetch_raw_limited(order_by=KEY_COL, order_dir="desc", filter_expr=flt)


async def _newest_load_ts_before(load_ts: str) -> str | None:
    """Return the newest load_timestamp strictly older than ``load_ts``."""
    rows = await fetch_raw_limited(
        limit=1, order_by=CURSOR, order_dir="desc",
        filter_expr=f"{CURSOR}<'{load_ts}'",
    )
    return rows[0].get(CURSOR) if rows else None


async def _newest_load_ts() -> str | None:
    """Return the newest load_timestamp in the dataset (for the first run)."""
    rows = await fetch_raw_limited(limit=1, order_by=CURSOR, order_dir="desc")
    return rows[0].get(CURSOR) if rows else None


async def run_backfill(settings) -> None:
    per_batch_cap = settings.dld_rent_max_per_load_ts
    run_budget = settings.dld_rent_records_per_run

    with _connect(settings) as conn:
        with conn.cursor() as cur:
            table_ready = _table_exists(cur)
            current_ts = _oldest_stored_load_ts(cur) if table_ready else None

    if current_ts is None:
        current_ts = await _newest_load_ts()
        if current_ts is None:
            print("No rent-contract records available from the DLD API.")
            return
        print(f"First run: starting at newest load_timestamp {current_ts}.")
    else:
        print(f"Resuming at oldest stored load_timestamp {current_ts}.")

    fetched_this_run = 0
    added_this_run = 0
    batches_touched = 0
    columns: list[str] | None = None

    while fetched_this_run < run_budget and batches_touched < MAX_LOAD_TS_PER_RUN:
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                have, min_id = (
                    _batch_progress(cur, current_ts)
                    if _table_exists(cur)
                    else (0, None)
                )

        if have >= per_batch_cap:
            batches_touched += 1
            if batches_touched >= MAX_LOAD_TS_PER_RUN:
                print(f"Batch {current_ts} at cap ({have}/{per_batch_cap}); "
                      f"reached MAX_LOAD_TS_PER_RUN={MAX_LOAD_TS_PER_RUN}.")
                break
            prev = await _newest_load_ts_before(current_ts)
            if prev is None:
                print(f"Backfill complete: reached the oldest batch ({current_ts}).")
                break
            print(f"Batch {current_ts} at cap ({have}/{per_batch_cap}); "
                  f"moving back to {prev}.")
            current_ts = prev
            continue

        page = await _fetch_page(current_ts, min_id)
        fetched_this_run += len(page)

        if not page:
            batches_touched += 1
            print(f"Batch {current_ts} exhausted ({have} rows).")
            if batches_touched >= MAX_LOAD_TS_PER_RUN:
                break
            prev = await _newest_load_ts_before(current_ts)
            if prev is None:
                print(f"Backfill complete: reached the oldest batch ({current_ts}).")
                break
            current_ts = prev
            continue

        room = per_batch_cap - have
        keep = page[:room]

        if columns is None:
            columns = _collect_columns(keep)
        with _connect(settings) as conn:
            with conn.cursor() as cur:
                _ensure_table(cur, columns)
                before = _batch_progress(cur, current_ts)[0]
                _upsert_records(cur, columns, keep)
                conn.commit()
                after = _batch_progress(cur, current_ts)[0]
        added_this_run += after - before
        print(f"  {current_ts}: +{after - before} (now {after}/{per_batch_cap}); "
              f"run total fetched {fetched_this_run}/{run_budget}.")

        if after >= per_batch_cap:
            batches_touched += 1

    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*), min({_quote_ident(CURSOR)}), "
                        f"max({_quote_ident(CURSOR)}) FROM {_quote_ident(TABLE)};")
            total, mn, mx = cur.fetchone()
    log.info("rent_backfill_run_done", fetched=fetched_this_run,
             added=added_this_run, total=total, batches=batches_touched)
    print(f"\nRun done: fetched {fetched_this_run}, added {added_this_run} new "
          f"across {batches_touched} load-batch(es). "
          f"Table now {total} rows spanning {mn} .. {mx} (load_timestamp).")


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
