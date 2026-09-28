"""Materialize quarterly transacted-price history into a snapshot table.

Recomputes the median registered sale price per square metre per (area,
canonical property type, calendar quarter) from the raw DLD sales table (see
``db/queries/price_history.py``) and replaces ``community_price_history`` with
the fresh set. That table backs the price chart on the results screen.

This exists for the same reason ``refresh_yield_medians`` does: the aggregation
is a seq scan with per-row regex matching and numeric casts over a 117k-row
all-TEXT table. One pass over every bucket measures ~3s against the Supabase
pooler, which is fine weekly and not something to repeat per request for an
optional panel; the snapshot turns the request path into an indexed lookup.

Run offline/periodically (alongside the yield-median refresh and model
training), from the backend/ dir:
    .venv/Scripts/python.exe -m scripts.refresh_price_history
"""

from __future__ import annotations

import sys

import structlog

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import get_settings
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident
from db.queries.price_history import PRICE_HISTORY_TABLE, compute_price_history

log = structlog.get_logger(__name__)


def _create_table_sql() -> str:
    t = _quote_ident(PRICE_HISTORY_TABLE)
    return f"""
    CREATE TABLE IF NOT EXISTS {t} (
        area                 TEXT NOT NULL,
        property_type        TEXT NOT NULL,
        year                 INTEGER NOT NULL,
        quarter              INTEGER NOT NULL,
        period               TEXT NOT NULL,
        median_aed_per_sqm   NUMERIC,
        median_aed_per_sqft  NUMERIC,
        n_sales              INTEGER,
        refreshed_at         TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (area, property_type, year, quarter)
    );
    """


def _create_index_sql() -> str:
    """Lookup index for the request path.

    The read filters on ``lower(trim(area))`` so the index has to be on the same
    expression; an index on the bare column would not be used and the endpoint
    would seq-scan the snapshot instead.
    """
    t = _quote_ident(PRICE_HISTORY_TABLE)
    return (
        f"CREATE INDEX IF NOT EXISTS {PRICE_HISTORY_TABLE}_lookup_idx "
        f"ON {t} (lower(trim(area)), property_type, year);"
    )


def _insert_sql() -> str:
    t = _quote_ident(PRICE_HISTORY_TABLE)
    return f"""
    INSERT INTO {t}
        (area, property_type, year, quarter, period,
         median_aed_per_sqm, median_aed_per_sqft, n_sales)
    VALUES (%(area)s, %(property_type)s, %(year)s, %(quarter)s, %(period)s,
            %(median_aed_per_sqm)s, %(median_aed_per_sqft)s, %(n_sales)s)
    ON CONFLICT (area, property_type, year, quarter) DO UPDATE SET
        period              = EXCLUDED.period,
        median_aed_per_sqm  = EXCLUDED.median_aed_per_sqm,
        median_aed_per_sqft = EXCLUDED.median_aed_per_sqft,
        n_sales             = EXCLUDED.n_sales,
        refreshed_at        = NOW();
    """


def main() -> None:
    settings = get_settings()
    if not (settings.db_parts_configured or settings.supabase_db_url):
        raise SystemExit("No Postgres connection configured in .env.")

    print("Computing quarterly price history over the raw DLD sales table...")
    rows = compute_price_history()
    print(f"Computed {len(rows)} (area, type, quarter) buckets.")

    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(_create_table_sql())
            cur.execute(_create_index_sql())
            cur.execute(f"TRUNCATE {_quote_ident(PRICE_HISTORY_TABLE)};")
            cur.executemany(_insert_sql(), rows)
            conn.commit()
            cur.execute(f"SELECT count(*) FROM {_quote_ident(PRICE_HISTORY_TABLE)};")
            total = cur.fetchone()[0]

    log.info("price_history_refreshed", buckets=total)
    if total == 0:
        print(
            f"{PRICE_HISTORY_TABLE} is empty. No (area, type) has a quarter with "
            "enough registered sales; check that the DLD sales backfill has run."
        )
        return

    print(f"{PRICE_HISTORY_TABLE} refreshed with {total} quarter buckets.")
    with _connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT area, property_type, count(*) AS quarters, "
            f"       min(period), max(period) "
            f"FROM {_quote_ident(PRICE_HISTORY_TABLE)} "
            f"GROUP BY 1, 2 ORDER BY quarters DESC LIMIT 5;"
        )
        print("Deepest series:")
        for area, ptype, quarters, lo, hi in cur.fetchall():
            print(f"  {area} / {ptype}: {quarters} quarters ({lo} to {hi})")


if __name__ == "__main__":
    main()
