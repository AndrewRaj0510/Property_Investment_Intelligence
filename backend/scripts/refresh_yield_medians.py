"""Materialize the sales↔rents gross-yield medians into a baseline table.

Recomputes gross rental yield per (area, property_type) from the raw DLD sales
and Ejari rent tables (see ``db/queries/yield_agg.py``) and replaces the
``community_yield_medians`` table with the fresh set. This baseline is what the
live-scoring verdict compares a listing's predicted yield against ("does it beat
the area+type median?").

Run offline/periodically (alongside model training), from the backend/ dir:
    .venv/Scripts/python.exe -m scripts.refresh_yield_medians

STAGING NOTE: staging's rents have scrambled area/type, so the join matches zero
buckets and the table is emptied - that is correct and expected. It fills only
against production Ejari data.
"""

from __future__ import annotations

import sys

import structlog

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import get_settings
from db.queries.yield_agg import (
    DISTRIBUTION_COLUMNS,
    MEDIANS_TABLE,
    compute_yield_medians,
)
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident

log = structlog.get_logger(__name__)


def _create_table_sql() -> str:
    t = _quote_ident(MEDIANS_TABLE)
    return f"""
    CREATE TABLE IF NOT EXISTS {t} (
        area                TEXT NOT NULL,
        property_type       TEXT NOT NULL,
        median_sale_price   NUMERIC,
        median_annual_rent  NUMERIC,
        gross_yield         NUMERIC,
        n_sales             INTEGER,
        n_rents             INTEGER,
        low_sample          BOOLEAN,
        p25_sale_price      NUMERIC,
        p75_sale_price      NUMERIC,
        p90_sale_price      NUMERIC,
        max_sale_price      NUMERIC,
        p25_annual_rent     NUMERIC,
        p75_annual_rent     NUMERIC,
        p90_annual_rent     NUMERIC,
        max_annual_rent     NUMERIC,
        refreshed_at        TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (area, property_type)
    );
    """


def _migrate_sql() -> list[str]:
    """Bring a table created before the distribution columns up to date.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    on any database where this script has run before, the new columns arrive
    only through an ALTER. Each one is IF NOT EXISTS, so running the refresh
    repeatedly is still safe.
    """
    t = _quote_ident(MEDIANS_TABLE)
    return [
        f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS {col} NUMERIC;"
        for col in DISTRIBUTION_COLUMNS
    ]


def _insert_sql() -> str:
    t = _quote_ident(MEDIANS_TABLE)
    extra_cols = ", ".join(DISTRIBUTION_COLUMNS)
    extra_vals = ", ".join(f"%({c})s" for c in DISTRIBUTION_COLUMNS)
    extra_set = ",\n        ".join(
        f"{c} = EXCLUDED.{c}" for c in DISTRIBUTION_COLUMNS
    )
    return f"""
    INSERT INTO {t}
        (area, property_type, median_sale_price, median_annual_rent,
         gross_yield, n_sales, n_rents, low_sample, {extra_cols})
    VALUES (%(area)s, %(property_type)s, %(median_sale_price)s,
            %(median_annual_rent)s, %(gross_yield)s, %(n_sales)s,
            %(n_rents)s, %(low_sample)s, {extra_vals})
    ON CONFLICT (area, property_type) DO UPDATE SET
        median_sale_price  = EXCLUDED.median_sale_price,
        median_annual_rent = EXCLUDED.median_annual_rent,
        gross_yield        = EXCLUDED.gross_yield,
        n_sales            = EXCLUDED.n_sales,
        n_rents            = EXCLUDED.n_rents,
        low_sample         = EXCLUDED.low_sample,
        {extra_set},
        refreshed_at       = NOW();
    """


def main() -> None:
    settings = get_settings()
    if not (settings.db_parts_configured or settings.supabase_db_url):
        raise SystemExit("No Postgres connection configured in .env.")

    rows = compute_yield_medians()
    print(f"Computed {len(rows)} (area, property_type) yield buckets.")

    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(_create_table_sql())
            for stmt in _migrate_sql():
                cur.execute(stmt)
            cur.execute(f"TRUNCATE {_quote_ident(MEDIANS_TABLE)};")
            ins = _insert_sql()
            for r in rows:
                cur.execute(ins, r)
            conn.commit()
            cur.execute(f"SELECT count(*) FROM {_quote_ident(MEDIANS_TABLE)};")
            total = cur.fetchone()[0]

    log.info("yield_medians_refreshed", buckets=total)
    if total == 0:
        print(
            f"{MEDIANS_TABLE} is empty. Expected on staging (rents have scrambled "
            "area/type → 0 join matches). Fills against production Ejari data."
        )
    else:
        print(f"{MEDIANS_TABLE} refreshed with {total} buckets.")
        with _connect(settings) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT area, property_type, round(gross_yield*100, 2) "
                f"FROM {_quote_ident(MEDIANS_TABLE)} "
                f"ORDER BY gross_yield DESC LIMIT 5;"
            )
            print("Top yields (%):")
            for a, p, y in cur.fetchall():
                print(f"  {a} / {p}: {y}%")


if __name__ == "__main__":
    main()
