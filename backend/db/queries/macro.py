"""Macro time-series access (Objective 3.1) - psycopg, not the Supabase ORM.

**Why this module was rewritten.** Every function here previously went through
``get_db()`` - the Supabase ORM client, which needs ``SUPABASE_URL``/
``SUPABASE_KEY`` that this deployment does not have (**F6**). The calls could
therefore never execute. That is the actual reason Objective 3.1 showed a
"World Bank client written but never run": the fetch worked, but it had nowhere
to land. Identical defect, and identical fix, to the OSM enrichment before
``community_features.py`` was moved to psycopg.

**Two tables, because the sources have two different true grains.**

* ``macro_timeseries`` - **monthly**. EIBOR (mean of daily observations), Brent
  (EIA monthly spot), and Dubai weather (means of daily ERA5 observations, plus
  the thermal discomfort index). These are genuinely monthly series.
* ``macro_annual`` - **yearly**. World Bank GDP growth, inflation and CPI index
  are published annually, plus annual aggregates of EIBOR and Brent. This is the
  grain the Phase-1 feature matrix joins on, since that matrix is built at
  (community × property type × bedrooms × **year**).

Splitting them is the point, not bureaucracy. The previous design wrote annual
World Bank values into a monthly table by repeating each value twelve times,
which manufactured eleven-twelfths of its own rows and made an annual series
indistinguishable from a measured monthly one. Every aggregate stored here
carries the observation count behind it, so a mean over four trading days is
never mistaken for a mean over twenty-two.
"""

from __future__ import annotations

import structlog

from config import get_settings
from db.pg import connect as _connect

log = structlog.get_logger(__name__)

MONTHLY_TABLE = "macro_timeseries"
ANNUAL_TABLE = "macro_annual"

_MONTHLY_DDL = f"""
CREATE TABLE IF NOT EXISTS {MONTHLY_TABLE} (
    month            DATE PRIMARY KEY,
    eibor_3m         NUMERIC,
    uae_gdp_growth   NUMERIC,
    uae_cpi          NUMERIC,
    brent_crude_usd  NUMERIC,
    temp_max_c       NUMERIC,
    humidity_pct     NUMERIC,
    thermal_index    NUMERIC,
    tourist_arrivals INTEGER,
    hotel_occupancy  NUMERIC,
    ramadan_flag     BOOLEAN DEFAULT FALSE,
    expo_dummy       BOOLEAN DEFAULT FALSE,
    ingested_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_ANNUAL_DDL = f"""
CREATE TABLE IF NOT EXISTS {ANNUAL_TABLE} (
    year               INTEGER PRIMARY KEY,
    temp_max_c         NUMERIC,
    humidity_pct       NUMERIC,
    thermal_index      NUMERIC,
    weather_n_days     INTEGER,
    weather_partial    BOOLEAN,
    eibor_3m           NUMERIC,
    eibor_3m_n_days    INTEGER,
    eibor_3m_n_months  INTEGER,
    eibor_partial      BOOLEAN,
    brent_crude_usd    NUMERIC,
    brent_n_months     INTEGER,
    brent_partial      BOOLEAN,
    uae_gdp_growth     NUMERIC,
    uae_inflation_pct  NUMERIC,
    uae_cpi_index      NUMERIC,
    ingested_at        TIMESTAMPTZ DEFAULT NOW()
);
"""

_MIGRATIONS = [
    f"ALTER TABLE {MONTHLY_TABLE} ADD COLUMN IF NOT EXISTS eibor_3m_n_days INTEGER",
    f"ALTER TABLE {MONTHLY_TABLE} ADD COLUMN IF NOT EXISTS weather_n_days INTEGER",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS temp_max_c NUMERIC",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS humidity_pct NUMERIC",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS thermal_index NUMERIC",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS weather_n_days INTEGER",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS weather_partial BOOLEAN",
    f"ALTER TABLE {MONTHLY_TABLE} ADD COLUMN IF NOT EXISTS ramadan_days INTEGER",
    f"ALTER TABLE {MONTHLY_TABLE} ADD COLUMN IF NOT EXISTS covid_period BOOLEAN",
    f"ALTER TABLE {MONTHLY_TABLE} ADD COLUMN IF NOT EXISTS covid_break BOOLEAN",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS ramadan_days INTEGER",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS expo_months INTEGER",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS covid_period_months INTEGER",
    f"ALTER TABLE {ANNUAL_TABLE} ADD COLUMN IF NOT EXISTS covid_break BOOLEAN",
]

MONTHLY_COLUMNS = [
    "month",
    "eibor_3m",
    "eibor_3m_n_days",
    "brent_crude_usd",
    "temp_max_c",
    "humidity_pct",
    "thermal_index",
    "weather_n_days",
    "ramadan_days",
    "ramadan_flag",
    "expo_dummy",
    "covid_period",
    "covid_break",
]

ANNUAL_COLUMNS = [
    "year",
    "eibor_3m",
    "eibor_3m_n_days",
    "eibor_3m_n_months",
    "eibor_partial",
    "brent_crude_usd",
    "brent_n_months",
    "brent_partial",
    "uae_gdp_growth",
    "uae_inflation_pct",
    "uae_cpi_index",
    "temp_max_c",
    "humidity_pct",
    "thermal_index",
    "weather_n_days",
    "weather_partial",
    "ramadan_days",
    "expo_months",
    "covid_period_months",
    "covid_break",
]


def ensure_tables() -> None:
    """Create both macro tables if absent (idempotent)."""
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(_MONTHLY_DDL)
            cur.execute(_ANNUAL_DDL)
            for statement in _MIGRATIONS:
                cur.execute(statement)
        conn.commit()
    log.info("macro_tables_ensured")


def _upsert(table: str, columns: list[str], key: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    settings = get_settings()
    placeholders = "(" + ", ".join(["%s"] * len(columns)) + ")"
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != key)
    sql = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES {placeholders} "
        f"ON CONFLICT ({key}) DO UPDATE SET {updates}, ingested_at = NOW()"
    )
    values = [tuple(r.get(c) for c in columns) for r in rows]
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, values)
        conn.commit()
    return len(rows)


def upsert_macro_monthly(rows: list[dict]) -> int:
    """Insert/update monthly macro rows, keyed on ``month``."""
    n = _upsert(MONTHLY_TABLE, MONTHLY_COLUMNS, "month", rows)
    log.info("macro_monthly_upserted", rows=n)
    return n


def upsert_macro_annual(rows: list[dict]) -> int:
    """Insert/update annual macro rows, keyed on ``year``."""
    n = _upsert(ANNUAL_TABLE, ANNUAL_COLUMNS, "year", rows)
    log.info("macro_annual_upserted", rows=n)
    return n


def all_macro_monthly() -> list[dict]:
    return _select(
        f"SELECT {', '.join(MONTHLY_COLUMNS)} FROM {MONTHLY_TABLE} ORDER BY month"
    )


def all_macro_annual() -> list[dict]:
    return _select(
        f"SELECT {', '.join(ANNUAL_COLUMNS)} FROM {ANNUAL_TABLE} ORDER BY year"
    )


def _select(sql: str) -> list[dict]:
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, rec)) for rec in cur.fetchall()]


def coverage() -> dict:
    """Row counts and ranges for both tables - what the ingest script reports."""
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*), MIN(month), MAX(month) FROM {MONTHLY_TABLE}"
            )
            m_count, m_first, m_last = cur.fetchone()
            cur.execute(
                f"SELECT COUNT(*), MIN(year), MAX(year), "
                f"COUNT(eibor_3m), COUNT(brent_crude_usd), COUNT(uae_gdp_growth), "
                f"COUNT(uae_inflation_pct), COUNT(uae_cpi_index), "
                f"COUNT(thermal_index) FROM {ANNUAL_TABLE}"
            )
            (
                a_count, a_first, a_last,
                n_eibor, n_brent, n_gdp, n_infl, n_cpi, n_thermal,
            ) = cur.fetchone()
    return {
        "monthly_rows": m_count,
        "monthly_first": m_first,
        "monthly_last": m_last,
        "annual_rows": a_count,
        "annual_first": a_first,
        "annual_last": a_last,
        "annual_non_null": {
            "eibor_3m": n_eibor,
            "brent_crude_usd": n_brent,
            "uae_gdp_growth": n_gdp,
            "uae_inflation_pct": n_infl,
            "uae_cpi_index": n_cpi,
            "thermal_index": n_thermal,
        },
    }
