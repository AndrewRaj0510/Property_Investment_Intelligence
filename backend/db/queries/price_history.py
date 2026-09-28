"""Quarterly transacted-price history per (area, canonical property type).

Backs the results-screen price chart: "has this area been getting more or less
expensive?", answered from the DLD sales table the rest of the platform already
trains and scores on.

    median(meter_sale_price) per calendar quarter, per (area, canonical type)

``meter_sale_price`` is DLD's own AED-per-square-metre figure for the sale, so
no division is done here. Medians (``percentile_cont(0.5)``) are used for the
same reason ``yield_agg`` uses them: a handful of penthouse or bulk sales in a
thin quarter would drag a mean around and invent a swing that did not happen.

THIS IS HISTORY, NOT A FORECAST. Every point is a median of sales that were
actually registered in that quarter. Nothing here extrapolates forward, and the
API deliberately exposes no projected point: the project has no price model
(the trained model predicts gross YIELD at annual grain), and the narrative and
Q&A prompts both forbid forecasting prices or capital growth. The series is
labelled in the UI accordingly.

**Materialized, like the yield baseline.** Every column of the raw table is
TEXT, so computing a series means a seq scan with per-row regex matching and
numeric casts over 117k rows. One full pass over EVERY bucket measures ~3s
against the Supabase pooler, which is cheap enough offline and too slow to
repeat on each request for a panel the user is already waiting on. So the same
pattern ``yield_agg`` uses for ``community_yield_medians`` applies here:
``compute_price_history`` does that pass offline
(``scripts/refresh_price_history.py``, 5,059 buckets), and the request path
reads ``community_price_history`` by indexed lookup instead.

Consistency rules shared with ``db/queries/yield_agg.py`` so the chart describes
the same population as the yield baseline shown beside it:

  * the SAME ``_SALES_TYPE_CASE`` normalisation, so "apartment" means the same
    stock in both places;
  * numeric TEXT columns cast defensively (every column in the raw table is TEXT);
  * off-plan and existing sales both included, exactly as the yield baseline
    includes them.

Sample-size honesty: a quarter with fewer than ``MIN_QUARTER_SALES`` registered
sales is DROPPED rather than plotted, and every point carries its own ``n`` so
the UI can show what each median rests on. A median of two sales is not a
market observation.
"""

from __future__ import annotations

import structlog

from config import get_settings
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident
from db.queries.yield_agg import (
    _POSITIVE_NUMBER,
    _SALES_TYPE_CASE,
    SALES_TABLE,
)

log = structlog.get_logger(__name__)

PRICE_HISTORY_TABLE = "community_price_history"

MIN_PRICE_PER_SQM = 1_000
MAX_PRICE_PER_SQM = 250_000

MIN_QUARTER_SALES = 4

MIN_YEAR = 2012

SQFT_PER_SQM = 10.7639


_COVERAGE_TTL_S = 600.0
_coverage_cache: tuple[str | None, float] | None = None


def coverage_end(max_age_s: float = _COVERAGE_TTL_S) -> str | None:
    """The newest ``instance_date`` in the sales table, or None.

    Surfaced so the chart can date its own last point. DLD publishes on a lag,
    so "the latest quarter we hold" and "the latest quarter that has happened"
    are different statements and the UI must not imply the first is the second.

    Cached for ``max_age_s``; pass 0 to force a read.
    """
    import time

    global _coverage_cache
    now = time.monotonic()
    if _coverage_cache is not None and now - _coverage_cache[1] < max_age_s:
        return _coverage_cache[0]

    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT max(instance_date) FROM {_quote_ident(SALES_TABLE)} "
                "WHERE instance_date ~ '^[0-9]{4}-[0-9]{2}'"
            )
            row = cur.fetchone()
    value = row[0] if row and row[0] else None
    _coverage_cache = (value, now)
    return value


def _latest_complete_quarter(cur) -> tuple[int, int]:
    """The last quarter that has FINISHED, as ``(year, quarter)`` exclusive bound.

    The table's newest ``instance_date`` sits mid-quarter (measured: 2026-07-07,
    one week into 2026-Q3), so that final bucket is a part-quarter median
    plotted beside full ones. On Palm Jumeirah apartments it rendered as AED
    3,615/sqft falling to AED 2,529/sqft off FOUR sales, a 30% "crash" that is
    an artefact of the quarter not having happened yet.

    The bound is resolved HERE and passed to the aggregation as a parameter
    rather than joined in as a CTE. Written as ``FROM tx, coverage`` the planner
    inlined the aggregate into the per-row filter and recomputed max() as it
    went: the same aggregation measured ~100s that way and ~3s with the bound
    passed in as two integers.
    """
    cur.execute(
        f"SELECT left(max(instance_date), 4)::int, "
        f"       ceil(substring(max(instance_date), 6, 2)::numeric / 3)::int "
        f"FROM {_quote_ident(SALES_TABLE)} "
        "WHERE instance_date ~ '^[0-9]{4}-[0-9]{2}'"
    )
    row = cur.fetchone()
    if not row or row[0] is None:
        return (MIN_YEAR, 1)
    return int(row[0]), int(row[1])


def _compute_sql() -> str:
    """Every (area, type, quarter) bucket in one pass. Offline use only."""
    return f"""
    WITH tx AS (
        SELECT
            area_name_en                                           AS area,
            ({_SALES_TYPE_CASE})                                   AS property_type,
            left(instance_date, 4)::int                            AS yr,
            ceil(substring(instance_date, 6, 2)::numeric / 3)::int  AS qtr,
            meter_sale_price::numeric                              AS psm
        FROM {_quote_ident(SALES_TABLE)}
        WHERE area_name_en IS NOT NULL
          AND property_type_en IS NOT NULL
          AND meter_sale_price ~ %(num)s
          AND meter_sale_price::numeric
              BETWEEN {MIN_PRICE_PER_SQM} AND {MAX_PRICE_PER_SQM}
          AND instance_date ~ '^[0-9]{{4}}-[0-9]{{2}}'
          AND left(instance_date, 4)::int >= %(min_year)s
    )
    SELECT
        area,
        property_type,
        yr,
        qtr,
        count(*)                                           AS n_sales,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY psm)   AS median_psm
    FROM tx
    WHERE (yr, qtr) < (%(max_yr)s, %(max_qtr)s)
      -- A whole-building sale price per square metre is not a per-unit price,
      -- the same exclusion the yield baseline makes.
      AND property_type <> 'building'
    GROUP BY 1, 2, 3, 4
    HAVING count(*) >= %(min_n)s
    ORDER BY 1, 2, 3, 4
    """


def _row(area: str, ptype: str, yr: int, qtr: int, n_sales: int, psm: float) -> dict:
    return {
        "area": area,
        "property_type": ptype,
        "period": f"{yr}-Q{qtr}",
        "year": yr,
        "quarter": qtr,
        "median_aed_per_sqm": round(psm, 1),
        "median_aed_per_sqft": round(psm / SQFT_PER_SQM, 1),
        "n_sales": n_sales,
    }


def compute_price_history(min_quarter_sales: int = MIN_QUARTER_SALES) -> list[dict]:
    """Every (area, canonical type, quarter) bucket. For the refresh script.

    One pass over the raw sales table, so this is the slow path by design: it
    runs offline alongside the yield-median refresh, never in a request.
    """
    settings = get_settings()
    rows: list[dict] = []
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            max_yr, max_qtr = _latest_complete_quarter(cur)
            cur.execute(
                _compute_sql(),
                {
                    "num": _POSITIVE_NUMBER,
                    "min_year": MIN_YEAR,
                    "max_yr": max_yr,
                    "max_qtr": max_qtr,
                    "min_n": max(int(min_quarter_sales), 1),
                },
            )
            for area, ptype, yr, qtr, n_sales, median_psm in cur.fetchall():
                rows.append(
                    _row(area, ptype, int(yr), int(qtr), int(n_sales), float(median_psm))
                )
    log.info(
        "price_history_computed",
        buckets=len(rows),
        excluded_from=f"{max_yr}-Q{max_qtr}",
    )
    return rows


def quarterly_price_history(
    area: str,
    property_type: str,
    from_year: int = MIN_YEAR,
) -> list[dict]:
    """Quarterly median transacted price for one (area, canonical type).

    Reads the materialized ``community_price_history`` snapshot, which is what
    makes this a request-path query at all (the live aggregation measured ~100s).

    ``area`` must already be resolved to DLD's ``area_name_en`` (the caller does
    this with the same resolver the model and the yield baseline use), and
    ``property_type`` must be a canonical type ("apartment", "villa", ...).

    Returns one dict per quarter, oldest first::

        {"period": "2024-Q1", "year": 2024, "quarter": 1,
         "median_aed_per_sqm": 21971.1, "median_aed_per_sqft": 2041.2,
         "n_sales": 20}

    Returns ``[]`` when the bucket has no quarter in the snapshot. That is a
    real answer, not an error: the UI shows the absence rather than a line drawn
    through two transactions.
    """
    settings = get_settings()
    sql = f"""
        SELECT year, quarter, n_sales, median_aed_per_sqm, median_aed_per_sqft
        FROM {_quote_ident(PRICE_HISTORY_TABLE)}
        WHERE lower(trim(area)) = %(area)s
          AND property_type = %(ptype)s
          AND year >= %(min_year)s
        ORDER BY year, quarter
    """
    params = {
        "area": (area or "").strip().lower(),
        "ptype": (property_type or "").strip().lower(),
        "min_year": max(int(from_year), MIN_YEAR),
    }
    rows: list[dict] = []
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            for yr, qtr, n_sales, psm, psf in cur.fetchall():
                rows.append(
                    {
                        "period": f"{int(yr)}-Q{int(qtr)}",
                        "year": int(yr),
                        "quarter": int(qtr),
                        "median_aed_per_sqm": float(psm),
                        "median_aed_per_sqft": float(psf),
                        "n_sales": int(n_sales),
                    }
                )
    log.info(
        "price_history_read",
        area=params["area"],
        property_type=params["ptype"],
        quarters=len(rows),
    )
    return rows
