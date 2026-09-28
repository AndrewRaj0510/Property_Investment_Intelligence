"""Sales↔rents gross-yield aggregation over the raw DLD tables.

Computes **gross rental yield** per ``area_name_en`` + property type by joining
the two raw DLD tables the backfill scripts populate:

    * ``dld_transactions_raw``    - sales (median sale price)
    * ``dld_rent_contracts_raw``  - Ejari rents (median annual rent)

    gross_yield = median(annual_rent) / median(sale_price)

Medians (``percentile_cont(0.5)``) are used rather than means so a few luxury
outliers don't distort a community's baseline. The output - one row per
(area, property type) with its median yield - is the **"area median" baseline**
the live-scoring verdict compares a listing against ("does this listing beat the
area+type median yield?").

All columns in the raw tables are TEXT (created dynamically from the source
schema), so the SQL casts the numeric fields defensively: only rows whose
price/rent/area parse as positive numbers are aggregated.

These raw tables are managed by the ingestion scripts via psycopg (not the
Supabase ORM client), so the aggregation runs as raw SQL over the same
connection, mirroring ``scripts/pull_and_store_dld.py``.

STAGING NOTE: on staging the rents table has fully scrambled ``area_name_en`` /
property-type values (random strings), so the join matches **zero** rows there.
The SQL is correct; it produces real medians only once the rents table holds
production Ejari data whose areas overlap the sales areas.
"""

from __future__ import annotations

import structlog

from config import get_settings
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident

log = structlog.get_logger(__name__)

SALES_TABLE = "dld_transactions_raw"
RENTS_TABLE = "dld_rent_contracts_raw"
MEDIANS_TABLE = "community_yield_medians"

_POSITIVE_NUMBER = r"^[0-9]+(\.[0-9]+)?$"

_MIN_YIELD = 0.01
_MAX_YIELD = 0.20

DISTRIBUTION_COLUMNS = (
    "p25_sale_price",
    "p75_sale_price",
    "p90_sale_price",
    "max_sale_price",
    "p25_annual_rent",
    "p75_annual_rent",
    "p90_annual_rent",
    "max_annual_rent",
)

DEFAULT_MIN_SAMPLE = 3


_SALES_TYPE_CASE = """
    CASE
        WHEN lower(trim(coalesce(property_sub_type_en,''))) LIKE 'hotel%%'
            THEN 'hotel_apartment'
        WHEN lower(trim(coalesce(property_sub_type_en,''))) = 'penthouse'
            THEN 'penthouse'
        WHEN lower(trim(coalesce(property_sub_type_en,''))) = 'studio'
            THEN 'studio'
        WHEN lower(trim(property_type_en)) = 'villa'          THEN 'villa'
        WHEN lower(trim(property_type_en)) = 'building'        THEN 'building'
        WHEN lower(trim(property_type_en)) = 'unit'            THEN 'apartment'
        ELSE lower(trim(property_type_en))
    END
"""

_RENTS_TYPE_CASE = """
    CASE
        WHEN lower(trim(ejari_property_type_en)) IN ('flat')            THEN 'apartment'
        WHEN lower(trim(ejari_property_type_en)) = 'studio'             THEN 'studio'
        WHEN lower(trim(ejari_property_type_en)) IN
                ('villa','complex villas','arabian house')             THEN 'villa'
        WHEN lower(trim(ejari_property_type_en)) = 'hotel apartments'   THEN 'hotel_apartment'
        WHEN lower(trim(ejari_property_type_en)) = 'penthouse'          THEN 'penthouse'
        WHEN lower(trim(ejari_property_type_en)) = 'building'           THEN 'building'
        ELSE lower(trim(ejari_property_type_en))
    END
"""


def _sales_median_cte() -> str:
    """CTE: sale price distribution + sample size per (area, canonical type).

    The median is the baseline the verdict compares against. The quartiles, the
    90th percentile and the outright maximum exist so the grounded Q&A can
    answer "what is the most expensive stock here?" from recorded transactions
    instead of refusing, and they cost nothing extra: the scan over this bucket
    was already happening.
    """
    return f"""
    sales AS (
        SELECT
            area_name_en                              AS area,
            ({_SALES_TYPE_CASE})                       AS property_type,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY actual_worth::numeric
            )                                          AS median_sale_price,
            percentile_cont(0.25) WITHIN GROUP (
                ORDER BY actual_worth::numeric
            )                                          AS p25_sale_price,
            percentile_cont(0.75) WITHIN GROUP (
                ORDER BY actual_worth::numeric
            )                                          AS p75_sale_price,
            percentile_cont(0.90) WITHIN GROUP (
                ORDER BY actual_worth::numeric
            )                                          AS p90_sale_price,
            max(actual_worth::numeric)                 AS max_sale_price,
            count(*)                                   AS n_sales
        FROM {_quote_ident(SALES_TABLE)}
        WHERE area_name_en IS NOT NULL
          AND property_type_en IS NOT NULL
          AND actual_worth ~ %(num)s
          AND actual_worth::numeric > 0
        GROUP BY 1, 2
    )"""


def _rents_median_cte() -> str:
    """CTE: annual rent distribution + sample size per (area, canonical type).

    Same reasoning as the sales CTE. "What is the highest rent achieved in this
    community?" is one of the first things a reader asks of a rental yield tool,
    and a median alone cannot answer it.

    NOTE the maximum is a RAW maximum over Ejari contracts: a whole-floor or
    bulk lease registered against the community lands in it, so it is labelled
    as an outlier everywhere it is surfaced rather than quietly winsorised here.
    """
    return f"""
    rents AS (
        SELECT
            area_name_en                              AS area,
            ({_RENTS_TYPE_CASE})                       AS property_type,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY annual_amount::numeric
            )                                          AS median_annual_rent,
            percentile_cont(0.25) WITHIN GROUP (
                ORDER BY annual_amount::numeric
            )                                          AS p25_annual_rent,
            percentile_cont(0.75) WITHIN GROUP (
                ORDER BY annual_amount::numeric
            )                                          AS p75_annual_rent,
            percentile_cont(0.90) WITHIN GROUP (
                ORDER BY annual_amount::numeric
            )                                          AS p90_annual_rent,
            max(annual_amount::numeric)                AS max_annual_rent,
            count(*)                                   AS n_rents
        FROM {_quote_ident(RENTS_TABLE)}
        WHERE area_name_en IS NOT NULL
          AND ejari_property_type_en IS NOT NULL
          AND annual_amount ~ %(num)s
          AND annual_amount::numeric > 0
        GROUP BY 1, 2
    )"""


def _yield_select() -> str:
    """Join the two medians and compute gross yield per (area, property_type).

    Excludes the ``building`` canonical type - a whole-building sale price and a
    whole-building rent are not a per-unit comparison, so their "yield" is
    meaningless for scoring an apartment/villa listing. Also clamps to a sane
    gross-yield band (``_MIN_YIELD``-``_MAX_YIELD``): values outside it are data
    artefacts (e.g. plot/partial sales with a tiny price, or a bulk lease), not
    real per-unit yields.
    """
    return f"""
    WITH
    {_sales_median_cte()},
    {_rents_median_cte()}
    SELECT
        s.area,
        s.property_type,
        s.median_sale_price,
        r.median_annual_rent,
        (r.median_annual_rent / s.median_sale_price)     AS gross_yield,
        s.n_sales,
        r.n_rents,
        s.p25_sale_price,
        s.p75_sale_price,
        s.p90_sale_price,
        s.max_sale_price,
        r.p25_annual_rent,
        r.p75_annual_rent,
        r.p90_annual_rent,
        r.max_annual_rent
    FROM sales s
    JOIN rents r
      ON r.area = s.area
     AND r.property_type = s.property_type
    WHERE s.median_sale_price > 0
      AND s.property_type <> 'building'
      AND (r.median_annual_rent / s.median_sale_price) BETWEEN {_MIN_YIELD} AND {_MAX_YIELD}
    ORDER BY s.area, s.property_type
    """


def compute_yield_medians(min_sample: int = DEFAULT_MIN_SAMPLE) -> list[dict]:
    """Return one dict per (area, property_type) with its gross-yield median.

    Each row: ``area``, ``property_type``, ``median_sale_price``,
    ``median_annual_rent``, ``gross_yield``, ``n_sales``, ``n_rents``, and a
    ``low_sample`` flag (True when either side has fewer than ``min_sample``
    rows). Returns ``[]`` when the two tables share no (area, property_type) -
    the expected result against staging's scrambled rents.
    """
    settings = get_settings()
    rows: list[dict] = []
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(_yield_select(), {"num": _POSITIVE_NUMBER})
            cols = [d.name for d in cur.description]
            for raw in cur.fetchall():
                rec = dict(zip(cols, raw))
                row = {
                    "area": rec["area"],
                    "property_type": rec["property_type"],
                    "median_sale_price": float(rec["median_sale_price"]),
                    "median_annual_rent": float(rec["median_annual_rent"]),
                    "gross_yield": float(rec["gross_yield"]),
                    "n_sales": int(rec["n_sales"]),
                    "n_rents": int(rec["n_rents"]),
                    "low_sample": rec["n_sales"] < min_sample
                    or rec["n_rents"] < min_sample,
                }
                for col in DISTRIBUTION_COLUMNS:
                    val = rec.get(col)
                    row[col] = None if val is None else float(val)
                rows.append(row)
    log.info("yield_medians_computed", buckets=len(rows), min_sample=min_sample)
    return rows


def area_type_median_yield(area: str, property_type: str) -> float | None:
    """Look up the gross-yield median for one (area, property_type), or None.

    The baseline the live-scoring verdict compares a listing's predicted yield
    against. Matching is case-insensitive on a trimmed property type, mirroring
    the aggregation.
    """
    settings = get_settings()
    sql = _yield_select().rstrip().rstrip(";")
    wrapped = (
        f"SELECT gross_yield FROM ({sql}) q "
        f"WHERE q.area = %(area)s AND q.property_type = %(ptype)s LIMIT 1"
    )
    params = {
        "num": _POSITIVE_NUMBER,
        "area": area,
        "ptype": (property_type or "").strip().lower(),
    }
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(wrapped, params)
            row = cur.fetchone()
    return float(row[0]) if row else None
