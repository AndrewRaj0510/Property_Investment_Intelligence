"""EXPERIMENTAL feature matrix V2 - bedroom-level grain + per-sqm features.

Isolated experiment (does NOT touch the live Phase-1 builder
``db/queries/feature_matrix_dld.py`` or any persisted table). Goal: test whether
finer granularity + density features push CV R² past ~0.70.

Changes vs the live V1 builder:
  * Grain: **(area × canonical_type × bedrooms × year)** - adds ``bedrooms``.
    Bedrooms is a real column on BOTH sides (no imputation):
      - sales: parsed from ``rooms_en``  ("1 B/R"→1, "Studio"→0).
      - rents: parsed from ``ejari_property_sub_type_en``
               ("1bed room+Hall"→1, "2 bed rooms+hall"→2, "Studio"→0).
  * Features add **price_per_sqm** and **rent_per_sqm** (explicit density signal),
    alongside the V1 features (size, metro, parking, year, sample sizes).
  * Optional **log-target** handled by the runner, not here.

Same honest-target design as V1: real medians on both sides, ``building``
excluded, yield clamped to 1-20 %, min-sample threshold per bucket. Uses the
shared type-normalization from ``yield_agg`` and the psycopg ``_connect`` path.
"""

from __future__ import annotations

import structlog

from config import get_settings
from db.queries.yield_agg import (
    _MAX_YIELD,
    _MIN_YIELD,
    _POSITIVE_NUMBER,
    _RENTS_TYPE_CASE,
    _SALES_TYPE_CASE,
    RENTS_TABLE,
    SALES_TABLE,
)
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident

log = structlog.get_logger(__name__)

MIN_YEAR = 2012
MAX_YEAR = 2026
MIN_SAMPLES = 3

_SALES_BEDROOMS = r"""
    CASE
        WHEN lower(trim(rooms_en)) = 'studio'          THEN 0
        WHEN rooms_en ~ '^[0-9]+ *B/R$'
            THEN split_part(trim(rooms_en), ' ', 1)::int
        ELSE NULL
    END
"""

_RENTS_BEDROOMS = r"""
    CASE
        WHEN lower(trim(ejari_property_sub_type_en)) = 'studio' THEN 0
        WHEN ejari_property_sub_type_en ~ '^[0-9]+'
            THEN (regexp_match(ejari_property_sub_type_en, '^[0-9]+'))[1]::int
        ELSE NULL
    END
"""


def _sales_cte() -> str:
    return f"""
    sales AS (
        SELECT
            area_name_en                                      AS community,
            ({_SALES_TYPE_CASE})                               AS property_type,
            ({_SALES_BEDROOMS})                                AS bedrooms,
            left(instance_date, 4)::int                        AS year,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY actual_worth::numeric)                AS sale_price_median_aed,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY procedure_area::numeric)
                FILTER (WHERE procedure_area ~ %(num)s)        AS floor_area_median_sqm,
            mode() WITHIN GROUP (ORDER BY NULLIF(trim(nearest_metro_en), ''))
                                                               AS nearest_metro,
            avg(CASE WHEN lower(trim(has_parking)) IN ('true','yes','1','y')
                     THEN 1.0 ELSE 0.0 END)                    AS parking_rate,
            count(*)                                           AS n_sales
        FROM {_quote_ident(SALES_TABLE)}
        WHERE area_name_en IS NOT NULL
          AND property_type_en IS NOT NULL
          AND actual_worth ~ %(num)s
          AND actual_worth::numeric > 0
          AND instance_date ~ '^[0-9]{{4}}'
          AND left(instance_date, 4)::int BETWEEN {MIN_YEAR} AND {MAX_YEAR}
          AND ({_SALES_BEDROOMS}) IS NOT NULL
        GROUP BY 1, 2, 3, 4
    )"""


def _rents_cte() -> str:
    return f"""
    rents AS (
        SELECT
            area_name_en                                      AS community,
            ({_RENTS_TYPE_CASE})                               AS property_type,
            ({_RENTS_BEDROOMS})                                AS bedrooms,
            left(contract_start_date, 4)::int                  AS year,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY annual_amount::numeric)               AS annual_rent_median_aed,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY actual_area::numeric)
                FILTER (WHERE actual_area ~ %(num)s)           AS rent_area_median_sqm,
            count(*)                                           AS n_rents
        FROM {_quote_ident(RENTS_TABLE)}
        WHERE area_name_en IS NOT NULL
          AND ejari_property_type_en IS NOT NULL
          AND annual_amount ~ %(num)s
          AND annual_amount::numeric > 0
          AND contract_start_date ~ '^[0-9]{{4}}'
          AND left(contract_start_date, 4)::int BETWEEN {MIN_YEAR} AND {MAX_YEAR}
          AND ({_RENTS_BEDROOMS}) IS NOT NULL
        GROUP BY 1, 2, 3, 4
    )"""


def _matrix_sql() -> str:
    return f"""
    WITH
    {_sales_cte()},
    {_rents_cte()}
    SELECT
        s.community,
        s.property_type,
        s.bedrooms,
        s.year,
        s.sale_price_median_aed,
        r.annual_rent_median_aed,
        round((r.annual_rent_median_aed / s.sale_price_median_aed * 100)::numeric, 3)
                                                             AS gross_yield_lt,
        s.floor_area_median_sqm,
        -- price per sqm (sales), rent per sqm (rents) - explicit density signal.
        CASE WHEN s.floor_area_median_sqm > 0
             THEN round((s.sale_price_median_aed / s.floor_area_median_sqm)::numeric, 2)
        END                                                  AS price_per_sqm,
        CASE WHEN r.rent_area_median_sqm > 0
             THEN round((r.annual_rent_median_aed / r.rent_area_median_sqm)::numeric, 2)
        END                                                  AS rent_per_sqm,
        s.nearest_metro,
        round(s.parking_rate::numeric, 3)                    AS parking_rate,
        s.n_sales,
        r.n_rents
    FROM sales s
    JOIN rents r
      ON  r.community = s.community
      AND r.property_type = s.property_type
      AND r.bedrooms = s.bedrooms
      AND r.year = s.year
    WHERE s.sale_price_median_aed > 0
      AND s.property_type <> 'building'
      AND s.n_sales >= {MIN_SAMPLES}
      AND r.n_rents >= {MIN_SAMPLES}
      AND (r.annual_rent_median_aed / s.sale_price_median_aed)
              BETWEEN {_MIN_YIELD} AND {_MAX_YIELD}
    ORDER BY s.community, s.property_type, s.bedrooms, s.year
    """


_COLUMNS = [
    "community",
    "property_type",
    "bedrooms",
    "year",
    "sale_price_median_aed",
    "annual_rent_median_aed",
    "gross_yield_lt",
    "floor_area_median_sqm",
    "price_per_sqm",
    "rent_per_sqm",
    "nearest_metro",
    "parking_rate",
    "n_sales",
    "n_rents",
]


def build_feature_matrix_v2(feature_version: int = 2) -> list[dict]:
    """Return one dict per (area, type, bedrooms, year) experimental training row."""
    settings = get_settings()
    rows: list[dict] = []
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(_matrix_sql(), {"num": _POSITIVE_NUMBER})
            for rec in cur.fetchall():
                row = dict(zip(_COLUMNS, rec))
                for k in (
                    "sale_price_median_aed",
                    "annual_rent_median_aed",
                    "gross_yield_lt",
                    "floor_area_median_sqm",
                    "price_per_sqm",
                    "rent_per_sqm",
                    "parking_rate",
                ):
                    row[k] = None if row[k] is None else float(row[k])
                row["bedrooms"] = int(row["bedrooms"])
                row["year"] = int(row["year"])
                row["n_sales"] = int(row["n_sales"])
                row["n_rents"] = int(row["n_rents"])
                row["feature_version"] = feature_version
                rows.append(row)
    log.info("v2_feature_matrix_built", rows=len(rows), version=feature_version)
    return rows
