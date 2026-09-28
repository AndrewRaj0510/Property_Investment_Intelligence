"""Marketing community name → DLD ``area_name_en`` resolution (F21).

**The problem.** Property portals label listings with *marketing* community names
("JVC District 14", "Dubai Marina", "Business Bay"), while DLD records them under
official sub-community names ("Al Barsha South Fourth", "Marsa Dubai"). The
scorer looks up its baseline by DLD area, so an unresolved name means no
baseline, and the listing is honestly-but-uselessly scored "Insufficient data".
A hand-written alias table previously covered 7 names against 127 DLD areas.

**The fix - derive the mapping from the data, don't assert it.** The raw DLD
transactions carry BOTH names on every row: ``master_project_en`` is the
marketing development, ``area_name_en`` is the official area. Grouping one by
the other yields a verified mapping - e.g. *Jumeirah Village Circle →
Al Barsha South Fourth* is backed by 3,305 real transactions, not by someone's
recollection of Dubai geography.

Where a master project spans several areas (they can), the area with the most
transactions wins, and the runner-up count is kept so the ambiguity is visible
rather than silently resolved.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

ALIAS_TABLE = "community_aliases"
SALES_TABLE = "dld_transactions_raw"

MIN_ALIAS_SAMPLE = 25


def compute_aliases(min_sample: int = MIN_ALIAS_SAMPLE) -> list[dict]:
    """Derive marketing-name → DLD-area pairs from the raw transactions.

    Returns one row per master project: its dominant area, how many transactions
    support that pairing, and how many fell in other areas (0 = unambiguous).
    """
    from config import get_settings
    from db.pg import connect as _connect
    from db.pg import quote_ident as _quote_ident

    sql = f"""
        WITH pairs AS (
            SELECT
                btrim(master_project_en) AS alias,
                btrim(area_name_en)      AS area,
                count(*)                 AS n
            FROM {_quote_ident(SALES_TABLE)}
            WHERE master_project_en IS NOT NULL
              AND btrim(master_project_en) <> ''
              AND area_name_en IS NOT NULL
              AND btrim(area_name_en) <> ''
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT
                alias, area, n,
                row_number() OVER (PARTITION BY alias ORDER BY n DESC) AS rk,
                sum(n)       OVER (PARTITION BY alias)                 AS total
            FROM pairs
        )
        SELECT alias, area, n, (total - n) AS n_other
        FROM ranked
        WHERE rk = 1 AND n >= %(min_sample)s
        ORDER BY n DESC
    """
    with _connect(get_settings()) as conn, conn.cursor() as cur:
        cur.execute(sql, {"min_sample": min_sample})
        rows = cur.fetchall()
    return [
        {"alias": a, "area": ar, "n_transactions": int(n), "n_other_areas": int(o)}
        for a, ar, n, o in rows
    ]


def load_aliases() -> dict[str, str]:
    """Read the materialized alias table as ``{normalized alias: DLD area}``.

    Returns an empty dict when the table is absent (it is built offline by
    ``scripts/refresh_community_aliases.py``) so resolution degrades to the
    static map rather than breaking scoring.
    """
    from config import get_settings
    from db.pg import connect as _connect
    from db.pg import quote_ident as _quote_ident

    try:
        with _connect(get_settings()) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT alias, area FROM {_quote_ident(ALIAS_TABLE)}")
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - table missing / DB unreachable
        log.warning("community_aliases_unavailable", error=str(exc))
        return {}
    return {str(a).strip().lower(): str(ar) for a, ar in rows if a and ar}
