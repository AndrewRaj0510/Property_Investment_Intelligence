"""``community_features`` table access - psycopg, not the Supabase ORM.

Holds the OpenStreetMap enrichment for each DLD community (Objective 1.4):
centroid, distance to the nearest rail/metro station, and amenity-POI density
within 1 km. Written by ``scripts/refresh_osm_features.py`` (and the
``ingest_osm`` pipeline node); read by ``db/queries/feature_matrix_dld.py``,
which LEFT JOINs it onto every training row.

**Why psycopg.** The Supabase ORM client needs ``SUPABASE_URL``/``SUPABASE_KEY``,
which this deployment does not have - every DB interaction goes through the
direct Postgres connection (the ``SUPABASE_DB_*`` parts via the Session Pooler).
The previous ORM-backed version of these functions could never execute, which is
why the OSM enrichment existed in code but never reached the database. This
module mirrors the ``ml/persist.py`` pattern: idempotent DDL plus plain SQL over
the same connection the ingestion scripts use.
"""

from __future__ import annotations

import structlog

from config import get_settings
from db.pg import connect as _connect

log = structlog.get_logger(__name__)

TABLE = "community_features"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    community          TEXT PRIMARY KEY,
    centroid_lat       DOUBLE PRECISION,
    centroid_lon       DOUBLE PRECISION,
    centroid_source    TEXT,
    nearest_metro_km   DOUBLE PRECISION,
    nearest_metro_name TEXT,
    poi_density_1km    INTEGER,
    osm_mirror         TEXT,
    osm_refreshed_at   TIMESTAMPTZ DEFAULT NOW()
);
"""

_MIGRATIONS = [
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS osm_mirror TEXT",
]

COLUMNS = [
    "community",
    "centroid_lat",
    "centroid_lon",
    "centroid_source",
    "nearest_metro_km",
    "nearest_metro_name",
    "poi_density_1km",
    "osm_mirror",
]


def ensure_table() -> None:
    """Create ``community_features`` if absent (idempotent).

    Never dropped/recreated: unlike ``feature_matrix`` this is *not* derived from
    data we still hold - refilling it costs hundreds of calls to a rate-limited
    public API, so an existing row is worth keeping.
    """
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
            for stmt in _MIGRATIONS:
                cur.execute(stmt)
        conn.commit()
    log.info("community_features_table_ensured")


def upsert_community_features(rows: list[dict]) -> int:
    """Insert/update one row per community, keyed on ``community``."""
    if not rows:
        return 0
    settings = get_settings()
    placeholders = "(" + ", ".join(["%s"] * len(COLUMNS)) + ")"
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in COLUMNS if c != "community"
    )
    sql = (
        f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) VALUES {placeholders} "
        f"ON CONFLICT (community) DO UPDATE SET {updates}, osm_refreshed_at = NOW()"
    )
    values = [tuple(r.get(c) for c in COLUMNS) for r in rows]
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, values)
        conn.commit()
    log.info("community_features_upserted", rows=len(rows))
    return len(rows)


def all_community_features() -> list[dict]:
    """Return every stored community-feature row as a dict."""
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(COLUMNS)}, osm_refreshed_at FROM {TABLE} "
                "ORDER BY community"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, rec)) for rec in cur.fetchall()]


def enriched_communities(mirror: str | None = None) -> set[str]:
    """Communities that already have a usable OSM row.

    "Usable" means the POI density landed - the metro distance has a legitimate
    sentinel value (``NO_STATION_KM``) that must not be mistaken for a failure,
    whereas a NULL density means the Overpass call never succeeded. Used by the
    refresh script to resume a run that a rate limit interrupted.

    ``mirror`` narrows "usable" to rows measured on THAT endpoint. Mirrors serve
    different planet snapshots, so when building a single-snapshot table a row
    measured elsewhere is not done - it is stale. Without this the sweep sees a
    fully-enriched table and refuses to finish the 23 rows a saturated pinned
    mirror had to skip.
    """
    settings = get_settings()
    sql = f"SELECT community FROM {TABLE} WHERE poi_density_1km IS NOT NULL"
    params: list = []
    if mirror:
        sql += " AND osm_mirror = %s"
        params.append(mirror)
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return {r[0] for r in cur.fetchall()}


def coverage() -> dict:
    """Row counts for the refresh script's summary line.

    ``by_source`` is built by GROUPing rather than by counting a hard-coded list
    of sources: an earlier version named three sources explicitly and silently
    under-reported once the Arabic fallback started producing ``nominatim_ar`` /
    ``nominatim_ar_relaxed`` / ``nominatim_poi`` rows (143 stored, 112 counted).
    A coverage report that omits rows is worse than none - it is the number the
    write-up quotes.
    """
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*), count(*) FILTER (WHERE poi_density_1km IS NOT NULL) "
                f"FROM {TABLE}"
            )
            total, enriched = cur.fetchone()
            cur.execute(
                f"SELECT coalesce(centroid_source, 'unknown'), count(*) "
                f"FROM {TABLE} GROUP BY 1 ORDER BY 2 DESC"
            )
            by_source = {src: n for src, n in cur.fetchall()}
    return {"rows": total, "enriched": enriched, "by_source": by_source}
