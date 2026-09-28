"""Refresh the OpenStreetMap enrichment for every DLD community (Objective 1.4).

For each community in the DLD sales table: resolve a centroid, then query
Overpass for the distance to the nearest rail/metro station and the count of
amenity POIs within 1 km, and store the result in ``community_features``. The
training feature matrix LEFT JOINs that table, so a community enriched here
becomes two extra model features for every one of its training rows.

Run from the backend/ directory (so .env and imports resolve):

    .venv/Scripts/python.exe -m scripts.refresh_osm_features
    .venv/Scripts/python.exe -m scripts.refresh_osm_features --limit 10
    .venv/Scripts/python.exe -m scripts.refresh_osm_features --force

**Resumable by design, because the public OSM APIs are the bottleneck.** Each
community costs one Nominatim call (only if it has no hand-verified centroid)
plus two Overpass calls, all against shared, rate-limited, frequently-saturated
endpoints. So the script skips communities already enriched unless ``--force``,
commits after every community rather than at the end, and treats a per-community
failure as a skip (logged, counted, reported) rather than an abort. Re-run it
until the coverage line stops moving.

Requires: ``SUPABASE_DB_*`` in .env and outbound internet. The DLD API is *not*
needed - this reads community names already stored locally.
"""

from __future__ import annotations

import argparse
import sys
import time

import structlog

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import get_settings
from db.queries.community_features import (
    all_community_features,
    coverage,
    ensure_table,
    enriched_communities,
    upsert_community_features,
)
from ingestion import osm_client
from ingestion.osm_client import (
    NOMINATIM_DELAY_S,
    NominatimError,
    OverpassError,
    _haversine_km,
    geocode_community,
    get_location_features,
    last_mirror,
    pin_mirror,
)
from db.pg import connect as _connect

log = structlog.get_logger(__name__)

SALES_TABLE = "dld_transactions_raw"

_MOVED_KM = 0.1


def _communities() -> list[tuple[str, str | None]]:
    """Distinct ``(area_name_en, area_name_ar)`` pairs in the DLD sales table.

    The sales side is the correct source: the feature matrix is built by joining
    rents *onto* sales, so a community absent from sales can never appear in a
    training row and enriching it would waste API budget.

    The Arabic name comes along because it geocodes what the English cannot -
    DLD and OSM transliterate differently (see ``geocode_community``). ``mode()``
    picks the dominant Arabic spelling per area in case the raw rows disagree.
    """
    settings = get_settings()
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT area_name_en, "
                f"       mode() WITHIN GROUP (ORDER BY NULLIF(trim(area_name_ar), '')) "
                f"FROM {SALES_TABLE} "
                "WHERE area_name_en IS NOT NULL AND trim(area_name_en) <> '' "
                "GROUP BY area_name_en "
                "ORDER BY area_name_en"
            )
            return [(r[0], r[1]) for r in cur.fetchall()]


def recheck_centroids(limit: int | None = None) -> dict:
    """Re-geocode stored communities; re-query Overpass only where the point moved.

    The cheap correction path. Geocoding is ~1s per community and free of the
    Overpass queue, so re-validating every stored centroid costs minutes, while a
    full ``--force`` sweep costs hours. Overpass is only re-hit for communities
    whose centroid actually changed - which is the only case where the metro
    distance and POI count can differ.

    Added after the ``_is_place_like`` fix: rows written before it may sit on a
    POI (e.g. Al Garhoud's centroid landed on the metro station, giving a
    circular 0.0 km). This re-points them without redoing the whole sweep.
    """
    ensure_table()
    stored = all_community_features()
    if limit:
        stored = stored[:limit]
    arabic = dict(_communities())
    print(f"re-checking {len(stored)} stored centroids\n", flush=True)

    moved, unchanged, failed = 0, 0, []
    for i, row in enumerate(stored, 1):
        community = row["community"]
        try:
            lat, lon, source = geocode_community(community, arabic.get(community))
            time.sleep(NOMINATIM_DELAY_S)
        except NominatimError as exc:
            failed.append((community, f"NominatimError: {exc}"))
            continue

        old = (row.get("centroid_lat"), row.get("centroid_lon"))
        if old[0] is not None and _haversine_km(lat, lon, old[0], old[1]) < _MOVED_KM:
            unchanged += 1
            continue

        try:
            metro_km, metro_name, poi = get_location_features(lat, lon)
        except OverpassError as exc:
            failed.append((community, f"OverpassError: {exc}"))
            print(f"[{i}/{len(stored)}] {community}: MOVED but Overpass failed",
                  flush=True)
            continue

        upsert_community_features([{
            "community": community,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "centroid_source": source,
            "nearest_metro_km": metro_km,
            "nearest_metro_name": metro_name,
            "poi_density_1km": poi,
            "osm_mirror": last_mirror(),
        }])
        moved += 1
        print(f"[{i}/{len(stored)}] {community}: CORRECTED "
              f"{row.get('nearest_metro_km')} -> {metro_km} km, "
              f"poi {row.get('poi_density_1km')} -> {poi} [{source}]", flush=True)

    print(f"\nCorrected: {moved}   unchanged: {unchanged}   failed: {len(failed)}")
    for community, reason in failed[:20]:
        print(f"  - {community}: {reason[:110]}")
    return {"corrected": moved, "unchanged": unchanged, "failed": failed}


def refresh(limit: int | None = None, force: bool = False) -> dict:
    """Enrich communities that need it; return a summary dict."""
    ensure_table()
    todo = _communities()
    if not force:
        done = enriched_communities(mirror=osm_client._pinned_mirror)
        todo = [(en, ar) for en, ar in todo if en not in done]
    if limit:
        todo = todo[:limit]

    print(f"{len(todo)} communities to enrich"
          f"{' (--force: re-doing all)' if force else ''}\n", flush=True)

    ok, failed = 0, []
    for i, (community, community_ar) in enumerate(todo, 1):
        try:
            lat, lon, source = geocode_community(community, community_ar)
            if source.startswith("nominatim"):
                time.sleep(NOMINATIM_DELAY_S)
            metro_km, metro_name, poi = get_location_features(lat, lon)
        except (NominatimError, OverpassError) as exc:
            failed.append((community, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(todo)}] {community}: SKIPPED ({type(exc).__name__})",
                  flush=True)
            continue

        upsert_community_features([{
            "community": community,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "centroid_source": source,
            "nearest_metro_km": metro_km,
            "nearest_metro_name": metro_name,
            "poi_density_1km": poi,
            "osm_mirror": last_mirror(),
        }])
        ok += 1
        print(f"[{i}/{len(todo)}] {community}: metro {metro_km} km "
              f"({metro_name or 'none in range'}), {poi} POIs/km2 [{source}]",
              flush=True)

    cov = coverage()
    print(f"\nEnriched this run: {ok}   skipped: {len(failed)}")
    if failed:
        print("Skipped (re-run to retry - most failures are transient Overpass "
              "saturation):")
        for community, reason in failed[:20]:
            print(f"  - {community}: {reason[:110]}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    print(f"\nCoverage: {cov['enriched']}/{cov['rows']} stored rows enriched")
    print("Centroid provenance:")
    for source, n in cov["by_source"].items():
        print(f"  {source:<24} {n}")
    return {"enriched": ok, "failed": failed, "coverage": cov}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N communities that still need enrichment",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-query communities that already have OSM data",
    )
    parser.add_argument(
        "--recheck-centroids", action="store_true",
        help="re-geocode stored rows; re-query Overpass only where the point moved",
    )
    parser.add_argument(
        "--mirror", default=None,
        help="pin Overpass to ONE endpoint so the whole table comes from a single "
             "planet snapshot (mirrors disagree - see osm_client.pin_mirror). "
             "Use the full interpreter URL.",
    )
    args = parser.parse_args()
    if args.mirror:
        pin_mirror(args.mirror)
    if args.recheck_centroids:
        recheck_centroids(limit=args.limit)
    else:
        refresh(limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
