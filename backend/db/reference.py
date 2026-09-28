"""Static reference data - real, non-synthetic lookups used by live ingestion.

This holds genuine reference values (not fabricated rows). Currently: the set of
Dubai communities the platform covers, each with its approximate geographic
centroid (latitude, longitude). The live OSM/Overpass client uses these
centroids to query nearest-metro distance and POI density per community.

These are static facts about Dubai geography, independent of any data source.
"""

from __future__ import annotations

COMMUNITY_CENTROIDS: dict[str, tuple[float, float]] = {
    "Dubai Marina": (25.080, 55.140),
    "Downtown Dubai": (25.197, 55.274),
    "Jumeirah Village Circle": (25.063, 55.207),
    "Business Bay": (25.186, 55.262),
    "Palm Jumeirah": (25.112, 55.139),
    "Dubai Hills Estate": (25.103, 55.247),
    "Jumeirah Lake Towers": (25.069, 55.143),
    "Arabian Ranches": (25.052, 55.270),
    "Dubai Sports City": (25.038, 55.218),
    "International City": (25.162, 55.408),
    "Mirdif": (25.217, 55.420),
    "Town Square": (25.000, 55.282),
    "Dubai Creek Harbour": (25.197, 55.350),
    "Al Furjan": (25.027, 55.145),
    "Dubai Silicon Oasis": (25.122, 55.378),
}

COMMUNITIES: list[str] = list(COMMUNITY_CENTROIDS.keys())
