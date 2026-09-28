"""OpenStreetMap client - community geocoding + Overpass locational features.

Supplies Objective 1.4: for every Dubai community in the DLD data, the
**distance to the nearest metro/rail station** and the **density of amenity POIs
within a 1 km radius**, both derived from OpenStreetMap.

Two OSM services are used, for two different jobs:

* **Nominatim** (``/search``) geocodes a DLD ``area_name_en`` to a centroid.
  ``db.reference.COMMUNITY_CENTROIDS`` only covers 15 hand-picked communities
  while the DLD sales table spans ~127 areas, so hard-coded centroids would have
  enriched a small minority of training rows - and "has OSM data" would then be
  a proxy for "is a famous community", which is exactly the leak to avoid.
  Hand-verified centroids still win where they exist; Nominatim fills the rest.
* **Overpass** (``/api/interpreter``) runs the two spatial queries.

**Operational notes learned the hard way.** Both services reject requests
without a descriptive ``User-Agent`` - ``httpx``'s default UA gets a bare
``406 Not Acceptable`` from ``overpass-api.de`` (this is why the previous
version of this module could never have succeeded against the live API). Both
also rate-limit aggressively, so callers must pace themselves (see
``scripts/refresh_osm_features.py``) and this module rotates over public
Overpass mirrors when one is saturated.

Live-only: there is no synthetic fallback. On failure the functions raise so the
caller records the gap rather than storing a fabricated distance.
"""

from __future__ import annotations

import math
import re
import time

import httpx
import structlog

from db.reference import COMMUNITY_CENTROIDS

log = structlog.get_logger(__name__)

__all__ = [
    "COMMUNITY_CENTROIDS",
    "NominatimError",
    "OverpassError",
    "geocode_community",
    "get_metro_distance",
    "get_poi_density",
]

OVERPASS_URLS = (
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)

OVERPASS_ATTEMPTS = 2

_PROBE_LAT, _PROBE_LON = 25.1972, 55.2744
_PROBE_QUERY = (
    f'[out:json][timeout:60];node["amenity"]'
    f"(around:1000,{_PROBE_LAT},{_PROBE_LON});out count;"
)
_planet_verified: dict[str, bool] = {}

_rotation = 0

_cooldown_until: dict[str, float] = {}
MIRROR_COOLDOWN_S = 600.0


_pinned_mirror: str | None = None


def pin_mirror(url: str | None) -> None:
    """Restrict (or release) Overpass to a single endpoint for this process."""
    global _pinned_mirror
    _pinned_mirror = url
    if url:
        log.info("overpass_mirror_pinned", url=url)


def _endpoints() -> tuple[str, ...]:
    return (_pinned_mirror,) if _pinned_mirror else OVERPASS_URLS


_last_mirror: str | None = None


def last_mirror() -> str | None:
    """The Overpass endpoint that served the most recent successful query."""
    return _last_mirror


def _available(url: str) -> bool:
    return time.monotonic() >= _cooldown_until.get(url, 0.0)


def _benched(url: str) -> None:
    _cooldown_until[url] = time.monotonic() + MIRROR_COOLDOWN_S

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

_USER_AGENT = (
    "PropertyInvestmentIntelligence/1.0 "
    "(MSc dissertation, Middlesex University Dubai)"
)
_HEADERS = {"User-Agent": _USER_AGENT}

METRO_SEARCH_RADIUS_M = 15000
NO_STATION_KM = 99.0

POI_RADIUS_M = 1000

OVERPASS_SERVER_TIMEOUT_S = 60

_TIMEOUT_S = 45.0
_RETRY_DELAY_S = 5.0
NOMINATIM_DELAY_S = 1.1


class OverpassError(RuntimeError):
    """Raised when every Overpass endpoint failed for a query."""


class NominatimError(RuntimeError):
    """Raised when a community could not be geocoded."""


_STOP_TOKENS = {
    "al", "el", "um", "umm", "abu", "bu", "wadi", "nad", "hor", "the",
    "ال", "أم", "ام", "ابو", "أبو", "وادي", "ند",
}


def _is_meaningful(rung: str) -> bool:
    """Whether a search string names somewhere, rather than being all articles."""
    return any(w.lower() not in _STOP_TOKENS for w in rung.split())


def _query_ladder(community: str) -> list[str]:
    """Progressively broader search strings for one DLD area name.

    Drops trailing words one at a time, keeping any rung that still names
    somewhere (see ``_is_meaningful``).

    Two bugs shaped this function, both found by running it:

    1. It originally dropped only ONE trailing word, which looked right for
       "Al Barsha South Fourth" but for deeper names made the ordinal strip and
       the word drop produce the SAME string - so "Al Goze Industrial First"
       never tried "Al Goze". Eight of the first forty communities were lost.
    2. The fix used a blunt TWO-WORD floor to avoid collapsing to "Al". That
       protected against a real hazard but over-blocked: "Jumeirah First",
       "Warsan Fourth", "Muhaisanah Second" and eleven others are two-word names
       whose parent is a single substantive word ("Jumeirah", "Warsan",
       "Muhaisanah" - all of which Nominatim resolves). They could never reach
       it. The stop-token test keeps the protection without the collateral.
    """
    words = community.split()
    ladder = [community]
    for cut in range(len(words) - 1, 0, -1):
        candidate = " ".join(words[:cut])
        if candidate and candidate not in ladder and _is_meaningful(candidate):
            ladder.append(candidate)
    return ladder


_PLACE_CLASSES = {"boundary", "place", "landuse"}


_CLASS_RANK = {"boundary": 0, "place": 1, "landuse": 2}

_SIBLING_SPREAD_KM = 10.0

_COORD_DP = 6


def _round_point(lat: float, lon: float) -> tuple[float, float]:
    return round(lat, _COORD_DP), round(lon, _COORD_DP)


def _select_centroid(hits: list[dict]) -> tuple[float, float] | None:
    """Choose ONE centroid from Nominatim's candidates, deterministically.

    **Why this is not just ``hits[0]``.** Querying "Muhaisanah, Dubai" returns its
    five constituent suburbs (محيصنة 1-5), *all with identical importance*
    (0.14674) - and Nominatim's ordering among equal-scoring ties varies between
    calls. Taking the first hit therefore picked a different sibling on different
    runs: re-running the centroid re-check flipped Muhaisanah Fourth between
    2.36 km and 2.039 km, and Al Thanyah First between 0.836 km and 0.758 km.
    Feature values that depend on which replica answered are not reproducible,
    which for a dissertation is a defect in its own right.

    The rule, in order:
      1. An administrative **boundary** wins outright - it is the area itself.
      2. Otherwise take every hit tied at the highest importance and use their
         **mean position**. This is both deterministic (a mean is independent of
         order) and semantically right: if asking for "Muhaisanah" returns its
         five sub-areas, the centre of that set *is* Muhaisanah.
      3. If those siblings are implausibly spread out, fall back to the lowest
         ``osm_id`` - arbitrary, but stable and never invented.
    """
    places = [h for h in hits if _is_place_like(h)]
    if not places:
        return None

    best_rank = min(_CLASS_RANK.get(h["class"], 9) for h in places)
    places = [h for h in places if _CLASS_RANK.get(h["class"], 9) == best_rank]
    if best_rank == 0:
        best = min(places, key=lambda h: (-float(h.get("importance") or 0),
                                          int(h.get("osm_id") or 0)))
        return _round_point(float(best["lat"]), float(best["lon"]))

    top = max(float(h.get("importance") or 0) for h in places)
    tied = [h for h in places if float(h.get("importance") or 0) == top]
    pts = [(float(h["lat"]), float(h["lon"])) for h in tied]

    spread = max(
        (_haversine_km(a[0], a[1], b[0], b[1]) for a in pts for b in pts),
        default=0.0,
    )
    if spread > _SIBLING_SPREAD_KM:
        best = min(tied, key=lambda h: int(h.get("osm_id") or 0))
        return _round_point(float(best["lat"]), float(best["lon"]))

    return _round_point(sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts))


def _is_place_like(hit: dict) -> bool:
    """Whether a Nominatim hit is an area, not a POI that merely shares the name.

    This guard exists because of a circularity found in the live sweep: querying
    "Al Garhoud, Dubai" returns the **Al Garhoud metro station**
    (``class=railway type=station``) as the top hit, ahead of the suburb boundary.
    Taking hit[0] therefore placed the community centroid *on a station*, and the
    nearest-metro feature came back as **0.0 km** - measuring "the geocoder
    matched a station", not the community's actual metro access. A handful of
    such rows would teach the model that certain areas sit at distance zero.
    """
    return hit.get("class") in _PLACE_CLASSES


def _normalise_arabic(name: str) -> str:
    """Normalise the orthographic variants that separate DLD's Arabic from OSM's.

    DLD writes ``الجافليه`` where OSM has ``الجافلية`` - a word-final haa where the
    standard spelling uses taa marbuta. Same word, and Nominatim matches only the
    exact form. Also folds the hamza-carrying alef forms, which vary freely.

    The haa→taa-marbuta swap is applied ONLY at the end of a word: ``ه`` mid-word
    is a real letter (``الهضبة``), and rewriting it would corrupt the name.
    """
    folded = re.sub("[أإآ]", "ا", name).replace("ى", "ي")
    return " ".join(
        w[:-1] + "ة" if w.endswith("ه") else w for w in folded.split()
    )


def _arabic_ladder(name_ar: str) -> list[str]:
    """Arabic search strings: each spelling variant, progressively broadened."""
    ladder: list[str] = []
    for variant in (name_ar, _normalise_arabic(name_ar)):
        for rung in _query_ladder(variant):
            if rung not in ladder:
                ladder.append(rung)
    return ladder


def _search(query: str) -> list[dict]:
    """One Nominatim lookup, returning raw hits."""
    params = {
        "q": query,
        "format": "json",
        "limit": 10,
        "countrycodes": "ae",
    }
    with httpx.Client(timeout=_TIMEOUT_S, headers=_HEADERS) as http:
        resp = http.get(NOMINATIM_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def geocode_community(
    community: str, community_ar: str | None = None
) -> tuple[float, float, str]:
    """Return ``(lat, lon, source)`` for a DLD area name.

    Source is one of:
      * ``"reference"``            - hand-verified centroid from ``db.reference``.
      * ``"nominatim"``            - exact match on the full English area name.
      * ``"nominatim_relaxed"``    - matched only after broadening to a parent
        area (see ``_query_ladder``).
      * ``"nominatim_ar"`` / ``"nominatim_ar_relaxed"`` - matched on DLD's Arabic
        name, exactly or broadened.
      * ``"nominatim_poi"``        - only a point of interest matched; kept for
        coverage but flagged as untrustworthy.

    Recorded distinctly so the write-up can report centroid precision honestly
    rather than implying every community was pinned exactly.

    **Why Arabic is tried at all.** DLD and OSM transliterate differently - DLD's
    ``Al Barshaa South`` / ``Al Jafliya`` / ``Al Goze`` are OSM's ``Al Barsha
    South`` / ``Al Jafiliya`` / ``Al Quoz`` - and no amount of broadening fixes a
    spelling difference: 9 of the first 30 communities in the live sweep were lost
    this way. DLD ships ``area_name_ar`` alongside ``area_name_en``, and the
    Arabic is the SAME string OSM stores, so geocoding by it sidesteps
    transliteration entirely. This is the F21 principle again - use the mapping
    already present in the data rather than asserting a hand-written spelling
    table that would be uncitable and silently wrong wherever memory failed.

    Raises ``NominatimError`` when nothing matches - the caller then records the
    community as un-enriched instead of inventing a location.
    """
    if community in COMMUNITY_CENTROIDS:
        lat, lon = COMMUNITY_CENTROIDS[community]
        return lat, lon, "reference"

    ladders = [("", _query_ladder(community))]
    if community_ar:
        ladders.append(("_ar", _arabic_ladder(community_ar)))

    fallback: tuple[float, float] | None = None
    tried: list[str] = []
    first_query = True

    for tag, ladder in ladders:
        suffix = ", دبي" if tag == "_ar" else ", Dubai, United Arab Emirates"
        for i, name in enumerate(ladder):
            if not first_query:
                time.sleep(NOMINATIM_DELAY_S)
            first_query = False
            tried.append(name)
            try:
                hits = _search(f"{name}{suffix}")
            except Exception as exc:  # noqa: BLE001
                log.error("nominatim_failed", community=community, error=str(exc))
                raise NominatimError(str(exc)) from exc

            point = _select_centroid(hits)
            if point is not None:
                source = f"nominatim{tag}" if i == 0 else f"nominatim{tag}_relaxed"
                return point[0], point[1], source
            if hits and fallback is None:
                fallback = (float(hits[0]["lat"]), float(hits[0]["lon"]))

    if fallback is not None:
        log.warning("nominatim_poi_centroid_only", community=community)
        return fallback[0], fallback[1], "nominatim_poi"

    raise NominatimError(f"no Nominatim match for {community!r} (tried {tried})")


def _stations_stmt(lat: float, lon: float) -> str:
    """Overpass statement selecting rail stations near a point.

    Dubai Metro stations carry ``railway=station``; ``station=subway`` catches
    the handful tagged only as subway.
    """
    return (
        f'(node["railway"="station"](around:{METRO_SEARCH_RADIUS_M},{lat},{lon});'
        f'node["station"="subway"](around:{METRO_SEARCH_RADIUS_M},{lat},{lon}););'
    )


def _amenities_stmt(lat: float, lon: float, radius_m: int) -> str:
    return f'node["amenity"](around:{radius_m},{lat},{lon});'


def _nearest_station(lat: float, lon: float, stations: list[dict]) -> tuple[float, str | None]:
    if not stations:
        return NO_STATION_KM, None
    nearest = min(stations, key=lambda s: _haversine_km(lat, lon, s["lat"], s["lon"]))
    km = _haversine_km(lat, lon, nearest["lat"], nearest["lon"])
    return round(km, 3), (nearest.get("tags") or {}).get("name")


def _count_from(elements: list[dict]) -> int:
    for el in elements:
        if el.get("type") == "count":
            return int(el.get("tags", {}).get("total", 0))
    raise OverpassError("count query returned no count element")


def get_location_features(
    lat: float, lon: float, radius_m: int = POI_RADIUS_M
) -> tuple[float, str | None, int]:
    """Both locational features in ONE Overpass request.

    Returns ``(metro_km, station_name, poi_count)``.

    Overpass allows several statements per request, and ``out count`` applies to
    the statement immediately before it - so stations and the amenity tally come
    back in a single response. Asking separately doubled the requests against a
    shared, rate-limited public API for no extra information: measured over the
    first sweep, two calls cost ~2 minutes per community, which is ~6 hours for
    the ~180 DLD areas. Halving the calls halves that.
    """
    query = (
        f"[out:json][timeout:{OVERPASS_SERVER_TIMEOUT_S}];"
        f"{_stations_stmt(lat, lon)}"
        "out;"
        f"{_amenities_stmt(lat, lon, radius_m)}"
        "out count;"
    )
    elements = _overpass(query)
    stations = [e for e in elements if e.get("type") == "node"]
    km, name = _nearest_station(lat, lon, stations)
    return km, name, _count_from(elements)


def get_metro_distance(lat: float, lon: float) -> tuple[float, str | None]:
    """Return ``(km_to_nearest_station, station_name)`` within the search radius.

    ``(NO_STATION_KM, None)`` when nothing is in range. Standalone form; the
    sweep uses ``get_location_features`` to fetch both features in one request.
    """
    query = (
        f"[out:json][timeout:{OVERPASS_SERVER_TIMEOUT_S}];"
        f"{_stations_stmt(lat, lon)}"
        "out;"
    )
    return _nearest_station(lat, lon, _overpass(query))


def get_poi_density(lat: float, lon: float, radius_m: int = POI_RADIUS_M) -> int:
    """Return the count of ``amenity`` POIs within ``radius_m`` of the point.

    Uses Overpass's ``out count`` so the server returns a single tally instead of
    every node - the same number for a fraction of the bandwidth.
    """
    query = (
        f"[out:json][timeout:{OVERPASS_SERVER_TIMEOUT_S}];"
        f"{_amenities_stmt(lat, lon, radius_m)}"
        "out count;"
    )
    return _count_from(_overpass(query))


def _post(url: str, query: str) -> list[dict]:
    """POST one query to one endpoint; return its elements.

    Raises on anything that is not a usable answer: a non-200, a non-JSON body
    (the HTML error pages Overpass serves when saturated), or a ``remark`` -
    which is how Overpass reports a *server-side* runtime error (timeout, memory)
    inside an otherwise-200 response, with an empty element list that would
    otherwise read as a genuine "nothing here".

    POST, never GET: two of the mirrors 504 on the query-string form of the very
    same request.
    """
    with httpx.Client(timeout=_TIMEOUT_S, headers=_HEADERS) as http:
        resp = http.post(url, data={"data": query})
    if resp.status_code != 200:
        raise OverpassError(f"HTTP {resp.status_code}")
    body = resp.json()
    if "remark" in body:
        raise OverpassError(f"server remark: {body['remark'][:120]}")
    return body.get("elements", [])


def _has_planet_data(url: str) -> bool:
    """Whether ``url`` serves worldwide data - probed once per process."""
    if url in _planet_verified:
        return _planet_verified[url]
    try:
        elements = _post(url, _PROBE_QUERY)
        total = int((elements[0].get("tags", {}) if elements else {}).get("total", 0))
    except Exception as exc:  # noqa: BLE001
        log.debug("overpass_probe_unreachable", url=url, error=str(exc))
        _benched(url)
        return False
    ok = total > 0
    _planet_verified[url] = ok
    if not ok:
        log.warning("overpass_mirror_lacks_dubai_data", url=url)
    return ok


def _overpass(query: str) -> list[dict]:
    """Run a query against the Overpass mirrors in turn; return its elements.

    Each mirror must pass the planet-coverage probe before its answer is used,
    and the whole list is swept ``OVERPASS_ATTEMPTS`` times because most failures
    are transient saturation. Raises ``OverpassError`` when every endpoint failed
    on every pass - the caller then records the community as un-enriched rather
    than storing a fabricated value.
    """
    global _rotation
    endpoints = _endpoints()
    start = _rotation % len(endpoints)
    _rotation += 1
    ordered = [endpoints[(start + i) % len(endpoints)] for i in range(len(endpoints))]

    errors: list[str] = []
    first = True
    for attempt in range(OVERPASS_ATTEMPTS):
        candidates = [u for u in ordered if _available(u)] if attempt == 0 else ordered
        for url in candidates:
            if not first:
                time.sleep(_RETRY_DELAY_S)
            first = False
            if not _has_planet_data(url):
                errors.append(f"{url}: no Dubai coverage / unreachable")
                continue
            try:
                elements = _post(url, query)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {type(exc).__name__} {exc}")
                _benched(url)
                continue
            global _last_mirror
            _last_mirror = url
            return elements

    log.error("overpass_failed", errors=errors)
    raise OverpassError("; ".join(errors))


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
