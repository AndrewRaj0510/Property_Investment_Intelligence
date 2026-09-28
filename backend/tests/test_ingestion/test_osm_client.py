"""Unit tests for the OSM enrichment client (Objective 1.4).

Every HTTP call is stubbed - these are offline tests of the parsing, fallback and
*trust* logic, not of OpenStreetMap's availability. The trust logic is the point:
the live sweep is only as sound as its refusal to record a value it did not
actually measure.
"""

from __future__ import annotations

import json

import pytest

from ingestion import osm_client
from ingestion.osm_client import (
    NO_STATION_KM,
    NominatimError,
    OverpassError,
    _query_ladder,
    geocode_community,
    get_metro_distance,
    get_poi_density,
)


class _Resp:
    """Minimal httpx.Response stand-in."""

    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, str):
            return json.loads(self._payload)
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _clear_mirror_state():
    """Probe verdicts and cooldowns are per-process caches; isolate each test."""
    osm_client._planet_verified.clear()
    osm_client._cooldown_until.clear()
    yield
    osm_client._planet_verified.clear()
    osm_client._cooldown_until.clear()


def _stub_overpass(monkeypatch, handler):
    """Route every Overpass POST through ``handler(url, query)``."""
    def fake_post(url, query):
        return handler(url, query)

    monkeypatch.setattr(osm_client, "_post", fake_post)
    monkeypatch.setattr(osm_client.time, "sleep", lambda *_: None)


def _count(total: int) -> list[dict]:
    return [{"type": "count", "id": 0, "tags": {"total": str(total)}}]


def test_regional_mirror_is_rejected_not_believed(monkeypatch):
    """A Switzerland-only mirror answers Dubai with a valid, empty 200.

    This really happened (overpass.osm.ch). Silently trusting it writes "no
    amenities within 1 km" for every Dubai community. The probe must reject the
    mirror, and with no other mirror available the call must RAISE rather than
    return a fabricated zero.
    """
    seen = []

    def handler(url, query):
        seen.append(url)
        return _count(0)

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(osm_client, "OVERPASS_URLS", ("https://regional.example/api",))

    with pytest.raises(OverpassError):
        get_poi_density(25.08, 55.14)
    assert len(seen) == 1


def test_planet_mirror_is_probed_once_then_reused(monkeypatch):
    calls = {"probe": 0, "query": 0}

    def handler(url, query):
        if query == osm_client._PROBE_QUERY:
            calls["probe"] += 1
            return _count(341)
        calls["query"] += 1
        return _count(12)

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(osm_client, "OVERPASS_URLS", ("https://planet.example/api",))

    assert get_poi_density(25.08, 55.14) == 12
    assert get_poi_density(25.09, 55.15) == 12
    assert calls["probe"] == 1
    assert calls["query"] == 2


def test_failing_mirror_is_benched_not_retried_every_call(monkeypatch):
    """A down mirror must not cost its timeout once per community.

    Re-probing a dead endpoint on every lookup turned a ~2s query into ~5
    minutes during the live sweep. After one failure the mirror is skipped until
    its cooldown expires.
    """
    hits = {"broken": 0}

    def handler(url, query):
        if "broken" in url:
            hits["broken"] += 1
            raise OverpassError("HTTP 504")
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        return _count(5)

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(
        osm_client, "OVERPASS_URLS",
        ("https://broken.example/api", "https://planet.example/api"),
    )
    monkeypatch.setattr(osm_client, "_rotation", 0)

    for _ in range(4):
        assert get_poi_density(25.08, 55.14) == 5
    assert hits["broken"] == 1


def test_failing_mirror_falls_through_to_the_next(monkeypatch):
    def handler(url, query):
        if "broken" in url:
            raise OverpassError("HTTP 504")
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        return _count(7)

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(
        osm_client, "OVERPASS_URLS",
        ("https://broken.example/api", "https://planet.example/api"),
    )
    assert get_poi_density(25.08, 55.14) == 7


def test_poi_density_reads_the_total_not_the_node_tag(monkeypatch):
    def handler(url, query):
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        return [{"type": "count", "id": 0,
                 "tags": {"nodes": "40", "ways": "0", "total": "40"}}]

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(osm_client, "OVERPASS_URLS", ("https://planet.example/api",))
    assert get_poi_density(25.08, 55.14) == 40


def test_count_query_without_a_count_element_raises(monkeypatch):
    """A count query that came back without a count did not run as asked."""
    def handler(url, query):
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        return []

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(osm_client, "OVERPASS_URLS", ("https://planet.example/api",))
    with pytest.raises(OverpassError):
        get_poi_density(25.08, 55.14)


def test_metro_distance_picks_the_nearest_station(monkeypatch):
    def handler(url, query):
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        return [
            {"type": "node", "lat": 25.20, "lon": 55.30, "tags": {"name": "Far"}},
            {"type": "node", "lat": 25.085, "lon": 55.145, "tags": {"name": "Near"}},
        ]

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(osm_client, "OVERPASS_URLS", ("https://planet.example/api",))
    km, name = get_metro_distance(25.08, 55.14)
    assert name == "Near"
    assert 0 < km < 1.5


def test_combined_query_splits_stations_from_the_count(monkeypatch):
    """One request returns station nodes AND the amenity tally; parse both."""
    queries = []

    def handler(url, query):
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        queries.append(query)
        return [
            {"type": "node", "lat": 25.085, "lon": 55.145, "tags": {"name": "Near"}},
            {"type": "node", "lat": 25.20, "lon": 55.30, "tags": {"name": "Far"}},
            {"type": "count", "id": 0, "tags": {"total": "257"}},
        ]

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(osm_client, "OVERPASS_URLS", ("https://planet.example/api",))

    km, name, poi = osm_client.get_location_features(25.08, 55.14)
    assert (name, poi) == ("Near", 257)
    assert 0 < km < 1.5
    assert len(queries) == 1


def test_combined_query_with_no_stations_still_returns_the_count(monkeypatch):
    def handler(url, query):
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        return [{"type": "count", "id": 0, "tags": {"total": "3"}}]

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(osm_client, "OVERPASS_URLS", ("https://planet.example/api",))
    assert osm_client.get_location_features(25.08, 55.14) == (NO_STATION_KM, None, 3)


def test_no_station_in_range_returns_the_sentinel(monkeypatch):
    """Must be the documented sentinel, never 0.0 - 0 km means 'at the station'."""
    def handler(url, query):
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        return []

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(osm_client, "OVERPASS_URLS", ("https://planet.example/api",))
    assert get_metro_distance(25.08, 55.14) == (NO_STATION_KM, None)


def test_query_ladder_broadens_one_word_at_a_time():
    assert _query_ladder("Al Barsha South Fourth") == [
        "Al Barsha South Fourth", "Al Barsha South", "Al Barsha",
    ]
    assert "Wadi Al Safa" in _query_ladder("Wadi Al Safa 5")
    assert _query_ladder("Mirdif") == ["Mirdif"]


def test_query_ladder_reaches_the_parent_of_a_deep_name():
    """Regression: 8 of the first 40 swept communities were skipped.

    "Al Goze Industrial First" needs THREE rungs to reach "Al Goze". The earlier
    ladder dropped only one trailing word, so the ordinal strip and the word drop
    produced the same string and it never tried the parent area.
    """
    assert _query_ladder("Al Goze Industrial First") == [
        "Al Goze Industrial First", "Al Goze Industrial", "Al Goze",
    ]


def test_query_ladder_never_searches_an_article_alone():
    """A confidently wrong centroid is worse than none.

    "Al, Dubai" resolves to a `place` hit at the city centre, so `_is_place_like`
    cannot catch it - the ladder itself must refuse to ask.
    """
    for name in ["Al Goze Industrial First", "Wadi Al Safa 5", "Al Barsha South Fourth"]:
        assert all(osm_client._is_meaningful(rung) for rung in _query_ladder(name))
        assert "Al" not in _query_ladder(name)
    assert "Wadi" not in _query_ladder("Wadi Al Safa 5")


def test_query_ladder_reaches_a_single_word_parent():
    """Regression: a two-word floor blocked 14 communities in the live sweep.

    "Jumeirah First" / "Warsan Fourth" / "Muhaisanah Second" all have a
    single-word parent that Nominatim resolves; the floor made them unreachable.
    """
    assert _query_ladder("Jumeirah First") == ["Jumeirah First", "Jumeirah"]
    assert _query_ladder("Warsan Fourth") == ["Warsan Fourth", "Warsan"]
    assert _query_ladder("Al Saffa First") == ["Al Saffa First", "Al Saffa"]


def test_hand_verified_centroid_wins_without_a_network_call(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("Nominatim must not be called for a known centroid")

    monkeypatch.setattr(osm_client.httpx, "Client", boom)
    lat, lon, source = geocode_community("Dubai Marina")
    assert source == "reference"
    assert (round(lat), round(lon)) == (25, 55)


def _stub_nominatim(monkeypatch, results_by_query):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            return _Resp(results_by_query.get(params["q"], []))

    monkeypatch.setattr(osm_client.httpx, "Client", FakeClient)
    monkeypatch.setattr(osm_client.time, "sleep", lambda *_: None)


def _place(lat, lon):
    return {"lat": lat, "lon": lon, "class": "boundary", "type": "administrative"}


def _station(lat, lon):
    return {"lat": lat, "lon": lon, "class": "railway", "type": "station"}


def test_exact_match_is_labelled_nominatim(monkeypatch):
    _stub_nominatim(monkeypatch, {
        "Marsa Dubai, Dubai, United Arab Emirates": [_place("25.08", "55.14")],
    })
    lat, lon, source = geocode_community("Marsa Dubai")
    assert (lat, lon, source) == (25.08, 55.14, "nominatim")


def test_station_hit_is_skipped_for_the_suburb_boundary(monkeypatch):
    """Regression: the live sweep gave Al Garhoud a metro distance of 0.0 km.

    Nominatim ranks the Al Garhoud METRO STATION above the Al Garhoud suburb, so
    taking hit[0] put the centroid on a station - the nearest-metro feature then
    measured the geocoder's choice, not the community.
    """
    _stub_nominatim(monkeypatch, {
        "Al Garhoud, Dubai, United Arab Emirates": [
            _station("25.2470", "55.3520"),
            _place("25.2440", "55.3480"),
        ],
    })
    lat, lon, source = geocode_community("Al Garhoud")
    assert (lat, lon) == (25.2440, 55.3480)
    assert source == "nominatim"


def test_poi_only_match_is_kept_but_labelled(monkeypatch):
    """No area anywhere on the ladder: keep coverage, flag the row as untrusted."""
    _stub_nominatim(monkeypatch, {
        "Some Tower, Dubai, United Arab Emirates": [_station("25.1", "55.2")],
    })
    lat, lon, source = geocode_community("Some Tower")
    assert (lat, lon) == (25.1, 55.2)
    assert source == "nominatim_poi"


def test_parent_area_fallback_is_labelled_relaxed(monkeypatch):
    """Sub-parcel names miss; the parent area is recorded as lower precision."""
    _stub_nominatim(monkeypatch, {
        "Al Barsha South, Dubai, United Arab Emirates": [_place("25.06", "55.21")],
    })
    lat, lon, source = geocode_community("Al Barsha South Fourth")
    assert (lat, lon) == (25.06, 55.21)
    assert source == "nominatim_relaxed"


def test_unmatched_community_raises_rather_than_guessing(monkeypatch):
    _stub_nominatim(monkeypatch, {})
    with pytest.raises(NominatimError):
        geocode_community("Nonexistent Parcel 999")


def _suburb(lat, lon, osm_id, importance=0.14674):
    return {"lat": lat, "lon": lon, "class": "place", "type": "suburb",
            "osm_id": osm_id, "importance": importance}


def test_tied_siblings_average_and_ignore_api_ordering():
    """Regression: re-running the sweep flipped Muhaisanah between 2.36 and 2.039 km.

    "Muhaisanah, Dubai" returns its five sub-suburbs, ALL with importance
    0.14674, and Nominatim orders equal-scoring ties differently between calls.
    Averaging is both order-independent and the right answer: the centre of an
    area's parts is the area's centre.
    """
    hits = [
        _suburb("25.26", "55.40", 13706209535),
        _suburb("25.27", "55.41", 13706209541),
        _suburb("25.28", "55.42", 13706209538),
    ]
    first = osm_client._select_centroid(hits)
    shuffled = osm_client._select_centroid(list(reversed(hits)))
    assert first == shuffled
    assert first == pytest.approx((25.27, 55.41))


def test_administrative_boundary_beats_place_nodes():
    """The boundary IS the area; a place node is just a point inside it."""
    hits = [
        _suburb("25.26", "55.40", 1, importance=0.9),
        {"lat": "25.30", "lon": "55.45", "class": "boundary",
         "type": "administrative", "osm_id": 18471781, "importance": 0.405},
    ]
    assert osm_client._select_centroid(hits) == pytest.approx((25.30, 55.45))


def test_scattered_siblings_are_not_averaged_into_a_gap():
    """Averaging far-apart hits would invent a centroid in empty desert."""
    hits = [
        _suburb("25.26", "55.40", 500),
        _suburb("24.90", "55.00", 200),
    ]
    assert osm_client._select_centroid(hits) == pytest.approx((25.26, 55.40)) or \
           osm_client._select_centroid(hits) == pytest.approx((24.90, 55.00))
    assert osm_client._select_centroid(hits) == osm_client._select_centroid(hits[::-1])


def test_pinned_mirror_is_the_only_endpoint_used(monkeypatch):
    """Mirrors disagree, so a dataset must come from ONE planet snapshot.

    Measured: the same centroid gave 8.536 km from maps.mail.ru and 11.352 km
    from another mirror, and rows flip-flopped between sweeps. When pinned, a
    failure must raise rather than quietly fall back to a different snapshot.
    """
    seen = []

    def handler(url, query):
        seen.append(url)
        if query == osm_client._PROBE_QUERY:
            return _count(341)
        return _count(9)

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(
        osm_client, "OVERPASS_URLS",
        ("https://a.example/api", "https://b.example/api", "https://c.example/api"),
    )
    osm_client.pin_mirror("https://b.example/api")
    try:
        assert get_poi_density(25.08, 55.14) == 9
        assert {u for u in seen} == {"https://b.example/api"}
    finally:
        osm_client.pin_mirror(None)


def test_pinned_mirror_failure_raises_instead_of_falling_back(monkeypatch):
    def handler(url, query):
        raise OverpassError("HTTP 504")

    _stub_overpass(monkeypatch, handler)
    monkeypatch.setattr(
        osm_client, "OVERPASS_URLS",
        ("https://a.example/api", "https://b.example/api"),
    )
    osm_client.pin_mirror("https://b.example/api")
    try:
        with pytest.raises(OverpassError):
            get_poi_density(25.08, 55.14)
    finally:
        osm_client.pin_mirror(None)


def test_no_place_like_candidates_returns_none():
    assert osm_client._select_centroid([_station("25.1", "55.2")]) is None
    assert osm_client._select_centroid([]) is None


def test_arabic_name_rescues_a_transliteration_mismatch(monkeypatch):
    """Regression: 9 of the first 30 swept communities failed on spelling alone.

    DLD writes "Al Barshaa South"; OSM writes "Al Barsha South". No amount of
    broadening fixes that - but DLD's own Arabic name is the string OSM stores.
    """
    _stub_nominatim(monkeypatch, {
        "البرشاء جنوب, دبي": [_place("25.0747", "55.2")],
    })
    lat, lon, source = geocode_community("Al Barshaa South", "البرشاء جنوب")
    assert (lat, lon) == (25.0747, 55.2)
    assert source == "nominatim_ar"


def test_arabic_normalisation_fixes_final_haa(monkeypatch):
    """DLD's الجافليه vs OSM's الجافلية - word-final haa where OSM has taa marbuta."""
    _stub_nominatim(monkeypatch, {
        "الجافلية, دبي": [_place("25.2376", "55.28")],
    })
    lat, lon, source = geocode_community("Al Jafliya", "الجافليه")
    assert (lat, lon) == (25.2376, 55.28)
    assert source.startswith("nominatim_ar")


def test_arabic_normalisation_leaves_mid_word_haa_alone():
    """ه inside a word is a real letter - rewriting it would corrupt the name."""
    assert osm_client._normalise_arabic("الهضبه") == "الهضبة"
    assert osm_client._normalise_arabic("القوز الصناعيه") == "القوز الصناعية"


def test_english_match_wins_before_arabic_is_tried(monkeypatch):
    """Arabic is a fallback, not a preference - don't spend a call when English works."""
    _stub_nominatim(monkeypatch, {
        "Al Barsha South, Dubai, United Arab Emirates": [_place("25.06", "55.21")],
        "البرشاء جنوب, دبي": [_place("99.0", "99.0")],
    })
    lat, lon, source = geocode_community("Al Barsha South", "البرشاء جنوب")
    assert (lat, lon, source) == (25.06, 55.21, "nominatim")
