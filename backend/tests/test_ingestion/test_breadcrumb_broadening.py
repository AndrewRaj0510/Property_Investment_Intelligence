"""Portals name a listing more specifically than DLD records it (F41).

Found by scoring a real Dubizzle listing whose breadcrumb was
`UAE > Dubai > DAMAC Lagoons > Costa Brava`. We took the deepest level, "Costa
Brava" - a cluster. DLD records the whole master community as ONE area,
`Al Hebiah Fifth` (456 villa sales, 103 rent contracts), and the mined alias
table already maps "DAMAC Lagoons" onto it. So a listing in a fully-covered
community came back as "Insufficient data" with an unanchored 12.11% yield.

This is the third instance of the same class after the JVT acronym gap (F36):
the area data was present and the NAME failed to reach it.
"""

from __future__ import annotations

from ingestion.listing_client import (
    _broader_locations,
    _normalize_bayut,
    _normalize_dubizzle,
)
from services.listing_scorer import _pick_area

REAL_PATH = [
    {"name": "UAE"},
    {"name": "Dubai"},
    {"name": "DAMAC Lagoons"},
    {"name": "Costa Brava"},
]


def test_only_broader_levels_are_offered():
    """Generalising a cluster to its master community is sound; guessing a
    specific tower from a community is not."""
    names = ["UAE", "Dubai", "DAMAC Lagoons", "Costa Brava"]
    assert _broader_locations(names, 3) == ["DAMAC Lagoons"]


def test_country_and_city_are_never_candidates():
    """"Dubai" would match nothing useful and "UAE" is meaningless as an area."""
    names = ["UAE", "Dubai", "Business Bay"]
    assert _broader_locations(names, 2) == []


def test_broader_levels_are_ordered_most_specific_first():
    names = ["UAE", "Dubai", "Master", "Sub", "Cluster"]
    assert _broader_locations(names, 4) == ["Sub", "Master"]


def test_no_levels_when_the_portal_already_chose_the_community():
    assert _broader_locations(["UAE", "Dubai", "Marsa Dubai"], 2) == []


def test_dubizzle_keeps_the_cluster_but_offers_the_master_community():
    out = _normalize_dubizzle({"listing_id": "1", "location_path": REAL_PATH}, "u")
    assert out["community"] == "Costa Brava"
    assert out["community_candidates"] == ["DAMAC Lagoons"]


def test_bayut_already_picks_the_community_level_so_offers_nothing():
    """Bayut's breadcrumb is Country > City > Community > Tower and
    `_bayut_community` takes index 2, so there is nothing broader to fall back
    to. The field exists so both portals present the same shape."""
    loc = [
        {"name": "UAE"},
        {"name": "Dubai"},
        {"name": "Jumeirah Village Triangle (JVT)"},
        {"name": "JVT District 4"},
        {"name": "Binghatti Luxuria"},
    ]
    out = _normalize_bayut({"externalID": "1", "location": loc}, "u")
    assert out["community"] == "Jumeirah Village Triangle (JVT)"
    assert out["community_candidates"] == []


def test_missing_location_yields_no_candidates():
    assert _normalize_dubizzle({"listing_id": "1"}, "u")["community_candidates"] == []


def _listing(community, candidates):
    return {"community": community, "community_candidates": candidates}


def test_portals_own_choice_wins_when_it_has_a_baseline(monkeypatch):
    """Broadening must never override a level that already matched - that would
    trade a precise segment for a coarser one."""
    import services.listing_scorer as ls

    monkeypatch.setattr(
        ls, "_baseline_for",
        lambda a, t: {"area": a, "n_sales": 1} if a == "Costa Brava" else None,
    )
    area, baseline = _pick_area(_listing("Costa Brava", ["DAMAC Lagoons"]), "villa")
    assert area == "Costa Brava"
    assert baseline["area"] == "Costa Brava"


def test_falls_back_to_the_master_community(monkeypatch):
    import services.listing_scorer as ls

    monkeypatch.setattr(
        ls, "_baseline_for",
        lambda a, t: {"area": "Al Hebiah Fifth"} if a == "DAMAC Lagoons" else None,
    )
    area, baseline = _pick_area(_listing("Costa Brava", ["DAMAC Lagoons"]), "villa")
    assert area == "DAMAC Lagoons"
    assert baseline["area"] == "Al Hebiah Fifth"


def test_no_baseline_anywhere_keeps_the_portals_name(monkeypatch):
    """The no-data path must be unchanged: the user still sees the community
    they pasted, and the verdict still refuses to judge."""
    import services.listing_scorer as ls

    monkeypatch.setattr(ls, "_baseline_for", lambda a, t: None)
    area, baseline = _pick_area(_listing("Nowhere", ["Also Nowhere"]), "villa")
    assert area == "Nowhere"
    assert baseline is None


def test_duplicate_levels_are_only_tried_once(monkeypatch):
    import services.listing_scorer as ls

    seen = []

    def fake(a, t):
        seen.append(a)
        return None

    monkeypatch.setattr(ls, "_baseline_for", fake)
    _pick_area(_listing("Same", ["Same", "Same"]), "villa")
    assert seen == ["Same"]


def test_listing_with_no_candidates_key_still_works(monkeypatch):
    """Fixtures and older captures predate the field."""
    import services.listing_scorer as ls

    monkeypatch.setattr(ls, "_baseline_for", lambda a, t: {"area": a})
    assert _pick_area({"community": "Marsa Dubai"}, "apartment")[0] == "Marsa Dubai"
