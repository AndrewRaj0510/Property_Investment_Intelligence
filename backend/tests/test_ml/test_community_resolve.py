"""Community resolution tests (F21).

The resolver bridges portal marketing names ("JVC District 14") to DLD's
official area names ("Al Barsha South Fourth"). Two properties matter:

  * it RESOLVES the names portals actually emit, and
  * it REFUSES to resolve ambiguous ones - picking the wrong area produces a
    confidently wrong verdict against another neighbourhood's medians, which is
    worse than "Insufficient data".

These run against injected lookup tables, so they do not need a database and do
not silently pass when the alias table is empty.
"""

from __future__ import annotations

import pytest

from ml import community_resolve as cr

ALIASES = {
    "jumeirah village circle": "Al Barsha South Fourth",
    "jumeirah village triangle": "Al Barsha South Fifth",
    "dubai marina": "Marsa Dubai",
    "downtown dubai": "Burj Khalifa",
    "dubai sports city": "Al Hebiah Fourth",
    "silicon oasis": "Nadd Hessa",
}
AREAS = {
    "al barsha south fourth": "Al Barsha South Fourth",
    "al barsha south fifth": "Al Barsha South Fifth",
    "al barsha south first": "Al Barsha South First",
    "marsa dubai": "Marsa Dubai",
    "burj khalifa": "Burj Khalifa",
    "business bay": "Business Bay",
    "palm jumeirah": "Palm Jumeirah",
    "nadd hessa": "Nadd Hessa",
    "al hebiah fourth": "Al Hebiah Fourth",
}


@pytest.fixture(autouse=True)
def _inject(monkeypatch):
    """Point the cached lookups at fixed tables (no DB, no cache bleed)."""
    cr.clear_cache()
    monkeypatch.setattr(cr, "_alias_map", lambda: ALIASES)
    monkeypatch.setattr(cr, "_known_areas", lambda: AREAS)
    yield
    cr.clear_cache()


def test_the_reported_case_resolves():
    """The listing that returned 'Insufficient data' in the UI."""
    assert cr.resolve_community("JVC District 14") == "Al Barsha South Fourth"


@pytest.mark.parametrize(
    "portal_name,expected",
    [
        ("JVC", "Al Barsha South Fourth"),
        ("jvc", "Al Barsha South Fourth"),
        ("Jumeirah Village Circle", "Al Barsha South Fourth"),
        ("JVT", "Al Barsha South Fifth"),
        ("Dubai Marina", "Marsa Dubai"),
        ("Marina", "Marsa Dubai"),
        ("Downtown", "Burj Khalifa"),
        ("Silicon Oasis", "Nadd Hessa"),
        ("Dubai Sports City", "Al Hebiah Fourth"),
    ],
)
def test_marketing_names_resolve(portal_name, expected):
    assert cr.resolve_community(portal_name) == expected


@pytest.mark.parametrize(
    "noisy",
    [
        "JVC District 14",
        "JVC - District 14",
        "Jumeirah Village Circle, District 12",
        "Dubai Marina - Marina Promenade",
        "Dubai Marina Tower 2",
    ],
)
def test_sub_locality_noise_is_stripped(noisy):
    assert cr.resolve_community(noisy) in {"Al Barsha South Fourth", "Marsa Dubai"}


def test_already_a_dld_area_passes_through():
    assert cr.resolve_community("Al Barsha South Fourth") == "Al Barsha South Fourth"


def test_name_that_is_its_own_dld_area_is_unchanged():
    """Business Bay / Palm Jumeirah are both the marketing AND the DLD name."""
    assert cr.resolve_community("Business Bay") == "Business Bay"
    assert cr.resolve_community("Palm Jumeirah") == "Palm Jumeirah"


def test_ambiguous_prefix_is_not_resolved():
    """'Al Barsha South' matches Fourth, Fifth AND First - resolving it to any
    one of them would score the listing against the wrong neighbourhood."""
    assert cr.resolve_community("Al Barsha South") == "Al Barsha South"


def test_unknown_community_is_returned_unchanged():
    assert cr.resolve_community("Nowhere Land 99") == "Nowhere Land 99"


def test_abbreviation_without_data_backing_is_not_invented(monkeypatch):
    """An abbreviation expands to a NAME, not an area. If that name isn't in the
    data-derived map, nothing is resolved - the expansion is not itself a claim
    about which DLD area the place belongs to."""
    monkeypatch.setattr(cr, "_alias_map", lambda: {})
    monkeypatch.setattr(cr, "_known_areas", lambda: {})
    assert cr.resolve_community("JVC") == "JVC"


def test_empty_and_whitespace_are_safe():
    assert cr.resolve_community("") == ""
    assert cr.resolve_community("   ") == ""


def test_short_names_do_not_partial_match():
    """A 3-char name must not latch onto a long area via containment."""
    assert cr.resolve_community("Al") == "Al"


def test_scorer_and_model_resolve_identically():
    """They look up different tables (feature matrix vs medians); if they
    disagreed, a prediction for one area would be scored against another's."""
    from services.listing_scorer import _resolve_area

    for name in ("JVC District 14", "Dubai Marina", "Nowhere Land 99"):
        assert _resolve_area(name) == cr.resolve_community(name)
