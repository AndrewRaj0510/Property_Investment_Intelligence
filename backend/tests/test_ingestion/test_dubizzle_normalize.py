"""Dubizzle normalizer regression tests.

Both cases here were found by scoring ONE real Dubizzle listing
(`.../residential/apartmentflat/.../high-floor-full-furnished-...`). They were
invisible until then because no real Dubizzle response had ever been through the
normalizer - every prior test used Bayut-shaped or hand-written data.
"""

from __future__ import annotations

import pytest

from ingestion.listing_client import _dubizzle_type, _normalize_dubizzle
from services.listing_scorer import canonical_type

REAL_URL = (
    "https://dubai.dubizzle.com/property-for-rent/residential/apartmentflat/"
    "2026/7/26/high-floor-full-furnished-close-to-exit-2-169339/"
)


def test_category_breadcrumb_is_not_mistaken_for_a_type():
    """Measured: the actor returned categories ending in 'Property for Rent'.
    Taking the last element yielded an unmappable type → canonical_type None →
    NoBaselineForListing → HTTP 422, despite every other field being usable."""
    d = {"categories": [{"name": "Property for Rent"}]}
    assert _dubizzle_type(d, REAL_URL) == "apartmentflat"
    assert canonical_type(_dubizzle_type(d, REAL_URL)) == "apartment"


def test_real_type_breadcrumb_is_preferred_over_the_url():
    d = {"categories": [{"name": "Property for Rent"}, {"name": "Villa"}]}
    assert canonical_type(_dubizzle_type(d, REAL_URL)) == "villa"


def test_deepest_type_breadcrumb_wins():
    d = {"categories": [{"name": "Residential"}, {"name": "Apartment/Flat"}]}
    assert canonical_type(_dubizzle_type(d, REAL_URL)) == "apartment"


def test_url_fallback_when_no_breadcrumbs():
    assert canonical_type(_dubizzle_type({}, REAL_URL)) == "apartment"


def test_unmappable_input_still_returns_none_not_a_guess():
    """No breadcrumb, no type in the URL → the scorer must still reject it
    rather than have the normalizer invent a plausible type."""
    d = {"categories": [{"name": "Property for Sale"}]}
    assert canonical_type(_dubizzle_type(d, "https://dubizzle.com/x/y/")) is None


@pytest.mark.parametrize(
    "token,expected",
    [
        ("apartmentflat", "apartment"),
        ("Apartment/Flat", "apartment"),
        ("Duplex", "apartment"),
        ("Townhouse", "villa"),
        ("Villa", "villa"),
    ],
)
def test_dubizzle_type_vocabulary_maps_to_the_model(token, expected):
    assert canonical_type(token) == expected


def test_multilingual_absolute_url_is_unwrapped():
    """Measured: `absolute_url` came back as {'en': ..., 'ar': ...}. It was the
    only multilingual field left unwrapped, so ScoreResponse rejected it with
    'Input should be a valid string' - a 500 immediately after the 422 was fixed.
    """
    item = {
        "absolute_url": {"en": REAL_URL, "ar": "https://ar.example/x"},
        "categories": [{"name": "Apartment/Flat"}],
        "price": 75000,
        "listing_id": 27666466,
    }
    out = _normalize_dubizzle(item, REAL_URL)
    assert isinstance(out["listing_url"], str)
    assert out["listing_url"] == REAL_URL


def test_plain_string_absolute_url_still_works():
    item = {"absolute_url": REAL_URL, "categories": [], "price": 1}
    assert _normalize_dubizzle(item, REAL_URL)["listing_url"] == REAL_URL


def test_missing_absolute_url_falls_back_to_the_pasted_url():
    item = {"categories": [], "price": 1}
    assert _normalize_dubizzle(item, REAL_URL)["listing_url"] == REAL_URL
