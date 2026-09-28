"""Tests for the offline listing-fixture path (F19).

The point of fixtures is to run the REAL scoring flow without a paid Apify call.
The risk they introduce is a fixture-backed result being mistaken for a live one,
so `source_mode` labelling is tested as carefully as the loading itself.
"""

from __future__ import annotations

import asyncio

import pytest

import config
from ingestion import listing_fixtures as fx
from ingestion.listing_client import fetch_listing

LISTING = {
    "source": "bayut",
    "listing_url": "https://www.bayut.com/property/details-1.html",
    "property_type": "Apartment",
    "community": "Dubai Marina",
    "bedrooms": 2,
    "area_sqft": 1200.0,
    "price": 1_800_000.0,
    "purpose": "for-sale",
}


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """Point the fixture store at a temp dir so tests never touch real captures."""
    monkeypatch.setattr(fx, "FIXTURE_DIR", tmp_path)
    fx.save("unit_test_listing", LISTING)
    return tmp_path


def test_detects_fixture_urls():
    assert fx.is_fixture_url("fixture://marina_2br")
    assert fx.is_fixture_url("FIXTURE://Marina")
    assert not fx.is_fixture_url("https://www.bayut.com/property/details-1.html")


def test_extracts_the_name():
    assert fx.fixture_name("fixture://marina_2br") == "marina_2br"


def test_save_then_load(saved):
    got = fx.load("unit_test_listing")
    assert got["community"] == "Dubai Marina"
    assert got["price"] == 1_800_000.0


def test_available_lists_saved(saved):
    assert "unit_test_listing" in fx.available()


def test_missing_fixture_names_the_remedy(saved):
    with pytest.raises(fx.FixtureNotFound, match="capture_listing"):
        fx.load("does_not_exist")


def test_incomplete_fixture_is_rejected(saved, tmp_path):
    """A fixture missing the normalized contract would test a shape production
    never produces - refuse it rather than score against it."""
    fx.save("broken", {"source": "bayut"})
    with pytest.raises(ValueError, match="missing required keys"):
        fx.load("broken")


def test_name_cannot_escape_the_fixture_dir(saved):
    with pytest.raises(fx.FixtureNotFound):
        fx.load("../../../etc/passwd")


def test_fetch_listing_short_circuits_on_fixture_url(saved):
    """No Apify token is configured here; a fixture URL must still resolve."""
    got = asyncio.run(fetch_listing("fixture://unit_test_listing"))
    assert got["community"] == "Dubai Marina"


def test_replay_mode_resolves_any_url(saved, monkeypatch):
    monkeypatch.setattr(config.get_settings(), "listing_fixture_mode", "replay", raising=False)
    got = asyncio.run(fetch_listing("https://www.bayut.com/property/details-999.html"))
    assert got["community"] == "Dubai Marina"


def test_detect_site_accepts_real_portals():
    from ingestion.listing_client import detect_site

    assert detect_site("https://www.bayut.com/property/details-1.html") == "bayut"
    assert detect_site("https://bayut.com/property/details-1.html") == "bayut"
    assert detect_site("https://dubai.dubizzle.com/property/x") == "dubizzle"
    assert detect_site("https://dubizzle.com/x") == "dubizzle"


@pytest.mark.parametrize(
    "url",
    [
        "https://bayut.com.attacker.net/x",
        "https://dubizzle.com.evil.net/x",
        "https://evil.example.com/?ref=bayut.com",
        "https://notbayut.com/x",
        "https://bayut.com.evil.co.uk/x",
    ],
)
def test_detect_site_rejects_lookalike_hosts(url):
    """A substring test on the host accepted these. The URL is handed to an
    Apify actor that fetches it and feeds the result into scoring + the LLM, so
    an attacker-controlled host must never pass as a supported portal."""
    from ingestion.listing_client import UnsupportedListingSite, detect_site

    with pytest.raises(UnsupportedListingSite):
        detect_site(url)


def test_live_mode_does_not_use_fixtures(saved, monkeypatch):
    """A normal URL in live mode must NOT silently resolve to a fixture - that
    would make a real request quietly return stale saved data."""
    monkeypatch.setattr(config.get_settings(), "listing_fixture_mode", "live", raising=False)
    monkeypatch.setattr(config.get_settings(), "apify_token", "", raising=False)
    from ingestion.listing_client import ListingSourceNotConfigured

    with pytest.raises(ListingSourceNotConfigured):
        asyncio.run(fetch_listing("https://www.bayut.com/property/details-999.html"))
