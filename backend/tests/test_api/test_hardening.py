"""Regression tests for the QA hardening pass.

Each test here pins a hole that was measured OPEN during the full code review and
is now closed. They are written against behaviour a client can observe, so they
keep meaning if the implementation moves.
"""

from __future__ import annotations

import pytest

import config
from api.schemas.score import ScoreRequest


def _minimal_context(description: str = "ok") -> dict:
    return {
        "listing": {"description": description},
        "scoring": {
            "actual_gross_yield": 0.05,
            "expected_gross_yield": 0.05,
            "expected_basis": "model",
            "annual_rent": 1,
            "sale_price": 1,
            "rent_basis": "listing",
            "price_basis": "listing",
            "yield_vs_expected_pp": 0,
        },
        "verdict": {
            "recommend": True,
            "label": "x",
            "reason": "y",
            "confidence": "high",
        },
    }


def test_oversized_body_is_rejected_before_parsing(client):
    body = {"question": "hi?", "context": _minimal_context("A" * 2_000_000)}
    r = client.post("/api/v1/explain", json=body)
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


def test_normal_sized_body_is_not_rejected(client):
    """The cap must not be so tight that a real analysis trips it."""
    r = client.post(
        "/api/v1/explain", json={"question": "why?", "context": _minimal_context()}
    )
    assert r.status_code != 413


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "gopher://internal/",
        "ftp://example.com/x",
        "",
    ],
)
def test_score_rejects_non_web_schemes(url):
    with pytest.raises(Exception):
        ScoreRequest(url=url)


def test_score_accepts_supported_schemes():
    assert ScoreRequest(url="https://www.bayut.com/property/details-1.html")
    assert ScoreRequest(url="fixture://marina_2br")


def test_score_rejects_absurdly_long_url(client):
    r = client.post("/api/v1/score", json={"url": "https://www.bayut.com/" + "a" * 3000})
    assert r.status_code == 422


def test_lookalike_domain_is_rejected(client):
    """`bayut.com.attacker.net` contains "bayut.com" but is not Bayut."""
    r = client.post("/api/v1/score", json={"url": "https://bayut.com.attacker.net/x"})
    assert r.status_code == 422


def test_fixtures_allowed_outside_production(monkeypatch):
    s = config.get_settings()
    monkeypatch.setattr(s, "app_env", "development", raising=False)
    monkeypatch.setattr(s, "listing_fixture_mode", "live", raising=False)
    assert s.fixtures_allowed is True


def test_fixtures_blocked_in_production(monkeypatch):
    s = config.get_settings()
    monkeypatch.setattr(s, "app_env", "production", raising=False)
    monkeypatch.setattr(s, "listing_fixture_mode", "live", raising=False)
    assert s.fixtures_allowed is False


def test_fixtures_allowed_in_production_when_replay_explicit(monkeypatch):
    """Replay mode is an explicit operator choice, and it already logs loudly."""
    s = config.get_settings()
    monkeypatch.setattr(s, "app_env", "production", raising=False)
    monkeypatch.setattr(s, "listing_fixture_mode", "replay", raising=False)
    assert s.fixtures_allowed is True


@pytest.mark.asyncio
async def test_fetch_listing_refuses_fixture_url_in_production(monkeypatch):
    from ingestion.listing_client import UnsupportedListingSite, fetch_listing

    s = config.get_settings()
    monkeypatch.setattr(s, "app_env", "production", raising=False)
    monkeypatch.setattr(s, "listing_fixture_mode", "live", raising=False)
    with pytest.raises(UnsupportedListingSite):
        await fetch_listing("fixture://marina_2br")


def test_predict_does_not_leak_exception_text(client, monkeypatch):
    """A failure inside predict() must not echo the exception to the caller.

    Before the fix this route returned `detail=str(exc)`, so a psycopg error
    would have handed an anonymous caller the DB host/user/name.
    """
    import api.routes.predict as route

    secret = "connection to server at 10.1.2.3, user=admin_svc, db=prod failed"

    def _boom(*a, **k):
        raise RuntimeError(secret)

    monkeypatch.setattr(route, "predict", _boom)
    r = client.post(
        "/api/v1/predict",
        json={
            "community": "Dubai Marina",
            "property_type": "apartment",
            "bedrooms": 2,
            "floor_area_sqft": 900,
        },
    )
    assert r.status_code == 500
    assert secret not in r.text
    assert "10.1.2.3" not in r.text


def test_api_responses_carry_a_restrictive_csp(client):
    h = client.get("/api/v1/health").headers
    assert h.get("Content-Security-Policy") == (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
        "form-action 'none'"
    )
    assert h.get("X-Content-Type-Options") == "nosniff"
    assert h.get("X-Frame-Options") == "DENY"
    assert h.get("Referrer-Policy") == "no-referrer"
    assert "max-age=" in h.get("Strict-Transport-Security", "")
    assert h.get("X-Permitted-Cross-Domain-Policies") == "none"
