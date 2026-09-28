"""Bayut detail-page regression tests (F37).

Found by scoring ONE real Bayut listing
(`bayut.com/property/details-16489473.html`) and reading the actor's own run
log. Invisible to every prior test because they all fed the normalizer a dict
that already had a `description` key - nothing asserted that we ever ASKED the
actor for one.
"""

from __future__ import annotations

import json

import pytest

from ingestion import listing_client as lc


def test_bayut_actor_input_requests_full_details():
    """`fullDetails` defaults to FALSE on memo23/apify-bayut-scraper. Without it
    the actor returns Bayut's search-index record - ~90 fields, no description
    text - so `_classify_character` short-circuited on every listing ever
    scored and the LLM narrative never saw the advertiser's own words."""
    assert lc._ACTOR_INPUT["bayut"]["fullDetails"] is True


def test_dubizzle_gets_no_extra_actor_input():
    """The Dubizzle actor has no `fullDetails` switch - its ordinary output
    already carries `description` - so sending one would be an undeclared
    field."""
    assert "dubizzle" not in lc._ACTOR_INPUT


@pytest.mark.asyncio
async def test_extra_input_is_merged_into_the_actor_payload(monkeypatch):
    """The per-site options must actually reach the actor, and the payload must
    not carry `proxyConfiguration` - that key is not in either actor's input
    schema (the declared name is `proxy`), so it was silently discarded."""
    sent: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [{"externalID": "1"}]

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, endpoint, headers=None, json=None):
            sent.update({"headers": headers, "json": json})
            return _Resp()

    monkeypatch.setattr(lc.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(
        lc, "get_settings",
        lambda: type(
            "S", (), {
                "listings_configured": True,
                "apify_token": "tok",
                "listing_request_timeout_s": 5,
            },
        )(),
    )

    await lc._run_actor_for_url("a~b", "https://www.bayut.com/property/details-1.html",
                                lc._ACTOR_INPUT["bayut"])

    assert sent["json"]["fullDetails"] is True
    assert sent["json"]["maxItems"] == 1
    assert "proxyConfiguration" not in sent["json"]
    assert sent["headers"]["Authorization"] == "Bearer tok"


def test_algolia_fallback_is_flagged_as_gated():
    """Measured on the real listing: bayut.com/api/listing/<id> answers 401, the
    actor's App Check re-mint returns 403, and it pushes the search-index record
    instead. The run SUCCEEDS and returns an item, so a gated listing is
    otherwise indistinguishable from one with a genuinely blank description."""
    out = lc._normalize_bayut({"externalID": "1", "source": "algolia_fallback"}, "u")
    assert out["detail_gated"] is True
    assert out["description"] is None


def test_normal_detail_response_is_not_flagged():
    out = lc._normalize_bayut(
        {"externalID": "1", "source": "detail", "description": "A studio."}, "u"
    )
    assert out["detail_gated"] is False
    assert out["description"] == "A studio."


def test_missing_source_is_not_flagged():
    """Absent provenance must not be read as a failure."""
    assert lc._normalize_bayut({"externalID": "1"}, "u")["detail_gated"] is False


def test_dubizzle_is_never_gated():
    assert lc._normalize_dubizzle({"listing_id": "1"}, "u")["detail_gated"] is False


def test_detail_gated_reaches_the_response_schema():
    from api.schemas.score import ScoredListing

    m = ScoredListing.model_validate(
        lc._normalize_bayut({"externalID": "1", "source": "algolia_fallback"}, "u")
    )
    assert m.detail_gated is True
    assert json.loads(m.model_dump_json())["detail_gated"] is True
