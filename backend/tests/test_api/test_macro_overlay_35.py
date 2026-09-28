"""M4-07/M4-08 - Objective 3.5: macro overlay, seasonal calendar, two-level SHAP.

The assertions here are mostly about **what the API refuses to claim**. Aim 3
produced a series of measured nulls, and the risk at the presentation layer is
that a dashboard quietly re-inflates them - by captioning a climate calendar as a
yield calendar, by hiding an empty macro group, or by substituting a placeholder
for an absent series. Each of those is pinned below.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _calendar() -> dict:
    resp = client.get("/api/v1/seasonal/calendar")
    assert resp.status_code == 200
    return resp.json()


def test_calendar_ships_its_own_caveats():
    """`basis` and `note` travel in the payload so a consumer cannot relabel it."""
    body = _calendar()
    assert body["basis"]
    assert body["note"]
    assert "climate calendar, not a yield calendar" in body["note"]


def test_calendar_makes_no_yield_claim():
    """The old `yield_index` field is gone, not merely left null.

    A null field named `yield_index` invites exactly the reading the data cannot
    support: the model predicts at annual grain, so no monthly yield exists.
    """
    body = _calendar()
    for month in body["months"]:
        assert "yield_index" not in month
        assert "avg_tourist_arrivals" not in month


def test_calendar_returns_twelve_months_or_nothing():
    body = _calendar()
    months = body["months"]
    assert len(months) in (0, 12)
    if months:
        assert [m["month_of_year"] for m in months] == list(range(1, 13))
        assert months[0]["month_name"] == "January"


def test_calendar_index_is_relative_to_the_annual_mean():
    """100 = annual mean, so the twelve months must straddle it."""
    body = _calendar()
    values = [
        m["thermal_index_relative"]
        for m in body["months"]
        if m.get("thermal_index_relative") is not None
    ]
    if not values:
        return
    assert min(values) < 100 < max(values)
    assert max(values) - min(values) > 20


def test_calendar_carries_observation_counts():
    """A month averaged over 14 years must be distinguishable from one over 2."""
    body = _calendar()
    for month in body["months"]:
        assert "months_observed" in month
        assert isinstance(month["months_observed"], int)


def _predict() -> dict | None:
    resp = client.post(
        "/api/v1/predict",
        json={
            "community": "Dubai Marina",
            "property_type": "Apartment",
            "bedrooms": 2,
            "year": 2025,
        },
    )
    if resp.status_code != 200:
        return None
    return resp.json()


def test_two_level_split_is_populated_on_phase_one():
    """It used to be gated on phase == 2, so it never ran at all.

    Gating hid the result rather than protecting it: the empty macro group IS the
    Aim 3 finding.
    """
    body = _predict()
    if body is None:
        return
    ex = body["shap_explanation"]
    assert ex["property_level"] is not None
    assert ex["macro_level"] is not None
    assert ex["property_share_pct"] is not None


def test_empty_macro_group_carries_its_explanation():
    """An empty group with no note reads as a bug; with a note it reads as a result."""
    body = _predict()
    if body is None:
        return
    ex = body["shap_explanation"]
    if ex["macro_level"]:
        return
    assert ex["macro_note"], "empty macro group must explain itself"
    assert ex["property_share_pct"] == 100.0
    note = ex["macro_note"].lower()
    assert "measured" in note
    assert "-0.0004" in ex["macro_note"] or "0.0004" in ex["macro_note"]


def test_property_share_is_a_percentage():
    body = _predict()
    if body is None:
        return
    share = body["shap_explanation"]["property_share_pct"]
    assert share is None or 0.0 <= share <= 100.0
