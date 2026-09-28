"""Endpoint tests for POST /api/v1/explain (grounded Q&A).

The LLM call itself is patched out - what matters here is the contract around
it: schema validation, the fail-loud 503 when OpenAI is absent, and error
mapping.
"""

from __future__ import annotations

import pytest

import config
from api.routes import explain as explain_route

VALID_CONTEXT = {
    "listing": {
        "title": "2BR in Marina",
        "property_type": "Apartment",
        "bedrooms": 2,
        "area_sqft": 1200.0,
        "price": 1_800_000.0,
        "purpose": "for-sale",
        "community": "Dubai Marina",
    },
    "scoring": {
        "actual_gross_yield": 0.0612,
        "expected_gross_yield": 0.0540,
        "expected_basis": "model",
        "annual_rent": 110_160.0,
        "sale_price": 1_800_000.0,
        "rent_basis": "model_x_median",
        "price_basis": "listing",
        "yield_vs_expected_pp": 0.72,
    },
    "verdict": {
        "recommend": True,
        "label": "Good investment",
        "reason": "Actual 6.12% vs expected 5.40%.",
        "confidence": "high",
    },
}


@pytest.fixture
def stub_answer(monkeypatch):
    """Patch the chain so no network call is made."""
    monkeypatch.setattr(
        explain_route, "answer", lambda ctx, q: "The rent is an estimate from DLD."
    )


def test_answers_a_grounded_question(client, stub_answer):
    r = client.post(
        "/api/v1/explain",
        json={"question": "Where did the rent come from?", "context": VALID_CONTEXT},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "The rent is an estimate from DLD."
    assert body["question"] == "Where did the rent come from?"


def test_503_when_llm_unconfigured(client, unconfigured_llm):
    r = client.post(
        "/api/v1/explain",
        json={"question": "Why is the confidence high?", "context": VALID_CONTEXT},
    )
    assert r.status_code == 503
    assert "OLLAMA" in r.json()["detail"]


def test_rejects_empty_question(client, stub_answer):
    r = client.post(
        "/api/v1/explain", json={"question": "", "context": VALID_CONTEXT}
    )
    assert r.status_code == 422


def test_rejects_oversized_question(client, stub_answer):
    r = client.post(
        "/api/v1/explain",
        json={"question": "x" * 501, "context": VALID_CONTEXT},
    )
    assert r.status_code == 422


def test_rejects_malformed_context(client, stub_answer):
    """A hand-crafted context missing required scoring fields is refused."""
    r = client.post(
        "/api/v1/explain",
        json={"question": "What is the yield?", "context": {"listing": {}}},
    )
    assert r.status_code == 422


def test_chain_failure_maps_to_502(client, monkeypatch, configured_llm):

    def boom(ctx, q):
        raise Exception("upstream timeout")

    monkeypatch.setattr(explain_route, "answer", boom)
    r = client.post(
        "/api/v1/explain",
        json={"question": "Why?", "context": VALID_CONTEXT},
    )
    assert r.status_code == 502


PRICE_HISTORY = {
    "area": "Marsa Dubai",
    "area_requested": "Dubai Marina",
    "property_type": "apartment",
    "points": [
        {
            "period": "2024-Q1",
            "year": 2024,
            "quarter": 1,
            "median_aed_per_sqm": 16_000.0,
            "median_aed_per_sqft": 1486.4,
            "n_sales": 120,
        },
        {
            "period": "2024-Q2",
            "year": 2024,
            "quarter": 2,
            "median_aed_per_sqm": 16_400.0,
            "median_aed_per_sqft": 1523.6,
            "n_sales": 131,
        },
    ],
    "direction": "rising",
    "basis": "Registered DLD sales, median price per sqm by quarter.",
    "note": "This is recorded transaction history, not a forecast.",
    "available": True,
}


def test_price_history_reaches_the_chain(client, monkeypatch):
    """The chart is a separate endpoint, so the client has to hand it back.

    Without this the model could not answer a question about a chart the reader
    was looking at while asking it.
    """
    seen: dict = {}

    def capture(ctx, q):
        seen.update(ctx)
        return "Prices rose across the window shown."

    monkeypatch.setattr(explain_route, "answer", capture)
    r = client.post(
        "/api/v1/explain",
        json={
            "question": "What has this community done on price?",
            "context": VALID_CONTEXT,
            "price_history": PRICE_HISTORY,
        },
    )
    assert r.status_code == 200
    assert seen["price_history"]["direction"] == "rising"
    assert len(seen["price_history"]["points"]) == 2


def test_price_history_is_optional(client, monkeypatch):
    """A client that drew no chart sends nothing, and the question still works."""
    seen: dict = {}

    def capture(ctx, q):
        seen.update(ctx)
        return "ok"

    monkeypatch.setattr(explain_route, "answer", capture)
    r = client.post(
        "/api/v1/explain",
        json={"question": "Why is confidence high?", "context": VALID_CONTEXT},
    )
    assert r.status_code == 200
    assert "price_history" not in seen


def test_malformed_price_history_is_rejected(client, stub_answer):
    """It is validated like the analysis is, not passed through to the prompt."""
    r = client.post(
        "/api/v1/explain",
        json={
            "question": "What has this community done on price?",
            "context": VALID_CONTEXT,
            "price_history": {"points": [{"period": "2024-Q1", "quarter": 9}]},
        },
    )
    assert r.status_code == 422
