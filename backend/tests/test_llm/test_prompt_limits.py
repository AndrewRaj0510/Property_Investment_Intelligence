"""Tests for the prompt input bounds (F19 review).

`/explain` accepts a CLIENT-SUPPLIED analysis. The field whitelist in
`prune_context` controls *which* fields reach the model but not how large they
are - during review an unbounded `title` pushed ~176k tokens into a single
request. Rate limiting caps requests per minute, not tokens per request, so the
size bound has to live here.
"""

from __future__ import annotations

import json

import config
from llm.chains.answer_question import prune_context
from llm.provider import clamp_for_prompt

BASE_SCORING = {
    "actual_gross_yield": 0.06,
    "expected_gross_yield": 0.05,
    "expected_basis": "model",
    "annual_rent": 1.0,
    "sale_price": 1.0,
    "rent_basis": "listing",
    "price_basis": "listing",
    "yield_vs_expected_pp": 1.0,
}
BASE_VERDICT = {"recommend": True, "label": "x", "reason": "r", "confidence": "high"}


def test_long_string_is_truncated_and_marked():
    limit = config.get_settings().llm_max_field_chars
    out = clamp_for_prompt("A" * (limit * 10))
    assert len(out) < limit + 50
    assert "[truncated]" in out


def test_short_string_is_untouched():
    assert clamp_for_prompt("a normal title") == "a normal title"


def test_long_list_is_capped_and_remainder_noted():
    limit = config.get_settings().llm_max_list_items
    out = clamp_for_prompt([f"item{i}" for i in range(limit + 25)])
    assert len(out) == limit + 1
    assert "25 more omitted" in out[-1]


def test_nested_structures_are_clamped():
    limit = config.get_settings().llm_max_field_chars
    out = clamp_for_prompt({"a": {"b": ["X" * (limit * 5)]}})
    assert "[truncated]" in out["a"]["b"][0]


def test_numbers_and_none_pass_through():
    assert clamp_for_prompt(5) == 5
    assert clamp_for_prompt(1.25) == 1.25
    assert clamp_for_prompt(None) is None
    assert clamp_for_prompt(True) is True


def test_oversized_context_cannot_blow_up_the_prompt():
    """The measured attack: unbounded title + amenities on a client-supplied
    context. Must land in a sane size band, not ~700k chars."""
    ctx = {
        "listing": {
            "title": "A" * 200_000,
            "amenities": ["x" * 1000] * 500,
            "community": "Marina",
        },
        "scoring": BASE_SCORING,
        "verdict": BASE_VERDICT,
    }
    size = len(json.dumps(prune_context(ctx)))
    assert size < 100_000, f"prompt context was {size} chars"


def test_normal_context_is_not_degraded():
    """A realistic analysis must survive clamping completely unchanged."""
    ctx = {
        "listing": {
            "title": "Spacious 2BR with Full Marina View",
            "amenities": ["Pool", "Gym", "Covered Parking"],
            "community": "Dubai Marina",
        },
        "scoring": BASE_SCORING,
        "verdict": BASE_VERDICT,
    }
    pruned = prune_context(ctx)
    assert pruned["listing"]["title"] == "Spacious 2BR with Full Marina View"
    assert pruned["listing"]["amenities"] == ["Pool", "Gym", "Covered Parking"]
    assert "truncated" not in json.dumps(pruned)
