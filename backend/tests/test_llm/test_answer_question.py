"""Tests for the grounded Q&A chain (llm/chains/answer_question.py).

The security property under test is that the prompt context is a WHITELIST: an
attacker (or a buggy client) cannot smuggle extra fields into the model's
context, and the analysis reaches it intact with its provenance notes.
"""

from __future__ import annotations

import json

import pytest

import config
from llm.chains import answer_question


@pytest.fixture
def result() -> dict:
    return {
        "listing": {
            "title": "2BR in Marina",
            "property_type": "Apartment",
            "bedrooms": 2,
            "area_sqft": 1200.0,
            "price": 1_800_000.0,
            "purpose": "for-sale",
            "community": "Dubai Marina",
            "amenities": ["Pool", "Gym"],
            "source": "bayut",
            "listing_url": "https://bayut.com/property/details-1.html",
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
        "model": {
            "expected_gross_yield": 0.0540,
            "segment_matched": True,
            "segment_fallback": False,
            "shap": {"bedrooms": 0.004},
        },
        "baseline": {
            "area": "Marsa Dubai",
            "property_type": "apartment",
            "median_gross_yield": 0.055,
            "median_sale_price": 2_040_000.0,
            "median_annual_rent": 112_200.0,
            "n_sales": 340,
            "n_rents": 512,
            "low_sample": False,
        },
    }


def test_unknown_fields_are_dropped(result):
    result["listing"]["internal_debug_token"] = "leak-me"
    result["secret_admin_block"] = {"db_password": "hunter2"}
    pruned = answer_question.prune_context(result)
    blob = json.dumps(pruned)
    assert "leak-me" not in blob
    assert "hunter2" not in blob


def test_analysis_core_survives_pruning(result):
    pruned = answer_question.prune_context(result)
    assert pruned["scoring"]["rent_basis"] == "model_x_median"
    assert pruned["verdict"]["confidence"] == "high"
    assert pruned["baseline"]["n_sales"] == 340
    assert pruned["model"]["segment_matched"] is True


def test_percentages_are_precomputed(result):
    """The model shouldn't have to multiply fractions by 100 - that invites slips."""
    pruned = answer_question.prune_context(result)
    assert pruned["scoring_percentages"]["actual_gross_yield_pct"] == 6.12
    assert pruned["scoring_percentages"]["expected_gross_yield_pct"] == 5.40


def test_provenance_notes_accompany_the_data(result):
    notes = answer_question.prune_context(result)["_field_notes"]
    assert "ESTIMATED" in notes["rent_basis / price_basis"]
    assert "fractions" in notes["yields"]


def test_character_is_labelled_context_only(result):
    result["character"] = {"finish_tier": "Luxury", "view_type": "Sea", "amenity_score": 4}
    pruned = answer_question.prune_context(result)
    assert "listing_character_context_only" in pruned
    assert "character" not in pruned


def test_pruning_survives_a_minimal_result():
    pruned = answer_question.prune_context({"scoring": None, "verdict": None})
    assert pruned["listing"] == {}


def test_empty_question_rejected(result, monkeypatch, configured_llm):
    with pytest.raises(ValueError):
        answer_question.answer(result, "   ")


def test_oversized_question_rejected(result, monkeypatch, configured_llm):
    with pytest.raises(ValueError):
        answer_question.answer(result, "x" * (answer_question.MAX_QUESTION_CHARS + 1))


def test_raises_when_llm_unconfigured(result, monkeypatch, unconfigured_llm):
    """Unlike the narrative, Q&A has no degraded mode - it must fail loudly."""
    with pytest.raises(RuntimeError, match="not configured"):
        answer_question.answer(result, "Why is the yield above the median?")


@pytest.fixture
def history() -> dict:
    return {
        "available": True,
        "area": "Marsa Dubai",
        "area_requested": "Dubai Marina",
        "property_type": "apartment",
        "points": [
            {
                "period": f"{2015 + i // 4}-Q{i % 4 + 1}",
                "year": 2015 + i // 4,
                "quarter": i % 4 + 1,
                "median_aed_per_sqm": 14_000.0 + i * 50,
                "median_aed_per_sqft": 1300.0 + i * 4.6666,
                "ma4_aed_per_sqft": None,
                "n_sales": 40 + i,
            }
            for i in range(40)
        ],
        "change_pct": 12.5,
        "yoy_change_pct": 3.2,
        "direction": "rising",
        "total_sales": 2_400,
        "basis": "Registered DLD sales.",
        "note": "This is recorded transaction history, not a forecast.",
    }


def test_rent_spread_reaches_the_prompt(result):
    """The percentiles are what let "the highest rent here" be answered."""
    result["baseline"].update(
        {
            "p75_annual_rent": 142_000.0,
            "p90_annual_rent": 190_000.0,
            "max_annual_rent": 45_000_000.0,
        }
    )
    pruned = answer_question.prune_context(result)
    assert pruned["baseline"]["p90_annual_rent"] == 190_000.0
    assert pruned["baseline"]["max_annual_rent"] == 45_000_000.0


def test_percentile_note_warns_that_the_max_can_be_a_bulk_deal(result):
    """The raw maximum is a real outlier risk, not a hypothetical one.

    Measured against the live snapshot: the largest Ejari annual rent recorded
    against Marsa Dubai apartments is AED 45,000,000, which is a bulk lease, not
    a flat. The model must be told to lead with p90 and label the maximum.
    """
    notes = answer_question.prune_context(result)["_field_notes"]
    text = " ".join(notes.values()).lower()
    assert "p90" in text
    assert "bulk" in text or "whole-building" in text


def test_price_history_is_carried_and_compacted(result, history):
    result["price_history"] = history
    pruned = answer_question.prune_context(result)
    carried = pruned["community_price_history"]
    assert carried["direction_of_last_4_quarters_vs_previous_4"] == "rising"
    assert len(carried["quarters"]) == answer_question.MAX_HISTORY_QUARTERS
    assert carried["quarters"][-1]["period"] == history["points"][-1]["period"]


def test_price_history_absent_is_simply_absent(result):
    """No chart drawn means no series, not an empty one to reason over."""
    assert "community_price_history" not in answer_question.prune_context(result)
    result["price_history"] = {"available": False, "points": []}
    assert "community_price_history" not in answer_question.prune_context(result)


def test_price_history_note_forbids_extending_it(result, history):
    result["price_history"] = history
    pruned = answer_question.prune_context(result)
    note = pruned["_field_notes"]["community_price_history"].lower()
    assert "forecast" in note
    assert "forecast" in pruned["community_price_history"]["note"].lower()


def test_history_stays_within_the_list_clamp(result, history, monkeypatch):
    """The trim has to sit under llm_max_list_items or the clamp truncates it.

    A truncated series would lose its newest quarters silently, which is the
    failure this window exists to prevent.
    """
    assert answer_question.MAX_HISTORY_QUARTERS <= config.get_settings().llm_max_list_items
    result["price_history"] = history
    dumped = json.dumps(answer_question.prune_context(result))
    assert "omitted" not in dumped
