"""Tests for the score-narrative chain (llm/chains/explain_score.py).

These test the DETERMINISTIC half - context building, provenance labelling and
graceful degradation. No OpenAI call is made; the LLM's prose is not asserted on
(it isn't deterministic), only the labelled inputs it is handed and the
guarantee that a failure never costs the caller their numeric verdict.
"""

from __future__ import annotations

import pathlib

import pytest

import config
from llm.chains import explain_score


@pytest.fixture
def result() -> dict:
    """A for-sale scorer result with a model-matched segment."""
    return {
        "listing": {
            "source": "bayut",
            "title": "2BR in Marina",
            "description": "Upgraded apartment with full sea view.",
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
            "reason": "Actual gross yield 6.12% vs the model's expected 5.40%.",
            "confidence": "high",
        },
        "model": {
            "expected_gross_yield": 0.0540,
            "segment_matched": True,
            "segment_fallback": False,
            "shap": {"bedrooms": 0.004, "area_sqm": -0.002, "community": 0.001},
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


def test_estimated_rent_is_labelled_as_an_estimate(result):
    ctx = explain_score.build_context(result)
    assert "ESTIMATED" in ctx["rent_basis"]
    assert "not stated on the listing" in ctx["rent_basis"]


def test_listing_price_is_labelled_as_real(result):
    ctx = explain_score.build_context(result)
    assert "stated on the listing itself" in ctx["price_basis"]
    assert "ESTIMATED" not in ctx["price_basis"]


def test_every_basis_label_the_scorer_emits_has_prompt_text():
    """A basis label with no mapping would reach the prompt as a bare token,
    losing the provenance warning. Guard the full vocabulary."""
    emitted = {
        "listing", "model", "model_x_median", "model_implied",
        "area_median", "model_low_confidence",
    }
    assert emitted <= set(explain_score._BASIS_TEXT)


def test_yields_are_rendered_as_percentages(result):
    ctx = explain_score.build_context(result)
    assert ctx["actual_yield"] == "6.12"
    assert ctx["expected_yield"] == "5.40"
    assert ctx["diff_pp"] == "+0.72"


def test_money_is_formatted_with_separators(result):
    ctx = explain_score.build_context(result)
    assert ctx["sale_price"] == "1,800,000"


def test_insufficient_data_forbids_implying_a_comparison(result):
    result["verdict"]["label"] = "Insufficient data"
    result["verdict"]["confidence"] = "very_low"
    result["baseline"] = None
    ctx = explain_score.build_context(result)
    assert "NO COMPARABLE SALES HISTORY" in ctx["caveats"]
    assert "must not imply any comparison" in ctx["caveats"]


def test_low_confidence_carries_an_explanation(result):
    result["verdict"]["confidence"] = "low"
    ctx = explain_score.build_context(result)
    assert "indicative only" in ctx["caveats"]


def test_missing_baseline_is_stated_not_omitted(result):
    result["baseline"] = None
    ctx = explain_score.build_context(result)
    assert "no comparable DLD" in ctx["baseline_summary"]


def test_low_sample_baseline_is_flagged(result):
    result["baseline"]["low_sample"] = True
    ctx = explain_score.build_context(result)
    assert "SMALL SAMPLE" in ctx["baseline_summary"]


def test_character_is_marked_as_not_a_model_input(result):
    result["character"] = {
        "finish_tier": "Luxury", "view_type": "Sea", "amenity_score": 4
    }
    ctx = explain_score.build_context(result)
    assert "NOT inputs to the yield model" in ctx["character"]
    assert "Luxury finish" in ctx["character"]


def test_character_absent_is_not_assessed(result):
    assert explain_score.build_context(result)["character"] == "not assessed"


def test_view_type_none_is_omitted(result):
    result["character"] = {"finish_tier": "Mid", "view_type": "None", "amenity_score": 1}
    ctx = explain_score.build_context(result)
    assert "view" not in ctx["character"].split(" - ")[0]


def test_drivers_ranked_by_absolute_contribution(result):
    ctx = explain_score.build_context(result)
    assert ctx["drivers"].index("bedrooms") < ctx["drivers"].index("area sqm")
    assert "raises" in ctx["drivers"] and "lowers" in ctx["drivers"]


def test_drivers_absent_when_no_shap(result):
    result["model"]["shap"] = None
    assert explain_score.build_context(result)["drivers"] == "not available for this segment"


def test_studio_rendered_as_studio_not_zero_bedroom(result):
    result["listing"]["bedrooms"] = 0
    assert "studio" in explain_score.build_context(result)["listing_summary"]


def test_missing_size_is_stated(result):
    result["listing"]["area_sqft"] = None
    assert "size not stated" in explain_score.build_context(result)["listing_summary"]


def test_returns_none_when_llm_unconfigured(result, monkeypatch, unconfigured_llm):
    assert explain_score.narrate_score(result) is None


def test_returns_none_when_chain_raises(result, monkeypatch, configured_llm):

    def boom():
        raise RuntimeError("ollama is down")

    monkeypatch.setattr(explain_score, "_get_chain", boom)
    assert explain_score.narrate_score(result) is None


def test_returns_none_on_malformed_result(monkeypatch, configured_llm):
    """A result missing 'scoring' must degrade, not raise into the endpoint."""
    assert explain_score.narrate_score({"listing": {}}) is None


def _json(**kwargs) -> str:
    import json

    return json.dumps(kwargs)


def test_parses_the_three_part_shape():
    out = explain_score.parse_narrative(
        _json(
            intro="A studio in JVT.",
            points=[
                {"label": "What it earns", "detail": "8.30%, about AED 53,927 a year."},
                {"label": "Against the area", "detail": "Comparables return 7.17%."},
            ],
            action="Check the service charge before committing.",
        )
    )
    assert out["intro"] == "A studio in JVT."
    assert [p["label"] for p in out["points"]] == ["What it earns", "Against the area"]
    assert out["action"].startswith("Check the service charge")


def test_json_inside_a_code_fence_is_still_parsed():
    """Models fence JSON constantly, whatever the prompt says."""
    out = explain_score.parse_narrative(
        "```json\n" + _json(intro="I.", points=[], action="A.") + "\n```"
    )
    assert out["intro"] == "I."


def test_plain_prose_is_kept_as_the_intro_not_discarded():
    """A model that answers in prose has still answered.

    Returning None here would blank the explanation card over a formatting
    miss, which is the opposite of the trade this layer is for.
    """
    out = explain_score.parse_narrative("The property yields 8.3% a year.")
    assert out["intro"] == "The property yields 8.3% a year."
    assert out["points"] == []


def test_points_flattened_to_strings_are_still_points():
    out = explain_score.parse_narrative(
        _json(intro="I.", points=["First finding.", "Second finding."], action="A.")
    )
    assert [p["detail"] for p in out["points"]] == ["First finding.", "Second finding."]
    assert all(p["label"] == "" for p in out["points"])


def test_empty_response_yields_none():
    assert explain_score.parse_narrative("") is None
    assert explain_score.parse_narrative("   ") is None


def test_every_field_is_dash_sanitised():
    """Each field goes through clean_output, not just the response as a whole."""
    em = chr(0x2014)
    out = explain_score.parse_narrative(
        _json(
            intro=f"Intro {em} here.",
            points=[{"label": f"A{em}B", "detail": f"Detail {em} here."}],
            action=f"Action {em} here.",
        )
    )
    blob = out["intro"] + out["points"][0]["label"] + out["points"][0]["detail"] + out["action"]
    assert em not in blob


def test_reasoning_markup_is_stripped_before_the_json_is_found():
    out = explain_score.parse_narrative(
        "<think>weighing it up</think>" + _json(intro="I.", points=[], action="A.")
    )
    assert out["intro"] == "I."


def test_flatten_keeps_every_part_for_the_string_field():
    """`narrative` stays a string for /explain and the UI fallback."""
    flat = explain_score.flatten_narrative(
        {
            "intro": "Intro.",
            "points": [{"label": "Label", "detail": "Detail."}],
            "action": "Action.",
        }
    )
    assert "Intro." in flat and "Label: Detail." in flat and "Action." in flat


def test_point_count_is_bounded():
    """A model that returns twenty findings does not get to render twenty."""
    out = explain_score.parse_narrative(
        _json(
            intro="I.",
            points=[{"label": f"L{i}", "detail": f"D{i}."} for i in range(20)],
            action="A.",
        )
    )
    assert len(out["points"]) == explain_score.MAX_POINTS


def test_prompt_forbids_forecasts_and_buy_sell_instructions():
    """The closing section gives due diligence, never a price call.

    There is no price model behind this platform, so a forecast in the one
    part of the output written as advice would be a claim with no method
    behind it.
    """
    from llm.prompts.score_narrative import SCORE_NARRATIVE_INSTRUCTIONS

    text = SCORE_NARRATIVE_INSTRUCTIONS
    assert "NO FORECASTS AND NO BUY/SELL INSTRUCTION" in text
    assert "EXPLAIN EVERY FIGURE" in text
    assert "em dash" in text


MEASURED_NARRATIVE_SECONDS = 27


def test_narrative_budget_clears_the_measured_generation():
    """A budget below the measured time is a permanently broken narrative."""
    budget = config.get_settings().llm_narrate_timeout_s
    assert budget >= MEASURED_NARRATIVE_SECONDS * 1.5, (
        f"The narrative takes about {MEASURED_NARRATIVE_SECONDS}s and is given "
        f"{budget}s. It fails silently, so too small a budget looks like a "
        "missing explanation, not an error."
    )


def test_narrative_gets_a_larger_budget_than_classification():
    """Two different jobs, two different budgets.

    Classification returns three short labels; the narrative returns an intro,
    several explained findings and a closing section. One number for both is
    what caused the failure.
    """
    s = config.get_settings()
    assert s.llm_narrate_timeout_s > s.llm_enrich_timeout_s


def test_chain_is_built_with_the_narrative_budget(monkeypatch, configured_llm):
    """The chain has to actually ASK for that budget, not just have one defined."""
    captured: dict = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here, the chain is not needed")

    monkeypatch.setattr(explain_score, "_chain", None)
    import llm.provider as provider

    monkeypatch.setattr(provider, "get_chat_model", fake_model)
    with pytest.raises(RuntimeError):
        explain_score._get_chain()
    monkeypatch.setattr(explain_score, "_chain", None)

    assert captured["timeout_s"] == config.get_settings().llm_narrate_timeout_s
    assert captured["num_predict"] >= 1200


def test_enrichment_budget_is_the_sum_of_both_calls():
    """The outer bound has to cover both inner ones or it cancels the second.

    It used to be `enrich * 2`, which was correct only while the two calls
    shared a budget.
    """
    import services.listing_scorer as scorer

    src = pathlib.Path(scorer.__file__).read_text(encoding="utf-8")
    assert "llm_enrich_timeout_s + settings.llm_narrate_timeout_s" in src
    s = config.get_settings()
    assert s.llm_enrich_timeout_s + s.llm_narrate_timeout_s + 5 <= 180, (
        "The whole enrichment must still fit inside the client's score timeout "
        "alongside a 90s listing fetch."
    )


def test_the_three_timeouts_still_form_a_chain():
    """Server worst case < browser client timeout < Next's proxy timeout.

    Three numbers in three files, and they only work as an ordered chain: the
    proxy must outlast the client so the client is the one that gives up (it can
    show a reason; a dead proxy returns a bare 500), and the client must outlast
    the server or work that succeeded is thrown away. Raising the narrative
    budget moved the server end of the chain, and the proxy number is the easy
    one to forget, so it is pinned here rather than rediscovered in a browser.
    """
    import re

    s = config.get_settings()
    front = pathlib.Path(__file__).resolve().parents[2].parent / "frontend" / "src"
    api_ts = (front / "lib" / "api.ts").read_text(encoding="utf-8")
    cfg = (front.parent / "next.config.mjs").read_text(encoding="utf-8")

    client_ms = int(
        re.search(r"SCORE_TIMEOUT_MS = ([\d_]+)", api_ts).group(1).replace("_", "")
    )
    proxy_ms = int(
        re.search(r"proxyTimeout: ([\d_]+)", cfg).group(1).replace("_", "")
    )
    server_s = (
        s.listing_request_timeout_s
        + s.llm_enrich_timeout_s
        + s.llm_narrate_timeout_s
        + 5
    )

    assert server_s * 1000 <= client_ms, (
        f"The backend can take {server_s}s but the browser gives up at "
        f"{client_ms / 1000}s, so a successful score would be discarded."
    )
    assert client_ms < proxy_ms, (
        "Next's proxy must outlast the client, or the browser gets a bare 500 "
        "instead of a timeout it can explain."
    )
