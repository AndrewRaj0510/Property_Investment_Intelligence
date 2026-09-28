"""Grounded Q&A chain (M3D-05) - live-only.

Backs POST /api/v1/explain. The user asks a free-text question about a score
result they already hold; the model answers strictly from that JSON.

Two guards keep this from becoming an open-ended chatbot that can invent Dubai
market facts:

  * **Context pruning** (``prune_context``) - only whitelisted fields are sent,
    so the prompt carries the analysis and nothing else.
  * **A refusal contract** in the prompt - if the answer is not in the JSON the
    model must say what is missing rather than fill the gap.

Unlike the narrative chain, this one RAISES when OpenAI is absent: a Q&A
endpoint with no model has no degraded mode to offer.
"""

from __future__ import annotations

import json

import structlog

from config import get_settings
from llm.prompts.qa_grounded import QA_GROUNDED_HUMAN, QA_GROUNDED_INSTRUCTIONS

log = structlog.get_logger(__name__)

_chain = None

MAX_QUESTION_CHARS = 500

_LISTING_FIELDS = (
    "title", "property_type", "bedrooms", "bathrooms", "area_sqft", "price",
    "purpose", "rent_frequency", "annual_rent", "community", "amenities",
    "source", "listing_url",
)

MAX_HISTORY_QUARTERS = 32


def _get_chain():
    global _chain
    if _chain is not None:
        return _chain
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from llm.provider import get_chat_model

    llm = get_chat_model(temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", QA_GROUNDED_INSTRUCTIONS), ("human", QA_GROUNDED_HUMAN)]
    )
    _chain = prompt | llm | StrOutputParser()
    return _chain


def prune_context(result: dict) -> dict:
    """Whitelist the analysis fields the model may see."""
    listing = result.get("listing") or {}
    pruned = {
        "listing": {k: listing.get(k) for k in _LISTING_FIELDS if listing.get(k) is not None},
        "scoring": result.get("scoring"),
        "verdict": result.get("verdict"),
    }
    scoring = result.get("scoring") or {}
    if scoring:
        pruned["scoring_percentages"] = {
            "actual_gross_yield_pct": round(scoring["actual_gross_yield"] * 100, 2),
            "expected_gross_yield_pct": round(scoring["expected_gross_yield"] * 100, 2),
        }
    if result.get("model"):
        model = result["model"]
        pruned["model"] = {
            "expected_gross_yield": model.get("expected_gross_yield"),
            "segment_matched": model.get("segment_matched"),
            "segment_fallback": model.get("segment_fallback"),
            "drivers": model.get("shap"),
        }
    if result.get("baseline"):
        pruned["baseline"] = result["baseline"]
    if result.get("character"):
        pruned["listing_character_context_only"] = result["character"]
    if result.get("narrative"):
        pruned["existing_narrative"] = result["narrative"]
    history = _price_history(result.get("price_history"))
    if history:
        pruned["community_price_history"] = history
    pruned["_field_notes"] = {
        "rent_basis / price_basis": (
            "'listing' = the figure was advertised; anything else means it was "
            "ESTIMATED from DLD/Ejari comparables and must be described as an "
            "estimate."
        ),
        "expected_basis": (
            "'model' = the trained model matched this segment; 'area_median' = "
            "fell back to the community median; 'model_low_confidence' = no "
            "segment match, weak evidence."
        ),
        "yields": "All *_gross_yield values are fractions (0.065 = 6.5%).",
        "baseline percentiles": (
            "baseline.p25/p75/p90_annual_rent and the _sale_price equivalents "
            "describe the SPREAD of recorded DLD sales and Ejari contracts for "
            "this whole community and property type, across all unit sizes. "
            "For 'what is the highest rent/price here', lead with p90 (the top "
            "of the normal range) and give the count of records behind it. "
            "max_annual_rent and max_sale_price are RAW maxima and can be a "
            "bulk or whole-building transaction rather than a single home, so "
            "quote them only as the largest single record and say that."
        ),
        "community_price_history": (
            "Registered DLD sale prices per square foot for this community and "
            "type, by calendar quarter. It is MEASURED HISTORY: the last point "
            "is the last complete quarter, there is no projected point, and "
            "direction/yoy_change_pct compare the last four measured quarters "
            "against the four before them. Never extend it into a forecast."
        ),
    }
    from llm.provider import clamp_for_prompt

    return clamp_for_prompt(pruned)


def _price_history(history: dict | None) -> dict | None:
    """Compact the price-history series for the prompt, or None.

    The chart beside the analysis is a separate endpoint, so a reader asking
    "what has this community done on price?" was asking about something the
    model could not see. This puts the same series in the context, trimmed to
    the recent window and to the fields that carry meaning.
    """
    if not history or not history.get("available"):
        return None
    points = history.get("points") or []
    if not points:
        return None
    recent = points[-MAX_HISTORY_QUARTERS:]
    return {
        "area": history.get("area"),
        "property_type": history.get("property_type"),
        "covers": f"{recent[0].get('period')} to {recent[-1].get('period')}",
        "direction_of_last_4_quarters_vs_previous_4": history.get("direction"),
        "change_pct_across_this_window": history.get("change_pct"),
        "last_4q_vs_previous_4q_change_pct": history.get("yoy_change_pct"),
        "total_sales_in_series": history.get("total_sales"),
        "basis": history.get("basis"),
        "note": history.get("note"),
        "quarters": [
            {
                "period": pt.get("period"),
                "median_aed_per_sqft": round(pt["median_aed_per_sqft"], 1)
                if pt.get("median_aed_per_sqft") is not None
                else None,
                "n_sales": pt.get("n_sales"),
            }
            for pt in recent
        ],
    }


def answer(result: dict, question: str) -> str:
    """Answer one question about one score result. Raises if OpenAI is absent."""
    settings = get_settings()
    if not settings.llm_configured:
        raise RuntimeError(
            "The LLM is not configured; grounded Q&A requires OLLAMA_BASE_URL "
            "and OLLAMA_MODEL (no template fallback)."
        )
    q = (question or "").strip()
    if not q:
        raise ValueError("Question must not be empty.")
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"Question is too long ({len(q)} chars; max {MAX_QUESTION_CHARS})."
        )
    from llm.provider import clean_output

    context_json = json.dumps(prune_context(result), indent=2, default=str)
    raw = _get_chain().invoke({"context_json": context_json, "question": q})
    return clean_output(raw)
