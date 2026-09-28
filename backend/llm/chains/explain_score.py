"""Live-listing verdict narrative chain (M3D-04) - live-only.

Composes the investor-facing explanation for POST /api/v1/score from the
scorer's structured result. Unlike ``explain_shap`` (which 503s the whole
prediction when OpenAI is absent), this chain is an OPTIONAL presentation layer:
``narrate_score`` returns ``None`` rather than raising, so a working numeric
verdict is never thrown away because the prose failed.

That is not a mock fallback - no text is fabricated when the LLM is unavailable;
the field is simply absent and the UI renders the deterministic ``reason``.
"""

from __future__ import annotations

import structlog

from config import get_settings
from llm.prompts.score_narrative import (
    SCORE_NARRATIVE_HUMAN,
    SCORE_NARRATIVE_INSTRUCTIONS,
)

log = structlog.get_logger(__name__)

_chain = None

_BASIS_TEXT = {
    "listing": "stated on the listing itself (a real advertised figure)",
    "model": "ESTIMATED by the model, not stated on the listing",
    "model_x_median": (
        "ESTIMATED - the model's expected yield applied to the comparable DLD "
        "median sale price, not stated on the listing"
    ),
    "model_implied": (
        "ESTIMATED - backed out from the rent and the expected yield, not "
        "stated on the listing"
    ),
    "area_median": (
        "the median for this community and property type from DLD records, not "
        "stated on the listing"
    ),
    "model_low_confidence": (
        "the model's estimate WITHOUT a matching market segment - weak evidence"
    ),
}

_CONFIDENCE_CAVEAT = {
    "high": "None - the model matched this exact community/type/bedroom segment.",
    "normal": (
        "The comparable sample for this segment is small (low_sample), so the "
        "baseline is thinner than usual."
    ),
    "low": (
        "Treat this as indicative only. Either the model had no exact segment "
        "match, or the listing omitted details the model needs, or the "
        "sale-vs-rent purpose had to be inferred."
    ),
    "very_low": (
        "There is no independent comparable sale price for this area and type, "
        "so the actual and expected yields are not genuinely comparable and no "
        "verdict on the asking price is possible."
    ),
}


def _get_chain():
    global _chain
    if _chain is not None:
        return _chain
    from langchain_core.messages import SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

    from llm.provider import get_chat_model

    llm = get_chat_model(
        temperature=0.3,
        timeout_s=get_settings().llm_narrate_timeout_s,
        num_predict=1400,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SCORE_NARRATIVE_INSTRUCTIONS),
            HumanMessagePromptTemplate.from_template(SCORE_NARRATIVE_HUMAN),
        ]
    )
    _chain = prompt | llm | StrOutputParser()
    return _chain


def _fmt_money(v) -> str:
    if v is None:
        return "not available"
    return f"{float(v):,.0f}"


def _listing_summary(listing: dict) -> str:
    beds = listing.get("bedrooms")
    bed_txt = "studio" if beds == 0 else (f"{beds}-bedroom" if beds else "unknown-bed")
    size = listing.get("area_sqft")
    size_txt = f"{float(size):,.0f} sqft" if size else "size not stated"
    ptype = listing.get("property_type") or "property"
    community = listing.get("community") or "an unspecified community"
    return f"{bed_txt} {ptype} in {community}, {size_txt}"


def _drivers_text(shap: dict | None) -> str:
    """Top SHAP drivers as plain phrases, strongest absolute contribution first."""
    if not shap:
        return "not available for this segment"
    ranked = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    parts = [
        f"{feat.replace('_', ' ')} ({'raises' if val > 0 else 'lowers'} the "
        f"expected yield)"
        for feat, val in ranked
        if val
    ]
    return "; ".join(parts) if parts else "not available for this segment"


def _character_text(character: dict | None) -> str:
    """Listing character (finish/view/amenities) from the classification chain.

    Explicitly labelled as context because these are NOT model features - the
    DLD/Ejari training data has no finish or view columns, so letting the
    narrative imply they drove the yield would misrepresent the model.
    """
    if not character:
        return "not assessed"
    bits = []
    if character.get("finish_tier"):
        bits.append(f"{character['finish_tier']} finish")
    if character.get("view_type") and character["view_type"].lower() != "none":
        bits.append(f"{character['view_type']} view")
    if character.get("amenity_score") is not None:
        bits.append(f"amenity score {character['amenity_score']}/5")
    if not bits:
        return "not assessed"
    return (
        ", ".join(bits)
        + " - these describe the unit but are NOT inputs to the yield model, so "
        "mention them only as possible context for how it is priced"
    )


def _baseline_text(baseline: dict | None) -> str:
    if not baseline:
        return "none - no comparable DLD sales/rent bucket for this area and type"
    txt = (
        f"median gross yield {baseline['median_gross_yield'] * 100:.2f}%, "
        f"median sale price AED {_fmt_money(baseline['median_sale_price'])}, "
        f"median annual rent AED {_fmt_money(baseline['median_annual_rent'])}, "
        f"from {baseline['n_sales']} sales and {baseline['n_rents']} rent contracts"
    )
    if baseline.get("low_sample"):
        txt += " (SMALL SAMPLE - treat with caution)"
    return txt


def build_context(result: dict) -> dict:
    """Flatten a scorer result into the prompt's labelled variables."""
    listing = result.get("listing") or {}
    scoring = result["scoring"]
    verdict = result["verdict"]
    model = result.get("model") or {}
    baseline = result.get("baseline")
    confidence = verdict.get("confidence", "low")

    caveats = _CONFIDENCE_CAVEAT.get(confidence, _CONFIDENCE_CAVEAT["low"])
    if verdict.get("label") == "Insufficient data":
        caveats = (
            "NO COMPARABLE SALES HISTORY. You must not imply any comparison was "
            "made or hint at a recommendation. Explain that the asking price "
            "cannot be judged for this area/type, and that a listing in a "
            "community with recorded DLD sales would be scoreable. " + caveats
        )

    return {
        "listing_summary": _listing_summary(listing),
        "purpose": listing.get("purpose") or "not stated by the portal (inferred)",
        "verdict_label": verdict.get("label", "unknown"),
        "confidence": confidence,
        "actual_yield": f"{scoring['actual_gross_yield'] * 100:.2f}",
        "expected_yield": f"{scoring['expected_gross_yield'] * 100:.2f}",
        "expected_basis": _BASIS_TEXT.get(
            scoring.get("expected_basis"), scoring.get("expected_basis", "unknown")
        ),
        "diff_pp": f"{scoring['yield_vs_expected_pp']:+.2f}",
        "annual_rent": _fmt_money(scoring.get("annual_rent")),
        "rent_basis": _BASIS_TEXT.get(
            scoring.get("rent_basis"), scoring.get("rent_basis", "unknown")
        ),
        "sale_price": _fmt_money(scoring.get("sale_price")),
        "price_basis": _BASIS_TEXT.get(
            scoring.get("price_basis"), scoring.get("price_basis", "unknown")
        ),
        "baseline_summary": _baseline_text(baseline),
        "drivers": _drivers_text(model.get("shap")),
        "character": _character_text(result.get("character")),
        "caveats": caveats,
    }


MAX_POINTS = 6
MAX_LABEL_CHARS = 60


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON object out of a model response.

    Same three candidates as the classifier chain: the whole response, a fenced
    block, then the first brace-delimited span.
    """
    import json
    import re

    if not text:
        return None
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for c in candidates:
        try:
            parsed = json.loads(c.strip())
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            continue
    return None


def _clean(value) -> str:
    """One field of model text, cleaned the way all user-facing output is."""
    from llm.provider import clean_output

    if value is None:
        return ""
    return clean_output(str(value))


def parse_narrative(raw: str) -> dict | None:
    """Parse a raw narrative response into ``{intro, points, action}``.

    Returns None when there is nothing usable. Never raises: every field is
    optional on the way in, and a response that is plain prose rather than JSON
    is kept as the intro rather than discarded, because readable prose in the
    wrong shape still beats an empty explanation card.
    """
    text = _clean(raw)
    if not text:
        return None

    data = _extract_json(text)
    if data is None:
        return {"intro": text, "points": [], "action": ""}

    intro = _clean(data.get("intro"))
    action = _clean(data.get("action"))

    points: list[dict] = []
    for item in (data.get("points") or [])[:MAX_POINTS]:
        if isinstance(item, dict):
            label = _clean(item.get("label"))[:MAX_LABEL_CHARS]
            detail = _clean(item.get("detail"))
        else:
            label, detail = "", _clean(item)
        if detail:
            points.append({"label": label, "detail": detail})

    if not (intro or points or action):
        return None
    return {"intro": intro, "points": points, "action": action}


def flatten_narrative(detail: dict) -> str:
    """The structured narrative as one plain-text string.

    ``ScoreResponse.narrative`` stays a string: it is what the UI falls back to,
    what /explain is handed as ``existing_narrative``, and what every existing
    consumer expects. The structure is carried alongside it, not instead of it.
    """
    parts: list[str] = []
    if detail.get("intro"):
        parts.append(detail["intro"])
    for point in detail.get("points") or []:
        label = point.get("label")
        parts.append(f"{label}: {point['detail']}" if label else point["detail"])
    if detail.get("action"):
        parts.append(detail["action"])
    return "\n\n".join(parts)


def narrate_score(result: dict) -> dict | None:
    """Structured investor-readable narrative, or None if unavailable.

    Returns ``{"intro": str, "points": [{"label", "detail"}], "action": str}``.
    Never raises: the numeric verdict stands on its own, so a missing key or a
    failed LLM call degrades this optional layer instead of the whole response.
    """
    settings = get_settings()
    if not settings.llm_configured:
        log.info("score_narrative_skipped", reason="llm_not_configured")
        return None
    try:
        raw = _get_chain().invoke(build_context(result))
        detail = parse_narrative(raw)
        if detail is None:
            log.warning("score_narrative_unparsed", sample=str(raw)[:200])
            return None
        log.info(
            "score_narrative_built",
            points=len(detail["points"]),
            has_action=bool(detail["action"]),
        )
        return detail
    except Exception as exc:  # noqa: BLE001 - presentation layer must not break scoring
        log.warning("score_narrative_failed", error=str(exc))
        return None
