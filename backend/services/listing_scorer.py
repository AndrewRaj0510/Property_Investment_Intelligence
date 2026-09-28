"""Score ONE pasted listing URL against the trained yield model + DLD baseline.

Ties the two halves of the system together:

  1. ``ingestion.listing_client`` fetches the pasted listing (Bayut/Dubizzle).
  2. ``ml.predict.predict_yield_for`` gives the **model's expected gross yield**
     for a property of this exact (community, type, bedrooms, size) - using the
     trained XGBoost model (features: bedrooms, size, price/rent-per-sqm,
     community, nearest-metro, parking).
  3. ``db.queries.yield_agg`` (``community_yield_medians``) supplies the segment's
     **median sale price / rent** - the reference figures used to complete the
     side of the ratio the listing doesn't state, and a fallback baseline.

**Scoring logic (model as "expected", verdict = actual vs expected):**
The model predicts the yield a property *like this one* should achieve. We then
compute the listing's **actual** gross yield by completing the missing half of
the ratio, and the verdict compares actual vs the model's expected:

    * **for-sale** (price real, rent not listed):
        expected_rent   = model_yield × segment_median_sale_price
        actual_yield    = expected_rent / LISTING price
        → a listing priced BELOW the segment norm yields more than expected → good.
    * **for-rent** (rent real, price not listed):
        buy_price       = segment_median_sale_price   (real cost to acquire)
        actual_yield    = LISTING rent / buy_price
        → a unit that rents ABOVE comparable stock at that price beats expected.

    verdict: actual_yield vs model_expected_yield (beats → good investment).

Every inferred number is labelled (``rent_basis`` / ``price_basis`` = "listing"
vs "model"/"area_median") so the UI never implies a figure was on the listing.

**LLM layer (optional, never load-bearing):**
After the numbers are settled, two LLM steps enrich the result:

  1. ``llm.chains.classify_listing`` reads the listing description into
     ``character`` (finish tier / view / amenity score). This is CONTEXT ONLY -
     those attributes are not model features (DLD/Ejari has no such columns), so
     they never touch the yield maths.
  2. ``llm.chains.explain_score`` narrates the verdict, preserving the basis
     labels above.

Both degrade to ``None`` on failure: the numeric verdict is the product, and a
missing API key or a flaky LLM call must not cost the user their analysis.

Honesty / confidence:
  * If the model has no real market context for the segment (unknown community or
    a spec with no feature-matrix bucket → ``segment_matched=False``), or the
    listing lacks bedrooms/size, the yield is far less reliable; we fall back to
    the area+type **median** yield as the expected baseline and flag low
    confidence, rather than trust a model extrapolation.
  * If neither the model nor a median baseline is available, we refuse
    (``NoBaselineForListing``) - we never invent a yield.

Live-only.
"""

from __future__ import annotations

import asyncio

import structlog

from config import get_settings
from psycopg.errors import UndefinedColumn

from db.queries.yield_agg import DISTRIBUTION_COLUMNS, MEDIANS_TABLE
from ingestion.listing_client import fetch_listing
from db.pg import connect as _connect
from db.pg import quote_ident as _quote_ident

log = structlog.get_logger(__name__)

_TYPE_ALIASES = {
    "apartment": "apartment",
    "apartments": "apartment",
    "flat": "apartment",
    "flats": "apartment",
    "residential": "apartment",
    "apartmentflat": "apartment",
    "apartment/flat": "apartment",
    "apartment flat": "apartment",
    "duplex": "apartment",
    "studio": "apartment",
    "studios": "apartment",
    "penthouse": "apartment",
    "penthouses": "apartment",
    "hotel apartment": "apartment",
    "hotel apartments": "apartment",
    "villa": "villa",
    "villas": "villa",
    "townhouse": "villa",
    "townhouses": "villa",
    "compound": "villa",
    "complex villas": "villa",
    "arabian house": "villa",
}


class NoBaselineForListing(RuntimeError):
    """Raised when we have no yield median for the listing's area + type."""


def canonical_type(raw_type: str | None) -> str | None:
    """Map a portal's property type to the model's canonical type, or None.

    The model's vocabulary is {apartment, villa}. A "studio" is an apartment with
    ``bedrooms=0`` (the portals report studios as Apartment/rooms=0 anyway, and the
    model learned studios in the 0-bed apartment bucket), so the type does not
    branch on bedroom count.
    """
    if not raw_type:
        return None
    return _TYPE_ALIASES.get(raw_type.strip().lower())


def _resolve_area(area: str) -> str:
    """Resolve a popular community name to DLD's area_name_en, using the SAME
    alias map the model uses - so the median lookup and the model agree on the
    area (e.g. a listing's "Dubai Marina" → "Marsa Dubai" in both)."""
    try:
        from ml.predict import _resolve_community
        return _resolve_community(area)
    except Exception:  # noqa: BLE001 - if model module unavailable, use as-is
        return area


def _pick_area(listing: dict, ptype: str) -> tuple[str | None, dict | None]:
    """Choose the breadcrumb level the DLD data actually knows.

    Portals name a listing at their DEEPEST level. Measured on a real Dubizzle
    listing, that was **"Costa Brava"** - a cluster inside DAMAC Lagoons. DLD
    records the whole master community as ONE area, `Al Hebiah Fifth`, which
    holds 456 villa sales and 103 rent contracts, and the mined alias table
    ALREADY maps "DAMAC Lagoons" onto it. But the cluster name matched nothing,
    so a listing in a fully-covered community was returned as "Insufficient
    data" with an unanchored 12.11% yield (F41).

    So: try the portal's own choice first, then progressively broader levels,
    and take the first with a real baseline. Falls back to the portal's choice
    when nothing resolves, leaving the no-data path exactly as it was.
    """
    primary = listing.get("community")
    seen: set[str] = set()
    for candidate in (primary, *(listing.get("community_candidates") or [])):
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        baseline = _baseline_for(candidate, ptype)
        if baseline is not None:
            if candidate != primary:
                log.info(
                    "community_broadened",
                    listed_as=primary,
                    matched=candidate,
                    resolved_area=baseline.get("area"),
                )
            return candidate, baseline
    return primary, None


def _baseline_for(area: str, ptype: str) -> dict | None:
    """The yield-median bucket for one (area, canonical type), or None.

    Reads the materialized ``community_yield_medians`` table (refreshed offline
    by ``scripts/refresh_yield_medians.py``) rather than re-running the full
    sales↔rents aggregation on every scoring request. Resolves popular area names
    to DLD's ``area_name_en`` first (shared with the model), then matches the area
    case-insensitively - the portals and DLD spell communities with different
    casing (e.g. "AL QUSAIS" vs "Al Qusais").
    """
    base_cols = (
        "area, property_type, median_sale_price, median_annual_rent, "
        "gross_yield, n_sales, n_rents, low_sample"
    )

    def _sql(cols: str) -> str:
        return f"""
        SELECT {cols}
        FROM {_quote_ident(MEDIANS_TABLE)}
        WHERE lower(trim(area)) = %(area)s AND property_type = %(ptype)s
        LIMIT 1
        """

    params = {"area": _resolve_area(area).strip().lower(), "ptype": ptype}
    wide = base_cols + ", " + ", ".join(DISTRIBUTION_COLUMNS)
    with _connect(get_settings()) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(_sql(wide), params)
            except UndefinedColumn:
                conn.rollback()
                log.warning(
                    "baseline_distribution_columns_missing",
                    table=MEDIANS_TABLE,
                    hint="run scripts/refresh_yield_medians.py",
                )
                with conn.cursor() as retry:
                    retry.execute(_sql(base_cols), params)
                    row, cols = retry.fetchone(), base_cols.replace(" ", "").split(",")
            else:
                row = cur.fetchone()
                cols = wide.replace(" ", "").split(",")
    if row is None:
        return None
    rec = dict(zip(cols, row))
    out = {
        "area": rec["area"],
        "property_type": rec["property_type"],
        "median_sale_price": float(rec["median_sale_price"]),
        "median_annual_rent": float(rec["median_annual_rent"]),
        "gross_yield": float(rec["gross_yield"]),
        "n_sales": int(rec["n_sales"]),
        "n_rents": int(rec["n_rents"]),
        "low_sample": bool(rec["low_sample"]),
    }
    for col in DISTRIBUTION_COLUMNS:
        val = rec.get(col)
        out[col] = None if val is None else float(val)
    return out


def _model_expected_yield(area, ptype, bedrooms, area_sqft):
    """Model's expected gross yield (fraction) for this spec, or None if the model
    is unavailable / has no real market context for the segment.

    Returns ``(yield_fraction, shap, segment_matched, segment_fallback)`` or
    ``None``.
    """
    try:
        from ml.predict import predict_yield_for
    except Exception as exc:  # noqa: BLE001 - model deps optional at import time
        log.warning("model_import_failed", error=str(exc))
        return None
    try:
        r = predict_yield_for(
            community=area,
            property_type=ptype,
            bedrooms=bedrooms,
            floor_area_sqft=area_sqft,
        )
    except Exception as exc:  # noqa: BLE001 - no active model, etc.
        log.warning("model_predict_failed", error=str(exc))
        return None
    return (
        r["gross_yield"],
        r["shap"],
        r["segment_matched"],
        r.get("segment_fallback", False),
    )


def _classify_character(listing: dict) -> dict | None:
    """Finish tier / view / amenity score for the pasted listing, or None.

    CONTEXT ONLY. These attributes do not exist in the DLD/Ejari training data,
    so they are deliberately kept out of the model's feature vector - surfacing
    them here explains how a unit is *presented*, never how its yield was
    computed. Failure is non-fatal: the score does not depend on this.
    """
    settings = get_settings()
    if not settings.llm_configured:
        return None
    description = (listing.get("description") or "").strip()
    if not description:
        return None
    try:
        from llm.chains.classify_listing import classify

        feats = classify(description)
    except Exception as exc:  # noqa: BLE001 - enrichment must not break scoring
        log.warning("listing_classify_failed", error=str(exc))
        return None
    out = {
        "finish_tier": feats.finish_tier,
        "view_type": feats.view_type,
        "amenity_score": feats.amenity_score,
    }
    if not any(v is not None for v in out.values()):
        return None
    return out


async def _enrich(scored: dict) -> None:
    """Attach LLM character + narrative to a scored result, in place.

    Both chains are synchronous and network-bound, so they run in worker threads
    rather than blocking the event loop. Classification runs first because the
    narrative quotes its output - they cannot be parallelised.

    Bounded twice, on purpose, because the two bounds do different jobs:

      * the **outer** ``wait_for`` budget bounds the REQUEST - measured: a hung
        chain returns control at the budget instead of the caller waiting.
      * the **inner** per-chain HTTP timeout bounds the WORKER THREAD. Python
        cannot cancel a thread, so ``wait_for`` alone would return promptly
        while the orphaned thread still held a threadpool slot until the HTTP
        call finished. Both are needed.

    The two chains have SEPARATE inner budgets (``llm_enrich_timeout_s`` for
    classification, ``llm_narrate_timeout_s`` for the narrative) because they
    are different sizes of job, so the outer budget is their sum rather than
    twice one of them. It used to be ``enrich * 2``, which silently starved the
    narrative (F44).

    On timeout the numeric verdict is returned as-is - the prose is optional.
    """
    scored.setdefault("character", None)
    scored.setdefault("narrative", None)
    scored.setdefault("narrative_detail", None)
    settings = get_settings()
    budget = settings.llm_enrich_timeout_s + settings.llm_narrate_timeout_s + 5

    async def _run() -> None:
        scored["character"] = await asyncio.to_thread(
            _classify_character, scored["listing"]
        )
        from llm.chains.explain_score import flatten_narrative, narrate_score

        detail = await asyncio.to_thread(narrate_score, scored)
        if detail:
            scored["narrative_detail"] = detail
            scored["narrative"] = flatten_narrative(detail)

    try:
        await asyncio.wait_for(_run(), timeout=budget)
    except Exception as exc:  # noqa: BLE001 - covers TimeoutError; NOT CancelledError
        log.warning("enrich_timed_out_or_failed", error=str(exc), budget_s=budget)


async def score_listing_url(url: str) -> dict:
    """Fetch a pasted listing and score it with the model + DLD baseline.

    Returns the listing, the baseline, the computed actual/expected yields, and a
    verdict. Raises ``NoBaselineForListing`` when neither the model nor a median
    baseline can inform the score; propagates listing_client errors when the URL
    can't be read.
    """
    listing = await fetch_listing(url)

    area = listing.get("community")
    bedrooms = listing.get("bedrooms")
    area_sqft = listing.get("area_sqft")
    ptype = canonical_type(listing.get("property_type"))
    if not area or not ptype:
        missing = []
        if not area:
            missing.append("which community it is in")
        if not ptype:
            missing.append("whether it is an apartment or a villa")
        raise NoBaselineForListing(
            "The portal did not tell us " + " or ".join(missing) + ", so this "
            "listing cannot be compared against Dubai Land Department records. "
            "This usually means the listing page is unavailable or has been "
            "removed."
        )

    area, baseline = _pick_area(listing, ptype)
    model = _model_expected_yield(area, ptype, bedrooms, area_sqft)

    if baseline is None and model is None:
        raise NoBaselineForListing(
            f"We have no sales or rental history for {ptype}s in {area}, so "
            "there is nothing to compare this listing against. The Dubai Land "
            "Department records behind this tool do not cover that "
            "community and property type."
        )

    scored = _score(listing, baseline, model, ptype)
    scored["listing"] = listing
    from ingestion import listing_fixtures as _fx

    scored["source_mode"] = (
        "fixture"
        if (_fx.is_fixture_url(url) or get_settings().replay_listings)
        else "live"
    )
    await _enrich(scored)
    log.info(
        "listing_scored",
        source=listing.get("source"),
        area=area,
        ptype=ptype,
        actual_yield=round(scored["scoring"]["actual_gross_yield"], 4),
        expected_yield=round(scored["scoring"]["expected_gross_yield"], 4),
        recommend=scored["verdict"]["recommend"],
        confidence=scored["verdict"]["confidence"],
        narrated=scored.get("narrative") is not None,
    )
    return scored


def _score(listing: dict, baseline: dict | None, model, ptype: str) -> dict:
    """Core scoring: complete the ratio, pick the expected-yield baseline, verdict.

    ``model`` is ``(yield_fraction, shap, segment_matched, segment_fallback)`` or
    None. ``baseline`` is the median bucket dict or None. At least one is non-None
    (checked upstream).
    """
    purpose = listing.get("purpose")
    price = listing.get("price")
    rent_listed = listing.get("annual_rent")
    bedrooms = listing.get("bedrooms")
    area_sqft = listing.get("area_sqft")

    price = price if price else None
    rent_listed = rent_listed if rent_listed else None

    purpose_ambiguous = purpose is None

    model_yield = model[0] if model else None
    shap = model[1] if model else None
    segment_matched = model[2] if model else False
    segment_fallback = model[3] if model and len(model) > 3 else False

    if model_yield is not None and segment_matched:
        expected_yield = model_yield
        expected_basis = "model"
    elif baseline is not None:
        expected_yield = baseline["gross_yield"]
        expected_basis = "area_median"
    elif model_yield is not None:
        expected_yield = model_yield
        expected_basis = "model_low_confidence"
    else:  # pragma: no cover - guarded upstream
        raise NoBaselineForListing("No expected-yield baseline available.")

    median_sale = baseline["median_sale_price"] if baseline else None

    is_rent = purpose == "for-rent" or (rent_listed is not None and price is None)

    if is_rent:
        rent = rent_listed or price
        if not rent:
            raise NoBaselineForListing("Listing has no usable rent figure.")
        buy_price = median_sale
        if not buy_price:
            buy_price = rent / expected_yield if expected_yield else None
        if not buy_price:
            raise NoBaselineForListing("Cannot establish a purchase price to score against.")
        actual_yield = rent / buy_price
        sale_price, rent_amount = buy_price, rent
        rent_basis, price_basis = "listing", ("area_median" if median_sale else "model_implied")
    else:
        if not price:
            raise NoBaselineForListing("Listing has no usable price figure.")
        ref_price = median_sale or price
        expected_rent = expected_yield * ref_price
        actual_yield = expected_rent / price
        sale_price, rent_amount = price, expected_rent
        rent_basis, price_basis = ("model" if median_sale is None else "model_x_median"), "listing"

    diff_pp = (actual_yield - expected_yield) * 100
    if abs(diff_pp) < 1e-6:
        diff_pp = 0.0
    beats = actual_yield > expected_yield

    no_reference = median_sale is None

    if no_reference:
        confidence = "very_low"
    elif purpose_ambiguous:
        confidence = "low"
    elif (
        expected_basis == "model"
        and segment_matched
        and not segment_fallback
        and bedrooms is not None
        and area_sqft
    ):
        confidence = "high" if not (baseline and baseline["low_sample"]) else "normal"
    elif expected_basis in ("model", "area_median"):
        confidence = "low"
    else:
        confidence = "very_low"

    if no_reference:
        reason = (
            f"Estimated gross yield ~{actual_yield * 100:.2f}% for this "
            f"{ptype}, but there is no comparable sales history for this area, "
            "so we can't judge whether the asking price is good value."
        )
    elif expected_basis.startswith("model"):
        reason = (
            f"Actual gross yield {actual_yield * 100:.2f}% vs the model's expected "
            f"{expected_yield * 100:.2f}% for a comparable {ptype} ({diff_pp:+.2f} pp)."
        )
    else:
        reason = (
            f"Actual gross yield {actual_yield * 100:.2f}% vs the {ptype} area "
            f"median of {expected_yield * 100:.2f}% ({diff_pp:+.2f} pp)."
        )

    verdict = {
        "recommend": bool(beats) and not no_reference,
        "label": (
            "Insufficient data" if no_reference
            else ("Good investment" if beats else "Below market")
        ),
        "reason": reason,
        "confidence": confidence,
    }

    out = {
        "baseline": None,
        "model": (
            {
                "expected_gross_yield": model_yield,
                "segment_matched": segment_matched,
                "segment_fallback": segment_fallback,
                "shap": shap,
            }
            if model
            else None
        ),
        "scoring": {
            "actual_gross_yield": actual_yield,
            "expected_gross_yield": expected_yield,
            "expected_basis": expected_basis,
            "annual_rent": rent_amount,
            "sale_price": sale_price,
            "rent_basis": rent_basis,
            "price_basis": price_basis,
            "yield_vs_expected_pp": diff_pp,
        },
        "verdict": verdict,
    }
    if baseline is not None:
        out["baseline"] = {
            "area": baseline["area"],
            "property_type": baseline["property_type"],
            "median_gross_yield": baseline["gross_yield"],
            "median_sale_price": baseline["median_sale_price"],
            "median_annual_rent": baseline["median_annual_rent"],
            "n_sales": baseline["n_sales"],
            "n_rents": baseline["n_rents"],
            "low_sample": baseline["low_sample"],
        }
        for col in DISTRIBUTION_COLUMNS:
            out["baseline"][col] = baseline.get(col)
    return out
