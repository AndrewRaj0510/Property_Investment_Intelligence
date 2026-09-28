"""Inference wrapper (M3D-01).

Loads the active model for a phase (cached in memory; reloaded when the active
run id changes) and returns a predicted yield plus per-feature SHAP values.
Builds the model input row by enriching the request with community-level context
(modal nearest metro, parking rate, sample depth) drawn from the feature matrix
so the user only supplies property facts.
"""

from __future__ import annotations

import pandas as pd
import structlog

from ml.artefacts import load_model
from ml.encoding import align_columns, encode
from ml.persist import active_run_row, feature_matrix_rows
from ml.shap_explainer import explain_single

log = structlog.get_logger(__name__)

_SQFT_PER_SQM = 10.7639

_cache: dict[int, dict] = {}

HUMAN_LABELS = {
    "floor_area_median_sqm": "Floor area",
    "bedrooms": "Bedrooms",
    "price_per_sqm": "Sale price per m²",
    "rent_per_sqm": "Rent per m²",
    "parking_rate": "Parking availability",
    "year": "Market year",
    "n_sales": "Sales-sample depth",
    "n_rents": "Rent-sample depth",
    "nearest_metro_km": "Distance to metro",
    "poi_density_1km": "Amenities within 1 km",
}


def _load_active(phase: int) -> dict:
    run = active_run_row(phase=phase)
    if run is None:
        raise RuntimeError(f"no active model for phase {phase}")
    cached = _cache.get(phase)
    if cached and cached["run_id"] == run["id"]:
        return cached["bundle"]
    bundle = load_model(run["artefact_path"])
    bundle["run_id"] = run["id"]
    _cache[phase] = {"run_id": run["id"], "bundle": bundle}
    return bundle


_TYPE_CANON = {
    "apartment": "apartment",
    "flat": "apartment",
    "unit": "apartment",
    "studio": "studio",
    "villa": "villa",
    "townhouse": "villa",
    "penthouse": "penthouse",
    "hotel apartment": "hotel_apartment",
}


def _canonical_type(raw: str) -> str:
    return _TYPE_CANON.get((raw or "").strip().lower(), (raw or "").strip().lower())


_COMMUNITY_ALIASES = {
    "dubai marina": "Marsa Dubai",
    "marina": "Marsa Dubai",
    "downtown": "Burj Khalifa",
    "downtown dubai": "Burj Khalifa",
    "palm": "Palm Jumeirah",
    "the palm": "Palm Jumeirah",
    "jumeirah": "Jumeirah First",
}


def _resolve_community(community: str) -> str:
    """Resolve a popular community name to DLD's area_name_en when we know it,
    else return the input unchanged (it may already be a real DLD area).

    Delegates to ``ml.community_resolve``, which matches against aliases MINED
    FROM the raw DLD data (``master_project_en`` → ``area_name_en``) rather than
    the 7 hand-written entries below - those covered 7 of 127 DLD areas, so most
    real portal names (e.g. "JVC District 14") resolved to nothing and the
    listing was scored "Insufficient data" (F21). ``_COMMUNITY_ALIASES`` is kept
    as a static fallback for when the alias table has not been built yet.
    """
    from ml.community_resolve import resolve_community

    resolved = resolve_community(community)
    if resolved != community:
        return resolved
    return _COMMUNITY_ALIASES.get((community or "").strip().lower(), community)


def _community_context(
    community: str, canon_type: str, bedrooms, version: int | None
) -> dict:
    """Pull metro / parking / sample depth / per-sqm density for this segment from
    the feature matrix, so the request only needs property facts. Prefers an exact
    (community, type, bedrooms) match in the latest year; falls back to the same
    (community, type) across bedrooms, then to neutral defaults."""
    rows = [
        r
        for r in feature_matrix_rows(version)
        if r.get("community") == community and r.get("property_type") == canon_type
    ]
    exact = [r for r in rows if bedrooms is not None and r.get("bedrooms") == bedrooms]
    pool = exact or rows
    latest = max(pool, key=lambda r: r.get("year", 0)) if pool else {}
    return {
        "year": latest.get("year", 2026),
        "nearest_metro": latest.get("nearest_metro"),
        "parking_rate": latest.get("parking_rate", 0.0),
        "price_per_sqm": latest.get("price_per_sqm", 0.0),
        "rent_per_sqm": latest.get("rent_per_sqm", 0.0),
        "n_sales": latest.get("n_sales", 0),
        "n_rents": latest.get("n_rents", 0),
        "nearest_metro_km": latest.get("nearest_metro_km"),
        "poi_density_1km": latest.get("poi_density_1km"),
        "exact_bedrooms_match": bool(exact),
        "has_rows": bool(rows),
    }


def _build_row(features: dict, version: int | None, phase: int = 1) -> dict:
    canon_type = _canonical_type(features.get("property_type", "Apartment"))
    community = _resolve_community(features.get("community", ""))
    bedrooms = features.get("bedrooms")
    ctx = _community_context(community, canon_type, bedrooms, version)
    sqft = features.get("floor_area_sqft")
    area_sqm = (float(sqft) / _SQFT_PER_SQM) if sqft else None
    row = {
        "community": community,
        "property_type": canon_type,
        "year": ctx["year"],
        "bedrooms": bedrooms,
        "floor_area_median_sqm": area_sqm,
        "price_per_sqm": ctx["price_per_sqm"],
        "rent_per_sqm": ctx["rent_per_sqm"],
        "nearest_metro": ctx["nearest_metro"],
        "parking_rate": ctx["parking_rate"],
        "n_sales": ctx["n_sales"],
        "n_rents": ctx["n_rents"],
        "nearest_metro_km": ctx["nearest_metro_km"],
        "poi_density_1km": ctx["poi_density_1km"],
    }
    if phase == 2:
        from ml.macro_features import macro_row_for_year

        row.update(macro_row_for_year(row["year"]))
    return row


def predict(features: dict, phase: int = 1) -> tuple[float, dict[str, float]]:
    bundle = _load_active(phase)
    version = bundle.get("feature_version")
    row = _build_row(features, version, phase=phase)
    X = align_columns(encode(pd.DataFrame([row])), bundle["feature_names"])
    pred = float(bundle["model"].predict(X)[0])
    shap_vals = explain_single(bundle["model"], X)
    return round(pred, 3), shap_vals


def segment_context(
    community: str,
    property_type: str,
    bedrooms: int | None = None,
    phase: int = 1,
) -> dict:
    """How much real market evidence backs a prediction for this spec.

    The model will return a number for ANY input - an unknown community simply
    encodes to the unseen-category sentinel and the trees extrapolate from the
    remaining features. Measured during review: ``community="Nowhere"`` produced a
    confident-looking 13.3% with no signal that nothing in the training data
    matched. That number is an extrapolation, not a prediction about a real place,
    and a caller cannot tell the difference from the yield alone.

    So this reports the evidence separately from the value:
      * ``segment_matched``  - the exact (community, type, bedrooms) bucket exists.
      * ``segment_fallback`` - the (community, type) exists at other bedroom counts.
      * neither              - no rows at all; the prediction is an extrapolation.

    Extracted from ``predict_yield_for`` so the /predict route and the listing
    scorer judge evidence by the same rule instead of only the scorer having one.
    """
    canon_type = _canonical_type(property_type)
    resolved = _resolve_community(community)
    bundle = _load_active(phase)
    ctx = _community_context(
        resolved, canon_type, bedrooms, bundle.get("feature_version")
    )
    has_data = (ctx.get("n_sales") or 0) > 0 or (ctx.get("n_rents") or 0) > 0
    segment_matched = bool(ctx.get("exact_bedrooms_match")) and bool(has_data)
    segment_fallback = (
        not ctx.get("exact_bedrooms_match")
        and bool(ctx.get("has_rows"))
        and bool(has_data)
    )
    return {
        "resolved_community": resolved,
        "canonical_type": canon_type,
        "segment_matched": segment_matched,
        "segment_fallback": segment_fallback,
        "n_sales": ctx.get("n_sales") or 0,
        "n_rents": ctx.get("n_rents") or 0,
    }


def predict_yield_for(
    community: str,
    property_type: str,
    bedrooms: int | None = None,
    floor_area_sqft: float | None = None,
    phase: int = 1,
) -> dict:
    """Model's expected gross yield for a specific property spec - for live scoring.

    Returns a dict:
      * ``gross_yield``    - predicted gross yield as a FRACTION (e.g. 0.0562), the
                             form the listing scorer works in (``predict`` returns a
                             percentage; this divides by 100).
      * ``gross_yield_pct``the same value as a percentage.
      * ``shap``           - {feature: contribution} for the prediction.
      * ``segment_matched``True only if the requested (community, type, BEDROOMS)
                             bucket exists in the feature matrix. A fallback across
                             other bedroom counts of the same (community, type) is
                             NOT a match (review F4) - it is reported via
                             ``segment_fallback`` instead, at lower confidence.
      * ``segment_fallback``True if no exact bedroom bucket existed but other
                             bedroom counts of the (community, type) did, so the
                             context came from a related-but-not-exact segment.
      * ``resolved_community`` - the DLD area the community resolved to.
      * ``canonical_type``     - the canonical type used.

    Raises the same errors as ``predict`` (no active model → RuntimeError).
    """
    seg = segment_context(community, property_type, bedrooms, phase=phase)
    canon_type = seg["canonical_type"]
    resolved = seg["resolved_community"]
    segment_matched = seg["segment_matched"]
    segment_fallback = seg["segment_fallback"]

    pct, shap = predict(
        {
            "community": community,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "floor_area_sqft": floor_area_sqft,
        },
        phase=phase,
    )
    return {
        "gross_yield": pct / 100.0,
        "gross_yield_pct": pct,
        "shap": shap,
        "segment_matched": segment_matched,
        "segment_fallback": segment_fallback,
        "resolved_community": resolved,
        "canonical_type": canon_type,
    }


def community_median_yield(community: str, phase: int = 1) -> float | None:
    """Mean long-term gross yield across a community's feature-matrix buckets."""
    community = _resolve_community(community)
    run = active_run_row(phase=phase) or active_run_row(phase=1)
    version = run.get("feature_version") if run else None
    rows = [r for r in feature_matrix_rows(version) if r["community"] == community]
    vals = [float(r["gross_yield_lt"]) for r in rows if r.get("gross_yield_lt") is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def top_drivers(shap_vals: dict[str, float], k: int = 5) -> list[dict]:
    """Top-k SHAP drivers by absolute contribution, with human labels."""
    ranked = sorted(shap_vals.items(), key=lambda t: abs(t[1]), reverse=True)[:k]
    out = []
    for feat, val in ranked:
        out.append(
            {
                "feature": feat,
                "contribution": round(val, 3),
                "label": _label_for(feat, val),
            }
        )
    return out


def _label_for(feature: str, value: float) -> str:
    pretty = HUMAN_LABELS.get(feature)
    if pretty is None:
        if feature.startswith("community_"):
            pretty = feature[len("community_"):]
        elif feature.startswith("nearest_metro_"):
            pretty = f"Near {feature[len('nearest_metro_'):]}"
        elif feature.startswith("property_type_"):
            pretty = feature[len("property_type_"):].title()
        else:
            pretty = feature.replace("_", " ").title()
    direction = "adds" if value >= 0 else "removes"
    return f"{pretty} {direction} ~{abs(value):.2f}%"
