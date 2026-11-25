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
from ml.community_resolve import resolve_area
from ml.encoding import align_columns, encode
from ml.persist import active_run_row, feature_matrix_rows
from ml.property_types import canonical_type_or_raw
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


def _segment(
    community: str, property_type: str, bedrooms, version: int | None
) -> tuple[str, str, dict]:
    """Resolve the area and type once and read the segment's context once."""
    canon_type = canonical_type_or_raw(property_type)
    resolved = resolve_area(community)
    return resolved, canon_type, _community_context(resolved, canon_type, bedrooms, version)


def _build_row(
    community: str,
    canon_type: str,
    bedrooms,
    floor_area_sqft,
    ctx: dict,
    phase: int = 1,
) -> dict:
    area_sqm = (float(floor_area_sqft) / _SQFT_PER_SQM) if floor_area_sqft else None
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


def _predict_row(bundle: dict, row: dict) -> tuple[float, dict[str, float]]:
    X = align_columns(encode(pd.DataFrame([row])), bundle["feature_names"])
    pred = float(bundle["model"].predict(X)[0])
    return round(pred, 3), explain_single(bundle["model"], X)


def _evidence(ctx: dict) -> dict:
    """How much real market evidence backs a prediction for this segment."""
    has_data = (ctx.get("n_sales") or 0) > 0 or (ctx.get("n_rents") or 0) > 0
    exact = bool(ctx.get("exact_bedrooms_match"))
    return {
        "segment_matched": exact and has_data,
        "segment_fallback": not exact and bool(ctx.get("has_rows")) and has_data,
        "n_sales": ctx.get("n_sales") or 0,
        "n_rents": ctx.get("n_rents") or 0,
    }


def predict(features: dict, phase: int = 1) -> tuple[float, dict[str, float]]:
    bundle = _load_active(phase)
    bedrooms = features.get("bedrooms")
    resolved, canon_type, ctx = _segment(
        features.get("community", ""),
        features.get("property_type", "Apartment"),
        bedrooms,
        bundle.get("feature_version"),
    )
    row = _build_row(
        resolved, canon_type, bedrooms, features.get("floor_area_sqft"), ctx, phase
    )
    return _predict_row(bundle, row)


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

    The /predict route and the listing scorer judge evidence by this same rule
    (``_evidence``) instead of only the scorer having one.
    """
    bundle = _load_active(phase)
    resolved, canon_type, ctx = _segment(
        community, property_type, bedrooms, bundle.get("feature_version")
    )
    return {"resolved_community": resolved, "canonical_type": canon_type, **_evidence(ctx)}


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

    Loads the model, resolves the area and reads the segment once each.
    Raises the same errors as ``predict`` (no active model → RuntimeError).
    """
    bundle = _load_active(phase)
    resolved, canon_type, ctx = _segment(
        community, property_type, bedrooms, bundle.get("feature_version")
    )
    row = _build_row(resolved, canon_type, bedrooms, floor_area_sqft, ctx, phase)
    pct, shap = _predict_row(bundle, row)
    ev = _evidence(ctx)
    return {
        "gross_yield": pct / 100.0,
        "gross_yield_pct": pct,
        "shap": shap,
        "segment_matched": ev["segment_matched"],
        "segment_fallback": ev["segment_fallback"],
        "resolved_community": resolved,
        "canonical_type": canon_type,
    }


def community_median_yield(community: str, phase: int = 1) -> float | None:
    """Mean long-term gross yield across a community's feature-matrix buckets."""
    community = resolve_area(community)
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
