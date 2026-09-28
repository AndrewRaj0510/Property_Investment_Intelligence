"""POST /api/v1/predict (M4-03)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request

from api.rate_limit import limiter
from api.schemas.predict import (
    PredictRequest,
    PredictResponse,
    ShapDriver,
    ShapExplanation,
    YieldRange,
)
from llm.chains.explain_shap import explain
from ml.persist import active_run_row
from ml.predict import (
    community_median_yield,
    predict,
    segment_context,
    top_drivers,
)
from ml.shap_explainer import decompose_two_level

log = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
@limiter.limit("30/minute")
async def predict_yield(request: Request, req: PredictRequest) -> PredictResponse:
    phase = 2 if req.use_extended_model else 1
    run = active_run_row(phase=phase)
    if run is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No active model for phase {phase}. Train and activate one with: "
                f"python -m scripts.train_model --phase {phase}"
            ),
        )

    try:
        pred, shap_vals = predict(req.model_dump(), phase=phase)
    except Exception as exc:  # noqa: BLE001
        log.error("predict_failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=500, detail="Prediction failed."
        ) from exc

    try:
        seg = segment_context(
            req.community, req.property_type, req.bedrooms, phase=phase
        )
    except Exception as exc:  # noqa: BLE001 - evidence is advisory, never fatal
        log.warning("segment_context_failed", error=str(exc))
        seg = {
            "segment_matched": False,
            "segment_fallback": False,
            "n_sales": 0,
            "n_rents": 0,
        }

    median = community_median_yield(req.community, phase=phase)
    drivers = top_drivers(shap_vals, k=5)
    try:
        narrative = explain(req.community, pred, median, drivers)
    except Exception as exc:  # noqa: BLE001
        log.error("narrative_failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="SHAP narrative generation requires an LLM; set OLLAMA_BASE_URL "
            "and OLLAMA_MODEL.",
        ) from exc

    predicted_st = None

    explanation = ShapExplanation(
        top_drivers=[ShapDriver(**d) for d in drivers],
        narrative=narrative,
    )

    split = decompose_two_level(shap_vals)
    explanation.property_level = [
        ShapDriver(**d) for d in top_drivers(split["property"], k=5)
    ]
    explanation.macro_level = [
        ShapDriver(**d) for d in top_drivers(split["macro"], k=5)
    ]

    prop_abs = sum(abs(v) for v in split["property"].values())
    macro_abs = sum(abs(v) for v in split["macro"].values())
    total_abs = prop_abs + macro_abs
    explanation.property_share_pct = (
        round(100.0 * prop_abs / total_abs, 1) if total_abs else None
    )

    if not split["macro"]:
        explanation.macro_note = (
            "No macroeconomic or environmental features are in the served model, "
            "so 100% of this prediction is attributed to property-level drivers. "
            "This is a measured result, not a gap: EIBOR, Brent, UAE CPI/GDP, "
            "Dubai weather and the seasonal calendar were all ingested and "
            "evaluated (findings F28-F31). At the model's annual join grain each "
            "collapses to a function of `year`, which is already a feature, and "
            "adding all of them changed cross-validated R-squared by -0.0004."
        )

    rmse = float(run.get("rmse") or 0.5)
    band = round(rmse / 2, 2)

    if seg["segment_matched"]:
        data_warning = None
    elif seg["segment_fallback"]:
        data_warning = (
            f"No {req.bedrooms}-bedroom {req.property_type.lower()} records exist "
            f"for {req.community} in the training data. This prediction is based "
            "on other bedroom counts in the same community and type - treat it as "
            "indicative rather than a like-for-like figure."
        )
    else:
        data_warning = (
            f"No DLD sales or Ejari rent records match {req.community} / "
            f"{req.property_type.lower()} in the training data. The model still "
            "returns a value, but it is an EXTRAPOLATION from other segments, not "
            "a measurement of this one. Do not rely on it."
        )

    return PredictResponse(
        segment_matched=seg["segment_matched"],
        segment_fallback=seg["segment_fallback"],
        n_sales=seg["n_sales"],
        n_rents=seg["n_rents"],
        data_warning=data_warning,
        predicted_yield_lt=pred,
        predicted_yield_st=predicted_st,
        yield_range=YieldRange(low=round(pred - band, 2), high=round(pred + band, 2)),
        community_median_yield=median,
        shap_explanation=explanation,
        model_phase=phase,
        model_run_id=run["id"],
    )
