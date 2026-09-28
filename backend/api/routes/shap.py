"""GET /api/v1/shap/summary (M4-06, M6-06)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from api.rate_limit import limiter
from api.schemas.shap import ShapFeature, ShapSummaryResponse
from ml.persist import active_run_row, shap_summary_rows

router = APIRouter()


@router.get("/shap/summary", response_model=ShapSummaryResponse, tags=["explainability"])
@limiter.limit("60/minute")
async def shap_summary(
    request: Request,
    use_extended_model: bool = Query(False),
) -> ShapSummaryResponse:
    phase = 2 if use_extended_model else 1
    run = active_run_row(phase=phase)
    if run is None:
        raise HTTPException(status_code=503, detail=f"No active model for phase {phase}.")

    rows = shap_summary_rows(run["id"])
    features = [ShapFeature(**r) for r in rows]

    resp = ShapSummaryResponse(
        model_run_id=run["id"], model_phase=phase, features=features
    )
    if phase == 2:
        resp.property_features = [f for f in features if f.feature_group != "macro"]
        resp.macro_features = [f for f in features if f.feature_group == "macro"]
    return resp
