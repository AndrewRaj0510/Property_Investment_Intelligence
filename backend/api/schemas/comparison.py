"""Schemas for GET /api/v1/models/comparison - the Aim 3 baseline-vs-extended result."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PhaseMetrics(BaseModel):
    """Cross-validated metrics for one trained model run."""

    run_id: str
    run_name: str = Field(max_length=200)
    model_type: str = Field(max_length=100)
    mae: float | None = None
    rmse: float | None = None
    r_squared: float | None = None
    cv_folds: int | None = None
    n_features: int = 0


class MacroDriver(BaseModel):
    feature: str = Field(max_length=200)
    mean_abs_shap: float


class MetricDelta(BaseModel):
    """Phase 2 minus Phase 1. A negative r_squared means the extended model is worse."""

    mae: float | None = None
    rmse: float | None = None
    r_squared: float | None = None


class ModelComparison(BaseModel):
    """Baseline (Phase 1) vs baseline + external factors (Phase 2)."""

    phase1: PhaseMetrics | None = None
    phase2: PhaseMetrics | None = None
    delta: MetricDelta | None = None
    macro_attribution_pct: float | None = None
    macro_drivers: list[MacroDriver] = []
    macro_feature_count: int = 0
    interpretation: str | None = Field(None, max_length=2000)
    available: bool = True
    detail: str | None = Field(None, max_length=500)
