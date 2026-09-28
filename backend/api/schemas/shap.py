"""SHAP summary schemas (M4-02)."""

from __future__ import annotations

from pydantic import BaseModel


class ShapFeature(BaseModel):
    feature_name: str
    mean_abs_shap: float
    rank: int
    feature_group: str | None = None


class ShapSummaryResponse(BaseModel):
    model_run_id: str
    model_phase: int
    features: list[ShapFeature]
    property_features: list[ShapFeature] | None = None
    macro_features: list[ShapFeature] | None = None
