"""Predict request/response schemas (M4-01)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Inputs to the yield model.

    These are exactly the property facts the trained model consumes (see
    ``ml/encoding.py``): area, canonical type, bedrooms, and floor area. The model
    supplies the segment's market context (nearest metro, parking rate, price/rent
    per sqm, sample depth) itself from the feature matrix, so the caller only sends
    what a listing states. Descriptive attributes that exist only in live listings
    (finish, view, amenities) are deliberately NOT inputs - they are not in the
    training data (train/serve feature parity; see DISSERTATION_FINDINGS F3/F8) and
    belong to the UI/narrative context layer instead.
    """

    community: str = Field(..., min_length=1, max_length=200)
    property_type: str = Field("Apartment", min_length=1, max_length=100)
    bedrooms: int = Field(1, ge=0, le=10)
    floor_area_sqft: float = Field(800, gt=0)
    use_extended_model: bool = False


class ShapDriver(BaseModel):
    feature: str = Field(max_length=200)
    contribution: float
    label: str = Field(max_length=500)


class YieldRange(BaseModel):
    low: float
    high: float


class ShapExplanation(BaseModel):
    top_drivers: list[ShapDriver]
    narrative: str
    property_level: list[ShapDriver] | None = None
    macro_level: list[ShapDriver] | None = None
    macro_note: str | None = None
    property_share_pct: float | None = None


class PredictResponse(BaseModel):
    predicted_yield_lt: float
    predicted_yield_st: float | None = None
    yield_range: YieldRange
    community_median_yield: float | None
    shap_explanation: ShapExplanation
    model_phase: int
    model_run_id: str
    segment_matched: bool = False
    segment_fallback: bool = False
    n_sales: int = 0
    n_rents: int = 0
    data_warning: str | None = Field(None, max_length=1000)
