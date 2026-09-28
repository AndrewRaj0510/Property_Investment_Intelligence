"""Schemas for GET /api/v1/market/price-history.

The transacted-price series behind the results-screen chart. Every field here
describes something that was RECORDED: there is no projected point, no fitted
forward value and no confidence band, because this platform has no price model
to project with (the trained model predicts gross yield at annual grain). The
``basis`` and ``note`` strings travel with the data so the client cannot render
the series without its provenance, the same contract the seasonal calendar uses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_SHORT_TEXT = 500


class PricePoint(BaseModel):
    """One calendar quarter of registered DLD sales."""

    period: str = Field(max_length=16, description='e.g. "2024-Q1"')
    year: int
    quarter: int = Field(ge=1, le=4)
    median_aed_per_sqm: float
    median_aed_per_sqft: float
    ma4_aed_per_sqft: float | None = None
    n_sales: int


class PriceHistory(BaseModel):
    area: str = Field(max_length=MAX_SHORT_TEXT)
    area_requested: str = Field(max_length=MAX_SHORT_TEXT)
    property_type: str = Field(max_length=MAX_SHORT_TEXT)

    points: list[PricePoint] = Field(default_factory=list)

    change_pct: float | None = None
    yoy_change_pct: float | None = None
    direction: str | None = Field(None, max_length=32)

    first_period: str | None = Field(None, max_length=16)
    last_period: str | None = Field(None, max_length=16)
    total_sales: int = 0
    coverage_to: str | None = Field(None, max_length=32)

    basis: str = Field(max_length=1_000)
    note: str = Field(max_length=1_000)

    available: bool = True
    detail: str | None = Field(None, max_length=1_000)
