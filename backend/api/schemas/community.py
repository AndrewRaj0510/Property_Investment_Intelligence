"""Community + comparison schemas (M4-02)."""

from __future__ import annotations

from pydantic import BaseModel


class CommunitySummary(BaseModel):
    community: str
    median_yield_lt: float | None = None
    median_yield_st: float | None = None
    median_sale_price_aed: float | None = None
    median_rent_aed: float | None = None
    data_points: int
    top_shap_driver: str | None = None
