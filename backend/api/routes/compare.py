"""GET /api/v1/compare (M4-05)."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from api.rate_limit import limiter
from api.schemas.community import CommunitySummary
from ml.community_stats import community_summaries

router = APIRouter()


@router.get("/compare", response_model=list[CommunitySummary], tags=["communities"])
@limiter.limit("60/minute")
async def compare(
    request: Request,
    communities: str = Query(
        ..., description="Comma-separated community names, e.g. Dubai Marina,Business Bay"
    ),
) -> list[CommunitySummary]:
    wanted = {c.strip() for c in communities.split(",") if c.strip()}
    summaries = [s for s in community_summaries() if s["community"] in wanted]
    return [CommunitySummary(**s) for s in summaries]
