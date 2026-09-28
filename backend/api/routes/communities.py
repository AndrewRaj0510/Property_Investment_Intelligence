"""GET /api/v1/communities (M4-04)."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from api.rate_limit import limiter
from api.schemas.community import CommunitySummary
from ml.community_stats import community_summaries

router = APIRouter()


@router.get("/communities", response_model=list[CommunitySummary], tags=["communities"])
@limiter.limit("60/minute")
async def list_communities(
    request: Request,
    community: str | None = Query(None, description="Filter to one community"),
) -> list[CommunitySummary]:
    summaries = community_summaries()
    if community:
        summaries = [s for s in summaries if s["community"] == community]
    return [CommunitySummary(**s) for s in summaries]
