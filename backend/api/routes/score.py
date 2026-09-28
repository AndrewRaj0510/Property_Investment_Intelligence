"""POST /api/v1/score - score a single pasted listing URL.

The live-scoring entry point: the user pastes ONE Bayut/Dubizzle listing URL; we
fetch just that listing (Apify), score it with the trained yield model against the
segment's real DLD/Ejari baseline, and return the listing, the model's expected
yield + SHAP drivers, the computed actual yield, and a verdict.

Error mapping (fail loud, never fabricate):
    * unsupported / unreadable URL      → 422 / 502
    * listing source not configured     → 503 (APIFY_TOKEN missing)
    * no model context AND no baseline   → 422 (we refuse to invent a yield)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request

from api.rate_limit import limiter
from api.schemas.score import ScoreRequest, ScoreResponse
from ingestion.listing_client import (
    ListingFetchError,
    ListingSourceNotConfigured,
    UnsupportedListingSite,
)
from services.listing_scorer import NoBaselineForListing, score_listing_url

log = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/score", response_model=ScoreResponse, tags=["scoring"])
@limiter.limit("10/minute")
async def score_listing(request: Request, req: ScoreRequest) -> ScoreResponse:
    url = str(req.url)
    try:
        result = await score_listing_url(url)
    except UnsupportedListingSite as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ListingSourceNotConfigured as exc:
        raise HTTPException(
            status_code=503,
            detail="Listing lookups are not configured (APIFY_TOKEN missing).",
        ) from exc
    except ListingFetchError as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not read the listing: {exc}"
        ) from exc
    except NoBaselineForListing as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error("score_failed", url=url, error=str(exc))
        raise HTTPException(status_code=500, detail="Scoring failed.") from exc

    return ScoreResponse(**result)
