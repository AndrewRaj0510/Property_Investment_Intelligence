"""GET /api/v1/market/price-history - transacted price history for one segment.

Answers "has this area been getting more or less expensive?" from the DLD sales
table, for the (area, property type) the results screen is already showing.

Deliberately NOT a forecast endpoint. Every point is a median of sales that were
registered in that quarter, and ``direction`` compares the latest four measured
quarters with the four before them. Nothing is extrapolated: the trained model
predicts gross YIELD at annual grain and there is no price model to project
with, so a projected point here would be a number with no method behind it.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

from api.rate_limit import limiter
from api.schemas.market import PriceHistory, PricePoint
from psycopg.errors import UndefinedTable

from db.queries.price_history import (
    MIN_QUARTER_SALES,
    MIN_YEAR,
    PRICE_HISTORY_TABLE,
    coverage_end,
    quarterly_price_history,
)

log = structlog.get_logger(__name__)
router = APIRouter()

MAX_AREA_CHARS = 200

_FLAT_BAND_PCT = 2.0

BASIS = (
    "Median registered sale price per square metre, by calendar quarter, from "
    "Dubai Land Department transaction records for this community and property "
    "type. Both off-plan and existing-property sales are included, the same "
    "population the yield baseline on this page is computed from."
)
NOTE = (
    "This is recorded transaction history, not a forecast. Past direction is "
    "not a prediction of future prices, and no projection is shown: this "
    "platform models gross rental yield, not price."
)


def _resolve_area(area: str) -> str:
    """Resolve a portal community name to DLD's ``area_name_en``.

    Uses the SAME resolver as the model and the yield baseline, so the chart and
    the numbers beside it describe one place. Falls back to the input when the
    model package cannot be imported, exactly as the scorer does.
    """
    try:
        from ml.predict import _resolve_community

        return _resolve_community(area)
    except Exception:  # noqa: BLE001 - resolver unavailable; use the name as given
        return area


def _canonical_type(raw: str) -> str:
    """Map a free-text property type onto the canonical vocabulary.

    The client passes whatever the listing said ("Apartment", "Residential /
    Villa"), so this reuses the scorer's alias map rather than a second one that
    could drift away from it.
    """
    try:
        from services.listing_scorer import canonical_type

        return canonical_type(raw) or (raw or "").strip().lower()
    except Exception:  # noqa: BLE001
        return (raw or "").strip().lower()


def _with_moving_average(rows: list[dict]) -> list[PricePoint]:
    """Attach a trailing 4-quarter mean to each point (null for the first three)."""
    points: list[PricePoint] = []
    for i, row in enumerate(rows):
        ma = None
        if i >= 3:
            window = [r["median_aed_per_sqft"] for r in rows[i - 3 : i + 1]]
            ma = round(sum(window) / 4, 1)
        points.append(PricePoint(**row, ma4_aed_per_sqft=ma))
    return points


def _window_mean(rows: list[dict], start: int, end: int) -> float | None:
    """Mean of ``median_aed_per_sqft`` over rows[start:end], or None if empty."""
    window = [r["median_aed_per_sqft"] for r in rows[start:end]]
    return sum(window) / len(window) if window else None


def _direction(rows: list[dict]) -> tuple[float | None, str | None]:
    """Year-on-year move and its direction label.

    Compares the mean of the last four quarters with the mean of the four before
    them. A single-quarter comparison would report sampling noise as a market
    move; two four-quarter windows is the smallest honest read on 20-odd sales a
    quarter. Returns ``(None, None)`` with fewer than eight quarters.
    """
    if len(rows) < 8:
        return None, None
    recent = _window_mean(rows, len(rows) - 4, len(rows))
    prior = _window_mean(rows, len(rows) - 8, len(rows) - 4)
    if not recent or not prior:
        return None, None
    change = (recent / prior - 1) * 100
    if abs(change) < _FLAT_BAND_PCT:
        label = "flat"
    else:
        label = "rising" if change > 0 else "falling"
    return round(change, 1), label


@router.get(
    "/market/price-history", response_model=PriceHistory, tags=["market"]
)
@limiter.limit("60/minute")
async def price_history(
    request: Request,
    area: str = Query(
        ...,
        min_length=1,
        max_length=MAX_AREA_CHARS,
        description="Community name; resolved to a DLD area_name_en.",
    ),
    property_type: str = Query(
        ...,
        min_length=1,
        max_length=MAX_AREA_CHARS,
        description='Property type; mapped to the canonical vocabulary ("apartment", "villa").',
    ),
    from_year: int = Query(
        2019,
        ge=MIN_YEAR,
        le=2100,
        description="Earliest calendar year to include.",
    ),
) -> PriceHistory:
    resolved = _resolve_area(area)
    ptype = _canonical_type(property_type)

    try:
        rows = quarterly_price_history(resolved, ptype, from_year=from_year)
        covered_to = coverage_end()
    except UndefinedTable as exc:
        log.warning("price_history_snapshot_missing", table=PRICE_HISTORY_TABLE)
        raise HTTPException(
            status_code=503,
            detail=(
                f"The {PRICE_HISTORY_TABLE} snapshot has not been built on this "
                "deployment. Run scripts.refresh_price_history."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - psycopg surfaces OperationalError here
        log.warning("price_history_unavailable", area=resolved, error=str(exc))
        raise HTTPException(
            status_code=503, detail="Price history is unavailable."
        ) from exc

    if not rows:
        return PriceHistory(
            area=resolved,
            area_requested=area,
            property_type=ptype,
            points=[],
            basis=BASIS,
            note=NOTE,
            coverage_to=covered_to,
            available=False,
            detail=(
                f"No quarter since {from_year} has at least {MIN_QUARTER_SALES} "
                f"registered {ptype} sales in {resolved}, so there is no price "
                "series to draw for this segment."
            ),
        )

    change_pct = None
    first, last = rows[0], rows[-1]
    if first["median_aed_per_sqft"]:
        change_pct = round(
            (last["median_aed_per_sqft"] / first["median_aed_per_sqft"] - 1) * 100, 1
        )
    yoy, direction = _direction(rows)

    return PriceHistory(
        area=resolved,
        area_requested=area,
        property_type=ptype,
        points=_with_moving_average(rows),
        change_pct=change_pct,
        yoy_change_pct=yoy,
        direction=direction,
        first_period=first["period"],
        last_period=last["period"],
        total_sales=sum(r["n_sales"] for r in rows),
        coverage_to=covered_to,
        basis=BASIS,
        note=NOTE,
        available=True,
    )
