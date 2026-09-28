"""GET /api/v1/macro/snapshot (M4-07)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.rate_limit import limiter
import structlog

from api.schemas.macro import MacroSnapshot
from db.queries.macro import all_macro_monthly

log = structlog.get_logger(__name__)
router = APIRouter()


def _latest_with(rows: list[dict], key: str) -> dict | None:
    """The most recent row whose ``key`` is populated. Rows arrive month-ascending."""
    for row in reversed(rows):
        if row.get(key) is not None:
            return row
    return None


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


@router.get("/macro/snapshot", response_model=MacroSnapshot, tags=["macro"])
@limiter.limit("60/minute")
async def macro_snapshot(request: Request) -> MacroSnapshot:
    try:
        rows = all_macro_monthly()
    except Exception as exc:  # noqa: BLE001
        log.warning("macro_snapshot_unavailable", error=str(exc))
        raise HTTPException(
            status_code=503, detail="Macro data unavailable."
        ) from exc
    if not rows:
        raise HTTPException(status_code=503, detail="No macro data ingested yet.")

    latest = rows[-1]
    last_12 = rows[-12:]

    eibor_row = _latest_with(rows, "eibor_3m")
    brent_row = _latest_with(rows, "brent_crude_usd")

    return MacroSnapshot(
        month=str(latest["month"]),
        eibor_3m=eibor_row.get("eibor_3m") if eibor_row else None,
        uae_gdp_growth=latest.get("uae_gdp_growth"),
        uae_cpi=latest.get("uae_cpi"),
        brent_crude_usd=brent_row.get("brent_crude_usd") if brent_row else None,
        temp_max_c=latest.get("temp_max_c"),
        humidity_pct=latest.get("humidity_pct"),
        thermal_index=latest.get("thermal_index"),
        tourist_arrivals=latest.get("tourist_arrivals"),
        hotel_occupancy=latest.get("hotel_occupancy"),
        ramadan_flag=bool(latest.get("ramadan_flag")),
        expo_dummy=bool(latest.get("expo_dummy")),
        model_phase=1,
        eibor_12m_avg=_avg(last_12, "eibor_3m"),
        brent_12m_avg=_avg(last_12, "brent_crude_usd"),
        eibor_as_of=str(eibor_row["month"]) if eibor_row else None,
        brent_as_of=str(brent_row["month"]) if brent_row else None,
    )
