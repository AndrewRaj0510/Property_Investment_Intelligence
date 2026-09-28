"""EIA Brent crude client (M1E-03) - Objective 3.1.

Fetches the monthly Europe Brent Spot Price FOB (series ``RBRTE``) from the EIA
petroleum API. Live-only: no synthetic fallback, and a missing key raises rather
than returning an empty series that would silently ingest as "no data".

**Source deviation from the proposal.** The proposal names **FRED** for Brent;
this uses the **EIA**, which is the upstream publisher of the series FRED
redistributes (FRED's ``DCOILBRENTEU`` is sourced from EIA). The EIA API is
documented, keyed but free, and returns monthly means directly, whereas FRED
returns daily observations that would need aggregating here. FRED was also
unreachable from this network on every attempt during development. Either the
objective wording or this client must change; the wording is the honest one to
change, and the reason is recorded in AIM.md.
"""

from __future__ import annotations

import httpx
import structlog

from config import get_settings

log = structlog.get_logger(__name__)

SERIES_ID = "RBRTE"
_ROUTE = "petroleum/pri/spt/data/"

_MIN_PLAUSIBLE = 1.0
_MAX_PLAUSIBLE = 500.0


class BrentSourceError(RuntimeError):
    """The EIA API is unconfigured, unreachable, or returned an unusable shape."""


def fetch_brent(start_year: int | None = None) -> list[dict]:
    """Monthly Brent spot price.

    Returns ``[{'month': 'YYYY-MM-01', 'brent_crude_usd': float}, ...]`` sorted
    by month. ``start_year`` trims the series at the source so the request stays
    small; the API holds data back to 1987.
    """
    settings = get_settings()
    if not settings.eia_api_key:
        raise BrentSourceError(
            "EIA_API_KEY is not set. Register free at "
            "https://www.eia.gov/opendata/register.php and add it to backend/.env. "
            "There is no synthetic fallback."
        )

    params = {
        "api_key": settings.eia_api_key,
        "frequency": "monthly",
        "data[0]": "value",
        "facets[series][]": SERIES_ID,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    if start_year is not None:
        params["start"] = f"{start_year}-01"

    url = f"{settings.eia_base_url.rstrip('/')}/{_ROUTE}"
    try:
        with httpx.Client(timeout=60) as http:
            resp = http.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        raise BrentSourceError(f"EIA request failed: {exc}") from exc

    try:
        records = payload["response"]["data"]
    except (KeyError, TypeError) as exc:
        raise BrentSourceError(
            f"Unexpected EIA response shape: {str(payload)[:200]}"
        ) from exc

    out: list[dict] = []
    skipped = 0
    for record in records:
        period = str(record.get("period", "")).strip()
        raw = record.get("value")
        if len(period) != 7 or raw is None:
            skipped += 1
            continue
        try:
            price = float(raw)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not _MIN_PLAUSIBLE <= price <= _MAX_PLAUSIBLE:
            skipped += 1
            continue
        out.append({"month": f"{period}-01", "brent_crude_usd": round(price, 4)})

    if not out:
        raise BrentSourceError(
            "EIA returned no usable Brent observations - check the series id and key."
        )

    out.sort(key=lambda r: r["month"])
    log.info(
        "brent_fetched",
        rows=len(out),
        first=out[0]["month"],
        last=out[-1]["month"],
        skipped=skipped,
    )
    return out


def fetch_brent_annual(
    start_year: int | None = None, monthly: list[dict] | None = None
) -> list[dict]:
    """Annual mean Brent price - the grain the Phase-1 matrix joins on.

    Carries the month count so a part-year mean (the current year, always) is
    distinguishable from a full twelve-month one.
    """
    from collections import defaultdict

    buckets: dict[int, list[float]] = defaultdict(list)
    for row in (monthly if monthly is not None else fetch_brent(start_year)):
        buckets[int(row["month"][:4])].append(row["brent_crude_usd"])

    return [
        {
            "year": year,
            "brent_crude_usd": round(sum(values) / len(values), 4),
            "brent_n_months": len(values),
            "partial_year": len(values) < 12,
        }
        for year, values in sorted(buckets.items())
    ]
