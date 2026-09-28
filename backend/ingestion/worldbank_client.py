"""World Bank UAE GDP growth + CPI client (M1E-02) - Objective 3.1.

Fetches UAE annual indicators from the World Bank REST API. Keyless.

**Two source deviations from the proposal**, both deliberate and both requiring
the objective wording to change rather than the code:

1. The proposal names the **UAE Federal Statistics Centre** for CPI and GDP. The
   FSC publishes irregularly and without a documented API; the World Bank REST
   API is stable, versioned, and keyless.
2. The proposal asks for **Dubai** GDP growth. The World Bank publishes at
   **national** level only, so ``uae_gdp_growth`` is a UAE-wide series standing
   in for an emirate-level one. Dubai Statistics Center does publish emirate GDP
   (dsc.gov.ae is reachable, unlike centralbank.ae), but only as manual
   PDF/Excel exports. That precision was judged not worth the manual-export cost
   because at the Phase-1 join grain every macro feature already collapses to one
   value per year - see ``scripts/ingest_macro.py`` for the full argument.

**Grain honesty.** These series are **annual**. An earlier version of this module
forward-filled each annual value across twelve months so it could be written to a
monthly table; that manufactured eleven-twelfths of its own rows and made an
annual series look monthly. It now returns annual rows at their true grain, and
the caller writes them to the annual table.

**CPI: both indicators are fetched, deliberately.** ``FP.CPI.TOTL.ZG`` is
inflation as an annual *percentage change*; ``FP.CPI.TOTL`` is the price *index*
itself. The previous code fetched the rate but stored it in a column named
``uae_cpi``, which reads as an index - the name misdescribed the contents. Both
are now carried under names that say which is which. They behave differently and
the distinction matters: the index trends monotonically and is therefore close to
collinear with ``year`` (a feature the model already has), while the rate is
stationary and is the economically meaningful driver.
"""

from __future__ import annotations

import httpx
import structlog

from config import get_settings

log = structlog.get_logger(__name__)

GDP_INDICATOR = "NY.GDP.MKTP.KD.ZG"
CPI_RATE_INDICATOR = "FP.CPI.TOTL.ZG"
CPI_INDEX_INDICATOR = "FP.CPI.TOTL"

COUNTRY = "AE"


class WorldBankSourceError(RuntimeError):
    """The World Bank API was unreachable or returned an unusable shape."""


def fetch_gdp_cpi(start_year: int | None = None) -> list[dict]:
    """Annual UAE macro indicators.

    Returns ``[{'year': int, 'uae_gdp_growth': float|None,
    'uae_inflation_pct': float|None, 'uae_cpi_index': float|None}, ...]``
    sorted by year.

    A missing indicator-year is returned as ``None``, never imputed or carried
    forward. The World Bank publishes with a lag, so the most recent one or two
    years are genuinely absent - leaving them NULL lets the model's native
    missing-value handling deal with it and keeps the gap visible in the data.
    """
    gdp = _fetch_indicator(GDP_INDICATOR)
    cpi_rate = _fetch_indicator(CPI_RATE_INDICATOR)
    cpi_index = _fetch_indicator(CPI_INDEX_INDICATOR)

    if not (gdp or cpi_rate or cpi_index):
        raise WorldBankSourceError(
            "World Bank returned no data for any of the three indicators; "
            "refusing to write an empty macro series."
        )

    years = sorted(set(gdp) | set(cpi_rate) | set(cpi_index))
    if start_year is not None:
        years = [y for y in years if y >= start_year]

    out = [
        {
            "year": year,
            "uae_gdp_growth": gdp.get(year),
            "uae_inflation_pct": cpi_rate.get(year),
            "uae_cpi_index": cpi_index.get(year),
        }
        for year in years
    ]
    log.info(
        "worldbank_fetched",
        years=len(out),
        first=out[0]["year"] if out else None,
        last=out[-1]["year"] if out else None,
        gdp_non_null=sum(1 for r in out if r["uae_gdp_growth"] is not None),
        inflation_non_null=sum(1 for r in out if r["uae_inflation_pct"] is not None),
        cpi_index_non_null=sum(1 for r in out if r["uae_cpi_index"] is not None),
    )
    return out


def _fetch_indicator(indicator: str) -> dict[int, float]:
    """One indicator as ``{year: value}``, omitting years the API reports null.

    A network or shape failure raises. The previous version swallowed every
    exception and returned ``{}``, which is indistinguishable from "the World
    Bank has no data for the UAE" - so a transient outage would have written a
    silently empty macro series instead of stopping the run.
    """
    settings = get_settings()
    url = f"{settings.worldbank_base_url.rstrip('/')}/country/{COUNTRY}/indicator/{indicator}"
    try:
        with httpx.Client(timeout=30) as http:
            resp = http.get(url, params={"format": "json", "per_page": 500})
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        raise WorldBankSourceError(
            f"World Bank request failed for {indicator}: {exc}"
        ) from exc

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise WorldBankSourceError(
            f"Unexpected World Bank response for {indicator}: {str(payload)[:200]}"
        )

    out: dict[int, float] = {}
    for record in payload[1]:
        value = record.get("value")
        if value is None:
            continue
        try:
            out[int(record["date"])] = float(value)
        except (TypeError, ValueError, KeyError):
            continue
    return out
