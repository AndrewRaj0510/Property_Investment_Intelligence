"""Scheduled maintenance tasks - macro and OSM refreshes.

These two were previously LangGraph nodes (``pipeline/nodes/ingest_macro.py``
and ``ingest_osm.py``). The graph around them was removed because it orchestrated
a data flow this project does not use: it wrote to ``dld_transactions`` and
``rental_listings``, neither of which exists in the deployed database, while the
model is actually built from ``dld_transactions_raw`` and
``dld_rent_contracts_raw`` loaded from DDA portal CSVs. See DISSERTATION_FINDINGS
F34.

These two survived because they are the only stages that did real, current work:
both persist over psycopg and both wrap the same scripts a human would run by
hand. They are plain async functions now - the ``state`` dict they used to thread
existed only to satisfy the graph's signature.
"""

from __future__ import annotations

import asyncio

import structlog

log = structlog.get_logger(__name__)

FROZEN_OBJECTIVES = {
    "3.3": "Tourism volumes + COVID/Ramadan/Expo dummies (data/tourism.csv absent)",
}


async def refresh_macro() -> dict:
    """Re-ingest the macro series (Objectives 3.1 + 3.2). Never raises.

    Covers EIBOR (monthly + annual), Brent, World Bank GDP/CPI, and Open-Meteo
    weather. Returns a small result dict; failures are logged and reported in
    ``errors`` so a scheduled run cannot take the process down.
    """
    from db.queries.macro import (
        ensure_tables,
        upsert_macro_annual,
        upsert_macro_monthly,
    )
    from ingestion.eia_client import fetch_brent, fetch_brent_annual
    from ingestion.eibor_client import (
        fetch_eibor,
        fetch_eibor_annual,
        fetch_eibor_daily,
    )
    from ingestion.openmeteo_client import (
        fetch_weather,
        fetch_weather_annual,
        fetch_weather_daily,
    )
    from ingestion.worldbank_client import fetch_gdp_cpi
    from scripts.ingest_macro import _merge_annual, _merge_monthly

    log.info("refresh_macro_start", scope="objectives_3.1_and_3.2")
    try:
        daily = fetch_eibor_daily()
        eibor_monthly = fetch_eibor(daily=daily)
        eibor_annual = fetch_eibor_annual(daily=daily)
        brent_monthly = fetch_brent()
        brent_annual = fetch_brent_annual(monthly=brent_monthly)
        worldbank = fetch_gdp_cpi()
        weather_daily, _ = fetch_weather_daily()
        weather_monthly = fetch_weather(daily=weather_daily)
        weather_annual = fetch_weather_annual(daily=weather_daily)

        ensure_tables()
        n_monthly = upsert_macro_monthly(
            _merge_monthly(eibor_monthly, brent_monthly, weather_monthly)
        )
        n_annual = upsert_macro_annual(
            _merge_annual(eibor_annual, brent_annual, worldbank, weather_annual)
        )
        log.info(
            "refresh_macro_done",
            monthly_rows=n_monthly,
            annual_rows=n_annual,
            frozen=sorted(FROZEN_OBJECTIVES),
        )
        return {
            "monthly_rows": n_monthly,
            "annual_rows": n_annual,
            "frozen": sorted(FROZEN_OBJECTIVES),
            "errors": [],
        }
    except Exception as exc:  # noqa: BLE001 - a scheduled job must not crash the app
        log.error("refresh_macro_failed", error=str(exc))
        return {"monthly_rows": 0, "annual_rows": 0, "errors": [str(exc)]}


async def refresh_osm() -> dict:
    """Re-derive OSM locational features for every DLD community. Never raises.

    Thin wrapper over ``scripts.refresh_osm_features.refresh``, which is also the
    standalone entrypoint. Blocking (hundreds of paced HTTP calls against
    rate-limited public OSM endpoints), so it runs off the event loop.
    """
    from scripts.refresh_osm_features import refresh

    log.info("refresh_osm_start")
    try:
        result = await asyncio.to_thread(refresh)
        log.info(
            "refresh_osm_done",
            enriched=result["enriched"],
            skipped=len(result["failed"]),
            coverage=result["coverage"],
        )
        return {"enriched": result["enriched"], "errors": []}
    except Exception as exc:  # noqa: BLE001
        log.error("refresh_osm_failed", error=str(exc))
        return {"enriched": 0, "errors": [str(exc)]}
