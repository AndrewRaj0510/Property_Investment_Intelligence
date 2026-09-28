"""Open-Meteo historical weather client (M1E-04) - Objective 3.2.

Fetches daily maximum temperature and mean relative humidity for Dubai from the
Open-Meteo **archive** API (ERA5 reanalysis), aggregates to month and year, and
computes the composite thermal discomfort index the objective calls for.

Keyless and unmetered, which is why this was the cheapest Aim 3 objective to
deliver. No synthetic fallback: a failed fetch raises.

**Provenance worth recording (the F22 lesson).** The API snaps the requested
point to the centre of its reanalysis grid cell - asking for (25.20, 55.27)
returns data for **(25.202108, 55.302376)**. The response carries the resolved
coordinates and elevation, and this module keeps them, because a value silently
measured somewhere other than where you asked is exactly the failure the OSM
sweep hit six times. The archive is also bounded at the current date and its most
recent days come from ERA5T (preliminary reanalysis), which can be revised
slightly later - so a re-run may move the last few days.

---

**THE THERMAL DISCOMFORT INDEX - the specified formula was dimensionally wrong.**

The formula carried in the project's task list and pipeline scaffolding was:

    thermal_index = temp_max_c + 0.33 * humidity_pct - 4

That is the Australian Bureau of Meteorology **apparent temperature** formula,
``AT = Ta + 0.33e - 0.70ws - 4.00`` (wind dropped, since the objective specifies
temperature and humidity only) - but ``e`` in that equation is **water vapour
pressure in hPa**, not **relative humidity in %**. Substituting a percentage for a
pressure is a unit error, and it is not a small one:

* it overstates the index by a mean of **9.0 °C** and by up to **21.5 °C**;
* it **halves the seasonal signal** - the January-to-July spread falls from
  **24.6 °C** (correct) to **13.6 °C** (wrong), and a seasonality feature that
  compresses its own seasonality is worthless;
* it inverts individual days: 2016-03-09 (23.5 °C, 87% RH) scored **48.2**, worse
  than a 42.9 °C August day at **42.9**.

**And it would have passed a spot check.** On the hottest day in the record
(2018-08-09, 47.2 °C, 41% RH) the two formulas agree to within 1 °C - 56.7 vs
57.6. Anyone validating the index by looking at peak summer would have seen a
plausible number. Same silent-plausible-wrong-answer trap as F22.

This module computes vapour pressure properly, via the Magnus-Tetens saturation
formula, and exposes both so the error is reproducible rather than asserted.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict

import httpx
import structlog

from config import get_settings

log = structlog.get_logger(__name__)

DUBAI_LAT, DUBAI_LON = 25.20, 55.27

_DAILY_VARS = "temperature_2m_max,relative_humidity_2m_mean"
_TIMEZONE = "Asia/Dubai"

_MIN_TEMP_C, _MAX_TEMP_C = -10.0, 60.0


class WeatherSourceError(RuntimeError):
    """Open-Meteo was unreachable or returned an unusable shape."""


def saturation_vapour_pressure_hpa(temp_c: float) -> float:
    """Saturation vapour pressure in hPa (Magnus-Tetens over water)."""
    return 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))


def vapour_pressure_hpa(temp_c: float, humidity_pct: float) -> float:
    """Actual water vapour pressure in hPa - the ``e`` the BoM formula wants."""
    return (humidity_pct / 100.0) * saturation_vapour_pressure_hpa(temp_c)


def thermal_index(temp_c: float, humidity_pct: float) -> float:
    """Composite thermal discomfort index (BoM apparent temperature, no wind).

    ``AT = Ta + 0.33e - 4.00`` with ``e`` in hPa. See the module docstring for
    why the previously specified ``+ 0.33 * humidity_pct`` form is wrong.
    """
    return temp_c + 0.33 * vapour_pressure_hpa(temp_c, humidity_pct) - 4.00


def legacy_thermal_index(temp_c: float, humidity_pct: float) -> float:
    """The dimensionally incorrect formula, kept ONLY so the error is measurable.

    Never persist this. It exists so the comparison in the write-up can be
    regenerated rather than quoted from a note.
    """
    return temp_c + 0.33 * humidity_pct - 4.00


def fetch_weather_daily(
    start_date: str = "2012-01-01", end_date: str | None = None
) -> tuple[list[dict], dict]:
    """Daily Dubai weather.

    Returns ``([{'date': date, 'temp_max_c': float, 'humidity_pct': float}, ...],
    provenance)``. ``end_date`` defaults to today; the archive refuses anything
    beyond that, and the refusal names its own bound, which is parsed and honoured
    rather than guessed at.
    """
    settings = get_settings()
    end = end_date or dt.date.today().isoformat()
    url = f"{settings.open_meteo_base_url.rstrip('/')}/archive"
    params = {
        "latitude": DUBAI_LAT,
        "longitude": DUBAI_LON,
        "start_date": start_date,
        "end_date": end,
        "daily": _DAILY_VARS,
        "timezone": _TIMEZONE,
    }

    payload = _get(url, params)
    if payload.get("error"):
        bound = _parse_range_bound(str(payload.get("reason", "")))
        if bound is None:
            raise WeatherSourceError(f"Open-Meteo rejected the request: {payload}")
        log.info("openmeteo_clamped_end_date", requested=end, allowed=bound)
        params["end_date"] = bound
        payload = _get(url, params)
        if payload.get("error"):
            raise WeatherSourceError(f"Open-Meteo rejected the request: {payload}")

    try:
        daily = payload["daily"]
        days = daily["time"]
        temps = daily["temperature_2m_max"]
        humidity = daily["relative_humidity_2m_mean"]
    except (KeyError, TypeError) as exc:
        raise WeatherSourceError(
            f"Unexpected Open-Meteo response shape: {str(payload)[:200]}"
        ) from exc

    provenance = {
        "resolved_lat": payload.get("latitude"),
        "resolved_lon": payload.get("longitude"),
        "elevation_m": payload.get("elevation"),
        "timezone": payload.get("timezone"),
        "requested_lat": DUBAI_LAT,
        "requested_lon": DUBAI_LON,
    }

    out: list[dict] = []
    skipped = 0
    for day, temp, rh in zip(days, temps, humidity):
        if temp is None or rh is None:
            skipped += 1
            continue
        if not _MIN_TEMP_C <= float(temp) <= _MAX_TEMP_C:
            skipped += 1
            continue
        out.append(
            {
                "date": dt.date.fromisoformat(day),
                "temp_max_c": float(temp),
                "humidity_pct": float(rh),
            }
        )

    if not out:
        raise WeatherSourceError("Open-Meteo returned no usable observations.")

    out.sort(key=lambda r: r["date"])
    log.info(
        "openmeteo_fetched",
        days=len(out),
        first=str(out[0]["date"]),
        last=str(out[-1]["date"]),
        skipped=skipped,
        **provenance,
    )
    return out, provenance


def _get(url: str, params: dict) -> dict:
    try:
        with httpx.Client(timeout=120) as http:
            resp = http.get(url, params=params)
            if resp.status_code == 400:
                return resp.json()
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        raise WeatherSourceError(f"Open-Meteo request failed: {exc}") from exc


def _parse_range_bound(reason: str) -> str | None:
    """Pull the archive's own upper bound out of its rejection message."""
    marker = " to "
    if "out of allowed range" not in reason or marker not in reason:
        return None
    tail = reason.rsplit(marker, 1)[-1].strip().rstrip(".")
    try:
        dt.date.fromisoformat(tail)
    except ValueError:
        return None
    return tail


def fetch_weather(
    start_date: str = "2012-01-01", end_date: str | None = None,
    daily: list[dict] | None = None,
) -> list[dict]:
    """Monthly means plus the thermal index.

    Returns ``[{'month': 'YYYY-MM-01', 'temp_max_c', 'humidity_pct',
    'thermal_index', 'weather_n_days'}, ...]``.

    The index is computed **per day and then averaged**, not computed from the
    monthly mean temperature and humidity. Vapour pressure is exponential in
    temperature, so the two differ (Jensen's inequality) - averaging the inputs
    first understates the index in months with wide daily swings.
    """
    rows = daily if daily is not None else fetch_weather_daily(start_date, end_date)[0]
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["date"].year, row["date"].month)].append(row)

    out = []
    for (year, month), group in sorted(buckets.items()):
        n = len(group)
        out.append(
            {
                "month": f"{year}-{month:02d}-01",
                "temp_max_c": round(sum(r["temp_max_c"] for r in group) / n, 2),
                "humidity_pct": round(sum(r["humidity_pct"] for r in group) / n, 2),
                "thermal_index": round(
                    sum(thermal_index(r["temp_max_c"], r["humidity_pct"]) for r in group)
                    / n,
                    2,
                ),
                "weather_n_days": n,
            }
        )
    return out


def fetch_weather_annual(
    start_date: str = "2012-01-01", end_date: str | None = None,
    daily: list[dict] | None = None,
) -> list[dict]:
    """Annual means - the grain the Phase-1 matrix joins on."""
    rows = daily if daily is not None else fetch_weather_daily(start_date, end_date)[0]
    buckets: dict[int, list[dict]] = defaultdict(list)
    months: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        buckets[row["date"].year].append(row)
        months[row["date"].year].add(row["date"].month)

    out = []
    for year, group in sorted(buckets.items()):
        n = len(group)
        out.append(
            {
                "year": year,
                "temp_max_c": round(sum(r["temp_max_c"] for r in group) / n, 2),
                "humidity_pct": round(sum(r["humidity_pct"] for r in group) / n, 2),
                "thermal_index": round(
                    sum(thermal_index(r["temp_max_c"], r["humidity_pct"]) for r in group)
                    / n,
                    2,
                ),
                "weather_n_days": n,
                "weather_partial": len(months[year]) < 12,
            }
        )
    return out
