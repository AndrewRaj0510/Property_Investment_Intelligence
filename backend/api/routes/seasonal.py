"""GET /api/v1/seasonal/calendar (M4-08) - Objective 3.5.

Aggregates ``macro_timeseries`` by month-of-year to expose Dubai's seasonal
pattern: mean maximum temperature, mean thermal discomfort index, that index
relative to the annual mean, and the mean number of Ramadan days each calendar
month contains.

**This is a CLIMATE calendar. It makes no yield claim, and that is deliberate.**

The original design returned a ``yield_index`` derived from monthly tourist
arrivals. Both halves of that failed, for different and independently sufficient
reasons:

1. The tourism series was **de-scoped on measured evidence** (F31) - the Dubai
   Statistics Centre publishes international visitor counts for 3 of the 15 years
   in the study period, and the best available monthly proxy changed model R² by
   −0.0033.
2. More fundamentally, **this project has no monthly yield.** The training matrix
   is built at (community × property type × bedrooms × **year**), so any
   month-level yield index would have to be manufactured. F29 measured the
   consequence from the other direction: 99.4% of the thermal signal's variance
   does not survive aggregation to the annual join grain.

So the calendar reports the seasonality that **was** measured - a 24.6 °C swing
in thermal discomfort between January and July - and says in the payload itself
that it is climate, not yield. A consumer cannot relabel it without contradicting
the `basis` and `note` fields it is handed.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.rate_limit import limiter
from api.schemas.macro import SeasonalCalendar, SeasonalMonth
from db.queries.macro import all_macro_monthly

router = APIRouter()

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

BASIS = (
    "Thermal discomfort index (BoM apparent temperature, T + 0.33e − 4, e in hPa), "
    "averaged by month-of-year over the ingested Open-Meteo record. "
    "100 = the annual mean."
)

NOTE = (
    "This is a climate calendar, not a yield calendar. The model predicts gross "
    "rental yield at annual grain only, so no monthly yield seasonality exists to "
    "report; tourist-arrival data was de-scoped on measured evidence (finding F31). "
    "Nothing here should be read as a seasonal yield expectation."
)


@router.get("/seasonal/calendar", response_model=SeasonalCalendar, tags=["macro"])
@limiter.limit("60/minute")
async def seasonal_calendar(request: Request) -> SeasonalCalendar:
    try:
        rows = all_macro_monthly()
    except Exception:  # noqa: BLE001
        return SeasonalCalendar(months=[], basis=BASIS, note=NOTE)

    buckets: dict[int, list[dict]] = {m: [] for m in range(1, 13)}
    for row in rows:
        buckets[int(str(row["month"]).split("-")[1])].append(row)

    def mean(group: list[dict], key: str) -> float | None:
        values = [g[key] for g in group if g.get(key) is not None]
        return sum(values) / len(values) if values else None

    observed = [r["thermal_index"] for r in rows if r.get("thermal_index") is not None]
    annual_mean = sum(observed) / len(observed) if observed else None

    months: list[SeasonalMonth] = []
    for month in range(1, 13):
        group = buckets[month]
        thermal = mean(group, "thermal_index")
        temp = mean(group, "temp_max_c")
        ramadan = mean(group, "ramadan_days")
        months.append(
            SeasonalMonth(
                month_of_year=month,
                month_name=MONTH_NAMES[month - 1],
                avg_temp_max_c=round(temp, 1) if temp is not None else None,
                avg_thermal_index=round(thermal, 1) if thermal is not None else None,
                thermal_index_relative=(
                    round(thermal / annual_mean * 100, 1)
                    if thermal is not None and annual_mean
                    else None
                ),
                avg_ramadan_days=round(ramadan, 1) if ramadan is not None else None,
                months_observed=len(group),
            )
        )

    return SeasonalCalendar(months=months, basis=BASIS, note=NOTE)
