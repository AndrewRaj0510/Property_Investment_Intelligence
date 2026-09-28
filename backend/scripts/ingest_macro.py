"""Ingest the Objective 3.1 macroeconomic series into Postgres.

Three sources, two grains, one run:

* **EIBOR 3-month** - parsed from the manually exported workbook at
  ``data/eibor_rates.xlsx`` (centralbank.ae is Cloudflare-walled; see
  ``ingestion/eibor_client.py`` for the full list of automated sources ruled out).
* **Brent crude** - EIA petroleum API, monthly spot, series ``RBRTE``.
* **UAE GDP growth, inflation, CPI index** - World Bank REST API, annual.
* **Dubai weather** (Objective 3.2) - Open-Meteo ERA5 archive, keyless: daily max
  temperature and mean relative humidity, plus the composite thermal discomfort
  index. The index uses proper vapour pressure; the formula previously specified
  in the task list was dimensionally wrong (see ``ingestion/openmeteo_client.py``).

Run from the backend/ directory (so .env and imports resolve):

    .venv/Scripts/python.exe -m scripts.ingest_macro
    .venv/Scripts/python.exe -m scripts.ingest_macro --dry-run
    .venv/Scripts/python.exe -m scripts.ingest_macro --start-year 2012

Requires ``SUPABASE_DB_*`` and ``EIA_API_KEY`` in .env, plus the EIBOR workbook.
Each source fails loudly and independently: a source that cannot be fetched
aborts the run rather than writing a partial series that would later read as
"the data says nothing happened".

---

**Read this before using these features in a model.**

This ingestion is deliberately scoped as ingestion. It does **not** retrain
anything, and ``MACRO_FEATURES`` in ``ml/features.py`` stays empty. The reason is
structural, was predicted before any row was fetched, and this script prints the
evidence for it at the end of every run.

The Phase-1 training matrix is built at **(community × property type × bedrooms
× year)**, and ``year`` is *already* one of its features. Joining an annual macro
series onto it makes every macro feature a deterministic function of ``year`` -
at most one distinct value per year, identical across every row sharing that
year. A gradient-boosted tree can already express any year-specific effect
through ``year`` itself, so these features cannot add information the model does
not already have access to.

That is the same failure mode as the Objective 1.4 OSM null result - a feature
constant within a group cannot explain variation *within* that group - with one
difference worth stating in the write-up: 1.4 was **measured** null after the
fact, whereas this one is **predicted** from the join grain before ingestion and
then confirmed. A predicted-then-confirmed null is the stronger result.

Getting a real signal out of these series would require re-graining the training
matrix to month, which changes the Phase-1 baseline rather than extending it.
That was considered and declined; see AIM.md.
"""

from __future__ import annotations

import argparse
import sys

import structlog

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from db.queries.macro import (
    ANNUAL_TABLE,
    MONTHLY_TABLE,
    coverage,
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
from ingestion.seasonal_calendar import annual_seasonal, monthly_seasonal
from ingestion.worldbank_client import fetch_gdp_cpi

log = structlog.get_logger(__name__)

TRAINING_FIRST_YEAR = 2012
TRAINING_LAST_YEAR = 2026


def _merge_monthly(
    eibor: list[dict],
    brent: list[dict],
    weather: list[dict] | None = None,
    seasonal: list[dict] | None = None,
) -> list[dict]:
    """Outer-join the two genuinely monthly series on month.

    Outer, not inner: EIBOR starts 2018-06 and Brent reaches back to 1987, so an
    inner join would silently discard three decades of Brent to match EIBOR's
    manual-export window. A month present in one source and absent in the other
    is a NULL, which is the truth.
    """
    merged: dict[str, dict] = {}
    for row in eibor:
        merged.setdefault(row["month"], {"month": row["month"]}).update(
            {
                "eibor_3m": row["eibor_3m"],
                "eibor_3m_n_days": row["eibor_3m_n_days"],
            }
        )
    for row in brent:
        merged.setdefault(row["month"], {"month": row["month"]}).update(
            {"brent_crude_usd": row["brent_crude_usd"]}
        )
    for row in weather or []:
        merged.setdefault(row["month"], {"month": row["month"]}).update(
            {
                "temp_max_c": row["temp_max_c"],
                "humidity_pct": row["humidity_pct"],
                "thermal_index": row["thermal_index"],
                "weather_n_days": row["weather_n_days"],
            }
        )
    for row in seasonal or []:
        if row["month"] in merged:
            merged[row["month"]].update(
                {
                    "ramadan_days": row["ramadan_days"],
                    "ramadan_flag": row["ramadan_flag"],
                    "expo_dummy": row["expo_dummy"],
                    "covid_period": row["covid_period"],
                    "covid_break": row["covid_break"],
                }
            )
    return [merged[m] for m in sorted(merged)]


def _merge_annual(
    eibor: list[dict],
    brent: list[dict],
    worldbank: list[dict],
    weather: list[dict] | None = None,
    seasonal: list[dict] | None = None,
) -> list[dict]:
    """Outer-join all three series on year."""
    merged: dict[int, dict] = {}

    for row in eibor:
        merged.setdefault(row["year"], {"year": row["year"]}).update(
            {
                "eibor_3m": row["eibor_3m"],
                "eibor_3m_n_days": row["eibor_3m_n_days"],
                "eibor_3m_n_months": row["eibor_3m_n_months"],
                "eibor_partial": row["partial_year"],
            }
        )
    for row in brent:
        merged.setdefault(row["year"], {"year": row["year"]}).update(
            {
                "brent_crude_usd": row["brent_crude_usd"],
                "brent_n_months": row["brent_n_months"],
                "brent_partial": row["partial_year"],
            }
        )
    for row in worldbank:
        merged.setdefault(row["year"], {"year": row["year"]}).update(
            {
                "uae_gdp_growth": row["uae_gdp_growth"],
                "uae_inflation_pct": row["uae_inflation_pct"],
                "uae_cpi_index": row["uae_cpi_index"],
            }
        )
    for row in weather or []:
        merged.setdefault(row["year"], {"year": row["year"]}).update(
            {
                "temp_max_c": row["temp_max_c"],
                "humidity_pct": row["humidity_pct"],
                "thermal_index": row["thermal_index"],
                "weather_n_days": row["weather_n_days"],
                "weather_partial": row["weather_partial"],
            }
        )
    for row in seasonal or []:
        if row["year"] in merged:
            merged[row["year"]].update(
                {
                    "ramadan_days": row["ramadan_days"],
                    "expo_months": row["expo_months"],
                    "covid_period_months": row["covid_period_months"],
                    "covid_break": row["covid_break"],
                }
            )
    return [merged[y] for y in sorted(merged)]


def _report_training_coverage(annual: list[dict]) -> None:
    """Print, per training year, which macro features would actually be present.

    This is the deliverable of Objective 3.1 as much as the rows are: it shows
    exactly how much of the training span each series can reach, and it makes the
    structural argument concrete rather than asserted.
    """
    by_year = {r["year"]: r for r in annual}
    fields = [
        ("eibor_3m", "EIBOR 3m"),
        ("brent_crude_usd", "Brent"),
        ("uae_gdp_growth", "GDP growth"),
        ("uae_inflation_pct", "Inflation"),
        ("uae_cpi_index", "CPI index"),
        ("thermal_index", "Thermal idx"),
    ]

    print()
    print("Coverage across the Phase-1 training span:")
    print(f"  {'year':<6}" + "".join(f"{label:>12}" for _, label in fields))
    counts = {key: 0 for key, _ in fields}
    for year in range(TRAINING_FIRST_YEAR, TRAINING_LAST_YEAR + 1):
        row = by_year.get(year, {})
        cells = []
        for key, _ in fields:
            value = row.get(key)
            if value is None:
                cells.append(f"{' - ':>12}")
            else:
                counts[key] += 1
                cells.append(f"{float(value):>12.3f}")
        print(f"  {year:<6}" + "".join(cells))

    span = TRAINING_LAST_YEAR - TRAINING_FIRST_YEAR + 1
    print()
    for key, label in fields:
        print(f"  {label:<12} {counts[key]:>2}/{span} training years covered")

    print()
    print("Distinct values available to the model at the Phase-1 join grain:")
    for key, label in fields:
        distinct = {
            by_year[y][key]
            for y in range(TRAINING_FIRST_YEAR, TRAINING_LAST_YEAR + 1)
            if y in by_year and by_year[y].get(key) is not None
        }
        print(f"  {label:<12} {len(distinct):>2} distinct values across 2,552 rows")
    print(
        "\n  `year` alone already separates 15 groups, and every figure above is a\n"
        "  function of it - so these features carry no information the model cannot\n"
        "  already reach. The null is structural, not empirical. See the module\n"
        "  docstring and AIM.md before writing this up."
    )


def _report_seasonal_loss(monthly: list[dict], annual: list[dict]) -> None:
    """Show what annual aggregation costs the weather series (Objective 3.2).

    The distinct-value count that damns the macro series understates the problem
    for weather, because weather's signal is not inter-annual at all - it is
    seasonal. Aggregating to the Phase-1 join grain does not merely coarsen it;
    it averages the entire signal away. This prints the decomposition so the
    claim is measured rather than asserted.
    """
    import statistics as stats
    from collections import defaultdict

    idx_monthly = [r["thermal_index"] for r in monthly if r.get("thermal_index") is not None]
    idx_annual = [
        r["thermal_index"]
        for r in annual
        if r.get("thermal_index") is not None and not r.get("weather_partial")
    ]
    if len(idx_monthly) < 24 or len(idx_annual) < 2:
        return

    by_month_of_year = defaultdict(list)
    for row in monthly:
        if row.get("thermal_index") is not None:
            by_month_of_year[int(row["month"][5:7])].append(row["thermal_index"])
    seasonal = [stats.mean(v) for _, v in sorted(by_month_of_year.items())]

    between = stats.pstdev(idx_annual)
    within = stats.pstdev(seasonal)

    print()
    print("Thermal index - what the annual join grain costs (Objective 3.2):")
    print(
        f"  between-year (annual means) spread {max(idx_annual) - min(idx_annual):6.2f} °C"
        f"   std {between:6.3f}"
    )
    print(
        f"  within-year  (seasonal)     spread {max(seasonal) - min(seasonal):6.2f} °C"
        f"   std {within:6.3f}"
    )
    if between:
        print(f"  seasonal variation is {within / between:.1f}x the inter-annual variation")
    destroyed = 100 * (1 - stats.pvariance(idx_annual) / stats.pvariance(idx_monthly))
    print(f"  variance destroyed by aggregating to year: {destroyed:.1f}%")
    print()
    print(
        "  Dubai weather barely varies between years - it varies WITHIN them. The"
    )
    print(
        "  Phase-1 matrix joins on year, so the seasonal signal, which is the entire"
    )
    print(
        "  signal, is averaged away before the model can see it. This is a sharper"
    )
    print("  case of the same grain problem as the macro series above.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Trim all series to this year onward (default: keep full history).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch, merge and report, but write nothing to the database.",
    )
    args = parser.parse_args()

    print("Fetching EIBOR 3-month (manual export)…")
    eibor_daily = fetch_eibor_daily()
    eibor_monthly = fetch_eibor(daily=eibor_daily)
    eibor_annual = fetch_eibor_annual(daily=eibor_daily)
    print(
        f"  {len(eibor_monthly)} months, {eibor_monthly[0]['month'][:7]} → "
        f"{eibor_monthly[-1]['month'][:7]}"
    )

    print("Fetching Brent crude (EIA)…")
    brent_monthly = fetch_brent(args.start_year)
    brent_annual = fetch_brent_annual(monthly=brent_monthly)
    print(
        f"  {len(brent_monthly)} months, {brent_monthly[0]['month'][:7]} → "
        f"{brent_monthly[-1]['month'][:7]}"
    )

    print("Fetching UAE GDP / inflation / CPI (World Bank)…")
    worldbank = fetch_gdp_cpi(args.start_year)
    print(f"  {len(worldbank)} years, {worldbank[0]['year']} → {worldbank[-1]['year']}")

    print("Fetching Dubai weather (Open-Meteo ERA5 archive)…")
    weather_start = f"{args.start_year or 2012}-01-01"
    weather_daily, weather_prov = fetch_weather_daily(start_date=weather_start)
    weather_monthly = fetch_weather(daily=weather_daily)
    weather_annual = fetch_weather_annual(daily=weather_daily)
    print(
        f"  {len(weather_daily)} days, {weather_daily[0]['date']} → "
        f"{weather_daily[-1]['date']}"
    )
    print(
        f"  grid cell {weather_prov['resolved_lat']}, {weather_prov['resolved_lon']} "
        f"(requested {weather_prov['requested_lat']}, {weather_prov['requested_lon']})"
    )

    print("Building seasonal calendar (Ramadan / Expo / COVID)…")
    first_year = args.start_year or TRAINING_FIRST_YEAR
    seasonal_monthly = monthly_seasonal(first_year, TRAINING_LAST_YEAR)
    seasonal_annual = annual_seasonal(first_year, TRAINING_LAST_YEAR)
    print(
        f"  {len(seasonal_monthly)} months; "
        f"{sum(1 for r in seasonal_monthly if r['ramadan_days'] > 0)} contain Ramadan days"
    )

    monthly = _merge_monthly(
        eibor_monthly, brent_monthly, weather_monthly, seasonal_monthly
    )
    annual = _merge_annual(
        eibor_annual, brent_annual, worldbank, weather_annual, seasonal_annual
    )
    if args.start_year is not None:
        monthly = [r for r in monthly if int(r["month"][:4]) >= args.start_year]
        annual = [r for r in annual if r["year"] >= args.start_year]

    print(f"\nMerged: {len(monthly)} monthly rows, {len(annual)} annual rows")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        _report_training_coverage(annual)
        return 0

    ensure_tables()
    n_monthly = upsert_macro_monthly(monthly)
    n_annual = upsert_macro_annual(annual)
    print(f"Wrote {n_monthly} rows to {MONTHLY_TABLE}, {n_annual} to {ANNUAL_TABLE}")

    stats = coverage()
    print(
        f"\n{MONTHLY_TABLE}: {stats['monthly_rows']} rows, "
        f"{stats['monthly_first']} → {stats['monthly_last']}"
    )
    print(
        f"{ANNUAL_TABLE}:     {stats['annual_rows']} rows, "
        f"{stats['annual_first']} → {stats['annual_last']}"
    )
    for key, value in stats["annual_non_null"].items():
        print(f"  {key:<20} {value} non-null")

    _report_training_coverage(annual)
    _report_seasonal_loss(monthly, annual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
