"""Seasonal calendar features (M1E) - Objective 3.3, dummy half.

Derived, not fetched: these need no external source, which is why they are
delivered while the tourism half of Objective 3.3 waits on a manual export.

Three feature families:

* **Ramadan** - computed from the Hijri calendar, not hardcoded.
* **Expo 2020 Dubai** - the event window (it ran Oct 2021 - Mar 2022 after the
  pandemic delay, despite the name).
* **COVID-19** - an acute-disruption pulse and a permanent structural-break step.

---

**Why Ramadan is computed rather than tabulated.** The previous implementation
carried a hardcoded set of ``(year, month)`` pairs:

    {(2018, 5), (2019, 5), (2020, 4), (2021, 4), (2022, 4), (2023, 3), (2024, 3)}

Checked against the actual Hijri calendar over the 2012-2026 training span, that
table is wrong three ways:

1. **It covers 7 of the 29 months that actually contain Ramadan days.**
2. **Eight of the fifteen training years carry no flag at all** - 2012-2017 and
   2025-2026 are simply absent, so those years read as "no Ramadan", which is
   silently false rather than missing.
3. **For 2020 and 2023 it flags the wrong month.** Ramadan 2020 ran 24 Apr -
   23 May: 7 days in April, 23 in May. The table flags **April**. Ramadan 2023 ran
   23 Mar - 20 Apr: 9 days in March, 20 in April. The table flags **March**. In
   both cases the flagged month is the one with *fewer* Ramadan days.

The root cause is that a lunar month rarely aligns with a Gregorian one - it
straddles two in 14 of the 15 training years (2025 is the sole exception) - so "which month is Ramadan in?" has no
single correct answer. Recording ``ramadan_days`` (how many days of the month
fall inside Ramadan) removes the question instead of answering it badly. The
binary flag the objective asks for is derived from that count, and is documented
as a threshold rather than a fact.

**A consequence worth stating in the write-up.** Because Ramadan is a fixed-length
lunar month, its *annual* day-count is ~29-30 in every single year. At the
Phase-1 annual join grain it is therefore **constant by construction** - not
merely low-variance like the macro series, but carrying exactly zero information,
derivable from first principles without looking at the data. Its real signal is
*which season* Ramadan falls in, which only exists at monthly grain.
"""

from __future__ import annotations

import datetime as dt
import warnings
from collections import defaultdict

import structlog

log = structlog.get_logger(__name__)

EXPO_START = dt.date(2021, 10, 1)
EXPO_END = dt.date(2022, 3, 31)

COVID_BREAK = dt.date(2020, 3, 11)
COVID_ACUTE_END = dt.date(2021, 6, 30)

RAMADAN_FLAG_MIN_DAYS = 15


def _hijri():
    """Import the Hijri converter, silencing its rename deprecation notice.

    ``hijri-converter`` was renamed to ``hijridate``; the pinned version still
    works and emits a DeprecationWarning on import. Swapping the dependency is a
    separate change from delivering this objective.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from hijri_converter import Hijri

        return Hijri


def ramadan_days_by_month(first_year: int, last_year: int) -> dict[tuple[int, int], int]:
    """``{(year, month): days of that month falling inside Ramadan}``.

    Ramadan is Hijri month 9. Hijri years are enumerated generously around the
    requested Gregorian span because the two calendars drift ~11 days a year.
    """
    Hijri = _hijri()
    counts: dict[tuple[int, int], int] = defaultdict(int)

    first_hijri = int((first_year - 622) * 33 / 32) - 2
    last_hijri = int((last_year - 622) * 33 / 32) + 2

    for hijri_year in range(first_hijri, last_hijri + 1):
        try:
            start = Hijri(hijri_year, 9, 1).to_gregorian()
            try:
                end = Hijri(hijri_year, 9, 30).to_gregorian()
            except (ValueError, OverflowError):
                end = Hijri(hijri_year, 9, 29).to_gregorian()
        except (ValueError, OverflowError):
            continue

        day = dt.date(start.year, start.month, start.day)
        final = dt.date(end.year, end.month, end.day)
        while day <= final:
            if first_year <= day.year <= last_year:
                counts[(day.year, day.month)] += 1
            day += dt.timedelta(days=1)

    return dict(counts)


def _month_bounds(year: int, month: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, month, 1)
    end = (
        dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
    ) - dt.timedelta(days=1)
    return start, end


def _overlap_days(a_start: dt.date, a_end: dt.date, b_start: dt.date, b_end: dt.date) -> int:
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return max(0, (earliest_end - latest_start).days + 1)


def monthly_seasonal(first_year: int = 2012, last_year: int = 2026) -> list[dict]:
    """One row per month with every calendar feature.

    Returns rows shaped ``{'month': 'YYYY-MM-01', 'ramadan_days': int,
    'ramadan_flag': bool, 'expo_dummy': bool, 'covid_period': bool,
    'covid_break': bool}``.
    """
    ramadan = ramadan_days_by_month(first_year, last_year)
    out = []
    for year in range(first_year, last_year + 1):
        for month in range(1, 13):
            start, end = _month_bounds(year, month)
            days = ramadan.get((year, month), 0)
            out.append(
                {
                    "month": f"{year}-{month:02d}-01",
                    "ramadan_days": days,
                    "ramadan_flag": days >= RAMADAN_FLAG_MIN_DAYS,
                    "expo_dummy": _overlap_days(start, end, EXPO_START, EXPO_END) > 0,
                    "covid_period": _overlap_days(
                        start, end, COVID_BREAK, COVID_ACUTE_END
                    ) > 0,
                    "covid_break": end >= COVID_BREAK,
                }
            )
    log.info(
        "seasonal_calendar_built",
        months=len(out),
        ramadan_months=sum(1 for r in out if r["ramadan_days"] > 0),
        ramadan_flagged=sum(1 for r in out if r["ramadan_flag"]),
        expo_months=sum(1 for r in out if r["expo_dummy"]),
        covid_acute_months=sum(1 for r in out if r["covid_period"]),
    )
    return out


def annual_seasonal(first_year: int = 2012, last_year: int = 2026) -> list[dict]:
    """Annual roll-up - the grain the Phase-1 matrix joins on.

    ``ramadan_days`` is included deliberately so the write-up can point at it:
    it is ~29-30 for **every** year, because Ramadan is a fixed-length lunar
    month. It is the cleanest example in the project of a feature that carries
    zero information at the annual grain, and unlike the macro series that can be
    shown from the calendar alone, without measuring anything.
    """
    monthly = monthly_seasonal(first_year, last_year)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in monthly:
        buckets[int(row["month"][:4])].append(row)

    return [
        {
            "year": year,
            "ramadan_days": sum(r["ramadan_days"] for r in rows),
            "expo_months": sum(1 for r in rows if r["expo_dummy"]),
            "covid_period_months": sum(1 for r in rows if r["covid_period"]),
            "covid_break": any(r["covid_break"] for r in rows),
        }
        for year, rows in sorted(buckets.items())
    ]
