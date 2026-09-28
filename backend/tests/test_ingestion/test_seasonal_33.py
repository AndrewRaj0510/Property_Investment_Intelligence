"""M1E - Objective 3.3, seasonal-dummy half (Ramadan / Expo / COVID).

The Ramadan tests pin the defect these features replace: a hardcoded
``(year, month)`` table that covered 7 of the 29 months actually containing
Ramadan days, left 8 of 15 training years unflagged, and for 2020 and 2023
flagged the month with *fewer* Ramadan days than the one it omitted.
"""

from __future__ import annotations

import pytest

from ingestion.seasonal_calendar import (
    COVID_BREAK,
    EXPO_END,
    EXPO_START,
    annual_seasonal,
    monthly_seasonal,
    ramadan_days_by_month,
)

_LEGACY_RAMADAN_MONTHS = {
    (2018, 5), (2019, 5), (2020, 4), (2021, 4), (2022, 4), (2023, 3), (2024, 3),
}


def test_ramadan_spans_two_gregorian_months_in_most_years():
    """A lunar month rarely aligns with a Gregorian one - this is the root cause."""
    counts = ramadan_days_by_month(2012, 2026)
    years = {}
    for (year, _), days in counts.items():
        years.setdefault(year, 0)
        years[year] += days
    for year in range(2012, 2026):
        assert 29 <= years[year] <= 30, (year, years[year])
    straddled = sum(
        1
        for year in range(2012, 2027)
        if len([1 for (y, _) in counts if y == year]) > 1
    )
    assert straddled >= 12


def test_legacy_table_missed_most_ramadan_months():
    counts = ramadan_days_by_month(2012, 2026)
    assert len(counts) == 29
    missed = set(counts) - _LEGACY_RAMADAN_MONTHS
    assert len(missed) == 22


def test_legacy_table_left_eight_training_years_unflagged():
    counts = ramadan_days_by_month(2012, 2026)
    covered = {y for y, _ in counts}
    legacy = {y for y, _ in _LEGACY_RAMADAN_MONTHS}
    assert covered - legacy == {2012, 2013, 2014, 2015, 2016, 2017, 2025, 2026}


@pytest.mark.parametrize(
    "year, legacy_month, correct_month",
    [(2020, 4, 5), (2023, 3, 4)],
)
def test_legacy_table_flagged_the_minority_month(year, legacy_month, correct_month):
    """2020 and 2023 are the cases where the old table is not merely incomplete."""
    counts = ramadan_days_by_month(2012, 2026)
    assert (year, legacy_month) in _LEGACY_RAMADAN_MONTHS
    assert counts[(year, correct_month)] > counts[(year, legacy_month)]

    flagged = {
        r["month"][:7] for r in monthly_seasonal(year, year) if r["ramadan_flag"]
    }
    assert f"{year}-{correct_month:02d}" in flagged
    assert f"{year}-{legacy_month:02d}" not in flagged


def test_exactly_one_ramadan_month_flagged_per_year():
    """The majority-share threshold must resolve every year to one month."""
    rows = monthly_seasonal(2012, 2026)
    per_year: dict[int, int] = {}
    for row in rows:
        if row["ramadan_flag"]:
            per_year[int(row["month"][:4])] = per_year.get(int(row["month"][:4]), 0) + 1
    assert set(per_year) == set(range(2012, 2027))
    assert set(per_year.values()) == {1}


def test_annual_ramadan_days_is_constant_by_construction():
    """The point of the Objective 3.3 write-up: zero information at annual grain.

    A fixed-length lunar month has the same duration every year, so the annual
    total can only ever be 29 or 30 - derivable from the calendar without looking
    at any data.
    """
    values = {r["ramadan_days"] for r in annual_seasonal(2012, 2026)}
    assert values <= {29, 30}


def test_expo_window_is_the_actual_event_not_the_name():
    """Expo *2020* ran Oct 2021 - Mar 2022 after the pandemic delay."""
    rows = {r["month"]: r for r in monthly_seasonal(2020, 2022)}
    assert rows["2020-10-01"]["expo_dummy"] is False
    assert rows["2021-10-01"]["expo_dummy"] is True
    assert rows["2022-03-01"]["expo_dummy"] is True
    assert rows["2022-04-01"]["expo_dummy"] is False
    assert EXPO_START.year == 2021 and EXPO_END.year == 2022


def test_covid_break_is_a_step_and_the_period_is_a_pulse():
    """A structural break is permanent; the acute disruption is not."""
    rows = {r["month"]: r for r in monthly_seasonal(2019, 2026)}
    assert rows["2020-02-01"]["covid_break"] is False
    assert rows["2020-03-01"]["covid_break"] is True
    assert rows["2026-01-01"]["covid_break"] is True

    assert rows["2020-06-01"]["covid_period"] is True
    assert rows["2022-01-01"]["covid_period"] is False
    assert COVID_BREAK.year == 2020


def test_covid_break_was_absent_from_the_codebase_before_this():
    """Regression guard for the gap AIM.md recorded: 'covid' matched nothing."""
    rows = monthly_seasonal(2012, 2026)
    assert any(r["covid_break"] for r in rows)
    assert any(r["covid_period"] for r in rows)


def test_every_month_in_span_is_present():
    rows = monthly_seasonal(2012, 2026)
    assert len(rows) == 15 * 12
    assert rows[0]["month"] == "2012-01-01"
    assert rows[-1]["month"] == "2026-12-01"
