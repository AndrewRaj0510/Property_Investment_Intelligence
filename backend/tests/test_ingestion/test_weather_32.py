"""M1E-04 - Objective 3.2 weather ingestion and the thermal discomfort index.

The index tests are the substantive ones. The formula the project previously
specified - ``temp_max_c + 0.33 * humidity_pct - 4`` - substitutes relative
humidity (%) for water vapour pressure (hPa) in the Australian BoM apparent
temperature equation. These tests pin the corrected physics and pin the size of
the error, so the finding in F29 can be regenerated rather than trusted.
"""

from __future__ import annotations

import datetime as dt

import pytest

from ingestion import openmeteo_client
from ingestion.openmeteo_client import (
    WeatherSourceError,
    _parse_range_bound,
    fetch_weather,
    fetch_weather_annual,
    legacy_thermal_index,
    saturation_vapour_pressure_hpa,
    thermal_index,
    vapour_pressure_hpa,
)


def test_saturation_vapour_pressure_matches_known_values():
    """Magnus-Tetens: ~6.11 hPa at 0 °C, ~23.4 hPa at 20 °C."""
    assert saturation_vapour_pressure_hpa(0.0) == pytest.approx(6.105, abs=0.01)
    assert saturation_vapour_pressure_hpa(20.0) == pytest.approx(23.4, abs=0.2)


def test_vapour_pressure_scales_with_relative_humidity():
    """At 50% RH the actual vapour pressure is half the saturation value."""
    assert vapour_pressure_hpa(30.0, 50.0) == pytest.approx(
        0.5 * saturation_vapour_pressure_hpa(30.0)
    )
    assert vapour_pressure_hpa(30.0, 0.0) == 0.0


def test_index_uses_vapour_pressure_not_raw_humidity():
    """The whole point of the fix: 0.33 multiplies hPa, not percent."""
    temp, rh = 35.0, 60.0
    expected = temp + 0.33 * vapour_pressure_hpa(temp, rh) - 4.00
    assert thermal_index(temp, rh) == pytest.approx(expected)
    assert thermal_index(temp, rh) != pytest.approx(legacy_thermal_index(temp, rh), abs=0.5)


def test_legacy_formula_overstates_most_at_mild_and_humid():
    """A mild, humid day is where the unit error does the real damage.

    2016-03-09 in the real record: 23.5 °C at 87% RH. The legacy formula scored
    it 48.2 - worse than a 42.9 °C August day - because it added 0.33 x 87
    instead of 0.33 x (vapour pressure).
    """
    mild_humid_legacy = legacy_thermal_index(23.5, 87.0)
    mild_humid_correct = thermal_index(23.5, 87.0)
    assert mild_humid_legacy > 47.0
    assert mild_humid_correct < 30.0
    assert mild_humid_legacy - mild_humid_correct > 18.0


def test_legacy_and_correct_agree_on_a_hot_dry_day():
    """Why the error survived review: a peak-summer spot check passes.

    2018-08-09: 47.2 °C at 41% RH - the two formulas land within ~1 °C, so
    validating the index against the hottest day in the record would have shown
    nothing wrong. Same silent-plausible-wrong-answer trap as F22.
    """
    assert abs(legacy_thermal_index(47.2, 41.0) - thermal_index(47.2, 41.0)) < 1.5


def test_index_is_monotonic_in_both_inputs():
    assert thermal_index(40.0, 60.0) > thermal_index(30.0, 60.0)
    assert thermal_index(35.0, 80.0) > thermal_index(35.0, 20.0)


def _days(*triples):
    return [
        {"date": dt.date(*d), "temp_max_c": t, "humidity_pct": h}
        for d, t, h in triples
    ]


def test_index_is_averaged_per_day_not_computed_from_monthly_means():
    """Vapour pressure is exponential in temperature, so the two differ.

    Computing the index from a month's mean temperature understates it whenever
    the daily swing is wide (Jensen's inequality).
    """
    daily = _days(((2020, 6, 1), 30.0, 60.0), ((2020, 6, 2), 46.0, 60.0))
    monthly = fetch_weather(daily=daily)
    per_day = (thermal_index(30.0, 60.0) + thermal_index(46.0, 60.0)) / 2
    from_means = thermal_index(38.0, 60.0)
    assert monthly[0]["thermal_index"] == pytest.approx(per_day, abs=0.01)
    assert monthly[0]["thermal_index"] > from_means


def test_monthly_and_annual_carry_day_counts_and_partial_flag():
    daily = _days(((2020, 1, 5), 24.0, 60.0), ((2020, 1, 6), 26.0, 62.0))
    monthly = fetch_weather(daily=daily)
    assert monthly[0]["month"] == "2020-01-01"
    assert monthly[0]["weather_n_days"] == 2
    assert monthly[0]["temp_max_c"] == pytest.approx(25.0)

    annual = fetch_weather_annual(daily=daily)
    assert annual[0]["weather_n_days"] == 2
    assert annual[0]["weather_partial"] is True


def test_full_year_is_not_flagged_partial():
    daily = _days(*(((2019, m, 15), 30.0, 55.0) for m in range(1, 13)))
    assert fetch_weather_annual(daily=daily)[0]["weather_partial"] is False


def test_archive_upper_bound_is_parsed_from_the_rejection():
    """The API names its own limit; honour it rather than guessing."""
    reason = (
        "Parameter 'end_date' is out of allowed range from 1940-01-01 to 2026-09-05"
    )
    assert _parse_range_bound(reason) == "2026-09-05"
    assert _parse_range_bound("some other error") is None
    assert _parse_range_bound("out of allowed range from x to not-a-date") is None


def test_unusable_response_raises(monkeypatch):
    monkeypatch.setattr(openmeteo_client, "_get", lambda url, params: {"unexpected": 1})
    with pytest.raises(WeatherSourceError, match="Unexpected"):
        openmeteo_client.fetch_weather_daily()


def test_all_null_observations_raise(monkeypatch):
    """An all-null window must stop the run, not persist as 'no weather'."""
    monkeypatch.setattr(
        openmeteo_client,
        "_get",
        lambda url, params: {
            "latitude": 25.2,
            "longitude": 55.3,
            "daily": {
                "time": ["2020-01-01", "2020-01-02"],
                "temperature_2m_max": [None, None],
                "relative_humidity_2m_mean": [None, None],
            },
        },
    )
    with pytest.raises(WeatherSourceError, match="no usable"):
        openmeteo_client.fetch_weather_daily()


def test_provenance_records_the_resolved_grid_cell(monkeypatch):
    """The API snaps to a grid centre; a value measured elsewhere than asked for
    is the F22 failure, so the resolved point is kept."""
    monkeypatch.setattr(
        openmeteo_client,
        "_get",
        lambda url, params: {
            "latitude": 25.202108,
            "longitude": 55.302376,
            "elevation": 0.0,
            "timezone": "Asia/Dubai",
            "daily": {
                "time": ["2020-01-01"],
                "temperature_2m_max": [24.0],
                "relative_humidity_2m_mean": [60.0],
            },
        },
    )
    _, provenance = openmeteo_client.fetch_weather_daily()
    assert provenance["resolved_lat"] == 25.202108
    assert provenance["requested_lat"] == openmeteo_client.DUBAI_LAT
    assert provenance["resolved_lon"] != provenance["requested_lon"]


def test_out_of_range_temperatures_are_rejected(monkeypatch):
    monkeypatch.setattr(
        openmeteo_client,
        "_get",
        lambda url, params: {
            "latitude": 25.2,
            "longitude": 55.3,
            "daily": {
                "time": ["2020-01-01", "2020-01-02"],
                "temperature_2m_max": [999.0, 24.0],
                "relative_humidity_2m_mean": [60.0, 60.0],
            },
        },
    )
    rows, _ = openmeteo_client.fetch_weather_daily()
    assert len(rows) == 1
    assert rows[0]["temp_max_c"] == 24.0
