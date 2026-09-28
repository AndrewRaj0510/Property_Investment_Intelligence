"""M1E - Objective 3.1 macro ingestion (EIBOR, Brent, World Bank).

Covers the three properties that actually matter for this objective:

1. **Nothing is fabricated.** A missing EIBOR export, a missing EIA key, or a
   World Bank outage raises - none of them returns an empty or filled-in series
   that would later read as "the data says nothing happened".
2. **Aggregation is honest.** Monthly and annual means are taken over the
   observations actually present, carry their observation counts, and flag
   part-years.
3. **The grain is not faked.** The World Bank client returns annual rows, not an
   annual value copied across twelve months.
"""

from __future__ import annotations

import datetime as dt

import pytest

from ingestion import eibor_client, worldbank_client
from ingestion.eia_client import BrentSourceError, fetch_brent, fetch_brent_annual
from ingestion.eibor_client import (
    EiborSourceError,
    _resolve_columns,
    fetch_eibor,
    fetch_eibor_annual,
)
from ingestion.worldbank_client import WorldBankSourceError


def test_missing_export_raises_rather_than_returning_empty():
    """An absent workbook must stop the run, not ingest as 'no data'."""
    with pytest.raises(EiborSourceError, match="not found"):
        eibor_client.fetch_eibor_daily("data/does_not_exist.xlsx")


def test_columns_resolved_by_name_not_position():
    header = ["Date", "Overnight", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
    assert _resolve_columns(header) == (0, 4)
    assert _resolve_columns(["3 Months", "Date"]) == (1, 0)


def test_missing_3m_tenor_raises():
    with pytest.raises(EiborSourceError, match="3 months"):
        _resolve_columns(["Date", "Overnight", "1 Year"])


def _daily(*pairs):
    return [{"date": dt.date(*d), "eibor_3m": v} for d, v in pairs]


def test_monthly_mean_carries_its_day_count():
    rows = fetch_eibor(
        daily=_daily(((2020, 1, 6), 2.0), ((2020, 1, 7), 4.0), ((2020, 2, 3), 1.0))
    )
    assert rows[0] == {"month": "2020-01-01", "eibor_3m": 3.0, "eibor_3m_n_days": 2}
    assert rows[1] == {"month": "2020-02-01", "eibor_3m": 1.0, "eibor_3m_n_days": 1}


def test_annual_mean_weights_by_day_not_by_month():
    """Two days in January, one in February - January must weigh double.

    Averaging the monthly means instead would give (3.0 + 1.0) / 2 = 2.0.
    """
    rows = fetch_eibor_annual(
        daily=_daily(((2020, 1, 6), 2.0), ((2020, 1, 7), 4.0), ((2020, 2, 3), 1.0))
    )
    assert rows[0]["eibor_3m"] == pytest.approx((2.0 + 4.0 + 1.0) / 3, abs=1e-5)
    assert rows[0]["eibor_3m"] != pytest.approx(2.0, abs=1e-3)
    assert rows[0]["eibor_3m_n_days"] == 3
    assert rows[0]["eibor_3m_n_months"] == 2


def test_partial_year_is_flagged():
    """The export starts mid-2018 and stops at the download date, so the first
    and last years average part of a year and are not comparable with a full one."""
    partial = fetch_eibor_annual(daily=_daily(((2018, 6, 11), 2.5)))
    assert partial[0]["partial_year"] is True

    full = fetch_eibor_annual(
        daily=_daily(*(((2019, m, 15), 2.0) for m in range(1, 13)))
    )
    assert full[0]["partial_year"] is False
    assert full[0]["eibor_3m_n_months"] == 12


def test_real_export_parses_if_present():
    """Guard against a re-export changing shape underneath the parser."""
    try:
        rows = eibor_client.fetch_eibor_daily()
    except EiborSourceError:
        pytest.skip("EIBOR export not present in this checkout")
    assert rows, "export parsed to zero rows"
    assert all(0.0 <= r["eibor_3m"] <= 20.0 for r in rows)
    assert rows == sorted(rows, key=lambda r: r["date"])


def test_brent_without_key_raises(monkeypatch):
    from config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EIA_API_KEY", "")
    with pytest.raises(BrentSourceError, match="EIA_API_KEY"):
        fetch_brent()
    get_settings.cache_clear()


def test_brent_annual_reuses_supplied_months_without_a_request():
    """Passing the monthly series in must not trigger a second API call."""
    monthly = [
        {"month": "2020-01-01", "brent_crude_usd": 60.0},
        {"month": "2020-02-01", "brent_crude_usd": 40.0},
        {"month": "2021-01-01", "brent_crude_usd": 70.0},
    ]
    rows = fetch_brent_annual(monthly=monthly)
    assert rows[0] == {
        "year": 2020,
        "brent_crude_usd": 50.0,
        "brent_n_months": 2,
        "partial_year": True,
    }
    assert rows[1]["year"] == 2021


def test_worldbank_returns_annual_rows_not_forward_filled_months(monkeypatch):
    """The previous client emitted 12 rows per year, inventing monthly variation."""
    monkeypatch.setattr(
        worldbank_client,
        "_fetch_indicator",
        lambda indicator: {2020: 1.0, 2021: 2.0},
    )
    rows = worldbank_client.fetch_gdp_cpi()
    assert [r["year"] for r in rows] == [2020, 2021]
    assert "month" not in rows[0]


def test_worldbank_missing_year_stays_none(monkeypatch):
    """World Bank publishes with a lag; absent years must not be carried forward."""
    series = {
        worldbank_client.GDP_INDICATOR: {2020: 1.0},
        worldbank_client.CPI_RATE_INDICATOR: {2020: 2.0, 2021: 3.0},
        worldbank_client.CPI_INDEX_INDICATOR: {2020: 100.0, 2021: 103.0},
    }
    monkeypatch.setattr(
        worldbank_client, "_fetch_indicator", lambda indicator: series[indicator]
    )
    rows = {r["year"]: r for r in worldbank_client.fetch_gdp_cpi()}
    assert rows[2021]["uae_gdp_growth"] is None
    assert rows[2021]["uae_inflation_pct"] == 3.0


def test_worldbank_total_outage_raises(monkeypatch):
    """An empty result from every indicator must not write an empty macro series.

    The previous client swallowed exceptions and returned {}, making a transient
    outage indistinguishable from 'the World Bank has no UAE data'.
    """
    monkeypatch.setattr(worldbank_client, "_fetch_indicator", lambda indicator: {})
    with pytest.raises(WorldBankSourceError):
        worldbank_client.fetch_gdp_cpi()


def test_start_year_trims(monkeypatch):
    monkeypatch.setattr(
        worldbank_client,
        "_fetch_indicator",
        lambda indicator: {2018: 1.0, 2019: 2.0, 2020: 3.0},
    )
    rows = worldbank_client.fetch_gdp_cpi(start_year=2019)
    assert [r["year"] for r in rows] == [2019, 2020]


def test_monthly_merge_is_outer_not_inner():
    """EIBOR starts 2018-06; an inner join would discard Brent's earlier decades."""
    from scripts.ingest_macro import _merge_monthly

    rows = _merge_monthly(
        [{"month": "2018-06-01", "eibor_3m": 2.5, "eibor_3m_n_days": 8}],
        [
            {"month": "2012-01-01", "brent_crude_usd": 110.0},
            {"month": "2018-06-01", "brent_crude_usd": 74.0},
        ],
    )
    by_month = {r["month"]: r for r in rows}
    assert set(by_month) == {"2012-01-01", "2018-06-01"}
    assert "eibor_3m" not in by_month["2012-01-01"]
    assert by_month["2018-06-01"]["eibor_3m"] == 2.5
