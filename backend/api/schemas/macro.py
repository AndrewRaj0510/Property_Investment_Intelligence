"""Macro + seasonal schemas (M4-02)."""

from __future__ import annotations

from pydantic import BaseModel


class MacroSnapshot(BaseModel):
    month: str
    eibor_3m: float | None
    uae_gdp_growth: float | None
    uae_cpi: float | None
    brent_crude_usd: float | None
    temp_max_c: float | None
    humidity_pct: float | None
    thermal_index: float | None
    tourist_arrivals: int | None
    hotel_occupancy: float | None
    ramadan_flag: bool
    expo_dummy: bool
    model_phase: int
    eibor_12m_avg: float | None = None
    brent_12m_avg: float | None = None
    eibor_as_of: str | None = None
    brent_as_of: str | None = None


class SeasonalMonth(BaseModel):
    """One month-of-year row of the seasonal calendar (Objective 3.5).

    **This is a CLIMATE calendar, not a yield calendar, and the distinction is
    not cosmetic.** The original schema carried ``yield_index`` - a relative
    demand index derived from monthly tourist arrivals. Two things killed it:
    the tourism series was de-scoped on measured evidence (F31), and, more
    fundamentally, **this project has no monthly yield to be seasonal about.**
    The training matrix is built at (community x type x bedrooms x YEAR), so a
    monthly yield index would have to be invented. It was removed rather than
    left permanently null, because a null field named ``yield_index`` invites
    exactly the reading it cannot support.

    What replaces it is real: ``thermal_index_relative`` indexes the measured
    thermal discomfort of each month against the annual mean (100 = average), so
    the 24.6 degC seasonal swing that F29 measured is displayed as what it is -
    a climate pattern, with no yield claim attached.
    """

    month_of_year: int
    month_name: str
    avg_temp_max_c: float | None
    avg_thermal_index: float | None
    thermal_index_relative: float | None = None
    avg_ramadan_days: float | None = None
    months_observed: int = 0


class SeasonalCalendar(BaseModel):
    """The seasonal calendar plus the caveats a reader needs to not misread it."""

    months: list[SeasonalMonth]
    basis: str
    note: str
