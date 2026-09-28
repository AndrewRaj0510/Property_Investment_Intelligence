"""UAE Central Bank EIBOR 3-month client (M1E-01) - Objective 3.1.

Parses a **manually downloaded** workbook at ``backend/data/eibor_rates.xlsx``.

**Why a manual export rather than a fetch.** ``centralbank.ae`` sits behind
Cloudflare: both the EIBOR rates page and its Umbraco surface endpoint return
HTTP 403 with a "Just a moment..." interstitial to any programmatic client. No
substitute machine-readable source exists either - the IMF ``dataservices``
endpoint is retired (DNS no longer resolves), the IMF SDMX API carries no IFS
dataflow for this series, IMF DataMapper returns 403, the UAE Bayanat open-data
API 404s, and FRED's 3-month interbank family (``IR3TIB01*``) covers OECD members
only, which the UAE is not. So the emirate's own central bank publishes its
benchmark rate in a form no automated pipeline can reach.

That is the **second** official Dubai/UAE source in this project blocked by bot
protection, after Bayut (DataDome) in Objective 1.2, and it is a finding about
data availability in the emirate rather than an implementation gap. The design
consequence is the same one ``tourism_client`` already makes: a human exports the
file, and this module fails loudly if it is absent so nothing fabricated is ever
persisted.

**Grain.** The export is *daily* (business days only). Nothing here forward-fills
or interpolates: monthly and annual figures are means over the days actually
present, and every aggregate carries the observation count that produced it, so a
thin month is visible in the data rather than hidden by a smooth-looking number.
"""

from __future__ import annotations

import datetime as dt
import os
from collections import defaultdict

import structlog

log = structlog.get_logger(__name__)

DEFAULT_XLSX = os.path.join("data", "eibor_rates.xlsx")

_DATE_HEADER = "date"
_TENOR_3M_HEADER = "3 months"

_DATE_FORMATS = ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y")

_MIN_PLAUSIBLE = 0.0
_MAX_PLAUSIBLE = 20.0


class EiborSourceError(RuntimeError):
    """The EIBOR export is missing, unreadable, or not shaped as expected."""


def _parse_date(value) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _resolve_columns(header: list) -> tuple[int, int]:
    """Locate the date and 3-month columns by name, not by position.

    Position would be the fragile choice: the export carries six tenors plus a
    trailing ``ValueDate``, and reading the wrong one yields a *plausible*
    interest rate rather than an error - the same silent-wrong-answer failure
    mode that the Objective 1.4 OSM sweep hit six times (F22).
    """
    normalised = [str(h).strip().lower() if h is not None else "" for h in header]
    try:
        date_idx = normalised.index(_DATE_HEADER)
    except ValueError as exc:
        raise EiborSourceError(
            f"No '{_DATE_HEADER}' column in the EIBOR export; found {header!r}"
        ) from exc
    try:
        rate_idx = normalised.index(_TENOR_3M_HEADER)
    except ValueError as exc:
        raise EiborSourceError(
            f"No '{_TENOR_3M_HEADER}' column in the EIBOR export; found {header!r}. "
            "Objective 3.1 specifies the 3-month tenor - do not substitute another."
        ) from exc
    return date_idx, rate_idx


def fetch_eibor_daily(xlsx_path: str = DEFAULT_XLSX) -> list[dict]:
    """Every business-day 3-month EIBOR observation in the export.

    Returns ``[{'date': date, 'eibor_3m': float}, ...]`` sorted by date.
    """
    if not os.path.exists(xlsx_path):
        raise EiborSourceError(
            f"EIBOR export not found at {xlsx_path}. Download the 3-month EIBOR "
            "history from centralbank.ae in a browser (the site blocks automated "
            "clients) and save it there. There is no synthetic fallback."
        )

    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise EiborSourceError(
            "openpyxl is required to read the EIBOR workbook; it is pinned in "
            "requirements.txt."
        ) from exc

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)

    try:
        header = list(next(rows))
    except StopIteration as exc:
        raise EiborSourceError(f"EIBOR export {xlsx_path} is empty") from exc
    date_idx, rate_idx = _resolve_columns(header)

    out: list[dict] = []
    skipped_date = skipped_value = skipped_range = 0
    for record in rows:
        if record is None or len(record) <= max(date_idx, rate_idx):
            skipped_date += 1
            continue
        day = _parse_date(record[date_idx])
        if day is None:
            skipped_date += 1
            continue
        raw = record[rate_idx]
        if raw is None or str(raw).strip() == "":
            skipped_value += 1
            continue
        try:
            rate = float(str(raw).strip())
        except (TypeError, ValueError):
            skipped_value += 1
            continue
        if not _MIN_PLAUSIBLE <= rate <= _MAX_PLAUSIBLE:
            skipped_range += 1
            continue
        out.append({"date": day, "eibor_3m": rate})

    workbook.close()
    if not out:
        raise EiborSourceError(
            f"EIBOR export {xlsx_path} parsed to zero usable rows - check the "
            "sheet layout."
        )

    out.sort(key=lambda r: r["date"])
    log.info(
        "eibor_parsed",
        rows=len(out),
        first=str(out[0]["date"]),
        last=str(out[-1]["date"]),
        skipped_date=skipped_date,
        skipped_value=skipped_value,
        skipped_out_of_range=skipped_range,
    )
    return out


def fetch_eibor(
    xlsx_path: str = DEFAULT_XLSX, daily: list[dict] | None = None
) -> list[dict]:
    """Monthly mean 3-month EIBOR.

    Returns ``[{'month': 'YYYY-MM-01', 'eibor_3m': float, 'eibor_3m_n_days': int}]``.
    The day count travels with the value so a month built from three
    observations is not mistaken for one built from twenty-two.
    """
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in (daily if daily is not None else fetch_eibor_daily(xlsx_path)):
        day = row["date"]
        buckets[(day.year, day.month)].append(row["eibor_3m"])

    return [
        {
            "month": f"{year}-{month:02d}-01",
            "eibor_3m": round(sum(values) / len(values), 5),
            "eibor_3m_n_days": len(values),
        }
        for (year, month), values in sorted(buckets.items())
    ]


def fetch_eibor_annual(
    xlsx_path: str = DEFAULT_XLSX, daily: list[dict] | None = None
) -> list[dict]:
    """Annual mean 3-month EIBOR - the grain the Phase-1 matrix joins on.

    The mean is taken over **days**, not over monthly means, so months with more
    trading days weigh proportionally. ``partial_year`` marks a year the export
    does not cover end to end: the manual export starts mid-2018 and stops at the
    download date, so the first and last years average a *part* of the year and
    their level is not comparable with a full year's. Recording that as data
    keeps the caveat attached to the number.
    """
    daily = daily if daily is not None else fetch_eibor_daily(xlsx_path)
    buckets: dict[int, list[float]] = defaultdict(list)
    months: dict[int, set[int]] = defaultdict(set)
    for row in daily:
        day = row["date"]
        buckets[day.year].append(row["eibor_3m"])
        months[day.year].add(day.month)

    out = []
    for year, values in sorted(buckets.items()):
        covered = sorted(months[year])
        out.append(
            {
                "year": year,
                "eibor_3m": round(sum(values) / len(values), 5),
                "eibor_3m_n_days": len(values),
                "eibor_3m_n_months": len(covered),
                "partial_year": len(covered) < 12,
                "first_month": min(covered),
                "last_month": max(covered),
            }
        )
    return out
