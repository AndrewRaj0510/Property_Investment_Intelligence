"""The baseline carries the bucket's rent/price SPREAD, not just its medians (F44).

A reader asked the live results page "what is the highest estimated annual rent
for this community?" and was refused, correctly: the analysis held medians and
nothing else, so the grounded Q&A had no figure to answer from. The medians
snapshot now carries quartiles, the 90th percentile and the maximum alongside
them.

Two properties are worth pinning:

  * the scorer READS them, so they reach the analysis and therefore the Q&A;
  * their absence cannot cost a caller a verdict. They are new columns on a
    table that already exists in deployed databases, and scoring itself does not
    use them, so a snapshot refreshed before they were added must degrade to the
    medians rather than fail the request.
"""

from __future__ import annotations

import pytest
from psycopg.errors import UndefinedColumn

from db.queries.yield_agg import DISTRIBUTION_COLUMNS
from services import listing_scorer

BASE_ROW = ("Marsa Dubai", "apartment", 2_040_000.0, 112_200.0, 0.055, 340, 512, False)
SPREAD = (
    1_300_000.0, 2_900_000.0, 4_100_000.0, 95_000_000.0,
    72_000.0, 142_000.0, 190_000.0, 45_000_000.0,
)


class _Cursor:
    def __init__(self, conn, row):
        self._conn = conn
        self._row = row

    def execute(self, sql, params=None):
        self._conn.statements.append(sql)
        if self._conn.narrow_table and "p90_annual_rent" in sql:
            raise UndefinedColumn("column p90_annual_rent does not exist")

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Conn:
    """Just enough psycopg surface for the baseline lookup."""

    def __init__(self, row, narrow_table=False):
        self.row = row
        self.narrow_table = narrow_table
        self.statements: list[str] = []
        self.rolled_back = False

    def cursor(self):
        row = BASE_ROW if self.narrow_table else self.row
        return _Cursor(self, row)

    def rollback(self):
        self.rolled_back = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def patched(monkeypatch):
    """Point the lookup at a fake connection, and stop area resolution hitting
    the database on its way there."""

    def install(conn):
        monkeypatch.setattr(listing_scorer, "_resolve_area", lambda a: a)
        monkeypatch.setattr(listing_scorer, "_connect", lambda settings: conn)
        return conn

    return install


def test_spread_reaches_the_analysis(patched):
    conn = patched(_Conn(BASE_ROW + SPREAD))
    baseline = listing_scorer._baseline_for("Marsa Dubai", "apartment")
    assert baseline["median_annual_rent"] == 112_200.0
    assert baseline["p90_annual_rent"] == 190_000.0
    assert baseline["max_annual_rent"] == 45_000_000.0
    assert baseline["p90_sale_price"] == 4_100_000.0


def test_every_distribution_column_is_present_on_the_baseline(patched):
    patched(_Conn(BASE_ROW + SPREAD))
    baseline = listing_scorer._baseline_for("Marsa Dubai", "apartment")
    for col in DISTRIBUTION_COLUMNS:
        assert col in baseline, col


def test_a_snapshot_without_the_columns_still_scores(patched):
    """The verdict is about the medians; the spread only widens what Q&A covers.

    So an older snapshot has to yield a working baseline with the extra fields
    empty, not an exception that costs the caller their analysis.
    """
    conn = patched(_Conn(BASE_ROW, narrow_table=True))
    baseline = listing_scorer._baseline_for("Marsa Dubai", "apartment")
    assert baseline["gross_yield"] == 0.055
    assert baseline["n_rents"] == 512
    assert all(baseline[col] is None for col in DISTRIBUTION_COLUMNS)


def test_the_failed_wide_query_is_rolled_back_before_retrying(patched):
    """Postgres aborts the whole transaction on a failed statement.

    Without the rollback the retry raises InFailedSqlTransaction and the
    fallback is no fallback at all.
    """
    conn = patched(_Conn(BASE_ROW, narrow_table=True))
    listing_scorer._baseline_for("Marsa Dubai", "apartment")
    assert conn.rolled_back
    assert len(conn.statements) == 2


def test_missing_bucket_is_still_none(patched):
    patched(_Conn(None))
    assert listing_scorer._baseline_for("Nowhere", "apartment") is None
