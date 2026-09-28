"""GET /market/price-history - the results-screen price chart.

The contract worth pinning is not the numbers (they move with every DLD
backfill) but the two properties that make the chart honest:

  * it describes the SAME population as the yield baseline shown beside it, and
  * it never projects. A forecast point here would be a number with no method
    behind it, and both LLM prompts in this project forbid predicting prices.
"""

from __future__ import annotations

from api.routes.market import _direction, _with_moving_average


def _rows(values: list[float]) -> list[dict]:
    """Minimal quarterly rows carrying only the fields these helpers read."""
    return [
        {
            "period": f"20{20 + i // 4}-Q{i % 4 + 1}",
            "year": 2020 + i // 4,
            "quarter": i % 4 + 1,
            "median_aed_per_sqm": v * 10.7639,
            "median_aed_per_sqft": v,
            "n_sales": 10,
        }
        for i, v in enumerate(values)
    ]


def test_moving_average_is_null_until_a_full_year_is_behind_it():
    """A 4-quarter mean drawn over 1 quarter of data is not a 4-quarter mean.

    Back-filling those first three points with a partial average would draw a
    smoothed line through a window that does not exist.
    """
    points = _with_moving_average(_rows([100, 200, 300, 400]))
    assert [p.ma4_aed_per_sqft for p in points[:3]] == [None, None, None]
    assert points[3].ma4_aed_per_sqft == 250.0


def test_direction_needs_two_full_years_before_it_will_call_a_trend():
    """With 20-odd sales a quarter, one quarter against another is noise."""
    change, label = _direction(_rows([100] * 7))
    assert change is None and label is None


def test_direction_compares_four_quarters_against_the_previous_four():
    rising = _rows([100, 100, 100, 100, 120, 120, 120, 120])
    change, label = _direction(rising)
    assert change == 20.0
    assert label == "rising"

    falling = _rows([120, 120, 120, 120, 100, 100, 100, 100])
    change, label = _direction(falling)
    assert change is not None and change < 0
    assert label == "falling"


def test_small_moves_are_reported_as_flat_not_as_a_trend():
    """A 1% difference between two sampled years is not a market direction."""
    _, label = _direction(_rows([100, 100, 100, 100, 101, 101, 101, 101]))
    assert label == "flat"


def test_response_carries_provenance_and_refuses_to_forecast():
    """basis/note travel with the series so a client cannot strip the caveat."""
    from api.routes.market import BASIS, NOTE

    assert "Dubai Land Department" in BASIS
    assert "not a forecast" in NOTE.lower()


def test_price_history_shares_the_yield_baseline_type_normalisation():
    """One type vocabulary, or the chart plots different stock from the verdict.

    If these drift apart, the chart's "apartment" and the baseline card's
    "apartment" become two different populations on the same screen.
    """
    from db.queries import price_history as ph
    from db.queries import yield_agg

    assert ph._SALES_TYPE_CASE is yield_agg._SALES_TYPE_CASE
    assert ph.SALES_TABLE == yield_agg.SALES_TABLE


def test_thin_quarters_are_dropped_rather_than_plotted():
    """A median of two sales is not a market observation."""
    from db.queries.price_history import MIN_QUARTER_SALES

    assert MIN_QUARTER_SALES >= 4
